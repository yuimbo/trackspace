"""Offline genre-space neighbourhood scoring — experiment surface only.

Compares musical neighbourhoods across aligned cached/extracted representations
(MAEST anchor, hidden blocks, MERT layers) without touching production: never
imported by ``backend.app``, runs no inference, trains no model.

:func:`recipes` — the fixed recipe set for a source list (normalised weights,
deterministic safe ids). ``recipe_set="baseline"`` is the original 13-recipe
sweep, unchanged; ``directed`` is the fixed 5-option contrastive preset
(:data:`DIRECTED_RECIPES`); ``logits-ab`` the fixed 2-option logits on/off
ablation (:data:`LOGITS_AB_RECIPES`).
:func:`compose` — one recipe's matrix like production: per-source L2
normalisation, sqrt-weight scaling, concatenation (weights renormalised to sum 1).
:func:`score_neighbours` — every query vs all other tracks (self excluded by
index), with micro/macro precision, retrieved-label confusion, and
artist/release concentration.
:func:`evaluate` — every recipe raw and in a PCA geometry fitted **only on dev
rows**; per model family the best (recipe, geometry) by **dev macro-per-label
precision of the unrestricted scoring** (ties: recipe id, then geometry id,
lexicographic). Every split/mode keeps the **full** :func:`score_neighbours`
report; heldout and the exclude_related diagnostic are never used for selection.

Callers hand aligned vectors and a track table and get a report. Pure by default;
``evaluate`` accepts an optional ``on_progress`` callback for long runs.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

MAEST_ANCHOR = "maest_logits"
MAEST_HIDDEN = "maest"
MAEST_LAYER7 = "maest_layer7"
MERT_L19 = "mert_l19"
MERT_L23 = "mert_l23"
MERT_LAYERS = (MERT_L19, MERT_L23)

#: Hidden-representation share sweep for the anchor+hidden recipes.
HIDDEN_SHARES = (0.2, 0.35, 0.5)

#: Raw anchor/hidden weights the ``maest_logits+maest`` recipe normalises from
#: (mirrors ``projection_config.FROZEN_SOURCE_WEIGHTS`` 0.45/0.25).
ANCHOR_MAEST_RAW_WEIGHTS = {MAEST_ANCHOR: 0.45, MAEST_HIDDEN: 0.25}

#: Allowed split values on a track record.
SPLITS = ("dev", "heldout")

#: Required keys on every track record.
TRACK_KEYS = ("fingerprint", "path", "label", "artist", "release", "split")

#: Fixed canonical source order (anchor first) — recipe ids and matrix column
#: order, so a weight sweep never renames the block layout. Unknown sources
#: fall back to lexical order after the known ones.
KNOWN_SOURCE_ORDER = (MAEST_ANCHOR, MAEST_HIDDEN, MAEST_LAYER7, MERT_L19, MERT_L23)
_KNOWN_SOURCE_RANK = {s: i for i, s in enumerate(KNOWN_SOURCE_ORDER)}

#: Named recipe sets accepted by :func:`recipes` (see module docstring).
RECIPE_SETS = ("baseline", "directed", "logits-ab")

#: Fixed "directed" preset: five stable, route/file-safe ids (verbatim, no
#: ``+``/``@`` slug) contrasting logits/hidden/layer7/MERT-L19, each normalised to sum 1.
DIRECTED_RECIPES: dict[str, dict[str, float]] = {
    "genre_control": {MAEST_ANCHOR: 0.65, MAEST_HIDDEN: 0.35},
    "genre_structure": {MAEST_ANCHOR: 0.65, MAEST_LAYER7: 0.35},
    "genre_without_layer7": {MAEST_ANCHOR: 0.65, MAEST_HIDDEN: 0.20, MERT_L19: 0.15},
    "genre_without_mert": {MAEST_ANCHOR: 0.55, MAEST_HIDDEN: 0.20, MAEST_LAYER7: 0.25},
    "genre_full": {MAEST_ANCHOR: 0.40, MAEST_HIDDEN: 0.20, MAEST_LAYER7: 0.25, MERT_L19: 0.15},
}

#: Readable source-of-truth labels for the directed ids (also the gallery labels
#: for those maps; baseline ids fall back to the default "PCA50 -> openTSNE (id)").
RECIPE_LABELS: dict[str, str] = {
    "genre_control": "Control: genre logits + final MAEST",
    "genre_structure": "Structure: genre logits + MAEST layer7",
    "genre_without_layer7": "Without layer7: logits + final MAEST + MERT19",
    "genre_without_mert": "Without MERT19: logits + final MAEST + layer7",
    "genre_full": "Full: logits + final MAEST + layer7 + MERT19",
    "genre_without_logits": "Without logits: final MAEST + layer7 + MERT19",
}


# ── recipes ──────────────────────────────────────────────────────────────────


def _normalise(weights):
    total = float(sum(weights.values()))
    return {s: w / total for s, w in weights.items()}


#: Fixed "logits-ab" preset: exactly two options — the shared ``genre_full``
#: (shallow copy of :data:`DIRECTED_RECIPES`, one source of truth) and its
#: ``genre_without_logits`` ablation (logits dropped, rest renormalised to sum 1).
LOGITS_AB_RECIPES: dict[str, dict[str, float]] = {
    "genre_full": dict(DIRECTED_RECIPES["genre_full"]),
    "genre_without_logits": _normalise(
        {s: w for s, w in DIRECTED_RECIPES["genre_full"].items() if s != MAEST_ANCHOR}
    ),
}

#: Fixed all-or-nothing presets :func:`recipes` returns verbatim (readable ids,
#: normalised weights). Each requires every source any of its recipes needs —
#: never a silently partial set, because each preset is only meaningful as a
#: complete comparison.
_FIXED_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "directed": DIRECTED_RECIPES,
    "logits-ab": LOGITS_AB_RECIPES,
}


def _canonical_order(weights):
    """Fixed source order: known sources by :data:`KNOWN_SOURCE_ORDER` (anchor
    first), unknown sources lexical after — never weight-dependent, so a sweep
    keeps the same recipe ids and matrix column layout."""
    return sorted(weights, key=lambda s: (_KNOWN_SOURCE_RANK.get(s, len(KNOWN_SOURCE_ORDER)), s))


def _recipe_id(weights):
    """Deterministic safe id: ``src`` or ``src1+src2@643-357`` (per-mille
    weights; only ``[a-z0-9_+@-]``, so file/JSON/CLI safe)."""
    order = _canonical_order(weights)
    if len(order) == 1:
        return order[0]
    per_mille = "-".join(f"{round(weights[s] * 1000):03d}" for s in order)
    return "+".join(order) + "@" + per_mille


def recipes(
    available_sources: set[str], *, recipe_set: str = "baseline"
) -> dict[str, dict[str, float]]:
    """The fixed genre-space recipes available for *available_sources*.

    ``recipe_set`` selects which fixed set to return (:data:`RECIPE_SETS`).
    ``baseline`` (default) keeps the original ids and build order unchanged:
    recipes needing a missing source are omitted; values are weight vectors
    normalised to sum 1, keys deterministic safe ids.

    ``directed`` returns :data:`DIRECTED_RECIPES` and ``logits-ab`` returns
    :data:`LOGITS_AB_RECIPES`, ids verbatim. Each is **all-or-nothing**: every
    source any of its recipes needs must be available, or it raises — never a
    silently partial set, because the point is to compare all of the set's
    options.
    """
    if recipe_set not in RECIPE_SETS:
        raise ValueError(
            f"unknown recipe_set {recipe_set!r} (choices: {', '.join(RECIPE_SETS)})"
        )
    if recipe_set in _FIXED_PRESETS:
        preset = _FIXED_PRESETS[recipe_set]
        required = set().union(*preset.values())
        missing = sorted(required - set(available_sources))
        if missing:
            raise ValueError(
                f"{recipe_set} recipe set requires every source; cannot compare all "
                f"options with missing source(s): {', '.join(missing)}"
            )
        return {rid: dict(w) for rid, w in preset.items()}
    avail = set(available_sources)
    out: dict[str, dict[str, float]] = {}

    def add(weights):
        if any(s not in avail for s in weights):
            return
        out[_recipe_id(weights)] = {s: float(w) for s, w in weights.items()}

    if MAEST_ANCHOR in avail:
        add({MAEST_ANCHOR: 1.0})
        if MAEST_HIDDEN in avail:
            add(_normalise(ANCHOR_MAEST_RAW_WEIGHTS))
        if MAEST_LAYER7 in avail:
            for share in HIDDEN_SHARES:
                add({MAEST_ANCHOR: 1.0 - share, MAEST_LAYER7: share})
    for mert in MERT_LAYERS:  # all MERT layers alone first
        if mert in avail:
            add({mert: 1.0})
    if MAEST_ANCHOR in avail:  # then the anchor+each-MERT sweeps
        for mert in MERT_LAYERS:
            if mert in avail:
                for share in HIDDEN_SHARES:
                    add({MAEST_ANCHOR: 1.0 - share, mert: share})
    return out


# ── compose ──────────────────────────────────────────────────────────────────


def compose(vectors: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """One recipe matrix like production: per-source L2 normalisation, sqrt-weight
    scaling, concatenation — weights renormalised to sum 1, blocks in
    :func:`_canonical_order`.

    Validates: vectors finite, 2-D, ≥1 column, aligned rows, nonzero rows (L2 of
    a zero row is undefined); weights non-empty, finite, strictly positive, all
    sources present. Inputs are never mutated; the result is ``float32`` like
    production.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    vecs = {s: np.asarray(v) for s, v in vectors.items()}
    unknown = [s for s in weights if s not in vecs]
    if unknown:
        raise ValueError(
            f"weights reference source(s) absent from vectors: {', '.join(unknown)}"
        )
    n_rows: int | None = None
    for s, v in vecs.items():
        if v.ndim != 2 or v.shape[1] < 1:
            raise ValueError(
                f"source {s!r} must be a 2-D array with at least one column, got shape {v.shape}"
            )
        if not np.isfinite(v).all():
            raise ValueError(f"source {s!r} contains non-finite values")
        if n_rows is None:
            n_rows = v.shape[0]
        elif v.shape[0] != n_rows:
            raise ValueError(
                f"source rows are not aligned: {s!r} has {v.shape[0]} rows, expected {n_rows}"
            )
        if np.any(np.linalg.norm(v, axis=1) <= 0.0):
            raise ValueError(f"source {s!r} contains a zero row (L2 normalisation is undefined)")
    for s, w in weights.items():
        if not np.isfinite(w) or w <= 0.0:
            raise ValueError(f"weight for source {s!r} must be a positive finite number, got {w!r}")

    total = float(sum(weights.values()))
    blocks: list[np.ndarray] = []
    for s in _canonical_order(weights):
        v = vecs[s].astype(np.float64)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        v = v * math.sqrt(weights[s] / total)
        blocks.append(v.astype(np.float32))
    return np.concatenate(blocks, axis=1)


# ── score_neighbours ─────────────────────────────────────────────────────────


def _known(value) -> str | None:
    """A non-empty string value, else ``None`` (unknown metadata)."""
    return value if isinstance(value, str) and value else None


def _validate_tracks(tracks: list[dict]) -> None:
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            raise ValueError(f"track {i} is not a dict")
        missing = [key for key in TRACK_KEYS if key not in t]
        if missing:
            raise ValueError(f"track {i} missing required key(s): {', '.join(missing)}")
        if not isinstance(t["label"], str) or not t["label"]:
            raise ValueError(f"track {i} must have a non-empty label")
        if t["split"] not in SPLITS:
            raise ValueError(f"track {i} split {t['split']!r} not in {SPLITS!r}")
        for key in ("fingerprint", "path"):
            if not isinstance(t[key], str) or not t[key]:
                raise ValueError(f"track {i} {key!r} must be a non-empty string")


def _related_to(track: dict, q_artist: str | None, q_release: str | None) -> bool:
    """Same *known* artist or same *known* release; missing metadata never equal."""
    a = _known(track["artist"])
    r = _known(track["release"])
    return (q_artist is not None and a == q_artist) or (
        q_release is not None and r == q_release
    )


def score_neighbours(
    matrix: np.ndarray,
    tracks: list[dict],
    *,
    k: int = 10,
    query_split: str = "heldout",
    exclude_related: bool = False,
) -> dict:
    """Score every track in *query_split* against all other tracks (both splits).

    Self is removed by explicit index — never by slicing the sorted list, which
    would drop a tied genuine neighbour; cosine ties break by input index. ``k``
    clamps per query; a no-candidate query is omitted and counted.
    ``exclude_related=True`` drops candidates sharing the query's *known*
    artist or release; missing metadata never counts as equal to anything.

    Returns the full report: query split, candidate pool, ``counts`` (queries,
    scored, omitted no-candidate, effective k min/max), ``precision`` (micro,
    macro per label, per-label values + query counts), row-normalised
    retrieved-label ``confusion`` (same label included), ``related``
    same-artist/same-release concentration with metadata coverage and
    denominators, and ``per_query`` (path/fingerprint/label, effective k,
    precision, ranked neighbours with path/fingerprint/label/cosine).
    """
    if query_split not in SPLITS:
        raise ValueError(f"query_split must be one of {SPLITS!r}, got {query_split!r}")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {M.shape}")
    if M.shape[0] != len(tracks):
        raise ValueError(f"matrix rows ({M.shape[0]}) must match tracks ({len(tracks)})")
    if not np.isfinite(M).all():
        raise ValueError("matrix must contain only finite values")
    norms = np.linalg.norm(M, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("matrix rows must be nonzero (cosine is undefined for a zero row)")
    _validate_tracks(tracks)
    n = len(tracks)

    query_idxs = [i for i, t in enumerate(tracks) if t["split"] == query_split]
    per_query: list[dict] = []
    per_label_agg: dict[str, list[float]] = {}  # label -> [sum precision, n queries]
    retrieved_by_query_label: dict[str, Counter] = {}
    effective_ks: list[int] = []
    correct_total = 0
    retrieved_total = 0
    artist_sum = 0.0
    artist_denom = 0
    release_sum = 0.0
    release_denom = 0
    omitted = 0

    for i in query_idxs:
        t = tracks[i]
        q_artist = _known(t["artist"])
        q_release = _known(t["release"])
        cands = [j for j in range(n) if j != i]  # explicit index exclusion
        if exclude_related:
            cands = [j for j in cands if not _related_to(tracks[j], q_artist, q_release)]
        if not cands:
            omitted += 1
            continue
        cos = (M[i] @ M[cands].T) / (norms[i] * norms[cands])
        ranked = sorted(zip(cands, cos), key=lambda p: (-p[1], p[0]))
        eff = min(k, len(ranked))
        picked = ranked[:eff]
        effective_ks.append(eff)

        neighbours: list[dict] = []
        correct = 0
        same_artist = 0
        same_release = 0
        for j, c in picked:
            nt = tracks[j]
            neighbours.append(
                {
                    "path": nt["path"],
                    "fingerprint": nt["fingerprint"],
                    "label": nt["label"],
                    "cosine": float(c),
                }
            )
            if nt["label"] == t["label"]:
                correct += 1
            if q_artist is not None and _known(nt["artist"]) == q_artist:
                same_artist += 1
            if q_release is not None and _known(nt["release"]) == q_release:
                same_release += 1
            retrieved_by_query_label.setdefault(t["label"], Counter())[nt["label"]] += 1

        precision = correct / eff
        correct_total += correct
        retrieved_total += eff
        agg = per_label_agg.setdefault(t["label"], [0.0, 0.0])
        agg[0] += precision
        agg[1] += 1.0
        if q_artist is not None:
            artist_sum += same_artist / eff
            artist_denom += 1
        if q_release is not None:
            release_sum += same_release / eff
            release_denom += 1
        per_query.append(
            {
                "path": t["path"],
                "fingerprint": t["fingerprint"],
                "label": t["label"],
                "effective_k": eff,
                "precision": precision,
                "neighbours": neighbours,
            }
        )

    scored = len(per_query)
    per_label = {label: agg[0] / agg[1] for label, agg in per_label_agg.items()}
    split_counts = Counter(t["split"] for t in tracks)
    return {
        "query_split": query_split,
        "k": k,
        "exclude_related": bool(exclude_related),
        "candidates": "all other tracks (both splits), self excluded by index",
        "candidate_pool": {s: split_counts.get(s, 0) for s in SPLITS},
        "counts": {
            "tracks": n,
            "queries": len(query_idxs),
            "queries_scored": scored,
            "queries_omitted_no_candidates": omitted,
            "effective_k_min": min(effective_ks) if effective_ks else None,
            "effective_k_max": max(effective_ks) if effective_ks else None,
        },
        "precision": {
            "micro": (correct_total / retrieved_total) if retrieved_total else None,
            "macro_per_label": (sum(per_label.values()) / len(per_label)) if per_label else None,
            "per_label": dict(sorted(per_label.items())),
            "label_query_counts": {
                label: int(agg[1]) for label, agg in sorted(per_label_agg.items())
            },
        },
        "confusion": {
            ql: {cl: cnt / sum(hist.values()) for cl, cnt in sorted(hist.items())}
            for ql, hist in sorted(retrieved_by_query_label.items())
        },
        "related": {
            "same_artist": {
                "concentration": (artist_sum / artist_denom) if artist_denom else None,
                "denominator_queries": artist_denom,
                "metadata_coverage": (artist_denom / scored) if scored else 0.0,
            },
            "same_release": {
                "concentration": (release_sum / release_denom) if release_denom else None,
                "denominator_queries": release_denom,
                "metadata_coverage": (release_denom / scored) if scored else 0.0,
            },
        },
        "per_query": per_query,
    }


# ── evaluate ─────────────────────────────────────────────────────────────────


def pca_dev_fit_transform(
    mat: np.ndarray, dev_indices, dim: int, seed: int = 42
) -> tuple[np.ndarray, dict]:
    """Project *mat* with a PCA fitted **only on the dev rows**.

    ``fit`` on ``mat[dev_indices]`` (a copy), then ``transform`` every row —
    never a ``fit_transform`` on the whole matrix, so heldout rows pass through
    a projection that never saw them. Components cap to
    ``min(dim, n_dev - 1, n_features - 1)``; fewer than 2 dev rows raises (the
    geometry is then reported unavailable by :func:`evaluate`). Inputs are
    never mutated. Returns ``(projected, info)``.
    """
    from sklearn.decomposition import PCA  # lazy — heavy import

    X = np.asarray(mat, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("matrix must contain only finite values")
    dev_idx = np.asarray(dev_indices, dtype=int)
    if dev_idx.ndim != 1 or dev_idx.size == 0:
        raise ValueError("dev_indices must be a non-empty 1-D index array")
    if dev_idx.min() < 0 or dev_idx.max() >= X.shape[0]:
        raise ValueError("dev_indices out of range for the matrix")
    if int(dim) < 1:
        raise ValueError("dim must be >= 1")
    n_dev = int(dev_idx.size)
    n_feat = int(X.shape[1])
    if n_dev < 2:
        raise ValueError(f"PCA needs at least 2 dev rows to fit, got {n_dev}")
    if n_feat < 2:
        raise ValueError(f"PCA needs at least 2 features, got {n_feat}")
    components = max(1, min(int(dim), n_dev - 1, n_feat - 1))
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(X[dev_idx])
    return pca.transform(X), {
        "components": components,
        "fit_on": "dev rows only",
        "n_dev_rows": n_dev,
        "n_features": n_feat,
    }


def _family(weights: dict[str, float]) -> str:
    """Model family a recipe belongs to: a MERT layer defines its own family;
    everything else (anchor + hidden blocks) is the MAEST family."""
    if MERT_L23 in weights:
        return MERT_L23
    if MERT_L19 in weights:
        return MERT_L19
    return MAEST_ANCHOR


def _score_summary(result: dict) -> dict:
    """Compact per-split/mode summary for the ``selected`` entries."""
    return {
        "queries_scored": result["counts"]["queries_scored"],
        "effective_k_min": result["counts"]["effective_k_min"],
        "effective_k_max": result["counts"]["effective_k_max"],
        "micro": result["precision"]["micro"],
        "macro_per_label": result["precision"]["macro_per_label"],
        "per_label": result["precision"]["per_label"],
    }


def evaluate(
    vectors: dict[str, np.ndarray],
    tracks: list[dict],
    *,
    k: int = 10,
    pca_dim: int = 50,
    seed: int = 42,
    recipe_set: str = "baseline",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Compare every recipe on the same full intersection, raw vs PCA geometry.

    The intersection is the rows finite and nonzero in every recipe-used source;
    every recipe is scored raw and in a PCA geometry fitted only on the dev rows,
    for both splits and both modes, keeping the **full** :func:`score_neighbours`
    report per entry. Selection is per model family: the (recipe, geometry) with
    the best **dev macro-per-label precision of the unrestricted scoring** (ties:
    recipe id, then geometry id, lexicographic); heldout and exclude_related are
    reported but never used. Requires ≥1 dev and ≥1 heldout track in the
    intersection and that every provided source is used by some recipe. Inputs
    are never mutated.

    ``on_progress(rid)`` fires twice per recipe (before composing/PCA and after
    its scoring); ``recipe_set`` selects which fixed :func:`recipes` set to
    evaluate (``baseline`` default).
    """
    if not vectors:
        raise ValueError("vectors must be non-empty")
    vecs = {s: np.asarray(v) for s, v in vectors.items()}
    for s, v in vecs.items():
        if v.ndim != 2:
            raise ValueError(f"source {s!r} must be 2-D, got shape {v.shape}")
        if v.shape[0] != len(tracks):
            raise ValueError(f"source {s!r} rows ({v.shape[0]}) must match tracks ({len(tracks)})")
    _validate_tracks(tracks)
    if len(tracks) == 0:
        raise ValueError("tracks must be non-empty")

    rcp = recipes(set(vecs), recipe_set=recipe_set)
    if not rcp:
        raise ValueError("no recipes available for the provided sources")
    used = sorted({s for weights in rcp.values() for s in weights})
    unused = sorted(set(vecs) - set(used))
    if unused:
        raise ValueError(f"source(s) provided but used by no recipe: {', '.join(unused)}")

    good = np.ones(len(tracks), dtype=bool)
    for s in used:
        v = vecs[s]
        good &= np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1) > 0.0)
    idx = np.flatnonzero(good)
    if len(idx) == 0:
        raise ValueError("no tracks in the intersection of the recipe sources")
    int_tracks = [tracks[i] for i in idx]
    int_vecs = {s: vecs[s][idx] for s in used}  # fancy-index copies; inputs untouched

    dev_n = sum(1 for t in int_tracks if t["split"] == "dev")
    held_n = len(int_tracks) - dev_n
    if dev_n < 1 or held_n < 1:
        raise ValueError(
            "evaluate needs at least one dev and one heldout track in the intersection "
            f"(got dev={dev_n}, heldout={held_n})"
        )
    dev_idx = [i for i, t in enumerate(int_tracks) if t["split"] == "dev"]
    pca_geom = f"pca{pca_dim}"

    def _both_modes(mat: np.ndarray) -> dict:
        return {
            split: {
                mode: score_neighbours(
                    mat,
                    int_tracks,
                    k=k,
                    query_split=split,
                    exclude_related=(mode == "exclude_related"),
                )
                for mode in ("unrestricted", "exclude_related")
            }
            for split in ("dev", "heldout")
        }

    scores: dict[str, dict] = {}
    for rid, weights in rcp.items():
        if on_progress is not None:
            on_progress(rid)  # before composing/PCA
        mat = compose({s: int_vecs[s] for s in weights}, weights)
        geometries: dict[str, dict] = {"raw": _both_modes(mat)}
        try:
            projected, pinfo = pca_dev_fit_transform(mat, dev_idx, pca_dim, seed=seed)
            geometries[pca_geom] = {**pinfo, **_both_modes(projected)}
        except ValueError as e:
            geometries[pca_geom] = {"status": "unavailable", "reason": str(e)}
        scores[rid] = {
            "family": _family(weights),
            "sources": sorted(weights),
            "weights": dict(weights),
            "geometries": geometries,
        }
        if on_progress is not None:
            on_progress(rid)  # recipe complete

    def _dev_unrestricted_macro(rid: str, geom: str) -> float:
        entry = scores[rid]["geometries"][geom]
        if "dev" not in entry:
            return float("-inf")
        macro = entry["dev"]["unrestricted"]["precision"]["macro_per_label"]
        return macro if macro is not None else float("-inf")

    selected: dict[str, dict] = {}
    for fam in sorted({s["family"] for s in scores.values()}):
        candidates = [
            (rid, geom)
            for rid, s in scores.items()
            if s["family"] == fam
            for geom in s["geometries"]
        ]
        rid, geom = min(candidates, key=lambda p: (-_dev_unrestricted_macro(*p), p[0], p[1]))
        dev_entry = scores[rid]["geometries"][geom]["dev"]
        hold_entry = scores[rid]["geometries"][geom]["heldout"]
        selected[fam] = {
            "recipe": rid,
            "geometry": geom,
            "selection_split": "dev",
            "selection_scoring": "unrestricted",
            "dev": {m: _score_summary(dev_entry[m]) for m in ("unrestricted", "exclude_related")},
            "heldout": {m: _score_summary(hold_entry[m]) for m in ("unrestricted", "exclude_related")},
        }

    return {
        "selection": "dev",
        "k": k,
        "pca_dim": pca_dim,
        "seed": seed,
        "dataset": {
            "tracks": len(int_tracks),
            "intersection_rows": len(int_tracks),
            "dropped_rows": len(tracks) - len(idx),
            "splits": {"dev": dev_n, "heldout": held_n},
        },
        "sources": used,
        "recipes": list(rcp),
        "geometries": ["raw", pca_geom],
        "scores": scores,
        "selected": selected,
        #: Chosen recipe per family (sorted by family) — unambiguous for a
        #: CLI/doc consumer to act on directly.
        "selected_recipe_ids": [selected[fam]["recipe"] for fam in sorted(selected)],
        "pca": {
            "fit_on": "dev rows only",
            "cap": "min(pca_dim, n_dev_rows - 1, n_features - 1)",
            "n_dev_rows": dev_n,
        },
        "notes": [
            "selection: best (recipe, geometry) per family by dev macro-per-label "
            "precision of the unrestricted scoring; ties broken by recipe id then "
            "geometry id (lexicographic)",
            "heldout is reported but never used for selection; exclude_related is a "
            "diagnostic only and never used for selection",
            "every split/mode entry holds the full score_neighbours report "
            "(per_query included) for CLI/CSV consumers",
            "no model is trained or tuned here; scoring is exact cosine over composed vectors",
            "PCA is fitted on dev rows only and applied to all rows; components capped "
            "to min(pca_dim, n_dev_rows - 1, n_features - 1)",
        ],
    }
