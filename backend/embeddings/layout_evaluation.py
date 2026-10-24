"""Offline layout evaluation: precomputed 2-D maps from cached embeddings.

This is a *read-only* evaluation surface. It gathers the same composite matrix the
interactive projection uses (via ``layout.prepare_projection_input``, so the input
rules — required sources, per-source weights, feature mask/blend — are identical),
but it fits its own UMAP / openTSNE models directly and never touches the
production layout or t-SNE fit caches, and it never mutates MP3 tags. Its only
output is a set of JSON artifacts under one ``--output`` directory.

The goal is a reproducible first gallery: eight maps over the *same* track set,
aligned (orthogonal Procrustes rotation/reflection only, no warp) onto a
reference map, then all normalised aspect-preservingly into the unit square.
The reference defaults to the ``baseline`` recipe but any registered recipe can
be the reference (the ``scale`` set aligns onto ``tsne_budget100``); the
reference is always fitted first and auto-included.

Nothing in this module runs at import time, and the heavy libraries (umap-learn,
openTSNE, scikit-learn) are imported lazily inside the fit functions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .linalg_utils import apply_procrustes, matched_indices, procrustes
from .projection_config import FROZEN_SOURCES

log = logging.getLogger(__name__)

#: Schema version of the manifest and per-recipe artifacts. Bump on a breaking
#: shape change so a reader can reject an old generation explicitly.
SCHEMA_VERSION = 1

#: Below this many tracks the baseline map (an openTSNE t-SNE fit) cannot be
#: produced: its perplexity clamp (min 5) cannot hold, and the gallery aligns
#: every other recipe onto that baseline. A PCA-only recipe could still fit on
#: fewer tracks, but without the baseline there is nothing to align onto, so one
#: shared floor applies to the whole gallery.
MIN_TRACKS = 6

#: Reference recipe everything else is aligned onto. It must be present in
#: :data:`RECIPES`.
BASELINE_ID = "baseline"


class MissingDependencyError(RuntimeError):
    """A recipe needs umap-learn or openTSNE and it is not installed."""


@dataclass(frozen=True)
class Recipe:
    """One precomputed map. Fields form the recipe's reproducible identity."""

    id: str
    label: str
    pca_dim: int
    method: str  # "pca" | "tsne" | "umap" | "umap_tsne" | "tsne_umap_init"
    tsne_n_iter: int = 500
    tsne_early_exaggeration_iter: int = 250
    #: Perplexity for the t-SNE methods. ``None`` keeps the production scalar
    #: rule (:func:`backend.embeddings.projection_backends._tsne_perplexity`);
    #: a one-element tuple fixes a scalar
    #: perplexity; two or more selects openTSNE's multiscale affinity mixture.
    #: ``None`` is omitted from the recipe's identity so the pre-existing recipes
    #: keep their exact revisions/artifacts.
    perplexities: tuple[float, ...] | None = None
    #: Explicit per-source mix for this map, as named ``(source, weight)`` pairs
    #: covering every source in :data:`FROZEN_SOURCES` (a complete vector, so the
    #: order is never positional). ``None`` means the gallery's default
    #: (control) source weights are used — the existing recipes all keep ``None``
    #: and their exact revisions. A non-``None`` value gives the recipe its own
    #: composite matrix and a distinct input fingerprint / revision.
    source_weights: tuple[tuple[str, float], ...] | None = None
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


#: The eight eval maps. The baseline is a *frozen control*: openTSNE over PCA50
#: with the production perplexity/seed and the recipe's own budget (default
#: 500/250 — the openTSNE library default), fully unsupervised; the rest probe
#: dimensionality, budget, and neighbours. This tuple is the *default*
#: generation; the budget extras in :data:`EXTRA_RECIPES` are selectable via
#: :data:`RECIPES_BY_ID` but are not part of the default eight.
#:
#: Production fits its own separate budget (the ``projection_config.OPENTSNE_*``
#: iteration constants); these eval recipes and their labels/revisions are
#: intentionally left frozen.
RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "baseline",
        "PCA50 → openTSNE (production default)",
        pca_dim=50,
        method="tsne",
    ),
    Recipe("pca2", "PCA-only 2D", pca_dim=2, method="pca"),
    Recipe("pca25", "PCA25 → openTSNE", pca_dim=25, method="tsne"),
    Recipe(
        "tsne_budget",
        "PCA50 → openTSNE (n_iter=250, early phase unchanged)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=250,
        tsne_early_exaggeration_iter=250,
    ),
    Recipe(
        "umap2_n15",
        "PCA50 → UMAP2 (n_neighbors=15, min_dist=0.1)",
        pca_dim=50,
        method="umap",
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    ),
    Recipe(
        "umap2_n60",
        "PCA50 → UMAP2 (n_neighbors=60, min_dist=0.1)",
        pca_dim=50,
        method="umap",
        umap_n_neighbors=60,
        umap_min_dist=0.1,
    ),
    Recipe(
        "umap_tsne",
        "PCA50 → UMAP15 (n_neighbors=30) → tSNE2",
        pca_dim=50,
        method="umap_tsne",
        umap_n_components=15,
        umap_n_neighbors=30,
    ),
    Recipe(
        "tsne_umap_init",
        "PCA50 → tSNE2 (UMAP2 initialization)",
        pca_dim=50,
        method="tsne_umap_init",
        umap_n_neighbors=15,
    ),
)

#: The t-SNE extras (the ``budget``, ``scale`` and ``early`` gallery sets), all
#: over the same default unsupervised composite input (PCA50, default weights,
#: seed 42, production perplexity rule). The ``budget`` recipes shorten only the
#: total iteration count (early-exaggeration phase unchanged at 250); the
#: ``scale`` recipes probe the perplexity axis at n_iter=100/early 250; the
#: ``early`` recipes shorten the early-exaggeration phase at
#: production's n_iter=100. Kept out of :data:`RECIPES` so the default
#: generation stays the original eight, but the recipes are fully selectable
#: through :data:`RECIPES_BY_ID`.
EXTRA_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_budget100",
        "PCA50 → openTSNE (n_iter=100, early phase unchanged)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
    ),
    Recipe(
        "tsne_budget150",
        "PCA50 → openTSNE (n_iter=150, early phase unchanged)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=150,
        tsne_early_exaggeration_iter=250,
    ),
    # The "scale" gallery set: n_iter=100 with early 250 over PCA50, probing the
    # perplexity axis. The control is ``tsne_budget100``
    # (perplexity 30, identity-identical to the recipe above); the rest raise the
    # neighbourhood scale or blend several via openTSNE's multiscale mixture.
    # Kept out of :data:`RECIPES` so the default eight are unchanged.
    Recipe(
        "tsne_p60",
        "PCA50 → openTSNE (perplexity 60, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(60.0,),
    ),
    Recipe(
        "tsne_p120",
        "PCA50 → openTSNE (perplexity 120, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(120.0,),
    ),
    Recipe(
        "tsne_multiscale30_120",
        "PCA50 → openTSNE multiscale (perplexities 30 & 120, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(30.0, 120.0),
    ),
    # The "early" gallery set: production's own iteration budget (n_iter=100)
    # over PCA50, probing the early-exaggeration phase length. The control is
    # ``tsne_budget100`` (early 250, identity-identical to the recipe above);
    # the variants shorten the early phase to 150 and 100 iterations. Kept out
    # of :data:`RECIPES` so the default eight are unchanged.
    Recipe(
        "tsne_early150",
        "PCA50 → openTSNE (early phase 150, final 100)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=150,
    ),
    Recipe(
        "tsne_early100",
        "PCA50 → openTSNE (early phase 100, final 100)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=100,
    ),
)

#: The source-mix extras (the ``mixes`` and ``ablation`` gallery sets). All share
#: the control geometry — PCA50, production scalar perplexity rule (30 at library
#: scale), n_iter=100/early 250, seed 42, unsupervised,
#: ``tsne_budget100`` as the alignment reference — and differ only in how the six
#: sources are weighted. The weights are explicit named pairs over
#: ``FROZEN_SOURCES`` (maest_logits, maest, rhythm, clap, supervised, features);
#: each source contributes ``sqrt(w)``× its block exactly as in production, and
#: ``features`` is always 0. Kept out of
#: :data:`RECIPES` so the default generation stays the original eight, but the
#: recipes are fully selectable through :data:`RECIPES_BY_ID`.
#:
#: The ``mixes`` recipes (``tsne_rhythm_forward``, ``tsne_texture_style``)
#: redistribute weight among the four learned sources with ``supervised`` fixed at
#: .15 and a total of 1.10, matching the production control weights
#: (``FROZEN_SOURCE_WEIGHTS`` also sums to 1.10) with no normalization.
#:
#: The ``ablation`` recipes drop whole blocks: ``tsne_maest_supervised`` keeps only
#: maest_logits, maest and supervised (sum .85); ``tsne_maest_only`` keeps only
#: maest_logits and maest (sum .70). The retained weights are preserved exactly
#: (.45/.25 and the supervised .15) — never renormalized — so the relative
#: contribution of the kept blocks matches the control; the *uniform* distance
#: factor simply adapts to the smaller column count. A zero-weight source drops
#: its columns from the composite matrix (the shared ``layout.prepare_projection_input``
#: skips weight<=0 blocks), so these variants are intentionally narrower than the
#: control: 1415 and 1287 columns versus the control's 1941 (maest_logits 519 +
#: maest 768 + supervised 128; the control adds rhythm 14 and clap 512). The track
#: universe is unchanged — every variant still gates on the same required sources
#: (maest_logits, maest) and places the identical path set.
#:
#: The labels state the *hypothesis* each mix is built to probe — they describe
#: the intended weighting, not a guarantee about the perceived semantics.
MIX_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_rhythm_forward",
        "PCA50 → openTSNE (rhythm-forward: rhythm .40, maest_logits .30)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.30),
            ("maest", 0.20),
            ("rhythm", 0.40),
            ("clap", 0.05),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
    Recipe(
        "tsne_texture_style",
        "PCA50 → openTSNE (texture-style: maest .35, clap .30)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.25),
            ("maest", 0.35),
            ("rhythm", 0.05),
            ("clap", 0.30),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
    Recipe(
        "tsne_maest_supervised",
        "PCA50 → openTSNE (ablation: maest_logits .45, maest .25, supervised .15)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.45),
            ("maest", 0.25),
            ("rhythm", 0.0),
            ("clap", 0.0),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
    Recipe(
        "tsne_maest_only",
        "PCA50 → openTSNE (ablation: maest_logits .45, maest .25)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.45),
            ("maest", 0.25),
            ("rhythm", 0.0),
            ("clap", 0.0),
            ("supervised", 0.0),
            ("features", 0.0),
        ),
    ),
)

#: Selectable registry: the original eight plus every extras recipe (budget,
#: scale, early, mixes, ablation), keyed by id. An explicit recipe list may
#: address any of them, while an unspecified generation still runs the original
#: eight (:data:`RECIPES`).
RECIPES_BY_ID: dict[str, Recipe] = {
    r.id: r for r in (*RECIPES, *EXTRA_RECIPES, *MIX_RECIPES)
}


def recipe_params(recipe: Recipe) -> dict[str, Any]:
    """Deterministic dict of a recipe's fields, for the artifact record.

    Fields left at their ``None`` default are omitted so the optional
    ``perplexities`` field does not alter an existing recipe's recorded params.
    """
    return {
        f.name: getattr(recipe, f.name)
        for f in fields(recipe)
        if getattr(recipe, f.name) is not None
    }


def resolve_recipe_set(
    recipe_ids: list[str] | None,
    reference_id: str,
) -> list[str]:
    """The effective ordered recipe set for one generation.

    ``None`` selects the original eight (:data:`RECIPES`); an explicit list is
    kept in caller order. Either way the reference is auto-included first (every
    other map aligns onto it) when absent. ``reference_id`` must already be a
    registered id — the caller validates that before calling.
    """
    if recipe_ids is None:
        chosen = [r.id for r in RECIPES]
    else:
        unknown = [rid for rid in recipe_ids if rid not in RECIPES_BY_ID]
        if unknown:
            raise ValueError(f"unknown recipe id(s): {', '.join(unknown)}")
        chosen = list(recipe_ids)
    if reference_id not in chosen:
        chosen = [reference_id] + chosen
    return [reference_id] + [rid for rid in chosen if rid != reference_id]


def validate_recipe_sources_for(
    chosen: list[str],
    sources: tuple[str, ...],
) -> None:
    """Reject a mix recipe that needs a source absent from the run's ``sources``.

    A source-mix recipe (:attr:`Recipe.source_weights` set) declares a complete
    weight vector over ``FROZEN_SOURCES``, but the composite gather only ever
    contributes the enabled ``sources``: a positive-weight source outside the
    subset would be silently dropped, so the fitted map would record a mix it
    never used. Only positive-weight sources matter — dropping a zero-weight
    block (``features`` at 0.0) cannot change the matrix.
    """
    for rid in chosen:
        pairs = RECIPES_BY_ID[rid].source_weights
        if pairs is None:
            continue
        missing = [
            src for src, w in pairs
            if float(w) > 0.0 and src not in sources
        ]
        if missing:
            raise ValueError(
                f"recipe {rid!r} needs source(s) absent from the run sources: "
                f"{', '.join(missing)} "
                f"(run sources: {', '.join(sources) or 'none'})"
            )


def _clamp_perplexity(requested: float, n_samples: int) -> float:
    # openTSNE needs k < n; clamp a requested perplexity down for small datasets
    # (a scalar or one scale of a multiscale mixture) so tiny fixtures stay valid.
    return min(float(requested), max(1.0, float(n_samples) - 1.0))


def _clamp_pca_dim(requested: int, n_samples: int, n_features: int) -> int:
    return max(1, min(int(requested), n_samples - 1, n_features - 1))


def _clamp_neighbors(requested: int, n_samples: int) -> int:
    return max(2, min(int(requested), n_samples - 1))


# ── fit primitives (uncached; never the production projection backends) ──────


def pca_reduce(mat: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, Any]:
    """Randomised-sklearn-PCA to *dim* components (clamped), returning (X, pca)."""
    from sklearn.decomposition import PCA  # lazy

    n_samples, n_features = mat.shape
    k = _clamp_pca_dim(dim, n_samples, n_features)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(mat), pca


def fit_opentsne(
    reduced: np.ndarray,
    *,
    recipe: Recipe,
    seed: int,
    n_jobs: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """openTSNE 2-D fit with the recipe's own (frozen) iteration budget.

    The recipe carries its ``tsne_n_iter`` / ``tsne_early_exaggeration_iter`` as a
    reproducible control; the *production* fit uses the separate
    ``projection_config.OPENTSNE_*`` iteration constants and is not routed here.

    ``init`` is a precomputed starting embedding (used by the UMAP-init recipe);
    it is scaled to the small variance openTSNE recommends for custom initials so
    the optimisation starts from a sane scale.
    """
    try:
        from openTSNE import TSNE
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "openTSNE is required for the layout-evaluation t-SNE recipes; "
            "install it (pip install opentsne)."
        ) from e

    base_kwargs: dict[str, Any] = dict(
        n_components=2,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
        n_iter=recipe.tsne_n_iter,
        early_exaggeration_iter=recipe.tsne_early_exaggeration_iter,
    )

    def _scaled_init() -> np.ndarray | None:
        if init is None:
            return None
        scaled = np.asarray(init, dtype=np.float64)
        return scaled / (float(scaled.std(axis=0).max()) + 1e-12) * 1e-4

    perps = recipe.perplexities
    if perps is not None and len(perps) > 1:
        # Multiscale: openTSNE's explicit-affinity mixture blends several
        # perplexities, then the t-SNE optimises against that affinity matrix.
        from openTSNE.affinity import MultiscaleMixture

        perplexities = [
            _clamp_perplexity(p, len(reduced)) for p in perps
        ]
        affinities = MultiscaleMixture(
            reduced,
            perplexities=perplexities,
            n_jobs=n_jobs,
            random_state=seed,
        )
        tsne = TSNE(**base_kwargs)
        scaled = _scaled_init()
        kwargs = dict(X=reduced, affinities=affinities)
        if scaled is not None:
            kwargs["initialization"] = scaled
        return np.asarray(tsne.fit(**kwargs), dtype=np.float32)

    # Scalar perplexity: a fixed one from the recipe, else the production rule.
    if perps is None:
        # Production's rule lives in projection_backends. The import is local:
        # that module pulls torch via ``clap`` at import time, and the heavy
        # libraries stay lazy in this module.
        from .projection_backends import _tsne_perplexity

        perplexity = _tsne_perplexity(len(reduced))
    else:
        perplexity = _clamp_perplexity(perps[0], len(reduced))
    kwargs: dict[str, Any] = {**base_kwargs, "perplexity": perplexity}
    tsne = TSNE(**kwargs)
    scaled = _scaled_init()
    if scaled is None:
        return np.asarray(tsne.fit(reduced), dtype=np.float32)
    return np.asarray(tsne.fit(reduced, initialization=scaled), dtype=np.float32)


def fit_umap(
    reduced: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    """umap-learn 2-D (or intermediate) fit. Neighbours are clamped for small n."""
    try:
        import umap
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "umap-learn is required for the layout-evaluation UMAP recipes; "
            "install it (pip install umap-learn)."
        ) from e
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=_clamp_neighbors(n_neighbors, len(reduced)),
        min_dist=float(min_dist),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )
    # UMAP's spectral initialisation needs dim + 1 < n_samples (its eigsh uses
    # k = dim + 1); on a tiny dataset (smoke tests) clamp the intermediate-dim
    # recipes down rather than letting that step raise. At real-library scale
    # this never triggers.
    if reducer.n_components >= len(reduced) - 1:
        reducer.n_components = max(1, len(reduced) - 2)
    return np.asarray(reducer.fit_transform(reduced), dtype=np.float32)


def fit_recipe(mat: np.ndarray, recipe: Recipe, *, seed: int, n_jobs: int):
    """Fit one recipe's 2-D map from the (already gathered) composite *mat*.

    Returns ``(coords, timings)`` where timings has per-stage seconds (each recipe
    runs its own PCA, so the fair total includes the prerequisite cost).
    """
    timings: dict[str, Any] = {}
    t0 = time.perf_counter()
    reduced, _pca = pca_reduce(mat, recipe.pca_dim, seed)
    timings["pca"] = time.perf_counter() - t0
    timings["tsne"] = 0.0
    timings["umap"] = 0.0

    if recipe.method == "pca":
        # Pure PCA map: the reduced coordinates already are the 2-D embedding.
        # A tiny dataset may clamp PCA to a single component (see
        # ``_clamp_pca_dim``); pad to a 2-D row so the map stays comparable.
        coords = np.asarray(reduced, dtype=np.float32)
        if coords.shape[1] < 2:
            pad = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=np.float32)
            coords = np.concatenate([coords, pad], axis=1)
    elif recipe.method == "tsne":
        t0 = time.perf_counter()
        coords = fit_opentsne(reduced, recipe=recipe, seed=seed, n_jobs=n_jobs)
        timings["tsne"] = time.perf_counter() - t0
    elif recipe.method == "umap":
        t0 = time.perf_counter()
        coords = fit_umap(
            reduced,
            n_components=recipe.umap_n_components,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
    elif recipe.method == "umap_tsne":
        t0 = time.perf_counter()
        intermediate = fit_umap(
            reduced,
            n_components=recipe.umap_n_components,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        coords = fit_opentsne(intermediate, recipe=recipe, seed=seed, n_jobs=n_jobs)
        timings["tsne"] = time.perf_counter() - t0
    elif recipe.method == "tsne_umap_init":
        t0 = time.perf_counter()
        init = fit_umap(
            reduced,
            n_components=2,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        coords = fit_opentsne(reduced, recipe=recipe, seed=seed, n_jobs=n_jobs, init=init)
        timings["tsne"] = time.perf_counter() - t0
    else:  # pragma: no cover - recipe set is fixed
        raise ValueError(f"unknown recipe method {recipe.method!r}")

    if not np.isfinite(coords).all():
        raise ValueError(f"recipe {recipe.id!r} produced non-finite coordinates")
    return coords, timings


# ── alignment + normalisation ─────────────────────────────────────────────────


def orthogonal_align_to(
    coords: np.ndarray,
    paths: list[str],
    ref_coords: np.ndarray,
    ref_paths: list[str],
) -> np.ndarray:
    """Rotate/reflect *coords* onto *ref_coords* using the shared tracks.

    Only an orthogonal (rotation + reflection) transform with translation — no
    scale, no warp — so the map's own geometry is preserved while its frame is
    made comparable to the reference. Raises on a non-finite result (a degenerate
    alignment must not write NaN positions into an artifact).
    """
    idx_new, idx_ref = matched_indices(paths, ref_paths)
    if len(idx_new) < 2:
        return coords
    rot, _scale, src_mu, dst_mu = procrustes(
        coords[idx_new], ref_coords[idx_ref]
    )
    aligned = apply_procrustes(coords, rot, 1.0, src_mu, dst_mu)
    if not np.all(np.isfinite(aligned)):
        raise ValueError(
            "alignment produced non-finite coordinates "
            f"({len(idx_new)} shared tracks)"
        )
    return aligned


def normalise_aspect(coords: np.ndarray) -> np.ndarray:
    """Center *coords* and scale it uniformly into the unit square.

    ``[0, 1]`` in both axes with the aspect ratio preserved — a *uniform* scale
    (never a per-axis stretch), applied identically to every map.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 1e-12:
        span = 1.0
    center = (mins + maxs) / 2.0
    return ((coords - center) / span + 0.5).astype(np.float32)


# ── identity hashes ───────────────────────────────────────────────────────────


def dataset_fingerprint(paths: list[str], mat: np.ndarray, comp_config: str) -> str:
    """Stable identity of one generation: input vectors + paths + params + versions.

    ``comp_config`` already folds in the source versions token and the full
    vector config (sources, mask, blend, weights), so hashing it plus the ordered
    paths and the matrix bytes makes the fingerprint change whenever any input
    changes and stay stable otherwise. The paths must arrive in the same row
    order as ``mat`` — the caller sorts before gathering, and this function never
    sorts them itself, because the digest must track the actual row order.
    """
    h = hashlib.sha256()
    h.update(comp_config.encode("utf-8"))
    for p in paths:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\0")
    h.update(np.ascontiguousarray(mat, dtype=np.float32).tobytes())
    return h.hexdigest()


def recipe_revision(
    dataset_fp: str,
    recipe: Recipe,
    *,
    seed: int = 42,
    n_jobs: int = -1,
    versions: dict[str, Any] | None = None,
) -> str:
    """Per-recipe identity: dataset fingerprint + recipe params + effective environment.

    The dataset fingerprint captures the *input* (paths, vectors, vector config);
    it deliberately stays unrelated to the environment. But the fitted map also
    depends on the seed, the embedding-source / projection-library versions, and
    the worker count — approximate-neighbour stages (UMAP) can vary with it — so
    those are folded in here too: a seed, thread-count, or library change must
    produce a different revision even when the input is identical.
    """
    h = hashlib.sha256(dataset_fp.encode("utf-8"))
    h.update(f"seed={seed}".encode("utf-8"))
    h.update(b"\0")
    h.update(f"n_jobs={n_jobs}".encode("utf-8"))
    h.update(b"\0")
    if versions:
        for k in sorted(versions):
            h.update(k.encode("utf-8"))
            h.update(repr(versions[k]).encode("utf-8"))
            h.update(b"\0")
    for f in fields(recipe):
        value = getattr(recipe, f.name)
        if value is None:
            # Optional fields at their ``None`` default are not part of the
            # identity, so the pre-existing recipes keep their exact revisions.
            continue
        h.update(f.name.encode("utf-8"))
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _source_versions(feature_cache) -> dict[str, int]:
    from .clap import EMBEDDING_VERSION
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_VERSION
    from .rhythm_features import RHYTHM_VERSION
    from .supervised import cache_version

    return {
        "clap": EMBEDDING_VERSION,
        "features": FEATURES_VERSION,
        "maest": MAEST_VERSION,
        "rhythm": RHYTHM_VERSION,
        "supervised": cache_version(feature_cache),
    }


def _backend_info() -> dict[str, str]:
    import platform

    info: dict[str, str] = {"python": platform.python_version()}
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import openTSNE
        info["openTSNE"] = openTSNE.__version__
    except Exception:
        pass
    try:
        import umap as _umap
        info["umap"] = _umap.__version__
    except Exception:
        pass
    return info


# ── atomic writes ─────────────────────────────────────────────────────────────


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── generation ────────────────────────────────────────────────────────────────


def build_manifest(
    *,
    status: str,
    dataset_fingerprint: str,
    count: int,
    sources: tuple[str, ...],
    source_weights: dict[str, float],
    required_sources: tuple[str, ...],
    feature_mask: np.ndarray,
    features_blend: float,
    seed: int,
    reference_id: str,
    backend: dict[str, str],
    versions: dict[str, int],
    recipes: dict[str, dict],
    gather: float,
    total: float,
) -> dict:
    """Assemble the generation manifest dict (the caller writes it atomically).

    ``backend`` / ``versions`` are the already-computed environment snapshots
    (projection-library versions vs. embedding-source versions); ``recipes`` is
    the per-recipe result dict keyed by id, in fit order.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset": {"count": count, "fingerprint": dataset_fingerprint},
        "sources": list(sources),
        "source_weights": {k: float(v) for k, v in source_weights.items()},
        "required_sources": list(required_sources),
        "feature_mask": [int(x) for x in feature_mask],
        "features_blend": float(features_blend),
        "seed": seed,
        "reference_id": reference_id,
        "backend": backend,
        "versions": versions,
        "notes": {
            "umap_jit": (
                "UMAP's first call includes a Numba JIT compile, so its "
                "timing on a fresh process is inflated."
            ),
        },
        "recipes": {
            rid: {
                "status": r["status"],
                **({"reason": r.get("reason")} if r.get("reason") else {}),
                **(
                    {
                        "label": r["label"],
                        "file": f"{rid}.json",
                        "revision": r["revision"],
                        "tracks": r["tracks"],
                        "timings": r["timings"],
                        "aligned_to": r.get("aligned_to"),
                        "input_fingerprint": r["input_fingerprint"],
                        "source_weights": r["source_weights"],
                    }
                    if r["status"] == "complete"
                    else {}
                ),
            }
            for rid, r in recipes.items()
        },
        "timings": {"gather": gather, "total": total},
        "created_at": _iso_now(),
    }


def _fit_one_recipe(
    *,
    rid: str,
    recipe: Recipe,
    mat: np.ndarray,
    fp: str,
    gather_t: float,
    variants: dict[str, tuple[np.ndarray, str, float] | None],
    paths: list[str],
    n: int,
    ref_id: str,
    ref_state: tuple[list[str], np.ndarray] | None,
    seed: int,
    n_jobs: int,
    env_versions: dict[str, Any],
    source_weights: dict[str, float],
    output_dir: str,
) -> tuple[dict, tuple[list[str], np.ndarray] | None]:
    """Fit, align, normalise and write ONE recipe's artifact.

    Returns ``(manifest entry, updated reference state)``. The reference state is
    ``(ref_paths, ref_coords)`` when *rid* is the reference recipe (the caller
    keeps it for the remaining recipes), else ``None``. Any failure — a missing
    dependency, a non-finite fit or alignment, a write error — records a clean
    per-recipe error entry and the run continues.
    """
    if n < MIN_TRACKS:
        return (
            {
                "status": "skipped",
                "reason": (
                    f"too few tracks ({n} < {MIN_TRACKS}): the t-SNE perplexity "
                    "clamp cannot hold"
                ),
            },
            None,
        )
    if recipe.source_weights is not None:
        if rid not in variants or variants[rid] is None:
            return (
                {
                    "status": "error",
                    "reason": (
                        "variant input path set diverged from the control; "
                        "refusing to fit/align on a different track universe"
                    ),
                },
                None,
            )
        mat_rid, input_fp_rid, gather_rid = variants[rid]
    else:
        mat_rid, input_fp_rid, gather_rid = mat, fp, gather_t

    ref_paths = ref_state[0] if ref_state is not None else None
    ref_coords = ref_state[1] if ref_state is not None else None
    try:
        coords, timings = fit_recipe(mat_rid, recipe, seed=seed, n_jobs=n_jobs)
        timings["gather"] = gather_rid
        aligned_to: str | None = None
        new_ref: tuple[list[str], np.ndarray] | None = None
        if rid == ref_id:
            new_ref = (list(paths), coords)
            timings["alignment"] = 0.0
        else:
            t0 = time.perf_counter()
            if ref_coords is not None and ref_paths is not None:
                coords = orthogonal_align_to(coords, paths, ref_coords, ref_paths)
                aligned_to = ref_id
            timings["alignment"] = time.perf_counter() - t0

        normed = normalise_aspect(coords)
        positions = [
            {"path": paths[i], "x": float(normed[i, 0]), "y": float(normed[i, 1])}
            for i in range(n)
        ]
        timings["total"] = (
            timings.get("pca", 0.0)
            + timings.get("tsne", 0.0)
            + timings.get("umap", 0.0)
            + timings.get("alignment", 0.0)
        )
        # The revision keys on the recipe's *actual* input fingerprint plus its
        # explicit params (including the source mix), so a different mix for the
        # same gallery dataset gets its own revision and rating identity.
        revision = recipe_revision(
            input_fp_rid, recipe, seed=seed, n_jobs=n_jobs, versions=env_versions
        )
        actual_weights = {
            k: float(v)
            for k, v in (
                dict(recipe.source_weights)
                if recipe.source_weights is not None
                else source_weights
            ).items()
        }
        artifact = {
            "id": rid,
            "label": recipe.label,
            "revision": revision,
            "dataset_fingerprint": fp,
            "input_fingerprint": input_fp_rid,
            "source_weights": actual_weights,
            "recipe": recipe_params(recipe),
            "tracks": n,
            "aligned_to": aligned_to,
            "timings": timings,
            "positions": positions,
        }
        _atomic_write_json(os.path.join(output_dir, f"{rid}.json"), artifact)
    except MissingDependencyError as e:
        return {"status": "error", "reason": str(e)}, None
    except Exception as e:
        log.exception("recipe %s failed", rid)
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}, None
    return (
        {
            "status": "complete",
            "label": recipe.label,
            "revision": revision,
            "tracks": n,
            "timings": timings,
            "aligned_to": aligned_to,
            "input_fingerprint": input_fp_rid,
            "source_weights": actual_weights,
            "file": f"{rid}.json",
        },
        new_ref,
    )


def run_generation(
    *,
    feature_cache,
    infos: list[dict],
    output_dir: str,
    sources: tuple[str, ...],
    source_weights: dict[str, float],
    required_sources: tuple[str, ...],
    feature_mask: np.ndarray,
    features_blend: float,
    seed: int = 42,
    n_jobs: int = -1,
    recipe_ids: list[str] | None = None,
    reference_id: str | None = None,
    limit: int | None = None,
    on_progress: Callable | None = None,
) -> dict[str, Any]:
    """Gather once, fit every requested recipe, and write the artifacts.

    Every non-reference map is aligned (orthogonal rotation/reflection only) onto
    the ``reference_id`` recipe, which is fitted first and auto-included if it was
    not already requested. Default reference is :data:`BASELINE_ID`.

    ``on_progress(stage, done, total)`` receives coarse progress (a string stage
    and counts) for the CLI's verbose output.

    Returns the final manifest dict. Only ``completed`` recipes are written and
    listed as valid; a partial run leaves the manifest ``in_progress`` so a reader
    never mistakes stale artifacts from a previous dataset for this one.
    """
    from .layout import prepare_projection_input

    os.makedirs(output_dir, exist_ok=True)

    ref_id = reference_id or BASELINE_ID
    if ref_id not in RECIPES_BY_ID:
        raise ValueError(
            f"unknown reference recipe id: {ref_id!r} "
            f"(choose from {', '.join(RECIPES_BY_ID)})"
        )

    chosen = resolve_recipe_set(recipe_ids, ref_id)

    # A mix recipe declares a complete weight vector; when the run gathers a
    # *subset* of sources, any positive-weight source outside that subset would be
    # silently dropped (only enabled sources contribute). Reject the run before
    # any gather or artifact write rather than record a map that claims a mix it
    # never actually used.
    validate_recipe_sources_for(chosen, sources)

    # 1. Control gather with the passed (default) weights → the gallery dataset
    # fingerprint. Every recipe without its own ``source_weights`` fits this
    # matrix, and the reference (default) map always does, so the artifact
    # ``dataset_fingerprint`` stays the gallery's control fingerprint.
    t_gather = time.perf_counter()
    inp = prepare_projection_input(
        infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        source_weights=source_weights,
        required_sources=required_sources,
    )
    gather_t = time.perf_counter() - t_gather

    paths = list(inp.paths)
    mat = inp.mat
    order: list[int] | None = None
    if limit is not None and limit > 0 and len(paths) > limit:
        order = sorted(range(len(paths)), key=lambda i: paths[i])
        order = order[: int(limit)]
        paths = [paths[i] for i in order]
        mat = mat[order]
    n = len(paths)

    comp_config = inp.comp_config
    fp = dataset_fingerprint(paths, mat, comp_config)

    # 2. Variant gathers: each recipe with its own explicit source mix gets its
    # own composite matrix through the same shared production helper, with only
    # the weights overlaid — no other parameter changes. The variant's track
    # universe must match the control's exactly (same paths, same order): a
    # different set would make the maps incomparable and is rejected outright,
    # never silently sliced or faked to the control's center.
    variants: dict[str, tuple[np.ndarray, str, float] | None] = {}
    for rid in chosen:
        recipe = RECIPES_BY_ID[rid]
        if recipe.source_weights is None:
            continue
        t_variant = time.perf_counter()
        inp_v = prepare_projection_input(
            infos,
            feature_cache,
            sources=sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            source_weights=dict(recipe.source_weights),
            required_sources=required_sources,
        )
        v_gather = time.perf_counter() - t_variant
        v_paths = list(inp_v.paths)
        v_mat = inp_v.mat
        if order is not None and len(v_paths) > limit:
            v_paths = [v_paths[i] for i in order]
            v_mat = v_mat[order]
        if v_paths != paths:
            # Path-set mismatch: reject the variant, no alignment/slicing/fake
            # center. It is recorded as an error in the fit loop below.
            variants[rid] = None
            continue
        v_fp = dataset_fingerprint(v_paths, v_mat, inp_v.comp_config)
        variants[rid] = (v_mat, v_fp, v_gather)

    def _progress(stage: str, done: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    t_total = time.perf_counter()
    results: dict[str, dict] = {}
    ref_state: tuple[list[str], np.ndarray] | None = None

    # Environment that determines each fitted map: the embedding-source versions
    # (a source's cached vectors) and the projection-library versions (the fit
    # primitives). Folded into every recipe's revision so a changed environment
    # invalidates a previous run's artifact even with identical input.
    source_versions = _source_versions(feature_cache)
    backend_info = _backend_info()
    env_versions: dict[str, Any] = {**source_versions, **backend_info}

    def _write_manifest(status: str) -> None:
        manifest = build_manifest(
            status=status,
            dataset_fingerprint=fp,
            count=n,
            sources=sources,
            source_weights=source_weights,
            required_sources=required_sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            seed=seed,
            reference_id=ref_id,
            backend=backend_info,
            versions=source_versions,
            recipes=results,
            gather=gather_t,
            total=time.perf_counter() - t_total,
        )
        _atomic_write_json(os.path.join(output_dir, "manifest.json"), manifest)

    _write_manifest("in_progress")
    _progress("gather", 0, n)

    for i, rid in enumerate(chosen):
        _progress(rid, i, len(chosen))
        entry, new_ref = _fit_one_recipe(
            rid=rid,
            recipe=RECIPES_BY_ID[rid],
            mat=mat,
            fp=fp,
            gather_t=gather_t,
            variants=variants,
            paths=paths,
            n=n,
            ref_id=ref_id,
            ref_state=ref_state,
            seed=seed,
            n_jobs=n_jobs,
            env_versions=env_versions,
            source_weights=source_weights,
            output_dir=output_dir,
        )
        if new_ref is not None:
            ref_state = new_ref
        results[rid] = entry
        _write_manifest("in_progress")
        _progress(rid, i + 1, len(chosen))

    _write_manifest("complete")
    _progress("complete", len(chosen), len(chosen))
    return manifest_dict(output_dir)


def manifest_dict(output_dir: str) -> dict | None:
    """Read the current manifest, or ``None`` when it does not exist or is malformed."""
    path = os.path.join(output_dir, "manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_completed(output_dir: str) -> dict[str, dict]:
    """Completed recipes of the current manifest, whatever its overall status.

    A long generation writes the manifest ``in_progress`` while each recipe's
    artifact lands, so the gallery can already show the finished maps of the
    current run. The manifest file is replaced atomically, so it always belongs
    to the current generation; staleness is guarded per-recipe by the
    fingerprint/revision check in :func:`load_recipe_json`. An unfinished recipe
    (anything not ``complete``) is excluded.
    """
    manifest = manifest_dict(output_dir)
    if not manifest:
        return {}
    recipes = manifest.get("recipes") or {}
    return {
        rid: r
        for rid, r in recipes.items()
        if isinstance(r, dict) and r.get("status") == "complete"
    }


def load_recipe_json(output_dir: str, recipe_id: str) -> dict | None:
    """Read and validate one completed recipe artifact, or ``None``.

    The current manifest may be ``in_progress``/``partial``/``failed`` during a
    long generation: a completed recipe of that same run is still served, but
    only when its dataset fingerprint *and* revision match the manifest's, so a
    stale file from a previous dataset or environment can never be served. A
    recipe the manifest does not mark ``complete`` is rejected.
    """
    manifest = manifest_dict(output_dir)
    if not manifest:
        return None
    recipes = manifest.get("recipes") or {}
    entry = recipes.get(recipe_id)
    if not isinstance(entry, dict) or entry.get("status") != "complete":
        return None
    expected_fp = manifest.get("dataset_fingerprint")
    expected_rev = entry.get("revision")
    path = os.path.join(output_dir, f"{recipe_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("positions"), list):
        return None
    if data.get("id") != recipe_id:
        return None
    if expected_fp and data.get("dataset_fingerprint") != expected_fp:
        return None
    if expected_rev and data.get("revision") != expected_rev:
        return None
    # A source-mix variant records its own actual input fingerprint (distinct
    # from the gallery dataset fingerprint). When the manifest entry records one,
    # the artifact must match it; a manifest/artifact pair from before this field
    # existed (neither records it) still passes the older checks above.
    expected_input_fp = entry.get("input_fingerprint")
    if expected_input_fp and data.get("input_fingerprint") != expected_input_fp:
        return None
    positions = data["positions"]
    for pos in positions:
        if not isinstance(pos, dict):
            return None
        x, y = pos.get("x"), pos.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
    return data


def _iso_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
