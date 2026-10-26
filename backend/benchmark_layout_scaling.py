#!/usr/bin/env python3
"""Reproducible wall-clock scaling benchmark for the layout-evaluation recipes.

It measures how the uncached ``fit_recipe`` cost scales with library size, from a
small subset up to the full eligible track set, reusing the exact canonical
composite input the interactive projection uses (``prepare_projection_input``
with the frozen sources/weights/mask/blend/required sources).

Scope: below ~10k tracks only. openTSNE auto-selects its Barnes-Hut engine below
10k and no gradient-method / FFT branch is implemented, so a library of
``>= 10k`` eligible tracks is refused rather than silently measured on the wrong
engine. "Eligible" is the *library-root* cohort (``load_infos`` scope
``library``, external roots excluded), matching the interactive library fit; the
all-roots production set can exceed 10k and is covered by
``benchmark_production_projection`` instead. The bound is on track count, not
output width: 2-D and 3-D maps use the same BH engine, so a paired dimension run
keeps the same below-10k scope. The reported numbers are timings on this machine
with this data, not claims of cross-library speedups.

Protocol per size: a seeded nested prefix of one row permutation is the track set;
every recipe gets one **warmup** fit (excluded from timings and scoring); then
``repeats`` measured **rounds**, each round visiting every recipe in its own
deterministic shuffle (so no recipe systematically runs first/last). The
numerical BLAS/OpenMP threads are bounded with ``threadpoolctl`` around the whole
size's fit block, and each fit's wall-clock ``t0`` is taken inside that context
immediately before the call, excluding threadpoolctl setup. Fidelity
(neighbourhood recall @k10, sample 256) is scored on the first measured map of
each recipe, outside the timed region.

This tool is read-only over the production caches (it gathers the composite
matrix exactly as the interactive projection does) and writes a single JSON
artifact. It never touches the production layout cache, the t-SNE fit cache, the
gallery artifacts, or MP3 tags.

Usage::

    python -m backend.benchmark_layout_scaling [--data-dir DIR] [--output FILE]
        [--sizes 500,1000,2000,4000] [--recipes id,id,...]
        [--repeats 3] [--threads 8] [--seed 42]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr

from backend.cache import TrackCache
from backend.embeddings.feature_cache import (
    LIBRARY_MEMORY_BYTES,
    LIBRARY_MEMORY_ENTRIES,
    FeatureCache,
)
from backend.embeddings.layout_evaluation import (
    RECIPES_BY_ID,
    _atomic_write_json,
    _backend_info,
    dataset_fingerprint,
    fit_recipe,
    pca_reduce,
    recipe_params,
    recipe_revision,
    resolve_multilevel_schedule,
)
from backend.embeddings.layout_fidelity import neighbourhood_recall
from backend.embeddings.layout import prepare_projection_input
from backend.embeddings.projection_config import (
    FROZEN_FEATURE_MASK,
    FROZEN_SOURCES,
    FROZEN_SOURCE_WEIGHTS,
    required_sources_for,
)
from backend.precompute_layout_evaluation import _data_dir, load_infos
from backend.services.roots_registry import load_roots_state
from backend.tuning import FROZEN_FEATURES_BLEND

#: Hard ceiling for the eligible track count (Barnes-Hut scope).
BELOW_10K_LIMIT = 10000

#: Recipe methods compatible with the benchmark's canonical PCA50 t-SNE input.
#: UMAP methods and multi-stage geometry change the effective high-dim space the
#: fidelity reference is measured against, and source-mix recipes carry their own
#: composite matrix — both would mislabel the fitted input, so they are rejected.
SUPPORTED_METHODS = {"tsne", "tsne_sample_init", "tsne_multilevel_init"}

#: Default recipe set probed, all over the canonical composite input. These span
#: plain, two-level and multilevel sample-init fits at library scale. Kept as the
#: historical seven; a paired 2-D/3-D run lists its ids explicitly via
#: ``--recipes`` (the registered 3-D siblings need no special treatment here).
DEFAULT_RECIPES = (
    "tsne_early100",
    "tsne50_plain",
    "tsne_sample1000_early0",
    "tsne_sample1000_coarse10e100",
    "tsne_sample1000_coarse100e0",
    "tsne_sample100_early0",
    "tsne_chain100_500_2000_final50",
)

#: Fidelity scoring configuration. The high-dim reference is the same PCA50
#: (canonical ``pca_reduce``, same seed) that the pca_dim=50 fits run on, so the
#: score measures how well each map preserves its own fitted neighbourhoods. It
#: is indicative, not a proof of global layout quality.
METRIC = {
    "name": "neighbourhood_recall",
    "k": 10,
    "sample": 256,
    "metric": "euclidean",
    "high_dim": "pca50 (canonical pca_reduce, same seed as the fits)",
    "scope": "first measured map per recipe; indicative, not proof of global quality",
}

#: All three untimed quality metrics computed on the first measured map of each
#: recipe, outside the measured wall-clock. The neighbourhood-recall @10 stays
#: under the back-compat ``fidelity`` field; the other two are additional
#: per-recipe fields.
METRICS = {
    "neighbourhood_recall": dict(METRIC),
    "neighbourhood_recall@100": {
        "name": "neighbourhood_recall@100",
        "k": 100,
        "sample": 256,
        "metric": "euclidean",
        "high_dim": METRIC["high_dim"],
        "scope": "first measured map per recipe; indicative, not proof of global quality",
    },
    "broad_distance_spearman": {
        "name": "broad_distance_spearman",
        "n_pairs": 2048,
        "metric": "euclidean",
        "high_dim": METRIC["high_dim"],
        "scope": (
            "Spearman rank correlation over 2048 seeded random pair distances "
            "(PCA50 Euclidean vs the map's low-dimensional Euclidean distance, "
            "2-D or 3-D) on the first measured map; a rank-correlation "
            "PROXY of whether broad pairwise distances are preserved — NOT proof "
            "that t-SNE distances are globally meaningful"
        ),
    },
}


def _validate_recipes(recipe_ids: list[str]) -> None:
    """Validate ids + benchmark compatibility; raise ValueError on any problem.

    The benchmark gathers one canonical PCA50 t-SNE composite input, so it only
    accepts recipes with ``pca_dim=50``, no explicit source mix, and a plain /
    sample-init / multilevel t-SNE method. Any registered output dimensionality
    is accepted: a 2-D recipe and its explicit 3-D sibling fit the same input
    matrix and are scored by the same metrics, so a paired run validates as one
    set. Any other registered recipe would fit a different input geometry (or its
    own weighted matrix), giving a misleading fidelity reference — rejected
    before any expensive gather.
    """
    ids = list(recipe_ids)
    if not ids:
        raise ValueError("at least one recipe id is required")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate recipe id(s) in recipe_ids")
    unknown = [r for r in ids if r not in RECIPES_BY_ID]
    if unknown:
        raise ValueError(f"unknown recipe id(s): {', '.join(unknown)}")
    bad: list[str] = []
    for rid in ids:
        r = RECIPES_BY_ID[rid]
        problems: list[str] = []
        if r.pca_dim != 50:
            problems.append(f"pca_dim={r.pca_dim}")
        if r.source_weights is not None:
            problems.append("explicit source mix")
        if r.method not in SUPPORTED_METHODS:
            problems.append(f"method={r.method!r}")
        if problems:
            bad.append(f"{rid} ({'; '.join(problems)})")
    if bad:
        raise ValueError(
            "recipe(s) incompatible with the canonical PCA50 t-SNE benchmark input: "
            + "; ".join(bad)
            + " (accept only pca_dim=50, no source mix, method in "
            + "tsne/tsne_sample_init/tsne_multilevel_init)"
        )


def _resolve_sizes(requested: list[int], n: int) -> list[int]:
    """Clamp *requested* sizes to *n*, dedupe, keep ascending, ensure *n* last.

    Rejects a non-positive size outright (never silently dropped). The full
    eligible set is always measured as the largest size; anything above *n*
    clamps down to it. An empty list means the full set only.
    """
    for s in requested:
        if int(s) < 2:
            raise ValueError(f"sizes must be integers >= 2 (got {s!r})")
    sizes = sorted({min(int(s), n) for s in requested})
    if not sizes or sizes[-1] != n:
        sizes.append(n)
    return sizes


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


def _broad_pairs(
    high_dim: np.ndarray, *, n_pairs: int, seed: int
) -> list[tuple[int, int]]:
    """Deterministic distinct row pairs with a nonzero high-dim distance.

    Drawn from *high_dim* with a fixed seed, so every recipe at a size (which
    shares the same PCA50 reference) is scored on the **same** pairs. Self-pairs
    and zero-distance high-dim pairs are excluded so Spearman never sees a
    meaningless degenerate edge. The count is capped to the distinct ordered pairs
    available so a tiny dataset is not held in a long rejection loop.
    """
    n = int(high_dim.shape[0])
    target = min(int(n_pairs), n * (n - 1))
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < target and attempts < target * 50:
        attempts += 1
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a == b or (a, b) in seen:
            continue
        if float(np.linalg.norm(high_dim[a] - high_dim[b])) == 0.0:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def broad_distance_spearman(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    n_pairs: int = 2048,
    seed: int = 42,
) -> float | None:
    """Spearman rank correlation of low-dimensional pair distances against PCA50.

    Works for any registered map width (2-D or 3-D): only row Euclidean
    distances are used, never the coordinate count. A **proxy** for whether the
    layout keeps *broad* (not just local) pairwise distances ranked the way the
    high-dim space does — NOT proof that t-SNE distances are globally
    meaningful. A degenerate layout (constant or non-finite distances) yields a
    NaN/constant Spearman, returned as ``None`` (a missing score, never a fake
    zero) so the writer's ``allow_nan=False`` stays safe.
    """
    hi = np.asarray(high_dim, dtype=float)
    lo = np.asarray(low_dim, dtype=float)
    if hi.ndim != 2 or lo.ndim != 2 or hi.shape[0] != lo.shape[0] or hi.shape[0] < 2:
        return None
    pairs = _broad_pairs(hi, n_pairs=n_pairs, seed=seed)
    if len(pairs) < 3:
        return None
    hi_d = np.array([float(np.linalg.norm(hi[a] - hi[b])) for a, b in pairs])
    lo_d = np.array([float(np.linalg.norm(lo[a] - lo[b])) for a, b in pairs])
    import warnings
    from scipy.stats import ConstantInputWarning

    # A degenerate (constant) low-dimensional distance vector is handled as
    # ``None`` below, so scipy's benign constant-input warning is suppressed here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, _ = spearmanr(hi_d, lo_d)
    if not np.isfinite(rho):
        return None
    return float(rho)


@contextlib.contextmanager
def _thread_limits(threads: int):
    """Bound the numerical BLAS/OpenMP threads (threadpoolctl) to *threads*.

    threadpoolctl is a direct dependency (see requirements.txt). ``threads``
    already bounds the sklearn/openTSNE ``n_jobs``; this additionally caps the
    underlying BLAS thread pools so wall-clock is a fair reflection of the
    requested count.
    """
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=max(1, int(threads))):
        yield


def run_scaling(
    mat: np.ndarray,
    paths: list[str],
    *,
    sizes: list[int],
    recipe_ids: list[str],
    seed: int = 42,
    repeats: int = 3,
    threads: int = 8,
    fit: Callable = fit_recipe,
    score: bool = True,
    on_progress: Callable[[str, str, int, float], None] | None = None,
    on_size_done: Callable[[int, dict], None] | None = None,
) -> dict[str, Any]:
    """Benchmark ``fit`` over nested subsets of *mat*/*paths*. Pure numpy, no IO.

    *mat* is the full composite matrix (one row per path); each requested size is
    a seeded nested prefix of one row permutation. ``fit`` is injectable for
    tests (defaults to the real :func:`fit_recipe`). ``on_progress(phase, rid,
    size, wall_seconds)`` fires after each fit (``phase`` is ``"warmup"`` or
    ``"fit"``) so the first (warmup) phase is never silent; ``on_size_done(size,
    size_results)`` fires per completed size (used by the CLI for checkpoints).

    Returns a results dict (see the CLI for the JSON envelope) whose per-size
    entry holds ``recipe_order`` — one permutation per measured round — and a
    ``recipes`` dict with the accumulated per-recipe samples. Never touches the
    filesystem or the caches.
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("mat must be 2-D (tracks x features)")
    n = int(mat.shape[0])
    if n < 2:
        raise ValueError("need at least 2 tracks to fit a layout")
    if n >= BELOW_10K_LIMIT:
        raise ValueError(
            f"below-{BELOW_10K_LIMIT} Barnes-Hut scope: refusing n={n} "
            "(no gradient-method / FFT branch is implemented)"
        )
    if not paths:
        raise ValueError("paths must be a non-empty list")
    if len(paths) != n:
        raise ValueError("len(paths) must equal mat.shape[0]")
    recipe_ids = list(recipe_ids)
    _validate_recipes(recipe_ids)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    sizes = _resolve_sizes(sizes, n)
    perm = np.random.default_rng(seed).permutation(n)

    results: dict[str, Any] = {
        "eligible_tracks": n,
        "sizes": sizes,
        "seed": seed,
        "repeats": repeats,
        "threads": threads,
        "results": {},
    }

    for i, s in enumerate(sizes):
        idx = perm[:s]
        sub = mat[idx]
        # One rng per size draws the warmup order and each round's order
        # independently, so every measured round visits the recipes in its own
        # deterministic shuffle.
        size_rng = np.random.default_rng(seed + i + 1)
        warmup_order = [recipe_ids[int(p)] for p in size_rng.permutation(len(recipe_ids))]
        round_orders = [
            [recipe_ids[int(p)] for p in size_rng.permutation(len(recipe_ids))]
            for _ in range(repeats)
        ]
        samples: dict[str, list[dict[str, Any]]] = {rid: [] for rid in recipe_ids}
        scores: dict[str, dict[str, float | None]] = {}
        # Dedicated seed for the distance-pair draw (distinct from the round-order
        # rng): one seed per size, so all recipes at that size share the same pairs.
        pair_seed = seed + i + 1000

        with _thread_limits(threads):
            # PCA scoring reference, once per size, under the same thread limits
            # but outside the timed fits.
            reduced = pca_reduce(sub, 50, seed)[0] if score else None

            # Warmup: every recipe once, excluded from timings and scoring.
            for rid in warmup_order:
                recipe = RECIPES_BY_ID[rid]
                t0 = time.perf_counter()
                fit(sub, recipe, seed=seed, n_jobs=threads)
                wall = time.perf_counter() - t0
                if on_progress is not None:
                    on_progress("warmup", rid, s, wall)

            # Measured rounds: each round visits every recipe in its own order.
            for r, order in enumerate(round_orders):
                for rid in order:
                    recipe = RECIPES_BY_ID[rid]
                    t0 = time.perf_counter()
                    coords, stages = fit(sub, recipe, seed=seed, n_jobs=threads)
                    wall = time.perf_counter() - t0
                    samples[rid].append({"total": wall, "stages": stages})
                    # All quality metrics are scored once, on the first measured
                    # map of each recipe, outside the timed region.
                    if score and r == 0 and rid not in scores:
                        scores[rid] = {
                            "fidelity": neighbourhood_recall(
                                reduced, coords, k=10, sample=256, seed=seed
                            ),
                            "fidelity100": neighbourhood_recall(
                                reduced, coords, k=100, sample=256, seed=seed
                            ),
                            "broad_distance_spearman": broad_distance_spearman(
                                reduced, coords, n_pairs=2048, seed=pair_seed
                            ),
                        }
                    if on_progress is not None:
                        on_progress("fit", rid, s, wall)

        recipes_out: dict[str, Any] = {}
        for rid in recipe_ids:
            sm = samples[rid]
            stage_keys = sorted({k for x in sm for k in x["stages"]})
            sc = scores.get(rid) if score else None
            recipes_out[rid] = {
                "samples": sm,
                "timings": {
                    "total": _summary([x["total"] for x in sm]),
                    **{
                        k: _summary([x["stages"].get(k, 0.0) for x in sm])
                        for k in stage_keys
                    },
                },
                "fidelity": sc.get("fidelity") if sc else None,
                "fidelity100": sc.get("fidelity100") if sc else None,
                "broad_distance_spearman": (
                    sc.get("broad_distance_spearman") if sc else None
                ),
            }
        size_res = {"recipe_order": round_orders, "recipes": recipes_out}
        # Size-adaptive multilevel recipes resolve their effective coarse schedule
        # from the actual size *s* (not the full library n), so a run records the
        # exact (count, normal, early, perp) stages each measured fit used. Only
        # multilevel recipes have a schedule; the key is omitted otherwise.
        eff = {
            rid: resolve_multilevel_schedule(RECIPES_BY_ID[rid], s)
            for rid in recipe_ids
            if RECIPES_BY_ID[rid].method == "tsne_multilevel_init"
        }
        if eff:
            size_res["effective_schedules"] = eff
        results["results"][str(s)] = size_res
        if on_size_done is not None:
            on_size_done(s, size_res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wall-clock scaling benchmark for the layout-evaluation recipes (below ~10k tracks)."
    )
    parser.add_argument("--data-dir", default=None, help="Cache dir holding roots.json + audio_features.db (default: data/)")
    parser.add_argument("--output", default=None, help="JSON output path (default: <data-dir>/layout-evaluation/hierarchy-scaling.json)")
    parser.add_argument("--sizes", default="500,1000,2000,4000", help="Comma-separated track counts; the full eligible set is appended/deduped")
    parser.add_argument("--recipes", default=None, help="Comma-separated recipe ids (default: the scaling set; only pca_dim=50, unmixed tsne/tsne_sample_init/tsne_multilevel_init recipes are accepted)")
    parser.add_argument("--repeats", type=int, default=3, help="Measured rounds per size after one warmup fit per recipe (default 3)")
    parser.add_argument("--threads", type=int, default=8, help="n_jobs for sklearn/openTSNE and the numerical thread cap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recipe_ids = (
        [r.strip() for r in args.recipes.split(",") if r.strip()]
        if args.recipes
        else list(DEFAULT_RECIPES)
    )
    try:
        _validate_recipes(recipe_ids)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    try:
        requested_sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"--sizes must be comma-separated integers (got {args.sizes!r})") from None
    if not requested_sizes:
        raise SystemExit("--sizes must list at least one positive track count")
    if any(s < 2 for s in requested_sizes):
        raise SystemExit("--sizes must be integers >= 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    data_dir = os.path.abspath(args.data_dir or _data_dir())
    output = os.path.abspath(args.output or os.path.join(data_dir, "layout-evaluation", "hierarchy-scaling.json"))

    # Roots (persisted only — the benchmark targets the configured library).
    roots: OrderedDict[str, str] = OrderedDict()
    root_roles: OrderedDict[str, str] = OrderedDict()
    state_path = os.path.join(data_dir, "roots.json")
    if not os.path.exists(state_path):
        raise SystemExit(
            f"no roots.json at {state_path}; start the Trackspace server once "
            "to register a library root"
        )
    load_roots_state(roots, state_path, default_root=None, root_roles=root_roles)
    if not roots:
        raise SystemExit("no library roots configured (roots.json is empty)")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    feature_cache = FeatureCache(
        os.path.join(data_dir, "audio_features.db"),
        max_memory=LIBRARY_MEMORY_ENTRIES,
        max_memory_bytes=LIBRARY_MEMORY_BYTES,
    )
    track_cache = TrackCache(os.path.join(data_dir, "track_cache.db"))
    infos, n_mp3s = load_infos(roots, track_cache, root_roles)
    if not infos:
        raise SystemExit(
            "no cached track fingerprints found under the library roots; "
            "run the analysis pipeline first"
        )
    # Refuse the >= 10k Barnes-Hut ceiling before any expensive gather.
    if len(infos) >= BELOW_10K_LIMIT:
        raise SystemExit(
            f"{len(infos)} eligible library-root tracks >= {BELOW_10K_LIMIT}: the "
            "benchmark is below-10k Barnes-Hut scope and no gradient-method / "
            "FFT branch is implemented; use benchmark_production_projection for "
            "the all-roots production set"
        )
    # Invalidate a previous completed report before the expensive gather: a
    # stale ``complete: true`` must never survive into this run's first size.
    _atomic_write_json(
        output,
        {
            "tool": "layout-scaling-benchmark",
            "schema_version": 1,
            "checkpoint": {"sizes_completed": [], "complete": False},
        },
    )
    # Gather the composite once with the canonical production defaults.
    required = required_sources_for(FROZEN_SOURCES)
    inp = prepare_projection_input(
        infos,
        feature_cache,
        sources=FROZEN_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=FROZEN_FEATURES_BLEND,
        source_weights=dict(FROZEN_SOURCE_WEIGHTS),
        required_sources=required,
    )
    paths, mat = list(inp.paths), inp.mat

    n = len(paths)
    print(f"roots: {len(roots)} | eligible tracks: {n} (of {n_mp3s} mp3s)")
    print(f"recipes: {', '.join(recipe_ids)}")
    print(f"sizes: {_resolve_sizes(requested_sizes, n)}")

    fp = dataset_fingerprint(paths, mat, inp.comp_config)
    backend = {**_backend_info(), "platform": platform.platform()}
    # ``scipy.stats.spearmanr`` backs the broad-distance metric, so its version is
    # folded into the envelope's backend record for metric reproducibility — but
    # NOT into ``_backend_info`` (that would bump the gallery's recipe revisions)
    # and NOT into the per-recipe revisions below (the fitted map does not depend
    # on scipy).
    backend_envelope = {**backend, "scipy": scipy.__version__}
    envelope: dict[str, Any] = {
        "tool": "layout-scaling-benchmark",
        "schema_version": 1,
        "eligible_tracks": n,
        "n_mp3s": n_mp3s,
        "dataset_fingerprint": fp,
        "sources": list(FROZEN_SOURCES),
        "source_weights": {k: float(v) for k, v in FROZEN_SOURCE_WEIGHTS.items()},
        "seed": args.seed,
        "repeats": args.repeats,
        "threads": args.threads,
        "backend": backend_envelope,
        "recipes": {
            rid: {
                "params": recipe_params(RECIPES_BY_ID[rid]),
                "revision": recipe_revision(
                    fp, RECIPES_BY_ID[rid], seed=args.seed, n_jobs=args.threads,
                    versions=backend,
                ),
            }
            for rid in recipe_ids
        },
        "metric": dict(METRIC),
        "metrics": METRICS,
        "scope_note": (
            f"Below {BELOW_10K_LIMIT} tracks openTSNE auto-selects its Barnes-Hut "
            "engine, so the whole 500 → full-library range here runs BH; the "
            f"benchmark refuses >= {BELOW_10K_LIMIT} eligible tracks (no FFT branch). "
            "'Eligible' is the library-root cohort (external roots excluded, "
            "matching the interactive library fit); the all-roots production set is "
            "measured by benchmark_production_projection instead. "
            "The bound is track count, not map width: 2-D and 3-D recipes use the "
            "same BH engine. These are timings on this machine/this data, not "
            "cross-library speedup claims."
        ),
        "results": {},
        "checkpoint": {"sizes_completed": [], "complete": False},
    }
    _atomic_write_json(output, envelope)

    def _on_progress(phase: str, rid: str, size: int, wall: float) -> None:
        print(f"[scaling] size={size:>5} {phase:<6} {rid:<40} fit={wall:7.3f}s", flush=True)

    def _on_size_done(size: int, size_res: dict) -> None:
        envelope["results"][str(size)] = size_res
        done = envelope["checkpoint"]["sizes_completed"]
        if size not in done:
            done.append(size)
        envelope["checkpoint"]["complete"] = False
        _atomic_write_json(output, envelope)
        print(f"[scaling] size={size} done ({len(done)}/{len(envelope['results'])} checkpoints)", flush=True)

    results = run_scaling(
        mat,
        paths,
        sizes=requested_sizes,
        recipe_ids=recipe_ids,
        seed=args.seed,
        repeats=args.repeats,
        threads=args.threads,
        on_progress=_on_progress,
        on_size_done=_on_size_done,
    )
    envelope["results"] = results["results"]
    envelope["checkpoint"]["complete"] = True
    _atomic_write_json(output, envelope)
    print(f"\nWrote {output}")
    print(f"Dataset fingerprint: {fp}")


if __name__ == "__main__":
    main()
