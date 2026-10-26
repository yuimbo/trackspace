"""Dimension-agnostic multilevel t-SNE: expansion, level loop, fraction math.

This module is the production-reusable core of the multilevel sample-init
strategy. It knows nothing about recipes, evaluation registries, the production
projection backend, or any cache: callers supply the schedule, the per-level
fit, and the final fit as plain callables. That keeps the winning optimiser
available to production without importing the evaluation module, and keeps the
heavy libraries (openTSNE, scikit-learn) lazy.

The evaluation module resolves a :class:`~backend.embeddings.layout_evaluation.Recipe`
into the same ``(count, normal, early, perplexity)`` schedule this module
consumes, and adapts its own openTSNE fit into ``fit_level`` / ``fit_final``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

#: Initial-embedding variance scale openTSNE recommends for custom initials.
_INIT_SCALE = 1e-4


def scale_initialization(init: np.ndarray) -> np.ndarray:
    """Scale a custom initial embedding to the variance openTSNE expects."""
    scaled = np.asarray(init, dtype=np.float64)
    return scaled / (float(scaled.std(axis=0).max()) + 1e-12) * _INIT_SCALE


def expand_landmark_init(
    reduced: np.ndarray,
    landmark_idx: np.ndarray,
    landmark_coords: np.ndarray,
    target_idx: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    """Expansion init for ``target_idx`` rows from landmark coords.

    Every target row starts at its nearest landmark's position (sklearn
    ``NearestNeighbors`` over the landmark rows) plus a tiny deterministic jitter
    scaled to the landmark spread in every coordinate of the map's dimension, so
    coincident landmarks do not collapse onto one point. ``landmark_idx`` must be
    a sorted subset of the sorted ``target_idx`` — landmark rows keep their exact
    coords (no jitter).
    """
    from sklearn.neighbors import NearestNeighbors  # lazy

    nn = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(reduced[landmark_idx])
    _dist, nbr = nn.kneighbors(reduced[target_idx])
    expanded = np.asarray(landmark_coords[nbr[:, 0]], dtype=np.float32)
    rng = np.random.default_rng(seed)
    coarse_scale = float(landmark_coords.std(axis=0).max()) or 1.0
    expanded = expanded + (rng.standard_normal(
        (len(target_idx), landmark_coords.shape[1])
    ) * coarse_scale * 1e-3).astype(np.float32)
    # Landmark rows keep their exact coarse coordinates (no jitter); the subset
    # invariant makes searchsorted give the exact target row of each landmark.
    expanded[np.searchsorted(target_idx, landmark_idx)] = landmark_coords
    return expanded


def fit_multilevel(
    reduced: np.ndarray,
    schedule: list[tuple[int, int, int, float | None]],
    *,
    fit_level: Callable[[np.ndarray, int, int, float | None, np.ndarray | None], Any],
    fit_final: Callable[[np.ndarray, np.ndarray | None], Any],
    seed: int,
    n_jobs: int,
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """Run the nested multilevel sample-init chain, then a full-data final fit.

    ``schedule`` holds ``(count, normal, early, perplexity)`` coarse levels in
    ascending count order. ``fit_level(subset, normal, early, perplexity, init)``
    fits one level (``init`` is ``None`` for the first level); each later level
    starts from its nearest-landmark expansion of the previous level's coords.
    ``fit_final(full, init)`` fits the full dataset at the caller's final budget
    (``init`` is ``None`` when the schedule is empty).

    Returns ``(final_object, coords, stage_timings)`` where ``coords`` is the
    ``float32`` numpy view of the final embedding and the timings are keyed
    ``tsne_level0..`` / ``tsne_expand`` / ``tsne_final`` (seconds). An empty
    schedule is a single full-data fit. Row order follows one seeded random
    permutation and every prefix is sorted, so the chain is deterministic.
    """
    n = len(reduced)
    stage_t: dict[str, float] = {}
    if not schedule:
        t0 = time.perf_counter()
        final = fit_final(reduced, None)
        stage_t["tsne_level0"] = 0.0
        stage_t["tsne_expand"] = 0.0
        stage_t["tsne_final"] = time.perf_counter() - t0
        return final, np.asarray(final, dtype=np.float32), stage_t

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    s0, n0, e0, p0 = schedule[0]
    prev_idx = np.sort(perm[:s0])
    t0 = time.perf_counter()
    prev_coords = fit_level(reduced[prev_idx], n0, e0, p0, None)
    stage_t["tsne_level0"] = time.perf_counter() - t0

    t_expand = 0.0
    for i, (si, ni, ei, pi) in enumerate(schedule[1:], start=1):
        idx_i = np.sort(perm[:si])
        t0 = time.perf_counter()
        expanded = expand_landmark_init(
            reduced, prev_idx, prev_coords, idx_i, seed=seed, n_jobs=n_jobs
        )
        t_expand += time.perf_counter() - t0
        t0 = time.perf_counter()
        prev_coords = fit_level(reduced[idx_i], ni, ei, pi, expanded)
        stage_t[f"tsne_level{i}"] = time.perf_counter() - t0
        prev_idx = idx_i

    t0 = time.perf_counter()
    expanded_full = expand_landmark_init(
        reduced, prev_idx, prev_coords, np.arange(n), seed=seed, n_jobs=n_jobs
    )
    t_expand += time.perf_counter() - t0
    stage_t["tsne_expand"] = t_expand

    t0 = time.perf_counter()
    final = fit_final(reduced, expanded_full)
    stage_t["tsne_final"] = time.perf_counter() - t0
    return final, np.asarray(final, dtype=np.float32), stage_t


def resolve_fraction_survivors(
    fractions: tuple[float, ...],
    n: int,
) -> list[tuple[float, int]]:
    """Resolve fraction levels to ``[(fraction, count), ...]`` surviving at *n*.

    Each fraction becomes ``count = max(2, round(n·f))``; a count at or above
    *n* (and every later, larger fraction) would duplicate the final full-data
    fit so it is dropped; rounding collisions are deduped to keep counts
    strictly increasing. This is the only place a runtime ``n`` enters the
    fraction schedule, so a library of any size reuses the same recipe.
    """
    survivors: list[tuple[float, int]] = []
    prev = 0
    for f in fractions:
        c = max(2, int(round(n * float(f))))
        if c >= n:
            break  # later fractions are larger, so also >= n
        if c > prev:
            survivors.append((float(f), c))
            prev = c
    return survivors


def allocate_normal_budget(budget: int, fractions: list[float]) -> list[int]:
    """Split *budget* normal iterations across coarse levels by inverse-sqrt weight.

    Largest remainder (Hamilton) with a 1-iteration floor per level and a
    deterministic index tie-break, so a given budget + fraction list always
    resolves to the same integer split — every level gets at least 1 and the
    shares sum to exactly *budget*.
    """
    weights = [1.0 / math.sqrt(f) for f in fractions]
    total = sum(weights)
    ideal = [budget * w / total for w in weights]
    base = [max(1, int(math.floor(s))) for s in ideal]
    remainder = budget - sum(base)
    if remainder < 0:
        raise ValueError(
            f"tsne_coarse_iteration_budget {budget} is smaller than the number "
            f"of coarse levels ({len(fractions)})"
        )
    order = sorted(
        range(len(ideal)),
        key=lambda i: (-(ideal[i] - math.floor(ideal[i])), i),
    )
    for i in order:
        if remainder <= 0:
            break
        base[i] += 1
        remainder -= 1
    return base


def fraction_perplexities(
    survivors: list[tuple[float, int]],
    n: int,
    *,
    gamma: float | None,
    start_fraction: float | None,
) -> list[float]:
    """Per-level perplexities for the surviving fraction levels.

    A ``None`` gamma means a flat requested perplexity 30 on every level.
    Otherwise the gamma schedule ``p(t)=30+50·(1−t)^gamma`` with ``t`` = the
    log-scaled position of the level's count between the schedule's bottom
    anchor and ``n`` (clamped 0..1), so gamma 0 pins every coarse level at 80.
    The anchor is ``n/64`` by default and ``n·start_fraction`` when that is set
    (with log span ``1/start_fraction``), so gamma 4 on the 1/256-anchored wide
    chain stays low for longer. Each value is clamped to the openTSNE affinity
    ceiling ``(count−1)/3``. Requested curves remain with the caller; effective
    schedules expose clipping on small levels (28 rows: 9).
    """
    g = float(gamma) if gamma is not None else None
    f_start = float(start_fraction) if start_fraction is not None else 1.0 / 64.0
    log_span = math.log(1.0 / f_start)
    perps: list[float] = []
    for _f, c in survivors:
        if g is None:
            p = 30.0
        else:
            t = math.log(c / (n * f_start)) / log_span
            t = min(1.0, max(0.0, t))
            p = 30.0 + 50.0 * (1.0 - t) ** g
        perps.append(min(p, (float(c) - 1.0) / 3.0))
    return perps


def resolve_fraction_schedule(
    fractions: tuple[float, ...],
    n: int,
    *,
    normal_budget: int | None,
    gamma: float | None,
    start_fraction: float | None,
    early_iter: int = 100,
) -> list[tuple[int, int, int, float | None]]:
    """Resolve a size-adaptive fraction schedule into per-level tuples.

    Returns ``(count, normal, early, perplexity)`` for every surviving coarse
    level (``count < n``), ascending. ``normal_budget=None`` uses the default
    ``round(50·sqrt(n/count))`` rule, otherwise the budget is split by
    :func:`allocate_normal_budget`. Only the first level gets ``early_iter``
    exaggeration iterations. An empty list means no coarse level survives.
    """
    survivors = resolve_fraction_survivors(fractions, n)
    if not survivors:
        return []
    level_fractions = [f for f, _ in survivors]
    counts = [c for _, c in survivors]
    if normal_budget is None:
        normals = [int(round(50.0 * math.sqrt(n / c))) for c in counts]
    else:
        normals = allocate_normal_budget(normal_budget, level_fractions)
    earlys = [early_iter if i == 0 else 0 for i in range(len(counts))]
    perps = fraction_perplexities(
        survivors, n, gamma=gamma, start_fraction=start_fraction
    )
    return [
        (counts[i], normals[i], earlys[i], perps[i])
        for i in range(len(counts))
    ]
