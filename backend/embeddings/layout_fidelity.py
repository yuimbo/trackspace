"""Neighbourhood-recall fidelity for a 2-D layout.

The one measure the projection's quality claims are scored against: for each point,
the fraction of its *k* nearest neighbours in the high-dimensional embedding that
are still among its *k* nearest neighbours in the 2-D layout. t-SNE only preserves
local neighbourhoods, so this is the property that matters — cluster counts and
distance-ratio spreads are not.

Pure and dependency-light: the only heavy import (sklearn's ``NearestNeighbors``)
is lazy, so importing this module costs nothing. *sample* bounds the O(n log n)
neighbour search at library scale.

Reference numbers measured on the real library (combined lens, n=14,163, PCA-50,
recall@10): full scikit-learn fit 0.342, full openTSNE fit 0.360, openTSNE
*placed* points 0.313, subsample-50%-and-interpolate 0.259. See
``audio_pipeline.md`` → *Incremental t-SNE (openTSNE)*.
"""

from __future__ import annotations

import numpy as np


def _knn_indices(mat: np.ndarray, k: int, query_idx: np.ndarray) -> np.ndarray:
    """k nearest neighbours of each row of ``mat[query_idx]`` within ``mat``."""
    from sklearn.neighbors import NearestNeighbors  # lazy — heavy import

    nn = NearestNeighbors(n_neighbors=k + 1).fit(mat)
    return nn.kneighbors(mat[query_idx], return_distance=False)[:, 1:]


def neighbourhood_recall(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    k: int = 10,
    sample: int | None = None,
    seed: int = 42,
) -> float:
    """Fraction of each point's *k* high-dim neighbours kept in 2-D.

    *sample* bounds the O(n log n) neighbour search at library scale: it scores
    a random *sample* of points against the whole set. A missing, empty, or
    mismatched input returns ``0.0`` rather than raising — a layout without
    fidelity is a score of zero, never a crashed harness.
    """
    hi = np.asarray(high_dim, dtype=np.float32)
    lo = np.asarray(low_dim, dtype=np.float32)
    if hi.ndim != 2 or lo.ndim != 2 or hi.shape[0] != lo.shape[0]:
        return 0.0
    n = int(hi.shape[0])
    if n < 2:
        return 0.0
    k = max(1, min(int(k), n - 1))
    if sample is not None and 0 < int(sample) < n:
        query_idx = np.random.default_rng(seed).choice(n, int(sample), replace=False)
    else:
        query_idx = np.arange(n)
    a = _knn_indices(hi, k, query_idx)
    b = _knn_indices(lo, k, query_idx)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))
