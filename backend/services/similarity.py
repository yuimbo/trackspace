"""Cosine similarity over the cached composite vectors.

The map, the clusters, and this index all describe the same weighted space: the
matrix is built with the same :func:`clustering.build_weighted_matrix` recipe
(standardise → L2-normalise per block → ``sqrt(weight)``), so "nearest" and
"anti" mean what the user sees. Read-only over ``FeatureCache`` — nothing
re-runs an encoder — and the built matrix is cached by
``(clustering revision, weights, library mtime)`` so a burst of inspect clicks
pays the gather once.

This is deliberately **not** an embedding source and never feeds projection or
clustering; it is an inspection aid.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from backend.embeddings.clustering import build_weighted_matrix

log = logging.getLogger(__name__)

#: Rebuild the cached matrix after this long even if the key has not changed.
CACHE_TTL_SECONDS = 120.0

#: Refuse to hold a matrix larger than this (float32 elements ≈ 4 bytes each).
MAX_CACHED_ELEMENTS = 120_000_000  # ~480 MB


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a floor, so a zero row cannot divide by zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-9, out=norms)
    return mat / norms


class SimilarityIndex:
    """Lazily-built, keyed cosine index over the weighted composite vectors.

    The key is ``(clustering revision, source weights, library mtime signature)``:
    a re-cluster, a weight change, or a library change each invalidate the cached
    matrix, and a TTL bounds staleness when none of those move.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key: tuple[Any, ...] | None = None
        self._built_at: float = 0.0
        self._paths: list[str] = []
        self._matrix: np.ndarray | None = None
        self._row: dict[str, int] = {}

    def lookup(
        self, key: tuple[Any, ...]
    ) -> tuple[list[str], np.ndarray, dict[str, int]] | None:
        """Return the cached index for *key*, or ``None`` when it must be rebuilt."""
        with self._lock:
            if (
                self._matrix is not None
                and self._key == key
                and time.time() - self._built_at < CACHE_TTL_SECONDS
            ):
                return self._paths, self._matrix, self._row
        return None

    def build(
        self,
        key: tuple[Any, ...],
        paths: list[str],
        blocks: dict[str, np.ndarray],
        weights: dict[str, float] | None,
    ) -> tuple[list[str], np.ndarray, dict[str, int]] | None:
        """Weight, normalise, and cache the matrix; ``None`` when unusable."""
        matrix, _used, _effective = build_weighted_matrix(blocks, weights)
        if matrix.size == 0:
            log.info("Similarity index: no usable feature sources")
            return None
        if matrix.shape[0] != len(paths):
            log.warning(
                "Similarity index: %d rows for %d paths", matrix.shape[0], len(paths)
            )
            return None
        if matrix.size > MAX_CACHED_ELEMENTS:
            log.info("Similarity index: too large to cache (%d elements)", matrix.size)
            return None
        norm = _l2_normalize_rows(matrix)
        row = {p: i for i, p in enumerate(paths)}
        with self._lock:
            self._key = key
            self._built_at = time.time()
            self._paths = list(paths)
            self._matrix = norm
            self._row = row
        return self._paths, self._matrix, self._row

    def rank(
        self,
        paths: list[str],
        matrix: np.ndarray,
        row: int,
        k: int,
        mode: str,
    ) -> list[dict[str, Any]]:
        """Rank other tracks by cosine similarity to row *row*.

        ``near`` returns the highest similarities, ``far`` the lowest (the
        "anti-track"); the queried row is always excluded, and ties keep
        path-sorted order because both sorts are stable.
        """
        scores = matrix @ matrix[row]
        order = (
            np.argsort(scores, kind="stable")
            if mode == "far"
            else np.argsort(-scores, kind="stable")
        )
        picks = [int(i) for i in order if int(i) != row][:k]
        return [{"path": paths[i], "score": float(scores[i])} for i in picks]
