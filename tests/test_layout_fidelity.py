"""Unit tests for the neighbourhood-recall fidelity harness.

The harness scores a 2-D layout against its high-dimensional embedding; these
tests pin the score's definition (an identical layout scores 1.0, a shuffled one
drops below 1.0) and its degenerate-input behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings.layout_fidelity import neighbourhood_recall


def _clustered(n: int, d: int = 2, seed: int = 0) -> np.ndarray:
    """A matrix with two well-separated clusters, so neighbours are unambiguous."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.01, size=(n // 2, d))
    b = rng.normal(loc=5.0, scale=0.01, size=(n - n // 2, d))
    return np.vstack([a, b]).astype(np.float32)


def test_perfect_layout_scores_one():
    hi = _clustered(40)
    # The layout *is* the embedding: every neighbour is trivially kept.
    assert neighbourhood_recall(hi, hi, k=5) == pytest.approx(1.0)


def test_shuffled_layout_scores_below_one():
    hi = _clustered(40)
    lo = hi.copy()
    # Move one point across the clusters: some of its neighbours are lost.
    lo[0] = lo[-1]
    score = neighbourhood_recall(hi, lo, k=5)
    assert 0.0 <= score < 1.0


def test_neighbourhood_recall_is_bounded_by_k():
    hi = _clustered(20)
    # k above n-1 is clamped rather than raising.
    assert neighbourhood_recall(hi, hi, k=1000) == pytest.approx(1.0)


def test_mismatched_or_empty_inputs_score_zero():
    hi = _clustered(10)
    lo = _clustered(9)
    assert neighbourhood_recall(hi, lo, k=3) == 0.0
    assert neighbourhood_recall(np.empty((0, 4)), np.empty((0, 2)), k=3) == 0.0
    # A 1-D input is malformed, not a two-column layout.
    assert neighbourhood_recall(hi[:, 0], hi[:, 0], k=3) == 0.0


def test_sampling_is_deterministic():
    # An 8-D embedding laid out on its first two dims is not a perfect match,
    # so the sampled score is a real number rather than a trivial 1.0.
    hi = _clustered(60, d=8)
    lo = np.asarray(hi[:, :2], dtype=np.float32)
    a = neighbourhood_recall(hi, lo, k=5, sample=20, seed=1)
    b = neighbourhood_recall(hi, lo, k=5, sample=20, seed=1)
    assert 0.0 <= a < 1.0
    assert a == b
