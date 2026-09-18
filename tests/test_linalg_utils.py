"""The SVD helper's fallback and the folder-PC guard against degenerate blocks.

The real trigger is a finite but numerically rank-deficient folder block, where LAPACK's
``gesdd`` (what ``np.linalg.svd`` drives) raises "SVD did not converge" while scipy's
``gesvd`` converges. The fallback is exercised deterministically here by forcing the
default driver to fail.
"""

from __future__ import annotations

import numpy as np

from backend.embeddings import linalg_utils
from backend.embeddings.semantic_weighting import _folder_orthogonal_pc_directions


def test_svd_singular_matches_numpy_on_a_normal_matrix():
    mat = np.random.default_rng(0).normal(size=(10, 6))
    _, s, vt = linalg_utils.svd_singular(mat)
    _, s_ref, vt_ref = np.linalg.svd(mat, full_matrices=False)
    assert np.allclose(s, s_ref)
    assert vt.shape == vt_ref.shape


def test_svd_singular_falls_back_when_gesdd_fails(monkeypatch):
    mat = np.random.default_rng(1).normal(size=(12, 5))
    _, s_ref, _ = np.linalg.svd(mat, full_matrices=False)

    def boom(*_a, **_k):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", boom)
    _, s, vt = linalg_utils.svd_singular(mat)
    assert np.allclose(s, s_ref)
    assert vt.shape == (5, 5)


def test_folder_pc_directions_skip_a_non_finite_block():
    # A NaN row must be skipped, not fed to the SVD (which would raise).
    block = np.random.default_rng(2).normal(size=(10, 6))
    block[0, 0] = np.nan
    d_unit = np.zeros(6)
    d_unit[0] = 1.0
    assert (
        _folder_orthogonal_pc_directions(
            block, d_unit, max_components=2, min_second_sv_ratio=0.18
        )
        == []
    )
