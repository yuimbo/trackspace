"""The scikit-learn t-SNE fallback pre-reduces with PCA.

Raw composite vectors are ~2,000-D and the library is past the mlx-vis point cap,
so the fallback used to run Barnes-Hut on the raw matrix — minutes-slow at 15k
tracks, long enough that the real layout never lands and the map stays on the PCA
preview. The fallback must mirror the openTSNE path and reduce to ~50 dims first.
"""

from __future__ import annotations

import types

import numpy as np

import backend.embeddings.projection_backends as pb


def _matrix(n_rows: int = 60, n_cols: int = 200) -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.standard_normal((n_rows, n_cols)).astype(np.float32)


def test_sklearn_tsne_pca_dim_default_and_env(monkeypatch):
    assert pb._sklearn_tsne_pca_dim(2000) == 50
    # Never wider than the input.
    assert pb._sklearn_tsne_pca_dim(10) == 9
    monkeypatch.setenv("TRACKSPACE_SKLEARN_TSNE_PCA_DIM", "0")
    assert pb._sklearn_tsne_pca_dim(2000) is None


def test_fallback_reduces_before_sklearn_fit(monkeypatch):
    captured: dict[str, int] = {}

    def fake_reduce(mat: np.ndarray, dim: int):
        captured["dim"] = dim
        return mat[:, :dim].copy(), object()

    def fake_tsne(**_kwargs):
        def fit_transform(x: np.ndarray) -> np.ndarray:
            captured["input_cols"] = int(x.shape[1])
            return np.zeros((x.shape[0], 2), dtype=np.float32)

        return types.SimpleNamespace(fit_transform=fit_transform)

    monkeypatch.setattr(pb, "_use_cuml_projection", lambda: False)
    monkeypatch.setattr(pb, "_use_mlx_vis_projection", lambda: False)
    monkeypatch.setattr(pb, "_pca_reduce", fake_reduce)
    monkeypatch.setattr("sklearn.manifold.TSNE", fake_tsne)

    out = pb._project_tsne_fallback(_matrix())
    assert captured == {"dim": 50, "input_cols": 50}
    assert out.shape == (60, 2)
