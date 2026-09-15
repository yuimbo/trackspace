"""openTSNE incremental projection: fit-cache reuse and transform semantics.

The prototype fits once per *vector config* (sources/mask/blend/weights — not the
track set), so a layout update after a generation batch transforms only the new
tracks instead of refitting the library. These tests pin that behaviour down with
small matrices; the timing comparison lives in ``test_profiling_embeddings.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.projection_backends as pb


def _matrix(n_rows: int, n_cols: int = 24, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n_rows, n_cols)).astype(np.float32)


@pytest.fixture(autouse=True)
def _enable_opentsne(monkeypatch):
    monkeypatch.setenv("TRACKSPACE_OPENTSNE", "1")
    pb.reset_projection_backend_cache()
    pb.reset_tsne_fit_cache()
    yield
    pb.reset_projection_backend_cache()
    pb.reset_tsne_fit_cache()


def _paths(n: int, offset: int = 0) -> list[str]:
    return [f"t{offset + i}.mp3" for i in range(n)]


def test_cold_fit_populates_cache():
    mat = _matrix(120)
    out = pb._project_tsne(mat, paths=_paths(120), config_key="cfg")
    assert out.shape == (120, 2)
    assert np.isfinite(out).all()
    assert pb.tsne_fit_stats() == {"fit": 1, "transform": 0, "reuse": 0, "evict": 0}


def test_subset_reuses_cached_coordinates_without_compute():
    mat = _matrix(120)
    full = pb._project_tsne(mat, paths=_paths(120), config_key="cfg")
    subset = pb._project_tsne(mat[:60], paths=_paths(60), config_key="cfg")
    assert np.allclose(subset, full[:60])
    assert pb.tsne_fit_stats()["reuse"] == 1
    assert pb.tsne_fit_stats()["fit"] == 1


def test_superset_transforms_only_new_tracks():
    mat = _matrix(120)
    full = pb._project_tsne(mat, paths=_paths(120), config_key="cfg")

    added = _matrix(10, seed=7)
    sup_paths = _paths(120) + _paths(10, offset=1000)
    extended = pb._project_tsne(
        np.vstack([mat, added]), paths=sup_paths, config_key="cfg",
    )

    assert extended.shape == (130, 2)
    assert np.isfinite(extended).all()
    # Known tracks are sliced from the cached embedding — they must not move.
    assert np.array_equal(extended[:120], full)
    assert pb.tsne_fit_stats()["transform"] == 1
    assert pb.tsne_fit_stats()["fit"] == 1


def test_config_change_refits():
    mat = _matrix(120)
    pb._project_tsne(mat, paths=_paths(120), config_key="cfg-a")
    pb._project_tsne(mat, paths=_paths(120), config_key="cfg-b")
    assert pb.tsne_fit_stats()["fit"] == 2
    assert pb.tsne_fit_stats()["reuse"] == 0


def test_too_many_new_tracks_refits(monkeypatch):
    monkeypatch.setenv("TRACKSPACE_OPENTSNE_REFIT_FRACTION", "0.3")
    mat = _matrix(100)
    pb._project_tsne(mat, paths=_paths(100), config_key="cfg")

    added = _matrix(60, seed=9)
    sup_paths = _paths(100) + _paths(60, offset=1000)
    pb._project_tsne(np.vstack([mat, added]), paths=sup_paths, config_key="cfg")

    stats = pb.tsne_fit_stats()
    assert stats["fit"] == 2
    assert stats["transform"] == 0


def test_fit_cache_evicts_past_cap():
    mat = _matrix(60)
    for i in range(pb._OPENTSNE_FIT_CACHE_MAX + 1):
        pb._project_tsne(mat, paths=_paths(60), config_key=f"cfg-{i}")
    assert pb.tsne_fit_stats()["evict"] == 1
    assert len(pb._tsne_fit_cache) == pb._OPENTSNE_FIT_CACHE_MAX


def test_disabled_flag_uses_fallback(monkeypatch):
    monkeypatch.setenv("TRACKSPACE_OPENTSNE", "0")
    pb.reset_projection_backend_cache()
    mat = _matrix(60)
    out = pb._project_tsne(mat, paths=_paths(60), config_key="cfg")
    assert out.shape == (60, 2)
    assert pb.tsne_fit_stats()["fit"] == 0
    assert len(pb._tsne_fit_cache) == 0


def test_no_paths_or_key_uses_fallback():
    mat = _matrix(60)
    out = pb._project_tsne(mat)
    assert out.shape == (60, 2)
    assert len(pb._tsne_fit_cache) == 0
