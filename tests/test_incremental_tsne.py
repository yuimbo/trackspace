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


def test_uncached_fits_do_not_evict_the_live_fit():
    """The Jacobian's throwaway fits must not evict the map's own fit.

    Each perturbation is a one-off config; caching it would evict the base fit past
    the LRU cap and make the next real layout a full refit.
    """
    mat = _matrix(120)
    pb._project_tsne(mat, paths=_paths(120), config_key="base")
    for i in range(pb._OPENTSNE_FIT_CACHE_MAX + 2):
        pb._project_tsne(
            mat, paths=_paths(120), config_key=f"pert-{i}", cache_fit=False
        )
    assert pb.tsne_fit_stats()["evict"] == 0
    pb._project_tsne(mat, paths=_paths(120), config_key="base")
    assert pb.tsne_fit_stats()["reuse"] == 1


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


def test_dims3_fit_returns_3d_coords():
    mat = _matrix(120)
    out = pb._project_tsne(mat, paths=_paths(120), config_key="cfg", dims=3)
    assert out.shape == (120, 3)
    assert np.isfinite(out).all()
    assert pb.tsne_fit_stats()["fit"] == 1


def test_dims2_and_dims3_fits_share_no_cache_key():
    """A 2-D and a 3-D embedding for the same vector config must not reuse each
    other: the fit cache key carries ``|dims=3`` for 3-D."""
    mat = _matrix(120)
    pb._project_tsne(mat, paths=_paths(120), config_key="cfg")
    out3 = pb._project_tsne(mat, paths=_paths(120), config_key="cfg", dims=3)
    assert out3.shape == (120, 3)
    assert pb.tsne_fit_stats()["fit"] == 2
    # The 2-D fit is still cached under its own (unchanged) key.
    out2 = pb._project_tsne(mat, paths=_paths(120), config_key="cfg")
    assert out2.shape == (120, 2)
    assert pb.tsne_fit_stats()["reuse"] == 1


def test_dims3_transform_places_new_tracks():
    mat = _matrix(120)
    full = pb._project_tsne(mat, paths=_paths(120), config_key="cfg", dims=3)
    assert full.shape == (120, 3)

    added = _matrix(10, seed=7)
    sup_paths = _paths(120) + _paths(10, offset=1000)
    extended = pb._project_tsne(
        np.vstack([mat, added]), paths=sup_paths, config_key="cfg", dims=3,
    )

    assert extended.shape == (130, 3)
    assert np.isfinite(extended).all()
    # Known tracks are sliced from the cached embedding — they must not move.
    assert np.array_equal(extended[:120], full)
    assert pb.tsne_fit_stats()["transform"] == 1
    assert pb.tsne_fit_stats()["fit"] == 1


class _FakeFitStore:
    """Minimal stand-in for FeatureCache's fit-blob sidecar."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def get_tsne_fit(self, key: str) -> bytes | None:
        return self.blobs.get(key)

    def put_tsne_fit(self, key: str, blob: bytes) -> None:
        self.blobs[key] = blob


def test_fit_survives_a_restart_via_disk():
    """A restart must transform, not refit, when the blob is on disk."""
    store = _FakeFitStore()
    mat = _matrix(120)
    pb._project_tsne(
        mat, paths=_paths(120), config_key="cfg", feature_cache=store
    )
    assert pb.tsne_fit_stats()["fit"] == 1

    # Simulate a restart: the in-process cache is empty, the disk is not.
    pb.reset_tsne_fit_cache()
    out = pb._project_tsne(
        mat, paths=_paths(120), config_key="cfg", feature_cache=store
    )
    assert out.shape == (120, 2)
    stats = pb.tsne_fit_stats()
    assert stats["fit"] == 0
    assert stats["reuse"] == 1


def test_persisted_fit_transforms_new_tracks_after_restart():
    store = _FakeFitStore()
    mat = _matrix(120)
    pb._project_tsne(
        mat, paths=_paths(120), config_key="cfg", feature_cache=store
    )

    pb.reset_tsne_fit_cache()
    added = _matrix(10, seed=7)
    sup_paths = _paths(120) + _paths(10, offset=1000)
    out = pb._project_tsne(
        np.vstack([mat, added]),
        paths=sup_paths,
        config_key="cfg",
        feature_cache=store,
    )
    assert out.shape == (130, 2)
    assert pb.tsne_fit_stats()["transform"] == 1
    assert pb.tsne_fit_stats()["fit"] == 0


def test_unreadable_persisted_fit_refits():
    store = _FakeFitStore()
    store.blobs["cfg"] = b"not a fit"
    mat = _matrix(120)
    out = pb._project_tsne(
        mat, paths=_paths(120), config_key="cfg", feature_cache=store
    )
    assert out.shape == (120, 2)
    assert pb.tsne_fit_stats()["fit"] == 1


def test_stale_persisted_fit_ignored_after_recipe_change():
    """A blob written before the recipe bump (old, un-namespaced key) is a miss.

    The fit cache key carries the ``|opentsne_recipe=…`` namespace, so an
    embedding fitted under an older iteration budget must never be loaded after a
    ``PROJECTION_RECIPE_VERSION`` bump — it refits and persists under the new key
    without touching the stale blob.
    """
    store = _FakeFitStore()
    stale_key = "cfg"  # the pre-namespace key
    store.blobs[stale_key] = b"stale blob from an older recipe"
    mat = _matrix(120)
    out = pb._project_tsne(
        mat, paths=_paths(120), config_key="cfg", feature_cache=store
    )
    assert out.shape == (120, 2)
    assert pb.tsne_fit_stats()["fit"] == 1
    # The stale blob is left untouched; the fresh fit lives under the namespaced key.
    assert store.blobs[stale_key] == b"stale blob from an older recipe"
    ns_key = f"cfg|opentsne_recipe={pb.PROJECTION_RECIPE_VERSION}"
    assert ns_key in store.blobs


def test_fit_uses_production_iter_budget(monkeypatch):
    """The production openTSNE fit passes the configured iteration budget."""
    import openTSNE

    from backend.embeddings.projection_config import (
        OPENTSNE_EARLY_EXAGGERATION_ITER,
        OPENTSNE_N_ITER,
    )

    calls: dict = {}

    class _RecordingTSNE:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit(self, x):
            return np.zeros((len(x), 2), dtype=np.float32)

    monkeypatch.setattr(openTSNE, "TSNE", _RecordingTSNE)
    pb._fit_tsne(_matrix(120))
    assert calls["n_iter"] == OPENTSNE_N_ITER
    assert calls["early_exaggeration_iter"] == OPENTSNE_EARLY_EXAGGERATION_ITER


def test_disk_fit_not_written_for_jacobian_fits():
    """The Jacobian's throwaway fits must not touch the persisted blob."""
    store = _FakeFitStore()
    mat = _matrix(60)
    pb._project_tsne(
        mat,
        paths=_paths(60),
        config_key="pert",
        cache_fit=False,
        feature_cache=store,
    )
    assert store.blobs == {}
