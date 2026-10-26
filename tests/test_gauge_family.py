"""Stable gauge families, uniform normalisation and persisted anchors."""

from __future__ import annotations

import types

import numpy as np
import pytest

from backend.embeddings import layout, lens_gauge
from backend.embeddings.projection_config import PROJECTION_LENSES
from backend.services.lens_layout import compute_lens_projection, gauge_scope_for
from backend.services.projection_request import (
    projection_params_for_lens,
    projection_params_for_recipe,
)


class _Cache:
    """In-memory stand-in for FeatureCache's metadata + layout tables."""

    def __init__(self) -> None:
        self.meta: dict[str, str] = {}
        self.layouts: dict[str, list[dict]] = {}

    def get_meta(self, key: str) -> str | None:
        return self.meta.get(key)

    def set_meta(self, key: str, value: object) -> None:
        self.meta[key] = str(value)

    def get_layout(self, revision: str) -> list[dict] | None:
        rows = self.layouts.get(revision)
        return [dict(p) for p in rows] if rows is not None else None

    def put_layout(self, revision: str, positions: list[dict]) -> None:
        self.layouts[revision] = [dict(p) for p in positions]


@pytest.fixture(autouse=True)
def _clean():
    lens_gauge.reset_gauge_cache()
    with layout._cache_lock:
        layout._revision_layout_cache.clear()
    yield
    lens_gauge.reset_gauge_cache()
    with layout._cache_lock:
        layout._revision_layout_cache.clear()


def _paths(n: int = 48, root: str = "lib") -> list[str]:
    return [f"{root}/genre/{i}.mp3" for i in range(n)]


def _cloud(n: int = 48, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 2))


def _rot(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _positions(paths: list[str], coords: np.ndarray) -> list[dict]:
    return [
        {"path": p, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
        for i, p in enumerate(paths)
    ]


def test_family_is_stable_across_lenses_and_a_track_subset():
    """The family keys on root scope + generations, never the exact track set or
    the lens's sources: a clap lens and the genre lens over the same library
    share one frame, and a folder subset stays in it."""
    paths = _paths()
    subset = paths[:32]
    assert layout._gauge_family(None, paths, None, 2) == layout._gauge_family(
        None, subset, None, 2
    )
    # A different scope or dimensionality is a different frame.
    assert layout._gauge_family(None, paths, None, 2) != layout._gauge_family(
        None, paths, None, 3
    )
    assert layout._gauge_family("lib", paths, None, 2) != layout._gauge_family(
        "lib\nother", paths, None, 2
    )
    # The source-version part is lens-independent: all projection generations.
    token = layout._source_versions_token(
        None, sources=("maest_layer7", "mert_l19")
    )
    assert layout._gauge_family(None, paths, None, 2).endswith(token)
    assert layout._gauge_family(None, paths, None, 2).startswith(
        f"2d{lens_gauge.GAUGE_VERSION2}|"
    )
    assert layout._gauge_family(None, paths, None, 3).startswith(
        f"3d{lens_gauge.GAUGE_VERSION2}.{layout.GAUGE_3D_VERSION}|"
    )


def test_subset_aligns_to_the_family_anchor_across_rotation_and_reflection():
    paths = _paths()
    pts = layout._normalise_coords(_cloud())
    family = layout._gauge_family(None, paths, None, 2)
    lens_gauge.gauge_layout(family, paths, pts)
    keep = list(range(0, len(paths), 2))
    moved = pts[keep] * np.array([-1.0, 1.0])
    moved = (0.6 * (_rot(73) @ moved.T)).T + np.array([3.0, -2.0])
    sub_paths = [paths[i] for i in keep]
    out = lens_gauge.gauge_layout(family, sub_paths, moved)
    _assert_same_orientation(layout._normalise_coords(out), pts[keep])


def test_axis_normalise_fills_every_axis_and_handles_a_degenerate_axis():
    coords = np.array([[0.0, 0.0], [10.0, 2.0], [4.0, 1.0]])
    out = layout._normalise_coords(coords)
    assert np.allclose(np.ptp(out, axis=0), 1.0)  # elongated cloud fills both axes
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)

    flat = np.array([[1.0, 3.0], [2.0, 3.0], [5.0, 3.0]])
    filled = layout._normalise_coords(flat)
    assert np.allclose(filled[:, 1], 0.5)  # zero-span axis centres, never divides


def test_axis_normalise_is_degenerate_safe_for_non_finite_span():
    coords = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 4.0]])
    out = layout._normalise_coords(coords)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(np.ptp(out[:, 1]), 1.0)


def _assert_same_orientation(a: np.ndarray, b: np.ndarray) -> None:
    """Axis-wise correlation, robust to the final per-axis display stretch."""
    for axis in range(a.shape[1]):
        corr = float(np.corrcoef(a[:, axis], b[:, axis])[0, 1])
        assert corr > 0.999, f"axis {axis} orientation differs: corr={corr}"


def test_gauge_aligns_orientation_and_ignores_scale():
    """Rotation, reflection and translation are undone; uniform scale is not."""
    paths = _paths(80)
    pts = layout._normalise_coords(_cloud(80, seed=3))
    family = layout._gauge_family(None, paths, None, 2)
    lens_gauge.gauge_layout(family, paths, pts)
    reflected = pts * np.array([-1.0, 1.0])
    moved = 0.35 * (_rot(-58) @ reflected.T).T + np.array([4.0, -2.0])
    out = lens_gauge.gauge_layout(family, paths, moved)
    assert np.allclose(layout._normalise_coords(out), pts, atol=1e-6)
    assert lens_gauge.gauge_stats()["reflected"] == 1
    # The served anchor already fills every axis, so the final per-axis display
    # normalisation is idempotent and keeps the aligned orientation.
    served = layout._normalise_coords(out)
    assert np.allclose(served, pts)
    assert np.allclose(np.ptp(served, axis=0), 1.0)


def test_tiny_layout_never_becomes_the_anchor():
    paths = _paths(4)
    pts = _cloud(4)
    family = layout._gauge_family(None, paths, None, 2)
    out = lens_gauge.gauge_layout(family, paths, pts)
    assert np.allclose(out, pts)
    assert lens_gauge.reference(family) is None
    assert lens_gauge.gauge_stats()["anchor"] == 0


def test_cache_hit_seeds_a_persisted_anchor_and_survives_reset():
    cache = _Cache()
    paths = _paths()
    pts = layout._normalise_coords(_cloud())
    cache.put_layout("rev-a", _positions(paths, pts))
    family = layout._gauge_family(None, paths, cache, 2)
    meta_key = layout._anchor_meta_key(family)

    # First read is a cache hit: it must register the family's frame and persist
    # the pointer, not merely return coordinates.
    hit = layout.revision_layout_cached("rev-a", cache)
    assert hit is not None
    assert cache.get_meta(meta_key) == "rev-a"
    assert lens_gauge.reference(family) is not None

    # Cold start: no in-memory frame and no LRU, only the persisted pointer.
    lens_gauge.reset_gauge_cache()
    with layout._cache_lock:
        layout._revision_layout_cache.clear()
    cache.put_layout(
        "rev-b",
        _positions(paths, 0.4 * (_rot(90) @ pts.T).T + np.array([2.0, 5.0])),
    )
    layout.revision_layout_cached("rev-b", cache)
    assert cache.get_meta(meta_key) == "rev-a"  # an existing live anchor is kept
    subset = paths[:24]
    moved = 0.4 * (_rot(90) @ pts[:24].T).T + np.array([2.0, 5.0])
    out = lens_gauge.gauge_layout(family, subset, moved)
    _assert_same_orientation(layout._normalise_coords(out), pts[:24])


def test_deleted_anchor_row_falls_back_to_the_current_layout():
    cache = _Cache()
    paths = _paths()
    pts = layout._normalise_coords(_cloud())
    cache.put_layout("rev-a", _positions(paths, pts))
    layout.revision_layout_cached("rev-a", cache)
    family = layout._gauge_family(None, paths, cache, 2)
    meta_key = layout._anchor_meta_key(family)
    assert cache.get_meta(meta_key) == "rev-a"

    del cache.layouts["rev-a"]
    lens_gauge.reset_gauge_cache()
    cache.put_layout("rev-b", _positions(paths, pts))
    layout.revision_layout_cached("rev-b", cache)
    assert cache.get_meta(meta_key) == "rev-b"


def test_serialized_2d_and_3d_coordinates_keep_their_frame():
    """The cache round-trip must not renormalise; 3-D keeps its stored x,y and z."""
    cache = _Cache()
    paths = _paths()
    pts2 = layout._normalise_coords(_cloud())
    cache.put_layout("rev-2d", _positions(paths, pts2))
    out2 = layout.revision_layout_cached("rev-2d", cache)
    assert out2 is not None
    back2 = np.asarray([[p["x"], p["y"]] for p in out2])
    assert np.allclose(back2, pts2.astype(np.float32), atol=1e-7)

    xyz = np.column_stack([pts2, np.linspace(-2.0, 3.0, len(paths))])
    cache.put_layout(
        "rev-3d",
        [
            {"path": p, "x": float(xyz[i, 0]), "y": float(xyz[i, 1]),
             "z": float(xyz[i, 2])}
            for i, p in enumerate(paths)
        ],
    )
    out3 = layout.revision_layout_cached("rev-3d", cache, dims=3)
    assert out3 is not None
    back3 = np.asarray([[p["x"], p["y"], p["z"]] for p in out3])
    assert np.allclose(back3[:, :2], pts2.astype(np.float32), atol=1e-7)

    bounded = layout._normalise_coords(back3)
    assert np.allclose(np.ptp(bounded[:, :2], axis=0), 1.0)
    assert bounded[:, 2].min() == 0.0 and bounded[:, 2].max() == 1.0


def test_gauge_scope_uses_configured_library_roots_and_falls_back_when_empty():
    assert gauge_scope_for(None) is None
    assert gauge_scope_for([]) is None
    assert gauge_scope_for(["b", "a", "a"]) == "library:a|b"


def test_gauge_scope_fallback_keeps_relative_roots_and_separates_absolute_trees():
    """Without configured roots the fallback must not merge unrelated absolute
    libraries under their shared empty first component."""
    assert layout._gauge_scope_for(["a/x.mp3", "a/sub/y.mp3"], None) == "a"
    assert layout._gauge_scope_for(["a/x.mp3"], "library:a") == "library:a"
    assert (
        layout._gauge_scope_for(["/libA/x.mp3", "/libA/sub/y.mp3"], None)
        == "abs:/libA"
    )
    assert layout._gauge_scope_for(["/libA/x.mp3"], None) != layout._gauge_scope_for(
        ["/libB/y.mp3"], None
    )


def test_every_lens_and_a_folder_subset_share_the_configured_scope():
    """The route and the prefetch pass the same library scope for every lens, so
    a lens switch or a folder request never re-anchors the family."""
    calls: list[str | None] = []

    class _Emb:
        def compute_projection(self, infos, *_a, **_k):
            calls.append(_k.get("gauge_scope"))
            return [{"path": t["path"], "x": 0.0, "y": 0.0} for t in infos]

    versions = types.SimpleNamespace(
        clap=1, features=1, maest=1, rhythm=1, supervised=0,
        maest_layer7=1, mert_l19=1,
    )
    lenses = [projection_params_for_lens(l.sources, l.weights) for l in PROJECTION_LENSES]
    lenses.append(projection_params_for_recipe("genre_full", genre_full_enabled=True))
    subsets = [
        [{"path": "a/x.mp3", "fingerprint": "fp-x"}],
        [{"path": "a/sub/y.mp3", "fingerprint": "fp-y"}],
        [{"path": "b/z.mp3", "fingerprint": "fp-z"}],
    ]
    for i, proj in enumerate(lenses):
        compute_lens_projection(
            subsets[i % len(subsets)],
            None,
            proj,
            embeddings_mod=_Emb(),
            batch_fetch_maps=lambda *_a, **_k: {},
            layout_revision_for_projection=lambda *_a, **_k: f"rev-{i}",
            source_versions=versions,
            embedding_version=1,
            library_root_ids=["a"],
        )
    assert calls == ["library:a"] * len(lenses)


def test_warm_hit_restores_the_scoped_reference_after_reset():
    """A warm cache hit seeds the stable family's persisted anchor, so a restart's
    next cold lens aligns to the pre-restart frame instead of re-anchoring."""
    cache = _Cache()
    paths = _paths()
    pts = layout._normalise_coords(_cloud())
    scope = "library:lib"
    cache.put_layout("rev-a", _positions(paths, pts))
    family = layout._gauge_family(scope, paths, cache, 2)

    layout.revision_layout_cached("rev-a", cache, gauge_scope=scope, dims=2)
    assert cache.get_meta(layout._anchor_meta_key(family)) == "rev-a"
    assert lens_gauge.reference(family) is not None

    lens_gauge.reset_gauge_cache()
    with layout._cache_lock:
        layout._revision_layout_cache.clear()
    moved = 0.4 * (_rot(90) @ pts.T).T + np.array([2.0, 5.0])
    cache.put_layout("rev-b", _positions(paths, moved))
    layout.revision_layout_cached("rev-b", cache, gauge_scope=scope, dims=2)
    assert cache.get_meta(layout._anchor_meta_key(family)) == "rev-a"

    out = lens_gauge.gauge_layout(family, paths[:24], moved[:24])
    _assert_same_orientation(layout._normalise_coords(out), pts[:24])


def test_tiny_first_layout_never_anchors_a_scoped_family():
    paths = _paths(4)
    pts = _cloud(4)
    scope = "library:lib"
    family = layout._gauge_family(scope, paths, None, 2)
    out = lens_gauge.gauge_layout(family, paths, pts)
    assert np.allclose(out, pts)
    assert lens_gauge.reference(family) is None
    assert lens_gauge.gauge_stats()["anchor"] == 0


def test_malformed_anchor_positions_are_skipped():
    cache = _Cache()
    paths = _paths()
    bad = [{"path": p, "x": float("nan"), "y": 0.0} for p in paths]
    family = layout._gauge_family("library:lib", paths, cache, 2)
    layout._store_gauge_anchor(family, cache, "rev-bad", bad, 2)
    assert cache.get_meta(layout._anchor_meta_key(family)) is None
