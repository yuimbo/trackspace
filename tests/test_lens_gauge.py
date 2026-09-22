"""A lens switch must not look like the map flipped."""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import lens_gauge


@pytest.fixture(autouse=True)
def _clean():
    lens_gauge.reset_gauge_cache()
    yield
    lens_gauge.reset_gauge_cache()


def _cloud(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


def _rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_first_layout_becomes_the_anchor_unchanged():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    out = lens_gauge.gauge_layout("fam", paths, pts)
    assert np.allclose(out, pts)
    assert lens_gauge.gauge_stats()["anchor"] == 1


def test_a_reflected_layout_is_flipped_back():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    flipped = pts * np.array([1.0, -1.0])
    out = lens_gauge.gauge_layout("fam", paths, flipped)
    assert np.allclose(out, pts, atol=1e-6)
    assert lens_gauge.gauge_stats()["reflected"] == 1


def test_rotation_and_scale_are_undone():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    moved = (2.5 * (_rot(37) @ pts.T)).T + np.array([4.0, -3.0])
    out = lens_gauge.gauge_layout("fam", paths, moved)
    assert np.allclose(out, pts, atol=1e-6)


def test_alignment_preserves_shape():
    """A similarity transform must not distort: distance ratios stay constant."""
    pts = _cloud(80, seed=3)
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    moved = (0.4 * (_rot(110) @ pts.T)).T
    out = lens_gauge.gauge_layout("fam", paths, moved)
    d0 = np.linalg.norm(moved[:, None] - moved[None], axis=-1)
    d1 = np.linalg.norm(out[:, None] - out[None], axis=-1)
    m = d0 > 1e-9
    ratio = d1[m] / d0[m]
    assert ratio.std() < 1e-9


def test_partial_overlap_still_aligns():
    pts = _cloud(60, seed=5)
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    keep = list(range(0, 60, 2))
    sub_paths = [paths[i] for i in keep]
    moved = (_rot(90) @ pts[keep].T).T
    out = lens_gauge.gauge_layout("fam", sub_paths, moved)
    assert np.allclose(out, pts[keep], atol=1e-6)


def test_too_little_overlap_is_left_alone():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    other = _cloud(20, seed=9)
    other_paths = [f"z{i}" for i in range(len(other))]
    out = lens_gauge.gauge_layout("fam", other_paths, other)
    assert np.allclose(out, other)
    assert lens_gauge.gauge_stats()["skipped"] == 1


def test_families_are_independent():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam-a", paths, pts)
    flipped = pts * np.array([-1.0, 1.0])
    out = lens_gauge.gauge_layout("fam-b", paths, flipped)
    assert np.allclose(out, flipped)
    assert lens_gauge.gauge_stats()["anchor"] == 2


def test_degenerate_input_is_returned_unchanged():
    pts = _cloud()
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    bad = np.zeros((4, 3))
    assert lens_gauge.gauge_layout("fam", ["a", "b", "c", "d"], bad).shape == (4, 3)


def test_track_set_key_is_lens_independent():
    """Every lens over the same tracks must land in one gauge family.

    The family key is derived from the track set only; if it ever picked up a
    lens-varying part (sources/mask/weights) each lens would anchor itself and the
    alignment would silently never fire.
    """
    from backend.embeddings.layout import track_set_key
    from backend.embeddings.projection_config import PROJECTION_LENSES

    paths = [f"r/{i}.mp3" for i in range(50)]
    assert len({track_set_key(paths) for _ in PROJECTION_LENSES}) == 1
    assert track_set_key(paths) != track_set_key(paths + ["r/extra.mp3"])
