"""A lens switch must not look like the map flipped."""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import lens_gauge
from backend.embeddings.layout import _normalise_coords


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
    pts = _normalise_coords(_cloud())
    paths = [f"t{i}" for i in range(len(pts))]
    lens_gauge.gauge_layout("fam", paths, pts)
    moved = (2.5 * (_rot(37) @ pts.T)).T + np.array([4.0, -3.0])
    out = lens_gauge.gauge_layout("fam", paths, moved)
    # Orientation is restored; the final per-axis display normalisation then
    # removes the input's uniform scale.
    assert np.allclose(_normalise_coords(out), pts, atol=1e-6)


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


# ─── 3-D head-on gauge ───────────────────────────────────────


def _cloud3(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def _rot3(rng):
    """A random proper 3-D rotation (QR of a random matrix, det fixed to +1)."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _ref_from_cloud3(pts, seed=0, scale=2.5):
    """A random proper rotation + scale + offset of *pts*; return the 2-D xy."""
    rng = np.random.default_rng(seed)
    rot = _rot3(rng)
    ref_full = scale * (rot @ pts.T).T + rng.normal(size=3)
    return ref_full[:, :2]


def test_head_on_fit_recovers_reference_xy():
    pts = _cloud3(200, seed=11)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_xy = _ref_from_cloud3(pts, seed=7, scale=2.5)
    out = lens_gauge.head_on_fit(paths, pts, paths, ref_xy)
    assert out is not None
    assert out.shape == (200, 3)
    assert np.abs(out[:, :2] - ref_xy).max() < 1e-6
    # Similarity: output distances are the input's times the fit's scale.
    d_in = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_out = np.linalg.norm(out[:, None] - out[None], axis=-1)
    m = d_in > 1e-9
    assert np.allclose(d_out[m] / d_in[m], d_out[m].max() / d_in[m].max(), rtol=1e-6)


def test_head_on_fit_partial_overlap_still_aligns_all():
    pts = _cloud3(200, seed=13)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_xy = _ref_from_cloud3(pts, seed=2, scale=0.8)
    keep = list(range(0, 200, 2))  # 50% of the tracks as the reference
    ref_paths = [paths[i] for i in keep]
    out = lens_gauge.head_on_fit(paths, pts, ref_paths, ref_xy[keep])
    assert out is not None
    assert np.abs(out[keep][:, :2] - ref_xy[keep]).max() < 1e-6
    # The transform applies to every track, shared or not.
    d_in = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_out = np.linalg.norm(out[:, None] - out[None], axis=-1)
    m = d_in > 1e-9
    assert np.allclose(d_out[m] / d_in[m], d_out[m].max() / d_in[m].max(), rtol=1e-6)


def test_head_on_fit_too_little_overlap_is_none_and_layout_unchanged():
    pts = _cloud3(60, seed=17)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_paths = [f"z{i}" for i in range(20)]  # disjoint: 0 shared tracks
    ref_xy = np.zeros((20, 2))
    assert lens_gauge.head_on_fit(paths, pts, ref_paths, ref_xy) is None
    out = lens_gauge.gauge_layout_3d("fam", paths, pts, ref_paths, ref_xy)
    assert np.allclose(out, pts)
    assert lens_gauge.gauge_stats()["skipped"] == 1


def test_head_on_fit_collinear_is_none_and_layout_unchanged():
    t = np.linspace(0, 1, 50)
    pts = np.column_stack([t, 2.0 * t, -t])
    paths = [f"t{i}" for i in range(len(pts))]
    rng = np.random.default_rng(0)
    ref_xy = rng.normal(size=(50, 2))
    assert lens_gauge.head_on_fit(paths, pts, paths, ref_xy) is None
    out = lens_gauge.gauge_layout_3d("fam", paths, pts, paths, ref_xy)
    assert np.allclose(out, pts)


def test_gauge_layout_3d_first_call_anchors_and_z_sign_is_fixed():
    pts = _cloud3(200, seed=23)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_xy = _ref_from_cloud3(pts, seed=5, scale=1.3)
    head = lens_gauge.head_on_fit(paths, pts, paths, ref_xy)
    assert head is not None
    out1 = lens_gauge.gauge_layout_3d("fam", paths, pts, paths, ref_xy)
    assert np.allclose(out1, head)
    assert lens_gauge.gauge_stats()["anchor"] == 1
    # The same cloud mirrored in z: the head-on fit gives the opposite z, so the
    # gauge flips it back to the anchor's sign — xy untouched by the flip.
    mirrored = pts * np.array([1.0, 1.0, -1.0])
    out2 = lens_gauge.gauge_layout_3d("fam", paths, mirrored, paths, ref_xy)
    assert np.abs(out2[:, :2] - ref_xy).max() < 1e-6
    z2, z1 = out2[:, 2] - out2[:, 2].mean(), out1[:, 2] - out1[:, 2].mean()
    corr = float((z2 * z1).sum()) / float(np.linalg.norm(z2) * np.linalg.norm(z1))
    assert corr > 0.99
    assert lens_gauge.gauge_stats()["aligned"] == 1
    assert lens_gauge.gauge_stats()["reflected"] == 1


def test_gauge_layout_3d_anchor_and_z_sign_come_from_2d_ref():
    """The head-on fit pins xy to the 2-D reference, not to the family's 2-D
    anchor: two families with the same head-on fit but different 2-D layouts
    keep their own xy frames."""
    pts = _cloud3(120, seed=41)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_a = _ref_from_cloud3(pts, seed=1, scale=1.0)
    ref_b = _ref_from_cloud3(pts, seed=2, scale=1.0)
    out_a = lens_gauge.gauge_layout_3d("fam-a", paths, pts, paths, ref_a)
    out_b = lens_gauge.gauge_layout_3d("fam-b", paths, pts, paths, ref_b)
    assert np.abs(out_a[:, :2] - ref_a).max() < 1e-6
    assert np.abs(out_b[:, :2] - ref_b).max() < 1e-6


def test_reset_gauge_cache_clears_3d_anchors():
    pts = _cloud3(100, seed=31)
    paths = [f"t{i}" for i in range(len(pts))]
    ref_xy = _ref_from_cloud3(pts, seed=0, scale=1.0)
    head = lens_gauge.head_on_fit(paths, pts, paths, ref_xy)
    lens_gauge.gauge_layout_3d("fam", paths, pts, paths, ref_xy)
    lens_gauge.reset_gauge_cache()
    # After the reset the mirrored cloud registers its own anchor (no flip).
    mirrored = pts * np.array([1.0, 1.0, -1.0])
    out = lens_gauge.gauge_layout_3d("fam", paths, mirrored, paths, ref_xy)
    assert lens_gauge.gauge_stats()["anchor"] == 1
    z = out[:, 2] - out[:, 2].mean()
    zh = head[:, 2] - head[:, 2].mean()
    corr = float((z * zh).sum()) / float(np.linalg.norm(z) * np.linalg.norm(zh))
    assert corr < -0.99
