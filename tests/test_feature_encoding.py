"""Golden vectors for the perceptual feature encodings.

These pin the *topology* (what is a neighbour of what), not the exact floats, so a
future tuning of the radii does not break them — but a change that moves a relative
major/minor pair apart, or a half/double tempo pair off the same ray, does.
"""

from __future__ import annotations

import math

import numpy as np

from backend.embeddings.feature_encoding import (
    BPM_SPIRAL_R_MAX,
    BPM_SPIRAL_R_MIN,
    CAMELOT_MAJOR_RADIUS,
    CAMELOT_MINOR_RADIUS,
    ENCODED_FEATURE_DIM,
    FEATURE_ENCODING_VERSION,
    bpm_spiral_xy,
    bpm_spiral_xy_norm,
    camelot_xy,
    encode_feature_vector,
)
from backend.embeddings.librosa_audio_features import _KEY_TO_FIFTHS
from backend.tuning import BPM_HI, BPM_LO


def _key_xy(key: str, mode: float) -> tuple[float, float]:
    """Camelot position for a named key via the raw circle-of-fifths encoding."""
    pos = _KEY_TO_FIFTHS[key]
    angle = 2.0 * math.pi * pos / 12.0
    return camelot_xy(math.cos(angle), math.sin(angle), mode)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_version_is_pinned():
    """Derived caches key on this; bump it when the encoding changes."""
    assert FEATURE_ENCODING_VERSION == 1
    assert ENCODED_FEATURE_DIM == 6


def _angle(p: tuple[float, float]) -> float:
    return math.atan2(p[1], p[0])


def test_relative_major_minor_share_a_ray():
    """C major and A minor are the same key signature — same ray, inner vs outer."""
    c_major = _key_xy("C", 1.0)
    a_minor = _key_xy("A", 0.0)
    assert math.isclose(_angle(c_major), _angle(a_minor), abs_tol=1e-9)
    assert math.hypot(*a_minor) < math.hypot(*c_major)


def test_relative_pairs_align_across_the_wheel():
    for major, minor in (("G", "E"), ("D", "B"), ("F", "D")):
        assert math.isclose(
            _angle(_key_xy(major, 1.0)), _angle(_key_xy(minor, 0.0)), abs_tol=1e-9
        )


def test_fifth_neighbours_are_angularly_adjacent():
    """C and G are a fifth apart; C and F# are tritone-distant."""
    d_fifth = _dist(_key_xy("C", 1.0), _key_xy("G", 1.0))
    d_tritone = _dist(_key_xy("C", 1.0), _key_xy("F#", 1.0))
    assert d_fifth < d_tritone


def test_relative_pair_is_closer_than_unrelated_same_mode():
    """Relative major/minor share a ray, so they beat two unrelated majors."""
    d_relative = _dist(_key_xy("C", 1.0), _key_xy("A", 0.0))
    d_unrelated = _dist(_key_xy("C", 1.0), _key_xy("A", 1.0))
    assert d_relative < d_unrelated


def test_major_and_minor_rings_are_distinct():
    for key in ("C", "G", "D"):
        assert math.isclose(
            math.hypot(*_key_xy(key, 1.0)), CAMELOT_MAJOR_RADIUS, abs_tol=1e-6
        )
        assert math.isclose(
            math.hypot(*_key_xy(key, 0.0)), CAMELOT_MINOR_RADIUS, abs_tol=1e-6
        )


def test_spiral_doubles_share_a_ray():
    """60/120/240 BPM are octaves apart: same angle, increasing radius."""
    a = bpm_spiral_xy(60.0)
    b = bpm_spiral_xy(120.0)
    c = bpm_spiral_xy(240.0)
    ang = lambda p: math.atan2(p[1], p[0])  # noqa: E731
    assert math.isclose(ang(a), ang(b), abs_tol=1e-9)
    assert math.isclose(ang(b), ang(c), abs_tol=1e-9)
    assert math.hypot(*a) < math.hypot(*b) < math.hypot(*c)


def test_spiral_radius_bounds():
    assert math.isclose(math.hypot(*bpm_spiral_xy(BPM_LO)), BPM_SPIRAL_R_MIN, abs_tol=1e-9)
    assert math.isclose(math.hypot(*bpm_spiral_xy(BPM_HI)), BPM_SPIRAL_R_MAX, abs_tol=1e-9)
    # Outside the window clamps to the ends rather than escaping the frame.
    assert math.isclose(math.hypot(*bpm_spiral_xy(30.0)), BPM_SPIRAL_R_MIN, abs_tol=1e-9)
    assert math.isclose(math.hypot(*bpm_spiral_xy(400.0)), BPM_SPIRAL_R_MAX, abs_tol=1e-9)


def test_spiral_from_norm_matches_bpm():
    for bpm in (60.0, 90.0, 128.0, 174.0, 200.0):
        norm = (bpm - BPM_LO) / (BPM_HI - BPM_LO)
        assert bpm_spiral_xy_norm(norm) == bpm_spiral_xy(bpm)


def test_encode_feature_vector_shape_and_order():
    raw = np.array([0.0, 1.0, 0.0, 1.0, 0.25, 0.75], dtype=np.float32)  # C major, 60 BPM
    out = encode_feature_vector(raw)
    assert out is not None
    assert out.shape == (ENCODED_FEATURE_DIM,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # C major is on the +x axis of the outer ring.
    assert math.isclose(float(out[0]), CAMELOT_MAJOR_RADIUS, abs_tol=1e-6)
    assert math.isclose(float(out[1]), 0.0, abs_tol=1e-6)
    # The two linear descriptors ride through untouched.
    assert float(out[4]) == 0.25
    assert float(out[5]) == 0.75


def test_encode_feature_vector_rejects_missing():
    assert encode_feature_vector(None) is None
    assert encode_feature_vector(np.zeros(3, dtype=np.float32)) is None
