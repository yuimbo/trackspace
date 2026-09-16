"""Tests for the librosa audio feature pipeline.

Run:  .venv/bin/python -m pytest tests/test_audio_features.py -v
  or: .venv/bin/python -m tests.test_audio_features       (standalone)

Uses two fixtures:
  test_track.mp3   — rhythmic content (good for tempo/danceability)
  test_track_2.mp3 — harmonic content (ID3: TKEY=Am, TBPM=119)
"""

import math
import os

import numpy as np
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_TRACK = os.path.join(FIXTURE_DIR, "test_track.mp3")
TEST_TRACK_HARMONIC = os.path.join(FIXTURE_DIR, "test_track_2.mp3")


def test_feature_excerpt_falls_back_when_seek_returns_empty():
    """Bad duration metadata can put the centered excerpt past EOF; librosa then
    returns an empty array without raising. Full-file decode must still succeed."""
    from unittest.mock import patch

    import backend.embeddings.librosa_audio_features as laf

    with patch.object(laf, "librosa_get_duration", return_value=600.0):
        audio, sr, warn = laf._load_feature_audio_excerpt(TEST_TRACK)
    assert audio is not None and audio.size > 0
    assert sr == laf._FEATURE_SR
    assert warn is not None
    assert "600" in warn


# ── Key detection ─────────────────────────────────────────────

def test_key_detection_harmonic():
    """Detect key on the harmonic track (ID3 says Am)."""
    import librosa
    from backend.embeddings.librosa_audio_features import detect_key

    audio, sr = librosa.load(TEST_TRACK_HARMONIC, sr=44100, mono=True)
    key, scale = detect_key(audio, sr)

    print(f"Detected key: {key} {scale}")
    assert key in ("C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B")
    assert scale in ("major", "minor")


def test_key_detection_returns_valid():
    """Key detection on the rhythmic track should still return a valid result."""
    import librosa
    from backend.embeddings.librosa_audio_features import detect_key

    audio, sr = librosa.load(TEST_TRACK, sr=44100, mono=True)
    key, scale = detect_key(audio, sr)

    print(f"Detected key: {key} {scale}")
    assert key in ("C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B")
    assert scale in ("major", "minor")


# ── Circle-of-fifths encoding ────────────────────────────────

def test_circle_of_fifths_mapping():
    from backend.embeddings.librosa_audio_features import _KEY_TO_FIFTHS

    all_keys = {"C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"}
    mapped_keys = set(_KEY_TO_FIFTHS.keys())
    assert all_keys.issubset(mapped_keys), f"Missing keys: {all_keys - mapped_keys}"

    positions = set(_KEY_TO_FIFTHS.values())
    assert positions == set(range(12)), f"Expected positions 0-11, got {positions}"


def test_circle_of_fifths_unit_circle():
    """Verify that the cos/sin encoding lies on the unit circle."""
    from backend.embeddings.librosa_audio_features import _KEY_TO_FIFTHS

    for key, pos in _KEY_TO_FIFTHS.items():
        angle = 2.0 * math.pi * pos / 12.0
        c, s = math.cos(angle), math.sin(angle)
        r = math.sqrt(c * c + s * s)
        assert abs(r - 1.0) < 1e-9, f"Key {key}: radius {r} != 1.0"
        assert -1.0 <= c <= 1.0
        assert -1.0 <= s <= 1.0


def test_harmonically_close_keys_are_geometrically_close():
    """Adjacent keys on the circle of fifths (e.g. C and G) should be
    closer in 2D than distant keys (e.g. C and F#)."""
    from backend.embeddings.librosa_audio_features import _KEY_TO_FIFTHS

    def encode(key: str) -> tuple[float, float]:
        pos = _KEY_TO_FIFTHS[key]
        angle = 2.0 * math.pi * pos / 12.0
        return math.cos(angle), math.sin(angle)

    c = encode("C")
    g = encode("G")
    fs = encode("F#")

    dist_cg = math.sqrt((c[0] - g[0])**2 + (c[1] - g[1])**2)
    dist_cfs = math.sqrt((c[0] - fs[0])**2 + (c[1] - fs[1])**2)

    print(f"C-G distance: {dist_cg:.4f}, C-F# distance: {dist_cfs:.4f}")
    assert dist_cg < dist_cfs, "C and G should be closer than C and F#"


# ── Full audio feature extraction ─────────────────────────────

def test_audio_features_shape():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK)
    print(f"Features: shape={feat.shape}, dtype={feat.dtype}, values={feat}")

    assert feat is not None
    assert feat.shape == (6,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()


def test_audio_features_shape_harmonic():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK_HARMONIC)
    print(f"Features (harmonic): shape={feat.shape}, values={feat}")

    assert feat is not None
    assert feat.shape == (6,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()


def test_audio_features_ranges():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK)
    tempo_norm, key_cos, key_sin, mode, energy_norm, danceability = feat

    assert 0.0 <= tempo_norm <= 1.0, f"tempo_norm={tempo_norm}"
    assert -1.0 <= key_cos <= 1.0, f"key_cos={key_cos}"
    assert -1.0 <= key_sin <= 1.0, f"key_sin={key_sin}"
    assert mode in (0.0, 1.0), f"mode={mode}"
    assert 0.0 <= energy_norm <= 1.0, f"energy_norm={energy_norm}"
    assert 0.0 <= danceability <= 1.0, f"danceability={danceability}"


def test_audio_features_ranges_harmonic():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK_HARMONIC)
    tempo_norm, key_cos, key_sin, mode, energy_norm, danceability = feat

    assert 0.0 <= tempo_norm <= 1.0
    assert -1.0 <= key_cos <= 1.0
    assert -1.0 <= key_sin <= 1.0
    assert mode in (0.0, 1.0)
    assert 0.0 <= energy_norm <= 1.0
    assert 0.0 <= danceability <= 1.0


def test_audio_features_deterministic():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    f1 = extract_audio_features(TEST_TRACK)
    f2 = extract_audio_features(TEST_TRACK)

    assert f1 is not None and f2 is not None
    np.testing.assert_array_equal(f1, f2)


def test_madmom_tempo_when_installed():
    from backend.embeddings import madmom_tempo as mt
    from backend.embeddings.librosa_audio_features import _load_feature_audio_excerpt

    if not mt.is_madmom_tempo_available():
        pytest.skip("madmom not installed or failed to import")
    audio, sr, _ = _load_feature_audio_excerpt(TEST_TRACK)
    assert audio is not None and sr is not None
    bpm = mt.estimate_tempo_bpm(audio, sr)
    assert bpm is not None
    assert math.isfinite(bpm)
    assert 40.0 < bpm < 250.0


# ── Standalone runner ─────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_circle_of_fifths_mapping,
        test_circle_of_fifths_unit_circle,
        test_harmonically_close_keys_are_geometrically_close,
        test_key_detection_harmonic,
        test_key_detection_returns_valid,
        test_audio_features_shape,
        test_audio_features_shape_harmonic,
        test_audio_features_ranges,
        test_audio_features_ranges_harmonic,
        test_audio_features_deterministic,
        test_madmom_tempo_when_installed,
    ]

    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
        print("  OK")

    print("\nAll tests passed.")
