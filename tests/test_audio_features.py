"""Tests for the EffNet embedding + librosa audio feature pipeline.

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


# ── Mel-spectrogram preprocessing ─────────────────────────────

def test_mel_preprocessing():
    import librosa
    from backend.audio_features import compute_mel_spectrogram, _EFFNET_SR

    audio, _ = librosa.load(TEST_TRACK, sr=_EFFNET_SR, mono=True)
    mel = compute_mel_spectrogram(audio)

    print(f"Mel shape: {mel.shape}, dtype: {mel.dtype}")
    assert mel.ndim == 2
    assert mel.shape[1] == 96, f"Expected 96 mel bins, got {mel.shape[1]}"
    assert mel.shape[0] > 0
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all(), "Mel contains NaN/Inf"
    assert mel.min() >= 0, "log10(10000*x+1) should be non-negative"


def test_mel_patching():
    import librosa
    from backend.audio_features import (
        compute_mel_spectrogram,
        patch_mel_spectrogram,
        _EFFNET_SR,
        _EFFNET_PATCH_FRAMES,
    )

    audio, _ = librosa.load(TEST_TRACK, sr=_EFFNET_SR, mono=True)
    mel = compute_mel_spectrogram(audio)
    patches = patch_mel_spectrogram(mel)

    print(f"Patches shape: {patches.shape}")
    assert patches.ndim == 3
    assert patches.shape[1] == _EFFNET_PATCH_FRAMES
    assert patches.shape[2] == 96
    assert patches.shape[0] >= 1, "Should produce at least one patch"


def test_mel_patching_short_audio():
    """A very short signal should still produce one zero-padded patch."""
    from backend.audio_features import patch_mel_spectrogram, _EFFNET_PATCH_FRAMES

    short_mel = np.random.rand(10, 96).astype(np.float32)
    patches = patch_mel_spectrogram(short_mel)

    assert patches.shape == (1, _EFFNET_PATCH_FRAMES, 96)
    assert np.allclose(patches[0, :10], short_mel)
    assert np.allclose(patches[0, 10:], 0.0)


# ── EffNet embeddings ─────────────────────────────────────────

@pytest.fixture(scope="module")
def effnet_loaded():
    from backend.audio_features import load_effnet, is_effnet_ready
    load_effnet()
    return is_effnet_ready()


def test_effnet_embedding(effnet_loaded):
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import generate_effnet_embedding

    emb = generate_effnet_embedding(TEST_TRACK)
    print(f"EffNet embedding: shape={emb.shape}, dtype={emb.dtype}")
    print(f"  min={emb.min():.4f}  max={emb.max():.4f}  mean={emb.mean():.4f}")

    assert emb is not None, "generate_effnet_embedding returned None"
    assert emb.ndim == 1, f"Expected 1-D vector, got {emb.ndim}-D"
    assert emb.shape[0] > 0
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all(), "Embedding contains NaN/Inf"
    assert np.linalg.norm(emb) > 0, "Embedding is all zeros"


def test_effnet_embedding_harmonic(effnet_loaded):
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import generate_effnet_embedding

    emb = generate_effnet_embedding(TEST_TRACK_HARMONIC)
    print(f"EffNet (harmonic): shape={emb.shape}")

    assert emb is not None
    assert emb.ndim == 1
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()


def test_effnet_deterministic(effnet_loaded):
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import generate_effnet_embedding

    emb1 = generate_effnet_embedding(TEST_TRACK)
    emb2 = generate_effnet_embedding(TEST_TRACK)

    assert emb1 is not None and emb2 is not None
    np.testing.assert_array_equal(emb1, emb2)


# ── Batched EffNet inference ──────────────────────────────────

def test_effnet_batch(effnet_loaded):
    """Batch inference should return an embedding for each valid path."""
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import generate_effnet_embeddings_batch

    paths = [TEST_TRACK, TEST_TRACK_HARMONIC]
    results = generate_effnet_embeddings_batch(paths)

    assert isinstance(results, dict)
    assert set(results.keys()) == set(paths)

    for p in paths:
        emb = results[p]
        assert emb is not None, f"Batch returned None for {p}"
        assert emb.ndim == 1
        assert emb.shape[0] > 0
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()
        assert np.linalg.norm(emb) > 0


def test_effnet_batch_matches_single(effnet_loaded):
    """Batch inference must produce identical embeddings to single-track calls."""
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import (
        generate_effnet_embedding,
        generate_effnet_embeddings_batch,
    )

    paths = [TEST_TRACK, TEST_TRACK_HARMONIC]
    single = {p: generate_effnet_embedding(p) for p in paths}
    batched = generate_effnet_embeddings_batch(paths)

    for p in paths:
        assert single[p] is not None and batched[p] is not None
        np.testing.assert_allclose(
            batched[p], single[p], rtol=1e-5, atol=1e-6,
            err_msg=f"Batch vs single mismatch for {os.path.basename(p)}",
        )


def test_effnet_batch_empty():
    """Empty path list should return empty dict without errors."""
    from backend.audio_features import generate_effnet_embeddings_batch

    assert generate_effnet_embeddings_batch([]) == {}


def test_effnet_batch_invalid_path(effnet_loaded):
    """Invalid paths should map to None, valid paths should still succeed."""
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.audio_features import generate_effnet_embeddings_batch

    bogus = "/nonexistent/file.mp3"
    results = generate_effnet_embeddings_batch([TEST_TRACK, bogus])

    assert results[TEST_TRACK] is not None
    assert results[TEST_TRACK].ndim == 1
    assert results[bogus] is None


# ── Key detection ─────────────────────────────────────────────

def test_key_detection_harmonic():
    """Detect key on the harmonic track (ID3 says Am)."""
    import librosa
    from backend.audio_features import detect_key

    audio, sr = librosa.load(TEST_TRACK_HARMONIC, sr=44100, mono=True)
    key, scale = detect_key(audio, sr)

    print(f"Detected key: {key} {scale}")
    assert key in ("C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B")
    assert scale in ("major", "minor")


def test_key_detection_returns_valid():
    """Key detection on the rhythmic track should still return a valid result."""
    import librosa
    from backend.audio_features import detect_key

    audio, sr = librosa.load(TEST_TRACK, sr=44100, mono=True)
    key, scale = detect_key(audio, sr)

    print(f"Detected key: {key} {scale}")
    assert key in ("C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B")
    assert scale in ("major", "minor")


# ── Circle-of-fifths encoding ────────────────────────────────

def test_circle_of_fifths_mapping():
    from backend.audio_features import _KEY_TO_FIFTHS

    all_keys = {"C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"}
    mapped_keys = set(_KEY_TO_FIFTHS.keys())
    assert all_keys.issubset(mapped_keys), f"Missing keys: {all_keys - mapped_keys}"

    positions = set(_KEY_TO_FIFTHS.values())
    assert positions == set(range(12)), f"Expected positions 0-11, got {positions}"


def test_circle_of_fifths_unit_circle():
    """Verify that the cos/sin encoding lies on the unit circle."""
    from backend.audio_features import _KEY_TO_FIFTHS

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
    from backend.audio_features import _KEY_TO_FIFTHS

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
    from backend.audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK)
    print(f"Features: shape={feat.shape}, dtype={feat.dtype}, values={feat}")

    assert feat is not None
    assert feat.shape == (6,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()


def test_audio_features_shape_harmonic():
    from backend.audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK_HARMONIC)
    print(f"Features (harmonic): shape={feat.shape}, values={feat}")

    assert feat is not None
    assert feat.shape == (6,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()


def test_audio_features_ranges():
    from backend.audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK)
    tempo_norm, key_cos, key_sin, mode, energy_norm, danceability = feat

    assert 0.0 <= tempo_norm <= 1.0, f"tempo_norm={tempo_norm}"
    assert -1.0 <= key_cos <= 1.0, f"key_cos={key_cos}"
    assert -1.0 <= key_sin <= 1.0, f"key_sin={key_sin}"
    assert mode in (0.0, 1.0), f"mode={mode}"
    assert 0.0 <= energy_norm <= 1.0, f"energy_norm={energy_norm}"
    assert 0.0 <= danceability <= 1.0, f"danceability={danceability}"


def test_audio_features_ranges_harmonic():
    from backend.audio_features import extract_audio_features

    feat = extract_audio_features(TEST_TRACK_HARMONIC)
    tempo_norm, key_cos, key_sin, mode, energy_norm, danceability = feat

    assert 0.0 <= tempo_norm <= 1.0
    assert -1.0 <= key_cos <= 1.0
    assert -1.0 <= key_sin <= 1.0
    assert mode in (0.0, 1.0)
    assert 0.0 <= energy_norm <= 1.0
    assert 0.0 <= danceability <= 1.0


def test_audio_features_deterministic():
    from backend.audio_features import extract_audio_features

    f1 = extract_audio_features(TEST_TRACK)
    f2 = extract_audio_features(TEST_TRACK)

    assert f1 is not None and f2 is not None
    np.testing.assert_array_equal(f1, f2)


# ── Standalone runner ─────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_mel_preprocessing,
        test_mel_patching,
        test_mel_patching_short_audio,
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
    ]

    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
        print("  OK")

    print("\n--- EffNet tests (require model download) ---")
    from backend.audio_features import load_effnet, is_effnet_ready
    load_effnet()
    if is_effnet_ready():
        test_effnet_embedding(True)
        print("  OK")
        test_effnet_embedding_harmonic(True)
        print("  OK")
        test_effnet_deterministic(True)
        print("  OK")
        test_effnet_batch(True)
        print("  OK")
        test_effnet_batch_matches_single(True)
        print("  OK")
        test_effnet_batch_empty()
        print("  OK")
        test_effnet_batch_invalid_path(True)
        print("  OK")
    else:
        print("  SKIPPED (onnxruntime not available)")

    print("\nAll tests passed.")
