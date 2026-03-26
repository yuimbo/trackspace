"""Smoke tests for the audio fingerprint → CLAP embedding pipeline.

Run:  .venv/bin/python -m pytest tests/ -v
  or: .venv/bin/python -m tests.test_embeddings       (standalone)
"""

import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_TRACK = os.path.join(FIXTURE_DIR, "test_track.mp3")


def test_fingerprint():
    from backend.fingerprint import compute_fingerprint, _FPCALC

    print(f"fpcalc binary: {_FPCALC}")
    assert _FPCALC, "fpcalc not found on PATH"

    fp = compute_fingerprint(TEST_TRACK)
    print(f"Fingerprint: {fp}")
    assert fp is not None, "compute_fingerprint returned None"
    assert len(fp) == 64, f"Expected 64-char hex digest, got {len(fp)}"


def test_audio_segment_loading():
    from backend.embeddings import _load_audio_segments, CLAP_SR

    segments = _load_audio_segments(TEST_TRACK)
    print(f"Segments: count={len(segments)}")
    assert len(segments) >= 1, "_load_audio_segments returned empty"
    for i, seg in enumerate(segments):
        print(f"  segment {i}: shape={seg.shape}, dtype={seg.dtype}, sr={CLAP_SR}")
        assert seg.ndim == 1
        assert len(seg) > 0
        assert seg.dtype == np.float32


def test_clap_embedding():
    from backend.embeddings import load_model, generate_embedding

    load_model()

    emb = generate_embedding(TEST_TRACK)
    print(f"Embedding: shape={emb.shape}, dtype={emb.dtype}")
    print(f"  min={emb.min():.4f}  max={emb.max():.4f}  mean={emb.mean():.4f}")
    assert emb is not None, "generate_embedding returned None"
    assert emb.ndim == 1, f"Expected 1-D vector, got {emb.ndim}-D"
    assert emb.shape[0] > 0
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all(), "Embedding contains NaN/Inf"


if __name__ == "__main__":
    test_fingerprint()
    print("  OK\n")
    test_audio_segment_loading()
    print("  OK\n")
    test_clap_embedding()
    print("  OK\n")
    print("All tests passed.")
