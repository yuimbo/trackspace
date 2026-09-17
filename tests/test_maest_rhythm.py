"""MAEST analysis and rhythm descriptors against the real fixture track.

MAEST tests are skipped when the checkpoint is unavailable (offline CI) rather
than failing, but when the model *is* present the assertions are real: shapes,
finiteness, determinism, and that decode failures degrade to ``None``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from backend.embeddings import maest as maest_mod
from backend.embeddings import rhythm_features as rf

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_track.mp3")

def _maest_ready() -> bool:
    """Load once at collection time; skip the suite if the checkpoint is absent."""
    if not maest_mod.is_model_ready():
        maest_mod.load_model()
    return maest_mod.is_model_ready()


maest_available = pytest.mark.skipif(
    not _maest_ready(),
    reason=f"MAEST checkpoint unavailable ({maest_mod.load_error()})",
)


# ── rhythm features ───────────────────────────────────────────


def test_rhythm_vector_shape_and_range() -> None:
    vec = rf.extract_rhythm_features(FIXTURE)
    assert vec is not None
    assert vec.shape == (rf.RHYTHM_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    # Every dimension is normalised so no single feature can dominate the block.
    assert (vec >= 0.0).all() and (vec <= 1.0).all()


def test_rhythm_names_match_dimension() -> None:
    assert len(rf.RHYTHM_FEATURE_NAMES) == rf.RHYTHM_FEATURE_DIM


def test_rhythm_is_deterministic() -> None:
    a = rf.extract_rhythm_features(FIXTURE)
    b = rf.extract_rhythm_features(FIXTURE)
    assert a is not None and b is not None
    assert np.allclose(a, b)


def test_rhythm_missing_file_returns_none() -> None:
    assert rf.extract_rhythm_features("/definitely/not/here.mp3") is None


def test_rhythm_batch_reports_per_path() -> None:
    out = rf.generate_rhythm_features_batch([FIXTURE, "/nope/x.mp3"])
    assert out[FIXTURE] is not None
    assert out["/nope/x.mp3"] is None


def test_rhythm_summary_is_labelled() -> None:
    vec = rf.extract_rhythm_features(FIXTURE)
    summary = rf.rhythm_feature_summary(vec)
    assert set(summary) == set(rf.RHYTHM_FEATURE_NAMES)
    assert rf.rhythm_feature_summary(None) == {}


# ── MAEST ─────────────────────────────────────────────────────


@maest_available
def test_maest_analysis_shapes() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    assert result.embedding.shape == (maest_mod.MAEST_EMBED_DIM,)
    assert result.logits.shape == (maest_mod.MAEST_LOGIT_DIM,)
    assert result.num_excerpts >= 1
    assert result.excerpt_embeddings.shape == (
        result.num_excerpts,
        maest_mod.MAEST_EMBED_DIM,
    )
    assert np.isfinite(result.embedding).all()
    assert np.isfinite(result.logits).all()


@maest_available
def test_maest_pooled_embedding_matches_excerpt_mean() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    expected = result.excerpt_embeddings.mean(axis=0)
    assert np.allclose(result.embedding, expected, atol=1e-5)


@maest_available
def test_maest_is_deterministic() -> None:
    a = maest_mod.analyze(FIXTURE)
    b = maest_mod.analyze(FIXTURE)
    assert a is not None and b is not None
    assert np.allclose(a.embedding, b.embedding, atol=1e-4)


@maest_available
def test_maest_missing_file_returns_none() -> None:
    assert maest_mod.analyze("/definitely/not/here.mp3") is None


@maest_available
def test_maest_batch_keys_every_requested_path() -> None:
    out = maest_mod.analyze_batch([FIXTURE, "/nope/x.mp3"])
    assert set(out) == {FIXTURE, "/nope/x.mp3"}
    assert out[FIXTURE] is not None
    assert out["/nope/x.mp3"] is None


@maest_available
def test_top_styles_are_ranked_discogs_labels() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    styles = maest_mod.top_styles(result.logits, k=5)
    assert len(styles) == 5
    scores = [s for _, s in styles]
    assert scores == sorted(scores, reverse=True)
    # Labels come from the Discogs taxonomy, formatted "Genre---Style".
    assert all("---" in name for name, _ in styles)


def test_discogs_label_table_size() -> None:
    labels = maest_mod.discogs_style_labels()
    if labels:
        assert len(labels) == maest_mod.MAEST_LOGIT_DIM


def test_empty_logits_yield_no_styles() -> None:
    assert maest_mod.top_styles(np.array([], dtype=np.float32)) == []
