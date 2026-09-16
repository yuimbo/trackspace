"""Compatibility barrel: resilient decode and librosa 6-D features.

Prefer importing from this package's `audio_decode` or `librosa_audio_features`.
Import via ``from backend import audio_features`` (shim) or
``backend.embeddings.audio_features``.
"""

from __future__ import annotations

# Re-export decode helpers (CLAP / cross-feature)
from .audio_decode import (
    decode_reliability_user_message,
    load_resilient_audio_segments,
)
from backend.decode_stderr import librosa_get_duration

# Librosa classical features
from .librosa_audio_features import (
    AUDIO_FEATURE_DIM,
    FEATURES_VERSION,
    audio_features_display_bpm_key,
    detect_key,
    extract_audio_features,
    extract_audio_features_and_warning,
    generate_audio_features_batch,
    warmup_audio_features,
)

# --- Private symbols referenced by tests (stable paths via this module) ---

from .librosa_audio_features import (  # noqa: F401
    _FEATURE_EXCERPT_SECONDS,
    _FEATURE_HOP,
    _FEATURE_SR,
    _KEY_TO_FIFTHS,
    _load_feature_audio_excerpt,
)
