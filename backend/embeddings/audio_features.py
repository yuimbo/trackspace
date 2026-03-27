"""Compatibility barrel: resilient decode, EffNet ONNX, and librosa 6-D features.

Prefer importing from this package's `audio_decode`, `effnet`, or `librosa_audio_features`.
Import via ``from backend import audio_features`` (shim) or ``backend.embeddings.audio_features``.
"""

from __future__ import annotations

# Re-export decode helpers (CLAP / cross-feature)
from .audio_decode import (
    decode_reliability_user_message,
    load_resilient_audio_segments,
)
from backend.decode_stderr import librosa_get_duration

# EffNet
from .effnet import (
    EFFNET_VERSION,
    compute_mel_spectrogram,
    generate_effnet_embedding,
    generate_effnet_embeddings_batch,
    is_effnet_ready,
    load_effnet,
    patch_mel_spectrogram,
)

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

from .effnet import (  # noqa: F401
    _EFFNET_BATCH_CAP,
    _EFFNET_HOP,
    _EFFNET_N_FFT,
    _EFFNET_N_MELS,
    _EFFNET_PATCH_FRAMES,
    _EFFNET_PATCH_HOP,
    _EFFNET_SEGMENT_SECONDS,
    _EFFNET_SR,
)
from .librosa_audio_features import (  # noqa: F401
    _FEATURE_EXCERPT_SECONDS,
    _FEATURE_HOP,
    _FEATURE_SR,
    _KEY_TO_FIFTHS,
    _load_feature_audio_excerpt,
)
