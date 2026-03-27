"""Classical audio descriptors via librosa (6-D: tempo, key circle, mode, energy, dance)."""

from __future__ import annotations

import logging
import math

import librosa
import numpy as np

from .audio_decode import (
    _mono_centered_window_or_full,
    _mono_full_short_metadata_decode,
    _probe_long_track_decode,
)
from backend.decode_stderr import librosa_get_duration

log = logging.getLogger(__name__)

FEATURES_VERSION = 3
AUDIO_FEATURE_DIM = 6

_KEY_TO_FIFTHS: dict[str, int] = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
    "F#": 6, "Gb": 6,
    "C#": 7, "Db": 7,
    "G#": 8, "Ab": 8,
    "D#": 9, "Eb": 9,
    "A#": 10, "Bb": 10,
    "F": 11,
}

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

_BPM_LO = 60.0
_BPM_HI = 200.0
_FEATURE_SR = 22050
_FEATURE_EXCERPT_SECONDS = 20.0
_FEATURE_HOP = 1024


def _load_feature_audio_excerpt(
    path: str,
) -> tuple[np.ndarray | None, int | None, str | None]:
    try:
        meta_dur = float(librosa_get_duration(path=path))
        if not math.isfinite(meta_dur) or meta_dur <= 0:
            return None, None, None

        done, audio, sr, warn = _mono_full_short_metadata_decode(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            meta_short_max=_FEATURE_EXCERPT_SECONDS * 1.25,
        )
        if done:
            if audio is None:
                return None, None, None
            return audio, sr, warn

        kind, audio, sr, warn = _probe_long_track_decode(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            segment_seconds=_FEATURE_EXCERPT_SECONDS,
        )
        if kind == "empty":
            return None, None, None
        if kind == "full":
            if audio is None:
                return None, None, None
            return audio, sr, warn

        return _mono_centered_window_or_full(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            window_seconds=_FEATURE_EXCERPT_SECONDS,
        )
    except Exception as e:
        log.warning("Audio excerpt load failed for %s: %s", path, e)
        return None, None, None


def detect_key(audio: np.ndarray, sr: int = 44100) -> tuple[str, str]:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=_FEATURE_HOP)
    mean_chroma = chroma.mean(axis=1)

    best_corr = -np.inf
    best_key = "C"
    best_scale = "major"

    for shift in range(12):
        rotated = np.roll(mean_chroma, -shift)
        corr_maj = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
        corr_min = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = _PITCH_CLASSES[shift]
            best_scale = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = _PITCH_CLASSES[shift]
            best_scale = "minor"

    return best_key, best_scale


def _compute_danceability(
    audio: np.ndarray,
    sr: int = 44100,
    onset_env: np.ndarray | None = None,
    tempo_bpm: float | None = None,
    hop_length: int = _FEATURE_HOP,
) -> float:
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    if len(onset_env) < 4:
        return 0.0

    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    if len(ac) < 2:
        return 0.0

    ac = ac / (ac[0] + 1e-9)

    if tempo_bpm is None:
        tempo_bpm = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if hasattr(tempo_bpm, "__len__"):
            tempo_bpm = float(tempo_bpm[0])

    frames_per_sec = sr / hop_length
    if tempo_bpm > 0:
        lag = int(round(60.0 * frames_per_sec / tempo_bpm))
    else:
        lag = 0

    if 0 < lag < len(ac):
        peak = ac[lag]
    else:
        peak = float(np.max(ac[1:])) if len(ac) > 1 else 0.0

    return float(np.clip(peak, 0.0, 1.0))


def extract_audio_features_and_warning(
    path: str,
) -> tuple[np.ndarray | None, str | None]:
    try:
        audio, sr, warn = _load_feature_audio_excerpt(path)
        if audio is None or sr is None or len(audio) == 0:
            return None, warn

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=_FEATURE_HOP)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0])
        tempo_norm = float(np.clip((tempo - _BPM_LO) / (_BPM_HI - _BPM_LO), 0.0, 1.0))

        key_str, scale_str = detect_key(audio, sr)
        fifths_pos = _KEY_TO_FIFTHS.get(key_str, 0)
        angle = 2.0 * math.pi * fifths_pos / 12.0
        key_cos = math.cos(angle)
        key_sin = math.sin(angle)
        mode = 1.0 if scale_str == "major" else 0.0

        rms = librosa.feature.rms(y=audio)
        mean_rms = float(rms.mean())
        energy_norm = float(np.clip(np.log1p(mean_rms * 100) / 5.0, 0.0, 1.0))

        danceability = _compute_danceability(
            audio, sr, onset_env=onset_env, tempo_bpm=tempo, hop_length=_FEATURE_HOP,
        )

        return (
            np.array(
                [tempo_norm, key_cos, key_sin, mode, energy_norm, danceability],
                dtype=np.float32,
            ),
            warn,
        )
    except Exception as e:
        log.warning("Audio feature extraction failed for %s: %s", path, e)
        return None, None


def extract_audio_features(path: str) -> np.ndarray | None:
    vec, _warn = extract_audio_features_and_warning(path)
    return vec


_FIFTHS_POS_NAMES = (
    "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F",
)


def audio_features_display_bpm_key(vec: np.ndarray | None) -> tuple[int | None, str | None]:
    if vec is None or int(vec.size) < AUDIO_FEATURE_DIM:
        return None, None
    tnorm = float(np.clip(float(vec[0]), 0.0, 1.0))
    bpm_f = tnorm * (_BPM_HI - _BPM_LO) + _BPM_LO
    bpm = int(round(bpm_f))
    key_cos = float(vec[1])
    key_sin = float(vec[2])
    mode_major = float(vec[3]) >= 0.5
    angle = math.atan2(key_sin, key_cos)
    fifths = int(round(12.0 * angle / (2.0 * math.pi))) % 12
    name = _FIFTHS_POS_NAMES[fifths]
    key_label = name if mode_major else f"{name}m"
    return bpm, key_label


def generate_audio_features_batch(
    paths: list[str],
    decode_warnings: dict[str, str] | None = None,
) -> dict[str, np.ndarray | None]:
    if not paths:
        return {}

    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    for p in paths:
        vec, w = extract_audio_features_and_warning(p)
        results[p] = vec
        if w and decode_warnings is not None:
            decode_warnings[p] = w

    return results


def warmup_audio_features() -> None:
    try:
        seconds = 4
        n = int(_FEATURE_SR * seconds)
        t = np.linspace(0.0, seconds, num=n, endpoint=False, dtype=np.float32)
        audio = 0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
        _ = librosa.onset.onset_strength(y=audio, sr=_FEATURE_SR, hop_length=_FEATURE_HOP)
        _ = librosa.feature.tempo(y=audio, sr=_FEATURE_SR, hop_length=_FEATURE_HOP)
        _ = librosa.feature.chroma_stft(y=audio, sr=_FEATURE_SR, n_fft=2048, hop_length=_FEATURE_HOP)
        _ = librosa.feature.rms(y=audio)
        _ = detect_key(audio, _FEATURE_SR)
        _ = _compute_danceability(audio, _FEATURE_SR, hop_length=_FEATURE_HOP)
    except Exception as e:
        log.debug("Audio feature warmup skipped: %s", e)
