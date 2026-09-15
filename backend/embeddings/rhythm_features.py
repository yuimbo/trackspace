"""Rhythm / style descriptors for electronic-microgenre discrimination (14-D).

The classical 6-D vector (``librosa_audio_features``) captures tempo, key, mode and
gross energy — enough to separate broad genres, but it spends most of its budget on
tonal harmony and collapses the micro-rhythmic detail that distinguishes electronic
sub-genres (techno vs. trance vs. house vs. breakbeat, half-time vs. double-time,
four-on-the-floor vs. syncopated grooves).

These dimensions are therefore kept as a **separate block** rather than appended to the
6-D vector: they measure *how the rhythm is articulated* — onset density and strength,
beat regularity, pulse clarity, syncopation, low-end (kick) periodicity, spectral shape,
dynamics and rhythmic entropy.  Living on a different scale from the tonal descriptors,
they are weighted as their own block in the distance function so neither family drowns
out the other.  Every value is finite and normalised to ``[0, 1]``.

Decoding mirrors ``librosa_audio_features._load_feature_audio_excerpt`` (same resilient
probe / centered-window strategy) but uses a ~30 s excerpt so rhythm is measured over a
longer window than the 20 s classical one.  librosa only — no Essentia.
"""

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

RHYTHM_VERSION = 1

RHYTHM_FEATURE_NAMES: tuple[str, ...] = (
    "onset_density",
    "onset_strength_mean",
    "onset_strength_std",
    "beat_confidence",
    "pulse_clarity",
    "syncopation",
    "kick_periodicity",
    "bass_ratio",
    "spectral_centroid_norm",
    "spectral_rolloff_norm",
    "spectral_flux",
    "dynamic_range",
    "loudness",
    "rhythmic_complexity",
)

RHYTHM_FEATURE_DIM = len(RHYTHM_FEATURE_NAMES)
assert RHYTHM_FEATURE_DIM == 14, "RHYTHM_FEATURE_DIM must match RHYTHM_FEATURE_NAMES"

# 22.05 kHz keeps the STFT / onset / beat stack cheap; rhythm only needs a
# longer window than the 20 s classical excerpt, not a higher sample rate.
_RHYTHM_SR = 22050
_RHYTHM_EXCERPT_SECONDS = 30.0
_RHYTHM_HOP = 512
_RHYTHM_N_FFT = 2048

# Low band (kick) upper edge and the bass-energy edge.
_KICK_HZ = 150.0
_BASS_HZ = 150.0

# Robust scaling constants: x / (x + k) maps [0, inf) -> [0, 1) monotonically.
_ONSET_MEAN_K = 5.0
_ONSET_STD_K = 5.0
_FLUX_K = 1.0
_ONSET_DENSITY_PER_SEC = 12.0
_DYNAMIC_RANGE_DB = 40.0
_LOUDNESS_FLOOR_DB = -60.0


def _load_rhythm_audio_excerpt(
    path: str,
) -> tuple[np.ndarray | None, int | None, str | None]:
    try:
        meta_dur = float(librosa_get_duration(path=path))
        if not math.isfinite(meta_dur) or meta_dur <= 0:
            return None, None, None

        done, audio, sr, warn = _mono_full_short_metadata_decode(
            path,
            sr=_RHYTHM_SR,
            meta_dur=meta_dur,
            meta_short_max=_RHYTHM_EXCERPT_SECONDS * 1.25,
        )
        if done:
            if audio is None:
                return None, None, None
            return audio, sr, warn

        kind, audio, sr, warn = _probe_long_track_decode(
            path,
            sr=_RHYTHM_SR,
            meta_dur=meta_dur,
            segment_seconds=_RHYTHM_EXCERPT_SECONDS,
        )
        if kind == "empty":
            return None, None, None
        if kind == "full":
            if audio is None:
                return None, None, None
            return audio, sr, warn

        return _mono_centered_window_or_full(
            path,
            sr=_RHYTHM_SR,
            meta_dur=meta_dur,
            window_seconds=_RHYTHM_EXCERPT_SECONDS,
        )
    except Exception as e:
        log.warning("Rhythm excerpt load failed for %s: %s", path, e)
        return None, None, None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _clip01(value: object, default: float = 0.0) -> float:
    return float(np.clip(_safe_float(value, default), 0.0, 1.0))


def _unit_scale(value: object, k: float) -> float:
    x = _safe_float(value, 0.0)
    if x <= 0.0 or k <= 0.0:
        return 0.0
    return _clip01(x / (x + k))


def _beat_lag_frames(tempo_bpm: float, sr: int, hop_length: int) -> int:
    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        return 0
    frames_per_sec = float(sr) / float(hop_length)
    return int(round(60.0 * frames_per_sec / tempo_bpm))


def _acf_peak_at_lag(ac: np.ndarray, lag: int) -> float:
    if ac.size < 2:
        return 0.0
    if 0 < lag < ac.size:
        peak = float(ac[lag])
    else:
        peak = float(np.max(ac[1:]))
    return _clip01(peak)


def _normalized_onset_autocorrelation(onset_env: np.ndarray) -> np.ndarray:
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    if ac.size < 2:
        return np.zeros(0, dtype=np.float64)
    return ac / (ac[0] + 1e-9)


def _compute_rhythm_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    hop = _RHYTHM_HOP
    n_fft = _RHYTHM_N_FFT

    # Single STFT reused for onset envelope, spectral shape and band energies.
    S = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop))
    if S.shape[1] < 2:
        return np.zeros(RHYTHM_FEATURE_DIM, dtype=np.float32)

    onset_env = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop)
    onset_env = np.asarray(onset_env, dtype=np.float64)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = freqs < _BASS_HZ
    if not np.any(low_mask):
        low_mask = freqs <= freqs[0]

    power = S.astype(np.float64) ** 2
    total_energy = float(power.sum()) + 1e-12

    duration = float(audio.shape[0]) / float(sr) if sr else 0.0
    n_frames = int(S.shape[1])

    # --- onset density -------------------------------------------------
    try:
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop, units="time",
        )
        n_onsets = int(np.size(onsets))
    except Exception:
        n_onsets = 0
    if duration > 0.0:
        onset_density = _clip01(
            (n_onsets / duration) / _ONSET_DENSITY_PER_SEC
        )
    else:
        onset_density = 0.0

    # --- onset strength -------------------------------------------------
    if onset_env.size:
        onset_strength_mean = _unit_scale(float(np.mean(onset_env)), _ONSET_MEAN_K)
        onset_strength_std = _unit_scale(float(np.std(onset_env)), _ONSET_STD_K)
    else:
        onset_strength_mean = 0.0
        onset_strength_std = 0.0

    # --- beat track (single call, reused everywhere) ---------------------
    try:
        tempo_arr, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames",
        )
        tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
        beat_frames = np.asarray(beats, dtype=int)
    except Exception:
        tempo_bpm = 0.0
        beat_frames = np.array([], dtype=int)

    if beat_frames.size >= 3:
        ibi = np.diff(beat_frames).astype(np.float64)
        mean_ibi = float(np.mean(ibi))
        if mean_ibi > 1e-6:
            cv = float(np.std(ibi)) / mean_ibi
            beat_confidence = _clip01(1.0 / (1.0 + cv))
        else:
            beat_confidence = 1.0
    else:
        beat_confidence = 0.0

    ac = _normalized_onset_autocorrelation(onset_env)
    lag = _beat_lag_frames(tempo_bpm, sr, hop)
    pulse_clarity = _acf_peak_at_lag(ac, lag)

    # --- syncopation: onset energy off the beat grid ---------------------
    total_onset = float(onset_env.sum()) if onset_env.size else 0.0
    if total_onset > 1e-9 and beat_frames.size:
        on_beat = np.zeros(n_frames, dtype=bool)
        for b in beat_frames:
            lo = max(0, int(b) - 1)
            hi = min(n_frames, int(b) + 2)
            on_beat[lo:hi] = True
        if on_beat.size == onset_env.size:
            off_energy = float(onset_env[~on_beat].sum())
        else:
            off_energy = total_onset
        syncopation = _clip01(off_energy / total_onset)
    else:
        syncopation = 0.0

    # --- kick periodicity: low-band onset autocorrelation ---------------
    low_env = power[low_mask].sum(axis=0)
    if low_env.size >= 2:
        low_onset = np.diff(low_env)
        low_onset = np.maximum(low_onset, 0.0)
        low_ac = librosa.autocorrelate(low_onset, max_size=low_onset.size)
        if low_ac.size >= 2 and low_ac[0] > 1e-12:
            low_ac = low_ac / (low_ac[0] + 1e-9)
            kick_periodicity = _acf_peak_at_lag(low_ac, lag)
        else:
            kick_periodicity = 0.0
    else:
        kick_periodicity = 0.0

    # --- bass ratio ----------------------------------------------------
    bass_ratio = _clip01(float(power[low_mask].sum()) / total_energy)

    # --- spectral shape ------------------------------------------------
    nyquist = float(sr) / 2.0
    try:
        centroid = librosa.feature.spectral_centroid(
            S=S, sr=sr, n_fft=n_fft, hop_length=hop,
        )
        spectral_centroid_norm = _clip01(
            float(np.mean(centroid)) / nyquist if nyquist > 0 else 0.0
        )
    except Exception:
        spectral_centroid_norm = 0.0

    try:
        rolloff = librosa.feature.spectral_rolloff(
            S=S, sr=sr, n_fft=n_fft, hop_length=hop,
        )
        spectral_rolloff_norm = _clip01(
            float(np.mean(rolloff)) / nyquist if nyquist > 0 else 0.0
        )
    except Exception:
        spectral_rolloff_norm = 0.0

    # --- spectral flux: mean frame-to-frame spectral difference ---------
    if S.shape[1] >= 2:
        diff = np.diff(S, axis=1)
        flux_raw = float(np.mean(np.abs(diff)))
        ref = float(np.mean(S)) + 1e-9
        spectral_flux = _unit_scale(flux_raw / ref, _FLUX_K)
    else:
        spectral_flux = 0.0

    # --- dynamics ------------------------------------------------------
    try:
        rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)
        rms = np.asarray(rms, dtype=np.float64).reshape(-1)
    except Exception:
        rms = np.array([], dtype=np.float64)

    if rms.size:
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        p5 = float(np.percentile(rms_db, 5.0))
        p95 = float(np.percentile(rms_db, 95.0))
        dynamic_range = _clip01((p95 - p5) / _DYNAMIC_RANGE_DB)
        loudness = _clip01(
            (float(np.mean(rms_db)) - _LOUDNESS_FLOOR_DB) / -_LOUDNESS_FLOOR_DB
        )
    else:
        dynamic_range = 0.0
        loudness = 0.0

    # --- rhythmic complexity: entropy of the onset ACF profile -----------
    if ac.size >= 2:
        pos = np.clip(ac, 0.0, None)
        s = float(pos.sum())
        if s > 1e-9:
            p = pos / s
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
            rhythmic_complexity = _clip01(entropy / math.log(ac.size))
        else:
            rhythmic_complexity = 0.0
    else:
        rhythmic_complexity = 0.0

    vec = np.array(
        [
            onset_density,
            onset_strength_mean,
            onset_strength_std,
            beat_confidence,
            pulse_clarity,
            syncopation,
            kick_periodicity,
            bass_ratio,
            spectral_centroid_norm,
            spectral_rolloff_norm,
            spectral_flux,
            dynamic_range,
            loudness,
            rhythmic_complexity,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def extract_rhythm_features_and_warning(
    path: str,
) -> tuple[np.ndarray | None, str | None]:
    try:
        audio, sr, warn = _load_rhythm_audio_excerpt(path)
        if audio is None or sr is None or len(audio) == 0:
            return None, warn
        vec = _compute_rhythm_vector(audio, int(sr))
        return vec, warn
    except Exception as e:
        log.warning("Rhythm feature extraction failed for %s: %s", path, e)
        return None, None


def extract_rhythm_features(path: str) -> np.ndarray | None:
    vec, _warn = extract_rhythm_features_and_warning(path)
    return vec


def generate_rhythm_features_batch(
    paths: list[str],
    decode_warnings: dict[str, str] | None = None,
) -> dict[str, np.ndarray | None]:
    if not paths:
        return {}

    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    for p in paths:
        vec, w = extract_rhythm_features_and_warning(p)
        results[p] = vec
        if w and decode_warnings is not None:
            decode_warnings[p] = w

    return results


def rhythm_feature_summary(vec: np.ndarray | None) -> dict[str, float]:
    if vec is None or int(np.size(vec)) != RHYTHM_FEATURE_DIM:
        return {}
    return {
        name: _clip01(vec[i])
        for i, name in enumerate(RHYTHM_FEATURE_NAMES)
    }


def warmup_rhythm_features() -> None:
    try:
        seconds = 4
        n = int(_RHYTHM_SR * seconds)
        t = np.linspace(0.0, seconds, num=n, endpoint=False, dtype=np.float32)
        audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        click = np.zeros_like(audio)
        click[:: int(_RHYTHM_SR * 0.5)] = 0.5
        audio = audio + click
        _ = _compute_rhythm_vector(audio, _RHYTHM_SR)
    except Exception as e:
        log.debug("Rhythm feature warmup skipped: %s", e)
