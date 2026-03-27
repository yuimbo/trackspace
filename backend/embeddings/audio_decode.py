"""Robust librosa-based decode: short/long track probes, seeks, and segment lists.

Used by CLAP, EffNet (via segment loader), and librosa classical features.  Keeps
MP3/metadata edge cases in one place."""
from __future__ import annotations

import math

import numpy as np

from backend.decode_stderr import librosa_get_duration, librosa_load


def decode_reliability_user_message(meta_seconds: float, decoded_seconds: float) -> str:
    """Human-readable note when container metadata and decoded audio length disagree."""
    if not math.isfinite(meta_seconds) or meta_seconds <= 0:
        return (
            "Audio could not be decoded to match the file metadata; "
            "the file may be damaged — try re-encoding or replacing it."
        )
    return (
        "Metadata reports ~"
        f"{meta_seconds:.0f}s"
        " of audio but only ~"
        f"{decoded_seconds:.1f}s"
        " decoded — file may be damaged or mis-tagged; re-encode for reliable analysis."
    )


def _three_segments_from_samples(
    y: np.ndarray,
    *,
    segment_seconds: float,
    sr: int,
) -> list[np.ndarray]:
    n = int(y.shape[0])
    if n <= 0:
        return []
    w = int(round(float(segment_seconds) * float(sr)))
    if w <= 0:
        return [y]
    total_dur = n / float(sr)
    if total_dur <= segment_seconds * 1.5:
        return [y]
    out: list[np.ndarray] = []
    for pos in (0.2, 0.5, 0.8):
        centre = int(round(pos * (n - 1)))
        start = max(0, centre - w // 2)
        end = min(n, start + w)
        start = max(0, end - w)
        chunk = y[start:end]
        if chunk.size > 0:
            out.append(chunk)
    return out


def _mono_full_short_metadata_decode(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    meta_short_max: float,
) -> tuple[bool, np.ndarray | None, int | None, str | None]:
    if meta_dur > meta_short_max:
        return False, None, None, None
    audio, sr_a = librosa_load(path, sr=sr, mono=True)
    if audio.size == 0:
        return True, None, None, None
    dec = float(audio.shape[0]) / float(sr_a)
    warn = None
    if meta_dur >= 2.0 and dec < meta_dur * 0.25:
        warn = decode_reliability_user_message(meta_dur, dec)
    return True, audio, sr_a, warn


def _probe_long_track_decode(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    segment_seconds: float,
) -> tuple[str, np.ndarray | None, int | None, str | None]:
    probe_len = min(segment_seconds, meta_dur)
    probe, sr_a = librosa_load(
        path, sr=sr, mono=True, offset=0.0, duration=probe_len,
    )
    if probe.size == 0:
        return "empty", None, None, None
    probe_dur = float(probe.shape[0]) / float(sr_a)
    if (
        probe_len >= 1.0
        and probe_dur < max(float(segment_seconds), probe_len) * 0.2
    ):
        audio, sr_a = librosa_load(path, sr=sr, mono=True)
        if audio.size == 0:
            return "empty", None, None, None
        tot_dur = float(audio.shape[0]) / float(sr_a)
        warn = decode_reliability_user_message(meta_dur, tot_dur)
        return "full", audio, sr_a, warn
    return "seek", None, None, None


def _try_librosa_seek_mono(
    path: str,
    *,
    sr: int,
    offset: float,
    read_dur: float,
) -> np.ndarray:
    try:
        audio, _ = librosa_load(
            path, sr=sr, mono=True, offset=offset, duration=read_dur,
        )
        return audio
    except Exception:
        return np.array([], dtype=np.float32)


def _mono_centered_window_or_full(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    window_seconds: float,
) -> tuple[np.ndarray | None, int | None, str | None]:
    offset = max(0.0, (meta_dur - window_seconds) * 0.5)
    read_dur = min(window_seconds, max(0.0, meta_dur - offset))
    if read_dur < 0.05:
        return None, None, None
    try:
        audio, sr_o = librosa_load(
            path, sr=sr, mono=True, offset=offset, duration=read_dur,
        )
        partial_ok = True
    except Exception:
        audio, sr_o = librosa_load(path, sr=sr, mono=True)
        partial_ok = False
    if audio.size == 0:
        audio, sr_o = librosa_load(path, sr=sr, mono=True)
        if audio.size == 0:
            return None, None, None
        dec = float(audio.shape[0]) / float(sr_o)
        warn = (
            decode_reliability_user_message(meta_dur, dec)
            if meta_dur > dec + 1.0
            else None
        )
        return audio, sr_o, warn
    dec = float(audio.shape[0]) / float(sr_o)
    warn = None
    if partial_ok:
        expected = min(float(read_dur), meta_dur)
        if expected >= 2.0 and dec < expected * 0.25:
            warn = decode_reliability_user_message(meta_dur, dec)
    else:
        if meta_dur > window_seconds * 1.25 and dec < meta_dur * 0.35:
            warn = decode_reliability_user_message(meta_dur, dec)
    return audio, sr_o, warn


def load_resilient_audio_segments(
    path: str,
    *,
    sr: int,
    segment_seconds: float,
) -> tuple[list[np.ndarray], str | None]:
    """Decode one or three segments; tolerate bad MP3 metadata and broken seeks."""
    meta_dur = float(librosa_get_duration(path=path))
    if not math.isfinite(meta_dur) or meta_dur <= 0:
        return [], None

    done, audio, sr_a, warn = _mono_full_short_metadata_decode(
        path,
        sr=sr,
        meta_dur=meta_dur,
        meta_short_max=segment_seconds * 1.5,
    )
    if done:
        if audio is None:
            return [], None
        return [audio], warn

    kind, audio, sr_a, warn = _probe_long_track_decode(
        path, sr=sr, meta_dur=meta_dur, segment_seconds=segment_seconds,
    )
    if kind == "empty":
        return [], None
    if kind == "full":
        assert audio is not None and sr_a is not None
        tot_dur = float(audio.shape[0]) / float(sr_a)
        if tot_dur <= segment_seconds * 1.5:
            return [audio], warn
        return (
            _three_segments_from_samples(
                audio, segment_seconds=segment_seconds, sr=int(sr_a),
            ),
            warn,
        )

    segments: list[np.ndarray] = []
    for pos in (0.2, 0.5, 0.8):
        centre = meta_dur * pos
        offset = max(0.0, centre - segment_seconds / 2)
        offset = min(offset, max(0.0, meta_dur - segment_seconds))
        read_dur = min(segment_seconds, max(0.0, meta_dur - offset))
        if read_dur < 0.05:
            continue
        audio = _try_librosa_seek_mono(
            path, sr=sr, offset=offset, read_dur=read_dur,
        )
        if audio.size > 0:
            segments.append(audio)

    if len(segments) >= 2:
        return segments, None

    audio, sr_a = librosa_load(path, sr=sr, mono=True)
    if audio.size == 0:
        return [], None
    tot_dur = float(audio.shape[0]) / float(sr_a)
    warn = None
    if meta_dur > segment_seconds * 1.5 and tot_dur < meta_dur * 0.35:
        warn = decode_reliability_user_message(meta_dur, tot_dur)
    if tot_dur <= segment_seconds * 1.5:
        return [audio], warn
    return (
        _three_segments_from_samples(
            audio, segment_seconds=segment_seconds, sr=int(sr_a),
        ),
        warn,
    )
