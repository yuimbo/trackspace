"""Track structure: intro / drop / break timestamps from a loudness envelope.

A whole-track mono decode at a low sample rate yields an RMS envelope; smoothing it
and thresholding against its own 95th-percentile level gives three coarse section
markers:

* **intro start** — first sustained crossing of ``INTRO_START_LEVEL_FRACTION`` of the
  track's peak level, i.e. where the audio actually begins after any lead-in silence;
  ``0`` when the track starts immediately.
* **intro end** — first sustained crossing of ``INTRO_LEVEL_FRACTION`` of the track's
  peak level (where the track "gets going"); ``0`` when it starts loud.
* **drop (peak)** — the start of the highest-energy ``PEAK_WINDOW_SECONDS`` window,
  the section a hover preview should audition.
* **break** — the longest sustained dip below ``BREAK_LEVEL_FRACTION`` of the peak
  level that returns to full energy afterwards; absent on tracks with no breakdown.

This is deliberately cheap DSP, not a model: it decodes the file once (the cost is the
decode, not the arithmetic) and caches six timestamps.  It is **not** an embedding
source — the result never feeds projection or clustering — so it is kept out of
``FROZEN_SOURCES`` and only appears in the feature cache as its own versioned blob.
Absent sections are stored as ``NaN`` and surface as ``null`` over the API.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np

from backend.decode_stderr import librosa_load

log = logging.getLogger(__name__)

#: Bump when the envelope, thresholds, or windowing change.
#: 2 — added ``intro_start`` (lead-in-silence skip); old blobs are purged and refilled.
STRUCTURE_VERSION = 2

#: Low sample rate: RMS needs no bandwidth, and a low rate keeps the resample cheap.
STRUCTURE_SR = 2205

#: Envelope frame rate (frames/second) before smoothing.
FRAME_RATE = 4.0

#: Moving-average window for the envelope, in seconds.
SMOOTH_SECONDS = 3.0

#: Width of the highest-energy window reported as the drop, in seconds.
PEAK_WINDOW_SECONDS = 30.0

#: Fraction of the peak level the intro must reach to be considered "started".
INTRO_LEVEL_FRACTION = 0.8

#: Fraction of the peak level above which the audio counts as "not lead-in silence".
INTRO_START_LEVEL_FRACTION = 0.05

#: How long the level must stay above the silence bar to mark the intro start, in seconds.
INTRO_START_HOLD_SECONDS = 1.0

#: Fraction of the peak level below which a sustained stretch counts as a break.
BREAK_LEVEL_FRACTION = 0.6

#: Minimum length of a sustained dip for it to count as a break, in seconds.
BREAK_MIN_SECONDS = 8.0

#: Sentinel for "no such section" inside the cached float32 blob.
_ABSENT = float("nan")

#: Blob slot order; the reader owns the layout (the blob carries no header).
STRUCTURE_FIELDS: tuple[str, ...] = (
    "intro_start",
    "intro_end",
    "peak_start",
    "peak_end",
    "break_start",
    "break_end",
)


@dataclass(frozen=True)
class TrackStructure:
    """Coarse section timestamps in seconds; ``None`` when a section is absent."""

    intro_start: float | None = None
    intro_end: float | None = None
    peak_start: float | None = None
    peak_end: float | None = None
    break_start: float | None = None
    break_end: float | None = None


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Edge-normalised moving average (window clamped to the signal length)."""
    n = int(x.size)
    if n == 0:
        return x.astype(np.float64)
    window = max(1, min(int(window), n))
    if window == 1:
        return x.astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(n)
    half = window // 2
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """RMS envelope of *y* at :data:`FRAME_RATE` frames per second."""
    hop = max(1, int(round(sr / FRAME_RATE)))
    frame_length = max(hop * 2, int(round(sr * 0.4)))
    audio = np.asarray(y, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float64)
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop
    )[0]
    return np.asarray(rms, dtype=np.float64).reshape(-1)


def _first_sustained(
    above: np.ndarray, hold: int
) -> int | None:
    """First frame index whose next *hold* frames are all ``True``."""
    if above.size == 0:
        return None
    if hold <= 1:
        idx = np.nonzero(above)[0]
        return int(idx[0]) if idx.size else None
    sustained = _moving_average(above.astype(np.float64), hold) >= 0.999
    idx = np.nonzero(sustained)[0]
    return int(idx[0]) if idx.size else None


def _longest_dip(
    below: np.ndarray, level: np.ndarray, threshold: float, min_frames: int
) -> tuple[int, int] | None:
    """Longest interior run of *below*, tie-broken by mean depth.

    Runs that touch either end are excluded: a quiet opening or a trailing fade is
    not a break — a break has energy on both sides.
    """
    n = int(below.size)
    if n == 0 or not bool(below.any()):
        return None
    delta = np.diff(below.astype(np.int8))
    starts = [int(i) + 1 for i in np.nonzero(delta == 1)[0]]
    ends = [int(i) + 1 for i in np.nonzero(delta == -1)[0]]
    if bool(below[0]):
        starts = [0, *starts]
    if bool(below[-1]):
        ends = [*ends, n]

    best: tuple[tuple[int, float], int, int] | None = None
    for s, e in zip(starts, ends):
        if e - s < min_frames:
            continue
        if s <= 0 or e >= n:
            continue
        depth = float(np.mean(threshold - level[s:e]))
        key = (e - s, depth)
        if best is None or key > best[0]:
            best = (key, s, e)
    if best is None:
        return None
    return best[1], best[2]


def detect_sections(envelope: np.ndarray, frame_rate: float) -> TrackStructure:
    """Locate intro end, drop window, and break from a raw RMS envelope.

    Pure over the envelope so it is unit-testable without audio.  Boundaries are
    compensated for the half-window lag the moving average introduces.
    """
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    n = int(env.size)
    if n == 0 or not math.isfinite(frame_rate) or frame_rate <= 0:
        return TrackStructure()
    duration = n / float(frame_rate)

    smooth_frames = max(1, int(round(SMOOTH_SECONDS * frame_rate)))
    level = _moving_average(env, smooth_frames)
    peak_level = float(np.percentile(level, 95.0))
    if not math.isfinite(peak_level) or peak_level <= 1e-9:
        return TrackStructure()

    # ── Drop: the highest-energy window ──
    w = max(1, int(round(PEAK_WINDOW_SECONDS * frame_rate)))
    if n <= w:
        peak_start, peak_end = 0.0, duration
    else:
        csum = np.concatenate(([0.0], np.cumsum(level)))
        means = (csum[w:] - csum[:-w]) / w
        # The first window at (within float noise of) the maximum, not an arbitrary
        # later equal-energy window: the drop starts where energy first peaks.
        best = float(means.max())
        i = int(np.argmax(means >= best - 1e-9))
        peak_start = max(0.0, i / float(frame_rate) - SMOOTH_SECONDS / 2.0)
        peak_end = min(duration, peak_start + PEAK_WINDOW_SECONDS)

    # ── Intro end: first sustained crossing of the intro level ──
    half_lag = SMOOTH_SECONDS / 2.0
    intro_idx = _first_sustained(
        level >= INTRO_LEVEL_FRACTION * peak_level, smooth_frames
    )
    intro_end = (
        None
        if intro_idx is None
        else max(0.0, intro_idx / float(frame_rate) - half_lag)
    )

    # ── Intro start: first sustained non-silence, so lead-in silence is skipped ──
    start_hold = max(1, int(round(INTRO_START_HOLD_SECONDS * frame_rate)))
    start_idx = _first_sustained(
        level >= INTRO_START_LEVEL_FRACTION * peak_level, start_hold
    )
    intro_start = (
        None
        if start_idx is None
        else max(0.0, start_idx / float(frame_rate) - half_lag)
    )
    if intro_start is not None and intro_end is not None:
        intro_start = min(intro_start, intro_end)

    # ── Break: longest interior sustained dip that returns to energy ──
    dip = _longest_dip(
        level <= BREAK_LEVEL_FRACTION * peak_level,
        level,
        BREAK_LEVEL_FRACTION * peak_level,
        max(1, int(round(BREAK_MIN_SECONDS * frame_rate))),
    )
    break_start = break_end = None
    if dip is not None:
        s, e = dip
        start = (s + smooth_frames // 2) / float(frame_rate)
        end = (e - smooth_frames // 2) / float(frame_rate)
        if end - start >= BREAK_MIN_SECONDS * 0.5:
            break_start, break_end = start, end

    return TrackStructure(
        intro_start=intro_start,
        intro_end=intro_end,
        peak_start=peak_start,
        peak_end=peak_end,
        break_start=break_start,
        break_end=break_end,
    )


def analyze_structure(path: str) -> TrackStructure | None:
    """Decode *path* and detect its coarse sections; ``None`` when undecodable."""
    try:
        y, sr = librosa_load(path, sr=STRUCTURE_SR, mono=True)
    except Exception as e:  # noqa: BLE001 — reported per job, never raised
        log.warning("Structure: failed to decode %s: %s", path, e)
        return None
    if y is None or np.asarray(y).size == 0:
        return None
    env = rms_envelope(y, int(sr))
    return detect_sections(env, FRAME_RATE)


def encode_structure(structure: TrackStructure) -> np.ndarray:
    """Pack a :class:`TrackStructure` into the fixed-length float32 blob."""
    return np.asarray(
        [
            _ABSENT if getattr(structure, f) is None else float(getattr(structure, f))
            for f in STRUCTURE_FIELDS
        ],
        dtype=np.float32,
    )


def decode_structure(blob: np.ndarray) -> dict[str, float | None]:
    """Unpack a cached blob to a JSON-safe ``{field: seconds | None}`` dict."""
    arr = np.asarray(blob, dtype=np.float32).reshape(-1)
    out: dict[str, float | None] = {}
    for i, field in enumerate(STRUCTURE_FIELDS):
        v = float(arr[i]) if i < arr.size else _ABSENT
        out[field] = None if not math.isfinite(v) else v
    return out


def compute_structure(path: str) -> dict[str, Any]:
    """Worker entrypoint: detect and pack, or report a per-track error.

    Module-level so a spawn process pool can pickle it by reference, mirroring
    ``features_worker.compute_track_features``.
    """
    try:
        structure = analyze_structure(path)
    except Exception as e:  # noqa: BLE001 — reported per job, never raised
        return {"structure": None, "error": f"{type(e).__name__}: {e}"}
    if structure is None:
        return {"structure": None, "error": "decode failed"}
    return {"structure": encode_structure(structure), "error": None}
