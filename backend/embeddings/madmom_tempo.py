"""Dominant tempo via madmom (RNN beat activations + comb-filter histogram).

Optional dependency: if madmom cannot be imported or fails at runtime, callers
fall back to librosa elsewhere.

madmom 0.16.x expects legacy NumPy aliases and ``collections.MutableSequence``;
we patch those before importing the package where needed.

**Sample rate:** ``RNNBeatProcessor`` expects 44.1 kHz mono. Classical features decode
at 22.05 kHz; we resample that buffer to 44.1 kHz here only (cheap vs. running all
librosa features at 44.1 kHz).

The default beat tracker loads an **8× BLSTM** ensemble; set env
``TRACKSPACE_MADMOM_FAST=1`` to use a single network (~4× faster per excerpt,
slightly less robust).

**Threading:** the BLSTM layers are stateful, so processors are *per thread* — see
:data:`_local`. Never share one processor instance between analysis workers.
"""

from __future__ import annotations

import logging
import os
import threading

import librosa
import numpy as np

log = logging.getLogger(__name__)

# Must match ``RNNBeatProcessor``'s internal ``SignalProcessor(sample_rate=…)``.
_MADMOM_TARGET_SR = 44100

#: madmom's BLSTM layers carry mutable hidden state across frames
#: (``RecurrentLayer.activate(data, reset=...)`` / ``reset()``), so one processor
#: instance cannot be called from two threads at once — concurrent activations
#: interleave the state and silently produce wrong beat activations. Each analysis
#: worker therefore owns its own processor pair. This also removes the lazy-init race.
_local = threading.local()

_madmom_checked: bool | None = None
_madmom_ok = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _apply_madmom_runtime_shims() -> None:
    import collections
    import collections.abc

    try:
        from collections import MutableSequence  # noqa: F401
    except ImportError:
        collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]

    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = np.int64  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = np.complex128  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]


def is_madmom_tempo_available() -> bool:
    global _madmom_checked, _madmom_ok
    if _madmom_checked is not None:
        return _madmom_ok
    _madmom_checked = True
    try:
        _apply_madmom_runtime_shims()
        import madmom  # noqa: F401
        from madmom.features.beats import RNNBeatProcessor  # noqa: F401
        from madmom.features.tempo import TempoEstimationProcessor  # noqa: F401

        _madmom_ok = True
    except Exception as e:
        log.debug("madmom tempo not available: %s", e)
        _madmom_ok = False
    return _madmom_ok


def _get_processors():
    """This thread's ``(beat, tempo)`` processors, built once on first use."""
    if not is_madmom_tempo_available():
        raise RuntimeError("madmom not available")
    procs = getattr(_local, "processors", None)
    if procs is None:
        _apply_madmom_runtime_shims()
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        from madmom.models import BEATS_BLSTM

        nn_kw = {}
        if _env_truthy("TRACKSPACE_MADMOM_FAST"):
            nn_kw["nn_files"] = [BEATS_BLSTM[0]]
        procs = (RNNBeatProcessor(**nn_kw), TempoEstimationProcessor(fps=100))
        _local.processors = procs
    return procs


def estimate_tempo_bpm(audio: np.ndarray, sr: int) -> float | None:
    """Strongest madmom tempo candidate in BPM, or ``None``."""
    if not is_madmom_tempo_available():
        return None
    if audio is None or not len(audio):
        return None
    try:
        from madmom.audio.signal import Signal

        beat_proc, tempo_proc = _get_processors()
        y = np.ascontiguousarray(audio, dtype=np.float32)
        if int(sr) != _MADMOM_TARGET_SR:
            y = librosa.resample(
                y,
                orig_sr=float(sr),
                target_sr=float(_MADMOM_TARGET_SR),
                res_type="soxr_qq",
            ).astype(np.float32)
        sig = Signal(y, sample_rate=_MADMOM_TARGET_SR)
        act = beat_proc(sig)
        if act is None or len(act) < 2:
            return None
        tempi = tempo_proc(act)
        if tempi is None or len(tempi) == 0:
            return None
        bpm = float(tempi[0][0])
        if not np.isfinite(bpm) or bpm <= 0:
            return None
        return bpm
    except Exception as e:
        log.debug("madmom tempo estimation failed: %s", e)
        return None


def warmup_madmom_tempo() -> None:
    if not is_madmom_tempo_available():
        return
    try:
        seconds = 0.5
        n = int(_MADMOM_TARGET_SR * seconds)
        t = np.linspace(0.0, seconds, num=n, endpoint=False, dtype=np.float32)
        y = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        _ = estimate_tempo_bpm(y, _MADMOM_TARGET_SR)
    except Exception as e:
        log.debug("madmom tempo warmup skipped: %s", e)
