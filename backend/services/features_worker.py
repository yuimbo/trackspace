"""Child-process entry point for the CPU-bound ``features`` stage.

The stage is independent per track, but it cannot be parallelised across *threads*:
madmom's BLSTM forward pass is GIL-bound, so several features threads do not run
in parallel and actively convoy (measured on one track: 0.8 s at 1 thread, 2.5 s at
2, 10.5 s at 4). The stage is parallelised across **processes** instead, which
scales near-linearly.

This module is deliberately import-light: a spawned child imports it (and only it) to
reach the two feature extractors, without pulling in torch or the embeddings
package's layout / projection stack. ``warm()`` is the pool initializer, so each
child builds its madmom processors once rather than on its first track.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def compute_track_features(
    abs_path: str,
    *,
    want_features: bool,
    want_rhythm: bool,
) -> dict[str, Any]:
    """Decode and compute the requested blocks for one track.

    Runs in a pool child (or in-process when the pool is disabled). Returns plain
    numpy arrays so the parent owns both the cache write and the decode-warning
    relay; a failure is reported in ``error`` rather than raised, so one bad track
    cannot take down a batch.
    """
    out: dict[str, Any] = {
        "features": None,
        "rhythm": None,
        "warnings": {},
        "error": None,
    }
    try:
        if want_features:
            from backend.embeddings import librosa_audio_features

            warns: dict[str, str] = {}
            out["features"] = librosa_audio_features.generate_audio_features_batch(
                [abs_path], decode_warnings=warns
            ).get(abs_path)
            out["warnings"].update(warns)
        if want_rhythm:
            from backend.embeddings import rhythm_features

            warns = {}
            out["rhythm"] = rhythm_features.generate_rhythm_features_batch(
                [abs_path], decode_warnings=warns
            ).get(abs_path)
            out["warnings"].update(warns)
    except Exception as e:  # noqa: BLE001 — reported per job, never raised
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def warm() -> None:
    """Pool initializer: build this child's decoders and madmom processors once."""
    try:
        from backend.embeddings import librosa_audio_features, rhythm_features

        librosa_audio_features.warmup_audio_features()
        rhythm_features.warmup_rhythm_features()
    except Exception:
        log.debug("features worker warmup skipped", exc_info=True)
