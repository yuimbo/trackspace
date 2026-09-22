from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from typing import Any

from backend.services.parallel_reads import read_paths_parallel

#: Tracks per SSE ``progress`` event. One event per track forced a JSON.parse and
#: an EventSource dispatch per track on the main thread (~14k at library scale);
#: chunking keeps the payload identical while cutting the event count.
DEFAULT_CHUNK = 500


def prefetch_track_cache(track_cache: Any, paths: Sequence[str]) -> None:
    """Warm *track_cache*'s memory LRU with one batched read per chunk.

    ``TrackCache.get`` opens a connection per call, so a library scan that reads
    each track individually pays ~730 us/track; ``get_many`` collapses it.
    """
    if track_cache is None or not paths:
        return
    pairs: list[tuple[str, float]] = []
    for p in paths:
        try:
            pairs.append((p, os.path.getmtime(p)))
        except OSError:
            continue
    if pairs:
        track_cache.get_many(pairs)


def emit_scan_progress_events(
    paths: Sequence[str],
    *,
    read_fn: Callable[[str], dict[str, Any]],
    executor: Executor,
    track_from_read: Callable[..., dict[str, Any]],
    emit: Callable[[dict[str, Any]], None],
    track_cache: Any | None = None,
    feature_cache: Any | None = None,
    features_version: int = 0,
    chunk_size: int = DEFAULT_CHUNK,
) -> None:
    """Read *paths* in chunks and emit one ``progress`` event per chunk.

    Reading is separated from emitting so the per-track feature lookup inside
    *track_from_read* becomes one batched cache read per chunk instead of one
    SQLite connection per track. *track_cache* warms the metadata LRU up front.
    """
    total = len(paths)
    done = 0
    chunk = max(1, int(chunk_size))
    for start in range(0, total, chunk):
        batch_paths = list(paths[start : start + chunk])
        prefetch_track_cache(track_cache, batch_paths)
        results = read_paths_parallel(
            batch_paths, read_fn=read_fn, executor=executor
        )
        feat_by_fp = None
        if feature_cache is not None:
            fps = [
                results[p].get("fingerprint")
                for p in batch_paths
                if results.get(p, {}).get("fingerprint")
            ]
            feat_by_fp = (
                feature_cache.get_all("features", fps, version=features_version)
                if fps
                else {}
            )
        tracks = [
            track_from_read(p, results.get(p, {}), feat_by_fp=feat_by_fp)
            for p in batch_paths
        ]
        done += len(batch_paths)
        emit({"type": "progress", "done": done, "total": total, "tracks": tracks})
    emit({"type": "done"})
