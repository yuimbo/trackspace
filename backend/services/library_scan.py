from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from typing import Any

from backend.services.parallel_reads import iter_parallel_reads


def emit_scan_progress_events(
    paths: Sequence[str],
    *,
    read_fn: Callable[[str], dict[str, Any]],
    executor: Executor,
    track_from_read: Callable[[str, dict[str, Any]], dict[str, Any]],
    emit: Callable[[dict[str, Any]], None],
) -> None:
    """Emit progress events while scanning tracks, then emit a done event."""
    total = len(paths)
    done = 0
    for path, info in iter_parallel_reads(paths, read_fn=read_fn, executor=executor):
        done += 1
        emit({
            "type": "progress",
            "done": done,
            "total": total,
            "track": track_from_read(path, info),
        })
    emit({"type": "done"})
