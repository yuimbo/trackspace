from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, as_completed
from typing import Any


def iter_parallel_reads(
    paths: Sequence[str],
    *,
    read_fn: Callable[[str], dict[str, Any]],
    executor: Executor,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (path, read_result) as each parallel read completes."""
    futures = {executor.submit(read_fn, path): path for path in paths}
    for fut in as_completed(futures):
        path = futures[fut]
        yield path, fut.result()


def read_paths_parallel(
    paths: Sequence[str],
    *,
    read_fn: Callable[[str], dict[str, Any]],
    executor: Executor,
) -> dict[str, dict[str, Any]]:
    """Return a path->read_result map by reading all paths in parallel."""
    return {path: info for path, info in iter_parallel_reads(paths, read_fn=read_fn, executor=executor)}
