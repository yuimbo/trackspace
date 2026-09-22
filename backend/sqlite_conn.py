"""Shared SQLite helpers.

**Closing connections.** ``with sqlite3.connect(path) as conn:`` is a trap: the stdlib
connection context manager only ends the transaction (commit on success, rollback on
exception) — it does **not** close the handle. A long-lived server that opens a
connection per cache read/write therefore leaks one connection and its db/wal/shm file
descriptors per call, until ``RLIMIT_NOFILE`` is exhausted and SQLite starts failing
with "unable to open database file" (observed: 112k open ``track_cache.db`` handles in a
3-minute-old process, after ~10 library-wide scans). Wrap the connection in
:class:`ClosingConnection` so the existing ``with self._conn() as conn:`` idiom keeps its
transaction semantics and closes the handle.

**Batching.** ``chunked`` splits a key list into ``IN (...)``-sized batches, since
SQLite's variable limit caps how many ``?`` one statement may carry.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence

#: Keys per ``IN (...)`` batch; well under SQLite's variable limit.
CHUNK = 500


class ClosingConnection:
    """Commit/roll back like the stdlib context manager, then close the handle."""

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            return self._conn.__exit__(exc_type, exc, tb)
        finally:
            self._conn.close()


def chunked(
    items: Sequence, size: int = CHUNK, unit: str = "?"
) -> Iterator[tuple[Sequence, str]]:
    """Yield ``(batch, "?,?,…")`` for each *size*-sized slice of *items*.

    *unit* is the placeholder for one item — ``"(?,?)"`` for a composite key like
    ``(path, mtime)`` — so a caller can bind a chunk straight into ``IN (...)``.
    """
    for i in range(0, len(items), size):
        batch = items[i : i + size]
        yield batch, ",".join(unit for _ in batch)
