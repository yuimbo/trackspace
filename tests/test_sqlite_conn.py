"""The connection wrapper that stops the cache FD leak.

``with sqlite3.connect(...) as conn`` ends the transaction but does **not** close the
handle, so a server that opens a connection per cache access leaked one connection (and
its db/wal/shm descriptors) per call — observed as 112k open ``track_cache.db`` handles
in a 3-minute-old process, then "unable to open database file" everywhere. These pin the
commit/rollback semantics *and* the close.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.cache import TrackCache
from backend.embeddings.feature_cache import FeatureCache
from backend.jobs.queue import JobQueue
from backend.sqlite_conn import ClosingConnection


def test_commits_then_closes(tmp_path):
    path = str(tmp_path / "t.db")
    with ClosingConnection(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    # The commit survived the closed handle.
    with ClosingConnection(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT x FROM t").fetchone()[0] == 1


def test_rolls_back_and_closes_on_error(tmp_path):
    path = str(tmp_path / "t.db")
    with ClosingConnection(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")

    with pytest.raises(ValueError):
        with ClosingConnection(sqlite3.connect(path)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise ValueError("boom")

    with ClosingConnection(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: TrackCache(str(p / "track.db")),
        lambda p: FeatureCache(str(p / "features.db")),
        lambda p: JobQueue(str(p / "jobs.db")),
    ],
)
def test_cache_connections_close(tmp_path, factory):
    cache = factory(tmp_path)
    with cache._conn() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
