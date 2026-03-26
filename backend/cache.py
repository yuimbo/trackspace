"""Two-layer cache for track metadata (ID3 tags + artist/title).

Layer 1 – in-memory LRU (``OrderedDict``, bounded by *max_memory* entries).
Layer 2 – SQLite on disk (survives server restarts).

Cache key: ``(absolute_path, mtime)``
  • mtime comes from ``os.path.getmtime()`` so any write to the file changes
    the key, naturally invalidating the old entry without explicit eviction.

Thread-safety:
  The memory dict is guarded by a ``threading.Lock``.
  SQLite connections are created per-call so the pool threads don't share
  a connection object (Python's sqlite3 is not safe to share across threads).
  WAL journal mode lets multiple readers proceed concurrently with a writer.
"""

import json
import os
import sqlite3
import threading
from collections import OrderedDict


class TrackCache:
    def __init__(self, db_path: str, max_memory: int = 4000) -> None:
        self._db_path = db_path
        self._max_memory = max_memory
        self._mem: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS track_cache (
                    path  TEXT    NOT NULL,
                    mtime REAL    NOT NULL,
                    data  TEXT    NOT NULL,
                    PRIMARY KEY (path, mtime)
                )
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _lru_put(self, key: tuple, data: dict) -> None:
        """Insert/promote *key* in the memory LRU, evicting LRU entry if full."""
        self._mem[key] = data
        self._mem.move_to_end(key)
        if len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, path: str, mtime: float) -> dict | None:
        """Return cached data for *(path, mtime)* or ``None`` on a miss."""
        key = (path, mtime)

        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]

        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT data FROM track_cache WHERE path=? AND mtime=?",
                    (path, mtime),
                ).fetchone()
        except Exception:
            return None

        if row is None:
            return None

        data: dict = json.loads(row[0])
        with self._lock:
            self._lru_put(key, data)
        return data

    def put(self, path: str, mtime: float, data: dict) -> None:
        """Store *data* under *(path, mtime)* in both layers."""
        key = (path, mtime)
        serialized = json.dumps(data, separators=(",", ":"))

        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO track_cache (path, mtime, data) VALUES (?,?,?)",
                    (path, mtime, serialized),
                )
        except Exception:
            pass

        with self._lock:
            self._lru_put(key, data)

    def remap(self, old_path: str, new_path: str) -> int:
        """Re-key all cache entries from *old_path* to *new_path*.

        Called after a single-file move so the entry survives under the new
        absolute path rather than becoming a stale orphan.  Returns the number
        of rows updated (0 or 1 in practice).
        """
        updated = 0
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE track_cache SET path=? WHERE path=?",
                    (new_path, old_path),
                )
                updated = cur.rowcount
        except Exception:
            pass

        with self._lock:
            keys = [k for k in self._mem if k[0] == old_path]
            for key in keys:
                data = self._mem.pop(key)
                new_key = (new_path, key[1])
                self._mem[new_key] = data
                self._mem.move_to_end(new_key)

        return updated

    def remap_prefix(self, old_prefix: str, new_prefix: str) -> int:
        """Re-key all cache entries whose path starts with *old_prefix*.

        Used when a folder is renamed: pass the old and new absolute folder
        paths (with a trailing separator) to bulk-update every entry inside.
        Returns the number of rows updated.
        """
        updated = 0
        try:
            with self._conn() as conn:
                escaped = old_prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
                rows = conn.execute(
                    "SELECT path, mtime FROM track_cache WHERE path LIKE ? ESCAPE '\\'",
                    (escaped + "%",),
                ).fetchall()
                for old_path, mtime in rows:
                    new_path = new_prefix + old_path[len(old_prefix):]
                    conn.execute(
                        "UPDATE track_cache SET path=? WHERE path=? AND mtime=?",
                        (new_path, old_path, mtime),
                    )
                    updated += 1
        except Exception:
            pass

        with self._lock:
            keys = [k for k in self._mem if k[0].startswith(old_prefix)]
            for key in keys:
                data = self._mem.pop(key)
                new_path = new_prefix + key[0][len(old_prefix):]
                new_key = (new_path, key[1])
                self._mem[new_key] = data
                self._mem.move_to_end(new_key)

        return updated

    def prune_missing(self) -> int:
        """Delete DB entries whose files no longer exist on disk.

        Safe to call at startup to keep the SQLite file from growing
        unboundedly after tracks are deleted or moved.  Returns the number
        of rows removed.
        """
        try:
            with self._conn() as conn:
                paths = [r[0] for r in conn.execute("SELECT DISTINCT path FROM track_cache")]
                dead = [p for p in paths if not os.path.exists(p)]
                if dead:
                    conn.executemany(
                        "DELETE FROM track_cache WHERE path=?",
                        [(p,) for p in dead],
                    )
                return len(dead)
        except Exception:
            return 0
