"""Content-addressed cache for audio features (CLAP embeddings, etc.).

Keyed by chromaprint fingerprint hash so identical audio content shares
one entry regardless of filename or location.

Same two-layer (memory LRU + SQLite) pattern as TrackCache.

Each entry carries a ``version`` integer that tracks the embedding strategy
used to generate it.  Callers pass the current version; entries with a
mismatched version are treated as missing so they get regenerated.
"""

import os
import sqlite3
import threading
from collections import OrderedDict

import numpy as np


class FeatureCache:
    def __init__(self, db_path: str, max_memory: int = 2000) -> None:
        self._db_path = db_path
        self._max_memory = max_memory
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_features (
                    fingerprint TEXT PRIMARY KEY,
                    clap_embedding BLOB NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Migrate older databases that lack the version column.
            try:
                conn.execute(
                    "ALTER TABLE audio_features ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _lru_put(self, key: str, data: np.ndarray) -> None:
        self._mem[key] = data
        self._mem.move_to_end(key)
        if len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API — all read methods filter by *version*.
    # The in-memory LRU is only populated from version-matching DB rows
    # (or from put_embedding), so the memory check is safe without an
    # explicit version comparison.
    # ------------------------------------------------------------------

    def get_embedding(self, fingerprint: str, version: int = 0) -> np.ndarray | None:
        with self._lock:
            if fingerprint in self._mem:
                self._mem.move_to_end(fingerprint)
                return self._mem[fingerprint]

        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT clap_embedding FROM audio_features WHERE fingerprint=? AND version=?",
                    (fingerprint, version),
                ).fetchone()
        except Exception:
            return None

        if row is None:
            return None

        arr = np.frombuffer(row[0], dtype=np.float32).copy()
        with self._lock:
            self._lru_put(fingerprint, arr)
        return arr

    def put_embedding(self, fingerprint: str, embedding: np.ndarray, version: int = 0) -> None:
        blob = embedding.astype(np.float32).tobytes()
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO audio_features (fingerprint, clap_embedding, version) VALUES (?,?,?)",
                    (fingerprint, blob, version),
                )
        except Exception:
            pass

        with self._lock:
            self._lru_put(fingerprint, embedding.astype(np.float32))

    def has_embedding(self, fingerprint: str, version: int = 0) -> bool:
        with self._lock:
            if fingerprint in self._mem:
                return True
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM audio_features WHERE fingerprint=? AND version=?",
                    (fingerprint, version),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def get_all_embeddings(self, fingerprints: list[str], version: int = 0) -> dict[str, np.ndarray]:
        """Bulk-fetch embeddings for a list of fingerprints."""
        result: dict[str, np.ndarray] = {}
        missing: list[str] = []

        with self._lock:
            for fp in fingerprints:
                if fp in self._mem:
                    self._mem.move_to_end(fp)
                    result[fp] = self._mem[fp]
                else:
                    missing.append(fp)

        if missing:
            try:
                with self._conn() as conn:
                    placeholders = ",".join("?" for _ in missing)
                    rows = conn.execute(
                        f"SELECT fingerprint, clap_embedding FROM audio_features "
                        f"WHERE fingerprint IN ({placeholders}) AND version=?",
                        [*missing, version],
                    ).fetchall()
                    for fp, blob in rows:
                        arr = np.frombuffer(blob, dtype=np.float32).copy()
                        result[fp] = arr
                        with self._lock:
                            self._lru_put(fp, arr)
            except Exception:
                pass

        return result

    def count(self, version: int | None = None) -> int:
        try:
            with self._conn() as conn:
                if version is not None:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE version=?",
                        (version,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) FROM audio_features").fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def purge_old_versions(self, current_version: int) -> int:
        """Delete entries with version < *current_version*."""
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM audio_features WHERE version < ?",
                    (current_version,),
                )
                return cursor.rowcount
        except Exception:
            return 0
