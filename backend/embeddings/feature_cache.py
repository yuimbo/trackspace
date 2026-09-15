"""Content-addressed cache for audio features (CLAP embeddings, EffNet
embeddings, MAEST embeddings/style logits, rhythm features, and extracted
audio features).

Keyed by chromaprint fingerprint hash so identical audio content shares
one entry regardless of filename or location.

Same two-layer (memory LRU + SQLite) pattern as TrackCache.

All sources share one generic implementation, driven by the :data:`_SOURCES`
registry. Each source carries its own ``version`` integer so that bumping one
source (e.g. CLAP) does not invalidate others (e.g. EffNet). The public API is
source-parameterised::

    cache.get("clap", fp)
    cache.put("rhythm", fp, vec)
    cache.has("maest", fp)
    cache.get_all("effnet", fingerprints)
    cache.count("features")
    cache.purge_old_versions("clap", current_version)
    cache.put_maest(fp, embedding, logits)   # paired escape hatch

Canonical source names are ``clap``, ``effnet``, ``features``, ``maest``,
``maest_logits`` and ``rhythm``.
"""

import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: ``WHERE ... IN (...)`` chunk size, safely below SQLite's variable limit.
_IN_CHUNK = 500


@dataclass(frozen=True)
class _SourceSpec:
    """Where one cached source lives in the shared ``audio_features`` table.

    ``purge_cols`` / ``purge_prefixes`` / ``count_cols`` default to the single
    blob column and LRU prefix. The MAEST pair overrides them because its
    embedding and logits share one version column and are written (and purged)
    together.
    """

    blob_col: str
    version_col: str
    lru_prefix: str
    purge_cols: tuple[str, ...] = ()
    purge_prefixes: tuple[str, ...] = ()
    count_cols: tuple[str, ...] = ()

    @property
    def null_cols(self) -> tuple[str, ...]:
        return self.purge_cols or (self.blob_col,)

    @property
    def evict_prefixes(self) -> tuple[str, ...]:
        return self.purge_prefixes or (self.lru_prefix,)

    @property
    def counted_cols(self) -> tuple[str, ...]:
        return self.count_cols or (self.blob_col,)


#: Canonical source registry. Names match the ``sources`` vocabulary shared by
#: projection, clustering and coverage.
_SOURCES: dict[str, _SourceSpec] = {
    "clap": _SourceSpec("clap_embedding", "version", "clap:"),
    "effnet": _SourceSpec("effnet_embedding", "effnet_version", "effnet:"),
    "features": _SourceSpec("features", "features_version", "feat:"),
    "maest": _SourceSpec(
        "maest_embedding",
        "maest_version",
        "maest:",
        purge_cols=("maest_embedding", "maest_logits"),
        purge_prefixes=("maest:", "maestlog:"),
        count_cols=("maest_embedding", "maest_logits"),
    ),
    "maest_logits": _SourceSpec("maest_logits", "maest_version", "maestlog:"),
    "rhythm": _SourceSpec("rhythm", "rhythm_version", "rhythm:"),
}

#: ``clap_embedding`` is NOT NULL, so writing another source to a new row pads
#: it with an empty blob.
_CLAP_PAD = b""


class FeatureCache:
    def __init__(self, db_path: str, max_memory: int = 2000) -> None:
        self._db_path = db_path
        self._max_memory = max_memory
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        #: Bumps on successful writes — invalidates embedding status caches.
        self._write_epoch = 0
        self._init_db()

    def write_epoch(self) -> int:
        """Monotonic counter; increases when cached embedding rows change."""
        with self._lock:
            return self._write_epoch

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if os.environ.get("TRACKSPACE_SQLITE_INTEGRITY", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                row = conn.execute("PRAGMA integrity_check").fetchone()
                if not row or row[0] != "ok":
                    log.warning(
                        "FeatureCache %s: PRAGMA integrity_check failed (%r). "
                        "Stop the server, back up the file if needed, and remove or "
                        "replace the corrupted DB; it will be recreated empty.",
                        self._db_path,
                        row[0] if row else None,
                    )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_features (
                    fingerprint TEXT PRIMARY KEY,
                    clap_embedding BLOB NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            for col, default in [
                ("version", "INTEGER NOT NULL DEFAULT 0"),
                ("effnet_embedding", "BLOB"),
                ("effnet_version", "INTEGER NOT NULL DEFAULT 0"),
                ("features", "BLOB"),
                ("features_version", "INTEGER NOT NULL DEFAULT 0"),
                ("maest_embedding", "BLOB"),
                ("maest_logits", "BLOB"),
                ("maest_version", "INTEGER NOT NULL DEFAULT 0"),
                ("rhythm", "BLOB"),
                ("rhythm_version", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE audio_features ADD COLUMN {col} {default}"
                    )
                except sqlite3.OperationalError:
                    pass

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _spec(source: str) -> _SourceSpec:
        try:
            return _SOURCES[source]
        except KeyError:
            raise ValueError(f"unknown source {source!r}") from None

    # ── Memory LRU ────────────────────────────────────────────
    # Keys are prefixed per source: "clap:fp", "effnet:fp", "feat:fp",
    # "maest:fp", "maestlog:fp", "rhythm:fp".

    def _lru_put(self, key: str, data: np.ndarray) -> None:
        self._mem[key] = data
        self._mem.move_to_end(key)
        if len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)

    def _evict_mem_keys_with_prefix_unlocked(self, prefix: str) -> None:
        dead = [k for k in self._mem if k.startswith(prefix)]
        for k in dead:
            del self._mem[k]

    # ------------------------------------------------------------------
    # Generic per-source API
    # ------------------------------------------------------------------

    def get(self, source: str, fingerprint: str, version: int = 0) -> np.ndarray | None:
        spec = self._spec(source)
        key = spec.lru_prefix + fingerprint
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None:
                self._mem.move_to_end(key)
                return hit
        try:
            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT {spec.blob_col} FROM audio_features "
                    f"WHERE fingerprint=? AND {spec.version_col}=? "
                    f"AND length({spec.blob_col}) > 0",
                    (fingerprint, version),
                ).fetchone()
        except Exception:
            return None
        if row is None or row[0] is None:
            return None
        arr = np.frombuffer(row[0], dtype=np.float32).copy()
        with self._lock:
            self._lru_put(key, arr)
        return arr

    def put(
        self, source: str, fingerprint: str, vec: np.ndarray, version: int = 0
    ) -> None:
        spec = self._spec(source)
        arr = np.asarray(vec, dtype=np.float32)
        cols = ["fingerprint", spec.blob_col, spec.version_col]
        vals: list = [fingerprint, arr.tobytes(), version]
        if spec.blob_col != "clap_embedding":
            cols.insert(1, "clap_embedding")
            vals.insert(1, _CLAP_PAD)
        sql = (
            f"INSERT INTO audio_features ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(fingerprint) DO UPDATE SET "
            f"{spec.blob_col}=excluded.{spec.blob_col}, "
            f"{spec.version_col}=excluded.{spec.version_col}"
        )
        try:
            with self._conn() as conn:
                conn.execute(sql, vals)
        except Exception:
            return
        with self._lock:
            self._write_epoch += 1
            self._lru_put(spec.lru_prefix + fingerprint, arr.astype(np.float32))

    def has(self, source: str, fingerprint: str, version: int = 0) -> bool:
        spec = self._spec(source)
        key = spec.lru_prefix + fingerprint
        with self._lock:
            if key in self._mem:
                return True
        try:
            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT 1 FROM audio_features WHERE fingerprint=? AND "
                    f"{spec.version_col}=? AND length({spec.blob_col}) > 0",
                    (fingerprint, version),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def get_all(
        self, source: str, fingerprints: list[str], version: int = 0
    ) -> dict[str, np.ndarray]:
        spec = self._spec(source)
        result: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._lock:
            for fp in fingerprints:
                key = spec.lru_prefix + fp
                hit = self._mem.get(key)
                if hit is not None:
                    self._mem.move_to_end(key)
                    result[fp] = hit
                else:
                    missing.append(fp)
        if missing:
            try:
                with self._conn() as conn:
                    for i in range(0, len(missing), _IN_CHUNK):
                        batch = missing[i : i + _IN_CHUNK]
                        placeholders = ",".join("?" for _ in batch)
                        rows = conn.execute(
                            f"SELECT fingerprint, {spec.blob_col} FROM audio_features "
                            f"WHERE fingerprint IN ({placeholders}) AND "
                            f"{spec.version_col}=? AND length({spec.blob_col}) > 0",
                            [*batch, version],
                        ).fetchall()
                        for fp, blob in rows:
                            if blob is None:
                                continue
                            arr = np.frombuffer(blob, dtype=np.float32).copy()
                            result[fp] = arr
                            with self._lock:
                                self._lru_put(spec.lru_prefix + fp, arr)
            except Exception:
                pass
        return result

    def count(self, source: str, version: int | None = None) -> int:
        spec = self._spec(source)
        present = " OR ".join(f"length({c}) > 0" for c in spec.counted_cols)
        try:
            with self._conn() as conn:
                if version is not None:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM audio_features WHERE "
                        f"{spec.version_col}=? AND ({present})",
                        (version,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM audio_features WHERE ({present})"
                    ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def purge_old_versions(self, source: str, current_version: int) -> int:
        """Blank out *source* rows whose version is older than *current_version*.

        Blobs are overwritten with an empty blob rather than NULL because
        ``clap_embedding`` is NOT NULL; empty blobs are treated as absent on
        every read path.
        """
        spec = self._spec(source)
        sets = ", ".join(f"{c}=X''" for c in spec.null_cols)
        present = " OR ".join(f"length({c}) > 0" for c in spec.null_cols)
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    f"UPDATE audio_features SET {sets} WHERE "
                    f"{spec.version_col} < ? AND ({present})",
                    (current_version,),
                )
                n = cursor.rowcount or 0
        except Exception:
            return 0
        if n:
            with self._lock:
                self._write_epoch += 1
                for prefix in spec.evict_prefixes:
                    self._evict_mem_keys_with_prefix_unlocked(prefix)
        return n

    def put_maest(
        self,
        fingerprint: str,
        embedding: np.ndarray,
        logits: np.ndarray,
        version: int = 0,
    ) -> None:
        """Write MAEST embedding + style logits and their shared version in one round-trip."""
        emb_arr = np.asarray(embedding, dtype=np.float32)
        log_arr = np.asarray(logits, dtype=np.float32)
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO audio_features
                           (fingerprint, clap_embedding, maest_embedding, maest_logits, maest_version)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(fingerprint) DO UPDATE SET
                           maest_embedding=excluded.maest_embedding,
                           maest_logits=excluded.maest_logits,
                           maest_version=excluded.maest_version""",
                    (fingerprint, _CLAP_PAD, emb_arr.tobytes(), log_arr.tobytes(), version),
                )
        except Exception:
            return
        with self._lock:
            self._write_epoch += 1
            self._lru_put("maest:" + fingerprint, emb_arr.astype(np.float32))
            self._lru_put("maestlog:" + fingerprint, log_arr.astype(np.float32))
