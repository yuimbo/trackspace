"""Content-addressed cache for audio features (CLAP embeddings, MAEST
embeddings/style logits, rhythm features, and extracted audio features).

Keyed by chromaprint fingerprint hash so identical audio content shares
one entry regardless of filename or location.

Same two-layer (memory LRU + SQLite) pattern as TrackCache.

All sources share one generic implementation, driven by the :data:`_SOURCES`
registry. Each source carries its own ``version`` integer so that bumping one
source (e.g. CLAP) does not invalidate others (e.g. MAEST). The public API is
source-parameterised::

    cache.get("clap", fp)
    cache.put("rhythm", fp, vec)
    cache.has("maest", fp)
    cache.get_all("supervised", fingerprints)
    cache.count("features")
    cache.purge_old_versions("clap", current_version)
    cache.put_maest(fp, embedding, logits, excerpts)   # paired escape hatch

Canonical source names are ``clap``, ``features``, ``maest``, ``maest_logits``,
``maest_excerpts`` and ``rhythm``.

``maest_excerpts`` stores the per-excerpt MAEST embeddings as one flat float32
blob of shape ``(n_excerpts, MAEST_EMBED_DIM)``. It shares ``maest_version``
because a single model pass produces the pooled embedding, the logits and the
excerpts together, and the reader owns the dimension (the blob carries no header).
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from backend.sqlite_conn import ClosingConnection, chunked

log = logging.getLogger(__name__)

#: Persisted layouts (2-D or 3-D) kept on disk, keyed by layout revision.
_LAYOUT_KEEP = 32

#: Persisted fitted t-SNE embeddings kept on disk, keyed by vector config. The
#: in-process fit cache dies with the process, so without this a restart refits
#: every lens (~20 s each) even though the layout coordinates are already cached.
_TSNE_FIT_KEEP = 8

#: Blob magic + format version. An unreadable or older blob reads as a miss, so a
#: library upgrade simply refits instead of loading an incompatible pickle.
_TSNE_FIT_MAGIC = b"TSF1"

#: Bounded memo for :meth:`FeatureCache.presence`. Keyed on the write epoch plus
#: the source versions and a fingerprint-set digest, so any row mutation or
#: version change is a miss; 8 entries cover the few status/projection shapes.
_PRESENCE_MEMO_MAX = 8

#: Library-wide in-memory budgets for the production FeatureCache, shared by the
#: server and the layout-evaluation CLI. The six hot projection sources decode
#: to ~112 MB; the byte budget leaves room for the excerpt/segment sources.
LIBRARY_MEMORY_ENTRIES = 1_000_000
LIBRARY_MEMORY_BYTES = 384 * 1024 * 1024


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
#: projection, clustering and coverage; ``structure`` is metadata-only and is not a
#: clustering/projection source.
_SOURCES: dict[str, _SourceSpec] = {
    "clap": _SourceSpec("clap_embedding", "version", "clap:"),
    #: Per-segment CLAP vectors (flat ``(n_segments, dim)``); shares the CLAP
    #: version column so bumping the encoder blanks both. Read back by
    #: ``/api/search`` to rank a query against the individual sections.
    "clap_segments": _SourceSpec("clap_segments", "version", "clapseg:"),
    #: Segment centre as a fraction of track duration, parallel to
    #: ``clap_segments`` rows; shares the CLAP version column.
    "clap_segment_pos": _SourceSpec("clap_segment_pos", "version", "clapsegpos:"),
    "features": _SourceSpec("features", "features_version", "feat:"),
    "maest": _SourceSpec(
        "maest_embedding",
        "maest_version",
        "maest:",
        purge_cols=("maest_embedding", "maest_logits", "maest_excerpts"),
        purge_prefixes=("maest:", "maestlog:", "maestexc:"),
        count_cols=("maest_embedding", "maest_logits"),
    ),
    "maest_logits": _SourceSpec("maest_logits", "maest_version", "maestlog:"),
    #: Per-excerpt MAEST vectors; shares the MAEST version column. Only written by
    #: ``put_maest`` and read back for weak-supervision hard-negative mining.
    "maest_excerpts": _SourceSpec("maest_excerpts", "maest_version", "maestexc:"),
    "rhythm": _SourceSpec("rhythm", "rhythm_version", "rhythm:"),
    #: Folder-supervised projection output. Versioned by training-run id, so a
    #: retrain rewrites every row under a new version and old rows read absent.
    "supervised": _SourceSpec("supervised_embedding", "supervised_version", "sup:"),
    #: Coarse track structure (intro/drop/break timestamps). Metadata, not a
    #: clustering/projection vector — kept out of ``FROZEN_SOURCES``.
    "structure": _SourceSpec("structure", "structure_version", "struct:"),
}

#: ``clap_embedding`` is NOT NULL, so writing another source to a new row pads
#: it with an empty blob.
_CLAP_PAD = b""


class FeatureCache:
    def __init__(
        self,
        db_path: str,
        max_memory: int = 2000,
        *,
        max_memory_bytes: int | None = None,
    ) -> None:
        self._db_path = db_path
        self._max_memory = max_memory
        #: Byte budget for the in-memory LRU. ``None`` keeps the historical
        #: entry-cap-only behaviour (bytes are still tracked, never evicted on).
        self._max_memory_bytes = max_memory_bytes
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_bytes = 0
        self._lock = threading.RLock()
        #: Bumps on successful writes — invalidates embedding status caches.
        self._write_epoch = 0
        #: ``presence()`` memo: key → result, bounded by ``_PRESENCE_MEMO_MAX``.
        self._presence_memo: OrderedDict[tuple, dict[str, dict[str, bool]]] = (
            OrderedDict()
        )
        self._init_db()

    def write_epoch(self) -> int:
        """Monotonic counter; increases when cached embedding rows change."""
        with self._lock:
            return self._write_epoch

    def memory_bytes(self) -> int:
        """Total bytes currently held by the in-memory LRU (read-only)."""
        with self._lock:
            return self._mem_bytes

    def clear_memory(self) -> None:
        """Empty the in-memory LRU and reset its byte counter."""
        with self._lock:
            self._mem.clear()
            self._mem_bytes = 0

    def _init_db(self) -> None:
        with ClosingConnection(sqlite3.connect(self._db_path)) as conn:
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fanout_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS fanout_journal_batch "
                "ON fanout_journal (batch_id, undone)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS origin_folders (
                    fingerprint TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS layout_cache (
                    revision   TEXT PRIMARY KEY,
                    paths      TEXT NOT NULL,
                    xy         BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tsne_fit_cache (
                    config_key TEXT PRIMARY KEY,
                    blob       BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            for col, default in [
                ("version", "INTEGER NOT NULL DEFAULT 0"),
                ("clap_segments", "BLOB"),
                ("clap_segment_pos", "BLOB"),
                ("features", "BLOB"),
                ("features_version", "INTEGER NOT NULL DEFAULT 0"),
                ("maest_embedding", "BLOB"),
                ("maest_logits", "BLOB"),
                ("maest_excerpts", "BLOB"),
                ("maest_version", "INTEGER NOT NULL DEFAULT 0"),
                ("supervised_embedding", "BLOB"),
                ("supervised_version", "INTEGER NOT NULL DEFAULT 0"),
                ("rhythm", "BLOB"),
                ("rhythm_version", "INTEGER NOT NULL DEFAULT 0"),
                ("structure", "BLOB"),
                ("structure_version", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE audio_features ADD COLUMN {col} {default}"
                    )
                except sqlite3.OperationalError:
                    pass

    def _conn(self) -> ClosingConnection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        # Writers (analysis workers + request threads) serialise on the WAL write
        # lock; wait rather than raising "database is locked" and silently
        # dropping a put(). Matches JobQueue.
        conn.execute("PRAGMA busy_timeout=30000")
        return ClosingConnection(conn)

    # ── Key/value metadata ─────────────────────────────────────
    # Used for values that are not per-track vectors but still belong to the
    # cache's identity — currently the supervised head's training-run id, which is
    # the effective cache version for the ``supervised`` source.

    def get_meta(self, key: str) -> str | None:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT value FROM feature_meta WHERE key=?", (key,)
                ).fetchone()
            return str(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def set_meta(self, key: str, value: object) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO feature_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)),
                )
        except Exception:
            return

    # ── Fan-out move journal ───────────────────────────────────
    # A folder fan-out moves files on disk, so each move is journalled before
    # it happens. The journal lets a single "Undo moves" restore the previous
    # layout, including the original names that a `` (n)`` collision suffix
    # would otherwise have changed.

    def journal_begin(self) -> int:
        """Reserve the next batch id for a fan-out apply; ``0`` on failure."""
        try:
            with self._lock:
                with self._conn() as conn:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(batch_id), 0) + 1 FROM fanout_journal"
                    ).fetchone()
                    return int(row[0]) if row else 0
        except Exception:
            return 0

    def journal_add(self, batch_id: int, source: str, target: str) -> int:
        """Record one pending move; returns its row id, or ``0`` on failure."""
        if int(batch_id) <= 0:
            return 0
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "INSERT INTO fanout_journal "
                    "(batch_id, source, target, created_at, undone) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (int(batch_id), source, target, time.time()),
                )
                return int(cursor.lastrowid or 0)
        except Exception:
            return 0

    def journal_remove(self, row_id: int) -> None:
        """Drop one pending move (it failed and must not be undone)."""
        if int(row_id) <= 0:
            return
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM fanout_journal WHERE id=?", (int(row_id),))
        except Exception:
            return

    def latest_move_batch(self) -> dict | None:
        """The most recent not-yet-undone batch, or ``None`` when there is none."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT batch_id, MAX(created_at) FROM fanout_journal "
                    "WHERE undone=0 GROUP BY batch_id "
                    "ORDER BY MAX(created_at) DESC, batch_id DESC LIMIT 1"
                ).fetchone()
                if not row:
                    return None
                batch_id = int(row[0])
                rows = conn.execute(
                    "SELECT id, source, target FROM fanout_journal "
                    "WHERE batch_id=? AND undone=0 ORDER BY id",
                    (batch_id,),
                ).fetchall()
            return {
                "batch_id": batch_id,
                "created_at": float(row[1] or 0.0),
                "moves": [
                    {"id": int(r[0]), "source": str(r[1]), "target": str(r[2])}
                    for r in rows
                ],
            }
        except Exception:
            return None

    def mark_batch_undone(self, batch_id: int) -> int:
        """Mark a whole batch undone; returns the number of rows flipped."""
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "UPDATE fanout_journal SET undone=1 WHERE batch_id=?",
                    (int(batch_id),),
                )
                return int(cursor.rowcount or 0)
        except Exception:
            return 0

    # ── Fan-out origin memory ───────────────────────────────────
    # A recursive fan-out flattens a nested tree, so a track's old relative
    # folder is lost once the move journal is pruned. This records it keyed by
    # fingerprint (content-addressed like every other row), so the original
    # layout stays inspectable after the fact.

    def record_origin(self, fingerprint: str, origin: str) -> None:
        """Remember one track's pre-fan-out relative folder (best-effort)."""
        fp = str(fingerprint or "")
        if not fp:
            return
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO origin_folders (fingerprint, origin, recorded_at) "
                    "VALUES (?, ?, ?) ON CONFLICT(fingerprint) DO UPDATE SET "
                    "origin=excluded.origin, recorded_at=excluded.recorded_at",
                    (fp, str(origin), time.time()),
                )
        except Exception:
            return

    def get_origins(self, fingerprints: list[str]) -> dict[str, str]:
        """Origin folder per fingerprint; fingerprints with no record are omitted."""
        fps = [str(f) for f in fingerprints if f]
        if not fps:
            return {}
        out: dict[str, str] = {}
        try:
            with self._conn() as conn:
                for batch, placeholders in chunked(fps):
                    rows = conn.execute(
                        "SELECT fingerprint, origin FROM origin_folders "
                        f"WHERE fingerprint IN ({placeholders})",
                        batch,
                    ).fetchall()
                    for fp, origin in rows:
                        out[str(fp)] = str(origin)
        except Exception:
            return {}
        return out

    # ── Revision-keyed layout persistence ────────────────────────
    # Layouts live only in an in-process LRU, so a server restart re-fits
    # every lens (~20 s each). The layout revision is a deterministic digest of
    # the exact configuration, so it is a stable key across restarts.

    def get_layout(self, revision: str) -> list[dict] | None:
        """Persisted positions for *revision*, or ``None``.

        The blob is row-major float32 with one row per path; its width (2 or 3)
        is inferred from the byte length, so 2-D and 3-D layouts share the table.
        """
        if not revision:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT paths, xy FROM layout_cache WHERE revision=?",
                    (revision,),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            paths = json.loads(row[0])
            blob = np.frombuffer(row[1], dtype=np.float32)
        except Exception:
            return None
        if not isinstance(paths, list):
            return None
        n = len(paths)
        if n == 0:
            return [] if blob.size == 0 else None
        if blob.size % n != 0:
            return None
        width = blob.size // n
        if width not in (2, 3):
            return None
        xy = blob.reshape(n, width)
        if width == 3:
            return [
                {
                    "path": str(p),
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "z": float(xy[i, 2]),
                }
                for i, p in enumerate(paths)
            ]
        return [
            {"path": str(p), "x": float(xy[i, 0]), "y": float(xy[i, 1])}
            for i, p in enumerate(paths)
        ]

    def _put_keyed(
        self,
        table: str,
        key_col: str,
        key: str,
        cols: tuple[str, ...],
        values: tuple,
        keep: int,
    ) -> None:
        """Upsert one keyed cache row and prune *table* to the newest *keep* rows.

        Best-effort: a cache write must never fail the request that produced it.
        """
        assignments = ", ".join(f"{c}=excluded.{c}" for c in cols)
        placeholders = ",".join("?" for _ in cols)
        try:
            with self._conn() as conn:
                conn.execute(
                    f"INSERT INTO {table} ({key_col},{','.join(cols)},created_at) "
                    f"VALUES (?,{placeholders},?) "
                    f"ON CONFLICT({key_col}) DO UPDATE SET "
                    f"{assignments}, created_at=excluded.created_at",
                    (key, *values, time.time()),
                )
                conn.execute(
                    f"DELETE FROM {table} WHERE {key_col} NOT IN ("
                    f"SELECT {key_col} FROM {table} "
                    f"ORDER BY created_at DESC LIMIT ?)",
                    (keep,),
                )
        except Exception:
            return

    def put_layout(self, revision: str, positions: list[dict]) -> None:
        """Persist *positions* under *revision* (best-effort).

        The blob is row-major float32 with width 3 when every position carries
        ``z`` (a 3-D layout) and width 2 otherwise, so the read side can infer
        the dimensionality from the byte length.
        """
        if not revision or not positions:
            return
        paths = [str(p["path"]) for p in positions]
        if all("z" in p for p in positions):
            xy = np.asarray(
                [[float(p["x"]), float(p["y"]), float(p["z"])] for p in positions],
                dtype=np.float32,
            )
        else:
            xy = np.asarray(
                [[float(p["x"]), float(p["y"])] for p in positions], dtype=np.float32
            )
        self._put_keyed(
            "layout_cache",
            "revision",
            revision,
            ("paths", "xy"),
            (json.dumps(paths), xy.tobytes()),
            _LAYOUT_KEEP,
        )

    # ── Vector-config-keyed fitted t-SNE persistence ─────────────
    # The fitted embedding is a pure function of the vector config (sources /
    # mask / blend / weights), not the track set, so it is keyed by that config
    # and a growing library can transform only its new tracks.

    def get_tsne_fit(self, config_key: str) -> bytes | None:
        """Persisted fitted t-SNE blob for *config_key*, or ``None``."""
        if not config_key:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT blob FROM tsne_fit_cache WHERE config_key=?",
                    (config_key,),
                ).fetchone()
        except Exception:
            return None
        if row is None or not row[0]:
            return None
        blob = bytes(row[0])
        if not blob.startswith(_TSNE_FIT_MAGIC):
            return None
        return blob[len(_TSNE_FIT_MAGIC) :]

    def put_tsne_fit(self, config_key: str, blob: bytes) -> None:
        """Persist a fitted t-SNE *blob* under *config_key* (best-effort)."""
        if not config_key or not blob:
            return
        self._put_keyed(
            "tsne_fit_cache",
            "config_key",
            config_key,
            ("blob",),
            (_TSNE_FIT_MAGIC + bytes(blob),),
            _TSNE_FIT_KEEP,
        )

    @staticmethod
    def _spec(source: str) -> _SourceSpec:
        try:
            return _SOURCES[source]
        except KeyError:
            raise ValueError(f"unknown source {source!r}") from None

    # ── Memory LRU ────────────────────────────────────────────
    # Keys are prefixed per source: "clap:fp", "feat:fp",
    # "maest:fp", "maestlog:fp", "rhythm:fp".

    @staticmethod
    def _nbytes(arr: np.ndarray) -> int:
        return int(arr.nbytes)

    def _lru_put(self, key: str, data: np.ndarray) -> None:
        old = self._mem.get(key)
        if old is not None:
            self._mem_bytes -= self._nbytes(old)
        self._mem[key] = data
        self._mem.move_to_end(key)
        self._mem_bytes += self._nbytes(data)
        # Evict least-recently-used entries until both the entry cap and the
        # byte budget hold; either limit may trigger. The ``len > 1`` guard
        # mirrors ``_composite_cache_put``: a single entry larger than the
        # budget must not be evicted by its own put in a loop that empties
        # the cache.
        while len(self._mem) > self._max_memory or (
            self._max_memory_bytes is not None
            and self._mem_bytes > self._max_memory_bytes
            and len(self._mem) > 1
        ):
            _k, popped = self._mem.popitem(last=False)
            self._mem_bytes -= self._nbytes(popped)

    def _evict_mem_keys_with_prefix_unlocked(self, prefix: str) -> None:
        dead = [k for k in self._mem if k.startswith(prefix)]
        for k in dead:
            self._mem_bytes -= self._nbytes(self._mem[k])
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
                    for batch, placeholders in chunked(missing):
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

    def presence(
        self, source_versions: dict[str, int], fingerprints: list[str]
    ) -> dict[str, dict[str, bool]]:
        """Which *fingerprints* have a non-empty row per source: ``{fp: True}``.

        One query with a boolean column per source. Coverage and layout revision
        only ever test ``fp in map``; decoding the blobs to answer that moved
        ~328 MB per status request. The composite gather still reads the vectors
        it needs through its own path.

        Results are memoised keyed on the write epoch, the source versions and a
        digest of the fingerprint set, so status polls and projection requests
        stop re-scanning every fingerprint. Every row mutation bumps the write
        epoch and the supervised run id is part of ``source_versions``.
        """
        out: dict[str, dict[str, bool]] = {s: {} for s in source_versions}
        if not source_versions or not fingerprints:
            return out
        digest = hashlib.sha1(
            "\n".join(sorted(fingerprints)).encode("utf-8")
        ).hexdigest()
        with self._lock:
            # Read the epoch before the query: a put racing the query can only
            # make the stored entry stale-keyed, never wrong.
            key = (
                self._write_epoch,
                tuple(sorted(source_versions.items())),
                digest,
            )
            hit = self._presence_memo.get(key)
            if hit is not None:
                self._presence_memo.move_to_end(key)
                return {s: dict(m) for s, m in hit.items()}
        specs = list(source_versions.items())
        cols = ", ".join(
            f"(length({self._spec(s).blob_col}) > 0 AND "
            f"{self._spec(s).version_col}=?)"
            for s, _v in specs
        )
        params = [int(v) for _s, v in specs]
        try:
            with self._conn() as conn:
                for batch, placeholders in chunked(fingerprints):
                    rows = conn.execute(
                        f"SELECT fingerprint, {cols} FROM audio_features "
                        f"WHERE fingerprint IN ({placeholders})",
                        [*params, *batch],
                    ).fetchall()
                    for row in rows:
                        fp = row[0]
                        for j, (s, _v) in enumerate(specs):
                            if row[j + 1]:
                                out[s][fp] = True
        except Exception:
            return out
        with self._lock:
            self._presence_memo[key] = out
            if len(self._presence_memo) > _PRESENCE_MEMO_MAX:
                self._presence_memo.popitem(last=False)
        # Callers may mutate the returned maps; hand back a copy, never the memo.
        return {s: dict(m) for s, m in out.items()}

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
        excerpts: np.ndarray | None = None,
        version: int = 0,
    ) -> None:
        """Write MAEST embedding + style logits (+ optional per-excerpt vectors).

        All three outputs come from one model pass and share ``maest_version``.
        When *excerpts* is omitted, an existing excerpt blob is left untouched
        (so a caller that only has pooled outputs cannot erase mined vectors).
        Excerpts are stored flattened; the reader reshapes by ``MAEST_EMBED_DIM``.
        """
        emb_arr = np.asarray(embedding, dtype=np.float32)
        log_arr = np.asarray(logits, dtype=np.float32)

        exc_arr: np.ndarray | None = None
        if excerpts is not None:
            exc_arr = np.asarray(excerpts, dtype=np.float32)
            if exc_arr.ndim == 1:
                exc_arr = exc_arr.reshape(1, -1)
            if exc_arr.size == 0:
                exc_arr = None

        cols = [
            "fingerprint",
            "clap_embedding",
            "maest_embedding",
            "maest_logits",
            "maest_version",
        ]
        vals: list = [
            fingerprint,
            _CLAP_PAD,
            emb_arr.tobytes(),
            log_arr.tobytes(),
            version,
        ]
        set_clause = (
            "maest_embedding=excluded.maest_embedding, "
            "maest_logits=excluded.maest_logits, "
            "maest_version=excluded.maest_version"
        )
        if exc_arr is not None:
            cols.insert(4, "maest_excerpts")
            vals.insert(4, exc_arr.tobytes())
            set_clause += ", maest_excerpts=excluded.maest_excerpts"

        sql = (
            f"INSERT INTO audio_features ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(fingerprint) DO UPDATE SET {set_clause}"
        )
        try:
            with self._conn() as conn:
                conn.execute(sql, vals)
        except Exception:
            return
        with self._lock:
            self._write_epoch += 1
            self._lru_put("maest:" + fingerprint, emb_arr.astype(np.float32))
            self._lru_put("maestlog:" + fingerprint, log_arr.astype(np.float32))
            if exc_arr is not None:
                self._lru_put("maestexc:" + fingerprint, exc_arr.reshape(-1).astype(np.float32))
