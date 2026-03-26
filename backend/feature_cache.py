"""Content-addressed cache for audio features (CLAP embeddings, EffNet
embeddings, and extracted audio features).

Keyed by chromaprint fingerprint hash so identical audio content shares
one entry regardless of filename or location.

Same two-layer (memory LRU + SQLite) pattern as TrackCache.

Each data type carries its own ``version`` integer so that bumping one
source (e.g. CLAP) does not invalidate others (e.g. EffNet).
"""

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
            for col, default in [
                ("version", "INTEGER NOT NULL DEFAULT 0"),
                ("effnet_embedding", "BLOB"),
                ("effnet_version", "INTEGER NOT NULL DEFAULT 0"),
                ("features", "BLOB"),
                ("features_version", "INTEGER NOT NULL DEFAULT 0"),
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

    # ── Memory LRU ────────────────────────────────────────────
    # Keys are prefixed: bare fp for CLAP, "effnet:fp" for EffNet,
    # "feat:fp" for audio features.

    def _lru_put(self, key: str, data: np.ndarray) -> None:
        self._mem[key] = data
        self._mem.move_to_end(key)
        if len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)

    # ------------------------------------------------------------------
    # CLAP embeddings (original API — unchanged signatures)
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
        if row is None or row[0] is None:
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
                    """INSERT INTO audio_features (fingerprint, clap_embedding, version)
                       VALUES (?,?,?)
                       ON CONFLICT(fingerprint) DO UPDATE SET clap_embedding=excluded.clap_embedding, version=excluded.version""",
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
                    "SELECT 1 FROM audio_features WHERE fingerprint=? AND version=? AND clap_embedding IS NOT NULL",
                    (fingerprint, version),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def get_all_embeddings(self, fingerprints: list[str], version: int = 0) -> dict[str, np.ndarray]:
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
                        f"WHERE fingerprint IN ({placeholders}) AND version=? AND clap_embedding IS NOT NULL",
                        [*missing, version],
                    ).fetchall()
                    for fp, blob in rows:
                        if blob is None:
                            continue
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
                        "SELECT COUNT(*) FROM audio_features WHERE version=? AND clap_embedding IS NOT NULL",
                        (version,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE clap_embedding IS NOT NULL"
                    ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def purge_old_versions(self, current_version: int) -> int:
        """NULL out CLAP embeddings with version < *current_version*."""
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "UPDATE audio_features SET clap_embedding=NULL WHERE version < ? AND clap_embedding IS NOT NULL",
                    (current_version,),
                )
                return cursor.rowcount
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # EffNet embeddings
    # ------------------------------------------------------------------

    def get_effnet_embedding(self, fingerprint: str, version: int = 0) -> np.ndarray | None:
        key = f"effnet:{fingerprint}"
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT effnet_embedding FROM audio_features WHERE fingerprint=? AND effnet_version=?",
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

    def put_effnet_embedding(self, fingerprint: str, embedding: np.ndarray, version: int = 0) -> None:
        blob = embedding.astype(np.float32).tobytes()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO audio_features (fingerprint, clap_embedding, effnet_embedding, effnet_version)
                       VALUES (?, X'', ?, ?)
                       ON CONFLICT(fingerprint) DO UPDATE SET effnet_embedding=excluded.effnet_embedding, effnet_version=excluded.effnet_version""",
                    (fingerprint, blob, version),
                )
        except Exception:
            pass
        with self._lock:
            self._lru_put(f"effnet:{fingerprint}", embedding.astype(np.float32))

    def has_effnet_embedding(self, fingerprint: str, version: int = 0) -> bool:
        key = f"effnet:{fingerprint}"
        with self._lock:
            if key in self._mem:
                return True
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM audio_features WHERE fingerprint=? AND effnet_version=? AND effnet_embedding IS NOT NULL",
                    (fingerprint, version),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def get_all_effnet_embeddings(self, fingerprints: list[str], version: int = 0) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._lock:
            for fp in fingerprints:
                key = f"effnet:{fp}"
                if key in self._mem:
                    self._mem.move_to_end(key)
                    result[fp] = self._mem[key]
                else:
                    missing.append(fp)
        if missing:
            try:
                with self._conn() as conn:
                    placeholders = ",".join("?" for _ in missing)
                    rows = conn.execute(
                        f"SELECT fingerprint, effnet_embedding FROM audio_features "
                        f"WHERE fingerprint IN ({placeholders}) AND effnet_version=? AND effnet_embedding IS NOT NULL",
                        [*missing, version],
                    ).fetchall()
                    for fp, blob in rows:
                        if blob is None:
                            continue
                        arr = np.frombuffer(blob, dtype=np.float32).copy()
                        result[fp] = arr
                        with self._lock:
                            self._lru_put(f"effnet:{fp}", arr)
            except Exception:
                pass
        return result

    def count_effnet(self, version: int | None = None) -> int:
        try:
            with self._conn() as conn:
                if version is not None:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE effnet_version=? AND effnet_embedding IS NOT NULL",
                        (version,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE effnet_embedding IS NOT NULL"
                    ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def purge_old_effnet_versions(self, current_version: int) -> int:
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "UPDATE audio_features SET effnet_embedding=NULL WHERE effnet_version < ? AND effnet_embedding IS NOT NULL",
                    (current_version,),
                )
                return cursor.rowcount
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Audio features (tempo, key, energy, danceability)
    # ------------------------------------------------------------------

    def get_audio_features(self, fingerprint: str, version: int = 0) -> np.ndarray | None:
        key = f"feat:{fingerprint}"
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT features FROM audio_features WHERE fingerprint=? AND features_version=?",
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

    def put_audio_features(self, fingerprint: str, features: np.ndarray, version: int = 0) -> None:
        blob = features.astype(np.float32).tobytes()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO audio_features (fingerprint, clap_embedding, features, features_version)
                       VALUES (?, X'', ?, ?)
                       ON CONFLICT(fingerprint) DO UPDATE SET features=excluded.features, features_version=excluded.features_version""",
                    (fingerprint, blob, version),
                )
        except Exception:
            pass
        with self._lock:
            self._lru_put(f"feat:{fingerprint}", features.astype(np.float32))

    def has_audio_features(self, fingerprint: str, version: int = 0) -> bool:
        key = f"feat:{fingerprint}"
        with self._lock:
            if key in self._mem:
                return True
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM audio_features WHERE fingerprint=? AND features_version=? AND features IS NOT NULL",
                    (fingerprint, version),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def get_all_audio_features(self, fingerprints: list[str], version: int = 0) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._lock:
            for fp in fingerprints:
                key = f"feat:{fp}"
                if key in self._mem:
                    self._mem.move_to_end(key)
                    result[fp] = self._mem[key]
                else:
                    missing.append(fp)
        if missing:
            try:
                with self._conn() as conn:
                    placeholders = ",".join("?" for _ in missing)
                    rows = conn.execute(
                        f"SELECT fingerprint, features FROM audio_features "
                        f"WHERE fingerprint IN ({placeholders}) AND features_version=? AND features IS NOT NULL",
                        [*missing, version],
                    ).fetchall()
                    for fp, blob in rows:
                        if blob is None:
                            continue
                        arr = np.frombuffer(blob, dtype=np.float32).copy()
                        result[fp] = arr
                        with self._lock:
                            self._lru_put(f"feat:{fp}", arr)
            except Exception:
                pass
        return result

    def count_features(self, version: int | None = None) -> int:
        try:
            with self._conn() as conn:
                if version is not None:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE features_version=? AND features IS NOT NULL",
                        (version,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM audio_features WHERE features IS NOT NULL"
                    ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def purge_old_feature_versions(self, current_version: int) -> int:
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "UPDATE audio_features SET features=NULL WHERE features_version < ? AND features IS NOT NULL",
                    (current_version,),
                )
                return cursor.rowcount
        except Exception:
            return 0
