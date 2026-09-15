"""Durable SQLite-backed job queue: leases, retries, resumability.

Design constraints that shaped this module:

* **Crash-safe.** State lives in SQLite, not in a Python dict. A server killed
  mid-batch resumes exactly where it stopped — ``running`` jobs whose lease has
  expired are reclaimed as ``pending`` by :meth:`JobQueue.reclaim_expired`.
* **Idempotent enqueue.** ``(kind, dedupe_key)`` is UNIQUE, so re-enqueuing the
  same unit of work while it is pending/running is a no-op instead of a
  duplicate. This is what makes "rescan on every filesystem event" safe.
* **Leases, not locks.** A worker claims a job for ``lease_seconds``; if it dies
  the lease lapses and another worker picks the job up. No global "running"
  flag can get stuck ``True`` after a crash.
* **Counts are SQL, not counters.** ``stats()`` aggregates the table, so the
  numbers the UI shows cannot drift away from reality.

Terminal states are ``done`` and ``failed``; ``failed`` is reached only after
``max_attempts`` attempts so transient decode/model errors self-heal.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: States a job can be reset from when the caller wants a forced re-run.
RESETTABLE = (DONE, FAILED)

DEFAULT_LEASE_SECONDS = 300.0
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Job:
    """One unit of work claimed from the queue."""

    id: int
    kind: str
    dedupe_key: str
    payload: dict[str, Any]
    attempts: int
    priority: int

    @property
    def path(self) -> str | None:
        """Convenience accessor — most Trackspace jobs are per-track."""
        p = self.payload.get("path")
        return p if isinstance(p, str) else None


@dataclass(frozen=True)
class QueueStats:
    """Aggregate snapshot of one job kind (or the whole queue)."""

    pending: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.running + self.done + self.failed

    @property
    def finished(self) -> int:
        """Jobs that will not be retried — the numerator for progress bars."""
        return self.done + self.failed

    @property
    def outstanding(self) -> int:
        return self.pending + self.running

    def as_dict(self) -> dict[str, int]:
        return {
            "pending": self.pending,
            "running": self.running,
            "done": self.done,
            "failed": self.failed,
            "total": self.total,
            "finished": self.finished,
            "outstanding": self.outstanding,
        }


@dataclass
class _Progress:
    """Last-completed breadcrumb per kind, for human-facing status text."""

    path: str | None = None
    ok: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = 0.0


class JobQueue:
    """Durable work queue backed by one SQLite table.

    Thread-safe. All mutating statements run inside ``IMMEDIATE`` transactions
    so two workers can never claim the same job.
    """

    def __init__(
        self,
        db_path: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._db_path = db_path
        self._lease_seconds = float(lease_seconds)
        self._max_attempts = int(max_attempts)
        self._lock = threading.RLock()
        self._worker_id = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        #: Bumps on every state transition so status caches can invalidate.
        self._epoch = 0
        self._progress: dict[str, _Progress] = {}
        self._init_db()

    # ── plumbing ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind         TEXT    NOT NULL,
                    dedupe_key   TEXT    NOT NULL,
                    payload      TEXT    NOT NULL DEFAULT '{}',
                    state        TEXT    NOT NULL DEFAULT 'pending',
                    priority     INTEGER NOT NULL DEFAULT 100,
                    attempts     INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_owner  TEXT,
                    lease_expires REAL,
                    last_error   TEXT,
                    created_at   REAL    NOT NULL,
                    updated_at   REAL    NOT NULL,
                    UNIQUE (kind, dedupe_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_claim "
                "ON jobs (kind, state, priority, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs (state, lease_expires)"
            )

    def _bump_epoch(self) -> None:
        with self._lock:
            self._epoch += 1

    def epoch(self) -> int:
        """Monotonic counter; changes whenever any job changes state."""
        with self._lock:
            return self._epoch

    # ── enqueue ───────────────────────────────────────────────

    def enqueue_many(
        self,
        kind: str,
        items: Iterable[tuple[str, dict[str, Any]]],
        *,
        priority: int = 100,
        max_attempts: int | None = None,
        requeue_terminal: bool = False,
    ) -> int:
        """Insert ``(dedupe_key, payload)`` pairs; return the number newly queued.

        Existing *pending* or *running* rows are left untouched — the work is
        already scheduled. Rows in a terminal state are revived only when
        *requeue_terminal* is set (used by "force re-analyse"), which also
        resets ``attempts`` so a previously exhausted job gets a clean budget.
        """
        now = time.time()
        attempts_cap = self._max_attempts if max_attempts is None else int(max_attempts)
        rows = [
            (kind, key, json.dumps(payload or {}), priority, attempts_cap, now, now)
            for key, payload in items
        ]
        if not rows:
            return 0

        inserted = 0
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.executemany(
                """
                INSERT INTO jobs
                    (kind, dedupe_key, payload, priority, max_attempts, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT (kind, dedupe_key) DO NOTHING
                """,
                rows,
            )
            inserted = cur.rowcount or 0

            if requeue_terminal:
                keys = [key for key, _ in items] if isinstance(items, list) else [r[1] for r in rows]
                revived = self._reset_keys(conn, kind, keys, now, priority, attempts_cap)
                inserted += revived

        if inserted:
            self._bump_epoch()
        return inserted

    def _reset_keys(
        self,
        conn: sqlite3.Connection,
        kind: str,
        keys: Sequence[str],
        now: float,
        priority: int,
        attempts_cap: int,
    ) -> int:
        """Move terminal rows back to pending. Caller holds the transaction."""
        total = 0
        for chunk_start in range(0, len(keys), 500):
            chunk = keys[chunk_start : chunk_start + 500]
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(
                f"""
                UPDATE jobs
                   SET state='pending', attempts=0, last_error=NULL,
                       lease_owner=NULL, lease_expires=NULL,
                       priority=?, max_attempts=?, updated_at=?
                 WHERE kind=? AND dedupe_key IN ({placeholders})
                   AND state IN ('done','failed')
                """,
                [priority, attempts_cap, now, kind, *chunk],
            )
            total += cur.rowcount or 0
        return total

    def prioritize(self, kind: str, dedupe_keys: Sequence[str], *, priority: int = 0) -> int:
        """Raise priority of still-pending jobs (lower number runs first).

        Used so tracks currently visible on screen are analysed first without
        disturbing queue membership.
        """
        if not dedupe_keys:
            return 0
        now = time.time()
        changed = 0
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for chunk_start in range(0, len(dedupe_keys), 500):
                chunk = list(dedupe_keys[chunk_start : chunk_start + 500])
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"""
                    UPDATE jobs SET priority=?, updated_at=?
                     WHERE kind=? AND state='pending' AND priority > ?
                       AND dedupe_key IN ({placeholders})
                    """,
                    [priority, now, kind, priority, *chunk],
                )
                changed += cur.rowcount or 0
        if changed:
            self._bump_epoch()
        return changed

    # ── claim / complete ──────────────────────────────────────

    def claim(self, kind: str, limit: int = 1) -> list[Job]:
        """Atomically lease up to *limit* pending jobs of *kind*."""
        if limit <= 0:
            return []
        now = time.time()
        expires = now + self._lease_seconds
        claimed: list[Job] = []

        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id, kind, dedupe_key, payload, attempts, priority
                  FROM jobs
                 WHERE kind=? AND state='pending'
                 ORDER BY priority ASC, id ASC
                 LIMIT ?
                """,
                (kind, limit),
            ).fetchall()
            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""
                    UPDATE jobs
                       SET state='running', attempts=attempts+1,
                           lease_owner=?, lease_expires=?, updated_at=?
                     WHERE id IN ({placeholders})
                    """,
                    [self._worker_id, expires, now, *ids],
                )
                for r in rows:
                    try:
                        payload = json.loads(r[3])
                    except (TypeError, ValueError):
                        payload = {}
                    claimed.append(
                        Job(
                            id=r[0],
                            kind=r[1],
                            dedupe_key=r[2],
                            payload=payload if isinstance(payload, dict) else {},
                            attempts=r[4] + 1,
                            priority=r[5],
                        )
                    )
        if claimed:
            self._bump_epoch()
        return claimed

    def renew(self, job_ids: Sequence[int]) -> None:
        """Extend leases for long-running work so it is not reclaimed mid-flight."""
        if not job_ids:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in job_ids)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE jobs SET lease_expires=?, updated_at=? "
                f"WHERE id IN ({placeholders}) AND state='running'",
                [now + self._lease_seconds, now, *job_ids],
            )

    def complete(
        self,
        job: Job | int,
        *,
        detail: dict[str, Any] | None = None,
        path: str | None = None,
    ) -> None:
        """Mark a job done."""
        job_id = job.id if isinstance(job, Job) else int(job)
        kind = job.kind if isinstance(job, Job) else None
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET state='done', lease_owner=NULL, lease_expires=NULL, "
                "last_error=NULL, updated_at=? WHERE id=?",
                (now, job_id),
            )
        if kind:
            self._note_progress(
                kind,
                path=path or (job.path if isinstance(job, Job) else None),
                ok=True,
                detail=detail or {},
            )
        self._bump_epoch()

    def fail(
        self,
        job: Job,
        error: str,
        *,
        detail: dict[str, Any] | None = None,
        retry: bool = True,
    ) -> bool:
        """Record a failure. Returns ``True`` if the job will be retried.

        A job goes to ``failed`` once its attempts reach ``max_attempts`` (or
        immediately when *retry* is ``False``, for errors that cannot heal —
        e.g. a file that no longer exists).
        """
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id=?", (job.id,)
            ).fetchone()
            attempts, cap = (row[0], row[1]) if row else (job.attempts, self._max_attempts)
            will_retry = bool(retry) and attempts < cap
            conn.execute(
                "UPDATE jobs SET state=?, lease_owner=NULL, lease_expires=NULL, "
                "last_error=?, updated_at=? WHERE id=?",
                (PENDING if will_retry else FAILED, error[:2000], now, job.id),
            )
        self._note_progress(
            job.kind, path=job.path, ok=False, detail={**(detail or {}), "error": error}
        )
        self._bump_epoch()
        return will_retry

    def reclaim_expired(self) -> int:
        """Return timed-out ``running`` jobs to ``pending``.

        Called at startup (recovering from a crash) and periodically by the
        worker loop (recovering from a wedged task).
        """
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                UPDATE jobs
                   SET state=CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                       lease_owner=NULL, lease_expires=NULL,
                       last_error=COALESCE(last_error, 'lease expired'),
                       updated_at=?
                 WHERE state='running' AND lease_expires IS NOT NULL AND lease_expires < ?
                """,
                (now, now),
            )
            n = cur.rowcount or 0
        if n:
            log.info("JobQueue: reclaimed %d expired job(s)", n)
            self._bump_epoch()
        return n

    # ── introspection ─────────────────────────────────────────

    def stats(self, kind: str | None = None) -> QueueStats:
        try:
            with self._conn() as conn:
                if kind is None:
                    rows = conn.execute(
                        "SELECT state, COUNT(*) FROM jobs GROUP BY state"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT state, COUNT(*) FROM jobs WHERE kind=? GROUP BY state",
                        (kind,),
                    ).fetchall()
        except Exception:
            return QueueStats()
        counts = {state: int(n) for state, n in rows}
        return QueueStats(
            pending=counts.get(PENDING, 0),
            running=counts.get(RUNNING, 0),
            done=counts.get(DONE, 0),
            failed=counts.get(FAILED, 0),
        )

    def stats_by_kind(self) -> dict[str, QueueStats]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT kind, state, COUNT(*) FROM jobs GROUP BY kind, state"
                ).fetchall()
        except Exception:
            return {}
        acc: dict[str, dict[str, int]] = {}
        for kind, state, n in rows:
            acc.setdefault(kind, {})[state] = int(n)
        return {
            kind: QueueStats(
                pending=c.get(PENDING, 0),
                running=c.get(RUNNING, 0),
                done=c.get(DONE, 0),
                failed=c.get(FAILED, 0),
            )
            for kind, c in acc.items()
        }

    def failures(self, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Recent permanently-failed jobs, for surfacing in the UI."""
        try:
            with self._conn() as conn:
                if kind is None:
                    rows = conn.execute(
                        "SELECT kind, dedupe_key, attempts, last_error, updated_at "
                        "FROM jobs WHERE state='failed' ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT kind, dedupe_key, attempts, last_error, updated_at "
                        "FROM jobs WHERE state='failed' AND kind=? "
                        "ORDER BY updated_at DESC LIMIT ?",
                        (kind, limit),
                    ).fetchall()
        except Exception:
            return []
        return [
            {
                "kind": r[0],
                "key": r[1],
                "attempts": r[2],
                "error": r[3],
                "at": r[4],
            }
            for r in rows
        ]

    def _note_progress(
        self, kind: str, *, path: str | None, ok: bool, detail: dict[str, Any]
    ) -> None:
        with self._lock:
            self._progress[kind] = _Progress(
                path=path, ok=ok, detail=detail, at=time.time()
            )

    def progress(self, kind: str) -> dict[str, Any] | None:
        with self._lock:
            p = self._progress.get(kind)
        if p is None:
            return None
        return {"path": p.path, "ok": p.ok, "detail": p.detail, "at": p.at}

    # ── maintenance ───────────────────────────────────────────

    def retry_failed(self, kind: str | None = None) -> int:
        """Reset ``failed`` jobs to ``pending`` with a fresh attempt budget."""
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if kind is None:
                cur = conn.execute(
                    "UPDATE jobs SET state='pending', attempts=0, last_error=NULL, "
                    "updated_at=? WHERE state='failed'",
                    (now,),
                )
            else:
                cur = conn.execute(
                    "UPDATE jobs SET state='pending', attempts=0, last_error=NULL, "
                    "updated_at=? WHERE state='failed' AND kind=?",
                    (now, kind),
                )
            n = cur.rowcount or 0
        if n:
            self._bump_epoch()
        return n

    def purge_done(self, kind: str | None = None, older_than: float = 0.0) -> int:
        """Delete completed rows. Keeps the table small over long sessions."""
        cutoff = time.time() - max(0.0, older_than)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if kind is None:
                cur = conn.execute(
                    "DELETE FROM jobs WHERE state='done' AND updated_at < ?", (cutoff,)
                )
            else:
                cur = conn.execute(
                    "DELETE FROM jobs WHERE state='done' AND kind=? AND updated_at < ?",
                    (kind, cutoff),
                )
            n = cur.rowcount or 0
        if n:
            self._bump_epoch()
        return n

    def drop_kind(self, kind: str) -> int:
        """Remove every job of one kind (e.g. after a version bump)."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute("DELETE FROM jobs WHERE kind=?", (kind,))
            n = cur.rowcount or 0
        if n:
            self._bump_epoch()
        return n
