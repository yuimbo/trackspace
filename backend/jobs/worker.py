"""Background worker that drains one :class:`~backend.jobs.queue.JobQueue` kind.

One worker thread per job *kind*, so a slow GPU stage (MAEST) never blocks a
cheap CPU stage (tag scan). The worker owns the whole lifecycle:

    claim(batch) → handler(jobs) → complete()/fail() → publish event

Handlers receive a *list* of jobs so GPU stages can batch (MAEST/EffNet want
batched inference), and report per-job outcomes back. A handler that raises is
treated as "every job in the batch failed" — the queue's retry budget then
decides whether they come back.

The loop is idle-cheap: when the queue is empty it sleeps on a
:class:`threading.Event` that ``wake()`` sets, so newly enqueued work starts
immediately without polling latency.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .queue import Job, JobQueue

log = logging.getLogger(__name__)

#: ``handler(jobs) -> {job_id: JobResult}``
JobHandler = Callable[[Sequence[Job]], "dict[int, JobResult]"]


@dataclass
class JobResult:
    """Outcome of one job as reported by a handler."""

    ok: bool
    detail: dict[str, Any] | None = None
    error: str | None = None
    #: ``False`` for permanent errors (missing file, unsupported codec) so the
    #: queue stops retrying immediately instead of burning the attempt budget.
    retry: bool = True

    @classmethod
    def success(cls, **detail: Any) -> JobResult:
        return cls(ok=True, detail=detail or None)

    @classmethod
    def failure(cls, error: str, *, retry: bool = True, **detail: Any) -> JobResult:
        return cls(ok=False, error=error, detail=detail or None, retry=retry)


class JobWorker:
    """Drains jobs of one *kind* on a daemon thread."""

    def __init__(
        self,
        queue: JobQueue,
        kind: str,
        handler: JobHandler,
        *,
        batch_size: int = 1,
        idle_sleep: float = 1.0,
        reclaim_interval: float = 60.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        gate: Callable[[], bool] | None = None,
    ) -> None:
        self._q = queue
        self._kind = kind
        self._handler = handler
        self._batch_size = max(1, int(batch_size))
        self._idle_sleep = float(idle_sleep)
        self._reclaim_interval = float(reclaim_interval)
        self._on_event = on_event
        #: Optional readiness predicate — worker idles until it returns True
        #: (used to wait for ML models to finish loading).
        self._gate = gate

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active = 0
        self._lock = threading.Lock()
        self._last_reclaim = 0.0

    @property
    def kind(self) -> str:
        return self._kind

    def is_busy(self) -> bool:
        with self._lock:
            return self._active > 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"trackspace-worker-{self._kind}"
        )
        self._thread.start()
        log.info("JobWorker[%s]: started", self._kind)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Signal that new work was enqueued."""
        self._wake.set()

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({"kind": self._kind, **event})
        except Exception:
            log.exception("JobWorker[%s]: event publish failed", self._kind)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("JobWorker[%s]: unexpected loop error", self._kind)
                time.sleep(self._idle_sleep)

    def _tick(self) -> None:
        now = time.time()
        if now - self._last_reclaim > self._reclaim_interval:
            self._last_reclaim = now
            self._q.reclaim_expired()

        if self._gate is not None and not self._gate():
            self._wake.wait(timeout=self._idle_sleep)
            self._wake.clear()
            return

        jobs = self._q.claim(self._kind, limit=self._batch_size)
        if not jobs:
            # Sleep until woken by an enqueue or the idle timeout elapses.
            self._wake.wait(timeout=self._idle_sleep)
            self._wake.clear()
            return

        with self._lock:
            self._active = len(jobs)
        try:
            self._run_batch(jobs)
        finally:
            with self._lock:
                self._active = 0

    def _run_batch(self, jobs: list[Job]) -> None:
        started = time.time()
        try:
            results = self._handler(jobs) or {}
        except Exception as e:
            log.exception("JobWorker[%s]: handler raised", self._kind)
            results = {
                j.id: JobResult.failure(f"{type(e).__name__}: {e}") for j in jobs
            }

        # Renew leases if the handler ran long — protects the *next* batch's
        # bookkeeping from a reclaim sweep racing our completion writes.
        if time.time() - started > 0.5 * DEFAULT_RENEW_FRACTION_GUARD:
            self._q.renew([j.id for j in jobs])

        for job in jobs:
            res = results.get(job.id)
            if res is None:
                res = JobResult.failure("handler returned no result for job")

            if res.ok:
                self._q.complete(job, detail=res.detail, path=job.path)
                self._emit(
                    {
                        "type": "progress",
                        "path": job.path,
                        "key": job.dedupe_key,
                        "ok": True,
                        **(res.detail or {}),
                    }
                )
            else:
                will_retry = self._q.fail(
                    job, res.error or "unknown error", detail=res.detail, retry=res.retry
                )
                self._emit(
                    {
                        "type": "progress",
                        "path": job.path,
                        "key": job.dedupe_key,
                        "ok": False,
                        "error": res.error,
                        "will_retry": will_retry,
                        "attempts": job.attempts,
                        **(res.detail or {}),
                    }
                )

        stats = self._q.stats(self._kind)
        self._emit({"type": "stats", **stats.as_dict()})
        if stats.outstanding == 0:
            self._emit({"type": "drained", **stats.as_dict()})


#: Leases are renewed when a batch takes longer than this many seconds, well
#: under the default 300 s lease so a slow batch is never reclaimed underneath us.
DEFAULT_RENEW_FRACTION_GUARD = 60.0


class WorkerPool:
    """Owns the worker threads for every job kind in the process."""

    def __init__(self) -> None:
        self._workers: dict[str, JobWorker] = {}

    def add(self, worker: JobWorker) -> JobWorker:
        self._workers[worker.kind] = worker
        return worker

    def get(self, kind: str) -> JobWorker | None:
        return self._workers.get(kind)

    def start_all(self) -> None:
        for w in self._workers.values():
            w.start()

    def stop_all(self) -> None:
        for w in self._workers.values():
            w.stop()

    def wake(self, kind: str | None = None) -> None:
        if kind is None:
            for w in self._workers.values():
                w.wake()
        elif kind in self._workers:
            self._workers[kind].wake()

    def any_busy(self) -> bool:
        return any(w.is_busy() for w in self._workers.values())

    def kinds(self) -> list[str]:
        return sorted(self._workers)
