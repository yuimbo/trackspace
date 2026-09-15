"""Durability semantics of the analysis job queue.

These tests exist because the queue's whole purpose is surviving things that
are awkward to reproduce by hand: a killed process, a wedged worker, a burst of
duplicate enqueues. Each test pins one of those behaviours.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from backend.jobs import DONE, FAILED, PENDING, JobQueue
from backend.jobs.worker import JobResult, JobWorker


@pytest.fixture()
def q() -> JobQueue:
    with tempfile.TemporaryDirectory() as d:
        yield JobQueue(os.path.join(d, "jobs.db"))


def _enqueue(q: JobQueue, kind: str, n: int, **kw) -> int:
    return q.enqueue_many(
        kind, [(f"k{i}", {"path": f"/m/{i}.mp3"}) for i in range(n)], **kw
    )


def test_enqueue_is_idempotent(q: JobQueue) -> None:
    assert _enqueue(q, "maest", 5) == 5
    # Re-enqueueing the same keys must not duplicate scheduled work.
    assert _enqueue(q, "maest", 5) == 0
    assert q.stats("maest").total == 5


def test_claim_leases_jobs_exclusively(q: JobQueue) -> None:
    _enqueue(q, "maest", 4)
    first = q.claim("maest", limit=2)
    second = q.claim("maest", limit=2)
    assert len(first) == 2 and len(second) == 2
    # No job may be handed to two workers.
    assert {j.id for j in first}.isdisjoint({j.id for j in second})
    assert q.claim("maest", limit=2) == []


def test_completion_is_durable_across_instances(q: JobQueue) -> None:
    _enqueue(q, "maest", 3)
    job = q.claim("maest")[0]
    q.complete(job)

    # A fresh JobQueue over the same file is the "after restart" case.
    reopened = JobQueue(q._db_path)
    assert reopened.stats("maest").done == 1
    assert reopened.stats("maest").pending == 2


def test_expired_lease_is_reclaimed(q: JobQueue) -> None:
    fast = JobQueue(q._db_path, lease_seconds=-1.0)  # already expired on claim
    _enqueue(fast, "maest", 2)
    claimed = fast.claim("maest", limit=2)
    assert len(claimed) == 2
    assert fast.stats("maest").running == 2

    assert fast.reclaim_expired() == 2
    # Work returns to the queue rather than being stranded in 'running'.
    assert fast.stats("maest").running == 0
    assert fast.stats("maest").pending == 2


def test_retry_budget_then_permanent_failure(q: JobQueue) -> None:
    budget = JobQueue(q._db_path, max_attempts=3)
    budget.enqueue_many("maest", [("a", {"path": "/m/a.mp3"})])

    for expected_attempt in (1, 2):
        job = budget.claim("maest")[0]
        assert job.attempts == expected_attempt
        assert budget.fail(job, "transient") is True

    job = budget.claim("maest")[0]
    assert budget.fail(job, "still broken") is False
    assert budget.stats("maest").failed == 1


def test_permanent_error_skips_remaining_attempts(q: JobQueue) -> None:
    q.enqueue_many("maest", [("gone", {"path": "/m/gone.mp3"})])
    job = q.claim("maest")[0]
    # A missing file will never succeed; do not burn the whole budget on it.
    assert q.fail(job, "file not found", retry=False) is False
    assert q.stats("maest").failed == 1


def test_retry_failed_restores_pending(q: JobQueue) -> None:
    q.enqueue_many("maest", [("a", {})])
    q.fail(q.claim("maest")[0], "boom", retry=False)
    assert q.retry_failed("maest") == 1
    assert q.stats("maest").pending == 1


def test_priority_orders_claims(q: JobQueue) -> None:
    _enqueue(q, "maest", 5)
    q.prioritize("maest", ["k4"], priority=0)
    # The prioritised key must come out first even though it was enqueued last.
    assert q.claim("maest")[0].dedupe_key == "k4"


def test_force_requeue_revives_terminal_jobs(q: JobQueue) -> None:
    q.enqueue_many("maest", [("a", {})])
    q.complete(q.claim("maest")[0])
    assert q.stats("maest").done == 1

    q.enqueue_many("maest", [("a", {})], requeue_terminal=True)
    assert q.stats("maest").pending == 1
    assert q.stats("maest").done == 0


def test_stats_by_kind_separates_pipelines(q: JobQueue) -> None:
    _enqueue(q, "maest", 3)
    _enqueue(q, "features", 2)
    by_kind = q.stats_by_kind()
    assert by_kind["maest"].total == 3
    assert by_kind["features"].total == 2


def test_epoch_advances_on_state_change(q: JobQueue) -> None:
    before = q.epoch()
    _enqueue(q, "maest", 1)
    assert q.epoch() > before


def test_worker_drains_queue_and_records_results() -> None:
    with tempfile.TemporaryDirectory() as d:
        queue = JobQueue(os.path.join(d, "jobs.db"))
        queue.enqueue_many(
            "maest", [(f"k{i}", {"path": f"/m/{i}.mp3"}) for i in range(4)]
        )

        seen: list[str] = []

        def handler(jobs):
            out = {}
            for j in jobs:
                seen.append(j.dedupe_key)
                # Make one job fail permanently to cover both paths.
                if j.dedupe_key == "k2":
                    out[j.id] = JobResult.failure("bad audio", retry=False)
                else:
                    out[j.id] = JobResult.success(excerpts=3)
            return out

        events: list[dict] = []
        worker = JobWorker(
            queue, "maest", handler, batch_size=2, on_event=events.append
        )
        # Drive the loop directly instead of starting a thread — deterministic.
        for _ in range(4):
            worker._tick()

        stats = queue.stats("maest")
        assert stats.done == 3
        assert stats.failed == 1
        assert stats.outstanding == 0
        assert sorted(seen) == ["k0", "k1", "k2", "k3"]
        assert any(e.get("type") == "drained" for e in events)


def test_worker_gate_defers_claiming() -> None:
    with tempfile.TemporaryDirectory() as d:
        queue = JobQueue(os.path.join(d, "jobs.db"))
        queue.enqueue_many("maest", [("a", {"path": "/m/a.mp3"})])

        ready = False
        calls: list[int] = []

        def handler(jobs):
            calls.append(len(jobs))
            return {j.id: JobResult.success() for j in jobs}

        worker = JobWorker(
            queue, "maest", handler, idle_sleep=0.01, gate=lambda: ready
        )
        worker._tick()
        # Model not ready: nothing claimed, job still queued for later.
        assert calls == []
        assert queue.stats("maest").pending == 1

        ready = True
        worker._tick()
        assert calls == [1]
        assert queue.stats("maest").done == 1


def test_handler_exception_fails_whole_batch_without_losing_jobs() -> None:
    with tempfile.TemporaryDirectory() as d:
        queue = JobQueue(os.path.join(d, "jobs.db"), max_attempts=1)
        queue.enqueue_many("maest", [("a", {}), ("b", {})])

        def handler(jobs):
            raise RuntimeError("model exploded")

        worker = JobWorker(queue, "maest", handler, batch_size=2)
        worker._tick()

        stats = queue.stats("maest")
        # Jobs are accounted for, not silently dropped.
        assert stats.failed == 2
        assert stats.total == 2
        assert queue.failures("maest")[0]["error"].startswith("RuntimeError")
