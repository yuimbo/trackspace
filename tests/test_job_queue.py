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
from tests.conftest import install_counting_conn


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


def test_stage_progress_is_scoped_per_kind_and_cleared_on_completion(q: JobQueue) -> None:
    q.note_stage_progress("cluster", phase="leiden", done=2, total=6)
    stage = q.stage_progress("cluster")
    assert stage is not None
    assert (stage["phase"], stage["done"], stage["total"]) == ("leiden", 2, 6)
    # An unrelated kind has no in-flight phase.
    assert q.stage_progress("maest") is None

    _enqueue(q, "cluster", 1)
    job = q.claim("cluster")[0]
    q.complete(job)
    # The phase describes a running job, so completion clears it.
    assert q.stage_progress("cluster") is None


def test_stage_progress_is_cleared_on_failure(q: JobQueue) -> None:
    q.note_stage_progress("supervise", phase="training", done=1, total=10)
    _enqueue(q, "supervise", 1)
    job = q.claim("supervise")[0]
    q.fail(job, "boom")
    assert q.stage_progress("supervise") is None


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


def test_automatic_requeue_does_not_revive_failed_jobs(q: JobQueue) -> None:
    """A broken file must not be retried by the automatic keep-warm enqueue.

    Only ``done`` rows are revived by default; a terminal failure is a real gap
    the user must retrigger, or every page load would re-run it forever.
    """
    one_shot = JobQueue(q._db_path, max_attempts=1)
    one_shot.enqueue_many("maest", [("broken", {})])
    job = one_shot.claim("maest")[0]
    assert one_shot.fail(job, "decode failed") is False
    assert one_shot.stats("maest").failed == 1

    one_shot.enqueue_many("maest", [("broken", {})], requeue_terminal=True)
    assert one_shot.stats("maest").pending == 0
    assert one_shot.stats("maest").failed == 1


def test_explicit_requeue_revives_failed_jobs(q: JobQueue) -> None:
    """An explicit retrigger (stage / track / force) does get a fresh budget."""
    one_shot = JobQueue(q._db_path, max_attempts=1)
    one_shot.enqueue_many("maest", [("broken", {})])
    one_shot.fail(one_shot.claim("maest")[0], "decode failed")

    one_shot.enqueue_many(
        "maest", [("broken", {})], requeue_terminal=True, requeue_failed=True
    )
    assert one_shot.stats("maest").pending == 1
    assert one_shot.stats("maest").failed == 0


def test_failed_keys_by_kind_groups_permanently_failed(q: JobQueue) -> None:
    one_shot = JobQueue(q._db_path, max_attempts=1)
    _enqueue(one_shot, "maest", 2)
    _enqueue(one_shot, "features", 1)
    one_shot.fail(one_shot.claim("maest")[0], "boom")
    one_shot.fail(one_shot.claim("features")[0], "boom")
    by_kind = one_shot.failed_keys_by_kind()
    assert by_kind["maest"] == {"k0"}
    assert by_kind["features"] == {"k0"}
    assert by_kind.get("structure") is None


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


# ── stats memo ────────────────────────────────────────────────


def test_stats_memo_avoids_repeated_sql(q: JobQueue) -> None:
    _enqueue(q, "maest", 3)
    counter = [0]
    install_counting_conn(q, counter)

    st = q.stats("maest")
    assert st.total == 3
    n0 = counter[0]
    assert n0 == 1  # one GROUP BY query

    assert q.stats("maest") == st
    assert counter[0] == n0  # memo hit

    by_kind = q.stats_by_kind()
    assert by_kind["maest"].total == 3
    n1 = counter[0]
    assert n1 == n0 + 1

    assert q.stats_by_kind() == by_kind
    assert counter[0] == n1  # memo hit

    failed = q.failed_keys_by_kind()
    assert failed == {}
    n2 = counter[0]
    assert n2 == n1 + 1

    assert q.failed_keys_by_kind() == failed
    assert counter[0] == n2  # memo hit


def test_stats_memo_invalidates_on_state_changes() -> None:
    with tempfile.TemporaryDirectory() as d:
        queue = JobQueue(os.path.join(d, "jobs.db"), lease_seconds=-1.0)
        counter = [0]
        install_counting_conn(queue, counter)

        _enqueue(queue, "maest", 3)
        base = counter[0]
        assert queue.stats("maest").pending == 3
        assert counter[0] == base + 1

        jobs = queue.claim("maest", limit=2)
        base = counter[0]
        assert queue.stats("maest").running == 2
        assert counter[0] == base + 1

        queue.complete(jobs[0])
        base = counter[0]
        st = queue.stats("maest")
        assert st.done == 1 and st.running == 1
        assert counter[0] == base + 1

        queue.fail(jobs[1], "boom", retry=False)
        base = counter[0]
        st = queue.stats("maest")
        assert st.failed == 1 and st.running == 0
        assert counter[0] == base + 1

        last = queue.claim("maest", limit=1)[0]
        base = counter[0]
        assert queue.stats("maest").running == 1
        assert counter[0] == base + 1

        assert queue.reclaim_expired() == 1  # the expired lease on *last*
        base = counter[0]
        st = queue.stats("maest")
        assert st.pending == 1 and st.running == 0
        assert counter[0] == base + 1

        queue.complete(last)  # done: jobs[0] and last
        assert queue.purge_done() == 2
        base = counter[0]
        st = queue.stats("maest")
        assert st.done == 0 and st.failed == 1 and st.pending == 0
        assert counter[0] == base + 1

        # Warm again: consecutive polls stay on the memo.
        assert queue.stats("maest").total == 1
        n = counter[0]
        assert queue.stats("maest").total == 1
        assert counter[0] == n


def test_stats_memo_ttl_serves_stale_then_recomputes(monkeypatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "backend.jobs.queue.time.monotonic", lambda: clock["t"]
    )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "jobs.db")
        ours = JobQueue(path)
        _enqueue(ours, "maest", 1)  # k0
        counter = [0]
        install_counting_conn(ours, counter)

        assert ours.stats("maest").pending == 1
        base = counter[0]

        # A second process (fresh instance, its own epoch) writes to the same DB.
        other = JobQueue(path)
        other.enqueue_many("maest", [("other-k0", {"path": "/m/o.mp3"})])

        # Within the TTL the stale memo is served...
        clock["t"] += 0.5
        assert ours.stats("maest").pending == 1
        assert counter[0] == base

        # ...and past it the query runs again and sees the new row.
        clock["t"] += 1.0
        assert ours.stats("maest").pending == 2
        assert counter[0] == base + 1

        # An in-process change invalidates immediately, TTL or not.
        _enqueue(ours, "maest", 2)  # k1 inserts; k0 conflicts
        n = counter[0]
        assert ours.stats("maest").pending == 3
        assert counter[0] == n + 1


def test_stats_memo_survives_lease_renewal(q: JobQueue) -> None:
    """renew() only extends lease_expires — invisible to the memoised stats."""
    _enqueue(q, "maest", 1)
    job = q.claim("maest")[0]
    counter = [0]
    install_counting_conn(q, counter)

    assert q.stats("maest").running == 1
    base = counter[0]

    q.renew([job.id])  # its own UPDATE is counted...
    assert q.stats("maest").running == 1
    # ...but no stats query: the memo is still warm (base + 1, not + 2).
    assert counter[0] == base + 1


def test_stats_memo_results_are_isolated_from_memo(q: JobQueue) -> None:
    one_shot = JobQueue(q._db_path, max_attempts=1)
    _enqueue(one_shot, "maest", 2)
    one_shot.fail(one_shot.claim("maest")[0], "boom")
    counter = [0]
    install_counting_conn(one_shot, counter)

    failed = one_shot.failed_keys_by_kind()
    assert failed["maest"] == {"k0"}
    failed["maest"].add("k1")  # mutate the inner set
    failed["features"] = {"k0"}  # mutate the outer dict

    by_kind = one_shot.stats_by_kind()
    by_kind.pop("maest", None)  # mutate the outer dict

    assert one_shot.failed_keys_by_kind() == {"maest": {"k0"}}
    assert "maest" in one_shot.stats_by_kind()
    assert one_shot.stats_by_kind()["maest"].total == 2


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
