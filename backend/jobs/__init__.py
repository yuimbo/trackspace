"""Durable job queue + background workers.

``queue.JobQueue`` persists work in SQLite (leases, retries, dedupe) and
``worker.JobWorker`` drains one job *kind* on a daemon thread. ``kinds`` names
the pipeline stages so enqueuers and workers cannot drift apart.
"""

from .kinds import (
    JOB_KINDS,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_SCAN,
    kind_label,
)
from .queue import DONE, FAILED, PENDING, RUNNING, Job, JobQueue, QueueStats
from .worker import JobHandler, JobResult, JobWorker, WorkerPool

__all__ = [
    "DONE",
    "FAILED",
    "JOB_KINDS",
    "KIND_CLUSTER",
    "KIND_FEATURES",
    "KIND_MAEST",
    "KIND_SCAN",
    "Job",
    "JobHandler",
    "JobQueue",
    "JobResult",
    "JobWorker",
    "PENDING",
    "RUNNING",
    "QueueStats",
    "WorkerPool",
    "kind_label",
]
