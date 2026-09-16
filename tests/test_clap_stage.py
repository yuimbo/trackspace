"""CLAP job stage: handler cache contract, failure mapping, and enqueue filter.

Uses a fake ``clap_mod`` so no model is loaded; the point is the handler/queue
contract (cache short-circuit, permanent vs retryable failures, disabled servers
never queue CLAP work), not CLAP inference quality.
"""

from __future__ import annotations

import numpy as np

from backend.embeddings.feature_cache import FeatureCache
from backend.jobs import JOB_KINDS, KIND_CLAP, KIND_MAEST, Job, JobQueue, kind_label
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_library_work,
    make_clap_handler,
)

CLAP_DIM = 8


class _FakeClap:
    NUM_SEGMENTS = 3

    def __init__(self, *, ready: bool = True, error: str | None = None):
        self._ready = ready
        self._error = error
        self.calls: list[list[str]] = []

    def is_model_ready(self) -> bool:
        return self._ready

    def load_error(self) -> str | None:
        return self._error

    def load_model(self) -> None:
        return None

    def analyze_batch(self, paths, *, on_decode_warning=None, max_batch=12):
        self.calls.append(list(paths))
        return {p: np.full(CLAP_DIM, len(p), np.float32) for p in paths}


def _deps(tmp_path, fc: FeatureCache, *, clap=None, clap_enabled: bool = True) -> AnalysisDeps:
    track = tmp_path / "track.mp3"
    track.write_bytes(b"ID3\x00fake")
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda _v: str(track),
        virtual_from_abs=lambda _a: "track.mp3",
        list_mp3s=lambda _f, _r: [str(track)],
        maest_mod=None,
        clap_mod=clap or _FakeClap(),
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        clap_version=1,
        clap_enabled=clap_enabled,
        data_dir=str(tmp_path),
    )


def _clap_job(job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind=KIND_CLAP,
        dedupe_key="track.mp3",
        payload={"path": "track.mp3"},
        attempts=0,
        priority=110,
    )


# ── registry ─────────────────────────────────────────────────


def test_clap_kind_is_registered_and_labelled():
    assert KIND_CLAP in JOB_KINDS
    assert kind_label(KIND_CLAP) == "CLAP analysis"


# ── handler ──────────────────────────────────────────────────


def test_clap_handler_writes_embedding_then_serves_cache(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeClap()
    handler = make_clap_handler(_deps(tmp_path, fc, clap=fake))

    out = handler([_clap_job()])
    assert out[1].ok is True
    assert fc.has("clap", "fp", version=1)
    assert fc.get("clap", "fp", version=1).shape == (CLAP_DIM,)

    # Second run is served from the cache — no model call.
    again = handler([_clap_job(2)])
    assert again[2].ok is True
    assert again[2].detail.get("cached") is True
    assert len(fake.calls) == 1


def test_clap_handler_batches_within_one_model_call(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeClap()
    handler = make_clap_handler(_deps(tmp_path, fc, clap=fake))

    handler([_clap_job(1), _clap_job(2)])
    assert len(fake.calls) == 1
    assert sorted(fake.calls[0]) == sorted([str(tmp_path / "track.mp3")] * 2)


def test_clap_handler_missing_file_is_permanent(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    deps = _deps(tmp_path, fc)
    deps.resolve_virtual_path = lambda _v: str(tmp_path / "does-not-exist.mp3")
    handler = make_clap_handler(deps)

    out = handler([_clap_job()])
    assert out[1].ok is False
    assert out[1].retry is False


def test_clap_handler_no_fingerprint_is_retryable(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    deps = _deps(tmp_path, fc)
    deps.cached_read_all = lambda _p: {"fingerprint": None}
    handler = make_clap_handler(deps)

    out = handler([_clap_job()])
    assert out[1].ok is False
    assert out[1].retry is True


def test_clap_handler_reports_model_unavailable(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeClap(ready=False, error="CUDA out of memory")
    handler = make_clap_handler(_deps(tmp_path, fc, clap=fake))

    out = handler([_clap_job()])
    assert out[1].ok is False
    assert "CUDA out of memory" in (out[1].error or "")
    assert not fc.has("clap", "fp", version=1)


# ── enqueue filter ──────────────────────────────────────────


def test_enqueue_clap_queues_only_missing_rows(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put("clap", "fp1", np.ones(CLAP_DIM, np.float32), version=1)
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_CLAP,)
    )
    assert queued[KIND_CLAP] == 1


def test_enqueue_clap_skipped_when_disabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    queued = enqueue_library_work(
        queue,
        infos,
        deps=_deps(tmp_path, fc, clap_enabled=False),
        kinds=(KIND_CLAP,),
    )
    assert queued.get(KIND_CLAP, 0) == 0
    assert KIND_CLAP not in queued


def test_enqueue_maest_does_not_queue_clap(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_MAEST,)
    )
    assert KIND_CLAP not in queued
