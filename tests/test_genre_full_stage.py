"""genre_full stage: MAEST layer-7 + MERT handler/queue contract.

Uses fake maest_mod/mert_mod so no model is loaded; the point is the handler/queue
contract: cache short-circuit, the layer-7 requirement when genre_full is enabled,
the separate MERT worker, and disabled servers never queue MERT.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.embeddings.feature_cache import FeatureCache
from backend.jobs import JOB_KINDS, KIND_MAEST, KIND_MERT, Job, JobQueue, kind_label
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_library_work,
    make_maest_handler,
    make_mert_handler,
)

MAEST_DIM = 768
L7_DIM = 2304
MERT_DIM = 1024


def _analysis(layer7=True, npad=1):
    return SimpleNamespace(
        embedding=np.full(MAEST_DIM, float(npad), np.float32),
        logits=np.full(519, float(npad), np.float32),
        excerpt_embeddings=np.full((1, MAEST_DIM), float(npad), np.float32),
        num_excerpts=1,
        layer7=np.full(L7_DIM, float(npad), np.float32) if layer7 else None,
    )


class _FakeMaest:
    def __init__(self, *, ready=True, error=None, layer7=True):
        self._ready = ready
        self._error = error
        self._layer7 = layer7
        self.calls = []
        self.reconcile_calls = 0

    def is_model_ready(self):
        return self._ready

    def load_error(self):
        return self._error

    def load_model(self):
        return None

    def reconcile_checkpoint(self, fc):
        self.reconcile_calls += 1
        return False

    def analyze_batch(self, paths, *, on_decode_warning=None, max_batch=12):
        self.calls.append(list(paths))
        return {p: _analysis(layer7=self._layer7, npad=len(p)) for p in paths}


class _FakeMert:
    def __init__(self, *, ready=True, error=None):
        self._ready = ready
        self._error = error
        self.calls = []

    def is_model_ready(self):
        return self._ready

    def load_error(self):
        return self._error

    def load_model(self):
        return None

    def analyze_batch(self, paths, *, on_decode_warning=None, max_batch=12):
        self.calls.append(list(paths))
        return {p: np.full(MERT_DIM, float(len(p)), np.float32) for p in paths}


def _deps(
    tmp_path,
    fc,
    *,
    maest=None,
    mert=None,
    layer7_enabled=False,
    mert_enabled=False,
) -> AnalysisDeps:
    track = tmp_path / "track.mp3"
    track.write_bytes(b"ID3\x00fake")
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda _v: str(track),
        virtual_from_abs=lambda _a: "track.mp3",
        list_mp3s=lambda _f, _r: [str(track)],
        maest_mod=maest or _FakeMaest(),
        mert_mod=mert or _FakeMert(),
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        maest_layer7_version=1,
        mert_version=1,
        layer7_enabled=layer7_enabled,
        mert_enabled=mert_enabled,
        data_dir=str(tmp_path),
    )


def _job(job_id: int = 1, kind: str = KIND_MAEST) -> Job:
    return Job(
        id=job_id,
        kind=kind,
        dedupe_key="track.mp3",
        payload={"path": "track.mp3"},
        attempts=0,
        priority=100,
    )


def test_mert_kind_is_registered_and_labelled():
    assert KIND_MERT in JOB_KINDS
    assert kind_label(KIND_MERT) == "MERT analysis"


# ── MAEST layer-7 handler ─────────────────────────────────────


def test_maest_handler_writes_layer7_when_enabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeMaest()
    handler = make_maest_handler(_deps(tmp_path, fc, maest=fake, layer7_enabled=True))

    out = handler([_job()])
    assert out[1].ok is True
    assert fc.has("maest", "fp", version=1)
    assert fc.has("maest_excerpts", "fp", version=1)
    assert fc.has("maest_layer7", "fp", version=1)
    assert fc.get("maest_layer7", "fp", version=1).shape == (L7_DIM,)

    again = handler([_job(2)])
    assert again[2].ok is True
    assert again[2].detail.get("cached") is True
    assert len(fake.calls) == 1


def test_maest_handler_without_layer7_requeues_when_enabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    # Model always reports no layer-7: the job must fail (and requeue), never
    # complete silently without the genre_full block.
    fake = _FakeMaest(layer7=False)
    handler = make_maest_handler(_deps(tmp_path, fc, maest=fake, layer7_enabled=True))
    out = handler([_job()])
    assert out[1].ok is False
    assert not fc.has("maest_layer7", "fp", version=1)


def test_maest_handler_ignores_layer7_when_disabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    # Default (no genre_full): a maest+excerpts row is cached without layer-7.
    fc.put_maest(
        "fp", np.ones(MAEST_DIM, np.float32), np.ones(519, np.float32),
        excerpts=np.ones((1, MAEST_DIM), np.float32), version=1,
    )
    fake = _FakeMaest()
    handler = make_maest_handler(_deps(tmp_path, fc, maest=fake))
    out = handler([_job()])
    assert out[1].ok is True
    assert out[1].detail.get("cached") is True
    assert len(fake.calls) == 0


def test_enqueue_maest_backfills_missing_layer7(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put_maest(
        "fp1", np.ones(MAEST_DIM, np.float32), np.ones(519, np.float32),
        excerpts=np.ones((1, MAEST_DIM), np.float32), version=1,
    )
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    # Without genre_full, the row is fully covered and nothing is queued.
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_MAEST,)
    )
    assert queued[KIND_MAEST] == 0
    # With genre_full, the missing layer-7 is re-queued on the same MAEST job.
    queued2 = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc, layer7_enabled=True),
        kinds=(KIND_MAEST,),
    )
    assert queued2[KIND_MAEST] == 1


# ── MERT handler ──────────────────────────────────────────────


def test_mert_handler_writes_source_then_serves_cache(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeMert()
    handler = make_mert_handler(_deps(tmp_path, fc, mert=fake, mert_enabled=True))

    out = handler([_job(kind=KIND_MERT)])
    assert out[1].ok is True
    assert fc.has("mert_l19", "fp", version=1)
    assert fc.get("mert_l19", "fp", version=1).shape == (MERT_DIM,)

    again = handler([_job(2, KIND_MERT)])
    assert again[2].ok is True
    assert again[2].detail.get("cached") is True
    assert len(fake.calls) == 1


def test_mert_handler_missing_file_is_permanent(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    deps = _deps(tmp_path, fc, mert_enabled=True)
    deps.resolve_virtual_path = lambda _v: str(tmp_path / "does-not-exist.mp3")
    handler = make_mert_handler(deps)
    out = handler([_job(kind=KIND_MERT)])
    assert out[1].ok is False
    assert out[1].retry is False


def test_mert_handler_model_unavailable(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fake = _FakeMert(ready=False, error="CC-BY-NC not accepted")
    handler = make_mert_handler(
        _deps(tmp_path, fc, mert=fake, mert_enabled=True)
    )
    out = handler([_job(kind=KIND_MERT)])
    assert out[1].ok is False
    assert "CC-BY-NC not accepted" in (out[1].error or "")
    assert not fc.has("mert_l19", "fp", version=1)


def test_mert_handler_produced_nothing_is_retryable(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))

    class _EmptyMert(_FakeMert):
        def analyze_batch(self, paths, **kw):
            return {p: None for p in paths}

    handler = make_mert_handler(
        _deps(tmp_path, fc, mert=_EmptyMert(), mert_enabled=True)
    )
    out = handler([_job(kind=KIND_MERT)])
    assert out[1].ok is False
    assert out[1].retry is True


# ── MERT enqueue filter ───────────────────────────────────────


def test_enqueue_mert_skipped_when_disabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_MERT,)
    )
    assert KIND_MERT not in queued


def test_enqueue_mert_queues_missing_when_enabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc, mert_enabled=True),
        kinds=(KIND_MERT,),
    )
    assert queued.get(KIND_MERT) == 1


def test_enqueue_mert_skips_fully_cached_when_enabled(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put("mert_l19", "fp1", np.ones(MERT_DIM, np.float32), version=1)
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc, mert_enabled=True),
        kinds=(KIND_MERT,),
    )
    assert queued[KIND_MERT] == 0
