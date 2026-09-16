"""Features stage: per-track cache filtering, process-pool dispatch, madmom threading.

The stage is parallelised across *processes* because madmom's BLSTM is GIL-bound
(threads convoy and get slower). These tests use injected fakes for the pool and the
feature modules so no child process is ever spawned; the real spawn path is exercised
by ``tests/profile_features_workers.py``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from backend.embeddings.feature_cache import FeatureCache
from backend.jobs import Job, JobResult
from backend.services import features_worker
from backend.services.analysis_pipeline import AnalysisDeps, make_features_handler


class _FakeFeatureMod:
    """Stand-in for ``librosa_audio_features`` / ``rhythm_features``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.batch_calls: list[list[str]] = []

    def _batch(self, paths, decode_warnings=None):
        self.batch_calls.append(list(paths))
        if decode_warnings is not None:
            decode_warnings[paths[0]] = "clipped"
        return {p: np.full(self.dim, 0.5, np.float32) for p in paths}

    def generate_audio_features_batch(self, paths, decode_warnings=None):
        return self._batch(paths, decode_warnings)

    def generate_rhythm_features_batch(self, paths, decode_warnings=None):
        return self._batch(paths, decode_warnings)


def _deps(tmp_path, *, audio_mod=None, rhythm_mod=None, feature_cache=None):
    track = tmp_path / "a.mp3"
    track.write_bytes(b"not really an mp3")

    def _read(_abs_path: str) -> dict:
        return {"fingerprint": "fp-a"}

    return (
        AnalysisDeps(
            feature_cache=feature_cache or FeatureCache(str(tmp_path / "f.db")),
            track_cache=None,
            cached_read_all=_read,
            resolve_virtual_path=lambda v: str(track),
            virtual_from_abs=lambda a: "a.mp3",
            list_mp3s=lambda folder, recursive: [str(track)],
            maest_mod=None,
            rhythm_mod=rhythm_mod or _FakeFeatureMod(14),
            audio_features_mod=audio_mod or _FakeFeatureMod(6),
            maest_version=0,
            rhythm_version=1,
            features_version=6,
        ),
        str(track),
    )


def _job(job_id: int, path: str) -> Job:
    return Job(
        id=job_id,
        kind="features",
        dedupe_key=path,
        payload={"path": path},
        attempts=1,
        priority=100,
    )


def test_serial_handler_uses_injected_modules_and_caches(tmp_path):
    audio_mod = _FakeFeatureMod(6)
    rhythm_mod = _FakeFeatureMod(14)
    deps, track = _deps(tmp_path, audio_mod=audio_mod, rhythm_mod=rhythm_mod)
    handler = make_features_handler(deps, pool_size=1)

    results = handler([_job(1, "a.mp3")])

    assert results[1].ok
    assert (results[1].detail or {})["sources"] == ["features", "rhythm"]
    assert deps.feature_cache.has("features", "fp-a", version=6)
    assert deps.feature_cache.has("rhythm", "fp-a", version=1)
    assert audio_mod.batch_calls == [[track]]
    assert rhythm_mod.batch_calls == [[track]]


def test_handler_skips_sources_already_cached(tmp_path):
    audio_mod = _FakeFeatureMod(6)
    rhythm_mod = _FakeFeatureMod(14)
    deps, _ = _deps(tmp_path, audio_mod=audio_mod, rhythm_mod=rhythm_mod)
    deps.feature_cache.put("features", "fp-a", np.ones(6, np.float32), version=6)
    deps.feature_cache.put("rhythm", "fp-a", np.ones(14, np.float32), version=1)
    handler = make_features_handler(deps, pool_size=1)

    results = handler([_job(1, "a.mp3")])

    assert results[1].ok
    assert (results[1].detail or {}).get("cached") is True
    # Nothing recomputed, nothing re-written.
    assert audio_mod.batch_calls == []
    assert rhythm_mod.batch_calls == []


def test_handler_dispatches_each_track_through_the_pool(tmp_path):
    audio_mod = _FakeFeatureMod(6)
    rhythm_mod = _FakeFeatureMod(14)
    deps, _ = _deps(tmp_path, audio_mod=audio_mod, rhythm_mod=rhythm_mod)

    submitted: list[str] = []

    def compute(abs_path, *, want_features, want_rhythm):
        submitted.append(abs_path)
        return {
            "features": np.ones(6, np.float32) if want_features else None,
            "rhythm": np.ones(14, np.float32) if want_rhythm else None,
            "warnings": {},
            "error": None,
        }

    def pool_factory(size: int):
        assert size == 3
        return ThreadPoolExecutor(max_workers=size)

    handler = make_features_handler(
        deps, pool_size=3, pool_factory=pool_factory, compute_fn=compute
    )

    results = handler([_job(1, "a.mp3"), _job(2, "a.mp3")])

    assert submitted == [str(tmp_path / "a.mp3")] * 2
    assert all(r.ok for r in results.values())
    assert deps.feature_cache.count("features", version=6) == 1
    assert deps.feature_cache.count("rhythm", version=1) == 1


def test_handler_resets_pool_when_dispatch_fails(tmp_path):
    deps, _ = _deps(tmp_path)

    class _BrokenPool:
        def __init__(self) -> None:
            self.shutdowns = 0

        def submit(self, *_a, **_k):
            raise RuntimeError("pool is broken")

        def shutdown(self, **_k) -> None:
            self.shutdowns += 1

    pools: list[_BrokenPool] = []

    def pool_factory(size: int):
        pool = _BrokenPool()
        pools.append(pool)
        return pool

    handler = make_features_handler(
        deps, pool_size=2, pool_factory=pool_factory,
        compute_fn=lambda *a, **k: {},
    )

    results = handler([_job(1, "a.mp3")])
    assert results[1].ok is False
    assert "pool" in (results[1].error or "")
    assert pools[0].shutdowns == 1

    # The broken pool is dropped, so the next batch builds a fresh one.
    handler([_job(2, "a.mp3")])
    assert len(pools) == 2


def test_compute_track_features_reports_error_without_raising():
    out = features_worker.compute_track_features(
        "/nonexistent/track.mp3", want_features=True, want_rhythm=True
    )
    assert set(out) == {"features", "rhythm", "warnings", "error"}
    assert out["features"] is None
    assert out["rhythm"] is None


def test_compute_track_features_skips_disabled_blocks():
    out = features_worker.compute_track_features(
        "/nonexistent/track.mp3", want_features=False, want_rhythm=False
    )
    assert out["features"] is None
    assert out["rhythm"] is None
    assert out["error"] is None


def test_madmom_processors_are_per_thread():
    from backend.embeddings import madmom_tempo

    if not madmom_tempo.is_madmom_tempo_available():
        pytest.skip("madmom not installed")

    other: dict[int, object] = {}

    def grab() -> None:
        other[threading.get_ident()] = madmom_tempo._get_processors()

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    main_procs = madmom_tempo._get_processors()
    assert other
    assert next(iter(other.values())) is not main_procs
    # Reused within a thread — the BLSTM state is not rebuilt per track.
    assert madmom_tempo._get_processors() is main_procs


def test_features_worker_count_env(monkeypatch):
    from backend import app as app_mod

    monkeypatch.delenv("TRACKSPACE_FEATURES_WORKERS", raising=False)
    assert app_mod._features_worker_count() == app_mod.DEFAULT_FEATURES_WORKERS

    monkeypatch.setenv("TRACKSPACE_FEATURES_WORKERS", "2")
    assert app_mod._features_worker_count() == 2

    for bad in ("0", "-3", "many"):
        monkeypatch.setenv("TRACKSPACE_FEATURES_WORKERS", bad)
        assert app_mod._features_worker_count() == app_mod.DEFAULT_FEATURES_WORKERS
