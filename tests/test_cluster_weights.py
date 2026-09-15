"""Requested source weights are sticky cluster state.

Re-clusters also fire automatically when an analysis stage drains, so weights must
live on the store rather than in a job payload — otherwise those automatic runs
would silently revert to the frozen defaults.
"""

from __future__ import annotations

from typing import cast

from backend.jobs import Job
from backend.services import analysis_pipeline as ap
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    ClusterStore,
    make_cluster_handler,
)

#: The handler only forwards *deps* to ``run_clustering``, which these tests stub.
_NO_DEPS = cast(AnalysisDeps, None)


def _job(job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind="cluster",
        dedupe_key="library",
        payload={"reason": "manual"},
        attempts=0,
        priority=200,
    )


def test_cluster_store_weights_round_trip():
    store = ClusterStore()
    assert store.weights() is None
    store.set_weights({"maest_logits": 0.5})
    assert store.weights() == {"maest_logits": 0.5}
    store.set_weights(None)
    assert store.weights() is None


def test_cluster_store_summary_exposes_requested_weights():
    store = ClusterStore()
    assert store.summary()["requested_weights"] is None
    store.set_weights({"rhythm": 0.4})
    assert store.summary()["requested_weights"] == {"rhythm": 0.4}


def test_cluster_handler_uses_stored_weights(monkeypatch):
    captured: dict = {}

    def _fake_run_clustering(deps, *, build_track_infos, weights=None):
        captured["weights"] = weights
        return None

    monkeypatch.setattr(ap, "run_clustering", _fake_run_clustering)

    store = ClusterStore()
    store.set_weights({"rhythm": 0.4})
    handler = make_cluster_handler(_NO_DEPS, store, build_track_infos=lambda f, r: [])
    out = handler([_job()])
    assert captured["weights"] == {"rhythm": 0.4}
    assert out[1].ok is True


def test_cluster_handler_falls_back_to_wiring_weights(monkeypatch):
    captured: dict = {}

    def _fake_run_clustering(deps, *, build_track_infos, weights=None):
        captured["weights"] = weights
        return None

    monkeypatch.setattr(ap, "run_clustering", _fake_run_clustering)

    handler = make_cluster_handler(
        _NO_DEPS, ClusterStore(), build_track_infos=lambda f, r: [], weights={"clap": 0.9}
    )
    handler([_job()])
    assert captured["weights"] == {"clap": 0.9}


def test_cluster_handler_empty_batch_is_noop():
    store = ClusterStore()
    store.set_weights({"rhythm": 0.4})
    handler = make_cluster_handler(_NO_DEPS, store, build_track_infos=lambda f, r: [])
    assert handler([]) == {}
