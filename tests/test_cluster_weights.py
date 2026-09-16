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

    def _fake_run_clustering(deps, *, build_track_infos, weights=None, on_progress=None):
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

    def _fake_run_clustering(deps, *, build_track_infos, weights=None, on_progress=None):
        captured["weights"] = weights
        return None

    monkeypatch.setattr(ap, "run_clustering", _fake_run_clustering)

    handler = make_cluster_handler(
        _NO_DEPS, ClusterStore(), build_track_infos=lambda f, r: [], weights={"clap": 0.9}
    )
    handler([_job()])
    assert captured["weights"] == {"clap": 0.9}


def test_cluster_handler_forwards_phase_progress(monkeypatch):
    """The handler's ``emit`` sink receives the pipeline's phase updates."""
    captured: dict = {}

    def _fake_run_clustering(deps, *, build_track_infos, weights=None, on_progress=None):
        assert on_progress is not None
        on_progress("leiden", 2, 6)
        return None

    monkeypatch.setattr(ap, "run_clustering", _fake_run_clustering)

    phases: list[tuple] = []
    handler = make_cluster_handler(
        _NO_DEPS,
        ClusterStore(),
        build_track_infos=lambda f, r: [],
        emit=lambda phase, done, total: phases.append((phase, done, total)),
    )
    handler([_job()])
    assert phases == [("leiden", 2, 6)]


def test_cluster_handler_empty_batch_is_noop():
    store = ClusterStore()
    store.set_weights({"rhythm": 0.4})
    handler = make_cluster_handler(_NO_DEPS, store, build_track_infos=lambda f, r: [])
    assert handler([]) == {}


def test_map_and_cluster_weights_stay_in_sync():
    """The 2-D map and the clusters must describe the same space.

    ``projection_config.FROZEN_SOURCE_WEIGHTS`` mirrors ``clustering.DEFAULT_WEIGHTS``
    deliberately; a drift here would make the layout disagree with the clusters.
    """
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.embeddings.projection_config import (
        FROZEN_SOURCE_WEIGHTS,
        FROZEN_SOURCES,
        SUPERVISED_INPUT_WEIGHTS,
    )

    assert FROZEN_SOURCE_WEIGHTS == DEFAULT_WEIGHTS
    assert set(FROZEN_SOURCES) == set(FROZEN_SOURCE_WEIGHTS)
    # The head's input must not include its own output.
    assert "supervised" not in SUPERVISED_INPUT_WEIGHTS
    assert SUPERVISED_INPUT_WEIGHTS.get("maest_logits") == FROZEN_SOURCE_WEIGHTS["maest_logits"]
