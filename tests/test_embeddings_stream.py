"""The embeddings SSE stream must speak the contract ``EmbeddingCoordinator`` reads.

The durable queue emits one ``progress`` event per job — carrying the path and
outcome but *no* queue totals — and one ``drained`` event per kind. The frontend
renders ``done/total`` off the progress event and treats ``done`` as terminal, so
the stream has to bridge the two vocabularies:

* merge the aggregate counts into every ``progress`` event, and
* only end once *every* analysis kind is idle — workers drain per kind, so the
  first ``drained`` (features) must not cut MAEST short.
"""

from __future__ import annotations

import json
import queue

from flask import Flask

from backend.factory import _register_error_handlers
from backend.jobs import KIND_FEATURES, KIND_MAEST
from backend.jobs.queue import QueueStats
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.sse import sse_response


class _FakeBroadcaster:
    """Hands out one shared queue the test pushes worker events onto."""

    def __init__(self) -> None:
        self.q: queue.Queue[dict] = queue.Queue()

    def subscribe(self, *, maxsize: int = 2000) -> queue.Queue[dict]:
        return self.q

    def unsubscribe(self, q: queue.Queue[dict]) -> None:
        return None


class _FakeQueue:
    """Stats the test mutates to model workers draining kind by kind."""

    def __init__(self, stats: dict) -> None:
        self.stats = stats

    def stats_by_kind(self) -> dict:
        return self.stats


def _stats(features_done: int, maest_done: int) -> dict:
    """Two jobs per kind; ``done`` grows until the kind drains."""
    return {
        KIND_FEATURES: QueueStats(done=features_done, pending=2 - features_done),
        KIND_MAEST: QueueStats(done=maest_done, pending=2 - maest_done),
    }


def _stream(client_queue: _FakeQueue, broadcaster: _FakeBroadcaster):
    bp = create_api_embeddings_blueprint(
        embeddings_mod=None,
        embedding_version=1,
        features_version=1,
        feature_cache=None,
        source_versions_cls=object,
        batch_fetch_maps=None,
        coverage_payload=None,
        layout_revision_for_projection=None,
        status_cache=None,
        library_mtime_signature=None,
        track_infos_cached=None,
        projection_query_fingerprint=None,
        parse_projection_params_fn=None,
        build_track_infos_fn=lambda folder, recursive: [],
        resolve_virtual_path=lambda rel: rel,
        virtual_from_abs_path=lambda abs_path: abs_path,
        broadcast_embed_event=lambda _e: None,
        emit_decode_warning_once=lambda *_a: None,
        broadcaster=broadcaster,
        sse_response=sse_response,
        enqueue_analysis=lambda **_k: {},
        job_queue=client_queue,
        maest_mod=None,
        rhythm_version=1,
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    return app


def _event(chunk: str) -> tuple[str, dict]:
    lines = chunk.strip().splitlines()
    etype = next(line for line in lines if line.startswith("event: "))[7:]
    data = json.loads(next(line for line in lines if line.startswith("data: "))[5:])
    return etype, data


def test_progress_events_carry_aggregate_totals_and_only_a_full_drain_ends():
    broadcaster = _FakeBroadcaster()
    job_queue = _FakeQueue(_stats(features_done=1, maest_done=0))
    app = _stream(job_queue, broadcaster)

    with app.test_request_context("/api/embeddings/stream"):
        response = app.view_functions[
            "api_embeddings.api_embeddings_stream"
        ]()
    stream = iter(response.response)

    # A per-job progress event has no queue totals of its own: the stream must
    # merge the aggregate the frontend renders as "done/total".
    broadcaster.q.put(
        {"type": "progress", "kind": KIND_FEATURES, "path": "a.mp3", "ok": True}
    )
    etype, data = _event(next(stream))
    assert etype == "progress"
    assert (data["done"], data["total"]) == (1, 4)
    assert data["path"] == "a.mp3"

    # The features worker drains while MAEST still has work: not terminal.
    job_queue.stats = _stats(features_done=2, maest_done=0)
    broadcaster.q.put({"type": "drained", "kind": KIND_FEATURES})
    broadcaster.q.put(
        {"type": "progress", "kind": KIND_MAEST, "path": "b.mp3", "ok": True}
    )
    etype, data = _event(next(stream))
    assert etype == "progress"
    assert (data["done"], data["total"]) == (2, 4)

    # Now every analysis kind is idle: the stream ends with ``done``.
    job_queue.stats = _stats(features_done=2, maest_done=2)
    broadcaster.q.put({"type": "drained", "kind": KIND_MAEST})
    etype, data = _event(next(stream))
    assert etype == "done"
    assert data == {}


def test_idle_queue_ends_immediately():
    broadcaster = _FakeBroadcaster()
    job_queue = _FakeQueue(_stats(features_done=2, maest_done=2))
    app = _stream(job_queue, broadcaster)

    with app.test_request_context("/api/embeddings/stream"):
        response = app.view_functions[
            "api_embeddings.api_embeddings_stream"
        ]()
    assert _event(next(iter(response.response))) == ("done", {})
