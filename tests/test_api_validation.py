"""Route-level validation and structured-error tests.

Blueprints are built directly with stub dependencies (their factories take
injected callables), so these tests exercise the real validation and error-handling
paths without wiring the full application.
"""

from __future__ import annotations

import pytest
from flask import Flask, abort

from backend.factory import _register_error_handlers
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.routes.api_folders import create_api_folders_blueprint
from backend.routes.api_jobs import create_api_jobs_blueprint
from backend.routes.api_tags import create_api_tags_blueprint
from backend.services.projection_request import parse_projection_params
from backend.services.request_validation import ApiValidationError


def _app(*blueprints) -> Flask:
    app = Flask(__name__)
    _register_error_handlers(app)
    for bp in blueprints:
        app.register_blueprint(bp)
    app.testing = True
    return app


# ─── global error handlers ───────────────────────────────────


def test_error_handlers_return_json():
    app = Flask(__name__)
    _register_error_handlers(app)

    @app.route("/boom-validation")
    def _v():
        raise ApiValidationError("nope")

    @app.route("/boom-404")
    def _n():
        abort(404)

    @app.route("/boom-os")
    def _o():
        raise OSError("disk full")

    @app.route("/boom-unexpected")
    def _u():
        raise RuntimeError("kaboom")

    app.testing = True
    c = app.test_client()

    r = c.get("/boom-validation")
    assert r.status_code == 400
    assert r.get_json() == {"ok": False, "error": "nope"}

    r = c.get("/boom-404")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False

    r = c.get("/boom-os")
    assert r.status_code == 500
    assert r.get_json() == {"ok": False, "error": "disk full"}

    r = c.get("/boom-unexpected")
    assert r.status_code == 500
    assert r.get_json() == {"ok": False, "error": "Internal server error"}


# ─── tags blueprint ─────────────────────────────────────────


@pytest.fixture
def tags_client(monkeypatch):
    written: list[tuple] = []
    monkeypatch.setattr(
        "backend.routes.api_tags.write_tag",
        lambda p, t, v: written.append((p, t, v)),
    )
    monkeypatch.setattr(
        "backend.routes.api_tags.delete_tag",
        lambda p, t: written.append((p, t, None)),
    )
    monkeypatch.setattr(
        "backend.routes.api_tags.rename_tag", lambda p, o, n: True
    )

    def resolve(rel: str) -> str:
        if rel == "bad":
            raise ApiValidationError("unresolvable path")
        return f"/music/{rel}"

    bp = create_api_tags_blueprint(
        list_mp3s=lambda folder, recursive: ["a.mp3", "bad.mp3"],
        resolve_virtual_path=resolve,
    )
    return _app(bp).test_client(), written


def test_update_tags_missing_path_is_400(tags_client):
    client, _ = tags_client
    r = client.post("/api/tracks/tags", json={"updates": [{"tag": "e", "value": 0.5}]})
    assert r.status_code == 400
    assert "path" in r.get_json()["error"]


def test_update_tags_missing_tag_is_400(tags_client):
    client, _ = tags_client
    r = client.post("/api/tracks/tags", json={"updates": [{"path": "a", "value": 0.5}]})
    assert r.status_code == 400
    assert "tag" in r.get_json()["error"]


def test_update_tags_bad_value_type_is_400(tags_client):
    client, _ = tags_client
    r = client.post(
        "/api/tracks/tags", json={"updates": [{"path": "a", "tag": "e", "value": "loud"}]}
    )
    assert r.status_code == 400


def test_update_tags_value_out_of_range_is_400(tags_client):
    client, _ = tags_client
    r = client.post(
        "/api/tracks/tags", json={"updates": [{"path": "a", "tag": "e", "value": 2.0}]}
    )
    assert r.status_code == 400


def test_update_tags_malformed_body_is_400(tags_client):
    client, _ = tags_client
    r = client.post("/api/tracks/tags", data="not json", content_type="application/json")
    assert r.status_code == 400


def test_update_tags_null_value_deletes(tags_client):
    client, written = tags_client
    r = client.post(
        "/api/tracks/tags", json={"updates": [{"path": "a", "tag": "e", "value": None}]}
    )
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "count": 1, "errors": []}
    assert written == [("/music/a", "e", None)]


def test_update_tags_partial_success(tags_client):
    client, written = tags_client
    r = client.post(
        "/api/tracks/tags",
        json={
            "updates": [
                {"path": "a", "tag": "e", "value": 0.5},
                {"path": "bad", "tag": "e", "value": 0.5},
            ]
        },
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["count"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["path"] == "bad"
    assert written == [("/music/a", "e", 0.5)]


def test_rename_tag_missing_field_is_400(tags_client):
    client, _ = tags_client
    r = client.post("/api/tags/rename", json={"old": "e"})
    assert r.status_code == 400
    assert "new" in r.get_json()["error"]


def test_delete_tag_missing_name_is_400(tags_client):
    client, _ = tags_client
    r = client.post("/api/tags/delete", json={"folder": ""})
    assert r.status_code == 400


# ─── jobs blueprint ─────────────────────────────────────────


class _FakeQueue:
    def stats_by_kind(self):
        return {}

    def progress(self, kind):
        return None

    def stage_progress(self, kind):
        # Only the clustering stage is mid-run in this fixture.
        if kind == "cluster":
            return {"phase": "leiden", "done": 2, "total": 6, "at": 1.0}
        return None

    def epoch(self):
        return 1

    def failures(self, kind, limit=50):
        return [{"kind": kind, "limit": limit}]

    def failed_keys_by_kind(self):
        return {}

    def retry_failed(self, kind):
        return 0


class _FakeWorkers:
    def wake(self, kind=None):
        pass


class _FakeMaest:
    def is_model_ready(self):
        return True

    def load_error(self):
        return None


@pytest.fixture
def jobs_client():
    bp = create_api_jobs_blueprint(
        job_queue=_FakeQueue(),
        workers=_FakeWorkers(),
        clusters=_FakeClusters(),
        build_track_infos_fn=lambda folder, recursive, **_k: [],
        enqueue_library_work=lambda *a, **k: {},
        enqueue_clustering=lambda *a, **k: 0,
        enqueue_supervised=lambda *a, **k: 0,
        analysis_deps=None,
        broadcaster=None,
        sse_response=None,
        maest_mod=_FakeMaest(),
    )
    return _app(bp).test_client()


def test_jobs_failures_bad_limit_is_400(jobs_client):
    r = jobs_client.get("/api/jobs/failures?limit=abc")
    assert r.status_code == 400


def test_jobs_failures_valid_limit(jobs_client):
    r = jobs_client.get("/api/jobs/failures?limit=10")
    assert r.status_code == 200
    assert r.get_json()["failures"][0]["limit"] == 10


def test_jobs_enqueue_unknown_kind_is_400(jobs_client):
    r = jobs_client.post("/api/jobs/enqueue", json={"kinds": ["bogus"]})
    assert r.status_code == 400
    assert "bogus" in r.get_json()["error"]


def test_jobs_enqueue_valid(jobs_client):
    r = jobs_client.post("/api/jobs/enqueue", json={})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_jobs_retry_unknown_kind_is_400(jobs_client):
    r = jobs_client.post("/api/jobs/retry", json={"kind": "bogus"})
    assert r.status_code == 400


def test_jobs_status_carries_singleton_stage_phase(jobs_client):
    """A mid-run singleton phase must survive the status snapshot, not just SSE."""
    body = jobs_client.get("/api/jobs/status").get_json()
    assert body["kinds"]["cluster"]["stage"] == {
        "phase": "leiden",
        "done": 2,
        "total": 6,
        "at": 1.0,
    }
    # Per-track kinds carry no phase.
    assert "stage" not in body["kinds"]["maest"]


# ─── folders blueprint ──────────────────────────────────────


@pytest.fixture
def folders_client():
    bp = create_api_folders_blueprint(
        roots={},
        root_roles={},
        folder_tree=lambda *a, **k: {},
        resolve_virtual_path=lambda rel: "/music",
        virtual_from_abs=lambda p: p,
        add_root=lambda p: (True, "r"),
        remove_root=lambda rid: True,
        set_root_role=lambda rid, role: None,
        schedule_root_watch=lambda rid: None,
        unschedule_root_watch=lambda rid: None,
        save_roots_state=lambda: None,
        track_cache=None,
        dir_color_fn=lambda p: "#000",
    )
    return _app(bp).test_client()


def test_move_tracks_paths_not_list_is_400(folders_client):
    r = folders_client.post("/api/tracks/move", json={"paths": "a.mp3", "dest": ""})
    assert r.status_code == 400


def test_move_tracks_paths_wrong_item_type_is_400(folders_client):
    r = folders_client.post("/api/tracks/move", json={"paths": ["a.mp3", 3], "dest": ""})
    assert r.status_code == 400


def test_create_folder_traversal_name_is_400(folders_client):
    r = folders_client.post("/api/folders/create", json={"parent": "", "name": "../evil"})
    assert r.status_code == 400


def test_create_folder_empty_name_is_400(folders_client):
    r = folders_client.post("/api/folders/create", json={"parent": "", "name": "  "})
    assert r.status_code == 400


def test_roots_add_missing_path_is_400(folders_client):
    r = folders_client.post("/api/roots/add", json={})
    assert r.status_code == 400


# ─── embeddings generate ────────────────────────────────────


@pytest.fixture
def embeddings_client():
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
        build_track_infos_fn=lambda folder, recursive, **_k: [],
        library_root_ids=lambda: [],
        resolve_virtual_path=None,
        virtual_from_abs_path=None,
        broadcast_embed_event=None,
        emit_decode_warning_once=None,
        broadcaster=None,
        sse_response=None,
        enqueue_analysis=lambda **kw: {"maest": 0},
        job_queue=None,
        maest_mod=None,
        rhythm_version=1,
        structure_version=1,
    )
    return _app(bp).test_client()


def test_generate_priority_paths_not_list_is_400(embeddings_client):
    r = embeddings_client.post(
        "/api/embeddings/generate", json={"priority_paths": "a.mp3"}
    )
    assert r.status_code == 400


def test_generate_priority_paths_wrong_item_type_is_400(embeddings_client):
    r = embeddings_client.post(
        "/api/embeddings/generate", json={"priority_paths": ["a.mp3", 7]}
    )
    assert r.status_code == 400


def test_generate_valid_empty_body(embeddings_client):
    r = embeddings_client.post("/api/embeddings/generate")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_generate_unknown_stage_is_400(embeddings_client):
    r = embeddings_client.post(
        "/api/embeddings/generate", json={"stages": ["bogus"]}
    )
    assert r.status_code == 400
    assert "stages" in r.get_json()["error"]


def test_generate_stages_not_list_is_400(embeddings_client):
    r = embeddings_client.post(
        "/api/embeddings/generate", json={"stages": "clap"}
    )
    assert r.status_code == 400


# ─── projection method enum ────────────────────────────────


class _Req:
    def __init__(self, args: dict[str, str]):
        self.args = args


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "tsne"), ("", "tsne"), ("tsne", "tsne"), ("pca", "pca"), ("umap", "tsne"), ("PCA", "pca")],
)
def test_projection_method_accepted(raw, expected):
    args = {} if raw is None else {"method": raw}
    assert parse_projection_params(_Req(args)).method == expected


def test_projection_method_unknown_is_400():
    with pytest.raises(ApiValidationError):
        parse_projection_params(_Req({"method": "bogus"}))


# ─── source weights ────────────────────────────────────────


def test_weight_map_fills_unspecified_from_defaults():
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    out = weight_map(
        {"rhythm": 0.5}, allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS
    )
    assert out["rhythm"] == 0.5
    assert out["maest_logits"] == DEFAULT_WEIGHTS["maest_logits"]
    assert set(out) == set(DEFAULT_WEIGHTS)


def test_weight_map_rejects_unknown_source():
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    with pytest.raises(ApiValidationError):
        weight_map({"bogus": 1.0}, allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_weight_map_rejects_out_of_range(value):
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    with pytest.raises(ApiValidationError):
        weight_map(
            {"rhythm": value}, allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS
        )


def test_weight_map_rejects_non_number():
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    with pytest.raises(ApiValidationError):
        weight_map(
            {"rhythm": "loud"}, allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS
        )


def test_weight_map_rejects_non_mapping():
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    with pytest.raises(ApiValidationError):
        weight_map([1, 2], allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS)


def test_weight_map_rejects_all_zero():
    """An all-zero map leaves no feature block: clustering returns nothing and the
    layout would index an empty matrix."""
    from backend.embeddings.clustering import DEFAULT_WEIGHTS
    from backend.services.request_validation import weight_map

    with pytest.raises(ApiValidationError):
        weight_map(
            {k: 0.0 for k in DEFAULT_WEIGHTS},
            allowed=DEFAULT_WEIGHTS,
            defaults=DEFAULT_WEIGHTS,
        )


def test_projection_all_zero_weights_is_400():
    from backend.embeddings.projection_config import FROZEN_SOURCE_WEIGHTS

    spec = ",".join(f"{k}:0" for k in FROZEN_SOURCE_WEIGHTS)
    with pytest.raises(ApiValidationError):
        parse_projection_params(_Req({"w": spec}))


def test_projection_weight_spec_parsed():
    from backend.embeddings.projection_config import FROZEN_SOURCE_WEIGHTS

    p = parse_projection_params(_Req({"w": "maest_logits:0.5,rhythm:0.2"}))
    assert p.source_weights is not None
    assert p.source_weights["maest_logits"] == 0.5
    assert p.source_weights["rhythm"] == 0.2
    # Unlisted sources keep their frozen default.
    assert p.source_weights["maest"] == FROZEN_SOURCE_WEIGHTS["maest"]


def test_projection_without_weights_is_none():
    assert parse_projection_params(_Req({})).source_weights is None


def test_projection_weight_spec_unknown_source_is_400():
    with pytest.raises(ApiValidationError):
        parse_projection_params(_Req({"w": "bogus:0.5"}))


def test_projection_weight_spec_malformed_is_400():
    with pytest.raises(ApiValidationError):
        parse_projection_params(_Req({"w": "maest"}))


def test_projection_weight_spec_out_of_range_is_400():
    with pytest.raises(ApiValidationError):
        parse_projection_params(_Req({"w": "rhythm:2"}))


# ─── clusters rebuild: source weights ──────────────────────


class _FakeClusters:
    """Minimal ClusterStore stand-in for the rebuild route."""

    def __init__(self):
        self._weights: dict[str, float] | None = None

    def set_weights(self, weights):
        self._weights = dict(weights) if weights else None

    def weights(self):
        return self._weights

    def get(self):
        return None

    def revision(self):
        return None

    def summary(self):
        return {"available": False, "requested_weights": self._weights}


@pytest.fixture
def rebuild_client():
    from backend.embeddings.clustering import DEFAULT_WEIGHTS

    clusters = _FakeClusters()
    enqueued: list[dict] = []

    def _enqueue_clustering(queue, *, reason="library-change"):
        enqueued.append({"queue": queue, "reason": reason})
        return 1

    bp = create_api_jobs_blueprint(
        job_queue=_FakeQueue(),
        workers=_FakeWorkers(),
        clusters=clusters,
        build_track_infos_fn=lambda folder, recursive, **_k: [],
        enqueue_library_work=lambda *a, **k: {},
        enqueue_clustering=_enqueue_clustering,
        enqueue_supervised=lambda *a, **k: 0,
        analysis_deps=None,
        broadcaster=None,
        sse_response=None,
        maest_mod=None,
    )
    return _app(bp).test_client(), clusters, enqueued, DEFAULT_WEIGHTS


def test_rebuild_without_weights_keeps_current(rebuild_client):
    client, clusters, enqueued, _ = rebuild_client
    r = client.post("/api/clusters/rebuild", json={})
    assert r.status_code == 200
    # Untouched: the worker falls back to the frozen defaults.
    assert clusters.weights() is None
    assert len(enqueued) == 1


def test_rebuild_with_weights_stores_and_queues(rebuild_client):
    client, clusters, enqueued, defaults = rebuild_client
    r = client.post("/api/clusters/rebuild", json={"weights": {"rhythm": 0.5}})
    assert r.status_code == 200
    stored = clusters.weights()
    assert stored is not None
    assert stored["rhythm"] == 0.5
    assert stored["maest_logits"] == defaults["maest_logits"]
    assert r.get_json()["weights"] == stored
    assert len(enqueued) == 1


def test_rebuild_unknown_source_is_400(rebuild_client):
    client, clusters, enqueued, _ = rebuild_client
    r = client.post("/api/clusters/rebuild", json={"weights": {"bogus": 0.5}})
    assert r.status_code == 400
    assert "bogus" in r.get_json()["error"]
    assert enqueued == []
    assert clusters.weights() is None


def test_rebuild_out_of_range_weight_is_400(rebuild_client):
    client, clusters, enqueued, _ = rebuild_client
    r = client.post("/api/clusters/rebuild", json={"weights": {"rhythm": 3.0}})
    assert r.status_code == 400
    assert enqueued == []


def test_rebuild_all_zero_weights_is_400(rebuild_client):
    client, clusters, enqueued, defaults = rebuild_client
    r = client.post(
        "/api/clusters/rebuild", json={"weights": {k: 0 for k in defaults}}
    )
    assert r.status_code == 400
    assert "positive weight" in r.get_json()["error"]
    assert enqueued == []
    assert clusters.weights() is None
