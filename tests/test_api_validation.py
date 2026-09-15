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

    def epoch(self):
        return 1

    def failures(self, kind, limit=50):
        return [{"kind": kind, "limit": limit}]

    def retry_failed(self, kind):
        return 0


class _FakeWorkers:
    def wake(self, kind=None):
        pass


@pytest.fixture
def jobs_client():
    bp = create_api_jobs_blueprint(
        job_queue=_FakeQueue(),
        workers=_FakeWorkers(),
        clusters=None,
        build_track_infos_fn=lambda folder, recursive: [],
        enqueue_library_work=lambda *a, **k: {},
        enqueue_clustering=lambda *a, **k: 0,
        analysis_deps=None,
        broadcaster=None,
        sse_response=None,
        maest_mod=None,
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


# ─── folders blueprint ──────────────────────────────────────


@pytest.fixture
def folders_client():
    bp = create_api_folders_blueprint(
        roots={},
        folder_tree=lambda *a, **k: {},
        resolve_virtual_path=lambda rel: "/music",
        virtual_from_abs=lambda p: p,
        add_root=lambda p: (True, "r"),
        remove_root=lambda rid: True,
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
        audio_features_mod=None,
        embedding_version=1,
        effnet_version=1,
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
