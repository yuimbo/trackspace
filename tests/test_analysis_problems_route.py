"""The /api/analysis/problems route and the per-path enqueue filter.

Problems is the actionable list behind the coverage panel's failure counts: tracks with
no fingerprint (fpcalc cannot read them) and tracks with a permanently-failed stage job.
"""

from __future__ import annotations

from flask import Flask

from backend.factory import _register_error_handlers
from backend.routes.api_embeddings import create_api_embeddings_blueprint


class _FakeQueue:
    def __init__(self, failures):
        self._failures = failures

    def failures(self, kind=None, limit=20):
        return list(self._failures)


class _MertStub:
    MERT_VERSION = 1

    def is_model_ready(self):
        return False

    def load_error(self):
        return None


def _client(*, infos, failures, enqueue=None):
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
        build_track_infos_fn=lambda _f, _r, **_k: infos,
        library_root_ids=lambda: [],
        resolve_virtual_path=lambda rel: rel,
        virtual_from_abs_path=lambda abs_path: abs_path,
        broadcast_embed_event=lambda _e: None,
        emit_decode_warning_once=lambda *_a: None,
        broadcaster=None,
        sse_response=None,
        enqueue_analysis=enqueue or (lambda **_k: {}),
        job_queue=_FakeQueue(failures),
        maest_mod=None,
        mert_mod=_MertStub(),
        genre_full_enabled=False,
        mert_enabled=False,
        rhythm_version=1,
        structure_version=1,
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client()


def test_problems_lists_unfingerprinted_and_failed_tracks():
    client = _client(
        infos=[
            {"path": "a.mp3", "fingerprint": "fp1"},
            {"path": "b.mp3", "fingerprint": None},
        ],
        failures=[
            {"kind": "structure", "key": "a.mp3", "attempts": 3, "error": "decode failed", "at": 0.0},
            {"kind": "maest", "key": "a.mp3", "attempts": 3, "error": "no embedding", "at": 0.0},
        ],
    )
    body = client.get("/api/analysis/problems").get_json()
    assert body["ok"] is True
    assert body["unfingerprinted"] == ["b.mp3"]
    # Failures are aggregated per path, one entry per kind.
    assert body["failed"] == [
        {
            "path": "a.mp3",
            "kinds": ["structure", "maest"],
            "errors": ["decode failed", "no embedding"],
        }
    ]


def test_generate_paths_restricts_to_those_tracks():
    seen: dict[str, list[str]] = {}

    def enqueue(*, infos, **_k):
        seen["paths"] = [t["path"] for t in infos]
        return {}

    client = _client(
        infos=[
            {"path": "a.mp3", "fingerprint": "fp1"},
            {"path": "b.mp3", "fingerprint": "fp2"},
        ],
        failures=[],
        enqueue=enqueue,
    )
    client.post(
        "/api/embeddings/generate",
        json={"folder": "", "recursive": True, "stages": ["structure"], "paths": ["b.mp3"]},
    )
    assert seen["paths"] == ["b.mp3"]
