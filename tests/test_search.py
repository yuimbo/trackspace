"""Semantic text search: ranking, validation, and CLAP gating.

The route is built with stub dependencies, so these exercise the real validation
and ranking paths without loading a model or a feature cache.
"""

from __future__ import annotations

import numpy as np
from flask import Flask

from backend.factory import _register_error_handlers
from backend.routes.api_search import (
    create_api_search_blueprint,
    rank_by_similarity,
)


def _app(*blueprints) -> Flask:
    app = Flask(__name__)
    _register_error_handlers(app)
    for bp in blueprints:
        app.register_blueprint(bp)
    app.testing = True
    return app


# ─── ranking ────────────────────────────────────────────────


def test_rank_orders_by_cosine_descending():
    tracks = [
        ("a", np.array([1.0, 0.0], dtype=np.float32), None),
        ("b", np.array([0.0, 1.0], dtype=np.float32), None),
        ("c", np.array([1.0, 1.0], dtype=np.float32), None),
    ]
    out = rank_by_similarity(np.array([1.0, 0.0]), tracks, limit=3)
    assert [r["path"] for r in out] == ["a", "c", "b"]
    assert out[0]["score"] == 1.0
    assert out[0]["section"] is None


def test_rank_normalises_unnormalised_inputs():
    """Cached CLAP rows are segment means, not unit vectors."""
    tracks = [
        ("a", np.array([10.0, 0.0], dtype=np.float32), None),
        ("b", np.array([0.0, 3.0], dtype=np.float32), None),
    ]
    out = rank_by_similarity(np.array([5.0, 0.0]), tracks, limit=2)
    assert out[0]["path"] == "a"
    assert abs(out[0]["score"] - 1.0) < 1e-6
    assert abs(out[1]["score"]) < 1e-6


def test_rank_uses_best_segment_and_reports_section():
    tracks = [
        (
            "a",
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            np.array([0.2, 0.8], dtype=np.float32),
        ),
    ]
    out = rank_by_similarity(np.array([1.0, 0.0]), tracks, limit=1)
    assert out[0]["score"] == 1.0
    assert abs(out[0]["section"] - 0.8) < 1e-6


def test_rank_ignores_absent_segments():
    """A NaN segment position must never win, even when its score is highest."""
    tracks = [
        (
            "a",
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            np.array([np.nan, 0.8], dtype=np.float32),
        ),
    ]
    out = rank_by_similarity(np.array([0.0, 1.0]), tracks, limit=1)
    assert abs(out[0]["section"] - 0.8) < 1e-6


def test_rank_limit_and_empty():
    tracks = [
        ("a", np.array([1.0, 0.0], dtype=np.float32), None),
        ("b", np.array([0.5, 0.5], dtype=np.float32), None),
        ("c", np.array([0.0, 1.0], dtype=np.float32), None),
    ]
    out = rank_by_similarity(np.array([1.0, 0.0]), tracks, limit=2)
    assert len(out) == 2
    assert rank_by_similarity(np.array([1.0, 0.0]), [], limit=5) == []


def test_rank_ties_keep_input_order():
    """Identical audio (same fingerprint, two paths) must not reorder nondeterministically."""
    tracks = [
        ("z.mp3", np.array([1.0, 0.0], dtype=np.float32), None),
        ("a.mp3", np.array([1.0, 0.0], dtype=np.float32), None),
    ]
    out = rank_by_similarity(np.array([1.0, 0.0]), tracks, limit=2)
    assert [r["path"] for r in out] == ["z.mp3", "a.mp3"]


# ─── route ─────────────────────────────────────────────────


_MISSING = object()


class _FakeClap:
    def __init__(self, *, ready: bool = True, error: str | None = None, vec=_MISSING):
        self._ready = ready
        self._error = error
        self._vec = (
            np.array([[1.0, 0.0]], dtype=np.float32) if vec is _MISSING else vec
        )
        self.queries: list[list[str]] = []

    def is_model_ready(self) -> bool:
        return self._ready

    def load_error(self) -> str | None:
        return self._error

    def generate_text_embeddings(self, texts):
        self.queries.append(texts)
        return self._vec


class _FakeCache:
    def __init__(
        self,
        rows: dict[str, np.ndarray],
        segments: dict[str, np.ndarray] | None = None,
        positions: dict[str, np.ndarray] | None = None,
    ):
        self.rows = rows
        self.segments = segments or {}
        self.positions = positions or {}

    def get_all(self, source, fps, version=None):
        src = {
            "clap": self.rows,
            "clap_segments": self.segments,
            "clap_segment_pos": self.positions,
        }[source]
        return {fp: src[fp] for fp in fps if fp in src}


_INFOS = [
    {"path": "a.mp3", "fingerprint": "fp-a"},
    {"path": "b.mp3", "fingerprint": "fp-b"},
    {"path": "c.mp3", "fingerprint": "fp-c"},
    {"path": "missing.mp3", "fingerprint": None},
]

_ROWS = {
    "fp-a": np.array([1.0, 0.0], dtype=np.float32),
    "fp-b": np.array([0.0, 1.0], dtype=np.float32),
    "fp-c": np.array([0.9, 0.1], dtype=np.float32),
}


def _client(*, clap=None, rows=None, segments=None, positions=None):
    bp = create_api_search_blueprint(
        clap_mod=clap or _FakeClap(),
        feature_cache=_FakeCache(
            _ROWS if rows is None else rows, segments, positions
        ),
        embedding_version=2,
        build_track_infos_fn=lambda folder, recursive: list(_INFOS),
    )
    return _app(bp).test_client()


def test_search_ranks_cached_tracks():
    c = _client()
    r = c.get("/api/search?q=techno")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert [x["path"] for x in body["results"]] == ["a.mp3", "c.mp3", "b.mp3"]
    assert body["searched"] == 3
    assert body["total_tracks"] == 4


def test_search_prefers_best_segment_and_reports_section():
    segments = {"fp-a": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)}
    positions = {"fp-a": np.array([0.2, 0.8], dtype=np.float32)}
    c = _client(segments=segments, positions=positions)
    body = c.get("/api/search?q=x").get_json()
    assert body["results"][0]["path"] == "a.mp3"
    assert body["results"][0]["score"] == 1.0
    assert abs(body["results"][0]["section"] - 0.8) < 1e-6


def test_search_falls_back_to_mean_without_segments():
    c = _client()
    body = c.get("/api/search?q=x").get_json()
    assert body["results"][0]["section"] is None


def test_search_passes_query_to_text_encoder():
    clap = _FakeClap()
    c = _client(clap=clap)
    c.get("/api/search?q=dark%20ambient")
    assert clap.queries == [["dark ambient"]]


def test_search_limit_is_applied():
    c = _client()
    body = c.get("/api/search?q=x&limit=1").get_json()
    assert len(body["results"]) == 1
    assert body["results"][0]["path"] == "a.mp3"


def test_search_missing_query_is_400():
    c = _client()
    r = c.get("/api/search")
    assert r.status_code == 400
    assert "q" in r.get_json()["error"]


def test_search_empty_query_is_400():
    c = _client()
    r = c.get("/api/search?q=%20%20")
    assert r.status_code == 400


def test_search_too_long_query_is_400():
    c = _client()
    r = c.get("/api/search?q=" + "x" * 201)
    assert r.status_code == 400


def test_search_bad_limit_is_400():
    c = _client()
    assert c.get("/api/search?q=x&limit=abc").status_code == 400
    assert c.get("/api/search?q=x&limit=0").status_code == 400
    assert c.get("/api/search?q=x&limit=201").status_code == 400


def test_search_model_not_ready_is_503():
    c = _client(clap=_FakeClap(ready=False))
    r = c.get("/api/search?q=x")
    assert r.status_code == 503
    assert r.get_json()["ok"] is False
    assert "loading" in r.get_json()["error"]


def test_search_model_load_error_is_surfaced():
    c = _client(clap=_FakeClap(ready=False, error="ImportError: no torch"))
    r = c.get("/api/search?q=x")
    assert r.status_code == 503
    assert r.get_json()["error"] == "ImportError: no torch"


def test_search_text_encoder_failure_is_503():
    c = _client(clap=_FakeClap(vec=None))
    r = c.get("/api/search?q=x")
    assert r.status_code == 503
    assert r.get_json()["ok"] is False


def test_search_no_cached_embeddings_returns_empty():
    c = _client(rows={})
    body = c.get("/api/search?q=x").get_json()
    assert body["ok"] is True
    assert body["results"] == []
    assert body["searched"] == 0
