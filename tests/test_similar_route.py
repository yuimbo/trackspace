"""The /api/tracks/similar route and its revision-keyed similarity index.

Built with stub dependencies and a fake feature cache, so no audio is decoded and no
real cache is touched. The stub vectors are 2-D so the expected near/far order is
checkable by hand.
"""

from __future__ import annotations

import numpy as np
from flask import Flask

from backend.factory import _register_error_handlers
from backend.routes.api_similar import MAX_K, create_api_similar_blueprint
from backend.services.analysis_pipeline import AnalysisDeps
from backend.services.similarity import SimilarityIndex

# t0 is closest to t1 (cos 0.99) and furthest from t2 (cos -0.69);
# t3 sits between them (cos -0.54).
_VECTORS = {
    "fp0": np.array([1.0, 0.0], np.float32),
    "fp1": np.array([0.9, 0.1], np.float32),
    "fp2": np.array([0.0, 1.0], np.float32),
    "fp3": np.array([-1.0, 0.0], np.float32),
}

_INFOS = [
    {"path": "t0.mp3", "fingerprint": "fp0"},
    {"path": "t1.mp3", "fingerprint": "fp1"},
    {"path": "t2.mp3", "fingerprint": "fp2"},
    {"path": "t3.mp3", "fingerprint": "fp3"},
]


class _FakeCache:
    """``source -> {fingerprint: vector}`` rows, filtered like FeatureCache.get_all."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def get_all(self, source, fps, version=None):
        self.calls += 1
        rows = self.rows.get(source, {})
        return {fp: rows[fp] for fp in fps if fp in rows}

    def get_meta(self, key):
        raise KeyError(key)


class _FakeClusters:
    """No clustering run: the index falls back to the requested weights."""

    def get(self):
        return None

    def weights(self):
        return {"maest": 1.0}


def _deps(cache: _FakeCache) -> AnalysisDeps:
    return AnalysisDeps(
        feature_cache=cache,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp0"},
        resolve_virtual_path=lambda p: f"/abs/{p}",
        virtual_from_abs=lambda a: a,
        list_mp3s=lambda _f, _r: [],
        maest_mod=None,
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        structure_version=1,
        clap_version=1,
    )


def _app(*, rows=None, build_calls=None, index=None, clusters=None):
    cache = _FakeCache(
        rows
        if rows is not None
        else {"maest": _VECTORS, "maest_logits": _VECTORS}
    )

    def build_track_infos(_folder, _recursive):
        if build_calls is not None:
            build_calls.append(1)
        return _INFOS

    bp = create_api_similar_blueprint(
        analysis_deps=_deps(cache),
        index=index or SimilarityIndex(),
        clusters=clusters or _FakeClusters(),
        build_track_infos_fn=build_track_infos,
        library_mtime_signature=lambda _f, _r: "sig",
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client(), cache


def test_similar_returns_nearest_first():
    client, _ = _app()
    body = client.get("/api/tracks/similar?path=t0.mp3&k=2").get_json()
    assert body["ok"] is True
    assert body["mode"] == "near"
    assert body["n_tracks"] == 4
    assert [r["path"] for r in body["results"]] == ["t1.mp3", "t3.mp3"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_far_returns_the_anti_track():
    client, _ = _app()
    body = client.get("/api/tracks/similar?path=t0.mp3&mode=far&k=2").get_json()
    # t2 points away from t0 (cos -0.69) and is the anti-track; t3 is next.
    assert [r["path"] for r in body["results"]] == ["t2.mp3", "t3.mp3"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores)


def test_similar_excludes_the_queried_track():
    client, _ = _app()
    body = client.get("/api/tracks/similar?path=t0.mp3&k=10").get_json()
    assert "t0.mp3" not in [r["path"] for r in body["results"]]
    assert len(body["results"]) == 3  # k is capped by the library size


def test_similar_reuses_the_cached_index():
    build_calls: list[int] = []
    client, _ = _app(build_calls=build_calls)
    client.get("/api/tracks/similar?path=t0.mp3")
    client.get("/api/tracks/similar?path=t1.mp3&mode=far")
    assert len(build_calls) == 1


def test_similar_unknown_track_is_404():
    client, _ = _app()
    r = client.get("/api/tracks/similar?path=nope.mp3")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


def test_similar_missing_path_is_400():
    client, _ = _app()
    r = client.get("/api/tracks/similar")
    assert r.status_code == 400
    assert "path" in r.get_json()["error"]


def test_similar_bad_mode_is_400():
    client, _ = _app()
    r = client.get("/api/tracks/similar?path=t0.mp3&mode=sideways")
    assert r.status_code == 400
    assert "mode" in r.get_json()["error"]


def test_similar_k_is_bounded():
    client, _ = _app()
    r = client.get(f"/api/tracks/similar?path=t0.mp3&k={MAX_K + 1}")
    assert r.status_code == 400
    assert "k" in r.get_json()["error"]


def test_similar_without_vectors_is_503():
    client, _ = _app(rows={"maest": {}, "maest_logits": {}})
    r = client.get("/api/tracks/similar?path=t0.mp3")
    assert r.status_code == 503
    assert r.get_json()["ok"] is False


def test_index_rebuilds_when_the_key_changes():
    index = SimilarityIndex()
    blocks = {"maest": np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)}
    paths = ["a.mp3", "b.mp3"]

    first = index.build(("r1",), paths, blocks, {"maest": 1.0})
    assert first is not None
    assert index.lookup(("r1",)) is not None
    assert index.lookup(("r2",)) is None  # a new revision must rebuild
    assert index.build(("r2",), paths, blocks, {"maest": 1.0}) is not None
    assert index.lookup(("r2",)) is not None
