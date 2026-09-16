"""The /api/tracks/structure route: validation, cache reads, and genre spanning.

Built with stub dependencies, so no audio is decoded and no real cache is touched.
"""

from __future__ import annotations

import numpy as np
from flask import Flask

from backend.embeddings.structure import TrackStructure, encode_structure
from backend.factory import _register_error_handlers
from backend.routes.api_structure import MAX_PATHS, create_api_structure_blueprint

_MAEST_DIM = 768


def _excerpts(*vectors: np.ndarray) -> np.ndarray:
    return np.stack(vectors).astype(np.float32)


def _app(*blueprints) -> Flask:
    app = Flask(__name__)
    _register_error_handlers(app)
    for bp in blueprints:
        app.register_blueprint(bp)
    app.testing = True
    return app


class _FakeCache:
    """``source -> {fingerprint: blob}`` rows, filtered like FeatureCache.get_all."""

    def __init__(self, rows: dict[str, dict[str, np.ndarray]]):
        self.rows = rows

    def get_all(self, source, fps, version=None):
        rows = self.rows.get(source, {})
        return {fp: rows[fp] for fp in fps if fp in rows}


_INFOS = [
    {"path": "a.mp3", "fingerprint": "fp-a"},
    {"path": "b.mp3", "fingerprint": "fp-b"},
    {"path": "missing.mp3", "fingerprint": None},
]

_STRUCTURES = {
    "fp-a": encode_structure(
        TrackStructure(intro_end=32.0, peak_start=64.0, peak_end=94.0, break_start=192.0)
    ),
}


def _client(*, rows=None, structure_version: int = 1):
    cache = _FakeCache({"structure": _STRUCTURES} if rows is None else rows)
    bp = create_api_structure_blueprint(
        feature_cache=cache,
        structure_version=structure_version,
        maest_version=1,
        build_track_infos_fn=lambda folder, recursive: list(_INFOS),
    )
    return _app(bp).test_client()


def test_structure_returns_cached_sections():
    body = _client().get("/api/tracks/structure?paths=a.mp3").get_json()
    assert body["ok"] is True
    assert body["tracks"] == [
        {
            "path": "a.mp3",
            "intro_end": 32.0,
            "peak_start": 64.0,
            "peak_end": 94.0,
            "break_start": 192.0,
            "break_end": None,
            "genre_spanning": False,
        }
    ]


def test_structure_omits_paths_without_a_row():
    body = _client().get("/api/tracks/structure?paths=a.mp3,b.mp3,missing.mp3").get_json()
    assert [t["path"] for t in body["tracks"]] == ["a.mp3"]


def test_structure_accepts_comma_separated_paths():
    body = _client().get("/api/tracks/structure?paths=a.mp3,b.mp3").get_json()
    assert body["ok"] is True
    assert [t["path"] for t in body["tracks"]] == ["a.mp3"]


def test_structure_missing_paths_is_400():
    r = _client().get("/api/tracks/structure")
    assert r.status_code == 400
    assert "paths" in r.get_json()["error"]


def test_structure_too_many_paths_is_400():
    paths = ",".join(f"p{i}.mp3" for i in range(MAX_PATHS + 1))
    r = _client().get(f"/api/tracks/structure?paths={paths}")
    assert r.status_code == 400
    assert "at most" in r.get_json()["error"]


def test_structure_flags_genre_spanning_tracks():
    a = np.zeros(_MAEST_DIM, np.float32)
    a[0] = 1.0
    b = np.zeros(_MAEST_DIM, np.float32)
    b[1] = 1.0  # orthogonal to a
    rows = {
        "structure": _STRUCTURES,
        "maest_excerpts": {"fp-a": _excerpts(a, b).reshape(-1)},
    }
    body = _client(rows=rows).get("/api/tracks/structure?paths=a.mp3").get_json()
    assert body["tracks"][0]["genre_spanning"] is True


def test_structure_not_spanning_without_two_excerpts():
    a = np.zeros(_MAEST_DIM, np.float32)
    a[0] = 1.0
    rows = {
        "structure": _STRUCTURES,
        "maest_excerpts": {"fp-a": _excerpts(a).reshape(-1)},
    }
    body = _client(rows=rows).get("/api/tracks/structure?paths=a.mp3").get_json()
    assert body["tracks"][0]["genre_spanning"] is False
