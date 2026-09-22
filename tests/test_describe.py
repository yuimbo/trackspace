"""Zero-shot CLAP descriptions: ranking, prompt-bank caching, and route gating.

The route is built with stub dependencies, so these exercise the real validation and
scoring paths without loading a model or a feature cache.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from backend.embeddings.clap_describe import (
    FACET_GENRE,
    FACET_MOOD,
    Axis,
    AxisDirections,
    Prompt,
    PromptBank,
    coordinates,
    rank_descriptions,
)
from backend.factory import _register_error_handlers
from backend.routes.api_describe import create_api_describe_blueprint


def _app(*blueprints) -> Flask:
    app = Flask(__name__)
    _register_error_handlers(app)
    for bp in blueprints:
        app.register_blueprint(bp)
    app.testing = True
    return app


_PROMPTS = (
    Prompt(FACET_GENRE, "a techno track"),
    Prompt(FACET_MOOD, "a calm track"),
)

_AXES = (
    Axis("sparse", "dense", mean=0.0, std=1.0, reliability=1.0),
    Axis("dark", "bright", mean=0.0, std=1.0, reliability=10.0),
)

# ─── ranking ────────────────────────────────────────────────


def test_rank_orders_by_cosine_descending():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    prompts = (
        Prompt(FACET_GENRE, "a"),
        Prompt(FACET_GENRE, "b"),
        Prompt(FACET_GENRE, "c"),
    )
    out = rank_descriptions(np.array([1.0, 0.0]), matrix, prompts, limit=3)
    assert [r["text"] for r in out] == ["a", "c", "b"]
    assert out[0]["score"] == 1.0
    assert out[0]["facet"] == FACET_GENRE


def test_rank_normalises_unnormalised_inputs():
    """Cached CLAP rows are segment means, not unit vectors."""
    matrix = np.array([[10.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    prompts = (Prompt(FACET_GENRE, "a"), Prompt(FACET_GENRE, "b"))
    out = rank_descriptions(np.array([5.0, 0.0]), matrix, prompts, limit=2)
    assert out[0]["text"] == "a"
    assert abs(out[0]["score"] - 1.0) < 1e-6
    assert abs(out[1]["score"]) < 1e-6


def test_rank_limit_and_shape_mismatch():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = rank_descriptions(np.array([1.0, 0.0]), matrix, _PROMPTS, limit=1)
    assert len(out) == 1
    # A matrix whose rows/columns do not match the bank or vector yields nothing.
    bad = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert rank_descriptions(np.array([1.0, 0.0]), bad, _PROMPTS, limit=2) == []


def test_prompt_bank_encodes_once_and_caches():
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    bank = PromptBank(_PROMPTS)
    first = bank.matrix(embed)
    second = bank.matrix(embed)
    assert first is not None
    assert first is second
    assert calls == [["a techno track", "a calm track"]]
    # Rows are unit-normalised.
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_prompt_bank_returns_none_when_encoder_unavailable():
    bank = PromptBank(_PROMPTS)
    assert bank.matrix(lambda texts: None) is None


# ─── coordinates ──────────────────────────────────────────


def test_coordinates_weight_by_reliability_and_label_the_pole():
    directions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = coordinates(np.array([0.5, 2.0]), directions, _AXES, limit=2)
    # Both axes are z=+2 raw, but the reliable axis is weighted first.
    assert out[0]["axis"] == "dark <-> bright"
    assert out[0]["pole"] == "dark"
    assert out[0]["percentile"] > 0.8
    assert out[1]["axis"] == "sparse <-> dense"


def test_coordinates_negative_leaning_pole():
    directions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = coordinates(np.array([-3.0, 0.0]), directions, _AXES, limit=1)
    assert out[0]["pole"] == "dense"
    assert out[0]["percentile"] < 0.2


def test_coordinates_shape_mismatch_is_empty():
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert coordinates(np.array([1.0, 0.0]), directions, _AXES, limit=2) == []


# ─── route ─────────────────────────────────────────────────


#: Deterministic per-text vectors: bank phrases and axis poles get distinct
#: directions, so ranking and coordinates are assertable without a model.
_VECS = {
    "a techno track": [1.0, 0.0],
    "a calm track": [0.0, 1.0],
    "a weak match": [0.2, 0.98],
    "sparse music": [1.0, 0.0],
    "dense music": [-1.0, 0.0],
    "dark music": [0.0, 1.0],
    "bright music": [0.0, -1.0],
}


class _FakeClap:
    def __init__(self, *, ready: bool = True, error: str | None = None, vec=_PROMPTS):
        self._ready = ready
        self._error = error
        self._vec = vec
        self.queries: list[list[str]] = []

    def is_model_ready(self) -> bool:
        return self._ready

    def load_error(self) -> str | None:
        return self._error

    def generate_text_embeddings(self, texts):
        self.queries.append(texts)
        if self._vec is None:
            return None
        if self._vec is not _PROMPTS:
            return self._vec
        return np.array(
            [_VECS.get(t, [1.0, 0.0]) for t in texts], dtype=np.float32
        )


class _FakeCache:
    def __init__(self, rows=None, segments=None):
        self.rows = {"fp-a": np.array([1.0, 0.0], dtype=np.float32)} if rows is None else rows
        self.segments = segments or {}

    def get_all(self, source, fps, version=None):
        src = {"clap": self.rows, "clap_segments": self.segments}[source]
        return {fp: src[fp] for fp in fps if fp in src}


def _client(*, clap=None, cache=None, fingerprint: str | None = "fp-a"):
    bp = create_api_describe_blueprint(
        clap_mod=clap or _FakeClap(),
        feature_cache=cache or _FakeCache(),
        embedding_version=3,
        resolve_virtual_path=lambda p: p,
        cached_read_all=lambda p: {"fingerprint": fingerprint} if fingerprint else {},
        bank=PromptBank(_PROMPTS),
        axis_directions=AxisDirections(_AXES),
    )
    return _app(bp).test_client()


def test_describe_ranks_a_cached_track():
    c = _client()
    body = c.get("/api/tracks/describe?path=a.mp3").get_json()
    assert body["ok"] is True
    assert body["n_descriptions"] == 2
    assert body["results"][0]["text"] == "a techno track"
    assert body["results"][0]["facet"] == FACET_GENRE


def test_describe_falls_back_to_segment_mean():
    segments = {"fp-a": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    c = _client(cache=_FakeCache(rows={}, segments=segments))
    body = c.get("/api/tracks/describe?path=a.mp3").get_json()
    assert body["ok"] is True
    assert body["results"][0]["text"] == "a techno track"


def test_describe_passes_the_bank_to_the_encoder():
    clap = _FakeClap()
    c = _client(clap=clap)
    c.get("/api/tracks/describe?path=a.mp3")
    # The bank is encoded first, then the positive poles, then the negative poles.
    assert clap.queries[0] == ["a techno track", "a calm track"]
    assert clap.queries[1] == [
        "sparse music",
        "dark music",
        "dense music",
        "bright music",
    ]


def test_describe_returns_coordinates():
    c = _client()
    body = c.get("/api/tracks/describe?path=a.mp3").get_json()
    assert body["ok"] is True
    assert {co["axis"] for co in body["coordinates"]} == {
        "sparse <-> dense",
        "dark <-> bright",
    }


def test_describe_limit_is_applied():
    c = _client()
    body = c.get("/api/tracks/describe?path=a.mp3&limit=1").get_json()
    assert len(body["results"]) == 1


def test_describe_drops_phrases_below_the_similarity_floor():
    # "a weak match" scores 0.20 against fp-a's [1, 0]; only the top survives.
    bp = create_api_describe_blueprint(
        clap_mod=_FakeClap(),
        feature_cache=_FakeCache(),
        embedding_version=3,
        resolve_virtual_path=lambda p: p,
        cached_read_all=lambda p: {"fingerprint": "fp-a"},
        bank=PromptBank(
            (Prompt(FACET_GENRE, "a techno track"), Prompt(FACET_GENRE, "a weak match"))
        ),
        axis_directions=AxisDirections(_AXES),
    )
    body = _app(bp).test_client().get("/api/tracks/describe?path=a.mp3").get_json()
    assert [r["text"] for r in body["results"]] == ["a techno track"]


def test_describe_unknown_track_is_404():
    c = _client(fingerprint=None)
    r = c.get("/api/tracks/describe?path=ghost.mp3")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


def test_describe_without_cached_embedding_is_404():
    c = _client(cache=_FakeCache(rows={}))
    r = c.get("/api/tracks/describe?path=a.mp3")
    assert r.status_code == 404
    assert "CLAP" in r.get_json()["error"]


def test_describe_missing_path_is_400():
    c = _client()
    r = c.get("/api/tracks/describe")
    assert r.status_code == 400
    assert "path" in r.get_json()["error"]


@pytest.mark.parametrize("limit", ["abc", "0", "21"])
def test_describe_bad_limit_is_400(limit):
    c = _client()
    assert c.get(f"/api/tracks/describe?path=a.mp3&limit={limit}").status_code == 400


def test_describe_model_not_ready_is_503():
    c = _client(clap=_FakeClap(ready=False))
    r = c.get("/api/tracks/describe?path=a.mp3")
    assert r.status_code == 503
    assert "loading" in r.get_json()["error"]


def test_describe_model_load_error_is_surfaced():
    c = _client(clap=_FakeClap(ready=False, error="ImportError: no torch"))
    r = c.get("/api/tracks/describe?path=a.mp3")
    assert r.status_code == 503
    assert r.get_json()["error"] == "ImportError: no torch"


def test_describe_text_encoder_failure_is_503():
    c = _client(clap=_FakeClap(vec=None))
    r = c.get("/api/tracks/describe?path=a.mp3")
    assert r.status_code == 503
    assert r.get_json()["ok"] is False
