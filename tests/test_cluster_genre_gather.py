"""genre_full source gathering for clustering.

The opt-in genre_full blocks (``maest_layer7``, ``mert_l19``) are gathered only
when the recipe needs them, each gated by its own enabled flag. The default
gather and the supervised trainer's input must stay byte-identical to before.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings.clustering import (
    CLUSTER_GENRE_FULL_WEIGHTS,
    DEFAULT_WEIGHTS,
)
from backend.embeddings.supervised import build_input_matrix
from backend.embeddings.projection_config import (
    SUPERVISED_INPUT_WEIGHTS,
    GENRE_FULL_RECIPE_WEIGHTS,
)
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    gather_source_blocks,
    run_clustering,
)

#: The default recipe's gathered source names, in gather order.
_DEFAULT_SOURCES = (
    "maest",
    "maest_logits",
    "rhythm",
    "features",
    "clap",
    "supervised",
)
_GENRE_BLOCKS = ("maest_layer7", "mert_l19")

_FPS = ["fa", "fb", "fc"]
_PATHS = ["a.mp3", "b.mp3", "c.mp3"]


class _FakeCache:
    """``source -> {fingerprint: vector}`` rows, recording the reads made."""

    def __init__(self, rows: dict[str, dict[str, np.ndarray]]):
        self.rows = rows
        self.requested: list[str] = []

    def get_all(self, source, fps, version=None):
        self.requested.append(source)
        rows = self.rows.get(source, {})
        return {fp: rows[fp] for fp in fps if fp in rows}

    def get_meta(self, key):
        return 0  # no trained head → supervised source absent


def _rows(dim: int = 8) -> dict[str, np.ndarray]:
    return {fp: np.zeros(dim, np.float32) for fp in _FPS}


_ALL_SOURCES = _DEFAULT_SOURCES + _GENRE_BLOCKS


def _deps(cache, *, layer7_enabled=True, mert_enabled=True) -> AnalysisDeps:
    return AnalysisDeps(
        feature_cache=cache,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": _FPS[0]},
        resolve_virtual_path=lambda p: f"/abs/{p}",
        virtual_from_abs=lambda a: a,
        list_mp3s=lambda _f, _r: [],
        maest_mod=None,
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        clap_version=1,
        maest_layer7_version=1,
        mert_version=1,
        layer7_enabled=layer7_enabled,
        mert_enabled=mert_enabled,
    )


def _infos() -> list[dict]:
    return [{"path": p, "fingerprint": fp} for p, fp in zip(_PATHS, _FPS)]


def test_default_gather_is_unchanged() -> None:
    """The default gather reads exactly the frozen sources — no genre blocks."""
    cache = _FakeCache({s: _rows() for s in _DEFAULT_SOURCES})
    deps = _deps(cache)
    paths, blocks, logits = gather_source_blocks(
        deps, _infos(), clap_version=deps.clap_version
    )
    assert paths == _PATHS
    assert logits is not None
    assert set(blocks) == set(_DEFAULT_SOURCES)
    assert not (set(_GENRE_BLOCKS) & set(cache.requested))
    assert cache.requested == list(_DEFAULT_SOURCES)


def test_genre_full_gather_adds_optional_blocks() -> None:
    cache = _FakeCache(
        {s: _rows() for s in _ALL_SOURCES}
    )
    deps = _deps(cache)
    paths, blocks, _ = gather_source_blocks(
        deps, _infos(), clap_version=deps.clap_version, include_genre_full=True
    )
    assert set(blocks) == set(_ALL_SOURCES)
    assert blocks["maest_layer7"].shape[1] == 8
    assert blocks["mert_l19"].shape[1] == 8


def test_genre_full_gather_gates_mert_by_enablement() -> None:
    cache = _FakeCache(
        {s: _rows() for s in _ALL_SOURCES}
    )
    deps = _deps(cache, layer7_enabled=True, mert_enabled=False)
    _, blocks, _ = gather_source_blocks(
        deps, _infos(), clap_version=deps.clap_version, include_genre_full=True
    )
    assert "maest_layer7" in blocks
    assert "mert_l19" not in blocks
    assert "mert_l19" not in cache.requested


def test_default_gather_ignores_genre_blocks_when_not_requested() -> None:
    """Even when the cache has layer-7 rows, the default gather does not read them."""
    cache = _FakeCache({s: _rows() for s in _ALL_SOURCES})
    deps = _deps(cache)
    _, blocks, _ = gather_source_blocks(
        deps, _infos(), clap_version=deps.clap_version
    )
    assert "maest_layer7" not in blocks
    assert not (set(_GENRE_BLOCKS) & set(cache.requested))


def test_run_clustering_forwards_genre_weight_signal(monkeypatch) -> None:
    """A genre_full weight vector triggers the genre gather and carries its versions."""
    captured: dict = {}

    def _fake_gather(deps, infos, *, clap_version, on_progress=None, include_genre_full=False):
        captured["include_genre_full"] = include_genre_full
        captured["clap_version"] = clap_version
        rows = {
            s: {fp: np.zeros(4, np.float32) for fp in _FPS}
            for s in DEFAULT_WEIGHTS
        }
        rows["maest_layer7"] = {fp: np.zeros(8, np.float32) for fp in _FPS}
        rows["mert_l19"] = {fp: np.zeros(8, np.float32) for fp in _FPS}
        return _PATHS, rows, rows["maest_logits"]

    def _fake_cluster_tracks(paths, blocks, *, weights=None, style_logits=None,
                             style_label_names=None, cache_versions=(), on_progress=None):
        captured["cache_versions"] = cache_versions
        captured["weights"] = weights
        captured["blocks"] = set(blocks)
        return None

    from backend.services import analysis_pipeline as ap

    monkeypatch.setattr(ap, "gather_source_blocks", _fake_gather)
    monkeypatch.setattr(ap.clustering_mod, "cluster_tracks", _fake_cluster_tracks)

    deps = _deps(_FakeCache({}), layer7_enabled=True, mert_enabled=True)
    deps.maest_mod = type("M", (), {"discogs_style_labels": lambda self: []})()
    run_clustering(deps, build_track_infos=lambda folder, recursive: _infos(),
                   weights=CLUSTER_GENRE_FULL_WEIGHTS)

    assert captured["include_genre_full"] is True
    assert captured["cache_versions"] == (
        1, 1, 1, 0, 1, 1, 1,
    )  # maest, rhythm, features, supervised, clap, layer7, mert
    assert captured["blocks"] >= {"maest_layer7", "mert_l19"}


def test_run_clustering_default_weights_skip_genre_gather(monkeypatch) -> None:
    captured: dict = {}

    def _fake_gather(deps, infos, *, clap_version, on_progress=None, include_genre_full=False):
        captured["include_genre_full"] = include_genre_full
        rows = {s: {fp: np.zeros(4, np.float32) for fp in _FPS} for s in DEFAULT_WEIGHTS}
        return _PATHS, rows, rows["maest_logits"]

    def _fake_cluster_tracks(*args, **kwargs):
        return None

    from backend.services import analysis_pipeline as ap

    monkeypatch.setattr(ap, "gather_source_blocks", _fake_gather)
    monkeypatch.setattr(ap.clustering_mod, "cluster_tracks", _fake_cluster_tracks)

    deps = _deps(_FakeCache({}))
    deps.maest_mod = type("M", (), {"discogs_style_labels": lambda self: []})()
    run_clustering(deps, build_track_infos=lambda folder, recursive: _infos(),
                   weights=None)

    assert captured["include_genre_full"] is False


def test_supervised_input_restricts_to_frozen_sources() -> None:
    """The supervised trainer's gather must never consume genre_full blocks.

    ``build_input_matrix`` iterates only the frozen sources, so even a blocks
    dict that (hypothetically) carried the genre blocks ignores them — its input
    is exactly the raw old sources under ``SUPERVISED_INPUT_WEIGHTS``.
    """
    assert SUPERVISED_INPUT_WEIGHTS == {
        k: v for k, v in DEFAULT_WEIGHTS.items() if k != "supervised"
    }
    rng = np.random.default_rng(0)
    blocks = {
        "maest_logits": rng.normal(size=(4, 8)).astype(np.float32),
        "maest": rng.normal(size=(4, 8)).astype(np.float32),
        # Genre blocks present in the dict — they must be ignored.
        "maest_layer7": rng.normal(size=(4, 16)).astype(np.float32),
        "mert_l19": rng.normal(size=(4, 8)).astype(np.float32),
    }
    mat, used = build_input_matrix(blocks, SUPERVISED_INPUT_WEIGHTS)
    assert not (set(_GENRE_BLOCKS) & set(used))
    assert set(used) <= set(DEFAULT_WEIGHTS) - {"supervised"}


def test_genre_full_weights_are_canonical_and_zero_others() -> None:
    """The clustering genre recipe mirrors the projection lens exactly."""
    assert CLUSTER_GENRE_FULL_WEIGHTS == GENRE_FULL_RECIPE_WEIGHTS
    for src in ("rhythm", "clap", "supervised", "features"):
        assert CLUSTER_GENRE_FULL_WEIGHTS[src] == 0.0
    for src in ("maest_logits", "maest", "maest_layer7", "mert_l19"):
        assert CLUSTER_GENRE_FULL_WEIGHTS[src] > 0.0
