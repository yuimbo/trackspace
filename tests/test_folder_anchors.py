"""Semantic folder colour anchors: which ancestor a folder is a variation of.

The pure decision is tested with hand-placed 2-D vectors so the centroids and
cosines are checkable; the provider is tested with a fake cache so no audio is
decoded and the library-signature cache is exercised.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from backend.services.analysis_pipeline import AnalysisDeps
from backend.services.folder_anchors import (
    FolderAnchors,
    compute_folder_anchors,
)


_DIM = 9
_GENRES = ["Rock", "Jazz", "Pop", "Classical", "Folk", "Blues", "Soul", "Funk", "Disco"]


def _axis(i: int) -> np.ndarray:
    """Unit vector along axis *i* — near-orthogonal to every other genre axis."""
    v = np.zeros(_DIM, np.float32)
    v[i] = 1.0
    return v


def _near(i: int, j: int) -> np.ndarray:
    """A direction a hair off axis *i* toward axis *j* — a sub-genre of *i*."""
    v = _axis(i) + 0.05 * _axis(j)
    return (v / np.linalg.norm(v)).astype(np.float32)


# Each genre owns one near-orthogonal axis, so GENRES's centroid is ~1/√9 from each
# genre (distinct). Rock's 70s sits a hair off Rock's axis (a variation).
_TRACKS: list[tuple[str, np.ndarray]] = []
for _i, _name in enumerate(_GENRES):
    _TRACKS.append((f"r/GENRES/{_name}/a.mp3", _axis(_i)))
    _TRACKS.append((f"r/GENRES/{_name}/b.mp3", _axis(_i)))
_TRACKS.append(("r/GENRES/Rock/70s/c.mp3", _near(0, 1)))
_TRACKS.append(("r/GENRES/Rock/70s/d.mp3", _near(0, 1)))
_TRACKS.append(("r/GENRES/Rock/70s/Live/e.mp3", _near(0, 1)))
_TRACKS.append(("r/GENRES/Rock/70s/Live/f.mp3", _near(0, 1)))
_PATHS = [p for p, _ in _TRACKS]
_MATRIX = np.stack([v for _, v in _TRACKS])


def _anchors(**kwargs):
    # Rock's family (Rock + 70s + Live) pulls GENRES's centroid toward it, so the
    # threshold is raised here to keep the genre/sub-genre split unambiguous.
    kwargs.setdefault("min_tracks", 2)
    kwargs.setdefault("variation_min_cosine", 0.8)
    return compute_folder_anchors(["r"], _PATHS, _MATRIX, **kwargs)


def test_root_and_top_level_start_their_own_family():
    anchors = _anchors(min_tracks=2)
    assert anchors["r"] == "r"
    assert anchors["r/GENRES"] == "r/GENRES"


def test_subgenre_tints_its_genre():
    anchors = _anchors(min_tracks=2)
    assert anchors["r/GENRES/Rock/70s"] == "r/GENRES/Rock"


def test_distinct_genre_starts_its_own_family():
    anchors = _anchors(min_tracks=2)
    for genre in _GENRES:
        assert anchors[f"r/GENRES/{genre}"] == f"r/GENRES/{genre}"


def test_deeper_folder_inherits_its_family_anchor():
    anchors = _anchors(min_tracks=2)
    # Live is a hair off 70s, which anchors to Rock, so Live stays in Rock's family.
    assert anchors["r/GENRES/Rock/70s/Live"] == "r/GENRES/Rock"


def test_thin_external_folder_is_distinct():
    paths = ["ext/a.mp3", "ext/randA/b.mp3", "ext/randB/c.mp3"]
    matrix = np.stack([_axis(0), _axis(1), _axis(2)])
    anchors = compute_folder_anchors(
        ["ext"], paths, matrix, external_root_ids={"ext"}, min_tracks=2
    )
    assert anchors["ext"] == "ext"
    assert anchors["ext/randA"] == "ext/randA"
    assert anchors["ext/randB"] == "ext/randB"


def test_thin_library_folder_keeps_the_structural_anchor():
    paths = ["lib/A/B/x.mp3"]
    matrix = _axis(0).reshape(1, -1)
    anchors = compute_folder_anchors(["lib"], paths, matrix, min_tracks=2)
    # Too thin to measure: B inherits the top-level hue (structural).
    assert anchors["lib/A"] == "lib/A"
    assert anchors["lib/A/B"] == "lib/A"


def test_empty_library_anchors_only_the_roots():
    assert compute_folder_anchors(["r"], [], np.zeros((0, _DIM), np.float32)) == {
        "r": "r"
    }
    assert compute_folder_anchors([], _PATHS, _MATRIX) == {}


class _FakeCache:
    """``source -> {fingerprint: vector}`` rows, filtered like FeatureCache.get_all."""

    def __init__(self, rows, gate=None):
        self.rows = rows
        self.calls = 0
        #: When set, the first read blocks until the test releases it, so the
        #: provider's non-blocking read can be observed deterministically.
        self.gate = gate

    def get_all(self, source, fps, version=None):
        if self.gate is not None:
            self.gate.wait()
        self.calls += 1
        rows = self.rows.get(source, {})
        return {fp: rows[fp] for fp in fps if fp in rows}

    def get_meta(self, key):
        raise KeyError(key)


def _provider(cache, *, build_calls=None, roots=None, on_progress=None):
    infos = [
        {"path": p, "fingerprint": f"fp{i}"} for i, p in enumerate(_PATHS)
    ]

    def build_track_infos(_folder, _recursive, **_kwargs):
        if build_calls is not None:
            build_calls.append(1)
        return infos

    deps = AnalysisDeps(
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
    return FolderAnchors(
        deps,
        roots=roots if roots is not None else {"r": "/abs/r"},
        root_roles={},
        build_track_infos=build_track_infos,
        library_mtime_signature=lambda _f, _r: "sig",
        on_progress=on_progress,
        min_tracks=2,
        variation_min_cosine=0.8,
    )


def _cache(gate=None):
    rows = {f"fp{i}": _MATRIX[i] for i in range(len(_PATHS))}
    return _FakeCache({"maest": rows, "maest_logits": rows}, gate=gate)


def test_provider_builds_in_the_background_and_caches_by_signature():
    gate = threading.Event()
    cache = _cache(gate)
    build_calls: list[int] = []
    provider = _provider(cache, build_calls=build_calls)
    # The first read is non-blocking (the tree request must not stall): it
    # returns empty while the background build waits on the gate.
    assert provider.get() == {}
    assert provider.get() == {}  # a second read does not queue another build
    gate.set()
    provider.wait()
    anchors = provider.get()
    # A sub-genre always shares its genre's anchor, whichever way Rock lands.
    assert anchors["r/GENRES/Rock/70s"] == anchors["r/GENRES/Rock"]
    assert anchors["r/GENRES/Rock/70s/Live"] == anchors["r/GENRES/Rock"]
    calls_after_first = cache.calls
    assert calls_after_first > 0
    assert provider.get() == anchors
    assert cache.calls == calls_after_first  # a repeat hit does not recompute
    assert len(build_calls) == 1


def test_provider_reports_its_phases():
    events: list[dict] = []
    provider = _provider(_cache(), on_progress=events.append)
    provider.get()
    provider.wait()
    phases = [e["phase"] for e in events]
    assert phases[0] == "scanning"
    assert "reading" in phases
    assert "matrix" in phases
    assert phases[-1] == "done"


def test_provider_without_vectors_returns_empty():
    provider = _provider(_FakeCache({"maest": {}, "maest_logits": {}}))
    provider.get()
    provider.wait()
    assert provider.get() == {}
