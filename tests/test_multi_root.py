"""Multi-root integration tests — the "no hidden scope" invariant.

These build the real library and embeddings blueprints with real service
implementations (virtual paths, track payload, coverage, layout revision) over two
temporary library roots. Only the expensive/unrelated edges are stubbed: tag reads
return fake fingerprints instead of running fpcalc, model handles report "not ready",
and projection is a no-op.

The invariant under test: any route that takes an empty ``folder`` must cover *every*
registered root, not just the first/default one — and scoping to one root must never
leak tracks from another.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import pytest
from flask import Flask

from backend.embeddings.coverage import (
    SourceVersions,
    batch_fetch_maps,
    coverage_payload,
)
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.librosa_audio_features import audio_features_display_bpm_key
from backend.factory import _register_error_handlers
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.routes.api_library import create_api_library_blueprint
from backend.services.embedding_status_cache import EmbeddingStatusCache
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.parallel_reads import iter_parallel_reads, read_paths_parallel
from backend.services.projection_request import (
    embedding_projection_query_fingerprint,
    parse_projection_params,
)
from backend.services.track_payload import build_track_list, track_dict_from_read
from backend.services.virtual_paths import list_mp3s, virtual_from_abs


class _FakeQueue:
    def stats_by_kind(self):
        return {}


class _FakeEmbeddings:
    def is_model_ready(self) -> bool:
        return False

    def compute_projection(self, *_a, **_k) -> list:
        return []


class _FakeMaest:
    MAEST_VERSION = 1

    def is_model_ready(self) -> bool:
        return False

    def load_error(self) -> None:
        return None


class _FakeAudioFeatures:
    def is_effnet_ready(self) -> bool:
        return False

    audio_features_display_bpm_key = staticmethod(audio_features_display_bpm_key)


class _FakeBroadcaster:
    def subscribe(self, *, maxsize: int = 2000):
        raise NotImplementedError

    def unsubscribe(self, q) -> None:
        return None


def _mp3(path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ID3\x00fake")
    return str(path)


@pytest.fixture
def roots(tmp_path):
    """Two roots, each with tracks, plus an empty folder under root a."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _mp3(a / "one.mp3")
    _mp3(a / "sub" / "deep.mp3")
    _mp3(b / "two.mp3")
    _mp3(b / "shared" / "three.mp3")
    (a / "empty").mkdir(parents=True, exist_ok=True)
    # Same-named subfolder in both roots — a scoped query must not bleed.
    _mp3(a / "shared" / "a_only.mp3")
    _mp3(b / "shared" / "b_only.mp3")
    return OrderedDict([("a", str(a)), ("b", str(b))])


@pytest.fixture
def client(roots, tmp_path):
    roots_map = roots
    executor = ThreadPoolExecutor(max_workers=2)

    def cached_read_all(path: str) -> dict:
        name = os.path.basename(path)
        return {"fingerprint": f"fp-{name}", "tags": {"energy": 0.5}, "title": name}

    def list_mp3s_fn(folder: str, recursive: bool) -> list[str]:
        return list_mp3s(folder, roots_map, recursive)

    def virtual_from_abs_fn(path: str) -> str:
        return virtual_from_abs(path, roots_map)

    def build_infos(folder: str, recursive: bool) -> list[dict]:
        paths = list_mp3s_fn(folder, recursive)
        infos = []
        for p, info in iter_parallel_reads(
            paths, read_fn=cached_read_all, executor=executor
        ):
            infos.append({
                "path": virtual_from_abs_fn(p),
                "fingerprint": info.get("fingerprint"),
            })
        infos.sort(key=lambda t: t["path"])
        return infos

    def library_signature(folder: str, recursive: bool) -> str:
        h = hashlib.sha256()
        for p in list_mp3s_fn(folder, recursive):
            h.update(virtual_from_abs_fn(p).encode())
            h.update(b"\0")
        return h.hexdigest()

    status_cache = EmbeddingStatusCache()
    feature_cache = FeatureCache(str(tmp_path / "features.db"))

    def track_infos_cached(folder: str, recursive: bool, lib_sig: str) -> list[dict]:
        cached = status_cache.get_track_infos((folder, recursive, lib_sig))
        if cached is not None:
            return cached
        infos = build_infos(folder, recursive)
        status_cache.put_track_infos((folder, recursive, lib_sig), infos)
        return infos

    library_bp = create_api_library_blueprint(
        list_mp3s=list_mp3s_fn,
        cached_read_all=cached_read_all,
        executor=executor,
        feature_cache=feature_cache,
        features_version=1,
        audio_features_mod=_FakeAudioFeatures(),
        read_paths_parallel=read_paths_parallel,
        iter_parallel_reads=iter_parallel_reads,
        track_dict_from_read_service=track_dict_from_read,
        build_track_list_service=build_track_list,
        emit_scan_progress_events=lambda *a, **k: None,
        sse_response=lambda gen: gen,
        virtual_from_abs=virtual_from_abs_fn,
    )

    embeddings_bp = create_api_embeddings_blueprint(
        embeddings_mod=_FakeEmbeddings(),
        audio_features_mod=_FakeAudioFeatures(),
        embedding_version=1,
        effnet_version=1,
        features_version=1,
        feature_cache=feature_cache,
        source_versions_cls=SourceVersions,
        batch_fetch_maps=batch_fetch_maps,
        coverage_payload=coverage_payload,
        layout_revision_for_projection=layout_revision_for_projection,
        status_cache=status_cache,
        library_mtime_signature=library_signature,
        track_infos_cached=track_infos_cached,
        projection_query_fingerprint=embedding_projection_query_fingerprint,
        parse_projection_params_fn=parse_projection_params,
        build_track_infos_fn=build_infos,
        resolve_virtual_path=lambda rel: rel,
        virtual_from_abs_path=virtual_from_abs_fn,
        broadcast_embed_event=lambda _e: None,
        emit_decode_warning_once=lambda *_a: None,
        broadcaster=_FakeBroadcaster(),
        sse_response=lambda gen: gen,
        enqueue_analysis=lambda **_k: {},
        job_queue=_FakeQueue(),
        maest_mod=_FakeMaest(),
        rhythm_version=1,
    )

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(library_bp)
    app.register_blueprint(embeddings_bp)
    app.testing = True
    return app.test_client()


# ─── full-library listing covers every root ───────────────────


def test_tracks_empty_folder_lists_all_roots(client):
    r = client.get("/api/tracks")
    assert r.status_code == 200
    paths = {t["path"] for t in r.get_json()}
    assert paths == {
        "a/one.mp3",
        "a/sub/deep.mp3",
        "a/shared/a_only.mp3",
        "b/two.mp3",
        "b/shared/three.mp3",
        "b/shared/b_only.mp3",
    }


def test_tracks_scoped_to_one_root(client):
    r = client.get("/api/tracks?folder=a&recursive=1")
    paths = {t["path"] for t in r.get_json()}
    assert paths == {"a/one.mp3", "a/sub/deep.mp3", "a/shared/a_only.mp3"}


def test_tracks_scope_does_not_leak_across_same_named_subfolder(client):
    r = client.get("/api/tracks?folder=a/shared&recursive=1")
    paths = {t["path"] for t in r.get_json()}
    assert paths == {"a/shared/a_only.mp3"}


def test_tracks_empty_folder_returns_empty(client):
    r = client.get("/api/tracks?folder=a/empty")
    assert r.status_code == 200
    assert r.get_json() == []


def test_tags_empty_folder_union_across_roots(client):
    r = client.get("/api/tags")
    assert r.status_code == 200
    assert r.get_json() == ["energy"]


# ─── embeddings status / projection across roots ──────────────


def test_status_empty_folder_covers_all_roots(client):
    r = client.get("/api/embeddings/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["tracks_total"] == 6
    assert body["tracks_with_fingerprint"] == 6
    assert body["tracks_with_maest"] == 0
    assert body["tracks_pending_maest"] == 6


def test_status_scoped_to_one_root(client):
    r = client.get("/api/embeddings/status?folder=b&recursive=1")
    assert r.status_code == 200
    assert r.get_json()["tracks_total"] == 3


def test_status_with_empty_folder_is_zero_not_error(client):
    r = client.get("/api/embeddings/status?folder=a/empty")
    assert r.status_code == 200
    body = r.get_json()
    assert body["tracks_total"] == 0
    assert body["tracks_with_fingerprint"] == 0
    assert body["tracks_pending_maest"] == 0


def test_projection_empty_folder_covers_all_roots(client):
    r = client.get("/api/embeddings/projection")
    assert r.status_code == 200
    body = r.get_json()
    assert body["positions"] == []
    assert "revision" in body


def test_projection_with_empty_folder_is_empty_not_error(client):
    r = client.get("/api/embeddings/projection?folder=a/empty")
    assert r.status_code == 200
    assert r.get_json()["positions"] == []
