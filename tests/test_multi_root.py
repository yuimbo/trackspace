"""Multi-root integration tests — the "no hidden scope" invariant.

These build the real library and embeddings blueprints with real service
implementations (virtual paths, track payload, coverage, layout revision) over two
temporary library roots. Only the expensive/unrelated edges are stubbed: tag reads
return fake fingerprints instead of running fpcalc, model handles report "not ready",
and projection echoes the infos it was scoped to.

The invariant under test: any route that takes an empty ``folder`` must cover *every*
registered root, not just the first/default one — and scoping to one root must never
leak tracks from another.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from flask import Flask

from backend.embeddings.coverage import (
    SourceVersions,
    batch_fetch_maps,
    coverage_payload,
)
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.layout import compute_projection as real_compute_projection
from backend.embeddings.librosa_audio_features import audio_features_display_bpm_key
from backend.embeddings.projection_config import FROZEN_SOURCES
from backend.factory import _register_error_handlers
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.routes.api_library import create_api_library_blueprint
from backend.services.embedding_status_cache import EmbeddingStatusCache
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.lens_layout import compute_lens_projection
from backend.services.parallel_reads import iter_parallel_reads, read_paths_parallel
from backend.services.projection_request import (
    embedding_projection_query_fingerprint,
    parse_projection_params,
    projection_params_for_lens,
)
from backend.services.track_payload import build_track_list, track_dict_from_read
from backend.services.virtual_paths import list_mp3s, virtual_from_abs


class _FakeQueue:
    def stats_by_kind(self):
        return {}

    def failed_keys_by_kind(self):
        return {}


class _FakeEmbeddings:
    def __init__(self, *, fail_on_dims2: bool = False):
        #: Recorded ``compute_projection`` kwargs, for asserting the 3-D route
        #: computes the 2-D reference first and passes it as ``align_ref``.
        self.calls: list[dict] = []
        self.fail_on_dims2 = fail_on_dims2

    def is_model_ready(self) -> bool:
        return False

    def compute_projection(self, infos, *_a, **_k):
        # Binding against the real signature makes every route test fail on a
        # wrong or renamed argument, which an accept-anything fake would hide.
        inspect.signature(real_compute_projection).bind(infos, *_a, **_k)
        dims = _k.get("dims", 2)
        self.calls.append({
            "dims": dims,
            "paths": [t["path"] for t in infos],
            "has_align_ref": "align_ref" in _k,
            "align_ref": _k.get("align_ref"),
            "gauge_scope": _k.get("gauge_scope"),
            "with_jacobian": bool(_k.get("with_jacobian")),
        })
        if self.fail_on_dims2 and dims == 2:
            raise RuntimeError("reference projection failed")
        # Echo the infos the route scoped for us, so a test can assert which
        # roots made it into the layout. dims=3 asks for a z coordinate.
        positions = [
            {
                "path": t["path"],
                "x": 0.0,
                "y": 0.0,
                **({"z": 0.0} if dims == 3 else {}),
            }
            for t in infos
        ]
        if _k.get("with_jacobian"):
            # Mirrors the real signature: (positions, per-source jacobian). With
            # fewer than two tracks there is no chart, so the real fallback yields
            # an empty (not empty-shaped) Jacobian.
            jacobian = (
                {"clap": np.zeros((len(infos), dims), dtype=np.float32)}
                if len(infos) >= 2
                else {}
            )
            return positions, jacobian
        return positions


class _FakeMaest:
    MAEST_VERSION = 1
    MAEST_LAYER7_VERSION = 1

    def is_model_ready(self) -> bool:
        return False

    def load_error(self) -> None:
        return None


class _FakeMert:
    MERT_VERSION = 1

    def is_model_ready(self) -> bool:
        return False

    def load_error(self) -> None:
        return None


class _FakeAudioFeatures:
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
def fake_embeddings():
    return _FakeEmbeddings()


@pytest.fixture
def client(roots, tmp_path, fake_embeddings):
    return _make_client(roots, tmp_path, fake_embeddings)


def _make_client(
    roots, tmp_path, fake_embeddings, *, genre_full_enabled=False, library_roots=None
):
    roots_map = roots
    library_roots = ["a"] if library_roots is None else list(library_roots)
    executor = ThreadPoolExecutor(max_workers=2)

    def cached_read_all(path: str) -> dict:
        name = os.path.basename(path)
        return {"fingerprint": f"fp-{name}", "tags": {"energy": 0.5}, "title": name}

    def list_mp3s_fn(folder: str, recursive: bool) -> list[str]:
        return list_mp3s(folder, roots_map, recursive)

    def virtual_from_abs_fn(path: str) -> str:
        return virtual_from_abs(path, roots_map)

    def build_infos(
        folder: str, recursive: bool, *, include_external: bool = True, **_kw
    ) -> list[dict]:
        # Root "b" stands in for an external root here, so the default layout
        # walks root "a" only.
        root_ids = None if include_external else ["a"]
        paths = list_mp3s(folder, roots_map, recursive, root_ids=root_ids)
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
        track_cache=None,
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
        embeddings_mod=fake_embeddings,
        embedding_version=1,
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
        parse_projection_params_fn=lambda req, **kw: parse_projection_params(
            req, genre_full_enabled=genre_full_enabled, **kw
        ),
        build_track_infos_fn=build_infos,
        library_root_ids=lambda: library_roots,
        resolve_virtual_path=lambda rel: rel,
        virtual_from_abs_path=virtual_from_abs_fn,
        broadcast_embed_event=lambda _e: None,
        emit_decode_warning_once=lambda *_a: None,
        broadcaster=_FakeBroadcaster(),
        sse_response=lambda gen: gen,
        enqueue_analysis=lambda **_k: {},
        job_queue=_FakeQueue(),
        maest_mod=_FakeMaest(),
        mert_mod=_FakeMert(),
        genre_full_enabled=genre_full_enabled,
        mert_enabled=False,
        rhythm_version=1,
        structure_version=1,
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
    maest = next(s for s in body["sources"] if s["id"] == "maest")
    assert maest["done"] == 0
    assert maest["pending"] == 6


def test_status_tolerates_disabled_genre_recipe(client):
    """A disabled server's status still answers ``recipe=genre_full``.

    The frontend capability bootstrap queries the recipe so it can learn
    ``genre_full_enabled`` (absent in the payload) and reconcile away from a genre
    map it cannot render. Unrelated invalid args are still a 400.
    """
    r = client.get("/api/embeddings/status?recipe=genre_full")
    assert r.status_code == 200
    assert "genre_full_enabled" not in r.get_json()
    # The status tolerance must not mask a genuinely bad request (sources= with a
    # recipe is still rejected).
    r2 = client.get("/api/embeddings/status?recipe=genre_full&sources=clap")
    assert r2.status_code == 400


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
    maest = next(s for s in body["sources"] if s["id"] == "maest")
    assert maest["pending"] == 0


def test_projection_empty_folder_covers_library_roots(client):
    """The default layout fits "my library" roots only (root b is external)."""
    r = client.get("/api/embeddings/projection")
    assert r.status_code == 200
    body = r.get_json()
    assert {p["path"] for p in body["positions"]} == {
        "a/one.mp3",
        "a/sub/deep.mp3",
        "a/shared/a_only.mp3",
    }
    assert "revision" in body


def test_projection_fit_scope_all_refits_with_external(client):
    """The old external-inclusive layout is now an explicit opt-in."""
    r = client.get("/api/embeddings/projection?include_external=1&fit_scope=all")
    assert r.status_code == 200
    body = r.get_json()
    assert {p["path"] for p in body["positions"]} == {
        "a/one.mp3",
        "a/sub/deep.mp3",
        "a/shared/a_only.mp3",
        "b/two.mp3",
        "b/shared/three.mp3",
        "b/shared/b_only.mp3",
    }


def test_projection_fit_scope_library_keeps_the_library_layout(client):
    """The default external scope places externals; it does not refit the map."""
    r = client.get("/api/embeddings/projection?include_external=1&fit_scope=library")
    assert r.status_code == 200
    body = r.get_json()
    # The fake cache has no feature rows, so the external roots cannot be placed;
    # the library tracks still come from the (unchanged) library fit.
    assert {p["path"] for p in body["positions"]} == {
        "a/one.mp3",
        "a/sub/deep.mp3",
        "a/shared/a_only.mp3",
    }


def test_projection_fit_scope_rejects_unknown_value(client):
    r = client.get("/api/embeddings/projection?fit_scope=everything")
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_projection_fit_scope_library_matches_default_revision(client):
    """The default and library fit scopes are the same chart (one revision)."""
    default = client.get("/api/embeddings/projection").get_json()
    scoped = client.get(
        "/api/embeddings/projection?include_external=1&fit_scope=library"
    ).get_json()
    assert scoped["revision"] == default["revision"]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?folder=a/sub",
        "?include_external=1",
        "?include_external=1&fit_scope=all",
        "?sources=clap",
        "?w=clap:1",
        "?dims=2",
        "?dims=3",
    ],
)
def test_projection_gauge_scope_is_the_configured_library_roots(
    fake_embeddings, client, query
):
    """Every production request — folder, external, source, weights, 2-D and the
    3-D reference/main pair — uses one workspace family, derived from the
    configured library roots rather than the selected paths."""
    fake_embeddings.calls.clear()
    r = client.get("/api/embeddings/projection" + query)
    assert r.status_code == 200
    assert fake_embeddings.calls
    assert {c["gauge_scope"] for c in fake_embeddings.calls} == {"library:a"}


def test_projection_genre_and_source_lenses_share_the_gauge_scope(
    roots, tmp_path, fake_embeddings
):
    client = _make_client(roots, tmp_path, fake_embeddings, genre_full_enabled=True)
    for query in ("?recipe=genre_full", "?recipe=genre_full&dims=3"):
        fake_embeddings.calls.clear()
        r = client.get("/api/embeddings/projection" + query)
        assert r.status_code == 200, r.get_json()
        assert {c["gauge_scope"] for c in fake_embeddings.calls} == {"library:a"}


def _status_and_projection_revisions(client, query: str) -> tuple[str, str]:
    status = client.get("/api/embeddings/status" + query)
    projection = client.get("/api/embeddings/projection" + query)
    assert status.status_code == 200, status.get_json()
    assert projection.status_code == 200, projection.get_json()
    return status.get_json()["layout_revision"], projection.get_json()["revision"]


def test_coordinate_revision_follows_the_configured_scope(roots, tmp_path):
    """The same visible slice under a different configured root set is a different
    frame: status and the projection route must agree, the prefetch must warm that
    same revision, and the fit cache must stay scope-independent."""
    client_a = _make_client(roots, tmp_path, _FakeEmbeddings(), library_roots=["a"])
    client_ab = _make_client(
        roots, tmp_path, _FakeEmbeddings(), library_roots=["a", "b"]
    )
    status_a, proj_a = _status_and_projection_revisions(client_a, "?folder=a")
    status_ab, proj_ab = _status_and_projection_revisions(client_ab, "?folder=a")
    assert status_a == proj_a
    assert status_ab == proj_ab
    assert status_a != status_ab

    # The prefetch path, through the same shared implementation the route uses.
    cache = FeatureCache(str(tmp_path / "prefetch.db"))
    versions = SourceVersions(
        clap=1,
        features=1,
        maest=1,
        rhythm=1,
        supervised=0,
        structure=1,
        maest_layer7=1,
        mert_l19=1,
    )
    for expected, library_roots in ((proj_a, ["a"]), (proj_ab, ["a", "b"])):
        _positions, revision = compute_lens_projection(
            [{"path": "a/one.mp3", "fingerprint": "fp-one.mp3"}],
            cache,
            projection_params_for_lens(FROZEN_SOURCES, None),
            embeddings_mod=_FakeEmbeddings(),
            batch_fetch_maps=batch_fetch_maps,
            layout_revision_for_projection=layout_revision_for_projection,
            source_versions=versions,
            embedding_version=1,
            user=False,
            library_root_ids=library_roots,
        )
        assert revision == expected


def test_list_mp3s_root_ids_limits_all_roots_walk(roots):
    assert len(list_mp3s("", roots, recursive=True)) == 6
    scoped = list_mp3s("", roots, recursive=True, root_ids=["a"])
    assert {os.path.basename(p) for p in scoped} == {
        "one.mp3",
        "deep.mp3",
        "a_only.mp3",
    }


def test_projection_with_empty_folder_is_empty_not_error(client):
    r = client.get("/api/embeddings/projection?folder=a/empty")
    assert r.status_code == 200
    assert r.get_json()["positions"] == []


def test_projection_dims3_returns_z_per_position(client):
    r = client.get("/api/embeddings/projection?dims=3")
    assert r.status_code == 200
    body = r.get_json()
    assert body["positions"]
    assert all("z" in p for p in body["positions"])
    # dims=3 owns its own revision, distinct from the 2-D default.
    default_rev = client.get("/api/embeddings/projection").get_json()["revision"]
    assert body["revision"] != default_rev


def test_projection_dims3_fits_all_visible_tracks(client):
    """dims=3 treats fit_scope as all: externals join the fit, not Nyström."""
    r = client.get("/api/embeddings/projection?dims=3&include_external=1")
    assert r.status_code == 200
    body = r.get_json()
    assert {p["path"] for p in body["positions"]} == {
        "a/one.mp3",
        "a/sub/deep.mp3",
        "a/shared/a_only.mp3",
        "b/two.mp3",
        "b/shared/three.mp3",
        "b/shared/b_only.mp3",
    }
    assert all("z" in p for p in body["positions"])


def test_projection_dims_bogus_is_400(client):
    r = client.get("/api/embeddings/projection?dims=bogus")
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_projection_empty_folder_jacobian_returns_no_chart(
    fake_embeddings, client, monkeypatch
):
    """A warm (or empty) layout with ``jacobian=1`` must stay a tuple response.

    The cache early-return used to hand the route a bare positions list, so the
    unpack raised and the request 500'd. Repeating the request hits the cached
    revision; an empty folder and a one-track folder both have no chart.
    """
    from backend.embeddings import landmark_projection as landmark_mod

    monkeypatch.setattr(landmark_mod, "landmark_model_for", lambda *a, **k: None)
    for _ in range(2):
        r = client.get("/api/embeddings/projection?folder=a/empty&jacobian=1")
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert body["positions"] == []
        assert body["jacobian"] == {}

    r = client.get("/api/embeddings/projection?folder=a/sub&jacobian=1")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert len(body["positions"]) == 1
    assert body["jacobian"] == {}


def test_projection_dims3_no_landmark_jacobian_returns_exact_and_empty(
    fake_embeddings, client, monkeypatch
):
    """3-D without a landmark chart returns the exact base layout with an empty
    Jacobian (weight-drag preview unavailable) instead of the 7 per-source Jacobian
    refits. The empty feature cache means no landmark chart can fit, so the route
    must NOT fall back to the per-source compute_projection refit."""
    from backend.embeddings import landmark_projection as landmark_mod

    monkeypatch.setattr(landmark_mod, "landmark_model_for", lambda *a, **k: None)
    fake_embeddings.calls.clear()
    r = client.get("/api/embeddings/projection?dims=3&jacobian=1")
    assert r.status_code == 200
    body = r.get_json()
    assert body["positions"]
    assert all("z" in p for p in body["positions"])
    # Exact base positions, but no per-source Jacobian columns: preview is
    # unavailable and the frontend holds the exact baseline.
    assert body["jacobian"] == {}
    assert set(body["positions"][0]) >= {"path", "x", "y", "z"}
    # Only the 2-D reference + the main 3-D fit ran; neither was a Jacobian refit.
    assert len(fake_embeddings.calls) == 2
    assert not any(c.get("with_jacobian") for c in fake_embeddings.calls)


# ─── 3-D gauge reference ─────────────────────────────────────


def _library_paths():
    return ["a/one.mp3", "a/shared/a_only.mp3", "a/sub/deep.mp3"]


def _all_paths():
    return _library_paths() + ["b/shared/b_only.mp3", "b/shared/three.mp3", "b/two.mp3"]


def test_projection_dims3_computes_2d_reference_first(fake_embeddings, client):
    """A dims=3 request runs the same-lens 2-D layout first and passes it as
    the alignment reference for the 3-D call."""
    fake_embeddings.calls.clear()
    r = client.get("/api/embeddings/projection?dims=3")
    assert r.status_code == 200
    assert r.get_json()["positions"]
    assert len(fake_embeddings.calls) == 2
    ref_call, main_call = fake_embeddings.calls
    assert ref_call["dims"] == 2
    assert not ref_call["has_align_ref"]
    # The reference uses the requested fit scope (default library): library only.
    assert ref_call["paths"] == _library_paths()
    assert main_call["dims"] == 3
    assert main_call["has_align_ref"]
    # The 3-D call always fits everything, and receives the 2-D positions.
    assert main_call["paths"] == _library_paths()
    assert [p["path"] for p in main_call["align_ref"]] == ref_call["paths"]
    assert all(set(p) == {"path", "x", "y"} for p in main_call["align_ref"])


def test_projection_dims3_reference_uses_raw_fit_scope(fake_embeddings, client):
    """The reference honours the requested fit_scope (library → external tracks
    placed, not fitted), while the 3-D call itself still fits everything."""
    fake_embeddings.calls.clear()
    r = client.get("/api/embeddings/projection?dims=3&include_external=1")
    assert r.status_code == 200
    assert r.get_json()["positions"]
    ref_call, main_call = fake_embeddings.calls
    # fit_scope=library: the fake cache cannot place the externals, so the
    # reference fit sees only the library tracks.
    assert ref_call["paths"] == _library_paths()
    assert main_call["paths"] == _all_paths()
    assert [p["path"] for p in main_call["align_ref"]] == ref_call["paths"]


def test_projection_dims3_reference_uses_only_fitted_tracks(
    fake_embeddings, client, monkeypatch
):
    """Externals placed by Nyström on the 2-D reference must not become the 3-D
    alignment target: placement is approximate, so only fitted rows are used."""
    from backend.embeddings import landmark_projection as landmark_mod

    class _Model:
        def __init__(self, paths):
            self.paths = paths

        def evaluate(self, _weights):
            return np.zeros((len(self.paths), 2), dtype=np.float32)

    def _model(revision, base_infos, feature_cache, base_positions, *, place_infos=None, **_kw):
        if not place_infos:
            return None
        paths = [p["path"] for p in base_infos] + [p["path"] for p in place_infos]
        return _Model(paths)

    monkeypatch.setattr(landmark_mod, "landmark_model_for", _model)
    fake_embeddings.calls.clear()
    r = client.get(
        "/api/embeddings/projection?dims=3&include_external=1&fit_scope=library"
    )
    assert r.status_code == 200
    ref_call, main_call = fake_embeddings.calls
    assert ref_call["paths"] == _library_paths()
    assert main_call["paths"] == _all_paths()
    # The reference contains the placed externals, but align_ref must not.
    assert [p["path"] for p in main_call["align_ref"]] == _library_paths()


def test_projection_dims3_reference_failure_still_returns_positions(
    fake_embeddings, client
):
    """A failed reference must never fail the 3-D request: it is logged and the
    3-D layout is served ungauged."""
    fake_embeddings.calls.clear()
    fake_embeddings.fail_on_dims2 = True
    r = client.get("/api/embeddings/projection?dims=3")
    assert r.status_code == 200
    body = r.get_json()
    assert body["positions"]
    assert all("z" in p for p in body["positions"])
    assert len(fake_embeddings.calls) == 2
    ref_call, main_call = fake_embeddings.calls
    assert ref_call["dims"] == 2
    assert main_call["dims"] == 3
    assert not main_call["has_align_ref"]
    assert main_call["align_ref"] is None


def test_projection_2d_makes_exactly_one_call(fake_embeddings, client):
    """The plain 2-D path is unchanged: a single projection, no reference."""
    fake_embeddings.calls.clear()
    r = client.get("/api/embeddings/projection")
    assert r.status_code == 200
    assert r.get_json()["positions"]
    assert len(fake_embeddings.calls) == 1
    call = fake_embeddings.calls[0]
    assert call["dims"] == 2
    assert not call["has_align_ref"]
    assert call["paths"] == _library_paths()
