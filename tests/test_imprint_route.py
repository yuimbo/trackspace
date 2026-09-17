"""The /api/supervised/imprint/* routes and the /api/roots picker listing.

The real roots registry and virtual-path helpers run over a scratch directory, so
the ephemeral-root add/remove and the path round trip are exercised for real. The
analysis enqueue is a recorder: the queue's own behaviour is covered by
``test_job_queue``, and here only the call shape (which sources, which folder)
matters.
"""

from __future__ import annotations

from flask import Flask

from backend.factory import _register_error_handlers
from backend.routes.api_imprint import create_api_imprint_blueprint
from backend.services.roots_registry import add_root, remove_root, save_roots_state
from backend.services.virtual_paths import virtual_from_abs


class _StubCache:
    """Nothing analysed: the readiness count is all-pending."""

    def get_all(self, *_a, **_k):
        return {}


class _StubDeps:
    feature_cache = _StubCache()
    maest_version = 1
    rhythm_version = 1
    clap_version = 0


def _imprint_client(tmp_path, *, infos, roots=None, temp_roots=None):
    roots = roots if roots is not None else {}
    temp_roots = temp_roots if temp_roots is not None else set()
    calls: list[tuple[int, tuple[str, ...]]] = []

    def enqueue(_queue, infos_arg, *, deps, kinds):
        calls.append((len(infos_arg), tuple(kinds)))
        return {k: len(infos_arg) for k in kinds}

    bp = create_api_imprint_blueprint(
        analysis_deps=_StubDeps(),
        roots=roots,
        temp_roots=temp_roots,
        add_root=lambda p: add_root(p, roots),
        remove_root=lambda rid: remove_root(rid, roots),
        schedule_root_watch=lambda _rid: None,
        unschedule_root_watch=lambda _rid: None,
        save_roots_state=lambda: save_roots_state(
            roots, str(tmp_path / "roots.json"), temp_roots=temp_roots
        ),
        build_track_infos_fn=lambda _f, _r: infos,
        enqueue_library_work_fn=enqueue,
        virtual_from_abs=lambda a: virtual_from_abs(a, roots),
        job_queue=object(),
        workers=None,
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client(), roots, temp_roots, calls


def test_imprint_analyze_registers_an_ephemeral_root(tmp_path):
    external = tmp_path / "Downloads"
    external.mkdir()
    infos = [{"path": "d/t.mp3", "fingerprint": "fp"}]
    client, roots, temp_roots, calls = _imprint_client(tmp_path, infos=infos)

    body = client.post(
        "/api/supervised/imprint/analyze", json={"path": str(external)}
    ).get_json()
    assert body["ok"] is True
    assert body["added"] is True
    assert body["tracks"] == 1
    root_id = body["root_id"]
    assert temp_roots == {root_id}
    assert roots[root_id] == str(external)
    # Analysis is queued for exactly the sources the head consumes.
    assert calls == [(1, ("features", "maest", "clap"))]

    finish = client.post(
        "/api/supervised/imprint/finish", json={"root_id": root_id}
    ).get_json()
    assert finish == {"ok": True, "removed": True}
    assert root_id not in roots and root_id not in temp_roots


def test_imprint_analyze_refuses_a_folder_containing_library_roots(tmp_path):
    library = tmp_path / "Music"
    library.mkdir()
    roots: dict = {}
    add_root(str(library), roots)
    client, roots, temp_roots, _ = _imprint_client(
        tmp_path, infos=[], roots=roots
    )

    r = client.post("/api/supervised/imprint/analyze", json={"path": str(tmp_path)})
    assert r.status_code == 400
    assert "library roots" in r.get_json()["error"]
    assert temp_roots == set()


def test_imprint_finish_leaves_a_library_folder_alone(tmp_path):
    library = tmp_path / "Music"
    library.mkdir()
    roots: dict = {}
    add_root(str(library), roots)
    rid = next(iter(roots))
    client, roots, temp_roots, _ = _imprint_client(
        tmp_path, infos=[], roots=roots
    )

    body = client.post(
        "/api/supervised/imprint/finish", json={"root_id": rid}
    ).get_json()
    assert body == {"ok": True, "removed": False}
    assert rid in roots


def test_imprint_status_requires_a_folder(tmp_path):
    client, *_ = _imprint_client(tmp_path, infos=[])
    r = client.get("/api/supervised/imprint/status")
    assert r.status_code == 400
    assert "folder" in r.get_json()["error"]


def test_imprint_temp_roots_lists_and_discards_stale_roots(tmp_path):
    external = tmp_path / "Downloads"
    external.mkdir()
    infos = [{"path": "d/t.mp3", "fingerprint": "fp"}]
    client, roots, temp_roots, _ = _imprint_client(tmp_path, infos=infos)
    body = client.post(
        "/api/supervised/imprint/analyze", json={"path": str(external)}
    ).get_json()
    root_id = body["root_id"]

    listed = client.get("/api/supervised/imprint/temp_roots").get_json()
    assert listed == {
        "ok": True,
        "roots": [{"root_id": root_id, "path": str(external)}],
    }

    discarded = client.post("/api/supervised/imprint/discard").get_json()
    assert discarded == {"ok": True, "discarded": 1}
    assert root_id not in roots and root_id not in temp_roots
    assert client.get("/api/supervised/imprint/temp_roots").get_json()["roots"] == []
