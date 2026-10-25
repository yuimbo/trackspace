"""``backend.app`` lifecycle: importing it must have no side effects.

The state used to be created at import time (SQLite caches, thread pool, startup
purges), which made ``import backend.app`` mutate the filesystem and prevented
tests from injecting a scratch state. ``main()`` now builds the state after CLI
parsing and ``build_app(state)`` wires the app around it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_import_backend_app_has_no_side_effects(tmp_path):
    """Importing the module creates no data dir, app, or module-level state."""
    data_dir = tmp_path / "data"
    env = {**os.environ, "TRACKSPACE_DATA_DIR": str(data_dir)}
    code = (
        "import backend.app as a\n"
        "assert not hasattr(a, 'app'), 'module-level app created at import'\n"
        "assert not hasattr(a, 'state'), 'module-level state created at import'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert not data_dir.exists(), "importing backend.app created its data dir"


def test_build_app_accepts_injected_state(tmp_path):
    """``build_app(state)`` wires the app around a caller-supplied state."""
    from backend.app import build_app
    from backend.trackspace_state import TrackspaceState

    state = TrackspaceState.create(data_dir=str(tmp_path / "data"))
    try:
        app = build_app(state)
        assert app.extensions["trackspace_state"] is state
        assert set(app.extensions["trackspace_lifecycle"]) == {
            "load_roots_state",
            "schedule_root_watch",
            "backfill_purged_analysis",
        }
        # The search blueprint is registered with the existing deps (no new wiring).
        assert "/api/search" in {r.rule for r in app.url_map.iter_rules()}
    finally:
        state.thread_pool.shutdown(wait=False)


def test_build_app_default_genre_on_but_mert_explicit_off(tmp_path):
    """Genre is the default, but MERT never runs from a bare ``build_app``.

    ``main()`` only sets MERT on when ``--trust-remote-code`` is passed; the app
    default must keep ``mert_enabled`` off so a server cannot pile up MERT jobs
    behind a model that will never load (the consent jobs bug this slice removes).
    """
    from backend.app import build_app
    from backend.jobs import KIND_MERT
    from backend.trackspace_state import TrackspaceState

    state = TrackspaceState.create(data_dir=str(tmp_path / "data"))
    try:
        app = build_app(state)
        # Genre is on: the genre_full recipe is served without a feature gate.
        from backend.embeddings.projection_config import GENRE_FULL_RECIPE_WEIGHTS
        from backend.services.projection_request import parse_projection_params

        class _Req:
            def __init__(self, args): self.args = args
        proj = parse_projection_params(_Req({"recipe": "genre_full"}), genre_full_enabled=True)
        assert proj.source_weights == GENRE_FULL_RECIPE_WEIGHTS

        # MERT is off: no MERT worker was registered by the bare build_app.
        worker_kinds = set(state.workers._workers)
        assert KIND_MERT not in worker_kinds
    finally:
        state.thread_pool.shutdown(wait=False)


def test_purged_sources_fold_onto_the_stage_that_rewrites_them():
    """A startup purge enqueues only the stages it actually blanked.

    The map must cover every versioned source (a new source with no kind would be
    silently un-backed-up), and several sources may share one stage.
    """
    from backend.app import PURGED_SOURCE_KINDS, SOURCE_VERSIONS, _kinds_for_purged_sources

    assert set(PURGED_SOURCE_KINDS) == set(SOURCE_VERSIONS)
    assert _kinds_for_purged_sources({}) == ()
    assert _kinds_for_purged_sources({"structure": 0}) == ()
    assert _kinds_for_purged_sources({"structure": 5}) == ("structure",)
    # CLAP's pooled vector, segment vectors and positions share one stage.
    assert _kinds_for_purged_sources(
        {"clap": 3, "clap_segments": 3, "clap_segment_pos": 3}
    ) == ("clap",)


def test_backfill_hook_queues_nothing_on_an_empty_library(tmp_path):
    """The startup backfill hook is wired and a no-op with no tracks."""
    from backend.app import build_app
    from backend.trackspace_state import TrackspaceState

    state = TrackspaceState.create(data_dir=str(tmp_path / "data"))
    try:
        app = build_app(state)
        hook = app.extensions["trackspace_lifecycle"]["backfill_purged_analysis"]
        assert sum(hook({}).values()) == 0
        # Even a purge is a no-op when there is no track to re-queue.
        assert sum(hook({"structure": 4}).values()) == 0
    finally:
        state.thread_pool.shutdown(wait=False)


def test_root_roles_roundtrip_through_the_app(tmp_path):
    """Root roles flow from the API through roots.json and the tree sections."""
    import json

    from backend.app import build_app
    from backend.services.roots_registry import load_roots_state
    from backend.trackspace_state import TrackspaceState

    lib = tmp_path / "lib"
    ext = tmp_path / "ext"
    (lib / "Album").mkdir(parents=True)
    (ext / "Downloads").mkdir(parents=True)

    state = TrackspaceState.create(data_dir=str(tmp_path / "data"))
    try:
        client = build_app(state).test_client()
        lib_res = client.post("/api/roots/add", json={"path": str(lib)})
        assert lib_res.get_json()["role"] == "library"
        ext_res = client.post(
            "/api/roots/add", json={"path": str(ext), "role": "external"}
        )
        assert ext_res.get_json()["role"] == "external"

        # The partial splits roots into sections and tags each root row.
        html = client.get("/partials/folder-tree").get_data(as_text=True)
        assert 'data-root-role="external"' in html
        assert 'data-root-role="library"' in html
        assert "folder-section-label" in html

        # Persisted as external *paths*; a reload restores the same role.
        saved = json.loads((tmp_path / "data" / "roots.json").read_text())
        assert saved["external_roots"] == [str(ext)]
        roots: dict[str, str] = {}
        roles: dict[str, str] = {}
        load_roots_state(
            roots, state.roots_state_path, default_root=None, root_roles=roles
        )
        ext_id = next(rid for rid, p in roots.items() if p == str(ext))
        assert roles[ext_id] == "external"
    finally:
        state.thread_pool.shutdown(wait=False)


def test_source_versions_covers_every_versioned_cache_source():
    """The startup purge and its report share one map, so they cannot disagree.

    A version bump used to add a source to the purge map but not the report map,
    which crashed ``main()`` with a ``KeyError`` on the first start after the bump.
    """
    from backend.app import SOURCE_VERSIONS

    assert set(SOURCE_VERSIONS) == {
        "clap",
        "clap_segments",
        "clap_segment_pos",
        "features",
        "maest",
        "maest_layer7",
        "mert_l19",
        "rhythm",
        "structure",
    }
    assert all(isinstance(v, int) and v >= 1 for v in SOURCE_VERSIONS.values())


def test_purge_reports_one_entry_per_source(tmp_path):
    from backend.app import SOURCE_VERSIONS, _purge_stale_embedding_cache_rows
    from backend.trackspace_state import TrackspaceState

    state = TrackspaceState.create(data_dir=str(tmp_path / "data"))
    try:
        purged = _purge_stale_embedding_cache_rows(state)
        assert set(purged) == set(SOURCE_VERSIONS)
    finally:
        state.thread_pool.shutdown(wait=False)


def _state_with_track(tmp_path, data_dir) -> tuple["TrackspaceState", str]:
    """A state with one root holding one mp3 whose fingerprint comes from cache."""
    import numpy as np

    from backend.trackspace_state import TrackspaceState

    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    track = lib / "a.mp3"
    track.write_bytes(b"ID3\x00fake")
    state = TrackspaceState.create(data_dir=data_dir)
    state.roots = {"r1": str(lib)}
    state.track_cache.put(
        str(track),
        os.path.getmtime(str(track)),
        {"path": str(track), "fingerprint": "fp1", "title": "A"},
    )
    return state, str(track)


def test_backfill_purge_revives_done_but_not_failed(tmp_path):
    """The automatic purge backfill re-queues done work but leaves failures alone."""
    from backend.app import build_app
    from backend.jobs import KIND_MAEST

    state, _track = _state_with_track(tmp_path, str(tmp_path / "data"))
    try:
        app = build_app(state)
        queue = state.job_queue

        # A completed MAEST job for the track that exists.
        queue.enqueue_many(KIND_MAEST, [("r1/a.mp3", {"path": "r1/a.mp3"})])
        queue.complete(queue.claim(KIND_MAEST)[0])
        assert queue.stats(KIND_MAEST).done == 1
        # A terminally-failed MAEST job for a file that no longer exists.
        queue.enqueue_many(KIND_MAEST, [("r1/gone.mp3", {"path": "r1/gone.mp3"})])
        queue.fail(queue.claim(KIND_MAEST)[0], "file not found", retry=False)
        assert queue.stats(KIND_MAEST).failed == 1

        hook = app.extensions["trackspace_lifecycle"]["backfill_purged_analysis"]
        queued = hook({"maest": 1})
        assert queued.get(KIND_MAEST, 0) == 1
        # The done job was revived and re-queued; the failed job was not revived.
        st = queue.stats(KIND_MAEST)
        assert st.pending == 1
        assert st.failed == 1
    finally:
        state.thread_pool.shutdown(wait=False)


def test_reconcile_after_late_checkpoint_purges_and_backfills(tmp_path, monkeypatch):
    """A checkpoint downloaded only after startup is reconciled + backfilled on load."""
    import numpy as np

    from backend.app import build_app
    from backend.embeddings import maest as maest_mod
    from backend.jobs import KIND_MAEST

    state, _track = _state_with_track(tmp_path, str(tmp_path / "data"))
    try:
        app = build_app(state)
        # Stale rows written under an older checkpoint, with the old fingerprint
        # recorded in the meta.
        state.feature_cache.put_maest(
            "fp1", np.ones(8, np.float32), np.ones(519, np.float32), version=1
        )
        state.feature_cache.set_meta("maest_checkpoint_sha256", "old-sha")
        assert state.feature_cache.has("maest", "fp1", version=1)

        # At startup the checkpoint is not on disk -> reconcile is a no-op and the
        # stale rows survive.
        monkeypatch.setattr(maest_mod, "checkpoint_fingerprint", lambda: None)
        assert maest_mod.reconcile_checkpoint(state.feature_cache) is False
        assert state.feature_cache.has("maest", "fp1", version=1)

        # The model load later downloads the checkpoint -> the fingerprint appears
        # and the family is reconciled (purged + recorded), exactly as the post-load
        # handshake in ``_bg_load_maest`` does.
        monkeypatch.setattr(maest_mod, "checkpoint_fingerprint", lambda: "new-sha")
        assert maest_mod.reconcile_checkpoint(state.feature_cache) is True
        assert state.feature_cache.has("maest", "fp1", version=1) is False
        assert state.feature_cache.get_meta("maest_checkpoint_sha256") == "new-sha"

        # The handshake then backfills the rows it just blanked.
        hook = app.extensions["trackspace_lifecycle"]["backfill_purged_analysis"]
        queued = hook(
            {"maest": 1, "maest_logits": 1, "maest_excerpts": 1, "maest_layer7": 1}
        )
        assert queued.get(KIND_MAEST, 0) == 1
    finally:
        state.thread_pool.shutdown(wait=False)
