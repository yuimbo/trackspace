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
        }
        # The search blueprint is registered with the existing deps (no new wiring).
        assert "/api/search" in {r.rule for r in app.url_map.iter_rules()}
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
