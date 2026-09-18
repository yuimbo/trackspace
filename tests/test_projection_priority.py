"""The background layout prefetch yields the projection slot to user requests.

A t-SNE fit is seconds long and holds the shared accelerator lock, so without this
gate a lens switch could wait behind several prefetched fits — the "never-ending
computing layout" case.
"""

from __future__ import annotations

import threading
import time
import types

from backend.services import projection_priority as pp


def _run(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def test_prefetch_yields_to_a_user_projection():
    user_entered = threading.Event()
    user_release = threading.Event()
    prefetch_entered = threading.Event()

    def user():
        with pp.user_projection():
            user_entered.set()
            user_release.wait(2)

    def prefetch():
        user_entered.wait(2)
        with pp.prefetch_projection():
            prefetch_entered.set()

    u = _run(user)
    p = _run(prefetch)
    assert user_entered.wait(2)
    # The prefetch must not start while the user holds the slot.
    time.sleep(0.1)
    assert not prefetch_entered.is_set()
    user_release.set()
    p.join(2)
    u.join(2)
    assert prefetch_entered.is_set()


def test_user_waits_for_an_active_prefetch():
    prefetch_entered = threading.Event()
    prefetch_release = threading.Event()
    user_entered = threading.Event()

    def prefetch():
        with pp.prefetch_projection():
            prefetch_entered.set()
            prefetch_release.wait(2)

    def user():
        prefetch_entered.wait(2)
        with pp.user_projection():
            user_entered.set()

    p = _run(prefetch)
    u = _run(user)
    assert prefetch_entered.wait(2)
    time.sleep(0.1)
    # The user waits for the one in-flight fit, then takes the slot.
    assert not user_entered.is_set()
    assert pp.user_projection_active()
    prefetch_release.set()
    u.join(2)
    p.join(2)
    assert user_entered.is_set()


def test_active_flag_clears_after_release():
    with pp.user_projection():
        assert pp.user_projection_active()
    assert not pp.user_projection_active()


def test_warmed_lens_short_circuits_before_the_slot():
    """A cached lens returns without waiting for (or taking) the projection slot."""
    from backend.embeddings import layout as layout_mod
    from backend.services import lens_layout as ll

    layout_mod._revision_layout_cache.clear()
    layout_mod._revision_layout_cache["rev-1"] = [
        {"path": "a.mp3", "x": 0.1, "y": 0.2}
    ]

    class _Proj:
        method = "tsne"
        tag_names: list = []
        explicit_folders: list = []
        scale_folders = False
        folder_boost = 3.0
        folder_depth_boost = 1.5
        sources = ("clap",)
        feature_mask = None
        features_blend = 0.42
        source_weights = None
        required_sources = ("clap",)

    class _Emb:
        def compute_projection(self, *a, **k):
            raise AssertionError("a cached revision must not refit")

    positions, revision = ll.compute_lens_projection(
        [{"path": "a.mp3", "fingerprint": "fp"}],
        object(),
        _Proj(),
        embeddings_mod=_Emb(),
        batch_fetch_maps=lambda *a, **k: {},
        layout_revision_for_projection=lambda *a, **k: "rev-1",
        source_versions=types.SimpleNamespace(
            clap=1, features=1, maest=1, rhythm=1, supervised=0
        ),
        embedding_version=1,
    )
    assert revision == "rev-1"
    assert positions == [{"path": "a.mp3", "x": 0.1, "y": 0.2}]
