"""Tests for embedding coverage / layout revision helpers (no heavy ML)."""

from backend.embeddings.coverage import (
    CachedSourceMaps,
    SourceVersions,
    coverage_payload,
    eligible_paths_for_projection,
)
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import PROJECTION_RECIPE_VERSION


def _source(payload, source):
    """One source row from the status payload, by id."""
    return next(s for s in payload["sources"] if s["id"] == source)


def test_coverage_payload_lists_sources_in_display_order():
    maps = CachedSourceMaps(clap={"fp1": object()}, features={})
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    p = coverage_payload(infos, maps, SourceVersions(2, 1, 1))
    assert [s["id"] for s in p["sources"]] == [
        "features",
        "rhythm",
        "structure",
        "maest",
        "clap",
        "supervised",
    ]
    assert _source(p, "clap")["label"] == "CLAP"


def test_coverage_genre_full_rows_gated_by_enablement():
    """genre_full blocks appear only when their stage is wired.

    maest_layer7 when genre_full is enabled; mert_l19 only when the MERT worker is
    actually enabled (genre_full AND not --no-mert). A disabled block must not show
    as permanently-pending.
    """
    maps = CachedSourceMaps(clap={"fp1": object()}, features={})
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]

    # Disabled: neither row appears.
    p = coverage_payload(infos, maps, SourceVersions(2, 1, 1))
    ids = [s["id"] for s in p["sources"]]
    assert "maest_layer7" not in ids and "mert_l19" not in ids

    # genre_full on but --no-mert: layer-7 shown, MERT hidden.
    p = coverage_payload(
        infos,
        maps,
        SourceVersions(2, 1, 1, maest_layer7=1, mert_l19=1),
        genre_full_enabled=True,
        mert_enabled=False,
    )
    ids = [s["id"] for s in p["sources"]]
    assert "maest_layer7" in ids
    assert "mert_l19" not in ids

    # Both on: both shown, with presence counted.
    maps2 = CachedSourceMaps(
        clap={"fp1": object()},
        features={},
        maest_layer7={"fp1": object()},
        mert_l19={},
    )
    p = coverage_payload(
        infos,
        maps2,
        SourceVersions(2, 1, 1, maest_layer7=1, mert_l19=1),
        genre_full_enabled=True,
        mert_enabled=True,
    )
    assert _source(p, "maest_layer7")["done"] == 1
    assert _source(p, "mert_l19")["done"] == 0
    assert _source(p, "mert_l19")["pending"] == 1


def test_coverage_payload_counts_tracks_with_duplicate_fingerprints():
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        features={},
    )
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp1"},
        {"path": "c.mp3", "fingerprint": "fp2"},
    ]
    p = coverage_payload(infos, maps, SourceVersions(2, 1, 1))
    assert p["tracks_with_fingerprint"] == 3
    assert _source(p, "clap")["done"] == 2
    assert _source(p, "clap")["pending"] == 1


def test_coverage_payload_reports_structure():
    """Structure is metadata-only but still versioned, so the coverage panel needs
    its own count — this is how a missing section timestamp becomes visible."""
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        features={"fp1": object()},
        structure={"fp1": object()},
    )
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    p = coverage_payload(infos, maps, SourceVersions(2, 1, 1, structure=3))
    assert _source(p, "structure")["done"] == 1
    assert _source(p, "structure")["pending"] == 1
    assert p["cache_versions"]["structure"] == 3


def test_coverage_pending_excludes_terminal_failures():
    """A terminally-failed stage is failed, not pending.

    Pending must reach zero once everything is either analysed or failed, or the
    frontend's "analysis ready" gate never opens and the map stays on the PCA
    preview instead of fetching the real t-SNE layout.
    """
    maps = CachedSourceMaps(clap={}, features={})
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    p = coverage_payload(
        infos,
        maps,
        SourceVersions(2, 1, 1),
        failed_by_source={"clap": {"fp1"}, "features": {"fp1"}},
    )
    assert _source(p, "clap")["pending"] == 1
    assert _source(p, "clap")["failed"] == 1
    assert _source(p, "features")["pending"] == 1
    assert _source(p, "features")["failed"] == 1


def test_coverage_failure_covered_by_duplicate_is_not_a_gap():
    """A failed path whose fingerprint is cached via another path is not failed."""
    maps = CachedSourceMaps(clap={"fp1": object()}, features={})
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp1"},
    ]
    p = coverage_payload(
        infos,
        maps,
        SourceVersions(2, 1, 1),
        failed_by_source={"clap": {"fp1"}},
    )
    assert _source(p, "clap")["pending"] == 0
    assert _source(p, "clap")["failed"] == 0


def test_eligible_paths_requires_all_sources():
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        features={},
    )
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    assert eligible_paths_for_projection(infos, maps, ("clap",)) == ["a.mp3"]
    assert eligible_paths_for_projection(infos, maps, ("clap", "features")) == []


def test_layout_revision_stable_for_same_inputs():
    r1 = compute_layout_revision(
        eligible_paths=["b.mp3", "a.mp3"],
        cache_versions=(2, 1, 1),
    )
    r2 = compute_layout_revision(
        eligible_paths=["a.mp3", "b.mp3"],
        cache_versions=(2, 1, 1),
    )
    assert r1 == r2


def _base_revision_kwargs():
    return dict(
        eligible_paths=["a.mp3", "b.mp3"],
        cache_versions=(2, 1, 1),
    )


def test_layout_revision_changes_when_cache_versions_change():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**kw)
    r_b = compute_layout_revision(**{**kw, "cache_versions": (3, 1, 1)})
    assert r_a != r_b


def test_layout_revision_changes_when_recipe_version_changes():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**kw)
    r_b = compute_layout_revision(
        **{**kw, "projection_recipe_version": PROJECTION_RECIPE_VERSION + 1},
    )
    assert r_a != r_b


def test_layout_revision_changes_with_source_weights():
    """A re-weighting must invalidate position caches, or the revision-keyed
    layout cache would serve one configuration's coordinates for another."""
    kw = _base_revision_kwargs()
    r_default = compute_layout_revision(**kw)
    r_custom = compute_layout_revision(**{**kw, "weights": {"rhythm": 0.5}})
    assert r_default != r_custom


def test_layout_revision_changes_with_sources():
    """Different lenses share the same eligible paths and weights; without the
    source set in the payload a CLAP lens could serve the combined lens's coords."""
    kw = _base_revision_kwargs()
    r_combined = compute_layout_revision(**{**kw, "sources": ("maest_logits", "maest")})
    r_clap = compute_layout_revision(**{**kw, "sources": ("clap",)})
    assert r_combined != r_clap


def test_layout_revision_changes_with_gauge_scope_but_not_when_absent():
    """The configured gauge scope selects the reference frame, and it is not
    derivable from the eligible paths: one slice under two root configurations
    must never share coordinates. No scope keeps the pre-scope hash."""
    kw = _base_revision_kwargs()
    r_unscoped = compute_layout_revision(**kw)
    assert r_unscoped == compute_layout_revision(**{**kw, "gauge_scope": None})
    r_a = compute_layout_revision(**{**kw, "gauge_scope": "library:a"})
    r_ab = compute_layout_revision(**{**kw, "gauge_scope": "library:a|b"})
    assert r_a not in (r_unscoped, r_ab)
    assert r_ab != r_unscoped


def test_layout_revision_dims3_differs_and_2d_matches_default():
    """dims=3 must fork the revision (3-D caches must never serve 2-D coords);
    dims=2 must be byte-identical to omitting the parameter, so existing 2-D
    persisted cache rows stay valid."""
    kw = _base_revision_kwargs()
    r_default = compute_layout_revision(**kw)
    r_2d = compute_layout_revision(**{**kw, "dims": 2})
    r_3d = compute_layout_revision(**{**kw, "dims": 3})
    assert r_default == r_2d
    assert r_3d != r_default


def test_layout_revision_3d_gauges_with_gauge3d_version():
    """Bumping the 3-D alignment version invalidates persisted 3-D layouts
    only: the 2-D revision must stay byte-identical."""
    from backend.embeddings.layout_revision import GAUGE_3D_VERSION

    kw = _base_revision_kwargs()
    r_2d = compute_layout_revision(**{**kw, "dims": 2})
    r_3d = compute_layout_revision(**{**kw, "dims": 3})
    assert r_3d != r_2d
    r_3d_new = compute_layout_revision(
        **{**kw, "dims": 3, "gauge_3d_version": GAUGE_3D_VERSION + 1}
    )
    r_2d_new = compute_layout_revision(
        **{**kw, "dims": 2, "gauge_3d_version": GAUGE_3D_VERSION + 1}
    )
    assert r_3d_new != r_3d
    assert r_2d_new == r_2d


def test_layout_revision_includes_the_shared_gauge_version_for_both_dims():
    """A gauge/normalisation change must invalidate 2-D *and* 3-D layouts."""
    from backend.embeddings.layout_revision import GAUGE_VERSION2

    kw = _base_revision_kwargs()
    r_2d = compute_layout_revision(**{**kw, "dims": 2})
    r_3d = compute_layout_revision(**{**kw, "dims": 3})
    assert compute_layout_revision(
        **{**kw, "dims": 2, "gauge_2d_version": GAUGE_VERSION2 + 1}
    ) != r_2d
    assert compute_layout_revision(
        **{**kw, "dims": 3, "gauge_2d_version": GAUGE_VERSION2 + 1}
    ) != r_3d


def test_layout_revision_stable_for_equivalent_weights():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**{**kw, "weights": {"rhythm": 0.5, "maest": 0.25}})
    # Key order and float repr noise must not fork the revision.
    r_b = compute_layout_revision(**{**kw, "weights": {"maest": 0.25, "rhythm": 0.5000000001}})
    assert r_a == r_b
