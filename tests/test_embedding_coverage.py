"""Tests for embedding coverage / layout revision helpers (no heavy ML)."""

from backend.embedding_coverage import (
    CachedSourceMaps,
    SourceVersions,
    coverage_payload,
    eligible_paths_for_projection,
)
from backend.layout_revision import compute_layout_revision


def test_coverage_payload_counts_tracks_with_duplicate_fingerprints():
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        effnet={"fp1": object()},
        features={},
    )
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp1"},
        {"path": "c.mp3", "fingerprint": "fp2"},
    ]
    p = coverage_payload(infos, maps, SourceVersions(2, 1, 1))
    assert p["tracks_with_fingerprint"] == 3
    assert p["tracks_with_clap"] == 2
    assert p["tracks_pending_clap"] == 1
    assert p["embedded"] == 2
    assert p["pending"] == 1


def test_eligible_paths_requires_all_sources():
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        effnet={},
        features={},
    )
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    assert eligible_paths_for_projection(infos, maps, ("clap",)) == ["a.mp3"]
    assert eligible_paths_for_projection(infos, maps, ("clap", "effnet")) == []


def test_layout_revision_stable_for_same_inputs():
    r1 = compute_layout_revision(
        eligible_paths=["b.mp3", "a.mp3"],
        method="tsne",
        sources=("clap", "effnet"),
        feature_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        features_blend=0.42,
        folder_boost=3.0,
        folder_depth_boost=1.5,
        context_tags=["energy", "mood"],
        context_folders=None,
        scale_folders=True,
        cache_versions=(2, 1, 1),
    )
    r2 = compute_layout_revision(
        eligible_paths=["a.mp3", "b.mp3"],
        method="tsne",
        sources=("clap", "effnet"),
        feature_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        features_blend=0.42,
        folder_boost=3.0,
        folder_depth_boost=1.5,
        context_tags=["mood", "energy"],
        context_folders=None,
        scale_folders=True,
        cache_versions=(2, 1, 1),
    )
    assert r1 == r2


def _base_revision_kwargs():
    return dict(
        eligible_paths=["a.mp3", "b.mp3"],
        method="tsne",
        sources=("clap", "effnet"),
        feature_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        features_blend=0.42,
        folder_boost=3.0,
        folder_depth_boost=1.5,
        context_tags=["mood"],
        context_folders=None,
        scale_folders=True,
        cache_versions=(2, 1, 1),
    )


def test_layout_revision_changes_when_cache_versions_change():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**kw)
    r_b = compute_layout_revision(**{**kw, "cache_versions": (3, 1, 1)})
    assert r_a != r_b


def test_layout_revision_changes_when_feature_mask_changes():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**kw)
    alt_mask = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    r_b = compute_layout_revision(**{**kw, "feature_mask": alt_mask})
    assert r_a != r_b


def test_layout_revision_changes_when_features_blend_changes():
    kw = _base_revision_kwargs()
    r_a = compute_layout_revision(**kw)
    r_b = compute_layout_revision(**{**kw, "features_blend": 0.9})
    assert r_a != r_b
