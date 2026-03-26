"""Single definition of per-track embedding coverage and generation work lists.

Cache rows are keyed by *fingerprint*; API stats and job queues are defined in
*track* space so duplicate audio does not inflate ``pending`` counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.feature_cache import FeatureCache


@dataclass(frozen=True)
class SourceVersions:
    clap: int
    effnet: int
    features: int


@dataclass(frozen=True)
class CachedSourceMaps:
    """Fingerprint → row maps from one batch read per source."""

    clap: dict[str, object]
    effnet: dict[str, object]
    features: dict[str, object]


def batch_fetch_maps(
    feature_cache: FeatureCache,
    fingerprints: list[str],
    versions: SourceVersions,
) -> CachedSourceMaps:
    return CachedSourceMaps(
        clap=feature_cache.get_all_embeddings(fingerprints, version=versions.clap),
        effnet=feature_cache.get_all_effnet_embeddings(fingerprints, version=versions.effnet),
        features=feature_cache.get_all_audio_features(fingerprints, version=versions.features),
    )


def tracks_with_fp_in_maps(
    infos: list[dict],
    maps: CachedSourceMaps,
    source: str,
) -> int:
    key = {"clap": maps.clap, "effnet": maps.effnet, "features": maps.features}[source]
    have = frozenset(key.keys())
    return sum(1 for t in infos if t.get("fingerprint") and t["fingerprint"] in have)


def pending_tracks_for_source(
    fingerprinted_tracks: int,
    embedded_tracks: int,
) -> int:
    return fingerprinted_tracks - embedded_tracks


def eligible_paths_for_projection(
    infos: list[dict],
    maps: CachedSourceMaps,
    sources: tuple[str, ...],
) -> list[str]:
    """Paths that have cache data for every enabled *sources* entry."""
    out: list[str] = []
    for t in infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        if "clap" in sources and fp not in maps.clap:
            continue
        if "effnet" in sources and fp not in maps.effnet:
            continue
        if "features" in sources and fp not in maps.features:
            continue
        out.append(t["path"])
    out.sort()
    return out


def build_generation_work(
    infos: list[dict],
    feature_cache: FeatureCache,
    sources: list[str],
    versions: SourceVersions,
) -> list[tuple[dict, list[str]]]:
    """(track_info, needed_source_names) for tracks missing at least one source."""
    work: list[tuple[dict, list[str]]] = []
    for t in infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        needed: list[str] = []
        if "clap" in sources and not feature_cache.has_embedding(fp, version=versions.clap):
            needed.append("clap")
        if "effnet" in sources and not feature_cache.has_effnet_embedding(
            fp, version=versions.effnet
        ):
            needed.append("effnet")
        if "features" in sources and not feature_cache.has_audio_features(
            fp, version=versions.features
        ):
            needed.append("features")
        if needed:
            work.append((t, needed))
    return work


def coverage_payload(
    infos: list[dict],
    maps: CachedSourceMaps,
    versions: SourceVersions,
) -> dict:
    """Stats for ``/api/embeddings/status`` (explicit + legacy field names)."""
    total = len(infos)
    fingerprinted = sum(1 for t in infos if t.get("fingerprint"))

    tracks_with_clap = tracks_with_fp_in_maps(infos, maps, "clap")
    tracks_with_effnet = tracks_with_fp_in_maps(infos, maps, "effnet")
    tracks_with_features = tracks_with_fp_in_maps(infos, maps, "features")

    pending_clap = pending_tracks_for_source(fingerprinted, tracks_with_clap)
    pending_effnet = pending_tracks_for_source(fingerprinted, tracks_with_effnet)
    pending_features = pending_tracks_for_source(fingerprinted, tracks_with_features)

    return {
        "tracks_total": total,
        "tracks_with_fingerprint": fingerprinted,
        "tracks_with_clap": tracks_with_clap,
        "tracks_pending_clap": pending_clap,
        "tracks_with_effnet": tracks_with_effnet,
        "tracks_pending_effnet": pending_effnet,
        "tracks_with_audio_features": tracks_with_features,
        "tracks_pending_audio_features": pending_features,
        "cache_versions": {
            "clap": versions.clap,
            "effnet": versions.effnet,
            "audio_features": versions.features,
        },
        # Legacy names (same values as track-space counts above)
        "total": total,
        "fingerprinted": fingerprinted,
        "embedded": tracks_with_clap,
        "pending": pending_clap,
        "effnet_embedded": tracks_with_effnet,
        "effnet_pending": pending_effnet,
        "features_extracted": tracks_with_features,
        "features_pending": pending_features,
        "embedding_version": versions.clap,
    }
