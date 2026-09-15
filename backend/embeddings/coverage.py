"""Single definition of per-track embedding coverage and generation work lists.

Cache rows are keyed by *fingerprint*; API stats and job queues are defined in
*track* space so duplicate audio does not inflate ``pending`` counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .feature_cache import FeatureCache


@dataclass(frozen=True)
class SourceVersions:
    clap: int
    effnet: int
    features: int
    maest: int = 0
    rhythm: int = 0


@dataclass(frozen=True)
class CachedSourceMaps:
    """Fingerprint → row maps from one batch read per source."""

    clap: Mapping[str, object]
    effnet: Mapping[str, object]
    features: Mapping[str, object]
    maest: Mapping[str, object] = field(default_factory=dict)
    maest_logits: Mapping[str, object] = field(default_factory=dict)
    rhythm: Mapping[str, object] = field(default_factory=dict)

    def for_source(self, source: str) -> Mapping[str, object]:
        return {
            "clap": self.clap,
            "effnet": self.effnet,
            "features": self.features,
            "maest": self.maest,
            "maest_logits": self.maest_logits,
            "rhythm": self.rhythm,
        }[source]


def batch_fetch_maps(
    feature_cache: FeatureCache,
    fingerprints: list[str],
    versions: SourceVersions,
) -> CachedSourceMaps:
    return CachedSourceMaps(
        clap=feature_cache.get_all_embeddings(fingerprints, version=versions.clap),
        effnet=feature_cache.get_all_effnet_embeddings(fingerprints, version=versions.effnet),
        features=feature_cache.get_all_audio_features(fingerprints, version=versions.features),
        maest=feature_cache.get_all_maest_embeddings(fingerprints, version=versions.maest),
        maest_logits=feature_cache.get_all_maest_logits(fingerprints, version=versions.maest),
        rhythm=feature_cache.get_all_rhythm_features(fingerprints, version=versions.rhythm),
    )


def tracks_with_fp_in_maps(
    infos: list[dict],
    maps: CachedSourceMaps,
    source: str,
) -> int:
    have = frozenset(maps.for_source(source).keys())
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
    required: tuple[str, ...] | None = None,
) -> list[str]:
    """Paths that can be placed on the map.

    A track needs cache data for every *required* source (defaulting to all of
    *sources*). Optional sources contribute a neutral zero row when missing, so
    demanding all of them would needlessly hide partially analysed tracks.
    """
    need = tuple(s for s in (required if required is not None else sources) if s in sources)
    out: list[str] = []
    for t in infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        if all(fp in maps.for_source(s) for s in need):
            out.append(t["path"])
    out.sort()
    return out


def build_generation_work(
    infos: list[dict],
    feature_cache: FeatureCache,
    sources: list[str],
    versions: SourceVersions,
) -> list[tuple[dict, list[str]]]:
    """(track_info, needed_source_names) for tracks missing at least one source.

    Uses one batched read per source rather than a ``has_*`` call per track per
    source — at ~7.4k tracks the per-track form issued tens of thousands of
    SQLite round-trips just to decide what to queue.
    """
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    maps = batch_fetch_maps(feature_cache, fps, versions)

    # "maest_logits" rides along with "maest" — one model pass produces both.
    checkable = [s for s in sources if s in ("clap", "effnet", "features", "maest", "rhythm")]

    work: list[tuple[dict, list[str]]] = []
    for t in infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        needed = [s for s in checkable if fp not in maps.for_source(s)]
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
    tracks_with_maest = tracks_with_fp_in_maps(infos, maps, "maest")
    tracks_with_rhythm = tracks_with_fp_in_maps(infos, maps, "rhythm")

    pending_clap = pending_tracks_for_source(fingerprinted, tracks_with_clap)
    pending_effnet = pending_tracks_for_source(fingerprinted, tracks_with_effnet)
    pending_features = pending_tracks_for_source(fingerprinted, tracks_with_features)
    pending_maest = pending_tracks_for_source(fingerprinted, tracks_with_maest)
    pending_rhythm = pending_tracks_for_source(fingerprinted, tracks_with_rhythm)

    return {
        "tracks_total": total,
        "tracks_with_fingerprint": fingerprinted,
        "tracks_with_clap": tracks_with_clap,
        "tracks_pending_clap": pending_clap,
        "tracks_with_effnet": tracks_with_effnet,
        "tracks_pending_effnet": pending_effnet,
        "tracks_with_audio_features": tracks_with_features,
        "tracks_pending_audio_features": pending_features,
        "tracks_with_maest": tracks_with_maest,
        "tracks_pending_maest": pending_maest,
        "tracks_with_rhythm": tracks_with_rhythm,
        "tracks_pending_rhythm": pending_rhythm,
        "cache_versions": {
            "clap": versions.clap,
            "effnet": versions.effnet,
            "audio_features": versions.features,
            "maest": versions.maest,
            "rhythm": versions.rhythm,
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
