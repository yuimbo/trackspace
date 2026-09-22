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
    features: int
    maest: int = 0
    rhythm: int = 0
    #: Training-run id of the supervised head (0 = untrained/absent).
    supervised: int = 0
    #: Structure is metadata-only (not a projection source) but still a versioned
    #: cache row, so coverage reports it. It must never enter the layout revision.
    structure: int = 0


@dataclass(frozen=True)
class CachedSourceMaps:
    """Fingerprint → presence sets from one batch read per source.

    Values are ``{fingerprint: True}`` placeholders, not decoded rows: every
    consumer only tests membership.
    """

    clap: Mapping[str, object]
    features: Mapping[str, object]
    maest: Mapping[str, object] = field(default_factory=dict)
    maest_logits: Mapping[str, object] = field(default_factory=dict)
    rhythm: Mapping[str, object] = field(default_factory=dict)
    supervised: Mapping[str, object] = field(default_factory=dict)
    structure: Mapping[str, object] = field(default_factory=dict)

    def for_source(self, source: str) -> Mapping[str, object]:
        return {
            "clap": self.clap,
            "features": self.features,
            "maest": self.maest,
            "maest_logits": self.maest_logits,
            "rhythm": self.rhythm,
            "supervised": self.supervised,
            "structure": self.structure,
        }[source]


def batch_fetch_maps(
    feature_cache: FeatureCache,
    fingerprints: list[str],
    versions: SourceVersions,
) -> CachedSourceMaps:
    """Presence-only per-source maps: ``{fingerprint: True}``.

    Coverage and layout revision only ever test ``fp in map``; decoding the blobs
    to answer that moved ~328 MB per status request. The composite gather reads
    the vectors it needs through its own path.
    """
    presence = feature_cache.presence(
        {
            "clap": versions.clap,
            "features": versions.features,
            "maest": versions.maest,
            "maest_logits": versions.maest,
            "rhythm": versions.rhythm,
            "supervised": versions.supervised,
            "structure": versions.structure,
        },
        list(fingerprints),
    )
    return CachedSourceMaps(
        clap=presence.get("clap", {}),
        features=presence.get("features", {}),
        maest=presence.get("maest", {}),
        maest_logits=presence.get("maest_logits", {}),
        rhythm=presence.get("rhythm", {}),
        supervised=presence.get("supervised", {}),
        structure=presence.get("structure", {}),
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
    failed_tracks: int = 0,
) -> int:
    """Outstanding tracks for one source.

    *failed_tracks* are tracks whose stage job is terminally failed (retries
    exhausted). Nothing will run them again without an explicit retrigger, so counting
    them as pending would leave the bar short forever — and, because the map's
    "analysis ready" gate keys on these counts, would keep the frontend from ever
    fetching the real t-SNE layout.
    """
    return max(0, fingerprinted_tracks - embedded_tracks - failed_tracks)


def failed_track_count(
    infos: list[dict],
    maps: CachedSourceMaps,
    source: str,
    failed_fingerprints: set[str] | None,
) -> int:
    """Tracks whose *source* job terminally failed and have no cached row.

    Counting in *track* space (not unique fingerprints) matches
    :func:`pending_tracks_for_source`; a failure whose fingerprint is already
    cached via another path is not a gap and is ignored.
    """
    if not failed_fingerprints:
        return 0
    have = maps.for_source(source)
    return sum(
        1
        for t in infos
        if t.get("fingerprint") in failed_fingerprints
        and t["fingerprint"] not in have
    )


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
    checkable = [s for s in sources if s in ("clap", "features", "maest", "rhythm")]

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
    failed_by_source: Mapping[str, set[str]] | None = None,
) -> dict:
    """Stats for ``/api/embeddings/status`` (explicit + legacy field names).

    *failed_by_source* maps a source name to the fingerprints whose stage job
    terminally failed (see :func:`failed_track_count`); those tracks are reported
    as failed, not pending.
    """
    total = len(infos)
    fingerprinted = sum(1 for t in infos if t.get("fingerprint"))
    failed = failed_by_source or {}

    tracks_with_clap = tracks_with_fp_in_maps(infos, maps, "clap")
    tracks_with_features = tracks_with_fp_in_maps(infos, maps, "features")
    tracks_with_maest = tracks_with_fp_in_maps(infos, maps, "maest")
    tracks_with_rhythm = tracks_with_fp_in_maps(infos, maps, "rhythm")
    tracks_with_supervised = tracks_with_fp_in_maps(infos, maps, "supervised")
    tracks_with_structure = tracks_with_fp_in_maps(infos, maps, "structure")

    def pending(source: str, embedded: int) -> int:
        return pending_tracks_for_source(
            fingerprinted,
            embedded,
            failed_track_count(infos, maps, source, failed.get(source)),
        )

    pending_clap = pending("clap", tracks_with_clap)
    pending_features = pending("features", tracks_with_features)
    pending_maest = pending("maest", tracks_with_maest)
    pending_rhythm = pending("rhythm", tracks_with_rhythm)
    pending_supervised = pending("supervised", tracks_with_supervised)
    pending_structure = pending("structure", tracks_with_structure)

    failed_clap = failed_track_count(infos, maps, "clap", failed.get("clap"))
    failed_features = failed_track_count(infos, maps, "features", failed.get("features"))
    failed_maest = failed_track_count(infos, maps, "maest", failed.get("maest"))
    failed_rhythm = failed_track_count(infos, maps, "rhythm", failed.get("rhythm"))
    failed_supervised = failed_track_count(
        infos, maps, "supervised", failed.get("supervised")
    )
    failed_structure = failed_track_count(
        infos, maps, "structure", failed.get("structure")
    )

    return {
        "tracks_total": total,
        "tracks_with_fingerprint": fingerprinted,
        "tracks_with_clap": tracks_with_clap,
        "tracks_pending_clap": pending_clap,
        "tracks_failed_clap": failed_clap,
        "tracks_with_audio_features": tracks_with_features,
        "tracks_pending_audio_features": pending_features,
        "tracks_failed_audio_features": failed_features,
        "tracks_with_maest": tracks_with_maest,
        "tracks_pending_maest": pending_maest,
        "tracks_failed_maest": failed_maest,
        "tracks_with_rhythm": tracks_with_rhythm,
        "tracks_pending_rhythm": pending_rhythm,
        "tracks_failed_rhythm": failed_rhythm,
        "tracks_with_supervised": tracks_with_supervised,
        "tracks_pending_supervised": pending_supervised,
        "tracks_failed_supervised": failed_supervised,
        "tracks_with_structure": tracks_with_structure,
        "tracks_pending_structure": pending_structure,
        "tracks_failed_structure": failed_structure,
        "cache_versions": {
            "clap": versions.clap,
            "audio_features": versions.features,
            "maest": versions.maest,
            "rhythm": versions.rhythm,
            "supervised": versions.supervised,
            "structure": versions.structure,
        },
        # Legacy names (same values as track-space counts above)
        "total": total,
        "fingerprinted": fingerprinted,
        "embedded": tracks_with_clap,
        "pending": pending_clap,
        "features_extracted": tracks_with_features,
        "features_pending": pending_features,
        "embedding_version": versions.clap,
    }
