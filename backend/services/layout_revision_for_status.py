from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.embeddings.coverage import CachedSourceMaps, eligible_paths_for_projection
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import FROZEN_SOURCES, REQUIRED_SOURCES


def layout_revision_for_projection(
    infos: list[dict[str, Any]],
    maps: CachedSourceMaps,
    proj: Any | None,
    *,
    cache_versions: Sequence[int],
) -> str | None:
    """Compute layout revision for the projection slice.

    *proj* carries the lens's sources and the client's source weights (``None`` =
    frozen defaults), both of which are part of the revision so a different lens or
    a re-weighting invalidates position caches.
    """
    sources = getattr(proj, "sources", None) or FROZEN_SOURCES
    required = getattr(proj, "required_sources", None) or REQUIRED_SOURCES
    eligible = eligible_paths_for_projection(infos, maps, sources, required)
    return compute_layout_revision(
        eligible_paths=eligible,
        cache_versions=tuple(cache_versions),
        weights=getattr(proj, "source_weights", None),
        sources=sources,
    )
