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

    *proj* carries the client's source weights (``None`` = frozen defaults), which
    are part of the revision so a re-weighting invalidates position caches.
    """
    eligible = eligible_paths_for_projection(
        infos, maps, FROZEN_SOURCES, REQUIRED_SOURCES
    )
    return compute_layout_revision(
        eligible_paths=eligible,
        cache_versions=tuple(cache_versions),
        weights=getattr(proj, "source_weights", None),
    )
