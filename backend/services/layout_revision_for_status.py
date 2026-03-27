from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.embeddings.coverage import CachedSourceMaps, eligible_paths_for_projection
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import FROZEN_SOURCES


def layout_revision_for_projection(
    infos: list[dict[str, Any]],
    maps: CachedSourceMaps,
    proj: Any | None,
    *,
    cache_versions: Sequence[int],
) -> str | None:
    """Compute layout revision (frozen sources; *proj* may be None — ignored)."""
    del proj
    eligible = eligible_paths_for_projection(infos, maps, FROZEN_SOURCES)
    return compute_layout_revision(
        eligible_paths=eligible,
        cache_versions=tuple(cache_versions),
    )
