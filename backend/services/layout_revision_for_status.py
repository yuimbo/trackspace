from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.embedding_coverage import eligible_paths_for_projection
from backend.layout_revision import compute_layout_revision


def layout_revision_for_projection(
    infos: list[dict[str, Any]],
    maps: dict[str, dict[str, Any]],
    proj: Any | None,
    *,
    cache_versions: Sequence[int],
) -> str | None:
    """Compute status/projection layout revision when projection params are provided."""
    if proj is None:
        return None
    eligible = eligible_paths_for_projection(infos, maps, proj.sources)
    return compute_layout_revision(
        eligible_paths=eligible,
        method=proj.method,
        sources=proj.sources,
        feature_mask=[float(x) for x in proj.feature_mask],
        features_blend=proj.features_blend,
        folder_boost=proj.folder_boost,
        folder_depth_boost=proj.folder_depth_boost,
        context_tags=proj.tag_names,
        context_folders=proj.explicit_folders if proj.explicit_folders else None,
        scale_folders=proj.scale_folders,
        cache_versions=tuple(cache_versions),
    )
