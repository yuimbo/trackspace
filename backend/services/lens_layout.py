"""One lens's 2-D projection, shared by the HTTP route and the prefetch job.

The route and the ``layout`` job must compute the *same* ``layout_revision`` so the
job warms the exact cache entry the client later reads. Keeping the gather →
revision → project sequence here, with the route and the job both calling it, is what
guarantees that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.embeddings.projection_config import FROZEN_SOURCE_WEIGHTS
from backend.services.projection_priority import (
    prefetch_projection,
    user_projection,
)


def versions_tuple(versions: Any) -> tuple[int, ...]:
    """Cache generations that feed the layout revision hash.

    Order matches ``embeddings/layout_revision.compute_layout_revision``'s
    ``cache_versions`` payload; ``structure`` is deliberately excluded (it is not a
    projection source).
    """
    return (
        versions.clap,
        versions.features,
        versions.maest,
        versions.rhythm,
        versions.supervised,
    )


def compute_lens_projection(
    infos: list[dict[str, Any]],
    feature_cache: Any,
    proj: Any,
    *,
    embeddings_mod: Any,
    batch_fetch_maps: Callable[..., Any],
    layout_revision_for_projection: Callable[..., str | None],
    source_versions: Any,
    embedding_version: int,
    user: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return ``(positions, revision)`` for one projection lens.

    ``proj`` is a :class:`~backend.services.projection_request.ProjectionParams`
    (from the route or :func:`projection_params_for_lens`); everything else is the
    wired projection stack.  ``user=False`` marks the background prefetch, which yields
    the projection slot to any waiting user request (see ``projection_priority``).
    """
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    maps = batch_fetch_maps(feature_cache, fps, source_versions)
    revision = layout_revision_for_projection(
        infos, maps, proj, cache_versions=versions_tuple(source_versions)
    )
    # An already-warmed lens must not wait behind a background fit for another.
    if proj.method == "tsne":
        from backend.embeddings.layout import revision_layout_cached

        hit = revision_layout_cached(revision)
        if hit is not None:
            return hit, revision
    slot = user_projection() if user else prefetch_projection()
    with slot:
        positions = embeddings_mod.compute_projection(
            infos,
            feature_cache,
            version=embedding_version,
            method=proj.method,
            context_tags=proj.tag_names,
            context_folders=proj.explicit_folders if proj.explicit_folders else None,
            scale_folders=proj.scale_folders,
            folder_boost=proj.folder_boost,
            folder_depth_boost=proj.folder_depth_boost,
            sources=proj.sources,
            feature_mask=proj.feature_mask,
            features_blend=proj.features_blend,
            layout_revision=revision,
            source_weights=proj.source_weights or FROZEN_SOURCE_WEIGHTS,
            required_sources=proj.required_sources,
            on_progress=on_progress,
        )
    return positions, revision
