"""One lens's projection, shared by the HTTP route and the prefetch job.

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


def versions_tuple(versions: Any, recipe_id: str | None = None) -> tuple[int, ...]:
    """Cache generations that feed the layout revision hash.

    Order matches ``embeddings/layout_revision.compute_layout_revision``'s
    ``cache_versions`` payload; ``structure`` is deliberately excluded (it is not a
    projection source). An opt-in genre recipe additionally appends the genre block
    generations (layer-7, MERT-L19) so a version bump invalidates the genre layout;
    the default recipe's tuple is byte-identical to before.
    """
    base = (
        versions.clap,
        versions.features,
        versions.maest,
        versions.rhythm,
        versions.supervised,
    )
    if recipe_id == "genre_full":
        return base + (versions.maest_layer7, versions.mert_l19)
    return base


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
    align_ref: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return ``(positions, revision)`` for one projection lens.

    ``proj`` is a :class:`~backend.services.projection_request.ProjectionParams`
    (from the route or :func:`projection_params_for_lens`); everything else is the
    wired projection stack.  ``user=False`` marks the background prefetch, which yields
    the projection slot to any waiting user request (see ``projection_priority``).
    ``align_ref`` is the same lens's 2-D layout: the 3-D gauge pins a 3-D layout's
    head-on (x,y) onto it. Only forwarded when set — the test fakes may not accept
    the keyword.
    """
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    maps = batch_fetch_maps(feature_cache, fps, source_versions)
    revision = layout_revision_for_projection(
        infos, maps, proj,
        cache_versions=versions_tuple(source_versions, getattr(proj, "recipe_id", None)),
    )
    # An already-warmed lens must not wait behind a background fit for another.
    if proj.method == "tsne":
        from backend.embeddings.layout import revision_layout_cached

        hit = revision_layout_cached(revision, feature_cache)
        if hit is not None:
            return hit, revision
    slot = user_projection() if user else prefetch_projection()
    with slot:
        # `align_ref` is only forwarded when set: it exists for the 3-D gauge, and
        # the route/job fakes in the tests do not accept it.
        extra = {} if align_ref is None else {"align_ref": align_ref}
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
            dims=proj.dims,
            recipe_id=getattr(proj, "recipe_id", None),
            tsne_n_iter=getattr(proj, "tsne_n_iter", None),
            tsne_early_exaggeration_iter=getattr(proj, "tsne_early_exaggeration_iter", None),
            on_progress=on_progress,
            **extra,
        )
    return positions, revision
