"""One lens's projection, shared by the HTTP route and the prefetch job.

The route and the ``layout`` job must compute the *same* ``layout_revision`` so the
job warms the exact cache entry the client later reads. Keeping the gather →
revision → project sequence here, with the route and the job both calling it, is what
guarantees that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from backend.services.projection_priority import (
    prefetch_projection,
    user_projection,
)
from backend.services.projection_request import effective_source_weights


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


def gauge_scope_for(library_root_ids: Sequence[str] | None) -> str | None:
    """Canonical gauge scope for the configured library roots.

    Every production request with configured library roots shares one workspace
    family, even when the visible tracks are a folder subset or only external
    tracks (a disjoint layout is then left unaligned by the overlap guard, never
    forced into a wrong frame). The scope deliberately comes from the configured
    roots, not the requested folder: deriving it from selected paths would give a
    folder request a different family from the whole-library request.

    ``None`` when no library roots are configured, leaving
    ``compute_projection``'s root-prefix fallback in charge.
    """
    ids = sorted({str(r) for r in (library_root_ids or ()) if str(r)})
    if not ids:
        return None
    return "library:" + "|".join(ids)


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
    library_root_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return ``(positions, revision)`` for one projection lens.

    ``proj`` is a :class:`~backend.services.projection_request.ProjectionParams`
    (from the route or :func:`projection_params_for_lens`); everything else is the
    wired projection stack.  ``user=False`` marks the background prefetch, which yields
    the projection slot to any waiting user request (see ``projection_priority``).
    ``align_ref`` is the same lens's 2-D layout: the 3-D gauge pins a 3-D layout's
    head-on (x,y) onto it.

    ``library_root_ids`` is the configured "my library" root id set; it becomes the
    stable gauge scope shared by every lens and by the 2-D/3-D pair, so a lens
    switch or a folder subset never re-anchors the map. It is also part of the
    coordinate revision, so changing the configured roots cannot reuse coordinates
    from their frame (the fit cache stays keyed on the vector config, so a re-scope
    reuses the fitted embedding). Keywords are only forwarded when set — the test
    fakes may not accept them.
    """
    scope = gauge_scope_for(library_root_ids)
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    maps = batch_fetch_maps(feature_cache, fps, source_versions)
    revision = layout_revision_for_projection(
        infos, maps, proj,
        cache_versions=versions_tuple(source_versions, getattr(proj, "recipe_id", None)),
        gauge_scope=scope,
    )
    # An already-warmed lens must not wait behind a background fit for another.
    if proj.method == "tsne":
        from backend.embeddings.layout import revision_layout_cached

        hit = revision_layout_cached(
            revision,
            feature_cache,
            gauge_scope=scope,
            dims=getattr(proj, "dims", 2),
        )
        if hit is not None:
            return hit, revision
    slot = user_projection() if user else prefetch_projection()
    with slot:
        # Optional keywords are only forwarded when set: they exist for the
        # shared gauge, and the route/job fakes in the tests need not accept them.
        extra = {} if align_ref is None else {"align_ref": align_ref}
        if scope is not None:
            extra["gauge_scope"] = scope
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
            source_weights=effective_source_weights(proj),
            required_sources=proj.required_sources,
            dims=proj.dims,
            recipe_id=getattr(proj, "recipe_id", None),
            tsne_n_iter=getattr(proj, "tsne_n_iter", None),
            tsne_early_exaggeration_iter=getattr(proj, "tsne_early_exaggeration_iter", None),
            on_progress=on_progress,
            **extra,
        )
    return positions, revision
