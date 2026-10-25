from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.embeddings.coverage import CachedSourceMaps, eligible_paths_for_projection
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import (
    FROZEN_SOURCES,
    REQUIRED_SOURCES,
    recipe_for,
)


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
    a re-weighting invalidates position caches. An opt-in recipe id and a non-default
    openTSNE iteration budget are encoded as deviations, so the default recipe's
    revision is byte-identical to before. The ``genre_full`` recipe is the exception:
    it encodes its *effective* budget absolutely (see the body) so its 100/100
    default never aliases the legacy 500/250 fit/layout under the same revision.
    """
    sources = getattr(proj, "sources", None) or FROZEN_SOURCES
    required = getattr(proj, "required_sources", None) or REQUIRED_SOURCES
    eligible = eligible_paths_for_projection(infos, maps, sources, required)

    recipe_id = getattr(proj, "recipe_id", None)
    recipe = recipe_for(recipe_id)
    # ``None`` = use the recipe default. Use an explicit ``is None`` check, never
    # truthiness: ``tsne_early_exaggeration_iter=0`` (disable early exaggeration)
    # is a real override, and ``0 or recipe.default`` would silently swallow it.
    eff_n_iter = getattr(proj, "tsne_n_iter", None)
    if eff_n_iter is None:
        eff_n_iter = recipe.tsne_n_iter
    eff_early = getattr(proj, "tsne_early_exaggeration_iter", None)
    if eff_early is None:
        eff_early = recipe.tsne_early_exaggeration_iter
    # The genre recipe encodes its *effective* budget absolutely: its default moved
    # from 500/250 to 100/100, and the legacy 500/250 default revision carried no
    # budget fields, so an absolute encoding is what stops the new 100/100 default
    # from hashing to (and reading) the legacy fit/layout under the same revision.
    # Every other recipe encodes only a deviation, keeping the default recipe's
    # revision byte-identical to before.
    is_genre = recipe_id == "genre_full"
    tsne_n_enc = eff_n_iter if (is_genre or eff_n_iter != recipe.tsne_n_iter) else None
    tsne_early_enc = (
        eff_early
        if (is_genre or eff_early != recipe.tsne_early_exaggeration_iter)
        else None
    )

    return compute_layout_revision(
        eligible_paths=eligible,
        cache_versions=tuple(cache_versions),
        weights=getattr(proj, "source_weights", None),
        sources=sources,
        dims=getattr(proj, "dims", 2),
        recipe_id=recipe_id if recipe_id != "default" else None,
        tsne_n_iter=tsne_n_enc,
        tsne_early_exaggeration_iter=tsne_early_enc,
    )
