"""Stable digest of the current library slice that has a full embedding row.

Revision changes only when eligible tracks (fingerprint + all frozen sources in
cache), embedding cache generations, or the source weights change — not when other
client "options" change, since those remain fixed in ``projection_config``.

Source weights are part of the payload because the UI can now re-weight the layout:
two requests that differ only in weights must hash differently, or the
revision-keyed position caches would serve one configuration's coordinates for the
other.

The configured gauge scope is part of the payload too: it selects the reference
frame a layout is aligned into, and it is independent of the visible tracks, so the
same slice under different configured library roots must not share coordinates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .lens_gauge import GAUGE_VERSION2
from .projection_config import PROJECTION_RECIPE_VERSION

# Bump when folder semantic *algorithm* changes (centroid + PCA extras, thresholds).
FOLDER_SEMANTIC_BASIS_VERSION = 1

# Bump when the 3-D alignment (head-on onto the lens's 2-D layout) changes; it
# invalidates persisted 3-D layouts.
GAUGE_3D_VERSION = 3

#: ``GAUGE_VERSION2`` (lens_gauge) is the shared 2-D gauge / normalisation
#: generation. It invalidates persisted 2-D layouts and participates in every
#: family key.


def compute_layout_revision(
    *,
    eligible_paths: list[str],
    cache_versions: Sequence[int],
    folder_semantic_basis_version: int = FOLDER_SEMANTIC_BASIS_VERSION,
    projection_recipe_version: int = PROJECTION_RECIPE_VERSION,
    weights: Mapping[str, float] | None = None,
    sources: Sequence[str] | None = None,
    gauge_scope: str | None = None,
    dims: int = 2,
    gauge_2d_version: int = GAUGE_VERSION2,
    gauge_3d_version: int = GAUGE_3D_VERSION,
    recipe_id: str | None = None,
    tsne_n_iter: int | None = None,
    tsne_early_exaggeration_iter: int | None = None,
) -> str:
    """Return a hex SHA-256 of the canonical payload (lexically sorted JSON keys).

    *recipe_id* and the openTSNE iteration budget are only included when they
    represent an opt-in deviation from the default recipe — the default recipe at
    its default 50/0 budget omits them, so unrelated recipes keep sharing hashes.
    """
    payload: dict[str, Any] = {
        "cache_versions": list(cache_versions),
        "eligible_paths": sorted(eligible_paths),
        "folder_semantic_basis_version": int(folder_semantic_basis_version),
        "projection_recipe_version": int(projection_recipe_version),
        # Lenses project different source subsets from the same library slice, so
        # two lenses with the same eligible paths and weights must not share a
        # revision — the revision-keyed position cache would serve one lens's
        # coordinates for the other.
        "sources": sorted(sources or ()),
        # The configured gauge scope decides which reference frame the coordinates
        # are aligned into. It is not derivable from ``eligible_paths``: the same
        # tracks under different configured library roots must not share
        # coordinates, or one workspace would be served positions in the other's
        # frame. An absent scope means the frame comes from the path-derived
        # fallback, which ``eligible_paths`` already determines.
        "gauge_scope": str(gauge_scope or ""),
        # The gauge/normalisation semantics decide the stored coordinates for a
        # slice, so a semantics change must invalidate every persisted layout,
        # 2-D included.
        "gauge": int(gauge_2d_version),
        # Formatted to 6 significant digits so float repr noise (0.1 vs 0.100000…)
        # cannot fork an otherwise identical configuration into two revisions.
        "source_weights": {
            str(k): f"{float(v):.6g}" for k, v in sorted((weights or {}).items())
        },
    }
    if recipe_id:
        payload["recipe"] = str(recipe_id)
    if tsne_n_iter is not None:
        payload["tsne_n_iter"] = int(tsne_n_iter)
    if tsne_early_exaggeration_iter is not None:
        payload["tsne_early_exaggeration_iter"] = int(tsne_early_exaggeration_iter)
    # Only the opt-in 3-D layout carries ``dims`` and its gauge version.
    if dims != 2:
        payload["dims"] = int(dims)
        payload["gauge3d"] = int(gauge_3d_version)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
