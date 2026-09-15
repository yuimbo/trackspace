"""Stable digest of the current library slice that has a full embedding row.

Revision changes only when eligible tracks (fingerprint + all frozen sources in
cache), embedding cache generations, or the source weights change — not when other
client "options" change, since those remain fixed in ``projection_config``.

Source weights are part of the payload because the UI can now re-weight the layout:
two requests that differ only in weights must hash differently, or the
revision-keyed position caches would serve one configuration's coordinates for the
other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .projection_config import PROJECTION_RECIPE_VERSION

# Bump when folder semantic *algorithm* changes (centroid + PCA extras, thresholds).
FOLDER_SEMANTIC_BASIS_VERSION = 1


def compute_layout_revision(
    *,
    eligible_paths: list[str],
    cache_versions: Sequence[int],
    folder_semantic_basis_version: int = FOLDER_SEMANTIC_BASIS_VERSION,
    projection_recipe_version: int = PROJECTION_RECIPE_VERSION,
    weights: Mapping[str, float] | None = None,
) -> str:
    """Return a hex SHA-256 of the canonical payload (lexically sorted JSON keys)."""
    payload: dict[str, Any] = {
        "cache_versions": list(cache_versions),
        "eligible_paths": sorted(eligible_paths),
        "folder_semantic_basis_version": int(folder_semantic_basis_version),
        "projection_recipe_version": int(projection_recipe_version),
        # Formatted to 6 significant digits so float repr noise (0.1 vs 0.100000…)
        # cannot fork an otherwise identical configuration into two revisions.
        "source_weights": {
            str(k): f"{float(v):.6g}" for k, v in sorted((weights or {}).items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
