"""Stable digest of the current library slice that has a full embedding row.

Revision changes only when eligible tracks (fingerprint + all frozen sources in
cache) or embedding cache generations change — not when client “options”
change, since those are fixed in ``projection_config``."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .projection_config import PROJECTION_RECIPE_VERSION

# Bump when folder semantic *algorithm* changes (centroid + PCA extras, thresholds).
FOLDER_SEMANTIC_BASIS_VERSION = 1


def compute_layout_revision(
    *,
    eligible_paths: list[str],
    cache_versions: tuple[int, int, int],
    folder_semantic_basis_version: int = FOLDER_SEMANTIC_BASIS_VERSION,
    projection_recipe_version: int = PROJECTION_RECIPE_VERSION,
) -> str:
    """Return a hex SHA-256 of the canonical payload (lexically sorted JSON keys)."""
    payload: dict[str, Any] = {
        "cache_versions": list(cache_versions),
        "eligible_paths": sorted(eligible_paths),
        "folder_semantic_basis_version": int(folder_semantic_basis_version),
        "projection_recipe_version": int(projection_recipe_version),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
