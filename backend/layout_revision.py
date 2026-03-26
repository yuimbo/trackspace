"""Stable digest of everything that defines a 2D embedding layout for the library.

Used so the client can skip redundant ``/api/embeddings/projection`` calls when
the server-side result would be identical."""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Bump when folder semantic basis changes (centroid + optional extras, thresholds).
FOLDER_SEMANTIC_BASIS_VERSION = 1


def compute_layout_revision(
    *,
    eligible_paths: list[str],
    method: str,
    sources: tuple[str, ...],
    feature_mask: list[float],
    features_blend: float,
    folder_boost: float,
    folder_depth_boost: float,
    context_tags: list[str],
    context_folders: list[str] | None,
    scale_folders: bool,
    cache_versions: tuple[int, int, int],
    folder_semantic_basis_version: int = FOLDER_SEMANTIC_BASIS_VERSION,
) -> str:
    """Return a hex SHA-256 of the canonical payload (lexically sorted JSON keys)."""
    folders_norm: list[str] | None
    if context_folders is None:
        folders_norm = None
    else:
        folders_norm = sorted(context_folders)

    payload: dict[str, Any] = {
        "cache_versions": list(cache_versions),
        "context_tags": sorted(context_tags),
        "context_folders": folders_norm,
        "eligible_paths": sorted(eligible_paths),
        "features_blend": float(features_blend),
        "folder_boost": float(folder_boost),
        "folder_depth_boost": float(folder_depth_boost),
        "method": method,
        "scale_folders": bool(scale_folders),
        "folder_semantic_basis_version": int(folder_semantic_basis_version),
        "sources": list(sources),
        "feature_mask": [float(x) for x in feature_mask],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
