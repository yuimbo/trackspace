from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from backend.embeddings.projection_config import (
    FROZEN_FEATURE_MASK,
    FROZEN_FOLDER_BOOST,
    FROZEN_FOLDER_DEPTH_BOOST,
    FROZEN_SCALE_FOLDERS,
    FROZEN_SOURCES,
)
from backend.services.request_validation import one_of
from backend.tuning import FROZEN_FEATURES_BLEND

#: Accepted projection methods. ``umap`` is the legacy alias for ``tsne``.
_METHODS: tuple[str, ...] = ("tsne", "pca", "umap")


@dataclass(frozen=True)
class ProjectionParams:
    method: str
    tag_names: list[str]
    explicit_folders: list[str]
    scale_folders: bool
    feature_mask: object  # np.ndarray
    features_blend: float
    folder_boost: float
    folder_depth_boost: float
    sources: tuple[str, ...]


def embedding_projection_query_fingerprint(req: Any) -> str:
    """Stable hash of query args that affect coverage counts or layout_revision.

    Projection hyperparameters are server-fixed, so optional client args are ignored.
    """
    skip = frozenset({"folder", "recursive", "models_only"})
    parts: list[str] = []
    for key in sorted(req.args.keys()):
        if key in skip:
            continue
        parts.append(f"{key}={req.args.get(key, '')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def parse_projection_params(
    req: Any,
    *,
    default_when_no_method: bool = False,
) -> ProjectionParams:
    """Parse method (``pca`` preview vs ``tsne``); all other fields are frozen.

    An unknown ``method`` is rejected with a structured 400 rather than silently
    falling back to t-SNE, so client bugs surface instead of being masked.
    """
    del default_when_no_method
    raw_method = (req.args.get("method") or "").strip().lower()
    if not raw_method:
        method = "tsne"
    else:
        one_of(raw_method, _METHODS, field="method")
        method = "tsne" if raw_method == "umap" else raw_method

    return ProjectionParams(
        method=method,
        tag_names=[],
        explicit_folders=[],
        scale_folders=FROZEN_SCALE_FOLDERS,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=FROZEN_FEATURES_BLEND,
        folder_boost=FROZEN_FOLDER_BOOST,
        folder_depth_boost=FROZEN_FOLDER_DEPTH_BOOST,
        sources=FROZEN_SOURCES,
    )
