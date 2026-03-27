from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from backend import embeddings


@dataclass(frozen=True)
class ProjectionParams:
    method: str
    tag_names: list[str]
    explicit_folders: list[str]
    scale_folders: bool
    feature_mask: object  # np.ndarray after parse
    features_blend: float
    folder_boost: float
    folder_depth_boost: float
    sources: tuple[str, ...]


def embedding_projection_query_fingerprint(req: Any) -> str:
    """Stable hash of query args that affect coverage counts or layout_revision."""
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
) -> ProjectionParams | None:
    """Parse projection query args shared by status + projection routes."""
    raw_method = req.args.get("method", "")
    if not raw_method and not default_when_no_method:
        return None
    method = raw_method or "umap"
    if method not in ("umap", "pca", "tsne"):
        method = "umap"

    raw_tags = req.args.get("context_tags", "")
    raw_folders = req.args.get("context_folders", "")
    raw_sources = req.args.get("sources", "clap")
    tag_names = [s.strip() for s in raw_tags.split(",") if s.strip()] if raw_tags else []
    explicit_folders = (
        [s.strip() for s in raw_folders.split(",") if s.strip()] if raw_folders else []
    )
    scale_folders = req.args.get("scale_folders", "") == "1"
    if not scale_folders and explicit_folders:
        scale_folders = True
    feature_mask = embeddings.parse_audio_feature_mask(req.args.get("feature_mask"))
    features_blend = embeddings.parse_features_blend(req.args.get("features_blend"))
    folder_boost = embeddings.parse_folder_boost(req.args.get("folder_boost"))
    folder_depth_boost = embeddings.parse_folder_depth_boost(
        req.args.get("folder_depth_boost"),
    )
    sources = tuple(
        s.strip() for s in raw_sources.split(",") if s.strip() in ("clap", "effnet", "features")
    )
    if not sources:
        sources = ("clap",)

    return ProjectionParams(
        method=method,
        tag_names=tag_names,
        explicit_folders=explicit_folders,
        scale_folders=scale_folders,
        feature_mask=feature_mask,
        features_blend=features_blend,
        folder_boost=folder_boost,
        folder_depth_boost=folder_depth_boost,
        sources=sources,
    )
