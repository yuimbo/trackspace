"""CLAP / EffNet / librosa features, FeatureCache, coverage, and 2-D layout.

Public symbols are loaded lazily so ``import backend.embeddings.audio_features`` does not
import the layout / projection stack.
"""

from __future__ import annotations

_PUBLIC = frozenset({
    "CLAP_SR",
    "EMBEDDING_VERSION",
    "_load_audio_segments",
    "batch_ensure_embeddings",
    "compute_projection",
    "compute_umap",
    "generate_embedding",
    "generate_text_embeddings",
    "is_model_ready",
    "load_model",
    "parse_audio_feature_mask",
    "parse_features_blend",
    "parse_folder_boost",
    "parse_folder_depth_boost",
    "reset_projection_backend_cache",
})


def __getattr__(name: str):
    if name in _PUBLIC:
        from . import layout

        return getattr(layout, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_PUBLIC)
