"""Canonical job-kind names for the analysis pipeline.

Kept in a leaf module so enqueuers, workers, and HTTP routes all agree on the
same strings without importing each other.

Pipeline order (each stage's output feeds the next):

    scan      → ID3 tags + Chromaprint fingerprint (cheap, CPU, parallel)
    features  → classical DSP / rhythm descriptors (CPU, librosa + madmom)
    structure → intro/drop/break timestamps (CPU, whole-track loudness envelope)
    maest     → MAEST embedding + 519 genre logits (GPU, batched)
    clap      → CLAP audio embedding (GPU, batched; text encoder shared with layout)
    supervise → folder-supervised projection head over cached vectors (CPU, singleton)
    cluster   → kNN graph + multi-resolution Leiden (whole-library, singleton)
    layout    → warm the per-lens 2-D projections (CPU/GPU, singleton per lens)
"""

from __future__ import annotations

KIND_SCAN = "scan"
KIND_FEATURES = "features"
KIND_STRUCTURE = "structure"
KIND_MAEST = "maest"
KIND_CLAP = "clap"
KIND_SUPERVISED = "supervise"
KIND_CLUSTER = "cluster"
KIND_LAYOUT = "layout"

#: Ordered by pipeline position — status UIs render them in this order.
JOB_KINDS: tuple[str, ...] = (
    KIND_SCAN,
    KIND_FEATURES,
    KIND_STRUCTURE,
    KIND_MAEST,
    KIND_CLAP,
    KIND_SUPERVISED,
    KIND_CLUSTER,
    KIND_LAYOUT,
)

_LABELS: dict[str, str] = {
    KIND_SCAN: "Library scan",
    KIND_FEATURES: "Audio features",
    KIND_STRUCTURE: "Track structure",
    KIND_MAEST: "MAEST analysis",
    KIND_CLAP: "CLAP analysis",
    KIND_SUPERVISED: "Folder supervision",
    KIND_CLUSTER: "Clustering",
    KIND_LAYOUT: "View layouts",
}


def kind_label(kind: str) -> str:
    """Human-readable name for a job kind."""
    return _LABELS.get(kind, kind)
