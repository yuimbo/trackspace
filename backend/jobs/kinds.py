"""Canonical job-kind names for the analysis pipeline.

Kept in a leaf module so enqueuers, workers, and HTTP routes all agree on the
same strings without importing each other.

Pipeline order (each stage's output feeds the next):

    scan      → ID3 tags + Chromaprint fingerprint (cheap, CPU, parallel)
    features  → classical DSP / rhythm descriptors (CPU, librosa + madmom)
    maest     → MAEST embedding + 519 genre logits (GPU, batched)
    cluster   → kNN graph + multi-resolution Leiden (whole-library, singleton)
"""

from __future__ import annotations

KIND_SCAN = "scan"
KIND_FEATURES = "features"
KIND_MAEST = "maest"
KIND_CLUSTER = "cluster"

#: Ordered by pipeline position — status UIs render them in this order.
JOB_KINDS: tuple[str, ...] = (KIND_SCAN, KIND_FEATURES, KIND_MAEST, KIND_CLUSTER)

_LABELS: dict[str, str] = {
    KIND_SCAN: "Library scan",
    KIND_FEATURES: "Audio features",
    KIND_MAEST: "MAEST analysis",
    KIND_CLUSTER: "Clustering",
}


def kind_label(kind: str) -> str:
    """Human-readable name for a job kind."""
    return _LABELS.get(kind, kind)
