"""Single typed reference for tuning constants scattered across the backend.

Each constant here is a *knob* — a threshold or range that changes behaviour but
is not itself a cache/recipe identity. Keeping them in one module makes them easy to
audit and retune; the owning modules import them rather than redefining them.

Constants that participate in cache identity (version numbers, projection recipe
weights) stay with their domain module — bumping those has correctness consequences
and belongs next to the code that documents them.
"""

from __future__ import annotations

from typing import Final

# --- Audio feature extraction (backend/embeddings/librosa_audio_features.py) ---

#: Tempo range used to normalise BPM to [0, 1] and to invert that back for
#: display. Tracks outside the range clip to the ends.
BPM_LO: Final[float] = 60.0
BPM_HI: Final[float] = 200.0

# --- EffNet ONNX model download (backend/embeddings/effnet.py) ---

#: Reject a downloaded/cached model smaller than this many bytes. The published
#: file is ~18 MB; this catches truncation before the checksum is even read.
EFFNET_MODEL_MIN_BYTES: Final[int] = 10_000_000

#: Download attempts before the loader gives up and raises.
EFFNET_DOWNLOAD_MAX_RETRIES: Final[int] = 3

# --- Projection recipe (backend/embeddings/projection_config.py) ---

#: Blend weight for the 6-D audio-feature block when it is folded into the
#: composite projection vector (0 disables the block).
FROZEN_FEATURES_BLEND: Final[float] = 0.42
