"""Single frozen recipe for embedding-space layout (no UI toggles).

Changing weights, sources, or the audio-feature mask is done here; bump
``PROJECTION_RECIPE_VERSION`` in ``layout_revision`` consumers when the recipe
changes so layout caches invalidate cleanly.
"""

from __future__ import annotations

import numpy as np

from .librosa_audio_features import AUDIO_FEATURE_DIM

# Bump in layout_revision.compute_layout_revision payload when this recipe changes.
PROJECTION_RECIPE_VERSION = 1

FROZEN_SOURCES: tuple[str, ...] = ("clap", "effnet", "features")

# Order: tempo_norm, key_cos, key_sin, mode, energy_norm, danceability — only tempo, energy, dance.
FROZEN_FEATURE_MASK = np.array([1, 0, 0, 0, 1, 1], dtype=np.float32)
assert int(FROZEN_FEATURE_MASK.size) == AUDIO_FEATURE_DIM

FROZEN_FEATURES_BLEND = 0.42
# Former UI slider maxima (“folder contrast” 0–6, “depth emphasis” 1–3).
FROZEN_FOLDER_BOOST = 6.0
FROZEN_FOLDER_DEPTH_BOOST = 3.0
FROZEN_SCALE_FOLDERS = True
