"""Default embedding-space layout recipe (weights are client-overridable).

``FROZEN_SOURCE_WEIGHTS`` is the *default* recipe: what the map uses when the
client sends no weights. The UI can override the source weights per request (the
``w=`` query param), and the effective weights are part of ``layout_revision``, so
each configuration caches its own coordinates. Sources and the audio-feature mask
remain fixed here.

The default source set follows the microgenre attack plan: MAEST's Discogs style
logits carry the genre signal, its hidden embedding carries audio semantics, and the
rhythm block carries the electronic-music groove distinctions that learned
semantic embeddings tend to blur. CLAP is retained at low weight because it is
the only source with a *text* encoder — the semantic folder/tag weighting in
``layout.py`` projects text directions into the CLAP column span.
"""

from __future__ import annotations

import numpy as np

from .librosa_audio_features import AUDIO_FEATURE_DIM

# Bump in layout_revision.compute_layout_revision payload when this recipe changes.
PROJECTION_RECIPE_VERSION = 5

FROZEN_SOURCES: tuple[str, ...] = (
    "maest_logits",
    "maest",
    "rhythm",
    "clap",
    "supervised",
    "features",
)

#: Plan §8 weighting. Mirrors ``clustering.DEFAULT_WEIGHTS`` so the 2-D map and
#: the cluster assignments describe the same space — a layout that disagreed
#: with the clustering would be actively misleading.
FROZEN_SOURCE_WEIGHTS: dict[str, float] = {
    "maest_logits": 0.45,
    "maest": 0.25,
    "rhythm": 0.15,
    "clap": 0.10,
    "supervised": 0.15,
    "features": 0.00,
}

#: Weights for the supervised head's *input*: the raw blocks, excluding the
#: head's own output (feeding ``supervised`` back in would be circular).
SUPERVISED_INPUT_WEIGHTS: dict[str, float] = {
    k: v for k, v in FROZEN_SOURCE_WEIGHTS.items() if k != "supervised"
}

#: Sources a track must have before it can be placed on the map.
REQUIRED_SOURCES: tuple[str, ...] = ("maest_logits", "maest")

# Order: tempo_norm, key_cos, key_sin, mode, energy_norm, danceability — only tempo, energy, dance.
FROZEN_FEATURE_MASK = np.array([1, 0, 0, 0, 1, 1], dtype=np.float32)
assert int(FROZEN_FEATURE_MASK.size) == AUDIO_FEATURE_DIM

# The audio-feature blend (``FROZEN_FEATURES_BLEND``) lives in ``backend.tuning``
# alongside the other tunable thresholds — see that module for the single reference.
# Former UI slider maxima ("folder contrast" 0–6, "depth emphasis" 1–3).
FROZEN_FOLDER_BOOST = 6.0
FROZEN_FOLDER_DEPTH_BOOST = 3.0
FROZEN_SCALE_FOLDERS = True
