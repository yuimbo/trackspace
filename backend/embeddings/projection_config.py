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

from dataclasses import dataclass

import numpy as np

from .librosa_audio_features import AUDIO_FEATURE_DIM

# Bump in layout_revision.compute_layout_revision payload when this recipe changes.
PROJECTION_RECIPE_VERSION = 8

#: openTSNE full-fit budget; evaluation recipes specify their own frozen budgets.
OPENTSNE_N_ITER = 100
OPENTSNE_EARLY_EXAGGERATION_ITER = 100

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


def _single_lens_weights(sources: tuple[str, ...]) -> dict[str, float]:
    """Exact weight vector the frontend sends for a per-source lens.

    Every source is listed explicitly (0 except the lens's own), matching the
    frontend's ``lensWeights`` so a warmed layout and the request that later reads
    it share one ``layout_revision``.
    """
    return {k: (1.0 if k in sources else 0.0) for k in FROZEN_SOURCE_WEIGHTS}


@dataclass(frozen=True)
class ProjectionLens:
    """One UI lens the backend can project.

    ``weights`` is ``None`` for the combined lens (the frozen defaults, which the
    client omits from the query string) and the complete explicit vector for a
    per-source lens.
    """

    id: str
    sources: tuple[str, ...]
    weights: dict[str, float] | None


#: Projection lenses the layout-prefetch job warms, in E-cycle order.
#: Must stay in step with ``frontend/src/lib/lenses.ts`` — the id, sources and
#: weights are what decide whether a prefetched layout is the one the client asks for.
PROJECTION_LENSES: tuple[ProjectionLens, ...] = (
    ProjectionLens("combined", FROZEN_SOURCES, None),
    ProjectionLens("maest", ("maest_logits", "maest"), _single_lens_weights(("maest_logits", "maest"))),
    ProjectionLens("clap", ("clap",), _single_lens_weights(("clap",))),
    ProjectionLens("rhythm", ("rhythm",), _single_lens_weights(("rhythm",))),
    ProjectionLens("supervised", ("supervised",), _single_lens_weights(("supervised",))),
)

#: Lens ids the prefetch job warms, for enqueue/status convenience.
PROJECTION_LENS_IDS: tuple[str, ...] = tuple(l.id for l in PROJECTION_LENSES)

#: Sources a track must have before it can be placed on the *combined* map.
REQUIRED_SOURCES: tuple[str, ...] = ("maest_logits", "maest")


def required_sources_for(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Sources a track must have to be placed in a lens.

    A single-source lens requires exactly its own source: there is no fallback
    block to zero-fill, so requiring the MAEST pair would hide every track the
    lens is about.  A multi-source lens keeps the frozen MAEST requirement when
    it is present, and a lens with no required source at all is rejected — it would
    place every fingerprinted track and make the "analysis ready" gate meaningless.
    """
    srcs = tuple(s for s in sources if s in FROZEN_SOURCES)
    if len(srcs) == 1:
        return srcs
    required = tuple(s for s in REQUIRED_SOURCES if s in srcs)
    if not required:
        raise ValueError(
            "a multi-source projection must include a required source "
            f"(one of: {', '.join(REQUIRED_SOURCES)})"
        )
    return required

# Encoded order: camelot_x, camelot_y, spiral_x, spiral_y, energy, dance — all
# six are meaningful now, so the whole block is kept.
FROZEN_FEATURE_MASK = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
assert int(FROZEN_FEATURE_MASK.size) == AUDIO_FEATURE_DIM

# The audio-feature blend (``FROZEN_FEATURES_BLEND``) lives in ``backend.tuning``
# alongside the other tunable thresholds — see that module for the single reference.
# Former UI slider maxima ("folder contrast" 0–6, "depth emphasis" 1–3).
FROZEN_FOLDER_BOOST = 6.0
FROZEN_FOLDER_DEPTH_BOOST = 3.0
FROZEN_SCALE_FOLDERS = True
