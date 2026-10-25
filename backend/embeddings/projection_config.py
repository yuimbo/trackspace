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

# ──────────────────────────────────────────────────────────────
# Canonical projection recipes
# ──────────────────────────────────────────────────────────────
#
# A recipe is a complete, named projection configuration: the source blocks, their
# exact weights, and the openTSNE full-fit iteration budget. ``FROZEN_SOURCE_WEIGHTS``
# / ``clustering.DEFAULT_WEIGHTS`` / ``SUPERVISED_INPUT_WEIGHTS`` and the shared
# ``frontend/lib/cluster-weights.json`` are the app *defaults* and stay untouched;
# these recipes are a separate, opt-in registry the frontend selects by id.
#
# ``default`` mirrors the pre-recipe behaviour byte-for-byte (frozen sources +
# weights, openTSNE 100/100) so a default request hashes and caches identically.
# ``genre_full`` is the pure-genre lens (logits .40 / maest .20 / layer-7 .25 /
# MERT-L19 .15, the tunable rhythm/clap/supervised at 0, all other sources zero)
# with its own 100/100 budget. Genre is the default, so the recipe is always
# served; ``--no-genre-full`` disables the layer-7/MERT analysis blocks on the
# server side (the recipe id itself is not gated). The gallery 500/250 profile is
# an explicit client override, not the recipe default.

#: Every source the projection layer can consume, including the opt-in genre_full
#: blocks. ``FROZEN_SOURCES`` (the default recipe's set) is the combined/subset
#: production set — the opt-in ``maest_layer7`` / ``mert_l19`` blocks appear only
#: here (they are not a suffix of ``FROZEN_SOURCES``).
AVAILABLE_SOURCES: tuple[str, ...] = (
    "maest_logits",
    "maest",
    "maest_layer7",
    "mert_l19",
    "rhythm",
    "clap",
    "supervised",
    "features",
)

#: The genre_full recipe's source set, in composite-gather order: the four genre
#: blocks plus the tunable non-genre contributors (rhythm/clap/supervised) that a
#: user can raise above zero in the weight preview. ``features`` stays excluded —
#: it is the audio-feature block, not a genre signal, and is never a valid genre
#: recipe override key.
GENRE_FULL_SOURCES: tuple[str, ...] = (
    "maest_logits",
    "maest",
    "maest_layer7",
    "mert_l19",
    "rhythm",
    "clap",
    "supervised",
)

#: Exact ``genre_full`` weights (mirrors ``backend/experiments/genre_space/scoring.py``).
#: A pure genre lens: the non-genre sources are explicitly zero so no budget blending
#: is invented. The MAEST pair is required; layer-7 and MERT are optional zero-fill.
GENRE_FULL_RECIPE_WEIGHTS: dict[str, float] = {
    "maest_logits": 0.40,
    "maest": 0.20,
    "maest_layer7": 0.25,
    "mert_l19": 0.15,
    "rhythm": 0.00,
    "clap": 0.00,
    "supervised": 0.00,
    "features": 0.00,
}

#: openTSNE iteration budget bounds for the opt-in quality fields. ``tsne_n_iter``
#: is the full-fit iteration count, ``tsne_early_exaggeration_iter`` the early
#: exaggeration phase (openTSNE accepts 0 to disable it).
TSNE_N_ITER_LO = 100
TSNE_N_ITER_HI = 2000
TSNE_EARLY_EXAGGERATION_ITER_LO = 0
TSNE_EARLY_EXAGGERATION_ITER_HI = 1000

#: Default recipe's openTSNE budget (unchanged from the pre-recipe constants).
DEFAULT_RECIPE_TSNE_N_ITER = OPENTSNE_N_ITER
DEFAULT_RECIPE_TSNE_EARLY_EXAGGERATION_ITER = OPENTSNE_EARLY_EXAGGERATION_ITER

#: genre_full recipe's openTSNE budget. The recipe default is 100/100 (the
#: user-agreed default); the gallery 500/250 profile is an explicit client
#: override. Because the genre budget is encoded *absolutely* in its cache keys,
#: this default can move without aliasing any earlier fit/layout.
GENRE_FULL_TSNE_N_ITER = 100
GENRE_FULL_TSNE_EARLY_EXAGGERATION_ITER = 100


@dataclass(frozen=True)
class ProjectionRecipe:
    """A complete, named projection configuration.

    ``weights`` is the complete vector over ``AVAILABLE_SOURCES`` so every source's
    contribution is explicit (a pure genre recipe zeros the non-genre blocks).
    ``tsne_n_iter`` / ``tsne_early_exaggeration_iter`` are the openTSNE full-fit
    budget used when the request does not override them.
    """

    id: str
    sources: tuple[str, ...]
    weights: dict[str, float]
    tsne_n_iter: int
    tsne_early_exaggeration_iter: int


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


#: Canonical recipe registry. ``default`` is the frozen app behaviour; ``genre_full``
#: is the opt-in pure-genre lens. ``PROJECTION_RECIPES["default"]`` must stay in
#: lock-step with ``FROZEN_SOURCE_WEIGHTS`` / ``clustering.DEFAULT_WEIGHTS`` and the
#: shared ``frontend/lib/cluster-weights.json``.
PROJECTION_RECIPES: dict[str, ProjectionRecipe] = {
    "default": ProjectionRecipe(
        id="default",
        sources=FROZEN_SOURCES,
        weights=dict(FROZEN_SOURCE_WEIGHTS),
        tsne_n_iter=DEFAULT_RECIPE_TSNE_N_ITER,
        tsne_early_exaggeration_iter=DEFAULT_RECIPE_TSNE_EARLY_EXAGGERATION_ITER,
    ),
    "genre_full": ProjectionRecipe(
        id="genre_full",
        sources=GENRE_FULL_SOURCES,
        weights=dict(GENRE_FULL_RECIPE_WEIGHTS),
        tsne_n_iter=GENRE_FULL_TSNE_N_ITER,
        tsne_early_exaggeration_iter=GENRE_FULL_TSNE_EARLY_EXAGGERATION_ITER,
    ),
}

#: Recipe ids the projection request accepts.
PROJECTION_RECIPE_IDS: tuple[str, ...] = tuple(PROJECTION_RECIPES.keys())


def recipe_for(recipe_id: str | None) -> ProjectionRecipe:
    """The recipe for an opt-in id (``None``/``"default"`` → the frozen default)."""
    return PROJECTION_RECIPES.get(recipe_id or "default", PROJECTION_RECIPES["default"])


def tsne_iteration_suffix(
    tsne_n_iter: int, tsne_early_exaggeration_iter: int, recipe: ProjectionRecipe
) -> str:
    """Encode a deviation from *recipe*'s openTSNE budget for cache keys.

    Only a *deviation* from the recipe's default budget is encoded, so the default
    recipe at its default 100/100 produces an empty suffix (byte-identical cache
    keys), while a user-chosen faster/slower budget changes the fit-cache key and
    the layout revision so the two configurations never share a fit.

    ``genre_full`` is the exception: its *effective* budget is always encoded
    (absolute), because the recipe default moved from 500/250 to 100/100 and the
    legacy 500/250 default fit was stored under a bare key. Encoding the genre
    budget absolutely guarantees the 100/100 default never aliases that legacy
    fit/layout under the same key, and that any future default change (or an
    explicit override like the 500/250 gallery profile) refits rather than reusing
    a stale fit. Every other recipe stays deviation-only so its default keys are
    byte-identical.
    """
    absolute = recipe.id == "genre_full"
    parts: list[str] = []
    if absolute or tsne_n_iter != recipe.tsne_n_iter:
        parts.append(f"n{int(tsne_n_iter)}")
    if absolute or tsne_early_exaggeration_iter != recipe.tsne_early_exaggeration_iter:
        parts.append(f"e{int(tsne_early_exaggeration_iter)}")
    return "|".join(parts)


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

#: The opt-in genre_full projection lens, warmed only when the server enabled
#: genre_full. Kept out of ``PROJECTION_LENSES`` so a disabled server's prefetch
#: set is byte-identical to before (``PROJECTION_LENS_IDS`` is unchanged).
GENRE_FULL_LENS: ProjectionLens = ProjectionLens(
    "genre_full", GENRE_FULL_SOURCES, dict(GENRE_FULL_RECIPE_WEIGHTS)
)

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
