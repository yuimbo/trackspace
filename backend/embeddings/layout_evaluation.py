"""Offline layout evaluation: precomputed maps from cached embeddings.

This is a *read-only* evaluation surface. It gathers the same composite matrix the
interactive projection uses (via ``layout.prepare_projection_input``, so the input
rules — required sources, per-source weights, feature mask/blend — are identical),
but it fits its own UMAP / openTSNE models directly and never touches the
production layout or t-SNE fit caches, and it never mutates MP3 tags. Its only
output is a set of JSON artifacts under one ``--output`` directory.

The goal is a reproducible gallery of maps over the *same* track set, aligned
(orthogonal Procrustes rotation/reflection only, no warp) onto a reference map,
then all normalised aspect-preservingly into the unit square. The reference
defaults to the ``baseline`` recipe but any registered recipe can be the
reference (the ``scale`` set aligns onto ``tsne_budget100``); the reference is
always fitted first and auto-included. The paired ``hierarchy-combination`` set
adds explicit 3-D siblings that align onto their own dimension's reference.

Nothing in this module runs at import time, and the heavy libraries (umap-learn,
openTSNE, scikit-learn) are imported lazily inside the fit functions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from .linalg_utils import apply_procrustes, matched_indices, procrustes
from .multilevel_tsne import (
    expand_landmark_init as _expand_landmark_init,
    fit_multilevel,
    resolve_fraction_schedule,
    scale_initialization,
)
from .projection_config import FROZEN_SOURCES

log = logging.getLogger(__name__)

#: Schema version of the manifest and per-recipe artifacts. Bump on a breaking
#: shape change so a reader can reject an old generation explicitly.
SCHEMA_VERSION = 1

#: Below this many tracks the baseline map (an openTSNE t-SNE fit) cannot be
#: produced: its perplexity clamp (min 5) cannot hold, and the gallery aligns
#: every other recipe onto that baseline. A PCA-only recipe could still fit on
#: fewer tracks, but without the baseline there is nothing to align onto, so one
#: shared floor applies to the whole gallery.
MIN_TRACKS = 6

#: Reference recipe everything else is aligned onto. It must be present in
#: :data:`RECIPES`.
BASELINE_ID = "baseline"


class MissingDependencyError(RuntimeError):
    """A recipe needs umap-learn or openTSNE and it is not installed."""


@dataclass(frozen=True)
class Recipe:
    """One precomputed map. Fields form the recipe's reproducible identity."""

    id: str
    label: str
    pca_dim: int
    method: str  # "pca" | "tsne" | "umap" | "umap_tsne" | "tsne_umap_init" | "tsne_sample_init" | "tsne_multilevel_init"
    tsne_n_iter: int = 500
    tsne_early_exaggeration_iter: int = 250
    #: Perplexity for the t-SNE methods. ``None`` keeps the production scalar
    #: rule (:func:`backend.embeddings.projection_backends._tsne_perplexity`);
    #: a one-element tuple fixes a scalar
    #: perplexity; two or more selects openTSNE's multiscale affinity mixture.
    #: ``None`` is omitted from the recipe's identity so the pre-existing recipes
    #: keep their exact revisions/artifacts.
    perplexities: tuple[float, ...] | None = None
    #: Explicit per-source mix for this map, as named ``(source, weight)`` pairs
    #: covering every source in :data:`FROZEN_SOURCES` (a complete vector, so the
    #: order is never positional). ``None`` means the gallery's default
    #: (control) source weights are used — the existing recipes all keep ``None``
    #: and their exact revisions. A non-``None`` value gives the recipe its own
    #: composite matrix and a distinct input fingerprint / revision.
    source_weights: tuple[tuple[str, float], ...] | None = None
    #: Two-level sample init (method ``"tsne_sample_init"``): coarse t-SNE on
    #: ``tsne_sample_n`` sampled rows (the ``tsne_sample_*`` budget), expand every
    #: full row onto its nearest landmark (tiny jitter), then refine a full-data
    #: t-SNE (this recipe's own ``tsne_*`` budget). ``None`` = plain recipe; each
    #: omitted from the identity so pre-existing recipes keep their revisions.
    tsne_sample_n: int | None = None
    tsne_sample_n_iter: int | None = None
    tsne_sample_early_exaggeration_iter: int | None = None
    #: Multilevel nested sample init (method ``"tsne_multilevel_init"``): a
    #: ``(size, normal, early)`` tuple per level, sizes strictly increasing; one
    #: seeded permutation's ascending prefixes make each level a nested superset of
    #: the previous. First level = coarse fit; each next expands+refines a full
    #: t-SNE at that size; the final full-data fit uses this recipe's own budget.
    #: ``None`` for a plain/two-level recipe; omitted from the identity.
    tsne_sample_levels: tuple[tuple[int, int, int], ...] | None = None
    #: Per-level perplexity override for the multilevel method: one explicit
    #: scalar per level of ``tsne_sample_levels``, matched *by position* against
    #: the original (pre-size-filter) levels so a tiny dataset never mis-pairs.
    #: ``None`` = every level uses the production scalar rule. Each value must be
    #: positive and finite. This never touches the *final* full-data fit, which
    #: keeps using ``perplexities``. Omitted from the identity when ``None``.
    tsne_level_perplexities: tuple[float, ...] | None = None
    #: Size-adaptive multilevel chains (fraction mode, method
    #: ``"tsne_multilevel_init"``): instead of fixed absolute level sizes, this is
    #: a strictly-increasing tuple of fractions ``0 < f < 1`` that resolves to
    #: level counts ``round(n·f)`` for the *actual* dataset size ``n`` (each at
    #: least 2, deduped strictly increasing, a level at/above ``n`` dropped).
    #: Mutually exclusive with ``tsne_sample_levels``. ``None`` = not fraction
    #: mode; omitted from the identity.
    tsne_sample_fractions: tuple[float, ...] | None = None
    #: Fraction mode only: the TOTAL normal-iteration budget shared across all
    #: coarse levels, split by inverse-sqrt-of-fraction weight (largest remainder,
    #: at least 1 each, deterministic). ``None`` = the automatic per-level formula
    #: ``round(50·√(n/count))``. Omitted from the identity when ``None``.
    tsne_coarse_iteration_budget: int | None = None
    #: Fraction mode only: per-level perplexity schedule exponent. ``None`` =
    #: every coarse level uses perplexity 30. A nonnegative finite gamma schedules
    #: each level's perplexity ``p(t)=30+50·(1−t)^gamma`` where ``t`` is the
    #: log-scaled position of the level's count between ``n/64`` and ``n``
    #: (clamped 0..1), so gamma 0 pins the coarse levels at 80. Omitted from the
    #: identity when ``None``.
    tsne_perplexity_gamma: float | None = None
    #: Fraction mode only: bottom anchor of the gamma perplexity schedule as a
    #: fraction of the dataset size. ``None`` keeps the original hard-coded
    #: ``n/64`` anchor (log span 64); ``1/256`` anchors at ``n/256`` with a log
    #: span of 256, so the curve stays low for longer on a wider level range.
    #: Requires a non-``None`` :attr:`tsne_perplexity_gamma`. Omitted from the
    #: identity when ``None``.
    tsne_perplexity_start_fraction: float | None = None
    #: Output dimensionality. ``None`` = 2 (every pre-existing recipe), so the
    #: optional field never changes an old identity. Only the explicitly
    #: registered 3-D siblings set ``3``; only the multilevel t-SNE method
    #: supports it.
    dims: int | None = None
    #: Paired-family id for the gallery (2-D base and 3-D sibling share it).
    #: ``None`` = the recipe id itself; only the new 3-D siblings set it.
    group: str | None = None
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


#: The frozen gallery control: openTSNE over PCA50 with the production
#: perplexity/seed. Its label and 500/250 budget are frozen artifact identity
#: (``label`` is hashed into the recipe revision), so they keep the historical
#: "production default" wording even though production is now the multilevel
#: 50/0 chain.
BASELINE_RECIPE: Recipe = Recipe(
    "baseline",
    "PCA50 → openTSNE (production default)",
    pca_dim=50,
    method="tsne",
)

#: The 250-iteration budget control. Also part of the ``budget`` gallery set,
#: hence a named constant rather than an inline tuple entry.
TSNE_BUDGET: Recipe = Recipe(
    "tsne_budget",
    "PCA50 → openTSNE (n_iter=250, early phase unchanged)",
    pca_dim=50,
    method="tsne",
    tsne_n_iter=250,
    tsne_early_exaggeration_iter=250,
)


#: The eight eval maps. The baseline is a *frozen control*: openTSNE over PCA50
#: with the production perplexity/seed and the recipe's own budget (default
#: 500/250 — the openTSNE library default), fully unsupervised; the rest probe
#: dimensionality, budget, and neighbours. This tuple is the *default*
#: generation; the extras below are selectable through :data:`GALLERY_SETS`
#: but are not part of the default eight.
#:
#: Production fits its own separate budget (the ``projection_config.OPENTSNE_*``
#: iteration constants); these eval recipes and their labels/revisions are
#: intentionally left frozen.
RECIPES: tuple[Recipe, ...] = (
    BASELINE_RECIPE,
    Recipe("pca2", "PCA-only 2D", pca_dim=2, method="pca"),
    Recipe("pca25", "PCA25 → openTSNE", pca_dim=25, method="tsne"),
    TSNE_BUDGET,
    Recipe(
        "umap2_n15",
        "PCA50 → UMAP2 (n_neighbors=15, min_dist=0.1)",
        pca_dim=50,
        method="umap",
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    ),
    Recipe(
        "umap2_n60",
        "PCA50 → UMAP2 (n_neighbors=60, min_dist=0.1)",
        pca_dim=50,
        method="umap",
        umap_n_neighbors=60,
        umap_min_dist=0.1,
    ),
    Recipe(
        "umap_tsne",
        "PCA50 → UMAP15 (n_neighbors=30) → tSNE2",
        pca_dim=50,
        method="umap_tsne",
        umap_n_components=15,
        umap_n_neighbors=30,
    ),
    Recipe(
        "tsne_umap_init",
        "PCA50 → tSNE2 (UMAP2 initialization)",
        pca_dim=50,
        method="tsne_umap_init",
        umap_n_neighbors=15,
    ),
)

#: The shared 100-normal / 250-early control of the ``scale``, ``early``,
#: ``mixes`` and ``ablation`` sets (perplexity 30 via the production rule).
#: Identity-identical to the ``budget`` set's 100-iteration recipe.
TSNE_BUDGET100: Recipe = Recipe(
    "tsne_budget100",
    "PCA50 → openTSNE (n_iter=100, early phase unchanged)",
    pca_dim=50,
    method="tsne",
    tsne_n_iter=100,
    tsne_early_exaggeration_iter=250,
)

#: The t-SNE extras (the ``budget``, ``scale`` and ``early`` gallery sets), all
#: over the same default unsupervised composite input (PCA50, default weights,
#: seed 42, production perplexity rule). Kept out of :data:`RECIPES` so the
#: default generation stays the original eight; :data:`GALLERY_SETS` is the
#: authoritative set membership.
BUDGET_RECIPES: tuple[Recipe, ...] = (
    TSNE_BUDGET100,
    Recipe(
        "tsne_budget150",
        "PCA50 → openTSNE (n_iter=150, early phase unchanged)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=150,
        tsne_early_exaggeration_iter=250,
    ),
)

#: ``scale``: n_iter=100 with early 250 over PCA50, probing the perplexity axis.
#: The control is :data:`TSNE_BUDGET100`; the rest raise the neighbourhood scale
#: or blend several via openTSNE's multiscale mixture.
SCALE_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_p60",
        "PCA50 → openTSNE (perplexity 60, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(60.0,),
    ),
    Recipe(
        "tsne_p120",
        "PCA50 → openTSNE (perplexity 120, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(120.0,),
    ),
    Recipe(
        "tsne_multiscale30_120",
        "PCA50 → openTSNE multiscale (perplexities 30 & 120, n_iter=100, early 250)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        perplexities=(30.0, 120.0),
    ),
)

#: The reference for the ``hierarchy``, ``hierarchy-budget`` and
#: ``hierarchy-depth`` two-level sets: production's own iteration budget at
#: 100 normal / 100 early, over PCA50.
TSNE_EARLY100: Recipe = Recipe(
    "tsne_early100",
    "PCA50 → openTSNE (early phase 100, final 100)",
    pca_dim=50,
    method="tsne",
    tsne_n_iter=100,
    tsne_early_exaggeration_iter=100,
)

#: ``early``: production's iteration budget over PCA50, probing the
#: early-exaggeration phase length. The control is :data:`TSNE_BUDGET100`; the
#: variants shorten the early phase to 150 and 100 iterations.
EARLY_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_early150",
        "PCA50 → openTSNE (early phase 150, final 100)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=150,
    ),
    TSNE_EARLY100,
)

#: The source-mix extras (the ``mixes`` and ``ablation`` gallery sets). All share
#: the control geometry — PCA50, production scalar perplexity rule (30 at library
#: scale), n_iter=100/early 250, seed 42, unsupervised,
#: ``tsne_budget100`` as the alignment reference — and differ only in how the six
#: sources are weighted. The weights are explicit named pairs over
#: ``FROZEN_SOURCES`` (maest_logits, maest, rhythm, clap, supervised, features);
#: each source contributes ``sqrt(w)``× its block exactly as in production, and
#: ``features`` is always 0. Kept out of
#: :data:`RECIPES` so the default generation stays the original eight, but the
#: recipes are fully selectable through :data:`RECIPES_BY_ID`.
#:
#: The ``mixes`` recipes (``tsne_rhythm_forward``, ``tsne_texture_style``)
#: redistribute weight among the four learned sources with ``supervised`` fixed at
#: .15 and a total of 1.10, matching the production control weights
#: (``FROZEN_SOURCE_WEIGHTS`` also sums to 1.10) with no normalization.
#:
#: The ``ablation`` recipes drop whole blocks: ``tsne_maest_supervised`` keeps only
#: maest_logits, maest and supervised (sum .85); ``tsne_maest_only`` keeps only
#: maest_logits and maest (sum .70). The retained weights are preserved exactly
#: (.45/.25 and the supervised .15) — never renormalized — so the relative
#: contribution of the kept blocks matches the control; the *uniform* distance
#: factor simply adapts to the smaller column count. A zero-weight source drops
#: its columns from the composite matrix (the shared ``layout.prepare_projection_input``
#: skips weight<=0 blocks), so these variants are intentionally narrower than the
#: control: 1415 and 1287 columns versus the control's 1941 (maest_logits 519 +
#: maest 768 + supervised 128; the control adds rhythm 14 and clap 512). The track
#: universe is unchanged — every variant still gates on the same required sources
#: (maest_logits, maest) and places the identical path set.
#:
#: The labels state the *hypothesis* each mix is built to probe — they describe
#: the intended weighting, not a guarantee about the perceived semantics.
MIX_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_rhythm_forward",
        "PCA50 → openTSNE (rhythm-forward: rhythm .40, maest_logits .30)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.30),
            ("maest", 0.20),
            ("rhythm", 0.40),
            ("clap", 0.05),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
    Recipe(
        "tsne_texture_style",
        "PCA50 → openTSNE (texture-style: maest .35, clap .30)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.25),
            ("maest", 0.35),
            ("rhythm", 0.05),
            ("clap", 0.30),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
)

ABLATION_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "tsne_maest_supervised",
        "PCA50 → openTSNE (ablation: maest_logits .45, maest .25, supervised .15)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.45),
            ("maest", 0.25),
            ("rhythm", 0.0),
            ("clap", 0.0),
            ("supervised", 0.15),
            ("features", 0.0),
        ),
    ),
    Recipe(
        "tsne_maest_only",
        "PCA50 → openTSNE (ablation: maest_logits .45, maest .25)",
        pca_dim=50,
        method="tsne",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=250,
        source_weights=(
            ("maest_logits", 0.45),
            ("maest", 0.25),
            ("rhythm", 0.0),
            ("clap", 0.0),
            ("supervised", 0.0),
            ("features", 0.0),
        ),
    ),
)

#: The plain 50/50 t-SNE control of the two-level ``hierarchy`` sets (shared by
#: ``hierarchy``, ``hierarchy-budget`` and ``hierarchy-depth``).
TSNE50_PLAIN: Recipe = Recipe(
    "tsne50_plain",
    "PCA50 → openTSNE (control: 50 iters normal / 50 early, plain)",
    pca_dim=50,
    method="tsne",
    tsne_n_iter=50,
    tsne_early_exaggeration_iter=50,
)

def _sample_recipe(
    rid: str,
    n: int,
    *,
    coarse: tuple[int, int],
    final: tuple[int, int],
    coarse_words: bool = False,
) -> Recipe:
    """One two-level sample-init recipe of a gallery set.

    A coarse t-SNE runs on *n* sampled rows at the ``coarse (normal, early)``
    budget, then the full-data refinement uses the ``final`` budget. Every such
    recipe is PCA50 with the production perplexity rule, so only these budgets
    vary and the label derives from them — it is part of the recipe's hashed
    identity, so the wording must stay stable. *coarse_words* selects the
    coarse-budget sweep's ``N normal / E early`` wording.
    """
    cn, ce = coarse
    fn, fe = final
    coarse_txt = f"{cn} normal / {ce} early" if coarse_words else f"{cn}/{ce}"
    return Recipe(
        rid,
        f"PCA50 → sample{n} (coarse {coarse_txt}, refine {fn}/{fe})",
        pca_dim=50,
        method="tsne_sample_init",
        tsne_n_iter=fn,
        tsne_early_exaggeration_iter=fe,
        tsne_sample_n=n,
        tsne_sample_n_iter=cn,
        tsne_sample_early_exaggeration_iter=ce,
    )


#: The sample1000 / final-100/0 anchor of the two-level coarse-budget sweeps
#: (shared by ``hierarchy``, ``hierarchy-budget`` and ``hierarchy-depth``).
TSNE_SAMPLE1000_EARLY0: Recipe = _sample_recipe(
    "tsne_sample1000_early0", 1000, coarse=(100, 100), final=(100, 0)
)

#: The two-level sample-init ``hierarchy`` gallery set. Every ``tsne_sample*``
#: map: coarse t-SNE on a sample, NN-expand, full refinement (the recipe's own
#: budgets). All PCA50, production perplexity rule, seed 42, reference the plain
#: 100/100 ``tsne_early100`` control. Out of :data:`RECIPES` so the default eight
#: stay the same.
HIERARCHY_RECIPES: tuple[Recipe, ...] = (
    TSNE50_PLAIN,
    _sample_recipe("tsne_sample500", 500, coarse=(100, 100), final=(100, 100)),
    _sample_recipe("tsne_sample1000", 1000, coarse=(100, 100), final=(100, 100)),
    _sample_recipe("tsne_sample2000", 2000, coarse=(100, 100), final=(100, 100)),
    TSNE_SAMPLE1000_EARLY0,
    _sample_recipe(
        "tsne_sample1000_final50", 1000, coarse=(100, 100), final=(50, 50)
    ),
    _sample_recipe(
        "tsne_sample1000_coarse50", 1000, coarse=(50, 50), final=(100, 100)
    ),
)

#: Coarse-budget sweep of the ``hierarchy-budget`` set. Every map is sample1000
#: with the final refinement held at 100 normal / 0 early (the
#: ``tsne_sample1000_early0`` anchor geometry); only the **coarse** budget moves
#: (coarse normal 10/25/50/500 at early 100, then coarse early 0/25/50 at normal
#: 100). The plain 100/100 coarse of the anchor is reused, not duplicated. Budgets
#: written ``coarse{N}e{E}`` in the ids. All PCA50, production perplexity rule,
#: seed 42, reference ``tsne_early100``; only existing ``tsne_sample_init`` fields.
HIERARCHY_BUDGET_RECIPES: tuple[Recipe, ...] = (
    _sample_recipe(
        "tsne_sample1000_coarse10e100", 1000,
        coarse=(10, 100), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse25e100", 1000,
        coarse=(25, 100), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse50e100", 1000,
        coarse=(50, 100), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse500e100", 1000,
        coarse=(500, 100), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse100e0", 1000,
        coarse=(100, 0), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse100e25", 1000,
        coarse=(100, 25), final=(100, 0), coarse_words=True,
    ),
    _sample_recipe(
        "tsne_sample1000_coarse100e50", 1000,
        coarse=(100, 50), final=(100, 0), coarse_words=True,
    ),
)

#: The shallowest multilevel chain (50→250→1000): coarse 100/100, intermediate
#: levels 100/0, final 100/0. Kept as a named constant so the
#: ``hierarchy-tuning`` variants derive from it by ``dataclasses.replace``,
#: changing only the intended field, id and label.
TSNE_CHAIN50_250_1000: Recipe = Recipe(
    "tsne_chain50_250_1000",
    "PCA50 → multilevel 50→250→1000 (coarse 100/100, levels 100/0, final 100/0)",
    pca_dim=50,
    method="tsne_multilevel_init",
    tsne_n_iter=100,
    tsne_early_exaggeration_iter=0,
    tsne_sample_levels=(
        (50, 100, 100),
        (250, 100, 0),
        (1000, 100, 0),
    ),
)

#: The shallow chain's final-fit halved: identical to :data:`TSNE_CHAIN50_250_1000`
#: except the final full-data fit runs 50 normal / 0 early. Kept as a named
#: constant so the ``hierarchy-distribution`` variants derive from it by
#: ``dataclasses.replace`` and the ``hierarchy-tuning`` set shares the same
#: reference object.
TSNE_CHAIN50_250_1000_FINAL50: Recipe = replace(
    TSNE_CHAIN50_250_1000,
    id="tsne_chain50_250_1000_final50",
    label="PCA50 → multilevel 50→250→1000 (coarse 100/100, levels 100/0, final 50/0)",
    tsne_n_iter=50,
)

#: Aggressive depth gallery (``hierarchy-depth``): how far down the sample
#: hierarchy can you go? Two new two-level maps (sample 100/250) plus three
#: **multilevel** chain recipes (method ``tsne_multilevel_init``): a single seeded
#: nested sample hierarchy — coarse fit on the smallest level, expand+refine up
#: each level (intermediates 100/0), then a full-data fit at the recipe's own
#: final budget. All PCA50, production perplexity rule, seed 42, reference the
#: plain 100/100 ``tsne_early100`` control. Out of :data:`RECIPES`.
HIERARCHY_DEPTH_RECIPES: tuple[Recipe, ...] = (
    _sample_recipe("tsne_sample100_early0", 100, coarse=(100, 100), final=(100, 0)),
    _sample_recipe("tsne_sample250_early0", 250, coarse=(100, 100), final=(100, 0)),
    Recipe(
        "tsne_chain100_500_2000_final100",
        "PCA50 → multilevel 100→500→2000 (coarse 100/100, levels 100/0, final 100/0)",
        pca_dim=50,
        method="tsne_multilevel_init",
        tsne_n_iter=100,
        tsne_early_exaggeration_iter=0,
        tsne_sample_levels=(
            (100, 100, 100),
            (500, 100, 0),
            (2000, 100, 0),
        ),
    ),
    Recipe(
        "tsne_chain100_500_2000_final50",
        "PCA50 → multilevel 100→500→2000 (coarse 100/100, levels 100/0, final 50/0)",
        pca_dim=50,
        method="tsne_multilevel_init",
        tsne_n_iter=50,
        tsne_early_exaggeration_iter=0,
        tsne_sample_levels=(
            (100, 100, 100),
            (500, 100, 0),
            (2000, 100, 0),
        ),
    ),
    TSNE_CHAIN50_250_1000,
)

#: ``hierarchy-tuning`` gallery: isolate the global-shape budget around the
#: shallow ``50→250→1000`` chain. Every map shares the reference chain's coarse
#: 100/100 and intermediate 100/0 levels and differs only in the final fit (or,
#: for the ``coarse10`` variants, in the coarse budget too). All PCA50,
#: production perplexity rule, seed 42, reference ``tsne_chain50_250_1000``.
#: Deriving each via ``replace`` from the named constant guarantees only the
#: intended fields, id and label differ. Out of :data:`RECIPES`.
HIERARCHY_TUNING_RECIPES: tuple[Recipe, ...] = (
    TSNE_CHAIN50_250_1000_FINAL50,
    replace(
        TSNE_CHAIN50_250_1000,
        id="tsne_chain50_250_1000_coarse10",
        label="PCA50 → multilevel 50→250→1000 (coarse 10/100, levels 100/0, final 100/0)",
        tsne_sample_levels=((50, 10, 100), (250, 100, 0), (1000, 100, 0)),
    ),
    replace(
        TSNE_CHAIN50_250_1000,
        id="tsne_chain50_250_1000_coarse10_final50",
        label="PCA50 → multilevel 50→250→1000 (coarse 10/100, levels 100/0, final 50/0)",
        tsne_sample_levels=((50, 10, 100), (250, 100, 0), (1000, 100, 0)),
        tsne_n_iter=50,
    ),
    replace(
        TSNE_CHAIN50_250_1000,
        id="tsne_chain50_250_1000_final2000",
        label="PCA50 → multilevel 50→250→1000 (coarse 100/100, levels 100/0, final 2000/0)",
        tsne_n_iter=2000,
    ),
)

#: ``hierarchy-distribution`` gallery: where the *normal-iteration* budget is
#: spent across the levels of the shallow ``50→250→1000`` chain. Every map keeps
#: the final fit fixed at 50 normal / 0 early (the ``tsne_chain50_250_1000_final50``
#: geometry) and moves **only** the per-level normal counts, keeping the early
#: phases unchanged (100 on the first level, 0 elsewhere) and the total added
#: normal iterations fixed at exactly 900. The reference
#: ``tsne_chain50_250_1000_final50`` contributes 100+100+100=300 normal
#: iterations; each variant redistributes a 900-iteration surplus over one level
#: (``early1000``/``middle1000``/``late1000``) or evenly across all three
#: (``spread400``: 400 each), so all four total the same 1200. All PCA50,
#: production perplexity rule, seed 42, method ``tsne_multilevel_init``. Each
#: label spells the per-level budget as ``level{size} {normal}/{early}`` so a
#: level size (50/250/1000) is never confused with a normal-iteration count.
#: Deriving via ``replace`` from the final50 constant guarantees only the
#: intended field, id and label differ. Out of :data:`RECIPES`.
HIERARCHY_DISTRIBUTION_RECIPES: tuple[Recipe, ...] = (
    replace(
        TSNE_CHAIN50_250_1000_FINAL50,
        id="tsne_chain50_250_1000_early1000_final50",
        label="PCA50 → multilevel 50→250→1000 (level50 1000/100, level250 100/0, level1000 100/0, final 50/0)",
        tsne_sample_levels=((50, 1000, 100), (250, 100, 0), (1000, 100, 0)),
    ),
    replace(
        TSNE_CHAIN50_250_1000_FINAL50,
        id="tsne_chain50_250_1000_middle1000_final50",
        label="PCA50 → multilevel 50→250→1000 (level50 100/100, level250 1000/0, level1000 100/0, final 50/0)",
        tsne_sample_levels=((50, 100, 100), (250, 1000, 0), (1000, 100, 0)),
    ),
    replace(
        TSNE_CHAIN50_250_1000_FINAL50,
        id="tsne_chain50_250_1000_late1000_final50",
        label="PCA50 → multilevel 50→250→1000 (level50 100/100, level250 100/0, level1000 1000/0, final 50/0)",
        tsne_sample_levels=((50, 100, 100), (250, 100, 0), (1000, 1000, 0)),
    ),
    replace(
        TSNE_CHAIN50_250_1000_FINAL50,
        id="tsne_chain50_250_1000_spread400_final50",
        label="PCA50 → multilevel 50→250→1000 (level50 400/100, level250 400/0, level1000 400/0, final 50/0)",
        tsne_sample_levels=((50, 400, 100), (250, 400, 0), (1000, 400, 0)),
    ),
)

def _geo_heuristic_recipe(
    rid: str,
    label: str,
    levels: tuple[tuple[int, int, int], ...],
    level_perps: tuple[float, ...],
) -> Recipe:
    """One frozen multilevel chain of the ``hierarchy-heuristic`` set.

    Every recipe is PCA50, ``tsne_multilevel_init``, final fit 50 normal / 0
    early at an explicit final perplexity 30 (``perplexities=(30.0,)``), with the
    given geometric level sizes/budgets and a per-level perplexity per level.
    """
    return Recipe(
        rid,
        label,
        pca_dim=50,
        method="tsne_multilevel_init",
        tsne_n_iter=50,
        tsne_early_exaggeration_iter=0,
        tsne_sample_levels=levels,
        tsne_level_perplexities=level_perps,
        perplexities=(30.0,),
    )


#: ``hierarchy-heuristic`` gallery: discrete **geometric** multilevel chains that
#: spend compute across exponentially growing levels (rather than rebuilding
#: affinities on every gradient iteration — the t-SNE gradient step keeps one
#: fixed affinity matrix, so the cost driver is the number and size of *separate*
#: optimisation stages). All PCA50, seed 42, final fit 50/0 at perplexity 30
#: (``perplexities=(30.0,)``). This tuple holds only the **six** new geometric
#: variants; the existing fast ``tsne_chain50_250_1000_final50`` is the gallery's
#: alignment reference but is *not* duplicated here — the gallery command lists it
#: explicitly alongside these six (production perplexity rule, unchanged).
#:
#: Level normal-iteration counts are the **source-of-truth** frozen schedule
#: ``round(50 · sqrt(7153 / level_size))`` calibrated for the 7153-track library
#: (``geo4``: 256→1024→4096 → 264/132/66; ``geo8`` shares the 256 and 2048 rows →
#: 264/93). The values are written as explicit constants so the schedule never
#: depends on a runtime ``n``:
#:
#: * ``tsne_geo4_fixed`` — fixed 100-normal levels, all perplexity 30 (the axis
#:   baseline for the other geo4 maps).
#: * ``tsne_geo4_sqrt`` — the sqrt schedule (264/132/66), all perplexity 30
#:   (isolates the normal-iteration axis against ``tsne_geo4_fixed``).
#: * ``tsne_geo4_perplexity`` — fixed 100-normal levels, per-level perplexities
#:   80/40/30 (isolates the perplexity axis against ``tsne_geo4_fixed``).
#: * ``tsne_geo4_combined`` — sqrt schedule + per-level perplexities 80/40/30
#:   (both axes at once, against ``tsne_geo4_fixed``).
#: * ``tsne_geo8_combined`` — coarser geo×8 levels (256→2048), sqrt schedule,
#:   per-level perplexities 80/30 (fewer, bigger stages).
#: * ``tsne_geo4_combined_double`` — the combined geometry with normal counts
#:   doubled (528/264/132); final stays 50/0 (not doubled).
#:
#: The 80 perplexity on the 256-row first level is valid (80 < 256, and the
#: existing ``_clamp_perplexity`` guards smaller datasets), so no clamp fires on
#: the first stage. Out of :data:`RECIPES` so the default eight stay the same.
#: The fixed 256→2048 geometric chain: the alignment reference of the
#: ``hierarchy-fractions`` and ``hierarchy-density`` sets.
TSNE_GEO8_COMBINED: Recipe = _geo_heuristic_recipe(
    "tsne_geo8_combined",
    "PCA50 → multilevel geo×8 256→2048 (coarse 264/100, perps 80/30, final 50/0, p30)",
    ((256, 264, 100), (2048, 93, 0)),
    (80.0, 30.0),
)

HIERARCHY_HEURISTIC_RECIPES: tuple[Recipe, ...] = (
    _geo_heuristic_recipe(
        "tsne_geo4_fixed",
        "PCA50 → multilevel geo×4 256→1024→4096 (coarse 100/100, levels 100/0, perps 30/30/30, final 50/0, p30)",
        ((256, 100, 100), (1024, 100, 0), (4096, 100, 0)),
        (30.0, 30.0, 30.0),
    ),
    _geo_heuristic_recipe(
        "tsne_geo4_sqrt",
        "PCA50 → multilevel geo×4 256→1024→4096 (coarse 264/100, levels 132/0, perps 30/30/30, final 50/0, p30)",
        ((256, 264, 100), (1024, 132, 0), (4096, 66, 0)),
        (30.0, 30.0, 30.0),
    ),
    _geo_heuristic_recipe(
        "tsne_geo4_perplexity",
        "PCA50 → multilevel geo×4 256→1024→4096 (coarse 100/100, levels 100/0, perps 80/40/30, final 50/0, p30)",
        ((256, 100, 100), (1024, 100, 0), (4096, 100, 0)),
        (80.0, 40.0, 30.0),
    ),
    _geo_heuristic_recipe(
        "tsne_geo4_combined",
        "PCA50 → multilevel geo×4 256→1024→4096 (coarse 264/100, levels 132/0, perps 80/40/30, final 50/0, p30)",
        ((256, 264, 100), (1024, 132, 0), (4096, 66, 0)),
        (80.0, 40.0, 30.0),
    ),
    TSNE_GEO8_COMBINED,
    _geo_heuristic_recipe(
        "tsne_geo4_combined_double",
        "PCA50 → multilevel geo×4 256→1024→4096 (coarse 528/100, levels 264/0, perps 80/40/30, final 50/0, p30)",
        ((256, 528, 100), (1024, 264, 0), (4096, 132, 0)),
        (80.0, 40.0, 30.0),
    ),
)


def _fraction_recipe(
    rid: str,
    label: str,
    fractions: tuple[float, ...],
    *,
    budget: int | None = None,
    gamma: float | None = None,
    start_fraction: float | None = None,
) -> Recipe:
    """One size-adaptive fraction-mode multilevel recipe of a gallery set.

    Every recipe is PCA50, ``tsne_multilevel_init``, final fit 50 normal / 0
    early at an explicit final perplexity 30 (``perplexities=(30.0,)``), with
    the given size-adaptive ``tsne_sample_fractions``. ``budget`` optionally
    fixes the TOTAL normal budget shared across the coarse levels; ``gamma``
    optionally schedules the per-level perplexity and ``start_fraction`` moves
    the schedule's bottom anchor from the default ``n/64``. (All default to
    ``None`` — the flat auto-budget galleries.)
    """
    return Recipe(
        rid,
        label,
        pca_dim=50,
        method="tsne_multilevel_init",
        tsne_n_iter=50,
        tsne_early_exaggeration_iter=0,
        tsne_sample_fractions=fractions,
        tsne_coarse_iteration_budget=budget,
        tsne_perplexity_gamma=gamma,
        tsne_perplexity_start_fraction=start_fraction,
        perplexities=(30.0,),
    )


#: Geometrically log-spaced fractions from 1/64 to 1/8 for the ``hierarchy-density``
#: depth gallery. Frozen as explicit literals (``np.geomspace(1/64, 1/8, num)``
#: float64 output, no runtime numpy dependency) so each recipe's identity is stable
#: across runs and JSON round-trips. ``density4`` lands on the exact powers
#: 1/64, 1/32, 1/16, 1/8; the ×8 and ×16 sets are the same geometric range more
#: finely sampled.
_FRACTIONS_DENSITY4: tuple[float, ...] = (1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0)
_FRACTIONS_DENSITY8: tuple[float, ...] = (
    0.015625,
    0.021029690509880565,
    0.02830386450824708,
    0.038094176693889856,
    0.051270959750477366,
    0.06900559460461328,
    0.09287464307105926,
    0.125,
)
_FRACTIONS_DENSITY16: tuple[float, ...] = (
    0.015625,
    0.017948411796828673,
    0.020617311105826472,
    0.023683071351724972,
    0.02720470510300388,
    0.03125,
    0.03589682359365735,
    0.041234622211652944,
    0.04736614270344994,
    0.05440941020600776,
    0.06249999999999999,
    0.07179364718731468,
    0.0824692444233059,
    0.09473228540689989,
    0.10881882041201552,
    0.125,
)
#: Geometrically log-spaced fractions from 1/256 to 1/16 (``np.geomspace`` float64
#: literals) for the wide continuous-stress comparison. The wider bottom anchor
#: (``n/256`` instead of ``n/64``) makes the first coarse level ~28 rows on the
#: 7153-track library and exposes the gamma curve's low-count clipping.
_FRACTIONS_DENSITY256_16: tuple[float, ...] = (
    0.00390625,
    0.004699316547195768,
    0.005653395458751476,
    0.006801176275750969,
    0.008181985334536145,
    0.009843133202303695,
    0.011841535675862483,
    0.014245663883722134,
    0.017137890307728526,
    0.020617311105826465,
    0.024803141437003108,
    0.029838800122200505,
    0.03589682359365735,
    0.04318477749899301,
    0.051952368508924235,
    0.0625,
)


def _three_d(recipe: Recipe) -> Recipe:
    """Explicit 3-D sibling of a registered 2-D recipe.

    The id/label append ``_3d`` / `` (3D)``, ``dims`` is 3, and ``group`` pins the
    pairing back to the base id. Everything else (fractions, budgets, gamma,
    perplexities) is inherited unchanged, so the pair differs only by dimension.
    """
    return replace(
        recipe,
        id=f"{recipe.id}_3d",
        label=f"{recipe.label} (3D)",
        dims=3,
        group=recipe.id,
    )


#: ``hierarchy-fractions`` gallery: size-ADAPTIVE multilevel chains. Instead of
#: fixed absolute level sizes (the ``hierarchy-heuristic`` recipes), each recipe
#: names level sizes as FRACTIONS of the actual dataset — ``round(n·f)`` at
#: runtime — so the same recipe adapts to any library size while keeping its
#: count-axis geometry. Every map is PCA50, seed 42, final fit 50/0 at an
#: explicit perplexity 30, flat per-level perplexity 30 (no gamma), and the
#: per-level normal budgets come from the automatic ``round(50·√(n/count))``
#: formula (NO fixed total budget), so the normal-iteration axis is deliberately
#: sqrt-linked to the count axis. The existing ``tsne_geo8_combined`` (fixed
#: 256→2048, perps 80/30) is the gallery's alignment reference but is *not*
#: duplicated here — the gallery command lists it explicitly alongside these
#: three. On this 7153-track library the counts/normals resolve to g4 447/1788 →
#: 200/100, g8 112/894 → 400/141, g16 28/447 → 799/200. ``tsne_fraction_g16``'s
#: 28-row first level clamps its requested perplexity 30 down to 27. Out of
#: :data:`RECIPES`.
HIERARCHY_FRACTION_RECIPES: tuple[Recipe, ...] = (
    _fraction_recipe(
        "tsne_fraction_g4",
        "PCA50 → multilevel fractions 1/16→1/4 (auto 200/100, perps 30/30, final 50/0, p30)",
        (1.0 / 16.0, 1.0 / 4.0),
    ),
    _fraction_recipe(
        "tsne_fraction_g8",
        "PCA50 → multilevel fractions 1/64→1/8 (auto 400/141, perps 30/30, final 50/0, p30)",
        (1.0 / 64.0, 1.0 / 8.0),
    ),
    _fraction_recipe(
        "tsne_fraction_g16",
        "PCA50 → multilevel fractions 1/256→1/16 (auto 799/200, perps 30/30, final 50/0, p30)",
        (1.0 / 256.0, 1.0 / 16.0),
    ),
)


#: ``hierarchy-density`` gallery: how deep a fixed-budget fraction chain can go.
#: Every map is PCA50, seed 42, final fit 50/0 at an explicit perplexity 30, flat
#: per-level perplexity 30, and shares a fixed TOTAL normal budget of 541 across
#: all coarse levels (split by inverse-sqrt-of-fraction weight, largest
#: remainder, first level early 100) — the same allocation policy at every depth.
#: ``tsne_density2`` (1/64→1/8, 2 levels) is the 2-level baseline and reproduces
#: the auto ``tsne_fraction_g8`` budget (400/141); ``density4/8/16`` log-space the
#: same 1/64→1/8 range into 4/8/16 levels; ``continuous16`` keeps density16's 16
#: fractions but gives the whole chain a total-normal budget of 16 (so 1 normal
#: iteration per stage, early 100 on the first) to isolate what a *dense rebuild*
#: of many tiny stages costs vs the full-budget density16 — a stress of the
#: per-stage affinity-rebuild overhead, NOT a literal continuous (per-iteration
#: affinity update) fit. The existing ``tsne_geo8_combined`` is the gallery's
#: alignment reference (fixed 256→2048, perps 80/30) but is *not* duplicated
#: here — the gallery command lists it explicitly. Out of :data:`RECIPES`.
CONTINUOUS16_RECIPE = _fraction_recipe(
    "tsne_continuous16",
    "PCA50 → multilevel continuous-stress ×16 (budget 16 → 1/level, perps 30×16, final 50/0, p30)",
    _FRACTIONS_DENSITY16,
    budget=16,
)

#: The 4-level, 541-normal density chain: the flat-30 control/reference of the
#: ``hierarchy-gamma`` set.
TSNE_DENSITY4: Recipe = _fraction_recipe(
    "tsne_density4",
    "PCA50 → multilevel density 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps 30×4, final 50/0, p30)",
    _FRACTIONS_DENSITY4,
    budget=541,
)

HIERARCHY_DENSITY_RECIPES: tuple[Recipe, ...] = (
    _fraction_recipe(
        "tsne_density2",
        "PCA50 → multilevel density 1/64→1/8 ×2 (budget 541 → 400/141, perps 30/30, final 50/0, p30)",
        (1.0 / 64.0, 1.0 / 8.0),
        budget=541,
    ),
    TSNE_DENSITY4,
    _fraction_recipe(
        "tsne_density8",
        "PCA50 → multilevel density 1/64→1/8 ×8 (budget 541 → 107/93/80/69/59/51/44/38, perps 30×8, final 50/0, p30)",
        _FRACTIONS_DENSITY8,
        budget=541,
    ),
    _fraction_recipe(
        "tsne_density16",
        "PCA50 → multilevel density 1/64→1/8 ×16 (budget 541 → 54/50/47/44/41/38/36/33/31/29/27/25/24/22/21/19, perps 30×16, final 50/0, p30)",
        _FRACTIONS_DENSITY16,
        budget=541,
    ),
    CONTINUOUS16_RECIPE,
)

#: ``hierarchy-gamma`` gallery: how the per-level perplexity *schedule* (the
#: ``tsne_perplexity_gamma`` exponent) shapes an otherwise-identical fixed-budget
#: 4-level fraction chain. Every map is the ``tsne_density4`` geometry — PCA50,
#: seed 42, final fit 50/0 at explicit perplexity 30, fractions 1/64→1/8 ×4, and
#: a fixed TOTAL normal budget of 541 split the same inverse-sqrt way
#: (211/149/106/75, first level early 100) — so the only thing that differs is
#: the per-level perplexity schedule ``p(t)=30+50·(1−t)^gamma`` with ``t`` the
#: log-scaled position of the level's count between ``n/64`` and ``n``. Gamma 0
#: pins every coarse level at 80 (a deliberate flat-high-coarse / final-30
#: discontinuity); higher gamma pulls the coarse levels down towards the final 30
#: faster. The existing ``tsne_density4`` (gamma ``None`` = flat 30 per level) is
#: the gallery's flat-30 control and is *not* duplicated here — the gallery
#: command lists it explicitly as the reference. Out of :data:`RECIPES`.
HIERARCHY_GAMMA_RECIPES: tuple[Recipe, ...] = (
    _fraction_recipe(
        "tsne_gamma0",
        "PCA50 → multilevel gamma0 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps 80×4 const, final 50/0, p30)",
        _FRACTIONS_DENSITY4,
        budget=541,
        gamma=0.0,
    ),
    _fraction_recipe(
        "tsne_gamma05",
        "PCA50 → multilevel gamma0.5 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps ~80/76/71/65, final 50/0, p30)",
        _FRACTIONS_DENSITY4,
        budget=541,
        gamma=0.5,
    ),
    _fraction_recipe(
        "tsne_gamma1",
        "PCA50 → multilevel gamma1 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps ~80/72/63/55, final 50/0, p30)",
        _FRACTIONS_DENSITY4,
        budget=541,
        gamma=1.0,
    ),
    _fraction_recipe(
        "tsne_gamma2",
        "PCA50 → multilevel gamma2 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps ~80/65/52/42, final 50/0, p30)",
        _FRACTIONS_DENSITY4,
        budget=541,
        gamma=2.0,
    ),
    _fraction_recipe(
        "tsne_gamma4",
        "PCA50 → multilevel gamma4 1/64→1/8 ×4 (budget 541 → 211/149/106/75, perps ~80/54/40/33, final 50/0, p30)",
        _FRACTIONS_DENSITY4,
        budget=541,
        gamma=4.0,
    ),
)

#: ``hierarchy-combination`` gallery: the continuous-stress winner against a
#: *wider* fraction span and the same span with a gamma-4 schedule. Every recipe
#: keeps the 16-level, 1-normal-per-stage continuous-stress budget and the
#: explicit final 50/0 at perplexity 30. The two new 2-D maps widen the levels
#: from 1/64→1/8 to 1/256→1/16; ``wide_gamma4`` also schedules the coarse
#: perplexity with the same ``p(t)=30+50·(1−t)^gamma`` curve but anchors ``t`` at
#: ``n/256`` (log span 256) instead of the hard-coded ``n/64``. The existing
#: ``tsne_continuous16`` is reused unchanged as the control, then each of the
#: three gains an explicit 3-D sibling (:data:`COMBINATION_3D_RECIPES`).
COMBINATION_RECIPES: tuple[Recipe, ...] = (
    CONTINUOUS16_RECIPE,
    _fraction_recipe(
        "tsne_continuous16_wide",
        "PCA50 → multilevel continuous-wide ×16 1/256→1/16 (budget 16 → 1/level, perps 30×16, final 50/0, p30)",
        _FRACTIONS_DENSITY256_16,
        budget=16,
    ),
    _fraction_recipe(
        "tsne_continuous16_wide_gamma4",
        "PCA50 → multilevel continuous-wide gamma4 ×16 1/256→1/16 (budget 16 → 1/level, perps 30+50(1−t)^4, final 50/0, p30)",
        _FRACTIONS_DENSITY256_16,
        budget=16,
        gamma=4.0,
        start_fraction=1.0 / 256.0,
    ),
)

#: Explicit 3-D counterparts of :data:`COMBINATION_RECIPES` (same groups). Not a
#: suffix parser: every id is registered here, and only these recipes are 3-D.
COMBINATION_3D_RECIPES: tuple[Recipe, ...] = tuple(
    _three_d(recipe) for recipe in COMBINATION_RECIPES
)


@dataclass(frozen=True)
class GallerySet:
    """One named gallery set the reader (and the CLI) can address.

    ``directory`` is the fixed child of the layout-evaluation base dir the set's
    artifacts live in (``None`` = the base dir itself). ``recipes`` is the exact
    generation order for a set this CLI can build — the first recipe is the
    alignment reference. ``None`` marks a set produced by another tool (the
    genre-space and production-folder benchmark CLIs), so it is served by the
    gallery but never claimed as a built-in generation here.
    """

    directory: str | None
    recipes: tuple[Recipe, ...] | None


#: The authoritative gallery-set registry. ``RECIPES_BY_ID`` and the route's
#: ``SET_SUBDIRS`` are both derived from this, so a new built-in set is
#: registered in exactly one place.
GALLERY_SETS: dict[str, GallerySet] = {
    "base": GallerySet(None, RECIPES),
    "budget": GallerySet(
        "budget", (BASELINE_RECIPE, *BUDGET_RECIPES, TSNE_BUDGET)
    ),
    "scale": GallerySet("scale", (TSNE_BUDGET100, *SCALE_RECIPES)),
    "early": GallerySet("early", (TSNE_BUDGET100, *EARLY_RECIPES)),
    "mixes": GallerySet("mixes", (TSNE_BUDGET100, *MIX_RECIPES)),
    "ablation": GallerySet("ablation", (TSNE_BUDGET100, *ABLATION_RECIPES)),
    "hierarchy": GallerySet("hierarchy", (TSNE_EARLY100, *HIERARCHY_RECIPES)),
    "hierarchy-budget": GallerySet(
        "hierarchy-budget",
        (
            TSNE_EARLY100,
            TSNE50_PLAIN,
            TSNE_SAMPLE1000_EARLY0,
            *HIERARCHY_BUDGET_RECIPES,
        ),
    ),
    "hierarchy-depth": GallerySet(
        "hierarchy-depth",
        (
            TSNE_EARLY100,
            TSNE50_PLAIN,
            TSNE_SAMPLE1000_EARLY0,
            *HIERARCHY_DEPTH_RECIPES,
        ),
    ),
    "hierarchy-tuning": GallerySet(
        "hierarchy-tuning", (TSNE_CHAIN50_250_1000, *HIERARCHY_TUNING_RECIPES)
    ),
    "hierarchy-distribution": GallerySet(
        "hierarchy-distribution",
        (TSNE_CHAIN50_250_1000_FINAL50, *HIERARCHY_DISTRIBUTION_RECIPES),
    ),
    "hierarchy-heuristic": GallerySet(
        "hierarchy-heuristic",
        (TSNE_CHAIN50_250_1000_FINAL50, *HIERARCHY_HEURISTIC_RECIPES),
    ),
    "hierarchy-fractions": GallerySet(
        "hierarchy-fractions", (TSNE_GEO8_COMBINED, *HIERARCHY_FRACTION_RECIPES)
    ),
    "hierarchy-density": GallerySet(
        "hierarchy-density", (TSNE_GEO8_COMBINED, *HIERARCHY_DENSITY_RECIPES)
    ),
    "hierarchy-gamma": GallerySet(
        "hierarchy-gamma", (TSNE_DENSITY4, *HIERARCHY_GAMMA_RECIPES)
    ),
    "hierarchy-combination": GallerySet(
        "hierarchy-combination",
        (*COMBINATION_RECIPES, *COMBINATION_3D_RECIPES),
    ),
    # Externally generated sets: served here, but their artifacts come from the
    # genre-space ``maps --gallery-output`` and production-projection benchmark
    # CLIs, so this module must not pretend to know fixed recipe ids for them.
    "genre": GallerySet("genre", None),
    "genre-directed": GallerySet("genre-directed", None),
    "genre-logits-ab": GallerySet("genre-logits-ab", None),
    "folder-separation": GallerySet("folder-separation", None),
    "folder-cheap": GallerySet("folder-cheap", None),
}

#: Selectable registry, derived from :data:`GALLERY_SETS`: the original eight
#: plus every built-in extras recipe, keyed by id. An explicit recipe list may
#: address any of them, while an unspecified generation still runs the original
#: eight (:data:`RECIPES`).
RECIPES_BY_ID: dict[str, Recipe] = {
    recipe.id: recipe
    for gallery_set in GALLERY_SETS.values()
    for recipe in (gallery_set.recipes or ())
}


def recipe_params(recipe: Recipe) -> dict[str, Any]:
    """Deterministic dict of a recipe's fields, for the artifact record.

    Fields left at their ``None`` default are omitted so the optional
    ``perplexities`` field does not alter an existing recipe's recorded params.
    """
    return {
        f.name: getattr(recipe, f.name)
        for f in fields(recipe)
        if getattr(recipe, f.name) is not None
    }


def resolve_recipe_set(
    recipe_ids: list[str] | None,
    reference_id: str,
) -> list[str]:
    """The effective ordered recipe set for one generation.

    ``None`` selects the original eight (:data:`RECIPES`); an explicit list is
    kept in caller order. Either way the reference is auto-included first (every
    other map aligns onto it) when absent. ``reference_id`` must already be a
    registered id — the caller validates that before calling.
    """
    if recipe_ids is None:
        chosen = [r.id for r in RECIPES]
    else:
        unknown = [rid for rid in recipe_ids if rid not in RECIPES_BY_ID]
        if unknown:
            raise ValueError(f"unknown recipe id(s): {', '.join(unknown)}")
        chosen = list(recipe_ids)
    return [reference_id] + [rid for rid in chosen if rid != reference_id]


def validate_recipe_sources_for(
    chosen: list[str],
    sources: tuple[str, ...],
) -> None:
    """Reject a mix recipe that needs a source absent from the run's ``sources``.

    A source-mix recipe (:attr:`Recipe.source_weights` set) declares a complete
    weight vector over ``FROZEN_SOURCES``, but the composite gather only ever
    contributes the enabled ``sources``: a positive-weight source outside the
    subset would be silently dropped, so the fitted map would record a mix it
    never used. Only positive-weight sources matter — dropping a zero-weight
    block (``features`` at 0.0) cannot change the matrix.
    """
    for rid in chosen:
        pairs = RECIPES_BY_ID[rid].source_weights
        if pairs is None:
            continue
        missing = [
            src for src, w in pairs
            if float(w) > 0.0 and src not in sources
        ]
        if missing:
            raise ValueError(
                f"recipe {rid!r} needs source(s) absent from the run sources: "
                f"{', '.join(missing)} "
                f"(run sources: {', '.join(sources) or 'none'})"
            )


def _clamp_perplexity(requested: float, n_samples: int) -> float:
    # openTSNE needs k < n; clamp a requested perplexity down for small datasets
    # (a scalar or one scale of a multiscale mixture) so tiny fixtures stay valid.
    return min(float(requested), max(1.0, float(n_samples) - 1.0))


def _clamp_pca_dim(requested: int, n_samples: int, n_features: int) -> int:
    return max(1, min(int(requested), n_samples - 1, n_features - 1))


def _clamp_neighbors(requested: int, n_samples: int) -> int:
    return max(2, min(int(requested), n_samples - 1))


# ── fit primitives (uncached; never the production projection backends) ──────


def pca_reduce(mat: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, Any]:
    """Randomised-sklearn-PCA to *dim* components (clamped), returning (X, pca)."""
    from sklearn.decomposition import PCA  # lazy

    n_samples, n_features = mat.shape
    k = _clamp_pca_dim(dim, n_samples, n_features)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(mat), pca


def fit_opentsne(
    reduced: np.ndarray,
    *,
    recipe: Recipe,
    seed: int,
    n_jobs: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """openTSNE fit with the recipe's own (frozen) iteration budget.

    The recipe carries its ``tsne_n_iter`` / ``tsne_early_exaggeration_iter`` as a
    reproducible control; the *production* fit uses the separate
    ``projection_config.OPENTSNE_*`` iteration constants and is not routed here.
    ``recipe.dims`` selects the output dimension (``None`` = 2); 3-D fits pin the
    Barnes-Hut negative-gradient method, since openTSNE's FFT approximation is
    2-D only.

    ``init`` is a precomputed starting embedding (used by the UMAP-init recipe);
    it is scaled to the small variance openTSNE recommends for custom initials so
    the optimisation starts from a sane scale.
    """
    try:
        from openTSNE import TSNE
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "openTSNE is required for the layout-evaluation t-SNE recipes; "
            "install it (pip install opentsne)."
        ) from e

    dims = recipe.dims or 2
    base_kwargs: dict[str, Any] = dict(
        n_components=dims,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
        n_iter=recipe.tsne_n_iter,
        early_exaggeration_iter=recipe.tsne_early_exaggeration_iter,
    )
    if dims == 3:
        base_kwargs["negative_gradient_method"] = "bh"

    def _scaled_init() -> np.ndarray | None:
        if init is None:
            return None
        return scale_initialization(init)

    perps = recipe.perplexities
    if perps is not None and len(perps) > 1:
        # Multiscale: openTSNE's explicit-affinity mixture blends several
        # perplexities, then the t-SNE optimises against that affinity matrix.
        from openTSNE.affinity import MultiscaleMixture

        perplexities = [
            _clamp_perplexity(p, len(reduced)) for p in perps
        ]
        affinities = MultiscaleMixture(
            reduced,
            perplexities=perplexities,
            n_jobs=n_jobs,
            random_state=seed,
        )
        tsne = TSNE(**base_kwargs)
        scaled = _scaled_init()
        kwargs = dict(X=reduced, affinities=affinities)
        if scaled is not None:
            kwargs["initialization"] = scaled
        return np.asarray(tsne.fit(**kwargs), dtype=np.float32)

    # Scalar perplexity: a fixed one from the recipe, else the production rule.
    if perps is None:
        # Production's rule lives in projection_backends. The import is local:
        # that module pulls torch via ``clap`` at import time, and the heavy
        # libraries stay lazy in this module.
        from .projection_backends import _tsne_perplexity

        perplexity = _tsne_perplexity(len(reduced))
    else:
        perplexity = _clamp_perplexity(perps[0], len(reduced))
    kwargs: dict[str, Any] = {**base_kwargs, "perplexity": perplexity}
    tsne = TSNE(**kwargs)
    scaled = _scaled_init()
    if scaled is None:
        return np.asarray(tsne.fit(reduced), dtype=np.float32)
    return np.asarray(tsne.fit(reduced, initialization=scaled), dtype=np.float32)


def _tsne_sample_indices(n: int, sample_n: int, seed: int) -> np.ndarray:
    """Deterministic sorted sample of ``min(sample_n, n)`` row indices.

    A fresh rng seeded with *seed* makes the sample independent of any other
    stage's rng usage.
    """
    k = min(int(sample_n), n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def _normalise_sample_levels(
    levels: tuple[tuple[int, int, int], ...] | None,
    n: int,
) -> list[tuple[int, int, int]] | None:
    """Validate multilevel levels, keeping only the ones smaller than *n*.

    After the strictly-increasing positive-size / valid-budget validation, a
    level at or above *n* would duplicate the final full-data fit, so only
    ``size < n`` levels survive. ``None`` means nothing smaller than *n*
    remains — the caller then does a single full-data fit.
    """
    if levels is None:
        return None
    if not isinstance(levels, tuple) or not levels:
        raise ValueError(
            "tsne_sample_levels must be a non-empty tuple of (size, normal, early)"
        )
    prev_size = 0
    for entry in levels:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise ValueError(
                f"each tsne_sample_levels entry must be (size, normal, early): "
                f"{entry!r}"
            )
        size, normal, early = entry
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"tsne_sample_levels size must be a positive int: {size!r}")
        if size <= prev_size:
            raise ValueError("tsne_sample_levels sizes must be strictly increasing")
        prev_size = size
        if not isinstance(normal, int) or normal <= 0:
            raise ValueError(
                f"tsne_sample_levels 'normal' must be a positive int: {normal!r}"
            )
        if not isinstance(early, int) or early < 0:
            raise ValueError(
                f"tsne_sample_levels 'early' must be a nonnegative int: {early!r}"
            )
    return [lv for lv in levels if lv[0] < n] or None


def _validate_fraction_fields(recipe: Recipe) -> None:
    """Validate the size-adaptive fraction-mode fields of a multilevel recipe."""
    fracs = recipe.tsne_sample_fractions
    if not isinstance(fracs, tuple) or not fracs:
        raise ValueError(
            "tsne_sample_fractions must be a non-empty tuple of fractions"
        )
    if recipe.tsne_sample_levels is not None:
        raise ValueError(
            "tsne_sample_fractions and tsne_sample_levels are mutually exclusive"
        )
    prev = 0.0
    for f in fracs:
        if isinstance(f, bool) or not isinstance(f, (int, float)) \
                or not np.isfinite(float(f)):
            raise ValueError(f"tsne_sample_fractions must be finite: {f!r}")
        f = float(f)
        if not 0.0 < f < 1.0:
            raise ValueError(
                f"tsne_sample_fractions must satisfy 0 < f < 1: {f!r}"
            )
        if f <= prev:
            raise ValueError("tsne_sample_fractions must be strictly increasing")
        prev = f
    budget = recipe.tsne_coarse_iteration_budget
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"tsne_coarse_iteration_budget must be a positive int: {budget!r}"
            )
    gamma = recipe.tsne_perplexity_gamma
    if gamma is not None:
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) \
                or not np.isfinite(float(gamma)) or float(gamma) < 0.0:
            raise ValueError(
                "tsne_perplexity_gamma must be a nonnegative finite number: "
                f"{gamma!r}"
            )
    start = recipe.tsne_perplexity_start_fraction
    if start is not None:
        if isinstance(start, bool) or not isinstance(start, (int, float)) \
                or not np.isfinite(float(start)) or not 0.0 < float(start) < 1.0:
            raise ValueError(
                "tsne_perplexity_start_fraction must satisfy 0 < f < 1: "
                f"{start!r}"
            )
        if gamma is None:
            raise ValueError(
                "tsne_perplexity_start_fraction requires tsne_perplexity_gamma"
            )


def resolve_multilevel_schedule(
    recipe: Recipe,
    n: int,
) -> list[tuple[int, int, int, float | None]]:
    """Resolve a multilevel recipe's effective coarse schedule for *n* rows.

    Returns a list of ``(count, normal, early, perp)`` tuples — one per
    surviving coarse level (``count < n``), ascending — where ``count`` is the
    level's row count, ``normal``/``early`` its normal/exaggeration-iteration
    budget, and ``perp`` the level's explicit perplexity (``None`` = production
    scalar rule). The final full-data fit is NOT included: it uses the recipe's
    own ``tsne_n_iter`` / ``tsne_early_exaggeration_iter`` / ``perplexities``.

    Fixed-level recipes (``tsne_sample_levels``) return their validated
    ``(size, normal, early)`` levels plus the per-level perplexity when
    ``tsne_level_perplexities`` is set (perps paired against the *original*
    levels so a tiny ``n`` that drops a level never shifts a perplexity onto a
    wrong neighbour). Fraction-mode recipes (``tsne_sample_fractions``) resolve
    their counts/budgets/perplexities from *n*. An empty list means no coarse
    level survives — the caller does a single full-data fit.
    """
    if recipe.tsne_sample_fractions is not None:
        _validate_fraction_fields(recipe)
        return resolve_fraction_schedule(
            recipe.tsne_sample_fractions,
            n,
            normal_budget=recipe.tsne_coarse_iteration_budget,
            gamma=recipe.tsne_perplexity_gamma,
            start_fraction=recipe.tsne_perplexity_start_fraction,
        )
    # Fixed-level path: the pre-existing behavior, byte-identical schedule.
    if recipe.tsne_sample_levels is None:
        raise ValueError(
            "a tsne_multilevel_init recipe must define tsne_sample_levels or "
            "tsne_sample_fractions"
        )
    # Fraction-mode-only fields are orphaned on a fixed-level recipe: silently
    # ignoring them would record a schedule that never used them. Reject instead.
    if recipe.tsne_coarse_iteration_budget is not None:
        raise ValueError(
            "tsne_coarse_iteration_budget requires tsne_sample_fractions "
            "(fraction mode); it is not applied to fixed tsne_sample_levels"
        )
    if recipe.tsne_perplexity_gamma is not None:
        raise ValueError(
            "tsne_perplexity_gamma requires tsne_sample_fractions "
            "(fraction mode); it is not applied to fixed tsne_sample_levels"
        )
    if recipe.tsne_perplexity_start_fraction is not None:
        raise ValueError(
            "tsne_perplexity_start_fraction requires tsne_sample_fractions "
            "(fraction mode); it is not applied to fixed tsne_sample_levels"
        )
    level_perps = recipe.tsne_level_perplexities
    if level_perps is not None:
        if len(level_perps) != len(recipe.tsne_sample_levels):
            raise ValueError(
                "tsne_level_perplexities must have one scalar per tsne_sample_levels "
                f"(got {len(level_perps)} for {len(recipe.tsne_sample_levels)} levels)"
            )
        for p in level_perps:
            if isinstance(p, bool) or not isinstance(p, (int, float)) \
                    or not float(p) > 0 or not np.isfinite(float(p)):
                raise ValueError(
                    f"tsne_level_perplexities must be positive finite numbers: {p!r}"
                )
    levels = _normalise_sample_levels(recipe.tsne_sample_levels, n)
    if levels is None:
        return []
    if level_perps is None:
        return [(sz, nm, er, None) for (sz, nm, er) in levels]
    surviving_perps = [
        level_perps[i]
        for i, lv in enumerate(recipe.tsne_sample_levels)
        if lv[0] < n
    ]
    return [
        (sz, nm, er, p)
        for (sz, nm, er), p in zip(levels, surviving_perps)
    ]


def _fit_tsne_sample_init(
    reduced: np.ndarray,
    *,
    recipe: Recipe,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Two-level sample init: coarse fit, NN expand, full refinement.

    When ``sample_n >= n`` the coarse/expand stages are pointless, so it falls
    through to a direct full-data fit. Returns ``(coords, stage_timings)`` keyed
    ``tsne_coarse`` / ``tsne_expand`` / ``tsne_refine`` (seconds).
    """
    n = len(reduced)
    k = min(int(recipe.tsne_sample_n), n)
    if k >= n:
        t0 = time.perf_counter()
        coords = fit_opentsne(reduced, recipe=recipe, seed=seed, n_jobs=n_jobs)
        dt = time.perf_counter() - t0
        return coords, {"tsne_coarse": 0.0, "tsne_expand": 0.0, "tsne_refine": dt}

    idx = _tsne_sample_indices(n, int(recipe.tsne_sample_n), seed)
    coarse_recipe = replace(
        recipe,
        method="tsne",
        tsne_n_iter=recipe.tsne_sample_n_iter,
        tsne_early_exaggeration_iter=recipe.tsne_sample_early_exaggeration_iter,
    )

    t0 = time.perf_counter()
    coarse = fit_opentsne(
        reduced[idx], recipe=coarse_recipe, seed=seed, n_jobs=n_jobs
    )
    coarse_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    expanded = _expand_landmark_init(
        reduced, idx, coarse, np.arange(n), seed=seed, n_jobs=n_jobs
    )
    expand_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    coords = fit_opentsne(
        reduced, recipe=recipe, seed=seed, n_jobs=n_jobs, init=expanded
    )
    refine_t = time.perf_counter() - t0
    return coords, {
        "tsne_coarse": coarse_t,
        "tsne_expand": expand_t,
        "tsne_refine": refine_t,
    }


def _fit_tsne_multilevel_init(
    reduced: np.ndarray,
    *,
    recipe: Recipe,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Multilevel nested sample init: coarse fit, then expand+refine up, then full.

    The effective coarse schedule comes from :func:`resolve_multilevel_schedule`
    (fixed ``tsne_sample_levels`` or size-adaptive ``tsne_sample_fractions``),
    so the fixed-level path is byte-identical to before the fraction fields
    existed. A missing schedule is rejected (not a silent plain fallback); when
    no level is smaller than the dataset the coarse chain is pointless, so it
    falls through to a single full-data fit. Returns ``(coords, stage_t)`` keyed
    ``tsne_level0..`` / ``tsne_expand`` / ``tsne_final`` (seconds).
    """
    schedule = resolve_multilevel_schedule(recipe, len(reduced))

    def _level_recipe(normal: int, early: int, perp: float | None) -> Recipe:
        if perp is None:
            # Production-rule level: no ``perplexities`` change, so a
            # production-scalar level fit is byte-identical to before the
            # per-level-perplexity field existed.
            return replace(
                recipe, method="tsne", tsne_n_iter=normal,
                tsne_early_exaggeration_iter=early,
            )
        return replace(
            recipe, method="tsne", tsne_n_iter=normal,
            tsne_early_exaggeration_iter=early, perplexities=(float(perp),),
        )

    def _fit_level(
        subset: np.ndarray,
        normal: int,
        early: int,
        perplexity: float | None,
        level_init: np.ndarray | None,
    ) -> np.ndarray:
        return fit_opentsne(
            subset,
            recipe=_level_recipe(normal, early, perplexity),
            seed=seed,
            n_jobs=n_jobs,
            init=level_init,
        )

    def _fit_final(full: np.ndarray, final_init: np.ndarray | None) -> np.ndarray:
        return fit_opentsne(
            full, recipe=recipe, seed=seed, n_jobs=n_jobs, init=final_init
        )

    _final, coords, stage_t = fit_multilevel(
        reduced,
        schedule,
        fit_level=_fit_level,
        fit_final=_fit_final,
        seed=seed,
        n_jobs=n_jobs,
    )
    return coords, stage_t


def fit_umap(
    reduced: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    """umap-learn 2-D (or intermediate) fit. Neighbours are clamped for small n."""
    try:
        import umap
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "umap-learn is required for the layout-evaluation UMAP recipes; "
            "install it (pip install umap-learn)."
        ) from e
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=_clamp_neighbors(n_neighbors, len(reduced)),
        min_dist=float(min_dist),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )
    # UMAP's spectral initialisation needs dim + 1 < n_samples (its eigsh uses
    # k = dim + 1); on a tiny dataset (smoke tests) clamp the intermediate-dim
    # recipes down rather than letting that step raise. At real-library scale
    # this never triggers.
    if reducer.n_components >= len(reduced) - 1:
        reducer.n_components = max(1, len(reduced) - 2)
    return np.asarray(reducer.fit_transform(reduced), dtype=np.float32)


def fit_recipe(mat: np.ndarray, recipe: Recipe, *, seed: int, n_jobs: int):
    """Fit one recipe's map from the (already gathered) composite *mat*.

    Returns ``(coords, timings)`` where ``coords`` is ``(n, recipe.dims or 2)``
    and timings has per-stage seconds (each recipe runs its own PCA, so the fair
    total includes the prerequisite cost). Only the multilevel t-SNE method
    supports 3-D; the other methods stay explicitly 2-D.
    """
    dims = recipe.dims or 2
    if dims not in (2, 3):
        raise ValueError(f"recipe {recipe.id!r} has unsupported dims {dims!r}")
    if dims == 3 and recipe.method != "tsne_multilevel_init":
        raise ValueError(
            f"recipe {recipe.id!r} requests 3-D output but method "
            f"{recipe.method!r} is 2-D only"
        )
    timings: dict[str, Any] = {}
    t0 = time.perf_counter()
    reduced, _pca = pca_reduce(mat, recipe.pca_dim, seed)
    timings["pca"] = time.perf_counter() - t0
    timings["tsne"] = 0.0
    timings["umap"] = 0.0

    if recipe.method == "pca":
        # Pure PCA map: the reduced coordinates already are the 2-D embedding.
        # A tiny dataset may clamp PCA to a single component (see
        # ``_clamp_pca_dim``); pad to a 2-D row so the map stays comparable.
        coords = np.asarray(reduced, dtype=np.float32)
        if coords.shape[1] < 2:
            pad = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=np.float32)
            coords = np.concatenate([coords, pad], axis=1)
    elif recipe.method == "tsne":
        t0 = time.perf_counter()
        coords = fit_opentsne(reduced, recipe=recipe, seed=seed, n_jobs=n_jobs)
        timings["tsne"] = time.perf_counter() - t0
    elif recipe.method == "tsne_sample_init":
        # ``tsne`` is the whole coarse+expand+refine pipeline; the per-stage keys
        # are informational and never summed into ``total`` again.
        t0 = time.perf_counter()
        coords, stage_t = _fit_tsne_sample_init(
            reduced, recipe=recipe, seed=seed, n_jobs=n_jobs
        )
        timings["tsne"] = time.perf_counter() - t0
        timings.update(stage_t)
    elif recipe.method == "tsne_multilevel_init":
        # ``tsne`` is the whole chain (level fits + expansions + final); the
        # per-level stage keys are informational — never summed into ``total``.
        t0 = time.perf_counter()
        coords, stage_t = _fit_tsne_multilevel_init(
            reduced, recipe=recipe, seed=seed, n_jobs=n_jobs
        )
        timings["tsne"] = time.perf_counter() - t0
        timings.update(stage_t)
    elif recipe.method == "umap":
        t0 = time.perf_counter()
        coords = fit_umap(
            reduced,
            n_components=recipe.umap_n_components,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
    elif recipe.method == "umap_tsne":
        t0 = time.perf_counter()
        intermediate = fit_umap(
            reduced,
            n_components=recipe.umap_n_components,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        coords = fit_opentsne(intermediate, recipe=recipe, seed=seed, n_jobs=n_jobs)
        timings["tsne"] = time.perf_counter() - t0
    elif recipe.method == "tsne_umap_init":
        t0 = time.perf_counter()
        init = fit_umap(
            reduced,
            n_components=2,
            n_neighbors=recipe.umap_n_neighbors,
            min_dist=recipe.umap_min_dist,
            seed=seed,
            n_jobs=n_jobs,
        )
        timings["umap"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        coords = fit_opentsne(reduced, recipe=recipe, seed=seed, n_jobs=n_jobs, init=init)
        timings["tsne"] = time.perf_counter() - t0
    else:  # pragma: no cover - recipe set is fixed
        raise ValueError(f"unknown recipe method {recipe.method!r}")

    if not np.isfinite(coords).all():
        raise ValueError(f"recipe {recipe.id!r} produced non-finite coordinates")
    return coords, timings


# ── alignment + normalisation ─────────────────────────────────────────────────


def orthogonal_align_to(
    coords: np.ndarray,
    paths: list[str],
    ref_coords: np.ndarray,
    ref_paths: list[str],
) -> np.ndarray:
    """Rotate/reflect *coords* onto *ref_coords* using the shared tracks.

    Only an orthogonal (rotation + reflection) transform with translation — no
    scale, no warp — so the map's own geometry is preserved while its frame is
    made comparable to the reference. Both maps must have the same width: a 3-D
    map aligns onto the 3-D reference, never a 2-D one (the gauge is
    dimension-specific). Raises on a non-finite result (a degenerate alignment
    must not write NaN positions into an artifact).
    """
    if coords.shape[1] != ref_coords.shape[1]:
        raise ValueError(
            f"cannot align a {coords.shape[1]}-D map onto a "
            f"{ref_coords.shape[1]}-D reference"
        )
    idx_new, idx_ref = matched_indices(paths, ref_paths)
    if len(idx_new) < 2:
        return coords
    rot, _scale, src_mu, dst_mu = procrustes(
        coords[idx_new], ref_coords[idx_ref]
    )
    aligned = apply_procrustes(coords, rot, 1.0, src_mu, dst_mu)
    if not np.all(np.isfinite(aligned)):
        raise ValueError(
            "alignment produced non-finite coordinates "
            f"({len(idx_new)} shared tracks)"
        )
    return aligned


def normalise_aspect(coords: np.ndarray) -> np.ndarray:
    """Center *coords* and scale it uniformly into the unit square.

    ``[0, 1]`` in both axes with the aspect ratio preserved — a *uniform* scale
    (never a per-axis stretch), applied identically to every map.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 1e-12:
        span = 1.0
    center = (mins + maxs) / 2.0
    return ((coords - center) / span + 0.5).astype(np.float32)


# ── identity hashes ───────────────────────────────────────────────────────────


def dataset_fingerprint(paths: list[str], mat: np.ndarray, comp_config: str) -> str:
    """Stable identity of one generation: input vectors + paths + params + versions.

    ``comp_config`` already folds in the source versions token and the full
    vector config (sources, mask, blend, weights), so hashing it plus the ordered
    paths and the matrix bytes makes the fingerprint change whenever any input
    changes and stay stable otherwise. The paths must arrive in the same row
    order as ``mat`` — the caller sorts before gathering, and this function never
    sorts them itself, because the digest must track the actual row order.
    """
    h = hashlib.sha256()
    h.update(comp_config.encode("utf-8"))
    for p in paths:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\0")
    h.update(np.ascontiguousarray(mat, dtype=np.float32).tobytes())
    return h.hexdigest()


def recipe_revision(
    dataset_fp: str,
    recipe: Recipe,
    *,
    seed: int = 42,
    n_jobs: int = -1,
    versions: dict[str, Any] | None = None,
) -> str:
    """Per-recipe identity: dataset fingerprint + recipe params + effective environment.

    The dataset fingerprint captures the *input* (paths, vectors, vector config);
    it deliberately stays unrelated to the environment. But the fitted map also
    depends on the seed, the embedding-source / projection-library versions, and
    the worker count — approximate-neighbour stages (UMAP) can vary with it — so
    those are folded in here too: a seed, thread-count, or library change must
    produce a different revision even when the input is identical.
    """
    h = hashlib.sha256(dataset_fp.encode("utf-8"))
    h.update(f"seed={seed}".encode("utf-8"))
    h.update(b"\0")
    h.update(f"n_jobs={n_jobs}".encode("utf-8"))
    h.update(b"\0")
    if versions:
        for k in sorted(versions):
            h.update(k.encode("utf-8"))
            h.update(repr(versions[k]).encode("utf-8"))
            h.update(b"\0")
    for f in fields(recipe):
        value = getattr(recipe, f.name)
        if value is None:
            # Optional fields at their ``None`` default are not part of the
            # identity, so the pre-existing recipes keep their exact revisions.
            continue
        h.update(f.name.encode("utf-8"))
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _source_versions(feature_cache) -> dict[str, int]:
    from .clap import EMBEDDING_VERSION
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_VERSION
    from .rhythm_features import RHYTHM_VERSION
    from .supervised import cache_version

    return {
        "clap": EMBEDDING_VERSION,
        "features": FEATURES_VERSION,
        "maest": MAEST_VERSION,
        "rhythm": RHYTHM_VERSION,
        "supervised": cache_version(feature_cache),
    }


def _backend_info() -> dict[str, str]:
    import importlib
    import platform

    info: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for key, module_name in (
        ("sklearn", "sklearn"),
        ("openTSNE", "openTSNE"),
        ("umap", "umap"),
    ):
        try:
            info[key] = importlib.import_module(module_name).__version__
        except Exception:
            pass
    return info


# ── atomic writes ─────────────────────────────────────────────────────────────


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write *obj* atomically; refuse non-finite floats rather than emit NaN."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # A failed write must not leave a partial temp file behind.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ── generation ────────────────────────────────────────────────────────────────


def build_manifest(
    *,
    status: str,
    dataset_fingerprint: str,
    count: int,
    sources: tuple[str, ...],
    source_weights: dict[str, float],
    required_sources: tuple[str, ...],
    feature_mask: np.ndarray,
    features_blend: float,
    seed: int,
    n_jobs: int | None,
    reference_id: str,
    reference_ids: dict[int, str],
    backend: dict[str, str],
    versions: dict[str, int],
    recipes: dict[str, dict],
    timings: dict[str, float],
    notes: dict[str, str] | None = None,
) -> dict:
    """Assemble the generation manifest dict (the caller writes it atomically).

    ``backend`` / ``versions`` are the already-computed environment snapshots
    (projection-library versions vs. embedding-source versions); ``recipes`` is
    the per-recipe result dict keyed by id, in fit order. ``reference_id`` stays
    the gallery's 2-D reference; ``reference_ids`` records the per-dimension
    reference actually used (``{2: …, 3: …}``). ``timings`` is the caller's
    measured breakdown (a shared gather/PCA plus per-artifact fits and
    alignments), never a fabricated per-artifact sum.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset": {"count": count, "fingerprint": dataset_fingerprint},
        "sources": list(sources),
        "source_weights": {k: float(v) for k, v in source_weights.items()},
        "required_sources": list(required_sources),
        "feature_mask": [int(x) for x in feature_mask],
        "features_blend": float(features_blend),
        "seed": seed,
        **({"n_jobs": int(n_jobs)} if n_jobs is not None else {}),
        "reference_id": reference_id,
        "reference_ids": {str(dim): rid for dim, rid in reference_ids.items()},
        "backend": backend,
        "versions": versions,
        "notes": dict(notes or {}),
        "recipes": {
            rid: {
                "status": r["status"],
                **({"reason": r.get("reason")} if r.get("reason") else {}),
                **(
                    {
                        "label": r["label"],
                        "file": f"{rid}.json",
                        "revision": r["revision"],
                        "tracks": r["tracks"],
                        "timings": r["timings"],
                        "aligned_to": r.get("aligned_to"),
                        "input_fingerprint": r["input_fingerprint"],
                        "source_weights": r["source_weights"],
                        "dims": r["dims"],
                        "group": r["group"],
                    }
                    if r["status"] == "complete"
                    else {}
                ),
            }
            for rid, r in recipes.items()
        },
        "timings": dict(timings),
        "created_at": _iso_now(),
    }


def _fit_one_recipe(
    *,
    rid: str,
    recipe: Recipe,
    mat: np.ndarray,
    fp: str,
    gather_t: float,
    variants: dict[str, tuple[np.ndarray, str, float] | None],
    paths: list[str],
    n: int,
    ref_id: str,
    ref_state: tuple[list[str], np.ndarray] | None,
    seed: int,
    n_jobs: int,
    env_versions: dict[str, Any],
    source_weights: dict[str, float],
    output_dir: str,
) -> tuple[dict, tuple[list[str], np.ndarray] | None]:
    """Fit, align, normalise and write ONE recipe's artifact.

    Returns ``(manifest entry, updated reference state)``. The reference state is
    ``(ref_paths, ref_coords)`` when *rid* is the reference recipe (the caller
    keeps it for the remaining recipes), else ``None``. Any failure — a missing
    dependency, a non-finite fit or alignment, a write error — records a clean
    per-recipe error entry and the run continues.
    """
    if n < MIN_TRACKS:
        return (
            {
                "status": "skipped",
                "reason": (
                    f"too few tracks ({n} < {MIN_TRACKS}): the t-SNE perplexity "
                    "clamp cannot hold"
                ),
            },
            None,
        )
    if recipe.source_weights is not None:
        if rid not in variants or variants[rid] is None:
            return (
                {
                    "status": "error",
                    "reason": (
                        "variant input path set diverged from the control; "
                        "refusing to fit/align on a different track universe"
                    ),
                },
                None,
            )
        mat_rid, input_fp_rid, gather_rid = variants[rid]
    else:
        mat_rid, input_fp_rid, gather_rid = mat, fp, gather_t

    ref_paths = ref_state[0] if ref_state is not None else None
    ref_coords = ref_state[1] if ref_state is not None else None
    try:
        coords, timings = fit_recipe(mat_rid, recipe, seed=seed, n_jobs=n_jobs)
        timings["gather"] = gather_rid
        aligned_to: str | None = None
        new_ref: tuple[list[str], np.ndarray] | None = None
        if rid == ref_id:
            new_ref = (list(paths), coords)
            timings["alignment"] = 0.0
        else:
            t0 = time.perf_counter()
            if ref_coords is not None and ref_paths is not None:
                coords = orthogonal_align_to(coords, paths, ref_coords, ref_paths)
                aligned_to = ref_id
            timings["alignment"] = time.perf_counter() - t0

        normed = normalise_aspect(coords)
        dims = recipe.dims or 2
        if dims == 3:
            positions = [
                {
                    "path": paths[i],
                    "x": float(normed[i, 0]),
                    "y": float(normed[i, 1]),
                    "z": float(normed[i, 2]),
                }
                for i in range(n)
            ]
        else:
            positions = [
                {"path": paths[i], "x": float(normed[i, 0]), "y": float(normed[i, 1])}
                for i in range(n)
            ]
        timings["total"] = (
            timings.get("pca", 0.0)
            + timings.get("tsne", 0.0)
            + timings.get("umap", 0.0)
            + timings.get("alignment", 0.0)
        )
        # The revision keys on the recipe's *actual* input fingerprint plus its
        # explicit params (including the source mix), so a different mix for the
        # same gallery dataset gets its own revision and rating identity.
        revision = recipe_revision(
            input_fp_rid, recipe, seed=seed, n_jobs=n_jobs, versions=env_versions
        )
        actual_weights = {
            k: float(v)
            for k, v in (
                dict(recipe.source_weights)
                if recipe.source_weights is not None
                else source_weights
            ).items()
        }
        artifact = {
            "id": rid,
            "label": recipe.label,
            "revision": revision,
            "dataset_fingerprint": fp,
            "input_fingerprint": input_fp_rid,
            "source_weights": actual_weights,
            "recipe": recipe_params(recipe),
            "tracks": n,
            "aligned_to": aligned_to,
            "dims": dims,
            "group": recipe.group or rid,
            "timings": timings,
            "positions": positions,
        }
        _atomic_write_json(os.path.join(output_dir, f"{rid}.json"), artifact)
    except MissingDependencyError as e:
        return {"status": "error", "reason": str(e)}, None
    except Exception as e:
        log.exception("recipe %s failed", rid)
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}, None
    return (
        {
            "status": "complete",
            "label": recipe.label,
            "revision": revision,
            "tracks": n,
            "timings": timings,
            "aligned_to": aligned_to,
            "input_fingerprint": input_fp_rid,
            "source_weights": actual_weights,
            "dims": dims,
            "group": recipe.group or rid,
        },
        new_ref,
    )


def run_generation(
    *,
    feature_cache,
    infos: list[dict],
    output_dir: str,
    sources: tuple[str, ...],
    source_weights: dict[str, float],
    required_sources: tuple[str, ...],
    feature_mask: np.ndarray,
    features_blend: float,
    seed: int = 42,
    n_jobs: int = -1,
    recipe_ids: list[str] | None = None,
    reference_id: str | None = None,
    limit: int | None = None,
    on_progress: Callable | None = None,
) -> dict[str, Any]:
    """Gather once, fit every requested recipe, and write the artifacts.

    Every non-reference map is aligned (orthogonal rotation/reflection only) onto
    the ``reference_id`` recipe, which is fitted first and auto-included if it was
    not already requested. Default reference is :data:`BASELINE_ID`.

    ``on_progress(stage, done, total)`` receives coarse progress (a string stage
    and counts) for the CLI's verbose output.

    Returns the final manifest dict. An empty ``in_progress`` manifest is
    written before the control gather, so a reader can never serve recipes from a
    previous generation while this one is preparing. The final status is
    ``complete`` when every requested recipe completed, ``partial`` when some
    did, and ``failed`` when none did; per-recipe serving rules are unchanged.
    """
    from .layout import prepare_projection_input

    os.makedirs(output_dir, exist_ok=True)

    ref_id = reference_id or BASELINE_ID
    if ref_id not in RECIPES_BY_ID:
        raise ValueError(
            f"unknown reference recipe id: {ref_id!r} "
            f"(choose from {', '.join(RECIPES_BY_ID)})"
        )
    if (RECIPES_BY_ID[ref_id].dims or 2) != 2:
        raise ValueError(
            f"reference recipe {ref_id!r} must be 2-D; 3-D maps align onto their "
            "explicit <reference>_3d sibling"
        )

    chosen = resolve_recipe_set(recipe_ids, ref_id)

    # Cross-dimension runs: the 2-D reference stays the gallery reference, and
    # every 3-D map aligns onto the explicit ``<reference>_3d`` sibling. Only the
    # registered sibling is used (no suffix parser for arbitrary ids); it is
    # auto-included when absent and fitted after the whole 2-D group so each
    # dimension's reference state is independent.
    reference_ids: dict[int, str] = {2: ref_id}
    if any((RECIPES_BY_ID[rid].dims or 2) == 3 for rid in chosen):
        ref_3d = f"{ref_id}_3d"
        if ref_3d not in RECIPES_BY_ID:
            raise ValueError(
                f"3-D recipes requested but reference {ref_id!r} has no "
                f"registered 3-D sibling {ref_3d!r}"
            )
        reference_ids[3] = ref_3d
        chosen_2d = [rid for rid in chosen if (RECIPES_BY_ID[rid].dims or 2) == 2]
        chosen_3d = [rid for rid in chosen if (RECIPES_BY_ID[rid].dims or 2) == 3]
        chosen = (
            chosen_2d
            + [ref_3d]
            + [rid for rid in chosen_3d if rid != ref_3d]
        )

    # A mix recipe declares a complete weight vector; when the run gathers a
    # *subset* of sources, any positive-weight source outside that subset would be
    # silently dropped (only enabled sources contribute). Reject the run before
    # any gather or artifact write rather than record a map that claims a mix it
    # never actually used.
    validate_recipe_sources_for(chosen, sources)

    # Environment that determines each fitted map: the embedding-source versions
    # (a source's cached vectors) and the projection-library versions (the fit
    # primitives). Folded into every recipe's revision so a changed environment
    # invalidates a previous run's artifact even with identical input.
    source_versions = _source_versions(feature_cache)
    backend_info = _backend_info()
    env_versions: dict[str, Any] = {**source_versions, **backend_info}
    notes: dict[str, str] = {}
    if any("umap" in RECIPES_BY_ID[rid].method for rid in chosen):
        notes["umap_jit"] = (
            "UMAP's first call includes a Numba JIT compile, so its "
            "timing on a fresh process is inflated."
        )

    # Invalidate a previous generation's ``complete`` manifest before the
    # expensive gather. Configuration is already validated above, so this write
    # never follows a rejected run; the fingerprint/count/timings are filled in
    # once the control gather has produced them.
    manifest_state: dict[str, Any] = {
        "fingerprint": "",
        "count": 0,
        "gather": 0.0,
        "fit_start": None,
    }

    def _write_manifest(status: str) -> None:
        fit_start = manifest_state["fit_start"]
        manifest = build_manifest(
            status=status,
            dataset_fingerprint=manifest_state["fingerprint"],
            count=manifest_state["count"],
            sources=sources,
            source_weights=source_weights,
            required_sources=required_sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            seed=seed,
            n_jobs=n_jobs,
            reference_id=ref_id,
            reference_ids=reference_ids,
            backend=backend_info,
            versions=source_versions,
            recipes=results,
            timings={
                "gather": manifest_state["gather"],
                "total": (
                    0.0
                    if fit_start is None
                    else time.perf_counter() - fit_start
                ),
            },
            notes=notes,
        )
        _atomic_write_json(os.path.join(output_dir, "manifest.json"), manifest)

    results: dict[str, dict] = {}
    _write_manifest("in_progress")

    # 1. Control gather with the passed (default) weights → the gallery dataset
    # fingerprint. Every recipe without its own ``source_weights`` fits this
    # matrix, and the reference (default) map always does, so the artifact
    # ``dataset_fingerprint`` stays the gallery's control fingerprint.
    t_gather = time.perf_counter()
    inp = prepare_projection_input(
        infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        source_weights=source_weights,
        required_sources=required_sources,
    )
    gather_t = time.perf_counter() - t_gather

    paths = list(inp.paths)
    mat = inp.mat
    order: list[int] | None = None
    if limit is not None and limit > 0 and len(paths) > limit:
        order = sorted(range(len(paths)), key=lambda i: paths[i])
        order = order[: int(limit)]
        paths = [paths[i] for i in order]
        mat = mat[order]
    n = len(paths)

    comp_config = inp.comp_config
    fp = dataset_fingerprint(paths, mat, comp_config)

    # 2. Variant gathers: each recipe with its own explicit source mix gets its
    # own composite matrix through the same shared production helper, with only
    # the weights overlaid — no other parameter changes. The variant's track
    # universe must match the control's exactly (same paths, same order): a
    # different set would make the maps incomparable and is rejected outright,
    # never silently sliced or faked to the control's center.
    variants: dict[str, tuple[np.ndarray, str, float] | None] = {}
    for rid in chosen:
        recipe = RECIPES_BY_ID[rid]
        if recipe.source_weights is None:
            continue
        t_variant = time.perf_counter()
        inp_v = prepare_projection_input(
            infos,
            feature_cache,
            sources=sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            source_weights=dict(recipe.source_weights),
            required_sources=required_sources,
        )
        v_gather = time.perf_counter() - t_variant
        v_paths = list(inp_v.paths)
        v_mat = inp_v.mat
        if order is not None and len(v_paths) > limit:
            v_paths = [v_paths[i] for i in order]
            v_mat = v_mat[order]
        if v_paths != paths:
            # Path-set mismatch: reject the variant, no alignment/slicing/fake
            # center. It is recorded as an error in the fit loop below.
            variants[rid] = None
            continue
        v_fp = dataset_fingerprint(v_paths, v_mat, inp_v.comp_config)
        variants[rid] = (v_mat, v_fp, v_gather)

    def _progress(stage: str, done: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    manifest_state.update(fingerprint=fp, count=n, gather=gather_t)
    manifest_state["fit_start"] = time.perf_counter()
    # Per-dimension reference state: a 3-D map must never align onto the 2-D
    # reference (or vice versa), so each dimension keeps its own paths/coords.
    ref_states: dict[int, tuple[list[str], np.ndarray] | None] = {}

    _write_manifest("in_progress")
    _progress("gather", 0, n)

    for i, rid in enumerate(chosen):
        _progress(rid, i, len(chosen))
        recipe = RECIPES_BY_ID[rid]
        dims = recipe.dims or 2
        entry, new_ref = _fit_one_recipe(
            rid=rid,
            recipe=recipe,
            mat=mat,
            fp=fp,
            gather_t=gather_t,
            variants=variants,
            paths=paths,
            n=n,
            ref_id=reference_ids[dims],
            ref_state=ref_states.get(dims),
            seed=seed,
            n_jobs=n_jobs,
            env_versions=env_versions,
            source_weights=source_weights,
            output_dir=output_dir,
        )
        if new_ref is not None:
            ref_states[dims] = new_ref
        results[rid] = entry
        _write_manifest("in_progress")
        _progress(rid, i + 1, len(chosen))

    complete = sum(1 for entry in results.values() if entry.get("status") == "complete")
    if complete == len(chosen):
        status = "complete"
    elif complete:
        status = "partial"
    else:
        status = "failed"
    _write_manifest(status)
    _progress(status, len(chosen), len(chosen))
    return manifest_dict(output_dir)


def manifest_dict(output_dir: str) -> dict | None:
    """Read the current manifest, or ``None`` when it does not exist or is malformed."""
    path = os.path.join(output_dir, "manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_completed(output_dir: str) -> dict[str, dict]:
    """Completed recipes of the current manifest, whatever its overall status.

    A long generation writes the manifest ``in_progress`` while each recipe's
    artifact lands, so the gallery can already show the finished maps of the
    current run. The manifest file is replaced atomically, so it always belongs
    to the current generation; staleness is guarded per-recipe by the
    fingerprint/revision check in :func:`load_recipe_json`. An unfinished recipe
    (anything not ``complete``) is excluded.
    """
    manifest = manifest_dict(output_dir)
    if not manifest:
        return {}
    recipes = manifest.get("recipes") or {}
    return {
        rid: r
        for rid, r in recipes.items()
        if isinstance(r, dict) and r.get("status") == "complete"
    }


def load_recipe_json(output_dir: str, recipe_id: str) -> dict | None:
    """Read and validate one completed recipe artifact, or ``None``.

    The current manifest may be ``in_progress``/``partial``/``failed`` during a
    long generation: a completed recipe of that same run is still served, but
    only when its dataset fingerprint *and* revision match the manifest's, so a
    stale file from a previous dataset or environment can never be served. A
    recipe the manifest does not mark ``complete`` is rejected.
    """
    manifest = manifest_dict(output_dir)
    if not manifest:
        return None
    recipes = manifest.get("recipes") or {}
    entry = recipes.get(recipe_id)
    if not isinstance(entry, dict) or entry.get("status") != "complete":
        return None
    expected_fp = manifest.get("dataset_fingerprint")
    expected_rev = entry.get("revision")
    path = os.path.join(output_dir, f"{recipe_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("positions"), list):
        return None
    if data.get("id") != recipe_id:
        return None
    if expected_fp and data.get("dataset_fingerprint") != expected_fp:
        return None
    if expected_rev and data.get("revision") != expected_rev:
        return None
    # A source-mix variant records its own actual input fingerprint (distinct
    # from the gallery dataset fingerprint). When the manifest entry records one,
    # the artifact must match it; a manifest/artifact pair from before this field
    # existed (neither records it) still passes the older checks above.
    expected_input_fp = entry.get("input_fingerprint")
    if expected_input_fp and data.get("input_fingerprint") != expected_input_fp:
        return None
    expected_dims = entry.get("dims", 2)
    data_dims = data.get("dims", 2)
    if expected_dims not in (2, 3) or data_dims != expected_dims:
        return None
    # The paired-family id must match too (defaulting to the recipe id on older
    # artifacts), so a 2-D artifact cannot masquerade as its 3-D sibling's group.
    expected_group = entry.get("group", recipe_id)
    if data.get("group", recipe_id) != expected_group:
        return None
    positions = data["positions"]
    for pos in positions:
        if not isinstance(pos, dict):
            return None
        x, y = pos.get("x"), pos.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if data_dims == 3:
            # A 3-D artifact must carry a finite z on every position. 2-D
            # artifacts ignore any legacy extra ``z`` key.
            z = pos.get("z")
            if not isinstance(z, (int, float)) or not np.isfinite(z):
                return None
    return data


def _iso_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
