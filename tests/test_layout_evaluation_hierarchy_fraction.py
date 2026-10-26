"""Size-adaptive fraction-mode multilevel galleries: resolver, validation, routes.

Adds the ``Recipe.tsne_sample_fractions`` / ``tsne_coarse_iteration_budget`` /
``tsne_perplexity_gamma`` fields (size-adaptive multilevel chains) and the
``HIERARCHY_FRACTION_RECIPES`` / ``HIERARCHY_DENSITY_RECIPES`` sets, plus the
public ``resolve_multilevel_schedule`` resolver both the fit and the scaling
benchmark now share.

Covers: field/recipe invariants, fraction validation (finite / strict / range /
mutual exclusion), the count + budget + perplexity resolver (auto vs fixed
budget, largest-remainder determinism, gamma schedule + clamping, tiny-N
level-filtering with no duplicate full fit), the unchanged fixed-level / ``None``
paths, and a real openTSNE smoke. Both sets' generation and route isolation are
covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("openTSNE")

from dataclasses import replace

import backend.embeddings.layout_evaluation as le

REFERENCE_ID = "tsne_geo8_combined"

FRACTION_SET_IDS = [
    "tsne_fraction_g4",
    "tsne_fraction_g8",
    "tsne_fraction_g16",
]
DENSITY_SET_IDS = [
    "tsne_density2",
    "tsne_density4",
    "tsne_density8",
    "tsne_density16",
    "tsne_continuous16",
]

N_LIB = 7153  # the current library size used for the resolver expectations.


# ── recipe invariants ────────────────────────────────────────────────────────


def test_fraction_recipe_parameters():
    # Every fraction/density map: PCA50 multilevel, final 50/0 at explicit perp 30,
    # no fixed absolute levels, flat per-level perplexity 30 (gamma None).
    expected_fractions = {
        "tsne_fraction_g4": (1 / 16, 1 / 4),
        "tsne_fraction_g8": (1 / 64, 1 / 8),
        "tsne_fraction_g16": (1 / 256, 1 / 16),
        "tsne_density2": (1 / 64, 1 / 8),
        "tsne_density4": le._FRACTIONS_DENSITY4,
        "tsne_density8": le._FRACTIONS_DENSITY8,
        "tsne_density16": le._FRACTIONS_DENSITY16,
        "tsne_continuous16": le._FRACTIONS_DENSITY16,
    }
    expected_budget = {
        **{rid: None for rid in FRACTION_SET_IDS},
        "tsne_density2": 541,
        "tsne_density4": 541,
        "tsne_density8": 541,
        "tsne_density16": 541,
        "tsne_continuous16": 16,
    }
    for rid in [*FRACTION_SET_IDS, *DENSITY_SET_IDS]:
        r = le.RECIPES_BY_ID[rid]
        assert r.method == "tsne_multilevel_init"
        assert r.pca_dim == 50
        assert r.source_weights is None
        assert r.tsne_sample_levels is None
        assert r.tsne_sample_fractions == expected_fractions[rid]
        assert r.tsne_coarse_iteration_budget == expected_budget[rid]
        assert r.tsne_perplexity_gamma is None
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
        assert r.perplexities == (30.0,)


# ── validation ───────────────────────────────────────────────────────────────


def test_fraction_validation_rejects_bad_fields():
    base = le.RECIPES_BY_ID["tsne_fraction_g8"]
    bad_fracs = [
        (),                                   # empty
        (0.5, float("nan")),                  # NaN
        (0.5, float("inf")),                  # +inf
        (0.5, float("-inf")),                 # -inf
        (0.0, 0.5),                           # zero
        (1.0, 0.5),                           # one
        (-0.5, 0.5),                          # negative
        (1.5, 0.5),                           # > 1
        (0.5, 0.5),                           # not strictly increasing
        (0.8, 0.5),                           # decreasing
        (0.5, True),                          # bool is not a fraction
        (0.5, "0.25"),                        # not a number
    ]
    for fracs in bad_fracs:
        with pytest.raises(ValueError):
            le.resolve_multilevel_schedule(replace(base, tsne_sample_fractions=fracs), 100)
    # Mutual exclusion with the fixed absolute-level field.
    with pytest.raises(ValueError):
        le.resolve_multilevel_schedule(
            replace(base, tsne_sample_levels=((10, 10, 10), (50, 10, 0))), 100
        )


def test_fraction_budget_and_gamma_validation():
    base = le.RECIPES_BY_ID["tsne_fraction_g8"]
    for budget in (0, -5, True, 1.5):
        with pytest.raises(ValueError):
            le.resolve_multilevel_schedule(replace(base, tsne_coarse_iteration_budget=budget), 100)
    for gamma in (-1.0, float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError):
            le.resolve_multilevel_schedule(replace(base, tsne_perplexity_gamma=gamma), 100)


# ── resolver: counts, budgets, perplexities derived from the contract ────────


def _expected_counts(recipe) -> list[int]:
    """``max(2, round(n·f))`` per fraction, ascending, dropped at/above ``n``."""
    counts: list[int] = []
    for f in recipe.tsne_sample_fractions:
        c = max(2, int(round(N_LIB * f)))
        if c >= N_LIB or (counts and c <= counts[-1]):
            break
        counts.append(c)
    return counts


def test_fraction_auto_schedule_follows_the_documented_rules():
    for rid in FRACTION_SET_IDS:
        recipe = le.RECIPES_BY_ID[rid]
        sched = le.resolve_multilevel_schedule(recipe, N_LIB)
        counts = _expected_counts(recipe)
        assert [s[0] for s in sched] == counts
        # Auto budget: round(50·√(n/count)) — the production allocation rule.
        assert [s[1] for s in sched] == [
            int(round(50.0 * math.sqrt(N_LIB / c))) for c in counts
        ]
        # Only the first (coarsest) level gets the early exaggeration phase.
        assert [s[2] for s in sched] == [100] + [0] * (len(counts) - 1)
        # Flat final perplexity 30, clamped to each level's affinity ceiling.
        assert [s[3] for s in sched] == [
            min(30.0, (c - 1) / 3.0) for c in counts
        ]


def test_density_budget_allocation_largest_remainder_deterministic():
    for rid in DENSITY_SET_IDS:
        recipe = le.RECIPES_BY_ID[rid]
        sched = le.resolve_multilevel_schedule(recipe, N_LIB)
        normals = [s[1] for s in sched]
        assert [s[0] for s in sched] == _expected_counts(recipe)
        # The fixed total is split exactly, with a 1-iteration floor per level.
        assert sum(normals) == recipe.tsne_coarse_iteration_budget
        assert all(n >= 1 for n in normals)
        # Largest remainder: each share is within one iteration of its ideal
        # inverse-sqrt weight, and the sum of floors is a lower bound.
        weights = [1.0 / math.sqrt(f) for f in recipe.tsne_sample_fractions]
        ideal = [recipe.tsne_coarse_iteration_budget * w / sum(weights) for w in weights]
        assert all(abs(n - s) < 1.0 for n, s in zip(normals, ideal))
        # Deterministic: repeated resolution is identical.
        again = le.resolve_multilevel_schedule(recipe, N_LIB)
        assert [s[1] for s in again] == normals
    # density2 and auto g8 resolve to the same 2-level budget: the auto rule and
    # the fixed 541 split agree on the same fractions.
    d2 = le.resolve_multilevel_schedule(le.RECIPES_BY_ID["tsne_density2"], N_LIB)
    g8 = le.resolve_multilevel_schedule(le.RECIPES_BY_ID["tsne_fraction_g8"], N_LIB)
    assert [s[1] for s in d2] == [s[1] for s in g8]


def test_budget_allocation_min_floor_and_exact_sum():
    # A budget equal to the level count forces exactly 1 per level (continuous16).
    sched = le.resolve_multilevel_schedule(le.RECIPES_BY_ID["tsne_continuous16"], N_LIB)
    assert [s[1] for s in sched] == [1] * len(sched)
    # A budget smaller than the level count is rejected, not silently mis-split.
    base = le.RECIPES_BY_ID["tsne_density2"]
    with pytest.raises(ValueError):
        le.resolve_multilevel_schedule(replace(base, tsne_coarse_iteration_budget=1), N_LIB)


# ── resolver: gamma perplexity schedule ──────────────────────────────────────


def test_gamma_schedule_zero_and_monotone_and_clamp():
    base = le.RECIPES_BY_ID["tsne_density2"]
    # gamma 0 pins every coarse level at perplexity 80.
    sched = le.resolve_multilevel_schedule(replace(base, tsne_perplexity_gamma=0.0), N_LIB)
    assert [s[3] for s in sched] == [37.0, 80.0]
    # Larger gamma drops the coarse perplexity towards the final 30; the first
    # level's (largest) requested perplexity is monotone decreasing in gamma.
    first_perp = []
    for g in (0.5, 1.0, 2.0):
        sched = le.resolve_multilevel_schedule(replace(base, tsne_perplexity_gamma=g), N_LIB)
        perps = [s[3] for s in sched]
        assert all(30.0 <= p <= 80.0 for p in perps)
        assert perps[0] == 37.0
        first_perp.append(perps[0])
    assert first_perp == [37.0, 37.0, 37.0]
    # A level near n (t→1) sits near the final 30, never below it.
    near = replace(base, tsne_sample_fractions=(0.5, 0.999))
    sched = le.resolve_multilevel_schedule(replace(near, tsne_perplexity_gamma=1.0), 1000)
    assert abs(sched[1][3] - 30.0) < 0.5
    # A tiny first level clamps the requested perplexity to (count−1)/3.
    tiny = replace(base, tsne_sample_fractions=(1 / 256, 1 / 8))
    sched = le.resolve_multilevel_schedule(replace(tiny, tsne_perplexity_gamma=0.0), 200)
    assert sched[0][3] == pytest.approx(1 / 3)  # count = max(2, round(200/256)) = 2 → ceiling 1/3


# ── tiny-N level filtering + no duplicate full fit ───────────────────────────


def test_resolver_dedupes_and_drops_levels_at_tiny_n():
    # Fractions resolve to small/deduped/at-or-above-n counts that must not
    # duplicate the final full-data fit.
    base = le.RECIPES_BY_ID["tsne_fraction_g8"]
    # n=4: round(4/64)=0→2, round(4/8)=0→2 (deduped) → [2]; no level == n.
    sched = le.resolve_multilevel_schedule(base, 4)
    assert [s[0] for s in sched] == [2]
    # n=3: round(3/64)=0→2, round(3/8)=0→2 (dedupe) → [2] < 3, survives.
    sched = le.resolve_multilevel_schedule(base, 3)
    assert [s[0] for s in sched] == [2]
    # n=2: the sole level resolves to 2 == n → dropped → empty (single full fit).
    assert le.resolve_multilevel_schedule(base, 2) == []
    # A level at/above n is dropped even when a smaller one survives.
    high = replace(base, tsne_sample_fractions=(0.4, 0.9))
    sched = le.resolve_multilevel_schedule(high, 10)  # round(10*0.9)=9 < 10 survives
    assert [s[0] for s in sched] == [4, 9]
    sched = le.resolve_multilevel_schedule(high, 5)  # round(5*0.9)=4 (dedupe with 2)
    assert [s[0] for s in sched] == [2, 4]


def test_fit_no_duplicate_full_fit_lengths(monkeypatch):
    # The coarse chain must never re-fit the full dataset (a level == n would
    # duplicate the final full-data fit). Sizes are strictly < n, and n appears
    # exactly once (the final fit).
    calls: list[int] = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append(len(reduced))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID["tsne_fraction_g8"]  # counts 112/894 at n=7153
    le._fit_tsne_multilevel_init(np.zeros((7153, 5)), recipe=recipe, seed=42, n_jobs=2)
    assert calls[-1] == 7153
    assert calls.count(7153) == 1
    assert all(c < 7153 for c in calls[:-1])


# ── fixed-level path unchanged (None fields, helper budgets) ─────────────────


def test_fixed_path_and_none_identity_unchanged():
    # The old fixed absolute-level helper still resolves its budgets.
    geo8 = le.RECIPES_BY_ID["tsne_geo8_combined"]
    assert le.resolve_multilevel_schedule(geo8, 7153) == [
        (256, 264, 100, 80.0), (2048, 93, 0, 30.0),
    ]
    # Production-perplexity rule levels record perp None (not a flat scalar).
    ref = le.RECIPES_BY_ID["tsne_chain50_250_1000"]
    assert le.resolve_multilevel_schedule(ref, 3000) == [
        (50, 100, 100, None), (250, 100, 0, None), (1000, 100, 0, None),
    ]
    # A multilevel recipe with neither levels nor fractions is rejected.
    nolevels = replace(le.RECIPES_BY_ID["tsne_chain50_250_1000"], tsne_sample_levels=None)
    with pytest.raises(ValueError):
        le.resolve_multilevel_schedule(nolevels, 3000)


# ── per-stage forwarding (fraction mode) ─────────────────────────────────────


def test_fraction_forwarded_per_stage(monkeypatch):
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), recipe.perplexities, recipe.tsne_n_iter,
                      recipe.tsne_early_exaggeration_iter, init is None))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID["tsne_fraction_g8"]
    le._fit_tsne_multilevel_init(np.zeros((7153, 5)), recipe=recipe, seed=42, n_jobs=2)
    # Level 0 coarse (no init), level 1 refine (init), then the final full fit.
    budget = [s[1] for s in le.resolve_multilevel_schedule(recipe, 7153)]
    counts = [s[0] for s in le.resolve_multilevel_schedule(recipe, 7153)]
    assert [(c[0], c[1]) for c in calls] == [
        (counts[0], (30.0,)), (counts[1], (30.0,)), (7153, (30.0,)),
    ]
    assert [c[2] for c in calls[:2]] == budget
    assert [c[3] for c in calls[:2]] == [100, 0]
    assert calls[0][4] is True and calls[1][4] is False and calls[2][4] is False


# ── real openTSNE smoke ──────────────────────────────────────────────────────


def test_fraction_real_smoke_deterministic_and_finite():
    recipe = le.Recipe(
        "smoke", "smoke", pca_dim=50, method="tsne_multilevel_init",
        tsne_n_iter=20, tsne_early_exaggeration_iter=0,
        tsne_sample_fractions=(0.25, 0.5),
        perplexities=(30.0,),
    )
    mat = np.random.RandomState(0).standard_normal((40, 50)).astype(np.float32)
    c1, timings = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    c2, _ = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    assert c1.shape == (40, 2)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1, c2)
    level_sum = timings["tsne_level0"] + timings["tsne_level1"] \
        + timings["tsne_expand"] + timings["tsne_final"]
    assert timings["tsne"] >= level_sum - 1e-6
