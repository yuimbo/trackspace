"""The ``hierarchy-gamma`` gallery: ``HIERARCHY_GAMMA_RECIPES``.

Covers the size-adaptive multilevel recipes that vary only the
``tsne_perplexity_gamma`` schedule over the ``tsne_density4`` geometry
(1/64→1/8 ×4, fixed TOTAL normal budget 541, final 50/0 at explicit perp 30):
registry/recipe invariants (the five gamma maps differ from the flat-30 control
``tsne_density4`` ONLY in id/label/gamma), the resolved schedule at the 7153
library size (counts/normals/earlys/final unchanged by gamma, coarse perps on
the ``30+50·(1−t)^gamma`` curve, budget summing to exactly 541), the
fraction-mode resolver invariants (no both fields, no orphan budget, no gamma on
fixed levels), and benchmark validation. The set's generation/route isolation is
covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("openTSNE")

from dataclasses import fields, replace

import backend.embeddings.layout_evaluation as le

#: The flat-30 control this gallery references (NOT a member of the set).
REFERENCE_ID = "tsne_density4"

GAMMA_SET_IDS = [
    "tsne_gamma0",
    "tsne_gamma05",
    "tsne_gamma1",
    "tsne_gamma2",
    "tsne_gamma4",
]

#: The gallery command / benchmark runs the reference first, then the five maps.
GAMMA_IDS = [REFERENCE_ID, *GAMMA_SET_IDS]

N_LIB = 7153  # the current library size used for the resolver expectations.


def _expected_coarse_perp(count: int, gamma: float, final: float = 30.0) -> float:
    """The documented curve ``p(t) = 30 + 50·(1−t)^gamma``, clamped by *count*.

    ``t`` is the log-scaled position of the level's count between the ``n/64``
    anchor and ``n``; the result is clamped to openTSNE's affinity ceiling
    ``(count−1)/3``.
    """
    start = 1.0 / 64.0
    t = math.log(count / (N_LIB * start)) / math.log(1.0 / start)
    t = min(1.0, max(0.0, t))
    requested = final + (80.0 - final) * (1.0 - t) ** gamma
    return min(requested, (count - 1) / 3.0)


# ── registry + recipe invariants ─────────────────────────────────────────────


def test_gamma_recipes_differ_from_density4_only_by_id_label_gamma():
    base = le.RECIPES_BY_ID[REFERENCE_ID]
    for r in le.HIERARCHY_GAMMA_RECIPES:
        for f in fields(base):
            if f.name in ("id", "label", "tsne_perplexity_gamma"):
                continue
            assert getattr(base, f.name) == getattr(r, f.name), (
                f"{r.id} must match {REFERENCE_ID} on {f.name}"
            )
        # The shared geometry/budget/final are exactly density4's.
        assert r.method == "tsne_multilevel_init"
        assert r.pca_dim == 50
        assert r.tsne_sample_fractions == le._FRACTIONS_DENSITY4
        assert r.tsne_coarse_iteration_budget == 541
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
        assert r.perplexities == (30.0,)
        assert r.tsne_sample_levels is None
        assert r.tsne_level_perplexities is None


# ── resolver: schedule invariants ────────────────────────────────────────────


def test_gamma_schedule_keeps_geometry_and_follows_the_curve():
    # Only the coarse perplexities move with gamma: counts, the fixed 541-normal
    # budget split, the first-level early phase and the final fit are density4's.
    base = le.resolve_multilevel_schedule(le.RECIPES_BY_ID[REFERENCE_ID], N_LIB)
    counts = [s[0] for s in base]
    normals = [s[1] for s in base]
    assert sum(normals) == 541  # fixed TOTAL budget, unchanged
    for rid in GAMMA_SET_IDS:
        r = le.RECIPES_BY_ID[rid]
        sched = le.resolve_multilevel_schedule(r, N_LIB)
        assert [s[0] for s in sched] == counts
        assert [s[1] for s in sched] == normals
        assert [s[2] for s in sched] == [100, 0, 0, 0]  # first level early 100
        # Per-level perplexity is the gamma curve evaluated at the level's share
        # of the library (clamped by the level's own count).
        expected = [
            _expected_coarse_perp(c, r.tsne_perplexity_gamma) for c in counts
        ]
        assert [s[3] for s in sched] == pytest.approx(expected, abs=1e-3)
        # The final fit stays 50/0 at explicit perp 30 for every gamma.
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
        assert r.perplexities == (30.0,)


def test_gamma_schedule_monotone_and_ranges():
    # The first (largest) coarse perp is ~80 for every gamma; the last level's
    # perp drops monotonically as gamma rises, from 80 (gamma 0) toward 33.
    first = [le.resolve_multilevel_schedule(le.RECIPES_BY_ID[rid], N_LIB)[0][3]
             for rid in GAMMA_SET_IDS]
    last = [le.resolve_multilevel_schedule(le.RECIPES_BY_ID[rid], N_LIB)[-1][3]
            for rid in GAMMA_SET_IDS]
    assert all(p == pytest.approx(37.0) for p in first)
    assert last == sorted(last, reverse=True)  # strictly decreasing in gamma
    assert last[0] == pytest.approx(80.0, abs=1e-6)  # gamma 0 flat 80
    assert last[-1] == pytest.approx(33.13, abs=0.05)  # gamma 4 near final 30
    # Every coarse perp stays within (final 30, coarse 80].
    for rid in GAMMA_SET_IDS:
        for s in le.resolve_multilevel_schedule(le.RECIPES_BY_ID[rid], N_LIB):
            assert 30.0 <= s[3] <= 80.0


def test_fraction_resolver_invariant_no_orphan_budget_or_gamma_on_fixed():
    # A fixed-level recipe silently ignoring a fraction-only field is a schedule
    # invariant violation: reject, don't record a schedule that never used it.
    base = le.RECIPES_BY_ID["tsne_geo8_combined"]
    assert base.tsne_sample_levels is not None
    assert base.tsne_coarse_iteration_budget is None
    assert base.tsne_perplexity_gamma is None
    with pytest.raises(ValueError, match="tsne_coarse_iteration_budget"):
        le.resolve_multilevel_schedule(replace(base, tsne_coarse_iteration_budget=100), 7153)
    with pytest.raises(ValueError, match="tsne_perplexity_gamma"):
        le.resolve_multilevel_schedule(replace(base, tsne_perplexity_gamma=1.0), 7153)
    # No both fields is still rejected in fraction mode.
    with pytest.raises(ValueError, match="mutually exclusive"):
        le.resolve_multilevel_schedule(
            replace(le.RECIPES_BY_ID["tsne_gamma1"], tsne_sample_levels=((10, 10, 10),)),
            100,
        )


# ── benchmark compatibility ──────────────────────────────────────────────────


def test_gamma_set_accepted_by_benchmark_validation():
    from backend import benchmark_layout_scaling as b
    b._validate_recipes(GAMMA_IDS)  # reference + 5 gamma maps
