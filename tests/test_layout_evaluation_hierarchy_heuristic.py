"""Geometric multilevel 'hierarchy-heuristic' gallery: per-level perplexities.

Adds the new ``Recipe.tsne_level_perplexities`` field (one explicit scalar per
level of the ``tsne_multilevel_init`` chain, matched by position against the
*original* levels before the tiny-N size filter) and the ``HIERARCHY_HEURISTIC_
RECIPES`` set — the fast ``tsne_chain50_250_1000_final50`` reference plus six
geometric chains (exponential level sizes, sqrt normal-iteration schedule,
80/40/30 or 30 per-level perplexities, final fit 50/0 at perplexity 30).

Covers the field's validation, identity independence, per-stage forwarding
(and the unchanged ``None`` path), tiny-N level-filter mapping and a real
openTSNE smoke. The set's registry entry, generation and route isolation are
covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le

REFERENCE_ID = "tsne_chain50_250_1000_final50"

# ── registry + recipe invariants ─────────────────────────────────────────────


def test_heuristic_recipe_parameters():
    # Every geometric chain: PCA50 multilevel, final 50/0 at an explicit
    # perplexity 30, and a per-level perplexity for every level.
    per_level = {
        "tsne_geo4_fixed": ((256, 100, 100), (1024, 100, 0), (4096, 100, 0)),
        "tsne_geo4_sqrt": ((256, 264, 100), (1024, 132, 0), (4096, 66, 0)),
        "tsne_geo4_perplexity": ((256, 100, 100), (1024, 100, 0), (4096, 100, 0)),
        "tsne_geo4_combined": ((256, 264, 100), (1024, 132, 0), (4096, 66, 0)),
        "tsne_geo8_combined": ((256, 264, 100), (2048, 93, 0)),
        "tsne_geo4_combined_double": ((256, 528, 100), (1024, 264, 0), (4096, 132, 0)),
    }
    level_perps = {
        "tsne_geo4_fixed": (30.0, 30.0, 30.0),
        "tsne_geo4_sqrt": (30.0, 30.0, 30.0),
        "tsne_geo4_perplexity": (80.0, 40.0, 30.0),
        "tsne_geo4_combined": (80.0, 40.0, 30.0),
        "tsne_geo8_combined": (80.0, 30.0),
        "tsne_geo4_combined_double": (80.0, 40.0, 30.0),
    }
    for rid, levels in per_level.items():
        r = le.RECIPES_BY_ID[rid]
        assert r.method == "tsne_multilevel_init"
        assert r.pca_dim == 50
        assert r.source_weights is None
        assert r.tsne_sample_levels == levels
        assert r.tsne_level_perplexities == level_perps[rid]
        # Every per-level perplexity is <= its level size (no clamp fires).
        assert all(p < size for (size, _, _), p in zip(levels, level_perps[rid]))
        # Final fit: 50 normal / 0 early at an explicit perplexity 30.
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
        assert r.perplexities == (30.0,)
    # The reference keeps the production perplexity rule (``None``) — pinned by
    # test_shared_chain_references_unchanged in the hierarchy hub.
    assert le.RECIPES_BY_ID[REFERENCE_ID].perplexities is None


# ── identity: new field independent, omitted when None ───────────────────────


def test_level_perplexities_drives_revision_and_params():
    from dataclasses import replace

    base = le.RECIPES_BY_ID["tsne_geo4_combined"]
    base_rev = le.recipe_revision("deadbeef", base)
    assert "tsne_level_perplexities" in le.recipe_params(base)
    # Changing only the per-level perplexities changes the identity.
    for variant in (
        replace(base, tsne_level_perplexities=(90.0, 40.0, 30.0)),
        replace(base, tsne_level_perplexities=(80.0, 40.0, 20.0)),
        replace(base, tsne_level_perplexities=None),
    ):
        assert le.recipe_revision("deadbeef", variant) != base_rev
    # Dropping the field (back to None) omits it from params and changes identity.
    plain = replace(base, tsne_level_perplexities=None)
    assert "tsne_level_perplexities" not in le.recipe_params(plain)


# ── validation ───────────────────────────────────────────────────────────────


def test_level_perplexities_invalid_values_rejected():
    from dataclasses import replace

    base = le.RECIPES_BY_ID["tsne_geo4_combined"]
    bad = [
        (80.0, 40.0),                     # wrong length (too short)
        (80.0, 40.0, 30.0, 30.0),         # wrong length (too long)
        (80.0, -1.0, 30.0),               # negative
        (80.0, 0.0, 30.0),                # zero
        (80.0, float("nan"), 30.0),       # NaN
        (80.0, float("inf"), 30.0),       # +inf
        (80.0, float("-inf"), 30.0),      # -inf
        (80.0, True, 30.0),               # bool is not a number
    ]
    for perps in bad:
        r = replace(base, tsne_level_perplexities=perps)
        with pytest.raises(ValueError):
            le._fit_tsne_multilevel_init(np.zeros((5000, 5)), recipe=r, seed=42, n_jobs=2)


# ── per-stage forwarding + tiny-N mapping + None path ────────────────────────


def test_level_perplexities_forwarded_per_stage_full_kwargs(monkeypatch):
    # Each level fit receives its own explicit scalar perplexity (as a one-element
    # tuple → scalar path), with the level's budget; the final full fit uses the
    # recipe's own final perplexity (30). The recipes differ so kwargs stay
    # distinct per stage.
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((
            len(reduced),
            recipe.perplexities,
            recipe.tsne_n_iter,
            recipe.tsne_early_exaggeration_iter,
            init is None,
        ))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID["tsne_geo4_combined"]  # 256→1024→4096, perps 80/40/30
    reduced = np.random.RandomState(0).standard_normal((5000, 5)).astype(np.float64)
    le._fit_tsne_multilevel_init(reduced, recipe=recipe, seed=42, n_jobs=2)

    assert [(c[0], c[1]) for c in calls] == [
        (256, (80.0,)), (1024, (40.0,)), (4096, (30.0,)), (5000, (30.0,)),
    ]
    # Level budgets are distinct (264/132/66) and the final fit is 50/0.
    assert [c[2] for c in calls[:3]] == [264, 132, 66]
    assert calls[3][2] == 50 and calls[3][3] == 0
    # Coarse level runs plain (no init); later levels and the final fit use init.
    assert calls[0][4] is True
    assert all(c[4] is False for c in calls[1:])


def test_level_perplexities_tiny_n_filter_maps_perps_to_surviving_levels(monkeypatch):
    # Perps are matched against the ORIGINAL levels before the size filter, so a
    # tiny N that drops later levels never shifts a perplexity onto the wrong
    # surviving level.
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), recipe.perplexities))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID["tsne_geo4_combined"]
    # n=300: only the 256 level survives (1024/4096 fall away); the final full fit
    # covers the rest at the recipe's final perplexity 30.
    le._fit_tsne_multilevel_init(np.zeros((300, 5)), recipe=recipe, seed=42, n_jobs=2)
    assert [(c[0], c[1]) for c in calls] == [(256, (80.0,)), (300, (30.0,))]
    # n=200: no level survives → a single full fit at the final perplexity.
    calls.clear()
    le._fit_tsne_multilevel_init(np.zeros((200, 5)), recipe=recipe, seed=42, n_jobs=2)
    assert [(c[0], c[1]) for c in calls] == [(200, (30.0,))]


def test_level_perplexities_old_none_path_unchanged(monkeypatch):
    # A recipe without tsne_level_perplexities (the pre-existing reference) keeps
    # production scalar perplexity (None) at EVERY level — byte-unchanged.
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), recipe.perplexities))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID[REFERENCE_ID]
    le._fit_tsne_multilevel_init(np.zeros((3000, 5)), recipe=recipe, seed=42, n_jobs=2)
    assert calls == [
        (50, None), (250, None), (1000, None), (3000, None),
    ]


# ── real openTSNE smoke ──────────────────────────────────────────────────────


def test_heuristic_real_smoke_deterministic_and_finite():
    recipe = le.Recipe(
        "smoke", "smoke", pca_dim=50, method="tsne_multilevel_init",
        tsne_n_iter=20, tsne_early_exaggeration_iter=0,
        tsne_sample_levels=((8, 20, 10), (16, 10, 0)),
        tsne_level_perplexities=(30.0, 30.0),
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
