"""Coarse-budget sweep + aggressive multilevel depth gallery.

Extends the two-level sample-init work with two new gallery sets:

* ``hierarchy-budget`` — ``HIERARCHY_BUDGET_RECIPES``: every map is sample1000
  with the final refinement held at 100/0 (the ``tsne_sample1000_early0`` anchor
  geometry); only the *coarse* budget moves (coarse normal 10/25/50/500 at
  early 100, and coarse early 0/25/50 at normal 100). Uses only the existing
  ``tsne_sample_init`` fields.
* ``hierarchy-depth`` — ``HIERARCHY_DEPTH_RECIPES``: two more two-level maps
  (sample 100 / 250) plus three **multilevel** chain recipes
  (``tsne_multilevel_init`` via the new ``tsne_sample_levels`` field).

Covers the new field's identity independence, the nested-sample-hierarchy
algorithm (membership, order, budget forwarding, clamp shortcuts, validation)
and a real openTSNE smoke. Both sets' registry entries, generation and route
isolation are covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le

# ── registry + budget-set recipe invariants ──────────────────────────────────


def test_budget_set_recipes_differ_by_one_coarse_axis_only():
    # Every new budget map: sample1000, final 100/0, no new field, and only the
    # coarse (normal or early) axis varies.
    for rid in le.HIERARCHY_BUDGET_RECIPES:
        r = le.RECIPES_BY_ID[rid.id]
        assert r.method == "tsne_sample_init"
        assert r.pca_dim == 50
        assert r.perplexities is None and r.source_weights is None
        assert r.tsne_sample_levels is None  # budget uses only existing fields
        assert r.tsne_sample_n == 1000
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (100, 0)

    coarse_normal = {
        rid.id: (rid.tsne_sample_n_iter, rid.tsne_sample_early_exaggeration_iter)
        for rid in le.HIERARCHY_BUDGET_RECIPES
    }
    assert coarse_normal == {
        "tsne_sample1000_coarse10e100": (10, 100),
        "tsne_sample1000_coarse25e100": (25, 100),
        "tsne_sample1000_coarse50e100": (50, 100),
        "tsne_sample1000_coarse500e100": (500, 100),
        "tsne_sample1000_coarse100e0": (100, 0),
        "tsne_sample1000_coarse100e25": (100, 25),
        "tsne_sample1000_coarse100e50": (100, 50),
    }

def test_depth_set_recipe_parameters():
    r100 = le.RECIPES_BY_ID["tsne_sample100_early0"]
    assert (r100.method, r100.tsne_sample_n, r100.tsne_sample_n_iter,
            r100.tsne_sample_early_exaggeration_iter, r100.tsne_n_iter,
            r100.tsne_early_exaggeration_iter) == (
        "tsne_sample_init", 100, 100, 100, 100, 0)
    r250 = le.RECIPES_BY_ID["tsne_sample250_early0"]
    assert (r250.tsne_sample_n, r250.tsne_n_iter, r250.tsne_early_exaggeration_iter) \
        == (250, 100, 0)

    chain = le.RECIPES_BY_ID["tsne_chain100_500_2000_final100"]
    assert chain.method == "tsne_multilevel_init"
    assert chain.tsne_sample_levels == (
        (100, 100, 100), (500, 100, 0), (2000, 100, 0))
    assert (chain.tsne_n_iter, chain.tsne_early_exaggeration_iter) == (100, 0)

    chain50 = le.RECIPES_BY_ID["tsne_chain100_500_2000_final50"]
    assert chain50.tsne_sample_levels == chain.tsne_sample_levels
    assert (chain50.tsne_n_iter, chain50.tsne_early_exaggeration_iter) == (50, 0)

    extreme = le.RECIPES_BY_ID["tsne_chain50_250_1000"]
    assert extreme.tsne_sample_levels == (
        (50, 100, 100), (250, 100, 0), (1000, 100, 0))
    assert (extreme.tsne_n_iter, extreme.tsne_early_exaggeration_iter) == (100, 0)


# ── identity: new field independent, omitted when None ───────────────────────


def test_sample_levels_drives_revision_and_params():
    from dataclasses import replace

    base = le.Recipe(
        "x", "x", pca_dim=50, method="tsne_multilevel_init",
        tsne_n_iter=100, tsne_early_exaggeration_iter=0,
        tsne_sample_levels=((100, 100, 100), (500, 100, 0), (2000, 100, 0)),
    )
    base_rev = le.recipe_revision("deadbeef", base)
    assert "tsne_sample_levels" in le.recipe_params(base)
    # A different level tuple (size, or a per-level budget) changes the identity.
    for variant in (
        replace(base, tsne_sample_levels=((100, 100, 100), (500, 100, 0), (1500, 100, 0))),
        replace(base, tsne_sample_levels=((100, 100, 100), (500, 50, 0), (2000, 100, 0))),
        replace(base, tsne_sample_levels=((100, 100, 100), (500, 100, 0))),
        replace(base, tsne_n_iter=50),
    ):
        assert le.recipe_revision("deadbeef", variant) != base_rev
    # Dropping the field (back to None) also changes the identity.
    plain = replace(base, tsne_sample_levels=None)
    assert le.recipe_revision("deadbeef", plain) != base_rev
    assert "tsne_sample_levels" not in le.recipe_params(plain)


# ── multilevel algorithm: nested membership, budgets, clamping ───────────────


def test_multilevel_nested_membership_and_order():
    # The levels come from sorted prefixes of ONE seeded permutation, so each
    # level's rows are a nested superset of the previous, all sorted ascending.
    n, seed = 3000, 42
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    idx0 = np.sort(perm[:100])
    idx1 = np.sort(perm[:500])
    idx2 = np.sort(perm[:2000])
    assert set(idx0) <= set(idx1) <= set(idx2)
    for arr in (idx0, idx1, idx2):
        assert arr.min() >= 0 and arr.max() < n
        assert np.all(arr[1:] > arr[:-1])  # sorted, no duplicates


def test_multilevel_budgets_forwarded_and_final_full_fit(monkeypatch):
    # Coarse fit at level0 with its budget; each following level expands+refines
    # at that level's budget; the final full fit uses the recipe's final budget.
    n = 3000
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    recipe = le.RECIPES_BY_ID["tsne_chain100_500_2000_final100"]
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init is None,
                      recipe.tsne_n_iter, recipe.tsne_early_exaggeration_iter))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    coords, stage_t = le._fit_tsne_multilevel_init(reduced, recipe=recipe, seed=42, n_jobs=2)
    # coarse@100 (100/100, no init), level1@500 (100/0, init), level2@2000
    # (100/0, init), final@3000 (recipe final 100/0, init).
    assert calls == [
        (100, True, 100, 100),
        (500, False, 100, 0),
        (2000, False, 100, 0),
        (3000, False, 100, 0),
    ]
    assert set(("tsne_level0", "tsne_level1", "tsne_level2", "tsne_expand",
                "tsne_final")) <= set(stage_t)
    assert stage_t["tsne_level0"] > 0.0
    assert np.isfinite(coords).all()


def test_multilevel_tiny_n_skips_redundant_fits(monkeypatch):
    # When no level is smaller than the dataset, the coarse chain is pointless:
    # exactly one full-data fit with no init (no duplicate intermediate fits).
    calls = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init))
        return np.zeros((len(reduced), 2), dtype=np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    reduced = np.random.RandomState(0).standard_normal((10, 5)).astype(np.float64)
    recipe = le.RECIPES_BY_ID["tsne_chain100_500_2000_final100"]  # min level 100 > 10
    coords, stage_t = le._fit_tsne_multilevel_init(reduced, recipe=recipe, seed=42, n_jobs=2)
    assert len(calls) == 1
    assert calls[0][0] == 10 and calls[0][1] is None
    assert stage_t["tsne_level0"] == 0.0
    assert stage_t["tsne_expand"] == 0.0
    assert stage_t["tsne_final"] > 0.0
    assert np.isfinite(coords).all()


def test_multilevel_clamped_intermediate_level_skipped(monkeypatch):
    # A level that clamps to n would be a duplicate full fit — it is dropped and
    # not refitted; the remaining smaller levels still run and the final full fit
    # covers the full data.
    calls = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    n = 1500  # level 2000 clamps to 1500 == n → dropped; 100 and 500 remain.
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    recipe = le.RECIPES_BY_ID["tsne_chain100_500_2000_final100"]
    coords, stage_t = le._fit_tsne_multilevel_init(reduced, recipe=recipe, seed=42, n_jobs=2)
    # coarse@100 (no init) + refine@500 (init) + final@1500 (init): no @1500 dup.
    assert [c[0] for c in calls] == [100, 500, 1500]
    assert [c[1] is None for c in calls] == [True, False, False]
    assert set(("tsne_level0", "tsne_level1", "tsne_expand", "tsne_final")) <= set(stage_t)
    assert "tsne_level2" not in stage_t
    assert np.isfinite(coords).all()


def test_multilevel_missing_levels_rejected_not_silent_plain_fit(monkeypatch):
    # A recipe whose method is tsne_multilevel_init but defines no levels must
    # raise — it must not silently fall through to a plain full-data fit (which
    # would claim a hierarchy it never used). A dataset whose levels are all >= n
    # is a different case: the coarse chain is pointless, so a direct full fit is
    # still valid, not an error.
    from dataclasses import replace

    base = le.RECIPES_BY_ID["tsne_chain50_250_1000"]
    no_levels = replace(base, tsne_sample_levels=None)
    with pytest.raises(ValueError):
        le._fit_tsne_multilevel_init(
            np.zeros((3000, 5)), recipe=no_levels, seed=42, n_jobs=2
        )

    # tsne_chain100_500_2000_final100 levels (100,500,2000) on n=150: only 100
    # survives, so coarse@100 then a direct full fit at 150.
    calls = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init))
        return np.zeros((len(reduced), 2), dtype=np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    recipe = le.RECIPES_BY_ID["tsne_chain100_500_2000_final100"]
    coords, stage_t = le._fit_tsne_multilevel_init(
        np.zeros((150, 5)), recipe=recipe, seed=42, n_jobs=2
    )
    assert [c[0] for c in calls] == [100, 150]
    assert stage_t["tsne_final"] > 0.0
    assert np.isfinite(coords).all()


def test_multilevel_invalid_levels_rejected():
    from dataclasses import replace

    base = le.RECIPES_BY_ID["tsne_chain50_250_1000"]
    bad = [
        (),  # empty
        ((0, 100, 100), (250, 100, 0)),  # non-positive size
        ((100, 100, 100), (100, 100, 0)),  # not strictly increasing
        ((250, 100, 100), (100, 100, 0)),  # decreasing
        ((100, 0, 100), (250, 100, 0)),  # non-positive normal
        ((100, 100, -1), (250, 100, 0)),  # negative early
        ((100, 100), (250, 100, 0)),  # malformed entry (len 2)
        ((100.0, 100, 100), (250, 100, 0)),  # non-int size
    ]
    for levels in bad:
        r = replace(base, tsne_sample_levels=levels)
        with pytest.raises(ValueError):
            le._fit_tsne_multilevel_init(
                np.zeros((3000, 5)), recipe=r, seed=42, n_jobs=2
            )
        with pytest.raises(ValueError):
            le._normalise_sample_levels(levels, 3000)


def test_multilevel_expand_preserves_landmark_coords():
    # The shared expand helper: landmark rows keep exact coords, others get
    # nearest-landmark + tiny jitter. Same geometry used for the two-level path.
    n = 40
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    idx = le._tsne_sample_indices(n, 8, seed=42)
    coarse = np.random.default_rng(1).standard_normal((8, 2)).astype(np.float32)
    expanded = le._expand_landmark_init(
        reduced, idx, coarse, np.arange(n), seed=42, n_jobs=2)
    assert expanded.shape == (n, 2)
    assert np.isfinite(expanded).all()
    np.testing.assert_array_equal(expanded[idx], coarse)
    scale = float(coarse.std(axis=0).max()) or 1.0
    for i in range(n):
        if i in idx:
            continue
        d = np.min(np.linalg.norm(expanded[i] - coarse, axis=1))
        assert d < scale * 1e-2  # near a landmark, but jittered
        assert tuple(expanded[i].tolist()) not in set(map(tuple, coarse.tolist()))


# ── real openTSNE smoke ──────────────────────────────────────────────────────


def test_multilevel_real_smoke_deterministic_and_finite():
    recipe = le.Recipe(
        "smoke", "smoke", pca_dim=50, method="tsne_multilevel_init",
        tsne_n_iter=20, tsne_early_exaggeration_iter=0,
        tsne_sample_levels=((8, 20, 10), (16, 10, 0)),
    )
    mat = np.random.RandomState(0).standard_normal((40, 50)).astype(np.float32)
    c1, timings = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    c2, _ = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    assert c1.shape == (40, 2)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1, c2)
    # tsne is the whole chain; the stage keys are informational and sum within it
    # (perf-counter overhead between windows means tsne >= their sum - epsilon).
    level_sum = timings["tsne_level0"] + timings["tsne_level1"] \
        + timings["tsne_expand"] + timings["tsne_final"]
    assert timings["tsne"] >= level_sum - 1e-6
    assert timings["tsne_level0"] > 0.0
    assert timings["tsne_final"] > 0.0
