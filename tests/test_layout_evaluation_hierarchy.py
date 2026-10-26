"""Hierarchy gallery hub: sample-init algorithm, registry, identity and routes.

Covers the ``tsne_sample_init`` method — coarse fit on a deterministic uniform
sample of rows, NN expansion onto landmark positions, then a full-data refinement
from that initialisation — plus the timing split (``tsne`` vs the
coarse/expand/refine stage keys, with no double-count in ``total``), the
independent identity fields, and the shared gallery harness for every built-in
hierarchy set: one pinned registry table, one global identity-uniqueness check,
and one parametrized generation + route-isolation test per set.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le


#: The alignment reference of the ``hierarchy`` set: the plain 100/100 control
#: ``tsne_early100`` (an existing recipe); the rest align onto it.
REFERENCE_ID = "tsne_early100"


# ── recipe fields and control identities ─────────────────────────────────────


def test_hierarchy_recipe_fields_and_control():
    # The alignment reference is the existing plain 100-normal/100-early control,
    # identity-identical to the early set's ``tsne_early100`` (100/250 would be
    # tsne_budget100 — the wrong control for this set).
    ref = le.RECIPES_BY_ID[REFERENCE_ID]
    assert ref.method == "tsne"
    assert ref.tsne_n_iter == 100
    assert ref.tsne_early_exaggeration_iter == 100
    assert ref.tsne_sample_n is None

    # The new plain control shortens both phases to 50/50, still plain.
    plain = le.RECIPES_BY_ID["tsne50_plain"]
    assert plain.method == "tsne"
    assert plain.tsne_n_iter == 50
    assert plain.tsne_early_exaggeration_iter == 50
    assert plain.tsne_sample_n is None

    # Every sampled map: PCA50, production scalar perplexity rule, no explicit
    # source mix, and the full sample + coarse/final budget set.
    sample_ids = [
        r.id for r in le.HIERARCHY_RECIPES if r.method == "tsne_sample_init"
    ]
    assert len(sample_ids) == 6
    for rid in sample_ids:
        r = le.RECIPES_BY_ID[rid]
        assert r.method == "tsne_sample_init"
        assert r.pca_dim == 50
        assert r.perplexities is None
        assert r.source_weights is None
        assert r.tsne_sample_n is not None
        assert r.tsne_sample_n_iter is not None
        assert r.tsne_sample_early_exaggeration_iter is not None


def test_hierarchy_sample_sizes_and_budgets():
    # Sample size axis: 500/1000/2000, coarse 100/100, final 100/100.
    for rid, n in (("tsne_sample500", 500), ("tsne_sample1000", 1000),
                   ("tsne_sample2000", 2000)):
        r = le.RECIPES_BY_ID[rid]
        assert r.tsne_sample_n == n
        assert r.tsne_sample_n_iter == 100
        assert r.tsne_sample_early_exaggeration_iter == 100
        assert r.tsne_n_iter == 100
        assert r.tsne_early_exaggeration_iter == 100

    # sample1000 variants: only the intended budget dimension moves.
    r = le.RECIPES_BY_ID["tsne_sample1000"]
    assert (r.tsne_sample_n, r.tsne_sample_n_iter,
            r.tsne_sample_early_exaggeration_iter,
            r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (
        1000, 100, 100, 100, 100)
    cases = {
        "tsne_sample1000_early0": (1000, 100, 100, 100, 0),     # final early off
        "tsne_sample1000_final50": (1000, 100, 100, 50, 50),    # final 50/50
        "tsne_sample1000_coarse50": (1000, 50, 50, 100, 100),   # coarse 50/50
    }
    for rid, (n, c_iter, c_early, f_iter, f_early) in cases.items():
        r = le.RECIPES_BY_ID[rid]
        assert r.tsne_sample_n == n
        assert r.tsne_sample_n_iter == c_iter
        assert r.tsne_sample_early_exaggeration_iter == c_early
        assert r.tsne_n_iter == f_iter
        assert r.tsne_early_exaggeration_iter == f_early


# ── registry: every built-in set's exact membership and order ────────────────
#
# One pinned table for the whole registry: each set's recipe ids in
# generation/alignment order (first = the reference). This replaces the
# per-set registry assertions that used to be copied into every
# ``test_layout_evaluation_hierarchy_*`` module.

GALLERY_SET_IDS = {
    "base": [
        "baseline", "pca2", "pca25", "tsne_budget",
        "umap2_n15", "umap2_n60", "umap_tsne", "tsne_umap_init",
    ],
    "budget": ["baseline", "tsne_budget100", "tsne_budget150", "tsne_budget"],
    "scale": ["tsne_budget100", "tsne_p60", "tsne_p120", "tsne_multiscale30_120"],
    "early": ["tsne_budget100", "tsne_early150", "tsne_early100"],
    "mixes": ["tsne_budget100", "tsne_rhythm_forward", "tsne_texture_style"],
    "ablation": ["tsne_budget100", "tsne_maest_supervised", "tsne_maest_only"],
    "hierarchy": [
        "tsne_early100", "tsne50_plain", "tsne_sample500", "tsne_sample1000",
        "tsne_sample2000", "tsne_sample1000_early0", "tsne_sample1000_final50",
        "tsne_sample1000_coarse50",
    ],
    "hierarchy-budget": [
        "tsne_early100", "tsne50_plain", "tsne_sample1000_early0",
        "tsne_sample1000_coarse10e100", "tsne_sample1000_coarse25e100",
        "tsne_sample1000_coarse50e100", "tsne_sample1000_coarse500e100",
        "tsne_sample1000_coarse100e0", "tsne_sample1000_coarse100e25",
        "tsne_sample1000_coarse100e50",
    ],
    "hierarchy-depth": [
        "tsne_early100", "tsne50_plain", "tsne_sample1000_early0",
        "tsne_sample100_early0", "tsne_sample250_early0",
        "tsne_chain100_500_2000_final100", "tsne_chain100_500_2000_final50",
        "tsne_chain50_250_1000",
    ],
    "hierarchy-tuning": [
        "tsne_chain50_250_1000", "tsne_chain50_250_1000_final50",
        "tsne_chain50_250_1000_coarse10", "tsne_chain50_250_1000_coarse10_final50",
        "tsne_chain50_250_1000_final2000",
    ],
    "hierarchy-distribution": [
        "tsne_chain50_250_1000_final50",
        "tsne_chain50_250_1000_early1000_final50",
        "tsne_chain50_250_1000_middle1000_final50",
        "tsne_chain50_250_1000_late1000_final50",
        "tsne_chain50_250_1000_spread400_final50",
    ],
    "hierarchy-heuristic": [
        "tsne_chain50_250_1000_final50", "tsne_geo4_fixed", "tsne_geo4_sqrt",
        "tsne_geo4_perplexity", "tsne_geo4_combined", "tsne_geo8_combined",
        "tsne_geo4_combined_double",
    ],
    "hierarchy-fractions": [
        "tsne_geo8_combined", "tsne_fraction_g4", "tsne_fraction_g8",
        "tsne_fraction_g16",
    ],
    "hierarchy-density": [
        "tsne_geo8_combined", "tsne_density2", "tsne_density4", "tsne_density8",
        "tsne_density16", "tsne_continuous16",
    ],
    "hierarchy-gamma": [
        "tsne_density4", "tsne_gamma0", "tsne_gamma05", "tsne_gamma1",
        "tsne_gamma2", "tsne_gamma4",
    ],
    "hierarchy-combination": [
        "tsne_continuous16", "tsne_continuous16_wide",
        "tsne_continuous16_wide_gamma4", "tsne_continuous16_3d",
        "tsne_continuous16_wide_3d", "tsne_continuous16_wide_gamma4_3d",
    ],
}


@pytest.mark.parametrize("set_name", list(GALLERY_SET_IDS))
def test_gallery_set_registry_matches_expected_ids(set_name):
    expected = GALLERY_SET_IDS[set_name]
    gallery_set = le.GALLERY_SETS[set_name]
    assert gallery_set.recipes is not None
    assert [r.id for r in gallery_set.recipes] == expected
    # Every set is one registry entry, sharing the recipe objects and ids.
    for rid in expected:
        assert le.RECIPES_BY_ID[rid] is gallery_set.recipes[expected.index(rid)]
    assert set(expected) <= set(le.RECIPES_BY_ID)
    # The default generation stays exactly the original eight; every hierarchy
    # set is additive and therefore disjoint from it (the older budget/scale/
    # early sets deliberately reuse controls from the eight).
    assert len(le.RECIPES) == 8
    if set_name.startswith("hierarchy"):
        assert not (set(expected) & {r.id for r in le.RECIPES})


def test_every_registered_recipe_has_a_distinct_revision():
    # Two recipes that resolve to the same identity would silently overwrite
    # each other's artifact, so identity uniqueness is global, not per set.
    revs = {
        rid: le.recipe_revision("deadbeef", recipe)
        for rid, recipe in le.RECIPES_BY_ID.items()
    }
    assert len(set(revs.values())) == len(revs)


BASELINE_DIGEST = "23a524bc54549fe9ab36d35a2f86172aabac597980e3b922829066308c302886"


@pytest.mark.parametrize(
    "rid, fields, digest",
    [
        # Every field added by the hierarchy work defaults to None and is
        # omitted from the identity, so the original eight keep their revisions.
        (
            "baseline",
            (
                "tsne_sample_n",
                "tsne_sample_n_iter",
                "tsne_sample_early_exaggeration_iter",
                "tsne_sample_levels",
                "tsne_level_perplexities",
                "tsne_sample_fractions",
                "tsne_coarse_iteration_budget",
                "tsne_perplexity_gamma",
            ),
            BASELINE_DIGEST,
        ),
        (
            "tsne_geo8_combined",
            ("tsne_sample_fractions",),
            "40ee796c921e1e46257f27babf3440438c0592deeb778d5525e8f89e22ec7803",
        ),
        (
            "tsne_continuous16",
            (
                "dims",
                "group",
                "tsne_perplexity_start_fraction",
                "tsne_perplexity_gamma",
            ),
            "52e6747b7a472a18fd89c7919ad465a29266e59a1b6928f4cd600176ab000490",
        ),
    ],
)
def test_optional_recipe_fields_are_omitted_from_identity(rid, fields, digest):
    recipe = le.RECIPES_BY_ID[rid]
    params = le.recipe_params(recipe)
    for field in fields:
        assert field not in params
    assert le.recipe_revision("deadbeef", recipe) == digest


#: The two pre-existing chain references the tuning/distribution/heuristic sets
#: build on, pinned as independent ``Recipe`` values (never compared to the
#: module's own constant) so a drift in either identity is caught.
CHAIN_REFERENCES = [
    (
        "tsne_chain50_250_1000",
        le.Recipe(
            id="tsne_chain50_250_1000",
            label="PCA50 → multilevel 50→250→1000 (coarse 100/100, levels 100/0, final 100/0)",
            pca_dim=50,
            method="tsne_multilevel_init",
            tsne_n_iter=100,
            tsne_early_exaggeration_iter=0,
            tsne_sample_levels=((50, 100, 100), (250, 100, 0), (1000, 100, 0)),
        ),
    ),
    (
        "tsne_chain50_250_1000_final50",
        le.Recipe(
            id="tsne_chain50_250_1000_final50",
            label="PCA50 → multilevel 50→250→1000 (coarse 100/100, levels 100/0, final 50/0)",
            pca_dim=50,
            method="tsne_multilevel_init",
            tsne_n_iter=50,
            tsne_early_exaggeration_iter=0,
            tsne_sample_levels=((50, 100, 100), (250, 100, 0), (1000, 100, 0)),
        ),
    ),
]


@pytest.mark.parametrize("rid, expected", CHAIN_REFERENCES)
def test_shared_chain_references_unchanged(rid, expected):
    ref = le.RECIPES_BY_ID[rid]
    assert le.recipe_params(ref) == le.recipe_params(expected)
    assert le.recipe_revision("deadbeef", ref) == le.recipe_revision(
        "deadbeef", expected
    )


# ── identity: new fields are independent ─────────────────────────────────────


def test_sample_fields_independent_revisions():
    from dataclasses import replace

    # Prove each sample field independently drives the identity: keep id, label,
    # method and the other budgets fixed and change exactly one field at a time.
    base = le.Recipe(
        "x", "x", pca_dim=50, method="tsne_sample_init",
        tsne_n_iter=100, tsne_early_exaggeration_iter=100,
        tsne_sample_n=500, tsne_sample_n_iter=100,
        tsne_sample_early_exaggeration_iter=100,
    )
    base_rev = le.recipe_revision("deadbeef", base)
    variants = {
        "tsne_sample_n": replace(base, tsne_sample_n=999),
        "tsne_sample_n_iter": replace(base, tsne_sample_n_iter=999),
        "tsne_sample_early_exaggeration_iter": replace(
            base, tsne_sample_early_exaggeration_iter=999),
        "tsne_n_iter": replace(base, tsne_n_iter=999),
        "tsne_early_exaggeration_iter": replace(base, tsne_early_exaggeration_iter=999),
    }
    for field, v in variants.items():
        assert le.recipe_revision("deadbeef", v) != base_rev, field
    # Dropping the sample path (fields back to None) also changes the identity.
    plain = replace(base, tsne_sample_n=None, tsne_sample_n_iter=None,
                    tsne_sample_early_exaggeration_iter=None)
    assert le.recipe_revision("deadbeef", plain) != base_rev
    assert "tsne_sample_n" in le.recipe_params(base)
    assert "tsne_sample_n" not in le.recipe_params(plain)


# ── sampling helper ──────────────────────────────────────────────────────────


def test_tsne_sample_indices_deterministic_and_clamped():
    idx1 = le._tsne_sample_indices(30, 8, seed=42)
    idx2 = le._tsne_sample_indices(30, 8, seed=42)
    assert idx1.shape == (8,)
    assert np.array_equal(idx1, idx2)
    assert idx1.min() >= 0 and idx1.max() < 30
    # Sample clamp: requested > n yields all rows (ascending), never duplicates.
    assert np.array_equal(le._tsne_sample_indices(5, 2000, seed=42), np.arange(5))
    # A different seed picks a different (still valid) sample.
    assert not np.array_equal(idx1, le._tsne_sample_indices(30, 8, seed=7))


# ── algorithm: mocked calls, tiny-N, landmarks ───────────────────────────────


def test_tiny_n_sample_covered_by_direct_full_fit(monkeypatch):
    # When sample_n >= n the coarse/expand stages are pointless: exactly one
    # full-data fit with no init (no duplicate coarse overhead).
    calls = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init))
        return np.zeros((len(reduced), 2), dtype=np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    reduced = np.random.RandomState(0).standard_normal((10, 5)).astype(np.float64)
    recipe = le.RECIPES_BY_ID["tsne_sample1000"]  # 1000 > 10
    coords, stage_t = le._fit_tsne_sample_init(reduced, recipe=recipe, seed=42, n_jobs=2)
    assert len(calls) == 1
    assert calls[0][0] == 10  # full data
    assert calls[0][1] is None  # no init → plain direct fit
    assert stage_t["tsne_coarse"] == 0.0
    assert stage_t["tsne_expand"] == 0.0
    assert stage_t["tsne_refine"] > 0.0
    assert np.isfinite(coords).all()


def test_sample_init_coarse_budget_and_full_refine_with_init(monkeypatch):
    # The coarse fit runs on the sampled rows with the sample budget; the refine
    # runs on all rows with the recipe's own budget, initialised from the
    # expanded positions. Landmarks (sampled rows) are preserved exactly.
    n, sample_n = 600, 500
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    recipe = le.RECIPES_BY_ID["tsne_sample500"]
    idx = le._tsne_sample_indices(n, sample_n, seed=42)
    coarse_ret: dict = {}
    calls: list = []

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        calls.append((len(reduced), init is None,
                      recipe.tsne_n_iter, recipe.tsne_early_exaggeration_iter))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        c = np.random.default_rng(seed).standard_normal(
            (len(reduced), 2)).astype(np.float32)
        coarse_ret["coords"] = c
        return c

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    coords, stage_t = le._fit_tsne_sample_init(reduced, recipe=recipe, seed=42, n_jobs=2)

    # Two fits: coarse (sample rows, sample budget, no init) then refine (all
    # rows, recipe budget, init supplied).
    assert len(calls) == 2
    coarse_call, refine_call = calls
    assert coarse_call[0] == sample_n
    assert coarse_call[1] is True  # no init on coarse
    assert coarse_call[2] == recipe.tsne_sample_n_iter  # coarse budget forwarded
    assert coarse_call[3] == recipe.tsne_sample_early_exaggeration_iter
    assert refine_call[0] == n
    assert refine_call[1] is False  # init passed to the full refinement
    assert refine_call[2] == recipe.tsne_n_iter  # final budget
    assert refine_call[3] == recipe.tsne_early_exaggeration_iter

    # The refinement ran with a non-coincident finite initialisation; the
    # sampled rows kept their exact coarse coordinates (landmarks preserved).
    assert np.isfinite(coords).all()
    np.testing.assert_array_equal(coords[idx], coarse_ret["coords"])


def test_sample_init_nonlandmark_rows_get_nearest_landmark_plus_jitter(monkeypatch):
    # Non-sampled rows start at their nearest landmark's coarse position plus a
    # tiny deterministic jitter — distinct from the landmark itself (so clusters
    # do not collapse onto a single 2-D point), yet close to it.
    n, sample_n = 20, 5
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    recipe = le.Recipe(
        "jitter", "jitter", pca_dim=50, method="tsne_sample_init",
        tsne_n_iter=10, tsne_early_exaggeration_iter=5,
        tsne_sample_n=sample_n, tsne_sample_n_iter=10,
        tsne_sample_early_exaggeration_iter=5,
    )
    idx = le._tsne_sample_indices(n, sample_n, seed=42)
    coarse_ret: dict = {}

    def _fake(reduced, *, recipe, seed, n_jobs, init=None):
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        # Coarse positions far apart so jitter (relative to coarse scale) is tiny.
        c = np.arange(len(reduced), dtype=np.float32)[:, None] * np.array(
            [[1.0, 0.0]], dtype=np.float32) * 10.0
        coarse_ret["coords"] = c
        return c

    monkeypatch.setattr(le, "fit_opentsne", _fake)
    coords, _ = le._fit_tsne_sample_init(reduced, recipe=recipe, seed=42, n_jobs=2)

    # Every full row's init position is near its nearest landmark (relative
    # jitter ~1e-3 of the coarse scale), and non-landmark rows do not sit
    # exactly on a landmark (tiny but nonzero offset).
    coarse = coarse_ret["coords"]
    landmark_positions = set(map(tuple, coarse.tolist()))
    coarse_scale = float(coarse.std(axis=0).max()) or 1.0
    for i in range(n):
        if i in idx:
            continue
        assert np.isfinite(coords[i]).all()
        # Close to some landmark (within jitter tolerance).
        assert (np.min(np.linalg.norm(coords[i] - coarse, axis=1)) < coarse_scale * 1e-2)
        # Not coincident with any landmark (deterministic jitter applied).
        assert tuple(coords[i].tolist()) not in landmark_positions
    # Landmarks unchanged.
    np.testing.assert_array_equal(coords[idx], coarse)


# ── real openTSNE smoke + timing split ───────────────────────────────────────


def test_sample_init_real_smoke_deterministic_and_finite():
    # A real openTSNE run through the coarse+expand+refine path (small custom
    # sample_n so the coarse stage actually runs on a tiny fixture). Fit twice →
    # identical coords; stage timings present and tsne = coarse+expand+refine.
    recipe = le.Recipe(
        "smoke", "smoke", pca_dim=50, method="tsne_sample_init",
        tsne_n_iter=20, tsne_early_exaggeration_iter=10,
        tsne_sample_n=12, tsne_sample_n_iter=20,
        tsne_sample_early_exaggeration_iter=10,
    )
    mat = np.random.RandomState(0).standard_normal((40, 50)).astype(np.float32)
    c1, timings = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    c2, _ = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    assert c1.shape == (40, 2)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1, c2)
    # tsne timing is the whole pipeline; the stage keys sum to it (total is
    # built elsewhere from pca+tsne+umap+alignment, so this must not double-count).
    # The stage windows are measured inside the tsne window, so tsne is at least
    # their sum (a little perf-counter overhead between stages is normal).
    assert timings["tsne"] >= timings["tsne_coarse"] + timings["tsne_expand"] \
        + timings["tsne_refine"] - 1e-6
    assert timings["tsne_coarse"] > 0.0
    assert timings["tsne_expand"] > 0.0
    assert timings["tsne_refine"] > 0.0


def test_sample_init_timing_split_in_fit_recipe():
    # fit_recipe routes tsne_sample_init to the pipeline and records the stage
    # keys without perturbing the existing tsne total.
    recipe = le.Recipe(
        "smoke2", "smoke2", pca_dim=50, method="tsne_sample_init",
        tsne_n_iter=10, tsne_early_exaggeration_iter=5,
        tsne_sample_n=8, tsne_sample_n_iter=10,
        tsne_sample_early_exaggeration_iter=5,
    )
    mat = np.random.RandomState(0).standard_normal((30, 50)).astype(np.float32)
    _coords, timings = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    assert set(("tsne_coarse", "tsne_expand", "tsne_refine")) <= set(timings)
    assert timings["tsne"] >= timings["tsne_refine"]


# ── generation: one parametrized run per built-in hierarchy set ──────────────

#: Every built-in hierarchy set, generated and route-isolated once. The first
#: recipe of each (``GALLERY_SETS[...].recipes[0]``) is its alignment reference.
HIERARCHY_GALLERY_SETS = [
    "hierarchy",
    "hierarchy-budget",
    "hierarchy-depth",
    "hierarchy-tuning",
    "hierarchy-distribution",
    "hierarchy-heuristic",
    "hierarchy-fractions",
    "hierarchy-density",
    "hierarchy-gamma",
    "hierarchy-combination",
]

#: Sets whose tiny fraction-derived coarse levels (2-3 rows) cannot be fitted by
#: real openTSNE; they use the deterministic stand-in fit.
FAKE_FIT_SETS = {
    "hierarchy-fractions",
    "hierarchy-density",
    "hierarchy-gamma",
    "hierarchy-combination",
}

#: Stage timing keys each multilevel method must record in its artifact.
STAGE_KEYS_BY_METHOD = {
    "tsne_sample_init": ("tsne_coarse", "tsne_expand", "tsne_refine"),
    "tsne_multilevel_init": ("tsne_level0", "tsne_expand", "tsne_final"),
}


@pytest.mark.parametrize("set_name", HIERARCHY_GALLERY_SETS)
def test_hierarchy_gallery_universe_alignment_and_route(
    set_name, tmp_path, layout_cache, layout_generate, layout_app, default_gallery
):
    ids = [r.id for r in le.GALLERY_SETS[set_name].recipes]
    reference = ids[0]
    fake = set_name in FAKE_FIT_SETS

    fc, infos = layout_cache(20)
    out = tmp_path / "set"
    manifest = layout_generate(fc, infos, out, ids, reference, fake=fake)

    # Same track universe, complete, in registry/generation order.
    assert manifest["status"] == "complete"
    assert list(manifest["recipes"]) == ids
    assert all(r["status"] == "complete" for r in manifest["recipes"].values())

    # Alignment onto the set's own reference — per dimension where the set pairs
    # 2-D and 3-D maps (only the combination set does).
    assert manifest["reference_id"] == reference
    assert manifest["reference_ids"]["2"] == reference
    for rid in ids:
        art = le.load_recipe_json(out, rid)
        dims = le.RECIPES_BY_ID[rid].dims or 2
        expected_ref = manifest["reference_ids"][str(dims)]
        assert art["aligned_to"] == (None if rid == expected_ref else expected_ref)
        assert art["dims"] == dims
        assert art["group"] == (le.RECIPES_BY_ID[rid].group or rid)
        # The artifact records the registered recipe verbatim (fields left None
        # are omitted, tuples arrive as JSON lists), so every set's expected
        # recipe params are pinned by the registry tests above.
        assert art["recipe"] == json.loads(
            json.dumps(le.recipe_params(le.RECIPES_BY_ID[rid]))
        )
        # Positions: the full universe, finite, with z only for 3-D maps.
        assert ("z" in art["positions"][0]) == (dims == 3)
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])
        if dims == 3:
            assert all(np.isfinite(p["z"]) for p in art["positions"])
        # tsne is the whole fit; its stage keys never leak into ``total``.
        assert {"tsne", "total", "alignment"} <= set(art["timings"])
        stages = STAGE_KEYS_BY_METHOD.get(le.RECIPES_BY_ID[rid].method, ())
        assert set(stages) <= set(art["timings"])
        if stages:
            assert art["timings"]["tsne"] >= max(art["timings"][k] for k in stages)

    # Same path set in every artifact of the set.
    paths = {p["path"] for p in le.load_recipe_json(out, reference)["positions"]}
    assert len(paths) == 20
    for rid in ids:
        assert {
            p["path"] for p in le.load_recipe_json(out, rid)["positions"]
        } == paths

    # ── route isolation ──
    root = tmp_path / "root"
    shutil.copytree(default_gallery, root)  # the default eight
    shutil.copytree(out, root / le.GALLERY_SETS[set_name].directory)
    client = layout_app(root).test_client()

    # set=<name> serves exactly this set's maps, each one resolvable by id.
    r = client.get(f"/api/layout-evaluation?set={set_name}")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(ids)
    for rid in ids:
        r = client.get(f"/api/layout-evaluation/{rid}?set={set_name}")
        assert r.status_code == 200
        assert r.get_json()["id"] == rid
    # None of the set's recipes leak into the base gallery, and a base recipe is
    # not served from the set; the allowlist still rejects unknown sets.
    for rid in ids:
        assert client.get(f"/api/layout-evaluation/{rid}").status_code == 404
    assert client.get(f"/api/layout-evaluation/baseline?set={set_name}").status_code == 404
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400
    # A sibling set with its own artifacts does not serve this set's maps: a
    # named set resolves only its own fixed directory.
    other = next(name for name in HIERARCHY_GALLERY_SETS if name != set_name)
    shutil.copytree(default_gallery, root / le.GALLERY_SETS[other].directory)
    r = client.get(f"/api/layout-evaluation?set={other}")
    assert r.status_code == 200
    for rid in ids:
        assert client.get(f"/api/layout-evaluation/{rid}?set={other}").status_code == 404
