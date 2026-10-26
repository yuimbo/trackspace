"""Paired 2-D/3-D continuous comparison gallery.

Adds the ``COMBINATION_RECIPES`` / ``COMBINATION_3D_RECIPES`` sets, the
``Recipe.dims`` / ``Recipe.group`` pairing fields, and the
``tsne_perplexity_start_fraction`` gamma anchor. Covers: pairing invariants,
fraction validation, the wide 1/256-anchored gamma schedule, a real 3-D openTSNE
smoke, per-dimension reference handling in generation, and artifact
serialisation/validation. The set's registry entry, universe/alignment and route
isolation are covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("openTSNE")

from dataclasses import replace

import backend.embeddings.layout_evaluation as le

REFERENCE_ID = "tsne_continuous16"
BASE_IDS = [
    "tsne_continuous16",
    "tsne_continuous16_wide",
    "tsne_continuous16_wide_gamma4",
]
THREE_D_IDS = [f"{rid}_3d" for rid in BASE_IDS]

N_LIB = 7153

#: Effective per-level perplexities of the wide gamma-4 chain at the current
#: library size (28..447 rows; the low-count levels clamp to (count−1)/3).
WIDE_GAMMA4_PERPS = [
    9.0, 11.0, 13.0, 16.0, 19.333333333333332, 23.0, 28.0,
    33.666666666666664, 40.666666666666664, 42.04465954706207,
    39.90238812004042, 38.0630016409218, 36.47281114436358,
    35.15376288699215, 34.039916771202804, 33.125644250473586,
]


# ── registry + pairing invariants ────────────────────────────────────────────


def test_combination_dims_and_groups():
    for rid in BASE_IDS:
        r = le.RECIPES_BY_ID[rid]
        assert r.dims is None and r.group is None
    for rid, base in zip(THREE_D_IDS, BASE_IDS):
        r = le.RECIPES_BY_ID[rid]
        assert r.dims == 3
        assert r.group == base
        assert r.label == f"{le.RECIPES_BY_ID[base].label} (3D)"
        # Everything except identity/dims/group is inherited.
        base_recipe = le.RECIPES_BY_ID[base]
        assert r.tsne_sample_fractions == base_recipe.tsne_sample_fractions
        assert r.tsne_coarse_iteration_budget == base_recipe.tsne_coarse_iteration_budget
        assert r.tsne_perplexity_gamma == base_recipe.tsne_perplexity_gamma
        assert (r.tsne_perplexity_start_fraction
                == base_recipe.tsne_perplexity_start_fraction)


def test_combination_wide_recipe_parameters_and_old_identity():
    wide = le.RECIPES_BY_ID["tsne_continuous16_wide"]
    gamma4 = le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4"]
    assert wide.tsne_sample_fractions == le._FRACTIONS_DENSITY256_16
    assert wide.tsne_sample_fractions[0] == pytest.approx(1 / 256)
    assert wide.tsne_sample_fractions[-1] == pytest.approx(1 / 16)
    assert wide.tsne_coarse_iteration_budget == 16
    assert wide.tsne_perplexity_gamma is None
    assert gamma4.tsne_perplexity_gamma == 4.0
    assert gamma4.tsne_perplexity_start_fraction == pytest.approx(1 / 256)
    # The old control's omitted optional fields and pinned identity are covered
    # by test_optional_recipe_fields_are_omitted_from_identity.


# ── validation ───────────────────────────────────────────────────────────────


def test_start_fraction_validation():
    base = le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4"]
    for bad in (0.0, 1.0, -0.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            le.resolve_multilevel_schedule(
                replace(base, tsne_perplexity_start_fraction=bad), 100
            )
    # The anchor is gamma-only.
    with pytest.raises(ValueError):
        le.resolve_multilevel_schedule(
            replace(base, tsne_perplexity_gamma=None), 100
        )
    # Orphaned on a fixed-level recipe.
    fixed = le.RECIPES_BY_ID["tsne_chain50_250_1000"]
    with pytest.raises(ValueError):
        le.resolve_multilevel_schedule(
            replace(fixed, tsne_perplexity_start_fraction=1 / 64), 3000
        )


def test_dims_validation_rejects_non_multilevel_3d():
    pca = replace(le.RECIPES_BY_ID["pca2"], dims=3)
    with pytest.raises(ValueError):
        le.fit_recipe(np.zeros((10, 5)), pca, seed=42, n_jobs=2)
    bad = replace(le.RECIPES_BY_ID["tsne_continuous16_3d"], dims=4)
    with pytest.raises(ValueError):
        le.fit_recipe(np.zeros((10, 5)), bad, seed=42, n_jobs=2)


# ── wide gamma schedule ──────────────────────────────────────────────────────


def test_wide_gamma4_schedule_anchors_at_256():
    recipe = le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4"]
    sched = le.resolve_multilevel_schedule(recipe, N_LIB)
    assert [s[0] for s in sched] == [
        28, 34, 40, 49, 59, 70, 85, 102, 123, 147, 177, 213, 257, 309, 372, 447,
    ]
    assert [s[1] for s in sched] == [1] * 16
    assert [s[2] for s in sched] == [100] + [0] * 15
    assert [s[3] for s in sched] == pytest.approx(WIDE_GAMMA4_PERPS, rel=1e-3)
    # The default (None = n/64) schedule's first small level keeps its old
    # requested-clamp behaviour; the wide anchor is what pulls it to 9.
    default = le.resolve_multilevel_schedule(
        replace(
            recipe,
            tsne_sample_fractions=le._FRACTIONS_DENSITY16,
            tsne_perplexity_start_fraction=None,
        ),
        N_LIB,
    )
    assert default[0][0] == 112


# ── real 3-D openTSNE smoke ──────────────────────────────────────────────────


def test_three_d_real_smoke_shape_finite_deterministic():
    recipe = le.Recipe(
        "smoke3d", "smoke3d", pca_dim=50, method="tsne_multilevel_init",
        tsne_n_iter=20, tsne_early_exaggeration_iter=0,
        tsne_sample_fractions=(0.25, 0.5), perplexities=(30.0,), dims=3,
    )
    mat = np.random.RandomState(0).standard_normal((40, 50)).astype(np.float32)
    c1, _ = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    c2, _ = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    assert c1.shape == (40, 3)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1, c2)


def test_expand_landmark_jitter_uses_coordinate_width():
    reduced = np.random.RandomState(0).standard_normal((10, 4)).astype(np.float32)
    idx = np.array([0, 2, 5])
    coords = np.random.RandomState(1).standard_normal((3, 3)).astype(np.float32)
    out = le._expand_landmark_init(
        reduced, idx, coords, np.arange(10), seed=42, n_jobs=2
    )
    assert out.shape == (10, 3)


def test_orthogonal_align_width_guard():
    paths = [f"p{i}" for i in range(4)]
    with pytest.raises(ValueError, match="2-D reference"):
        le.orthogonal_align_to(
            np.zeros((4, 3), dtype=np.float32), paths,
            np.zeros((4, 2), dtype=np.float32), paths,
        )


# ── paired generation + artifact validation ─────────────────────────────────


def _run(layout_cache, layout_generate, out, recipe_ids, reference=REFERENCE_ID):
    fc, infos = layout_cache(20)
    return layout_generate(fc, infos, out, recipe_ids, reference, fake=True)


def test_paired_generation_per_dimension_references(
    tmp_path, layout_cache, layout_generate
):
    out = tmp_path / "combination"
    # Deliberately omit both references from the explicit list: the 2-D base
    # reference and its explicit 3-D sibling are auto-included.
    requested = [
        "tsne_continuous16_wide",
        "tsne_continuous16_wide_gamma4",
        "tsne_continuous16_wide_3d",
        "tsne_continuous16_wide_gamma4_3d",
    ]
    manifest = _run(layout_cache, layout_generate, out, requested)

    expected = [
        *BASE_IDS,
        *THREE_D_IDS,
    ]
    assert list(manifest["recipes"]) == expected  # 2-D group first, then 3-D
    assert manifest["reference_id"] == REFERENCE_ID
    assert manifest["reference_ids"] == {
        "2": REFERENCE_ID,
        "3": f"{REFERENCE_ID}_3d",
    }
    for rid in BASE_IDS:
        entry = manifest["recipes"][rid]
        assert entry["dims"] == 2 and entry["group"] == rid
        art = le.load_recipe_json(out, rid)
        assert art["dims"] == 2 and art["group"] == rid
        assert all("z" not in p for p in art["positions"])
        if rid != REFERENCE_ID:
            assert art["aligned_to"] == REFERENCE_ID
    for rid, base in zip(THREE_D_IDS, BASE_IDS):
        entry = manifest["recipes"][rid]
        assert entry["dims"] == 3 and entry["group"] == base
        art = le.load_recipe_json(out, rid)
        assert art["dims"] == 3 and art["group"] == base
        assert all(np.isfinite(p["z"]) for p in art["positions"])
        if rid != f"{REFERENCE_ID}_3d":
            assert art["aligned_to"] == f"{REFERENCE_ID}_3d"


def test_paired_load_rejects_3d_missing_z(tmp_path, layout_cache, layout_generate):
    out = tmp_path / "combination"
    _run(
        layout_cache, layout_generate, out,
        ["tsne_continuous16_wide", "tsne_continuous16_wide_3d"],
    )
    path = out / "tsne_continuous16_wide_3d.json"
    data = json.loads(path.read_text())
    for pos in data["positions"]:
        pos.pop("z")
    path.write_text(json.dumps(data))
    assert le.load_recipe_json(str(out), "tsne_continuous16_wide_3d") is None


def test_three_d_without_registered_sibling_reference_is_rejected(
    tmp_path, layout_cache, layout_generate
):
    with pytest.raises(ValueError, match="no registered 3-D sibling"):
        _run(
            layout_cache, layout_generate, tmp_path / "bad",
            ["tsne_continuous16_wide_3d"], reference="baseline",
        )
