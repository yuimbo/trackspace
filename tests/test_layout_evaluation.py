"""Layout-evaluation: recipe fits, alignment/normalisation, manifest, and GET routes.

Covers the offline eval surface added for the browser gallery: the eight recipes
fit deterministically and with correct small-n clamping, alignment is
rotation/reflection only, normalisation is aspect-preserving, the manifest never
serves stale artifacts, the generation never touches the production fit caches,
and the read-only GET routes validate safely.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le
import backend.embeddings.projection_backends as pb
from backend.embeddings.clap import EMBEDDING_VERSION
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.layout import prepare_projection_input
from backend.embeddings.librosa_audio_features import FEATURES_VERSION
from backend.embeddings.maest import MAEST_VERSION
from backend.embeddings.projection_config import (
    FROZEN_FEATURE_MASK,
    FROZEN_SOURCES,
    FROZEN_SOURCE_WEIGHTS,
    REQUIRED_SOURCES,
)
from backend.embeddings.rhythm_features import RHYTHM_VERSION


def _build_cache(tmp_path, n: int, seed: int = 0):
    fc = FeatureCache(str(tmp_path / "audio_features.db"))
    rng = np.random.RandomState(seed)
    fps = [f"fp{i:03d}" for i in range(n)]
    for i, fp in enumerate(fps):
        fc.put("maest_logits", fp, rng.standard_normal(519).astype(np.float32), MAEST_VERSION)
        fc.put("maest", fp, rng.standard_normal(768).astype(np.float32), MAEST_VERSION)
        fc.put("rhythm", fp, rng.standard_normal(14).astype(np.float32), RHYTHM_VERSION)
        fc.put("clap", fp, rng.standard_normal(512).astype(np.float32), EMBEDDING_VERSION)
    infos = [{"path": f"lib/{fp}.mp3", "fingerprint": fp} for fp in fps]
    return fc, infos


def _run(fc, infos, out, *, recipe_ids=None, limit=None):
    return le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=out,
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=2,
        recipe_ids=recipe_ids,
        limit=limit,
    )


@pytest.fixture(scope="module")
def gallery(tmp_path_factory):
    """One real eight-recipe generation, shared by the read-only tests.

    Fitting the default eight maps is the expensive part of this suite (UMAP's
    first call includes a Numba JIT compile); tests that only read a completed
    generation reuse this one, copying the dir when they must mutate it. The
    cache includes the ``supervised`` block so the source-mix sets (ablation)
    can reuse the same fixture.
    """
    pytest.importorskip("umap")
    root = tmp_path_factory.mktemp("gallery")
    fc, infos = _build_cache_with_supervised(root, 10)
    out = root / "layout-evaluation"
    _run(fc, infos, out)
    return fc, infos, out


# ── input preparation is shared ──────────────────────────────────────────────


def test_prepare_projection_input_no_supervision_matches_gather(tmp_path):
    fc, infos = _build_cache(tmp_path, 8)
    inp = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
    )
    # With no folder/tag supervision the projection matrix is the raw composite
    # and the t-SNE config is the plain vector config — exactly what production
    # computes before the (absent) semantic weighting.
    assert inp.mat_proj.shape == inp.mat.shape
    assert np.array_equal(inp.mat_proj, inp.mat)
    assert inp.tsne_config == inp.comp_config
    assert inp.mat.shape[0] == 8


def test_prepare_projection_input_one_track_skips_semantic(tmp_path, monkeypatch):
    fc, infos = _build_cache(tmp_path, 1)
    called = []
    monkeypatch.setattr(
        "backend.embeddings.layout._build_semantic_basis",
        lambda *a, **k: called.append(1),
    )
    monkeypatch.setattr("backend.embeddings.layout.is_model_ready", lambda: True)
    inp = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        scale_folders=True,
    )
    # A single track cannot project or re-weight: return the unweighted bundle
    # without building any semantic basis, even with a ready model.
    assert called == []
    assert inp.mat_proj is inp.mat
    assert inp.tsne_config == inp.comp_config


def test_prepare_projection_input_allzero_safe(tmp_path, monkeypatch):
    fc, _infos = _build_cache(tmp_path, 3)
    called = []
    monkeypatch.setattr(
        "backend.embeddings.layout._build_semantic_basis",
        lambda *a, **k: called.append(1),
    )
    monkeypatch.setattr("backend.embeddings.layout.is_model_ready", lambda: True)
    # No cached fingerprints means no track is placed: an empty matrix. The
    # degenerate guard must short-circuit without any semantic inference.
    infos = [{"path": f"lib/{i}.mp3", "fingerprint": f"missing{i}"} for i in range(3)]
    inp = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        scale_folders=True,
    )
    assert called == []
    assert inp.mat.shape == (0, 0)
    assert inp.mat_proj.shape == (0, 0)
    assert inp.tsne_config == inp.comp_config


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_pca_clamps_to_feature_dim():
    mat = np.random.RandomState(0).standard_normal((20, 6)).astype(np.float32)
    reduced, pca = le.pca_reduce(mat, 50, seed=42)
    assert reduced.shape == (20, 5)  # n_features-1
    assert pca.n_components == 5


def test_fit_recipe_is_deterministic():
    mat = np.random.RandomState(0).standard_normal((40, 50)).astype(np.float32)
    c1, _ = le.fit_recipe(mat, le.RECIPES_BY_ID["baseline"], seed=42, n_jobs=2)
    c2, _ = le.fit_recipe(mat, le.RECIPES_BY_ID["baseline"], seed=42, n_jobs=2)
    assert c1.shape == (40, 2)
    assert np.array_equal(c1, c2)


def test_fit_recipe_small_n_neighbors_clamped():
    mat = np.random.RandomState(0).standard_normal((8, 50)).astype(np.float32)
    coords, timings = le.fit_recipe(mat, le.RECIPES_BY_ID["umap2_n60"], seed=42, n_jobs=2)
    assert coords.shape == (8, 2)
    assert np.isfinite(coords).all()


def test_fit_recipe_rejects_nonfinite(monkeypatch):
    mat = np.random.RandomState(0).standard_normal((20, 8)).astype(np.float32)

    def _nan(*args, **kwargs):
        return np.full((20, 2), np.nan, dtype=np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _nan)
    with pytest.raises(ValueError, match="non-finite"):
        le.fit_recipe(mat, le.RECIPES_BY_ID["baseline"], seed=42, n_jobs=2)


def test_fit_opentsne_scalar_perplexity(monkeypatch):
    # A one-element recipe.perplexities fixes the scalar perplexity on TSNE.
    recorded: dict = {}

    class FakeTSNE:
        def __init__(self, **kw):
            recorded["kwargs"] = kw

        def fit(self, X, **kw):
            return np.zeros((len(X), 2), dtype=np.float32)

    monkeypatch.setattr("openTSNE.TSNE", FakeTSNE)
    reduced = np.random.RandomState(0).standard_normal((200, 5)).astype(np.float64)
    le.fit_opentsne(reduced, recipe=le.RECIPES_BY_ID["tsne_p60"], seed=42, n_jobs=2)
    assert recorded["kwargs"]["perplexity"] == 60.0
    assert recorded["kwargs"]["n_iter"] == 100
    assert recorded["kwargs"]["early_exaggeration_iter"] == 250


def test_fit_opentsne_multiscale_uses_multiscale_mixture(monkeypatch):
    # Two or more perplexities route through openTSNE's MultiscaleMixture and
    # the fitted map is produced against that explicit affinity matrix.
    calls: dict = {}

    class FakeAffinity:
        def __init__(self, data, perplexities=None, n_jobs=1, random_state=None):
            calls["affinity"] = (perplexities, n_jobs, random_state)

    class FakeTSNE:
        def __init__(self, **kw):
            calls["tsne_kwargs"] = kw

        def fit(self, X=None, affinities=None, initialization=None):
            calls["fit"] = (affinities is not None, X is not None)
            return np.zeros((len(X), 2), dtype=np.float32)

    monkeypatch.setattr("openTSNE.affinity.MultiscaleMixture", FakeAffinity)
    monkeypatch.setattr("openTSNE.TSNE", FakeTSNE)
    reduced = np.random.RandomState(0).standard_normal((200, 5)).astype(np.float64)
    le.fit_opentsne(
        reduced, recipe=le.RECIPES_BY_ID["tsne_multiscale30_120"], seed=42, n_jobs=2
    )
    assert calls["affinity"][0] == [30.0, 120.0]
    assert calls["affinity"][1] == 2  # n_jobs forwarded
    assert calls["fit"] == (True, True)  # affinities + X passed to tsne.fit
    assert "perplexity" not in calls["tsne_kwargs"]


def test_scale_recipes_safe_on_tiny_n():
    # Requested perplexities exceed the track count: the clamp must keep both the
    # scalar and the multiscale recipes finite and 2-D on a tiny fixture.
    mat = np.random.RandomState(0).standard_normal((8, 50)).astype(np.float32)
    for rid in ("tsne_p60", "tsne_p120", "tsne_multiscale30_120"):
        coords, _ = le.fit_recipe(mat, le.RECIPES_BY_ID[rid], seed=42, n_jobs=2)
        assert coords.shape == (8, 2)
        assert np.isfinite(coords).all()


def test_clamp_perplexity_for_small_n():
    assert le._clamp_perplexity(60.0, 8) == 7.0
    assert le._clamp_perplexity(120.0, 8) == 7.0
    # No clamp at real library scale.
    assert le._clamp_perplexity(120.0, 2000) == 120.0


def test_pca2_recipe_is_pure_pca(monkeypatch):
    recipe = le.RECIPES_BY_ID["pca2"]
    assert recipe.method == "pca"
    assert recipe.pca_dim == 2
    mat = np.random.RandomState(0).standard_normal((12, 50)).astype(np.float32)
    calls: list[str] = []

    def _tsne(*a, **k):
        calls.append("tsne")
        return np.zeros((12, 2), dtype=np.float32)

    def _umap(*a, **k):
        calls.append("umap")
        return np.zeros((12, 2), dtype=np.float32)

    monkeypatch.setattr(le, "fit_opentsne", _tsne)
    monkeypatch.setattr(le, "fit_umap", _umap)
    coords, timings = le.fit_recipe(mat, recipe, seed=42, n_jobs=2)
    # A pure-PCA recipe must never call the t-SNE or UMAP primitives.
    assert calls == []
    assert coords.shape == (12, 2)
    assert np.isfinite(coords).all()
    assert timings["tsne"] == 0.0
    assert timings["umap"] == 0.0


def test_pca2_pads_tiny_dim_to_2d():
    # One feature forces the PCA clamp down to a single component; the map is
    # padded to a 2-D row so it stays comparable with the other recipes.
    mat = np.random.RandomState(0).standard_normal((8, 1)).astype(np.float32)
    coords, timings = le.fit_recipe(mat, le.RECIPES_BY_ID["pca2"], seed=42, n_jobs=2)
    assert coords.shape == (8, 2)
    assert np.isfinite(coords).all()
    assert timings["tsne"] == 0.0
    assert timings["umap"] == 0.0


def test_normalise_aspect_preserves_aspect_ratio():
    # Wide x, narrow y.
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 5.0], [10.0, 5.0]], dtype=np.float64)
    normed = le.normalise_aspect(coords)
    xs = normed[:, 0].max() - normed[:, 0].min()
    ys = normed[:, 1].max() - normed[:, 1].min()
    # Uniform scale: x range stays 2x the y range (aspect preserved), inside [0,1].
    assert abs(xs - 2.0 * ys) < 1e-6
    assert normed.min() >= -1e-9 and normed.max() <= 1.0 + 1e-9
    assert abs(xs - 1.0) < 1e-6  # the wider axis fills the unit square


def test_orthogonal_align_is_rotation_only():
    base = np.random.RandomState(1).standard_normal((20, 2))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = (rot @ base.T).T + np.array([5.0, -3.0])  # rotate + translate
    aligned = le.orthogonal_align_to(
        transformed, [f"p{i}" for i in range(20)],
        base, [f"p{i}" for i in range(20)],
    )
    # Rotation + translation is an isometry: pairwise distances are preserved.
    a = base - base.mean(axis=0)
    b = aligned - aligned.mean(axis=0)
    assert np.allclose((a**2).sum(axis=1), (b**2).sum(axis=1), atol=1e-6)


def test_orthogonal_align_matches_legacy_formula():
    # The shared matched-index + Procrustes refactor must stay numerically
    # identical to the original inline formula for finite inputs, including a
    # partial overlap (tracks present in only one of the two maps).
    rng = np.random.RandomState(3)
    ref_paths = [f"p{i}" for i in range(24)]
    paths = [f"p{i}" for i in range(20) if i % 3 != 0] + ["only-a"]
    ref_coords = rng.standard_normal((24, 2))
    coords = rng.standard_normal((len(paths), 2))
    # The pre-refactor implementation, embedded verbatim.
    ref_index = {p: i for i, p in enumerate(ref_paths)}
    idx_new = [i for i, p in enumerate(paths) if p in ref_index]
    idx_ref = [ref_index[p] for p in paths if p in ref_index]
    rot, _scale, src_mu, dst_mu = le.procrustes(
        coords[idx_new], ref_coords[idx_ref]
    )
    expected = le.apply_procrustes(coords, rot, 1.0, src_mu, dst_mu)
    got = le.orthogonal_align_to(coords, paths, ref_coords, ref_paths)
    np.testing.assert_array_equal(got, expected)


def test_orthogonal_align_rejects_nonfinite(monkeypatch):
    # A degenerate alignment must not write NaN positions into an artifact: it
    # raises a clear error so the recipe records error status instead.
    rng = np.random.RandomState(0)
    coords = rng.standard_normal((10, 2))
    ref_coords = rng.standard_normal((10, 2))
    paths = [f"p{i}" for i in range(10)]
    monkeypatch.setattr(
        le,
        "apply_procrustes",
        lambda *a, **k: np.full_like(coords, np.nan),
    )
    with pytest.raises(ValueError, match="non-finite"):
        le.orthogonal_align_to(coords, paths, ref_coords, paths)


# ── identity hashes ──────────────────────────────────────────────────────────


def test_dataset_fingerprint_stability_and_sensitivity():
    paths = ["a.mp3", "b.mp3"]
    mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cfg = "C1F1M1R1S0|maest_logits,maest|00|0.42|maest_logits:0.45"
    fp1 = le.dataset_fingerprint(paths, mat, cfg)
    assert le.dataset_fingerprint(paths, mat, cfg) == fp1
    assert le.dataset_fingerprint(["a.mp3", "c.mp3"], mat, cfg) != fp1
    assert le.dataset_fingerprint(paths, mat + 1.0, cfg) != fp1
    assert le.dataset_fingerprint(paths, mat, cfg + "x") != fp1


def test_recipe_revision_differs_across_recipes():
    fp = "deadbeef"
    revs = {rid: le.recipe_revision(fp, r) for rid, r in le.RECIPES_BY_ID.items()}
    assert len(set(revs.values())) == len(revs)


def test_recipe_revision_changes_with_seed_and_versions():
    recipe = le.RECIPES_BY_ID["baseline"]
    fp = "deadbeef"
    versions = {"clap": 1, "maest": 2}
    base = le.recipe_revision(fp, recipe, seed=42, versions=versions)
    # Stable given the same inputs.
    assert le.recipe_revision(fp, recipe, seed=42, versions=versions) == base
    # A different seed changes the fitted output, so the revision must change.
    assert le.recipe_revision(fp, recipe, seed=43, versions=versions) != base
    # A changed embedding-source / library version changes the output identity.
    assert le.recipe_revision(fp, recipe, seed=42, versions={"clap": 2, "maest": 2}) != base
    # The effective worker count is part of the fitted-map identity: approximate
    # neighbour stages can vary with it.
    assert le.recipe_revision(fp, recipe, seed=42, n_jobs=2, versions=versions) != base
    # The fingerprint is input-specific and deliberately does not capture the
    # environment, so two runs with different environments but the same input
    # must still diverge at the revision level.
    assert le.recipe_revision(fp, recipe, seed=42, versions=None) != base


def test_recipe_identity_preserved_for_original_recipes():
    # The optional perplexities field must not change the identity of recipes
    # that don't set it: their recorded params omit it and their revision hash
    # is unchanged (pinned against the pre-scale-known values).
    r = le.RECIPES_BY_ID["baseline"]
    assert "perplexities" not in le.recipe_params(r)
    assert le.recipe_revision("deadbeef", r) == (
        "23a524bc54549fe9ab36d35a2f86172aabac597980e3b922829066308c302886"
    )
    assert le.recipe_revision("deadbeef", le.RECIPES_BY_ID["tsne_budget100"]) == (
        "0fc99b8ff67652ea8bc4034545fe02c491f96be62c7a8abf55a45bba4304189b"
    )
    # Setting a perplexity changes the revision vs the identical recipe without
    # it, and is recorded in the recipe params.
    base = dict(pca_dim=50, method="tsne", tsne_n_iter=100,
                tsne_early_exaggeration_iter=250)
    plain = le.Recipe("x", "x", **base)
    with_perps = le.Recipe("x", "x", **base, perplexities=(30.0, 120.0))
    assert "perplexities" in le.recipe_params(with_perps)
    assert "perplexities" not in le.recipe_params(plain)
    assert le.recipe_revision("deadbeef", with_perps) != le.recipe_revision(
        "deadbeef", plain
    )


# ── gallery-set registry ────────────────────────────────────────────────────


def test_gallery_set_registry_is_derived_from_one_mapping():
    # The default generation is still exactly the original eight.
    assert [r.id for r in le.RECIPES] == [
        "baseline", "pca2", "pca25", "tsne_budget",
        "umap2_n15", "umap2_n60", "umap_tsne", "tsne_umap_init",
    ]
    assert le.GALLERY_SETS["base"].recipes is le.RECIPES
    assert le.GALLERY_SETS["base"].directory is None

    # RECIPES_BY_ID is the exact union of the built-in sets — no second,
    # hand-maintained list of recipe ids.
    builtin_ids = {
        recipe.id
        for gallery_set in le.GALLERY_SETS.values()
        if gallery_set.recipes is not None
        for recipe in gallery_set.recipes
    }
    assert builtin_ids == set(le.RECIPES_BY_ID)

    for name, gallery_set in le.GALLERY_SETS.items():
        if gallery_set.recipes is None:
            continue
        # The documented control/reference leads the alignment order.
        assert gallery_set.recipes[0].id in le.RECIPES_BY_ID, name


def test_gallery_sets_share_the_same_recipe_objects():
    # A registry refactor must not fork a frozen identity: shared controls are
    # the same object in every set and in the registry.
    for gallery_set in le.GALLERY_SETS.values():
        for recipe in gallery_set.recipes or ():
            assert le.RECIPES_BY_ID[recipe.id] is recipe
    assert le.RECIPES_BY_ID["tsne_chain50_250_1000_final50"] is (
        le.TSNE_CHAIN50_250_1000_FINAL50
    )
    assert le.GALLERY_SETS["hierarchy-heuristic"].recipes[0] is (
        le.TSNE_CHAIN50_250_1000_FINAL50
    )
    assert le.GALLERY_SETS["hierarchy-density"].recipes[0] is le.TSNE_GEO8_COMBINED
    assert le.GALLERY_SETS["hierarchy-gamma"].recipes[0] is le.TSNE_DENSITY4


def test_gallery_set_routes_derive_and_external_sets_are_honest():
    from backend.routes.api_layout_evaluation import SET_SUBDIRS

    assert SET_SUBDIRS == {
        name: gallery_set.directory
        for name, gallery_set in le.GALLERY_SETS.items()
    }
    assert SET_SUBDIRS["base"] is None
    for name in (
        "genre", "genre-directed", "genre-logits-ab",
        "folder-separation", "folder-cheap",
    ):
        assert SET_SUBDIRS[name] == name
        assert le.GALLERY_SETS[name].recipes is None


# ── generation: end-to-end, small-n, cache isolation, stale handling ─────────


def test_generation_produces_all_eight(gallery):
    _fc, _infos, out = gallery
    manifest = le.manifest_dict(out)
    assert manifest["status"] == "complete"
    assert manifest["n_jobs"] == 2
    assert "umap_jit" in manifest["notes"]
    assert len(manifest["recipes"]) == 8
    assert all(r["status"] == "complete" for r in manifest["recipes"].values())
    # An unspecified generation runs exactly the original eight — the registry
    # also holds the selectable extras (budget/scale/early/mixes/ablation).
    assert len(le.RECIPES) == 8
    assert set(r.id for r in le.RECIPES) < set(le.RECIPES_BY_ID)
    assert sorted(manifest["recipes"]) == sorted(r.id for r in le.RECIPES)
    # Same path set in every artifact.
    paths = {p["path"] for p in le.load_recipe_json(out, "baseline")["positions"]}
    for rid in manifest["recipes"]:
        art = le.load_recipe_json(out, rid)
        assert {p["path"] for p in art["positions"]} == paths
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])
    # Baseline is the alignment reference; others align onto it.
    assert manifest["recipes"]["baseline"]["aligned_to"] is None
    assert manifest["recipes"]["pca2"]["aligned_to"] == "baseline"


def test_explicit_budget_set_accepted(tmp_path):
    pytest.importorskip("umap")
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "layout-evaluation"
    budget = ["baseline", "tsne_budget100", "tsne_budget150", "tsne_budget"]
    manifest = _run(fc, infos, out, recipe_ids=budget)
    assert manifest["status"] == "complete"
    assert sorted(manifest["recipes"]) == sorted(budget)
    assert all(r["status"] == "complete" for r in manifest["recipes"].values())
    # Baseline is the alignment reference even within a reduced set.
    assert manifest["recipes"]["baseline"]["aligned_to"] is None
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] == "baseline"


def test_budget_recipes_use_fixed_early_exaggeration():
    for rid, n_iter in (
        ("tsne_budget100", 100),
        ("tsne_budget150", 150),
        ("tsne_budget", 250),
    ):
        recipe = le.RECIPES_BY_ID[rid]
        assert recipe.method == "tsne"
        assert recipe.pca_dim == 50
        assert recipe.tsne_n_iter == n_iter
        assert recipe.tsne_early_exaggeration_iter == 250


def test_early_recipes_use_expected_iter_budget():
    # The "early" gallery set: production's n_iter=100 with only the
    # early-exaggeration phase shortened. The control (tsne_budget100) keeps its
    # identity; the two variants set early 150 and 100. Nothing else differs.
    for rid, early in (
        ("tsne_budget100", 250),
        ("tsne_early150", 150),
        ("tsne_early100", 100),
    ):
        recipe = le.RECIPES_BY_ID[rid]
        assert recipe.method == "tsne"
        assert recipe.pca_dim == 50
        assert recipe.tsne_n_iter == 100
        assert recipe.tsne_early_exaggeration_iter == early
        assert recipe.perplexities is None  # production scalar perplexity rule
        assert recipe.source_weights is None  # default weights, unsupervised
    # The variants live only in the registry, never the default eight.
    assert "tsne_early150" not in {r.id for r in le.RECIPES}
    assert "tsne_early100" not in {r.id for r in le.RECIPES}


def test_fit_opentsne_early_recipes_pass_their_early_phase(monkeypatch):
    # The recipe's early-exaggeration budget is forwarded to openTSNE verbatim:
    # n_iter stays 100 (production's final budget), only the early phase shortens.
    recorded: dict = {}

    class FakeTSNE:
        def __init__(self, **kw):
            recorded["kwargs"] = kw

        def fit(self, X, **kw):
            return np.zeros((len(X), 2), dtype=np.float32)

    monkeypatch.setattr("openTSNE.TSNE", FakeTSNE)
    reduced = np.random.RandomState(0).standard_normal((200, 5)).astype(np.float64)
    for rid, early in (("tsne_early150", 150), ("tsne_early100", 100)):
        le.fit_opentsne(reduced, recipe=le.RECIPES_BY_ID[rid], seed=42, n_jobs=2)
        assert recorded["kwargs"]["n_iter"] == 100
        assert recorded["kwargs"]["early_exaggeration_iter"] == early


def test_early_recipes_distinct_revisions_and_original_unchanged():
    assert len(le.RECIPES) == 8
    # Each early recipe gets its own distinct revision (the early field is part
    # of the identity); the pinned identities of the original recipes live in
    # test_recipe_identity_preserved_for_original_recipes.
    revs = {
        rid: le.recipe_revision("deadbeef", le.RECIPES_BY_ID[rid])
        for rid in ("tsne_budget100", "tsne_early150", "tsne_early100")
    }
    assert len(set(revs.values())) == 3


def _run_scale(fc, infos, out, *, n_jobs=2):
    scale = ["tsne_budget100", "tsne_p60", "tsne_p120", "tsne_multiscale30_120"]
    return le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=str(out),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=n_jobs,
        recipe_ids=scale,
        reference_id="tsne_budget100",
    )


def test_scale_gallery_four_maps_reference_control_no_auto_baseline(tmp_path):
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "scale"
    manifest = _run_scale(fc, infos, out)
    scale = ["tsne_budget100", "tsne_p60", "tsne_p120", "tsne_multiscale30_120"]
    assert manifest["status"] == "complete"
    assert manifest["n_jobs"] == 2
    assert "umap_jit" not in manifest["notes"]
    # Exactly the four scale recipes — the baseline (500-iteration) map must NOT
    # be auto-added just to serve as the alignment reference.
    assert sorted(manifest["recipes"]) == sorted(scale)
    assert manifest["reference_id"] == "tsne_budget100"
    # The control is the reference (aligned_to None); the rest align onto it.
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] is None
    for rid in ("tsne_p60", "tsne_p120", "tsne_multiscale30_120"):
        assert manifest["recipes"][rid]["aligned_to"] == "tsne_budget100"
    # Same path set in every artifact, all finite.
    paths = {p["path"] for p in le.load_recipe_json(out, "tsne_budget100")["positions"]}
    for rid in scale:
        art = le.load_recipe_json(out, rid)
        assert {p["path"] for p in art["positions"]} == paths
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])


def test_early_gallery_three_maps_reference_control(tmp_path):
    # The "early" set: the budget-100 control plus the two shortened
    # early-phase variants, fitted at production's n_iter=100 and aligned onto
    # the control — no baseline (500-iteration) map is auto-added.
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "early"
    early = ["tsne_budget100", "tsne_early150", "tsne_early100"]
    manifest = le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=str(out),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=2,
        recipe_ids=early,
        reference_id="tsne_budget100",
    )
    assert manifest["status"] == "complete"
    assert sorted(manifest["recipes"]) == sorted(early)
    assert manifest["reference_id"] == "tsne_budget100"
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] is None
    for rid in ("tsne_early150", "tsne_early100"):
        assert manifest["recipes"][rid]["aligned_to"] == "tsne_budget100"
    # Same path set in every artifact, all finite; control revision matches the
    # standalone budget-100 map (identity-identical recipe + same input).
    paths = {p["path"] for p in le.load_recipe_json(out, "tsne_budget100")["positions"]}
    for rid in early:
        art = le.load_recipe_json(out, rid)
        assert {p["path"] for p in art["positions"]} == paths
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])


def _run_mixes(fc, infos, out, *, n_jobs=2):
    mixes = ["tsne_budget100", "tsne_rhythm_forward", "tsne_texture_style"]
    return le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=str(out),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=n_jobs,
        recipe_ids=mixes,
        reference_id="tsne_budget100",
    )


def test_mix_recipes_are_complete_vectors_over_frozen_sources():
    # The two mix extras are complete named-pair vectors in FROZEN_SOURCES order,
    # summing to 1.10 (matching the production default) with features at 0 and the
    # required sources positive, exactly as in production.
    for rid in ("tsne_rhythm_forward", "tsne_texture_style"):
        r = le.RECIPES_BY_ID[rid]
        assert r.method == "tsne" and r.pca_dim == 50
        assert r.tsne_n_iter == 100 and r.tsne_early_exaggeration_iter == 250
        assert r.perplexities is None
        assert r.source_weights is not None
        d = dict(r.source_weights)
        assert list(d) == list(FROZEN_SOURCES)
        assert abs(sum(d.values()) - 1.10) < 1e-9
        assert d["features"] == 0.0
        assert d["supervised"] == 0.15
        assert d["maest_logits"] > 0 and d["maest"] > 0
    # The control (budget-100) stays a no-explicit-mix recipe.
    assert le.RECIPES_BY_ID["tsne_budget100"].source_weights is None
    # The two mixes live only in the registry, never the default eight.
    assert "tsne_rhythm_forward" not in {r.id for r in le.RECIPES}


def test_validate_recipe_sources_for_rejects_missing_positive_source():
    # --sources maest gathers only maest; tsne_rhythm_forward declares a positive
    # weight on rhythm/maest_logits/clap/supervised the run cannot supply. It must
    # be rejected, never silently fit on the subset while recording the full mix.
    chosen = le.resolve_recipe_set(
        ["tsne_budget100", "tsne_rhythm_forward"], "tsne_budget100"
    )
    with pytest.raises(ValueError, match="absent from the run sources") as e:
        le.validate_recipe_sources_for(chosen, ("maest",))
    msg = str(e.value)
    assert "tsne_rhythm_forward" in msg
    for src in ("rhythm", "maest_logits", "clap", "supervised"):
        assert src in msg
    # The no-mix control and a mix whose every positive source is present pass.
    le.validate_recipe_sources_for(chosen, FROZEN_SOURCES)


def test_validate_recipe_sources_for_allows_subset_missing_only_zero_weight():
    # Dropping a zero-weight source (features) from the run sources cannot change
    # the composite matrix, so a mix recipe is still valid when every positive
    # source is present.
    sources = tuple(s for s in FROZEN_SOURCES if s != "features")
    for rid in ("tsne_rhythm_forward", "tsne_texture_style"):
        chosen = le.resolve_recipe_set([rid], "tsne_budget100")
        le.validate_recipe_sources_for(chosen, sources)
        # A positive source dropped is exactly the rejected case.
        dropped_positive = tuple(s for s in sources if s != "rhythm")
        with pytest.raises(ValueError, match="absent from the run sources"):
            le.validate_recipe_sources_for(chosen, dropped_positive)


def test_mix_recipe_rejected_before_any_artifact_write(tmp_path):
    # A source-mix recipe selected under a --sources subset that omits a
    # positively-weighted source raises before any manifest/artifact is written:
    # no lying map, no partial generation to confuse a reader.
    fc, infos = _build_cache(tmp_path, 8)
    out = tmp_path / "mixes"
    with pytest.raises(ValueError, match="absent from the run sources"):
        le.run_generation(
            feature_cache=fc, infos=infos, output_dir=str(out),
            sources=("maest",),
            source_weights={"maest": 1.0},
            required_sources=("maest",),
            feature_mask=FROZEN_FEATURE_MASK.copy(),
            features_blend=0.42, seed=42, n_jobs=2,
            recipe_ids=["tsne_rhythm_forward"],
            reference_id="tsne_budget100",
        )
    assert not (out / "manifest.json").exists()


def test_source_weights_absent_from_identity_when_none():
    # The new field defaults to None and must not change the identity of recipes
    # that don't set it (the pinned value lives in
    # test_recipe_identity_preserved_for_original_recipes).
    r = le.RECIPES_BY_ID["baseline"]
    assert "source_weights" not in le.recipe_params(r)
    # A mix recipe records its explicit weights and gets a distinct revision.
    mix = le.RECIPES_BY_ID["tsne_rhythm_forward"]
    assert "source_weights" in le.recipe_params(mix)
    assert le.recipe_revision("deadbeef", mix) != le.recipe_revision("deadbeef", r)


def test_mixes_gallery_three_maps_same_universe(tmp_path):
    pytest.importorskip("umap")
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "mixes"
    mixes = ["tsne_budget100", "tsne_rhythm_forward", "tsne_texture_style"]
    manifest = _run_mixes(fc, infos, out)
    assert manifest["status"] == "complete"
    assert sorted(manifest["recipes"]) == sorted(mixes)
    assert all(r["status"] == "complete" for r in manifest["recipes"].values())
    # Control is the alignment reference; the two variants align onto it.
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] is None
    for rid in ("tsne_rhythm_forward", "tsne_texture_style"):
        assert manifest["recipes"][rid]["aligned_to"] == "tsne_budget100"
    # Same path set in every artifact, all finite.
    paths = {p["path"] for p in le.load_recipe_json(out, "tsne_budget100")["positions"]}
    fps = set()
    for rid in mixes:
        art = le.load_recipe_json(out, rid)
        assert {p["path"] for p in art["positions"]} == paths
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])
        entry = manifest["recipes"][rid]
        assert "input_fingerprint" in entry and "source_weights" in entry
        assert set(art["source_weights"]) == set(FROZEN_SOURCES)
        fps.add(art["input_fingerprint"])
    # Control's actual input fingerprint is the gallery dataset fingerprint; each
    # variant records a distinct one (its own composite matrix).
    ctrl = le.load_recipe_json(out, "tsne_budget100")
    assert ctrl["input_fingerprint"] == manifest["dataset_fingerprint"]
    assert ctrl["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert len(fps) == 3


def test_mixes_control_revision_matches_budget100(tmp_path):
    # The control is identity-identical to the budget/scale tsne_budget100 map:
    # same recipe (no explicit mix), same gallery dataset fingerprint, so its
    # revision must match the standalone budget-100 run.
    fc, infos = _build_cache(tmp_path, 12)
    budget_out = tmp_path / "budget"
    _run(fc, infos, budget_out, recipe_ids=["tsne_budget100"])
    mixes_out = tmp_path / "mixes"
    _run_mixes(fc, infos, mixes_out)
    budget_art = le.load_recipe_json(budget_out, "tsne_budget100")
    mix_art = le.load_recipe_json(mixes_out, "tsne_budget100")
    assert mix_art["revision"] == budget_art["revision"]
    assert mix_art["input_fingerprint"] == budget_art["input_fingerprint"]


# ── ablation recipes: dropped-block source mixes ─────────────────────────────


def _build_cache_with_supervised(tmp_path, n, seed=0):
    """Like ``_build_cache`` but also caches the ``supervised`` block (run id 1)."""
    fc, infos = _build_cache(tmp_path, n, seed=seed)
    fc.set_meta("supervised_run_id", 1)
    rng = np.random.RandomState(seed)
    for fp in [t["fingerprint"] for t in infos]:
        fc.put("supervised", fp, rng.standard_normal(128).astype(np.float32), version=1)
    return fc, infos


ABLATION_IDS = ["tsne_budget100", "tsne_maest_supervised", "tsne_maest_only"]


def _run_ablation(fc, infos, out, *, n_jobs=2):
    return le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=str(out),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=n_jobs,
        recipe_ids=ABLATION_IDS,
        reference_id="tsne_budget100",
    )


def test_ablation_recipes_are_complete_vectors_with_honest_sums():
    # Both are complete vectors over FROZEN_SOURCES with features at 0, the
    # required pair positive, and rhythm/clap (and supervised for "only") zeroed.
    # The retained weights (.45/.25 and the supervised .15) are preserved exactly,
    # never renormalized: the sums are the honest .85 and .70, not 1.10.
    for rid, expected_sum, expected_sup in (
        ("tsne_maest_supervised", 0.85, 0.15),
        ("tsne_maest_only", 0.70, 0.0),
    ):
        r = le.RECIPES_BY_ID[rid]
        assert r.method == "tsne" and r.pca_dim == 50
        assert r.tsne_n_iter == 100 and r.tsne_early_exaggeration_iter == 250
        assert r.perplexities is None
        d = dict(r.source_weights)
        assert list(d) == list(FROZEN_SOURCES)
        assert abs(sum(d.values()) - expected_sum) < 1e-9
        assert d["features"] == 0.0
        assert d["supervised"] == expected_sup
        assert d["rhythm"] == 0.0 and d["clap"] == 0.0
        assert d["maest_logits"] == 0.45 and d["maest"] == 0.25
    # The control stays a no-explicit-mix recipe.
    assert le.RECIPES_BY_ID["tsne_budget100"].source_weights is None
    # Both live only in the registry, never the default eight.
    assert set(ABLATION_IDS[1:]) <= set(le.RECIPES_BY_ID)
    assert not ({r.id for r in le.RECIPES} & set(ABLATION_IDS[1:]))


def test_ablation_keeps_original_identities_and_default_eight():
    assert len(le.RECIPES) == 8
    assert set(r.id for r in le.RECIPES) < set(le.RECIPES_BY_ID)
    # Each mix/ablation recipe gets its own distinct revision and records weights
    # (the original recipes' pinned identities live in
    # test_recipe_identity_preserved_for_original_recipes).
    revs = {
        rid: le.recipe_revision("deadbeef", le.RECIPES_BY_ID[rid])
        for rid in ("tsne_rhythm_forward", "tsne_texture_style",
                    "tsne_maest_supervised", "tsne_maest_only")
    }
    assert len(set(revs.values())) == len(revs)
    for rid in ABLATION_IDS[1:]:
        assert "source_weights" in le.recipe_params(le.RECIPES_BY_ID[rid])


def test_ablation_zero_weight_blocks_dropped_and_retained_preserved(tmp_path):
    # A zero-weight source drops its columns from the composite matrix (the shared
    # layout helper skips weight<=0 blocks), and the retained blocks keep their
    # exact sqrt(w) scaling — the variants are intentionally narrower, never
    # renormalized to the control's total.
    fc, infos = _build_cache_with_supervised(tmp_path, 8)
    ctrl = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
    )
    sup = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=dict(le.RECIPES_BY_ID["tsne_maest_supervised"].source_weights),
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
    )
    only = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=dict(le.RECIPES_BY_ID["tsne_maest_only"].source_weights),
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
    )
    # Same path universe in every variant (required sources unchanged).
    assert ctrl.paths == sup.paths == only.paths
    # Column layout: maest_logits 519 | maest 768 | rhythm 14 | clap 512 |
    # supervised 128. Control has all five (features 0).
    assert ctrl.mat.shape[1] == 519 + 768 + 14 + 512 + 128  # 1941
    # Ablation narrows the matrix by dropping rhythm/clap (+supervised for "only").
    assert sup.mat.shape[1] == 519 + 768 + 128  # 1415
    assert only.mat.shape[1] == 519 + 768       # 1287
    # Retained blocks carry their exact scaling (identical to the control's), so
    # the shared columns are byte-identical.
    np.testing.assert_array_equal(only.mat[:, :519], ctrl.mat[:, :519])
    np.testing.assert_array_equal(only.mat[:, 519:1287], ctrl.mat[:, 519:1287])
    np.testing.assert_array_equal(sup.mat[:, :519], ctrl.mat[:, :519])
    np.testing.assert_array_equal(sup.mat[:, 519:1287], ctrl.mat[:, 519:1287])
    # The supervised block (cols 1287:1415) matches the control's supervised block
    # (cols 1813:1941, past rhythm+clap), both scaled sqrt(.15).
    np.testing.assert_array_equal(sup.mat[:, 1287:], ctrl.mat[:, 1813:])


def test_ablation_missing_optional_supervised_stays_eligible(tmp_path):
    # ``supervised`` is optional (required = maest_logits, maest). A library with
    # no cached supervised rows still places every track; the optional block is
    # simply dropped, narrowing the matrix to the two required blocks.
    fc, infos = _build_cache(tmp_path, 6)
    fc.set_meta("supervised_run_id", 1)
    sup = prepare_projection_input(
        infos, fc, sources=FROZEN_SOURCES,
        source_weights=dict(le.RECIPES_BY_ID["tsne_maest_supervised"].source_weights),
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
    )
    assert len(sup.paths) == 6
    assert sup.mat.shape[1] == 519 + 768


def test_ablation_gallery_three_maps_same_universe(tmp_path):
    fc, infos = _build_cache_with_supervised(tmp_path, 12)
    out = tmp_path / "ablation"
    manifest = _run_ablation(fc, infos, out)
    assert manifest["status"] == "complete"
    assert sorted(manifest["recipes"]) == sorted(ABLATION_IDS)
    assert all(r["status"] == "complete" for r in manifest["recipes"].values())
    assert manifest["reference_id"] == "tsne_budget100"
    # Control is the alignment reference; the two ablation variants align onto it.
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] is None
    for rid in ABLATION_IDS[1:]:
        assert manifest["recipes"][rid]["aligned_to"] == "tsne_budget100"
    # Same path set in every artifact, all finite.
    paths = {p["path"] for p in le.load_recipe_json(out, "tsne_budget100")["positions"]}
    fps = set()
    for rid in ABLATION_IDS:
        art = le.load_recipe_json(out, rid)
        assert {p["path"] for p in art["positions"]} == paths
        assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in art["positions"])
        assert set(art["source_weights"]) == set(FROZEN_SOURCES)
        expected = (
            FROZEN_SOURCE_WEIGHTS
            if rid == "tsne_budget100"
            else dict(le.RECIPES_BY_ID[rid].source_weights)
        )
        assert art["source_weights"] == expected
        fps.add(art["input_fingerprint"])
    # Control's actual input fingerprint is the gallery dataset fingerprint; each
    # variant records a distinct one (its own weight vector and matrix width).
    ctrl = le.load_recipe_json(out, "tsne_budget100")
    assert ctrl["input_fingerprint"] == manifest["dataset_fingerprint"]
    assert len(fps) == 3


def test_get_set_ablation_routes_three_maps_isolation(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)  # root default eight
    _run_ablation(fc, infos, out / "ablation")
    client = _make_app(out).test_client()

    r = client.get("/api/layout-evaluation?set=ablation")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(ABLATION_IDS)
    # A variant resolves within the ablation set and serves its own positions.
    r = client.get("/api/layout-evaluation/tsne_maest_only?set=ablation")
    assert r.status_code == 200
    assert r.get_json()["id"] == "tsne_maest_only"
    assert len(r.get_json()["positions"]) == 10
    # Ablation-only recipes are not present in the root set (cross-set isolation).
    assert client.get("/api/layout-evaluation/tsne_maest_supervised").status_code == 404
    assert client.get("/api/layout-evaluation/tsne_maest_only?set=budget").status_code == 404
    # The allowlist still rejects unknown sets.
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400


def test_variant_path_mismatch_rejected(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    fc, infos = _build_cache(tmp_path, 8)
    out = tmp_path / "mixes"
    real = prepare_projection_input
    from backend.embeddings.layout import ProjectionInput

    def _fake(track_infos, feature_cache, **kw):
        inp = real(track_infos, feature_cache, **kw)
        # Only a variant's gather (a non-default explicit mix) diverges from the
        # control's track universe; the control gather keeps the default weights.
        if kw.get("source_weights") and kw["source_weights"] != FROZEN_SOURCE_WEIGHTS:
            # A variant whose track universe diverges from the control's.
            extra_paths = list(inp.paths) + ["lib/extra.mp3"]
            extra_mat = np.vstack([inp.mat, inp.mat[0:1]])
            return ProjectionInput(
                extra_paths, extra_mat, inp.clap_dim, inp.clap_col_lo,
                inp.comp_key, inp.comp_config, inp.spans, extra_mat, inp.tsne_config,
            )
        return inp

    monkeypatch.setattr("backend.embeddings.layout.prepare_projection_input", _fake)
    manifest = _run_mixes(fc, infos, out)
    # Control fits normally; each variant is rejected with a clear status, never
    # silently sliced or faked to the control's center.
    assert manifest["recipes"]["tsne_budget100"]["status"] == "complete"
    for rid in ("tsne_rhythm_forward", "tsne_texture_style"):
        assert manifest["recipes"][rid]["status"] == "error"
        assert "diverged" in manifest["recipes"][rid]["reason"]
    assert le.load_completed(out) == {"tsne_budget100": manifest["recipes"]["tsne_budget100"]}


def test_reference_auto_included_and_must_be_registered(tmp_path):
    fc, infos = _build_cache(tmp_path, 10)
    out = tmp_path / "out"
    # A reference not in the requested set is auto-included (and fitted first).
    manifest = le.run_generation(
        feature_cache=fc,
        infos=infos,
        output_dir=str(out),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=2,
        recipe_ids=["tsne_p60"],
        reference_id="tsne_budget100",
    )
    assert sorted(manifest["recipes"]) == ["tsne_budget100", "tsne_p60"]
    assert manifest["recipes"]["tsne_p60"]["aligned_to"] == "tsne_budget100"
    assert manifest["recipes"]["tsne_budget100"]["aligned_to"] is None
    # An unknown reference id is rejected up front.
    with pytest.raises(ValueError, match="reference"):
        le.run_generation(
            feature_cache=fc,
            infos=infos,
            output_dir=str(tmp_path / "out2"),
            sources=FROZEN_SOURCES,
            source_weights=FROZEN_SOURCE_WEIGHTS,
            required_sources=REQUIRED_SOURCES,
            feature_mask=FROZEN_FEATURE_MASK.copy(),
            features_blend=0.42,
            seed=42,
            n_jobs=2,
            recipe_ids=["tsne_p60"],
            reference_id="nope",
        )


def test_generation_does_not_touch_production_fit_cache(tmp_path):
    pytest.importorskip("umap")
    pb.reset_tsne_fit_cache()
    fc, infos = _build_cache(tmp_path, 10)
    _run(fc, infos, tmp_path / "out")
    # The eval fits its own openTSNE models; the production incremental-fit cache
    # must not gain entries, and nothing is persisted under it.
    assert pb.tsne_fit_stats()["fit"] == 0
    assert pb.tsne_fit_stats()["transform"] == 0
    assert len(pb._tsne_fit_cache) == 0


def test_too_few_tracks_skips_with_clear_reason(tmp_path):
    fc, infos = _build_cache(tmp_path, 3)
    manifest = _run(fc, infos, tmp_path / "out")
    # No requested recipe completed, so the generation is failed, not complete.
    assert manifest["status"] == "failed"
    assert all(
        r["status"] == "skipped" and "too few tracks" in r["reason"]
        for r in manifest["recipes"].values()
    )
    # No artifacts are written for a skipped run.
    assert le.load_completed(tmp_path / "out") == {}


def test_initial_manifest_written_before_gather(tmp_path, monkeypatch):
    fc, infos = _build_cache(tmp_path, 8)
    out = tmp_path / "out"

    def boom(*_args, **_kwargs):
        raise RuntimeError("gather failed")

    monkeypatch.setattr(
        "backend.embeddings.layout.prepare_projection_input", boom
    )
    with pytest.raises(RuntimeError, match="gather failed"):
        _run(fc, infos, out)
    manifest = le.manifest_dict(out)
    assert manifest["status"] == "in_progress"
    assert manifest["recipes"] == {}
    assert manifest["dataset_fingerprint"] == ""
    assert manifest["n_jobs"] == 2


def test_stale_artifacts_never_appear_valid(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # Dataset A: the shared eight-recipe generation (copied above).
    manifest_a = le.manifest_dict(out)
    # Dataset B: a changed library, only the baseline regenerated.
    fc_b, infos_b = _build_cache(tmp_path, 11, seed=7)
    manifest_b = _run(fc_b, infos_b, out, recipe_ids=["baseline"])
    assert manifest_a["dataset_fingerprint"] != manifest_b["dataset_fingerprint"]
    assert sorted(le.load_completed(out)) == ["baseline"]
    # A stale pca2.json from dataset A remains on disk but is not served.
    stale_path = out / "pca2.json"
    assert stale_path.exists()
    assert le.load_recipe_json(out, "pca2") is None


def test_partial_generation_records_errors(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "out"
    _real = le.fit_recipe

    def _fail(mat, recipe, *, seed, n_jobs):
        if recipe.id == "pca25":
            raise RuntimeError("boom")
        return _real(mat, recipe, seed=seed, n_jobs=n_jobs)

    monkeypatch.setattr(le, "fit_recipe", _fail)
    manifest = _run(fc, infos, out)
    assert manifest["status"] == "partial"
    assert manifest["recipes"]["pca25"]["status"] == "error"
    assert "RuntimeError: boom" in manifest["recipes"]["pca25"]["reason"]
    assert all(
        r["status"] == "complete" for rid, r in manifest["recipes"].items() if rid != "pca25"
    )
    assert "pca25" not in le.load_completed(out)


def test_limit_caps_tracks_deterministically(tmp_path):
    pytest.importorskip("umap")
    fc, infos = _build_cache(tmp_path, 12)
    out = tmp_path / "out"
    manifest = _run(fc, infos, out, limit=8)
    assert manifest["dataset"]["count"] == 8


# ── GET routes ───────────────────────────────────────────────────────────────


def _make_app(layout_dir):
    from flask import Flask

    from backend.factory import _register_error_handlers
    from backend.routes.api_layout_evaluation import create_api_layout_evaluation_blueprint

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(create_api_layout_evaluation_blueprint(str(layout_dir)))
    return app


def test_get_index_and_recipe_after_generation(gallery):
    _fc, _infos, out = gallery
    client = _make_app(out).test_client()

    r = client.get("/api/layout-evaluation")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert len(r.get_json()["manifest"]["recipes"]) == 8

    r = client.get("/api/layout-evaluation/baseline")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["positions"]) == 10


def test_get_no_generation_is_404(tmp_path):
    client = _make_app(tmp_path / "empty").test_client()
    r = client.get("/api/layout-evaluation")
    assert r.status_code == 404
    assert r.get_json() == {"ok": False, "error": "no layout-evaluation generation found"}


def test_get_traversal_and_bad_id_rejected(gallery):
    _fc, _infos, out = gallery
    client = _make_app(out).test_client()
    for bad in ["../secret", "baseline/../../etc", "a/b", "..%2Fsecret"]:
        r = client.get(f"/api/layout-evaluation/{bad}")
        assert r.status_code == 400, bad
        assert r.get_json()["ok"] is False


def test_get_unknown_recipe_is_404(gallery):
    _fc, _infos, out = gallery
    client = _make_app(out).test_client()
    r = client.get("/api/layout-evaluation/nope")
    assert r.status_code == 404


def test_get_set_routes_to_named_subdir(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # Root gallery: the default eight (copied).
    # Named "budget" gallery under out/budget: only the four budget recipes.
    budget = ["baseline", "tsne_budget100", "tsne_budget150", "tsne_budget"]
    _run(fc, infos, out / "budget", recipe_ids=budget)
    client = _make_app(out).test_client()

    # No set → root (eight).
    r = client.get("/api/layout-evaluation")
    assert r.status_code == 200
    assert len(r.get_json()["manifest"]["recipes"]) == 8
    # set=budget → the budget subdir (four).
    r = client.get("/api/layout-evaluation?set=budget")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(budget)
    # set=base is the allowlisted explicit alias for the root dir.
    r = client.get("/api/layout-evaluation?set=base")
    assert r.status_code == 200
    assert len(r.get_json()["manifest"]["recipes"]) == 8
    # A recipe within a named set resolves to that set's own artifact.
    r = client.get("/api/layout-evaluation/tsne_budget100?set=budget")
    assert r.status_code == 200
    assert r.get_json()["id"] == "tsne_budget100"
    assert len(r.get_json()["positions"]) == 10
    # A budget-only recipe is not present in the root set.
    r = client.get("/api/layout-evaluation/tsne_budget100")
    assert r.status_code == 404


def test_get_set_scale_and_cross_set_isolation(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # Root gallery: the default eight (copied).
    # Named "budget" gallery: the four budget recipes.
    budget = ["baseline", "tsne_budget100", "tsne_budget150", "tsne_budget"]
    _run(fc, infos, out / "budget", recipe_ids=budget)
    # Named "scale" gallery: the four scale recipes.
    _run_scale(fc, infos, out / "scale")
    scale = ["tsne_budget100", "tsne_p60", "tsne_p120", "tsne_multiscale30_120"]
    client = _make_app(out).test_client()

    # set=scale serves exactly the four scale maps.
    r = client.get("/api/layout-evaluation?set=scale")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(scale)
    # The control resolves within the scale set.
    r = client.get("/api/layout-evaluation/tsne_budget100?set=scale")
    assert r.status_code == 200
    assert r.get_json()["id"] == "tsne_budget100"
    # A scale-only recipe is not present in the root or budget sets.
    assert client.get("/api/layout-evaluation/tsne_p60").status_code == 404
    assert client.get("/api/layout-evaluation/tsne_p60?set=budget").status_code == 404
    # A budget-only recipe is not present in the scale set (cross-set isolation).
    assert client.get("/api/layout-evaluation/tsne_budget150?set=scale").status_code == 404
    # The allowlist still rejects unknown sets.
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400


def test_get_set_early_routes_three_maps_isolation(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # Root gallery: the default eight (copied).
    # Named "early" gallery: the budget-100 control + the two early variants.
    early = ["tsne_budget100", "tsne_early150", "tsne_early100"]
    le.run_generation(
        feature_cache=fc, infos=infos, output_dir=str(out / "early"),
        sources=FROZEN_SOURCES, source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES, feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42, seed=42, n_jobs=2, recipe_ids=early,
        reference_id="tsne_budget100",
    )
    client = _make_app(out).test_client()

    # set=early serves exactly the three early maps.
    r = client.get("/api/layout-evaluation?set=early")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(early)
    # The control resolves within the early set.
    r = client.get("/api/layout-evaluation/tsne_budget100?set=early")
    assert r.status_code == 200
    assert r.get_json()["id"] == "tsne_budget100"
    assert len(r.get_json()["positions"]) == 10
    # An early-only recipe is not present in the root or budget sets
    # (cross-set isolation).
    assert client.get("/api/layout-evaluation/tsne_early150").status_code == 404
    assert client.get("/api/layout-evaluation/tsne_early150?set=budget").status_code == 404
    # The allowlist still rejects unknown sets.
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400


def test_get_set_mixes_routes_three_maps(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    mixes = ["tsne_budget100", "tsne_rhythm_forward", "tsne_texture_style"]
    le.run_generation(
        feature_cache=fc, infos=infos, output_dir=str(out / "mixes"),
        sources=FROZEN_SOURCES, source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES, feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42, seed=42, n_jobs=2, recipe_ids=mixes,
        reference_id="tsne_budget100",
    )
    client = _make_app(out).test_client()

    r = client.get("/api/layout-evaluation?set=mixes")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(mixes)
    # A mix variant resolves within the mixes set and serves its own positions.
    r = client.get("/api/layout-evaluation/tsne_rhythm_forward?set=mixes")
    assert r.status_code == 200
    assert r.get_json()["id"] == "tsne_rhythm_forward"
    assert len(r.get_json()["positions"]) == 10
    # A mix-only recipe is not present in the root set (cross-set isolation).
    assert client.get("/api/layout-evaluation/tsne_rhythm_forward").status_code == 404
    # An ungenerated named set has no manifest.
    assert client.get("/api/layout-evaluation?set=scale").status_code == 404


def test_get_invalid_or_traversal_set_rejected(gallery):
    _fc, _infos, out = gallery
    client = _make_app(out).test_client()
    for bad in ["../etc", "..", "budget/../../etc", "not-a-set", ""]:
        r = client.get(f"/api/layout-evaluation?set={bad}")
        assert r.status_code == 400, bad
        assert r.get_json()["ok"] is False


def test_get_named_set_with_no_generation_is_404(tmp_path):
    client = _make_app(tmp_path / "empty").test_client()
    r = client.get("/api/layout-evaluation?set=budget")
    assert r.status_code == 404
    assert r.get_json() == {"ok": False, "error": "no layout-evaluation generation found"}


def test_get_corrupt_artifact_is_500(gallery, tmp_path):
    _fc, _infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # Corrupt the baseline artifact file after generation.
    (out / "baseline.json").write_text("not json")
    client = _make_app(out).test_client()
    r = client.get("/api/layout-evaluation/baseline")
    assert r.status_code == 500
    assert "artifact is missing or invalid" in r.get_json()["error"]


def test_get_nonfinite_coords_rejected(gallery, tmp_path):
    _fc, _infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    art = json.loads((out / "pca2.json").read_text())
    art["positions"][0]["x"] = float("nan")
    (out / "pca2.json").write_text(json.dumps(art))
    client = _make_app(out).test_client()
    r = client.get("/api/layout-evaluation/pca2")
    assert r.status_code == 500


def test_recipe_artifact_fingerprint_must_match_manifest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # Manifest claims a completed recipe for dataset fingerprint "manifest-fp".
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset_fingerprint": "manifest-fp",
        "recipes": {"baseline": {"status": "complete"}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    # The artifact on disk carries a different (stale) dataset fingerprint.
    stale = {
        "id": "baseline",
        "label": "x",
        "dataset_fingerprint": "stale-fp",
        "positions": [{"path": "a.mp3", "x": 0.1, "y": 0.2}],
    }
    (out / "baseline.json").write_text(json.dumps(stale))
    assert le.load_recipe_json(out, "baseline") is None


def test_variant_input_fingerprint_must_match_manifest(tmp_path):
    # A source-mix artifact that matches the gallery dataset fingerprint and the
    # manifest revision but carries a stale *actual* input fingerprint is rejected.
    out = tmp_path / "out"
    out.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset_fingerprint": "gallery-fp",
        "recipes": {
            "tsne_rhythm_forward": {
                "status": "complete",
                "revision": "rev",
                "input_fingerprint": "actual-fp",
            }
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    good = {
        "id": "tsne_rhythm_forward",
        "label": "x",
        "dataset_fingerprint": "gallery-fp",
        "revision": "rev",
        "input_fingerprint": "actual-fp",
        "positions": [{"path": "a.mp3", "x": 0.1, "y": 0.2}],
    }
    (out / "tsne_rhythm_forward.json").write_text(json.dumps(good))
    assert le.load_recipe_json(out, "tsne_rhythm_forward") == good
    # Same gallery fp + revision, but a different actual input fingerprint.
    stale = dict(good)
    stale["input_fingerprint"] = "other-fp"
    (out / "tsne_rhythm_forward.json").write_text(json.dumps(stale))
    assert le.load_recipe_json(out, "tsne_rhythm_forward") is None


def test_old_artifact_without_input_fingerprint_still_valid(tmp_path):
    # A manifest/artifact pair from before input_fingerprint existed (neither
    # records it) must still pass via the older gallery-fp + revision checks — the
    # new check is additive, never a relaxed path for stale artifacts.
    out = tmp_path / "out"
    out.mkdir()
    rev = le.recipe_revision("fp", le.RECIPES_BY_ID["baseline"], seed=42, versions={"clap": 1})
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset_fingerprint": "fp",
        "recipes": {"baseline": {"status": "complete", "revision": rev}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    old = {
        "id": "baseline",
        "label": "x",
        "dataset_fingerprint": "fp",
        "revision": rev,
        "positions": [{"path": "a.mp3", "x": 0.1, "y": 0.2}],
    }
    (out / "baseline.json").write_text(json.dumps(old))
    assert le.load_recipe_json(out, "baseline") == old


def test_partial_manifest_serves_current_completed_recipe(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rev = le.recipe_revision("fp", le.RECIPES_BY_ID["baseline"], seed=42, versions={"clap": 1})
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "dataset_fingerprint": "fp",
        "recipes": {
            "baseline": {"status": "complete", "revision": rev},
            "pca2": {"status": "pending"},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    good = {
        "id": "baseline",
        "label": "x",
        "dataset_fingerprint": "fp",
        "revision": rev,
        "positions": [{"path": "a.mp3", "x": 0.1, "y": 0.2}],
    }
    (out / "baseline.json").write_text(json.dumps(good))
    # A completed recipe of the current (in-progress) run is served.
    assert le.load_completed(out) == {"baseline": manifest["recipes"]["baseline"]}
    assert le.load_recipe_json(out, "baseline") == good
    # An unfinished recipe is never served.
    assert le.load_recipe_json(out, "pca2") is None


def test_partial_manifest_rejects_stale_revision(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rev = le.recipe_revision("fp", le.RECIPES_BY_ID["baseline"], seed=42, versions={"clap": 1})
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "dataset_fingerprint": "fp",
        "recipes": {"baseline": {"status": "complete", "revision": rev}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    stale = {
        "id": "baseline",
        "label": "x",
        "dataset_fingerprint": "fp",
        "revision": "stale-revision",
        "positions": [{"path": "a.mp3", "x": 0.1, "y": 0.2}],
    }
    (out / "baseline.json").write_text(json.dumps(stale))
    # Same dataset fingerprint but a different revision (e.g. changed environment)
    # is a stale artifact and must never be served from a partial manifest.
    assert le.load_recipe_json(out, "baseline") is None


def test_partial_manifest_with_no_completed_recipes_is_empty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "in_progress",
        "dataset_fingerprint": "fp",
        "recipes": {"baseline": {"status": "pending"}},
    }))
    assert le.load_completed(out) == {}
    assert le.load_recipe_json(out, "baseline") is None


def test_get_stale_recipe_not_served(gallery, tmp_path):
    fc, infos, gallery_out = gallery
    out = tmp_path / "out"
    shutil.copytree(gallery_out, out)
    # A stale recipe file from an older dataset with no manifest entry.
    (out / "pca2.json").unlink()
    # Re-run with only baseline so pca2 is not in the manifest.
    _run(fc, infos, out, recipe_ids=["baseline"])
    client = _make_app(out).test_client()
    r = client.get("/api/layout-evaluation/pca2")
    assert r.status_code == 404


# ── missing dependencies surface as clean per-recipe errors ──────────────────


def test_missing_dependency_is_clean_per_recipe_error(tmp_path, monkeypatch):
    # A recipe whose fit needs an uninstalled library raises MissingDependencyError
    # with a clear message; the per-recipe handler records it as a clean error
    # status (never a traceback, never an abort of the rest of the run).
    fc, infos = _build_cache(tmp_path, 10)
    out = tmp_path / "out"

    def _missing(*a, **k):
        raise le.MissingDependencyError(
            "umap-learn is required for the layout-evaluation UMAP recipes; "
            "install it (pip install umap-learn)."
        )

    monkeypatch.setattr(le, "fit_umap", _missing)
    manifest = _run(fc, infos, out, recipe_ids=["umap2_n15"])
    assert manifest["recipes"]["umap2_n15"]["status"] == "error"
    assert "pip install umap-learn" in manifest["recipes"]["umap2_n15"]["reason"]
    # The auto-included reference still fits; only the UMAP recipe failed.
    assert manifest["recipes"]["baseline"]["status"] == "complete"
    assert le.load_completed(out) == {"baseline": manifest["recipes"]["baseline"]}


# ── precompute CLI: clean early exits before any artifact write ──────────────


def _cli_main(monkeypatch, tmp_path, *argv):
    """Run the precompute CLI with fake cached infos (only failure paths hit)."""
    import backend.precompute_layout_evaluation as cli

    (tmp_path / "music").mkdir()
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        cli,
        "load_infos",
        lambda *a, **k: ([{"path": "lib/a.mp3", "fingerprint": "fp0"}], 1),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["precompute_layout_evaluation", "--root", str(tmp_path / "music"),
         "--data-dir", str(tmp_path / "data"), *argv],
    )
    return cli.main()


def test_cli_unknown_recipe_clean_exit(tmp_path, monkeypatch):
    # An unknown --recipes id is rejected by run_generation before any artifact
    # write; the CLI surfaces it as a clean SystemExit (exit 1, no traceback).
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _cli_main(monkeypatch, tmp_path, "--output", str(out), "--recipes", "nope")
    assert "unknown recipe id(s): nope" in str(ei.value)
    assert not (out / "manifest.json").exists()


def test_cli_unknown_reference_clean_exit(tmp_path, monkeypatch):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _cli_main(monkeypatch, tmp_path, "--output", str(out), "--reference", "nope")
    assert "unknown reference recipe id" in str(ei.value)
    assert not (out / "manifest.json").exists()


def test_cli_mix_with_sources_subset_rejected_clean_exit(tmp_path, monkeypatch):
    # A mix recipe selected under a --sources subset that omits a positively
    # weighted source is rejected before any artifact write.
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as ei:
        _cli_main(
            monkeypatch, tmp_path, "--output", str(out),
            "--sources", "maest", "--recipes", "tsne_rhythm_forward",
            "--reference", "tsne_budget100",
        )
    assert "absent from the run sources" in str(ei.value)
    assert not (out / "manifest.json").exists()


def test_cli_set_selects_recipes_reference_and_default_output(tmp_path, monkeypatch):
    import backend.precompute_layout_evaluation as cli

    captured = {}

    def _fake_run_generation(**kwargs):
        captured.update(kwargs)
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        return {
            "dataset_fingerprint": "fp",
            "sources": [],
            "source_weights": {},
            "dataset": {"count": 1},
            "recipes": {},
        }

    monkeypatch.setattr(cli, "run_generation", _fake_run_generation)
    _cli_main(monkeypatch, tmp_path, "--set", "hierarchy-heuristic")

    gallery_set = le.GALLERY_SETS["hierarchy-heuristic"]
    assert captured["recipe_ids"] == [r.id for r in gallery_set.recipes]
    assert captured["reference_id"] == gallery_set.recipes[0].id
    assert captured["output_dir"] == str(
        tmp_path / "data" / "layout-evaluation" / "hierarchy-heuristic"
    )


def test_cli_external_set_rejected_clean_exit(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as ei:
        _cli_main(monkeypatch, tmp_path, "--set", "genre")
    assert "externally generated" in str(ei.value)
    assert not (tmp_path / "data" / "layout-evaluation" / "genre").exists()
