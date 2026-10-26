"""Generic multilevel t-SNE core: expansion, level loop, fraction schedule."""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import layout_evaluation as le
from backend.embeddings import multilevel_tsne as ml


def _fake_adapter(calls: list, dims: int = 2):
    def fit_level(subset, normal, early, perplexity, init):
        calls.append(("level", len(subset), normal, early, perplexity, init))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.random.default_rng(len(subset)).standard_normal(
            (len(subset), dims)
        ).astype(np.float32)

    def fit_final(full, init):
        calls.append(("final", len(full), init))
        if init is not None:
            return np.asarray(init, dtype=np.float32)
        return np.zeros((len(full), dims), dtype=np.float32)

    return fit_level, fit_final


def test_empty_schedule_fits_only_final_and_returns_object():
    reduced = np.zeros((10, 4))
    calls: list = []
    fit_level, fit_final = _fake_adapter(calls)
    marker = np.arange(20, dtype=np.float32).reshape(10, 2)

    final, coords, stage_t = ml.fit_multilevel(
        reduced, [], fit_level=fit_level, fit_final=lambda full, init: marker,
        seed=42, n_jobs=1,
    )
    assert final is marker
    np.testing.assert_array_equal(coords, marker)
    assert coords.dtype == np.float32
    assert calls == []
    assert stage_t["tsne_level0"] == 0.0
    assert stage_t["tsne_expand"] == 0.0
    assert stage_t["tsne_final"] >= 0.0


def test_schedule_loop_forwards_stage_params_and_fresh_init():
    n = 60
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float64)
    schedule = [(10, 7, 5, 30.0), (30, 6, 0, None)]
    calls: list = []
    fit_level, fit_final = _fake_adapter(calls)

    _final, coords, stage_t = ml.fit_multilevel(
        reduced, schedule, fit_level=fit_level, fit_final=fit_final,
        seed=7, n_jobs=2,
    )
    assert coords.shape == (n, 2)
    assert [c[0:2] for c in calls] == [
        ("level", 10), ("level", 30), ("final", 60),
    ]
    assert calls[0][2:5] == (7, 5, 30.0)
    assert calls[1][2:5] == (6, 0, None)
    assert calls[0][5] is None  # coarse level starts uninitialised
    assert calls[1][5] is not None  # expansion init, fresh array per stage
    assert calls[2][2] is not None
    assert set(("tsne_level0", "tsne_level1", "tsne_expand", "tsne_final")) <= set(stage_t)


def test_dimension_agnostic_three_d_shape():
    n = 40
    reduced = np.random.RandomState(1).standard_normal((n, 5))
    calls: list = []
    fit_level, fit_final = _fake_adapter(calls, dims=3)
    _final, coords, _t = ml.fit_multilevel(
        reduced, [(8, 5, 5, 30.0), (20, 5, 0, 30.0)],
        fit_level=fit_level, fit_final=fit_final, seed=3, n_jobs=1,
    )
    assert coords.shape == (n, 3)
    idx = np.array([0, 2, 5])
    landmark = np.random.RandomState(2).standard_normal((3, 3)).astype(np.float32)
    expanded = ml.expand_landmark_init(
        reduced, idx, landmark, np.arange(n), seed=3, n_jobs=1
    )
    assert expanded.shape == (n, 3)


def test_production_fraction_preset_shape_and_range():
    from backend.embeddings.projection_config import (
        FROZEN_FOLDER_BASIS_MODE,
        FROZEN_FOLDER_BOOST,
        FROZEN_FOLDER_DEPTH_BOOST,
        OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER,
        OPENTSNE_COARSE_ITERATION_BUDGET,
        OPENTSNE_EARLY_EXAGGERATION_ITER,
        OPENTSNE_FINAL_EXAGGERATION,
        OPENTSNE_MULTILEVEL_FRACTIONS,
        OPENTSNE_N_ITER,
        OPENTSNE_PERPLEXITY_GAMMA,
        OPENTSNE_PERPLEXITY_START_FRACTION,
        PROJECTION_RECIPE_VERSION,
        PROJECTION_RECIPES,
    )

    fr = OPENTSNE_MULTILEVEL_FRACTIONS
    assert len(fr) == 16
    assert fr[0] == 1 / 256
    assert fr[-1] == 1 / 16
    assert all(b > a for a, b in zip(fr, fr[1:]))
    ratio = 16 ** (1 / 15)
    assert all(b / a == pytest.approx(ratio) for a, b in zip(fr, fr[1:]))
    assert OPENTSNE_COARSE_ITERATION_BUDGET == 16
    assert OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER == 100
    assert OPENTSNE_PERPLEXITY_GAMMA == 4.0
    assert OPENTSNE_PERPLEXITY_START_FRACTION == fr[0]
    assert (OPENTSNE_N_ITER, OPENTSNE_EARLY_EXAGGERATION_ITER) == (50, 0)
    assert OPENTSNE_FINAL_EXAGGERATION == pytest.approx(1.25)
    assert PROJECTION_RECIPE_VERSION == 10
    assert (FROZEN_FOLDER_BASIS_MODE, FROZEN_FOLDER_BOOST,
            FROZEN_FOLDER_DEPTH_BOOST) == ("centroids", 2.4, 3.0)
    for rid in ("default", "genre_full"):
        r = PROJECTION_RECIPES[rid]
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)


def test_eval_fraction_recipes_keep_their_own_frozen_literals():
    """Evaluation recipes must not derive from the mutable production preset."""
    for rid, fractions in (
        ("tsne_fraction_g16", (1 / 256, 1 / 16)),
        ("tsne_fraction_g8", (1 / 64, 1 / 8)),
    ):
        r = le.RECIPES_BY_ID[rid]
        assert r.tsne_sample_fractions == fractions
        assert r.tsne_coarse_iteration_budget is None
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
    density = le.RECIPES_BY_ID["tsne_density16"]
    assert density.tsne_sample_fractions == le._FRACTIONS_DENSITY16
    assert density.tsne_coarse_iteration_budget == 541


def _production_schedule(n: int):
    from backend.embeddings.projection_config import (
        OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER,
        OPENTSNE_COARSE_ITERATION_BUDGET,
        OPENTSNE_MULTILEVEL_FRACTIONS,
        OPENTSNE_PERPLEXITY_GAMMA,
        OPENTSNE_PERPLEXITY_START_FRACTION,
    )

    return ml.resolve_fraction_schedule(
        OPENTSNE_MULTILEVEL_FRACTIONS,
        n,
        normal_budget=OPENTSNE_COARSE_ITERATION_BUDGET,
        gamma=OPENTSNE_PERPLEXITY_GAMMA,
        start_fraction=OPENTSNE_PERPLEXITY_START_FRACTION,
        early_iter=OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER,
    )


def test_production_schedule_uses_all_16_levels_one_iteration_each():
    sched = _production_schedule(7153)
    assert len(sched) == 16
    counts = [s[0] for s in sched]
    assert counts == [28, 34, 40, 49, 59, 70, 85, 102, 123, 147, 177, 213,
                      257, 309, 372, 447]
    assert [s[1] for s in sched] == [1] * 16
    assert [s[2] for s in sched] == [100] + [0] * 15
    # gamma 4 anchored at 1/256: the requested coarse perplexity never exceeds
    # the 80 top of the schedule, and the affinity ceiling clamps the small levels.
    assert sched[0][3] == pytest.approx(9.0)  # (28−1)/3 ceiling
    assert sched[-1][3] >= 30.0
    assert all(s[3] <= 80.0 for s in sched)


def test_small_n_schedule_dedupes_and_spreads_budget_16():
    for n in (3, 4, 10, 60, 120, 447):
        sched = _production_schedule(n)
        assert len(sched) <= 16
        counts = [s[0] for s in sched]
        assert counts == sorted(set(counts))
        assert all(2 <= c < n for c in counts)
        assert sum(s[1] for s in sched) == 16
        assert all(s[1] >= 1 for s in sched)
        assert sched[0][2] == 100
        assert all(s[2] == 0 for s in sched[1:])


def test_fraction_schedule_matches_eval_wrapper_and_min_two_rows():
    recipe = le.Recipe(
        "frac", "frac", pca_dim=50, method="tsne_multilevel_init",
        tsne_sample_fractions=(0.01, 0.1, 0.3),
        tsne_coarse_iteration_budget=120,
        tsne_perplexity_gamma=2.0,
        tsne_perplexity_start_fraction=0.25,
    )
    for n in (100, 1000, 7153):
        assert ml.resolve_fraction_schedule(
            recipe.tsne_sample_fractions, n,
            normal_budget=recipe.tsne_coarse_iteration_budget,
            gamma=recipe.tsne_perplexity_gamma,
            start_fraction=recipe.tsne_perplexity_start_fraction,
        ) == le.resolve_multilevel_schedule(recipe, n)
    # No 16-row floor: the smallest level is max(2, round(n·f)).
    assert ml.resolve_fraction_survivors((0.0001,), 10_000) == [(0.0001, 2)]


def test_expand_preserves_landmarks_and_jitters_coordinate_width():
    n = 30
    reduced = np.random.RandomState(0).standard_normal((n, 4))
    idx = np.array([0, 5, 9])
    coords = np.random.RandomState(1).standard_normal((3, 3)).astype(np.float32)
    out = ml.expand_landmark_init(
        reduced, idx, coords, np.arange(n), seed=42, n_jobs=1
    )
    assert out.shape == (n, 3)
    np.testing.assert_array_equal(out[idx], coords)
    others = np.setdiff1d(np.arange(n), idx)
    scale = float(coords.std(axis=0).max())
    nearest = np.min(
        np.linalg.norm(out[others][:, None, :] - coords[None, :, :], axis=2),
        axis=1,
    )
    assert np.all(nearest < scale * 1e-2)


def _real_fit_adapters(dims: int):
    from openTSNE import TSNE

    def _fit(X, init, normal, early):
        tsne = TSNE(
            n_components=dims, perplexity=min(30.0, max(5.0, (len(X) - 1) / 3.0)),
            random_state=42, n_jobs=1, verbose=False,
            n_iter=normal, early_exaggeration_iter=early,
            **({"negative_gradient_method": "bh"} if dims == 3 else {}),
        )
        kwargs = {}
        if init is not None:
            kwargs["initialization"] = ml.scale_initialization(init)
        return tsne.fit(X, **kwargs)

    def fit_level(subset, normal, early, perplexity, init):
        return _fit(subset, init, normal, early)

    def fit_final(full, init):
        return _fit(full, init, 20, 0)

    return fit_level, fit_final


@pytest.mark.parametrize("dims", [2, 3])
def test_real_small_seed_is_deterministic(dims):
    n = 24
    reduced = np.random.RandomState(0).standard_normal((n, 5)).astype(np.float32)
    schedule = [(6, 10, 5, 30.0), (12, 10, 0, 30.0)]
    fit_level, fit_final = _real_fit_adapters(dims)
    _final, c1, _t = ml.fit_multilevel(
        reduced, schedule, fit_level=fit_level, fit_final=fit_final,
        seed=42, n_jobs=1,
    )
    c2 = ml.fit_multilevel(
        reduced, schedule, fit_level=fit_level, fit_final=fit_final,
        seed=42, n_jobs=1,
    )[1]
    assert c1.shape == (n, dims)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1, c2)
