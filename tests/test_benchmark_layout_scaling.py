"""Unit tests for the layout-scaling benchmark (``backend.benchmark_layout_scaling``).

Scope is below ~10k tracks (the whole 500 → full-library benchmark range runs
openTSNE's Barnes-Hut engine, and the benchmark refuses n >= 10k). The tests
exercise the pure ``run_scaling`` routine with injected fake fits and fidelity
scores — no cache IO, no filesystem writes, no real fits (one tiny real openTSNE
smoke excepted).
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le
from backend import benchmark_layout_scaling as b

# Real registered recipe ids used as stand-ins for the fakes.
RID = "tsne_early100"
RID2 = "tsne50_plain"
RID3 = "tsne_sample1000_early0"

#: The registered paired 2-D/3-D combination set (three 2-D bases + their explicit
#: 3-D siblings). The benchmark is dimension-agnostic; a paired run lists all six.
COMBINATION_BASE_IDS = (
    "tsne_continuous16",
    "tsne_continuous16_wide",
    "tsne_continuous16_wide_gamma4",
)
COMBINATION_IDS = (
    *COMBINATION_BASE_IDS,
    *(f"{rid}_3d" for rid in COMBINATION_BASE_IDS),
)


def _mat(n, seed=0):
    return np.random.RandomState(seed).standard_normal((n, 60)).astype(np.float32)


def _paths(n):
    return [f"p{i}" for i in range(n)]


def _fake_fit(mat, recipe, *, seed, n_jobs, stages=None):
    """Return constant coords + a fixed stage dict; never touches real data."""
    return np.zeros((mat.shape[0], 2), dtype=np.float32), dict(stages or {"pca": 0.1, "tsne": 0.1})


def _per_recipe(res, size, rid):
    return res["results"][str(size)]["recipes"][rid]


# ── size resolution ──────────────────────────────────────────────────────────


def test_resolve_sizes_clamps_dedupes_and_appends_full():
    # Clamp to n, dedupe, keep ascending, and always end on the full eligible set.
    assert b._resolve_sizes([500, 1000, 500, 2000, 4000], 300) == [300]
    assert b._resolve_sizes([500, 1000], 4000) == [500, 1000, 4000]
    assert b._resolve_sizes([], 50) == [50]
    assert b._resolve_sizes([50, 60], 50) == [50]
    # A non-positive size is rejected, not silently dropped.
    with pytest.raises(ValueError, match=">= 2"):
        b._resolve_sizes([0, 100], 300)
    with pytest.raises(ValueError, match=">= 2"):
        b._resolve_sizes([-5, 100], 300)


# ── nested subsets ───────────────────────────────────────────────────────────


def test_nested_deterministic_subsets(monkeypatch):
    n = 40
    mat = _mat(n)
    seen: list[list[float]] = []

    def fake(mat, recipe, *, seed, n_jobs):
        seen.append(list(mat[:, 0]))
        return _fake_fit(mat, recipe, seed=seed, n_jobs=n_jobs)

    res = b.run_scaling(
        mat, _paths(n), sizes=[10, 20], recipe_ids=[RID],
        seed=42, repeats=2, threads=2, fit=fake, score=False,
    )
    # sizes resolved to [10, 20, 40]; per size one warmup + repeats measured fits.
    assert res["sizes"] == [10, 20, 40]
    # First fit of each size is its warmup: blocks of 3 calls per size.
    s10 = set(seen[0]); s20 = set(seen[3]); s40 = set(seen[6])
    assert s10 <= s20 <= s40  # nested supersets
    assert len(s10) == 10 and len(s20) == 20 and len(s40) == 40
    # Determinism: same seed → same subsets.
    seen2: list[list[float]] = []

    def fake2(mat, recipe, *, seed, n_jobs):
        seen2.append(list(mat[:, 0]))
        return _fake_fit(mat, recipe, seed=seed, n_jobs=n_jobs)

    b.run_scaling(
        mat, _paths(n), sizes=[10, 20], recipe_ids=[RID],
        seed=42, repeats=2, threads=2, fit=fake2, score=False,
    )
    assert seen2[0] == seen[0] and seen2[3] == seen[3]


# ── per-round recipe orders + repeat count ───────────────────────────────────


def test_round_orders_distinct_reproducible_and_each_repeated(monkeypatch):
    n = 15
    mat = _mat(n)
    ids = [RID, RID2, RID3]
    calls: list[str] = []

    def fake(mat, recipe, *, seed, n_jobs):
        calls.append(recipe.id)
        return _fake_fit(mat, recipe, seed=seed, n_jobs=n_jobs)

    res = b.run_scaling(
        mat, _paths(n), sizes=[15], recipe_ids=ids,
        seed=7, repeats=3, threads=2, fit=fake, score=False,
    )
    # Per-round orders live at the size level (one permutation per measured round),
    # never a misleading identical per-recipe "recipe_order".
    round_orders = res["results"]["15"]["recipe_order"]
    assert len(round_orders) == 3
    for order in round_orders:
        assert sorted(order) == sorted(ids)
    # Rounds do not share a single fixed order.
    assert len({tuple(o) for o in round_orders}) >= 2
    # Reproducible: same seed → identical per-round orders.
    calls2: list[str] = []

    def fake2(mat, recipe, *, seed, n_jobs):
        calls2.append(recipe.id)
        return _fake_fit(mat, recipe, seed=seed, n_jobs=n_jobs)

    res2 = b.run_scaling(
        mat, _paths(n), sizes=[15], recipe_ids=ids,
        seed=7, repeats=3, threads=2, fit=fake2, score=False,
    )
    assert res2["results"]["15"]["recipe_order"] == round_orders
    # Per recipe: one warmup + repeats measured fits.
    assert Counter(calls) == {r: 4 for r in ids}
    for rid in ids:
        assert len(_per_recipe(res, 15, rid)["samples"]) == 3


def test_warmups_precede_measured_rounds(monkeypatch):
    # All warmups run first (each recipe once), then the measured rounds.
    n = 15
    mat = _mat(n)
    ids = [RID, RID2]
    calls: list[str] = []

    def fake(mat, recipe, *, seed, n_jobs):
        calls.append(recipe.id)
        return _fake_fit(mat, recipe, seed=seed, n_jobs=n_jobs)

    b.run_scaling(
        mat, _paths(n), sizes=[15], recipe_ids=ids,
        seed=3, repeats=2, threads=2, fit=fake, score=False,
    )
    # First 2 calls are the warmups (both recipes); the rest are the measured rounds.
    assert set(calls[:2]) == set(ids)
    assert len(calls) == len(ids) * (1 + 2)


# ── warmup excluded from timings and scoring ────────────────────────────────


def test_warmup_excluded_from_timings_and_scoring(monkeypatch):
    n = 20
    mat = _mat(n)
    fit_sleep = 0.005
    fit_call = {"n": 0}
    scored = []

    def fake(mat, recipe, *, seed, n_jobs):
        fit_call["n"] += 1
        time.sleep(fit_sleep)
        # Encode the fit-call index in the coords so we can tell which fit was scored.
        return np.full((mat.shape[0], 2), fit_call["n"], dtype=np.float32), {"pca": 0.001, "tsne": 0.001}

    def fake_recall(high, low, *, k, sample, seed):
        scored.append(int(low[0, 0]))
        return 0.5

    monkeypatch.setattr(b, "neighbourhood_recall", fake_recall)
    res = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=3, threads=2, fit=fake,
    )
    # warmup is fit call #1; the first measured map is fit call #2 → scored on #2
    # for both k=10 and k=100 (the untimed quality metrics).
    assert scored == [2, 2]
    assert len(_per_recipe(res, 20, RID)["samples"]) == 3
    totals = [s["total"] for s in _per_recipe(res, 20, RID)["samples"]]
    assert all(fit_sleep - 1e-3 <= t < fit_sleep * 3 for t in totals)
    assert all(t < fit_sleep + 0.01 for t in totals)


# ── threadpoolctl setup excluded from the timed total ───────────────────────


def test_thread_limits_setup_excluded_from_total(monkeypatch):
    n = 20
    mat = _mat(n)
    fit_sleep = 0.005

    def fake(mat, recipe, *, seed, n_jobs):
        time.sleep(fit_sleep)
        return np.zeros((mat.shape[0], 2), dtype=np.float32), {"pca": 0.001, "tsne": 0.001}

    @contextlib.contextmanager
    def slow_limits(threads):  # emulates a costly threadpoolctl setup
        time.sleep(0.05)
        yield

    monkeypatch.setattr(b, "_thread_limits", slow_limits)
    res = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=3, threads=2, fit=fake, score=False,
    )
    # t0 is taken inside the limits context, immediately before each fit, so the
    # 0.05s setup never leaks into a sample's total.
    totals = [s["total"] for s in _per_recipe(res, 20, RID)["samples"]]
    assert all(fit_sleep - 1e-3 <= t < fit_sleep + 0.02 for t in totals)


# ── stage summaries: no double count ────────────────────────────────────────


def test_stage_summaries_no_double_count(monkeypatch):
    n = 20
    mat = _mat(n)

    def fake(mat, recipe, *, seed, n_jobs):
        # Stages sum to more than the real (tiny) wall — the total must NOT be
        # built by re-summing the stage keys.
        return np.zeros((mat.shape[0], 2), dtype=np.float32), {"pca": 1.0, "tsne": 1.0}

    res = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=3, threads=2, fit=fake, score=False,
    )
    timings = _per_recipe(res, 20, RID)["timings"]
    assert set(timings) == {"total", "pca", "tsne"}
    assert timings["total"]["median"] < 0.01  # independent measured wall
    assert timings["pca"]["median"] == 1.0
    assert timings["tsne"]["median"] == 1.0
    assert list(timings["total"]) == ["median", "min", "max"]


# ── no cache / production / filesystem touch ────────────────────────────────


def test_run_scaling_performs_no_filesystem_io(tmp_path, monkeypatch):
    n = 10
    mat = _mat(n)
    monkeypatch.chdir(tmp_path)
    b.run_scaling(
        mat, _paths(n), sizes=[10], recipe_ids=[RID],
        seed=42, repeats=1, threads=2, fit=_fake_fit, score=False,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_scaling_invalid_recipe_id_clean_error():
    mat = _mat(10)
    with pytest.raises(ValueError, match="unknown recipe id"):
        b.run_scaling(
            mat, _paths(10), sizes=[10], recipe_ids=["not-a-recipe"],
            seed=42, repeats=1, threads=2, fit=_fake_fit, score=False,
        )


def test_incompatible_registered_recipes_rejected():
    # Only the canonical PCA50 t-SNE input is benchmarked, so a source-mix
    # recipe, a non-PCA50 map, and a UMAP recipe are rejected (they would fit a
    # different input geometry / their own weighted matrix).
    mat = _mat(10)
    paths = _paths(10)
    for rid, reason in (
        ("tsne_rhythm_forward", "source mix"),   # MIX_RECIPES
        ("pca2", "pca_dim=2"),                   # non-PCA50
        ("umap2_n15", "method='umap'"),          # UMAP geometry
    ):
        with pytest.raises(ValueError, match="incompatible"):
            b.run_scaling(mat, paths, sizes=[10], recipe_ids=[rid],
                          seed=42, repeats=1, threads=2, fit=_fake_fit, score=False)
    # The default scaling set is all compatible.
    b._validate_recipes(list(b.DEFAULT_RECIPES))


def test_validate_recipes_accepts_paired_combination_set():
    # The six registered combination ids (three 2-D bases + explicit 3-D siblings)
    # all fit the canonical PCA50 input; dimension is not a compatibility axis.
    b._validate_recipes(list(COMBINATION_IDS))
    assert [le.RECIPES_BY_ID[rid].dims or 2 for rid in COMBINATION_IDS] == [
        2, 2, 2, 3, 3, 3,
    ]
    # An explicit 3-D recipe runs through the same validation as its base.
    for rid in COMBINATION_IDS:
        assert le.RECIPES_BY_ID[rid].group in (None, *COMBINATION_BASE_IDS)


def test_default_scaling_set_unchanged():
    # Adding the paired 2-D/3-D comparison must not silently enlarge or reorder
    # the historical default benchmark set; a 3-D run lists its ids explicitly.
    assert b.DEFAULT_RECIPES == (
        "tsne_early100",
        "tsne50_plain",
        "tsne_sample1000_early0",
        "tsne_sample1000_coarse10e100",
        "tsne_sample1000_coarse100e0",
        "tsne_sample100_early0",
        "tsne_chain100_500_2000_final50",
    )
    assert not (set(COMBINATION_IDS) & set(b.DEFAULT_RECIPES))


def test_run_scaling_paths_len_mismatch_clean_error():
    mat = _mat(10)
    with pytest.raises(ValueError, match="mat.shape"):
        b.run_scaling(
            mat, _paths(9), sizes=[10], recipe_ids=[RID],
            seed=42, repeats=1, threads=2, fit=_fake_fit, score=False,
        )


# ── input validators ────────────────────────────────────────────────────────


def _expect_value_error(mat, paths, **over):
    kwargs = dict(sizes=[20], recipe_ids=[RID], repeats=3, threads=2,
                  fit=_fake_fit, score=False)
    kwargs.update(over)
    with pytest.raises(ValueError):
        b.run_scaling(mat, paths, **kwargs)


def test_run_scaling_rejects_bad_inputs():
    mat = _mat(30)
    paths = _paths(30)
    # Repeat / thread counts.
    _expect_value_error(mat, paths, repeats=0)
    _expect_value_error(mat, paths, repeats=-1)
    _expect_value_error(mat, paths, threads=0)
    # Too few tracks.
    _expect_value_error(_mat(1), _paths(1))
    # Non-2-D matrix.
    _expect_value_error(np.zeros(30, dtype=np.float32), paths)
    # Empty / mismatched paths.
    _expect_value_error(mat, [])
    # Empty / duplicate recipes.
    _expect_value_error(mat, paths, recipe_ids=[])
    _expect_value_error(mat, paths, recipe_ids=[RID, RID])
    # Non-positive sizes.
    _expect_value_error(mat, paths, sizes=[0, 100])
    _expect_value_error(mat, paths, sizes=[-5])


def test_run_scaling_rejects_10k_scope():
    # The benchmark refuses n >= 10k (no FFT/gradient branch). n=10000 hits the
    # ceiling; a just-below library is fine.
    mat = _mat(10000)
    with pytest.raises(ValueError, match="Barnes-Hut"):
        b.run_scaling(mat, _paths(10000), sizes=[10000], recipe_ids=[RID],
                      seed=42, repeats=1, threads=2, fit=_fake_fit, score=False)
    mat9 = _mat(9999)
    res = b.run_scaling(mat9, _paths(9999), sizes=[9999], recipe_ids=[RID],
                        seed=42, repeats=1, threads=2, fit=_fake_fit, score=False)
    assert res["eligible_tracks"] == 9999


# ── fidelity outside the timed region ───────────────────────────────────────


def test_fidelity_scored_outside_measured_wall(monkeypatch):
    n = 20
    mat = _mat(n)
    fit_sleep = 0.01
    scored_count = {"n": 0}

    def fake(mat, recipe, *, seed, n_jobs):
        time.sleep(fit_sleep)
        return np.zeros((mat.shape[0], 2), dtype=np.float32), {"pca": 0.001, "tsne": 0.001}

    def fake_recall(high, low, *, k, sample, seed):
        scored_count["n"] += 1
        time.sleep(0.05)  # fidelity cost that must NOT leak into the timed total
        return 0.0

    monkeypatch.setattr(b, "neighbourhood_recall", fake_recall)
    res = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=3, threads=2, fit=fake,
    )
    totals = [s["total"] for s in _per_recipe(res, 20, RID)["samples"]]
    assert all(t < fit_sleep + 0.02 for t in totals)  # 0.05s fidelity not timed
    assert scored_count["n"] == 2  # k=10 and k=100 scored once each, on the first measured map


# ── quality metrics: fidelity@100 + broad-distance Spearman ──────────────────


def _monotone_line_pair(n):
    x = np.linspace(0.0, 1.0, n)[:, None]
    hi = np.concatenate([x, np.ones((n, 1)), np.zeros((n, 48))], axis=1)  # 50-D
    lo = np.concatenate([x * 3.7 + 1.2, -x * 0.5 + 2.0], axis=1)          # 2-D
    return hi.astype(np.float32), lo.astype(np.float32)


def test_broad_distance_spearman_line_scaled_high_correlation():
    # A 2-D layout that is a monotone (distance-order-preserving) function of the
    # high-dim line keeps broad pair distances ranked almost perfectly.
    hi, lo = _monotone_line_pair(500)
    rho = b.broad_distance_spearman(hi, lo, n_pairs=2048, seed=1)
    assert rho is not None
    assert rho > 0.99


def test_broad_distance_spearman_permuted_low():
    # Shuffling the low-dim rows scrambles the pair-distance ranking → low rho.
    hi, lo = _monotone_line_pair(500)
    lo_perm = lo[np.random.RandomState(0).permutation(500)]
    rho = b.broad_distance_spearman(hi, lo_perm, n_pairs=2048, seed=1)
    assert rho is not None
    assert abs(rho) < 0.5


def test_broad_distance_spearman_degenerate_is_null_not_zero():
    # A constant (degenerate) 2-D layout yields a missing score (None), never a
    # fabricated zero, so the writer's allow_nan=False stays safe.
    hi, lo = _monotone_line_pair(200)
    lo_const = np.zeros_like(lo)
    assert b.broad_distance_spearman(hi, lo_const, n_pairs=2048, seed=1) is None
    # A row-count mismatch between the high- and low-dim inputs is also a null.
    assert b.broad_distance_spearman(hi, lo[:100], n_pairs=16, seed=1) is None


def test_broad_distance_spearman_deterministic_same_pairs():
    # Same seed → identical pairs → identical score (reproducible across calls,
    # and across recipes at a size because they share the PCA50 reference).
    hi, lo = _monotone_line_pair(400)
    assert b.broad_distance_spearman(hi, lo, n_pairs=2048, seed=7) == \
        b.broad_distance_spearman(hi, lo, n_pairs=2048, seed=7)
    # A different seed draws different pairs (different score).
    assert b.broad_distance_spearman(hi, lo, n_pairs=2048, seed=7) != \
        b.broad_distance_spearman(hi, lo, n_pairs=2048, seed=8)


def test_run_scaling_reports_all_three_untimed_metrics(monkeypatch):
    # The default (score=True) run records fidelity@10, fidelity@100 and the
    # broad-distance Spearman per recipe, all keyed on the first measured map.
    n = 20
    mat = _mat(n)

    def fake(mat, recipe, *, seed, n_jobs):
        return np.zeros((mat.shape[0], 2), dtype=np.float32), {"pca": 0.001, "tsne": 0.001}

    def fake_recall(high, low, *, k, sample, seed):
        return 0.5

    monkeypatch.setattr(b, "neighbourhood_recall", fake_recall)
    res = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=1, threads=2, fit=fake,
    )
    entry = _per_recipe(res, 20, RID)
    assert entry["fidelity"] == 0.5          # back-compat @10
    assert entry["fidelity100"] == 0.5       # @100
    assert entry["broad_distance_spearman"] is None  # degenerate all-zero layout
    # With score=False every metric is absent (None), not computed.
    res2 = b.run_scaling(
        mat, _paths(n), sizes=[20], recipe_ids=[RID],
        seed=42, repeats=1, threads=2, fit=fake, score=False,
    )
    entry2 = _per_recipe(res2, 20, RID)
    assert entry2["fidelity"] is None and entry2["fidelity100"] is None
    assert entry2["broad_distance_spearman"] is None


def test_run_scaling_scores_three_d_recipe_any_width(monkeypatch):
    # A registered 3-D recipe fits (n, 3) coords; both untimed quality metrics must
    # accept the wider map, so a paired 2-D/3-D benchmark records comparable scores.
    n = 30
    mat = _mat(n)
    widths: list[int] = []

    def fake_recall(high, low, *, k, sample, seed):
        widths.append(low.shape[1])
        return 0.5

    def fake_fit(mat, recipe, *, seed, n_jobs):
        dims = recipe.dims or 2
        coords = np.random.default_rng(seed).standard_normal(
            (mat.shape[0], dims)).astype(np.float32)
        return coords, {"pca": 0.001, "tsne": 0.001}

    monkeypatch.setattr(b, "neighbourhood_recall", fake_recall)
    res = b.run_scaling(
        mat, _paths(n), sizes=[n],
        recipe_ids=["tsne_continuous16_wide_gamma4_3d", "tsne_continuous16_wide_gamma4"],
        seed=42, repeats=1, threads=2, fit=fake_fit,
    )
    entry_3d = _per_recipe(res, n, "tsne_continuous16_wide_gamma4_3d")
    entry_2d = _per_recipe(res, n, "tsne_continuous16_wide_gamma4")
    assert sorted(widths) == [2, 2, 3, 3]  # k=10 and k=100 per recipe, first map
    for entry in (entry_3d, entry_2d):
        assert entry["fidelity"] == 0.5
        assert entry["fidelity100"] == 0.5
        assert isinstance(entry["broad_distance_spearman"], float)
    # The 3-D entry records its dims/group metadata via the registry params; the
    # 2-D base omits them (None = 2-D, group defaults to the recipe id).
    assert le.recipe_params(le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4_3d"])[
        "dims"
    ] == 3
    assert le.recipe_params(le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4_3d"])[
        "group"
    ] == "tsne_continuous16_wide_gamma4"
    assert "dims" not in le.recipe_params(
        le.RECIPES_BY_ID["tsne_continuous16_wide_gamma4"]
    )
    json.dumps(res, allow_nan=False)


def test_broad_distance_spearman_accepts_three_and_four_dims():
    hi, lo = _monotone_line_pair(300)
    # Embed the same monotone line in a 3-D and a 4-D layout: only row distances
    # matter, so the broad rank correlation stays high at any map width.
    for extra in (1, 2):
        padded = np.concatenate(
            [lo, np.ones((lo.shape[0], extra), dtype=np.float32)], axis=1
        )
        rho = b.broad_distance_spearman(hi, padded, n_pairs=1024, seed=1)
        assert rho is not None and rho > 0.99
    # A degenerate constant 3-D layout is still a null score, not a fake zero.
    pad_const = np.zeros((lo.shape[0], 3), dtype=np.float32)
    assert b.broad_distance_spearman(hi, pad_const, n_pairs=1024, seed=1) is None


# ── real openTSNE smoke ─────────────────────────────────────────────────────


def test_real_fit_recipe_smoke_tiny():
    # A real openTSNE fit through run_scaling (real pca_reduce + fidelity), on a
    # tiny matrix so it stays fast. Proves the default fit path integrates with
    # the benchmark and the output is JSON-serialisable (allow_nan=False).
    mat = _mat(12)
    res = b.run_scaling(
        mat, _paths(12), sizes=[12], recipe_ids=["tsne50_plain"],
        seed=42, repeats=2, threads=2, fit=le.fit_recipe,
    )
    assert res["sizes"] == [12]
    entry = _per_recipe(res, 12, "tsne50_plain")
    assert len(entry["samples"]) == 2
    assert entry["fidelity"] is not None
    assert 0.0 <= entry["fidelity"] <= 1.0
    assert entry["timings"]["total"]["median"] >= 0.0
    json.dumps(res, allow_nan=False)  # must not contain NaN


# ── atomic writer (the CLI's write path) ────────────────────────────────────


def test_atomic_write_json(tmp_path):
    out = tmp_path / "out.json"
    b._atomic_write_json(str(out), {"a": 1, "b": [1.5, 2]})
    assert out.exists()
    assert not (tmp_path / "out.json.tmp").exists()  # no leftover temp file
    assert json.loads(out.read_text()) == {"a": 1, "b": [1.5, 2]}


def test_atomic_write_json_rejects_nan(tmp_path):
    out = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        b._atomic_write_json(str(out), {"x": float("nan")})
    # A rejected write leaves no partial file behind.
    assert not out.exists()
    assert not (tmp_path / "nan.json.tmp").exists()


def test_single_track_size_rejected():
    with pytest.raises(ValueError, match=">= 2"):
        b._resolve_sizes([1], 100)


def test_main_writes_incomplete_checkpoint_before_gather(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "roots.json").write_text("{}")
    output = tmp_path / "scaling.json"
    # A previous completed report must not survive the new run's gather.
    output.write_text(json.dumps({
        "tool": "layout-scaling-benchmark",
        "checkpoint": {"sizes_completed": [10], "complete": True},
    }))

    def fake_roots(roots, _path, default_root=None, root_roles=None):
        roots["lib"] = "/music"
        if root_roles is not None:
            root_roles["lib"] = "library"

    monkeypatch.setattr(b, "load_roots_state", fake_roots)
    monkeypatch.setattr(b, "FeatureCache", lambda *a, **k: object())
    monkeypatch.setattr(b, "TrackCache", lambda *a, **k: object())
    monkeypatch.setattr(b, "load_infos", lambda *a, **k: ([{"path": "a"}], 1))

    def fail(*_args, **_kwargs):
        raise RuntimeError("gather failed")

    monkeypatch.setattr(b, "prepare_projection_input", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_layout_scaling",
            "--data-dir", str(data),
            "--output", str(output),
            "--sizes", "10",
            "--repeats", "1",
        ],
    )
    with pytest.raises(RuntimeError, match="gather failed"):
        b.main()
    payload = json.loads(output.read_text())
    assert payload["checkpoint"] == {"sizes_completed": [], "complete": False}
