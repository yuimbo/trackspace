"""Tests for the production projection benchmark and its fit seam."""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np
import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.projection_backends as pb
from backend import benchmark_production_projection as b
from backend.cache import TrackCache
from backend.precompute_layout_evaluation import load_infos
from backend.services.roots_registry import ROLE_EXTERNAL, ROLE_LIBRARY, add_root


# ── load_infos scope ─────────────────────────────────────────────────────────


def _cached_roots(tmp_path):
    lib = tmp_path / "lib"
    ext = tmp_path / "external"
    lib.mkdir()
    ext.mkdir()
    lib_file = lib / "a.mp3"
    ext_file = ext / "b.mp3"
    lib_file.write_bytes(b"a")
    ext_file.write_bytes(b"b")

    roots: OrderedDict[str, str] = OrderedDict()
    add_root(str(lib), roots)
    add_root(str(ext), roots)
    lib_id = next(rid for rid, p in roots.items() if p == str(lib))
    ext_id = next(rid for rid, p in roots.items() if p == str(ext))
    roles = OrderedDict({lib_id: ROLE_LIBRARY, ext_id: ROLE_EXTERNAL})

    cache = TrackCache(str(tmp_path / "track_cache.db"))
    for path, fp in ((lib_file, "fp-lib"), (ext_file, "fp-ext")):
        cache.put(str(path), os.path.getmtime(path), {"fingerprint": fp})
    return roots, roles, cache


def test_load_infos_default_scope_is_library_only(tmp_path):
    roots, roles, cache = _cached_roots(tmp_path)
    infos, n_mp3s = load_infos(roots, cache, roles)
    assert n_mp3s == 1
    assert [t["fingerprint"] for t in infos] == ["fp-lib"]


def test_load_infos_scope_all_includes_external_roots(tmp_path):
    roots, roles, cache = _cached_roots(tmp_path)
    infos, n_mp3s = load_infos(roots, cache, roles, scope="all")
    assert n_mp3s == 2
    assert sorted(t["fingerprint"] for t in infos) == ["fp-ext", "fp-lib"]


def test_load_infos_unknown_scope_rejected(tmp_path):
    roots, roles, cache = _cached_roots(tmp_path)
    with pytest.raises(ValueError, match="unknown scope"):
        load_infos(roots, cache, roles, scope="everything")


# ── fit seam ─────────────────────────────────────────────────────────────────


def test_fit_tsne_production_defaults_and_3d_bh_pin(monkeypatch):
    import openTSNE

    from backend.embeddings.projection_config import OPENTSNE_FINAL_EXAGGERATION

    calls: list[dict] = []

    class _RecordingTSNE:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

        def fit(self, x, initialization=None):
            dims = calls[-1]["n_components"]
            return np.zeros((len(x), dims), dtype=np.float32)

    monkeypatch.setattr(openTSNE, "TSNE", _RecordingTSNE)
    mat = np.random.RandomState(0).standard_normal((120, 24)).astype(np.float32)

    pb._fit_tsne(mat)
    assert calls[-1]["n_jobs"] == -1
    assert "theta" not in calls[-1]
    # The canonical late exaggeration is on the final fit only.
    assert all("exaggeration" not in c for c in calls[:-1])
    assert calls[-1]["exaggeration"] == pytest.approx(
        OPENTSNE_FINAL_EXAGGERATION
    )
    assert "negative_gradient_method" not in calls[-1]

    calls.clear()
    _emb, coords = pb._fit_tsne(mat, dims=3)
    assert coords.shape == (120, 3)
    assert calls[-1]["negative_gradient_method"] == "bh"
    assert calls[-1]["exaggeration"] == pytest.approx(
        OPENTSNE_FINAL_EXAGGERATION
    )

    # An explicit comparison value (the benchmark's ×1 control) is not replaced.
    calls.clear()
    pb._fit_tsne(mat, final_exaggeration=1.0)
    assert calls[-1]["exaggeration"] == pytest.approx(1.0)


def test_fit_tsne_tuning_only_changes_the_requested_knobs(monkeypatch):
    import openTSNE

    calls: list[dict] = []

    class _RecordingTSNE:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

        def fit(self, x, initialization=None):
            dims = calls[-1]["n_components"]
            return np.zeros((len(x), dims), dtype=np.float32)

    monkeypatch.setattr(openTSNE, "TSNE", _RecordingTSNE)
    mat = np.random.RandomState(0).standard_normal((120, 24)).astype(np.float32)
    timings: list[dict[str, float]] = []

    pb._fit_tsne(
        mat,
        dims=2,
        n_jobs=4,
        negative_gradient_method="bh",
        theta=0.7,
        final_exaggeration=2.0,
        on_timings=timings.append,
    )

    assert all(c["n_jobs"] == 4 for c in calls)
    assert all(c["negative_gradient_method"] == "bh" for c in calls)
    assert all(c["theta"] == pytest.approx(0.7) for c in calls)
    # Late exaggeration is on the final fit only; the coarse early phase is unchanged.
    assert all("exaggeration" not in c for c in calls[:-1])
    assert calls[-1]["exaggeration"] == pytest.approx(2.0)
    assert calls[-1]["early_exaggeration"] == 12
    assert calls[-1]["early_exaggeration_iter"] == 0
    assert calls[0]["early_exaggeration_iter"] == 100
    assert timings and "fit" in timings[-1]
    assert "tsne_final" in timings[-1]


def test_fit_tsne_rejects_fft_in_3d():
    mat = np.random.RandomState(0).standard_normal((20, 6)).astype(np.float32)
    with pytest.raises(ValueError, match="Barnes-Hut"):
        pb._fit_tsne(mat, dims=3, negative_gradient_method="fft")


def test_small_real_multilevel_fit_reports_stage_timings():
    mat = np.random.RandomState(1).standard_normal((64, 12)).astype(np.float32)
    seen: list[dict[str, float]] = []
    _emb, coords = pb._fit_tsne(
        mat, dims=2, n_jobs=2, on_timings=seen.append,
    )
    assert coords.shape == (64, 2)
    assert np.isfinite(coords).all()
    assert seen and "fit" in seen[0]
    assert "tsne_final" in seen[0]
    assert "tsne_expand" in seen[0]


# ── harness CLI / orchestration ──────────────────────────────────────────────


def test_parse_cases_and_dims():
    cases = b._parse_cases("native,threads4,folders-off,bh,fft,theta07,exag2", None)
    by_name = {c.name: c for c in cases}
    assert by_name["native"].threads is None
    assert by_name["native"].basis_mode == "full"
    assert by_name["threads4"].threads == 4
    assert by_name["folders-off"].folders is False
    assert by_name["folders-off"].basis_mode == "none"
    assert by_name["bh"].engine == "bh"
    assert by_name["fft"].engine == "fft"
    assert by_name["theta07"].theta == pytest.approx(0.7)
    assert by_name["theta07"].engine == "bh"
    assert by_name["exag2"].final_exaggeration == pytest.approx(2.0)

    # --threads appends capped thread cases and dedupes by name.
    with_threads = b._parse_cases("native", "4,8")
    assert [c.name for c in with_threads] == ["native", "threads4", "threads8"]
    assert b._parse_dims("3,2,2") == [2, 3]
    with pytest.raises(ValueError):
        b._parse_dims("4")
    with pytest.raises(ValueError, match="unknown case"):
        b._parse_cases("nope", None)


def test_named_cheap_cases_map_modes_and_exaggerations():
    cases = b._parse_cases(",".join(b.FOLDER_CHEAP_CASES), None)
    by_name = {c.name: c for c in cases}
    assert [c.name for c in cases] == list(b.FOLDER_CHEAP_CASES)
    assert by_name["folders-off"].basis_mode == "none"
    assert by_name["folders-off"].final_exaggeration is None
    assert by_name["centroids"].basis_mode == "centroids"
    assert by_name["approx"].basis_mode == "approx"
    assert by_name["off-exag15"].basis_mode == "none"
    assert by_name["off-exag15"].final_exaggeration == pytest.approx(1.5)
    assert by_name["off-exag2"].final_exaggeration == pytest.approx(2.0)
    assert by_name["centroids-exag15"].basis_mode == "centroids"
    assert by_name["centroids-exag15"].final_exaggeration == pytest.approx(1.5)
    # The named case wins over the legacy dynamic exagN parser (which reads 15).
    assert b._parse_cases("exag15", None)[0].final_exaggeration == 15.0
    with pytest.raises(ValueError, match="basis mode"):
        b.CaseSpec("bad", basis_mode="legacy")


def test_resolved_engines_match_production_auto_rule():
    auto = b.CaseSpec("native")
    engines_2d = b._engines(auto, 2, 15_571)
    assert engines_2d[-1] == {"level": "tsne_final", "count": 15_571, "engine": "fft"}
    assert any(e["engine"] == "bh" for e in engines_2d[:-1])
    # 3-D pins BH even above the auto-FFT row count.
    assert all(e["engine"] == "bh" for e in b._engines(auto, 3, 15_571))
    # An explicit FFT is skipped for 3-D, never run.
    assert b._skip_reason(b.CaseSpec("fft", engine="fft"), 3)
    # The auto note follows the resolved engines, not a fixed 15k assumption.
    notes_2d = b._case_notes(auto, 2, engines_2d, False)
    assert any("full-data level uses FFT" in n for n in notes_2d)
    notes_bh = b._case_notes(auto, 2, b._engines(auto, 2, 20), False)
    assert any("every level resolves to Barnes-Hut" in n for n in notes_bh)


def _fake_input(
    n: int,
    *,
    prepare=1.0,
    weighted_prepare=1.0,
    gather_unweighted=0.5,
    basis=0.5,
    pca=0.25,
    folders=True,
    basis_mode=None,
    warm=False,
):
    mode = basis_mode or ("full" if folders else "none")
    rng = np.random.default_rng(0)
    return b.CaseInput(
        paths=[f"p{i}" for i in range(n)],
        mat_proj=rng.standard_normal((n, 8)).astype(np.float32),
        reduced=rng.standard_normal((n, 4)).astype(np.float32),
        comp_config="cfg",
        tsne_config="cfg|basis" if folders else "cfg",
        prepare_s=prepare,
        weighted_prepare_s=weighted_prepare if folders else None,
        gather_unweighted_s=gather_unweighted,
        basis_estimate_s=basis if folders else None,
        prepare_warm=warm,
        pca_s=pca,
        folder_weighting=folders,
        fingerprint="fp",
        basis_mode=mode,
    )


def test_run_benchmark_tasks_checkpoints_and_timings():
    cases = [
        b.CaseSpec("native"),
        b.CaseSpec("threads4", threads=4),
        b.CaseSpec("fft", engine="fft"),
        b.CaseSpec("folders-off", basis_mode="none"),
    ]
    inputs = {
        "full": _fake_input(20),
        "none": _fake_input(20, folders=False),
    }
    done: list[tuple[str, int]] = []
    calls: list[tuple[str, int, int]] = []

    def fake_fit(reduced, dims, spec):
        calls.append((spec.name, dims, len(reduced)))
        return b.FitOutcome(
            coords=np.zeros((len(reduced), dims), dtype=np.float32),
            stages={"tsne_final": 0.2, "tsne_expand": 0.1, "fit": 0.3},
            threadpool_info=[{"num_threads": spec.threads or -1}],
            wall=0.3,
        )

    results = b.run_benchmark(
        cases,
        [2, 3],
        inputs,
        repeats=2,
        fit=fake_fit,
        score=False,
        on_case_done=lambda name, dim, result: done.append((name, dim)),
    )

    # 4 cases × 2 dims, minus the FFT/3-D skip, times (warmup + 2 measured).
    assert len(calls) == (4 * 2 - 1) * 3
    assert len(done) == 4 * 2
    assert ("fft", 3) in done
    assert results["cases"]["fft"]["3"]["skipped"]

    native2 = results["cases"]["native"]["2"]
    assert len(native2["samples"]) == 2
    # prepare + PCA + fit, never the double-counted gather+basis sum.
    assert native2["timings"]["pipeline_estimate"]["median"] == pytest.approx(
        1.0 + 0.25 + 0.3
    )
    assert native2["timings"]["weighted_prepare"] == 1.0
    assert native2["timings"]["gather_unweighted"] == 0.5
    assert native2["timings"]["basis_estimate"] == 0.5
    assert native2["threads"]["n_jobs"] == -1
    assert native2["engines"][-1]["engine"] == "bh"  # 20 rows is below the auto FFT boundary
    folder_off = results["cases"]["folders-off"]["2"]["timings"]
    assert folder_off["basis_estimate"] is None
    assert folder_off["pipeline_estimate"]["median"] == pytest.approx(1.0 + 0.25 + 0.3)
    assert results["cases"]["threads4"]["2"]["threads"]["n_jobs"] == 4


def test_theta_effective_requires_an_explicit_theta():
    cases = [b.CaseSpec("native"), b.CaseSpec("theta07", engine="bh", theta=0.7)]
    inputs = {"full": _fake_input(20)}
    results = b.run_benchmark(
        cases,
        [2],
        inputs,
        repeats=1,
        fit=lambda reduced, dims, spec: b.FitOutcome(
            coords=np.zeros((len(reduced), dims), dtype=np.float32),
            stages={"tsne_final": 0.1},
            threadpool_info=[],
            wall=0.1,
        ),
        score=False,
    )
    assert results["cases"]["native"]["2"]["theta_effective"] is False
    assert results["cases"]["theta07"]["2"]["theta_effective"] is True


def test_select_infos_preserves_full_order_and_sorts_subsets():
    infos = [{"path": f"p{i:03d}"} for i in range(20)]
    full = b._select_infos(infos, [20], seed=42)
    assert full[20] == infos  # exact native order, no permutation
    subsets = b._select_infos(infos, [5, 10, 20], seed=42)
    for s in (5, 10):
        idx = [int(t["path"][1:]) for t in subsets[s]]
        assert idx == sorted(idx)  # original order retained
        assert set(idx) <= {int(t["path"][1:]) for t in subsets[20]}
    assert subsets[20] == infos


def test_needs_clap_is_lazy_for_genre_full():
    from types import SimpleNamespace

    genre = SimpleNamespace(
        sources=("maest_logits", "maest", "clap"),
        source_weights={"clap": 0.0},
        tag_names=[],
    )
    default = SimpleNamespace(
        sources=("clap",),
        source_weights={"clap": 0.1},
        tag_names=[],
    )
    tagged = SimpleNamespace(
        sources=("maest",),
        source_weights={"clap": 0.0},
        tag_names=["techno"],
    )
    assert not b._needs_clap(genre)
    assert b._needs_clap(default)
    assert b._needs_clap(tagged)


def test_prepare_inputs_times_weighted_first_and_does_not_double_count(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_prepare(infos, cache, *, scale_folders, **kwargs):
        calls.append(scale_folders)
        n = len(infos)
        mat = np.random.default_rng(0).standard_normal((n, 4)).astype(np.float32)
        return SimpleNamespace(
            paths=[t["path"] for t in infos],
            mat=mat,
            mat_proj=mat + (1.0 if scale_folders else 0.0),
            comp_config="cfg",
            tsne_config="cfg|basis" if scale_folders else "cfg",
        )

    monkeypatch.setattr(b, "prepare_projection_input", fake_prepare)
    params = SimpleNamespace(
        sources=("clap",),
        source_weights={"clap": 0.0},
        tag_names=[],
        explicit_folders=[],
        required_sources=("clap",),
        feature_mask=np.ones(4, dtype=np.float32),
        features_blend=0.42,
        folder_boost=6.0,
        folder_depth_boost=3.0,
    )
    infos = [{"path": f"p{i}"} for i in range(6)]
    inputs, meta = b.prepare_inputs(
        infos, object(), params, {"full", "none"}
    )
    # Weighted production prepare first, then the unweighted diagnostic.
    assert calls == [True, False]
    assert inputs["full"].folder_weighting
    assert inputs["full"].prepare_warm is False
    assert inputs["none"].prepare_warm is True
    assert inputs["full"].basis_estimate_s == pytest.approx(
        max(0.0, meta["weighted_prepare_s"] - meta["gather_unweighted_s"])
    )
    assert inputs["none"].basis_estimate_s is None


def test_prepare_inputs_reports_selected_count_after_required_filter(monkeypatch):
    from types import SimpleNamespace

    def fake_prepare(infos, cache, *, scale_folders, **kwargs):
        # Simulate the required-source filter dropping one track.
        kept = infos[:-1]
        mat = np.random.default_rng(0).standard_normal((len(kept), 4)).astype(
            np.float32
        )
        return SimpleNamespace(
            paths=[t["path"] for t in kept],
            mat=mat,
            mat_proj=mat,
            comp_config="cfg",
            tsne_config="cfg",
        )

    monkeypatch.setattr(b, "prepare_projection_input", fake_prepare)
    params = SimpleNamespace(
        sources=("clap",),
        source_weights={"clap": 0.0},
        tag_names=[],
        explicit_folders=[],
        required_sources=("clap",),
        feature_mask=np.ones(4, dtype=np.float32),
        features_blend=0.42,
        folder_boost=6.0,
        folder_depth_boost=3.0,
    )
    infos = [{"path": f"p{i}"} for i in range(6)]
    inputs, meta = b.prepare_inputs(infos, object(), params, {"none"})
    assert meta["tracks"] == 5
    assert len(inputs["none"].paths) == 5


def test_prepare_inputs_prepares_each_basis_mode_once_in_order(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_prepare(infos, cache, *, scale_folders, **kwargs):
        calls.append((scale_folders, kwargs.get("folder_basis_mode")))
        n = len(infos)
        mat = np.random.default_rng(0).standard_normal((n, 4)).astype(np.float32)
        return SimpleNamespace(
            paths=[t["path"] for t in infos],
            mat=mat,
            mat_proj=mat + (1.0 if scale_folders else 0.0),
            comp_config="cfg",
            tsne_config="cfg|basis" if scale_folders else "cfg",
        )

    monkeypatch.setattr(b, "prepare_projection_input", fake_prepare)
    params = SimpleNamespace(
        sources=("clap",),
        source_weights={"clap": 0.0},
        tag_names=[],
        explicit_folders=[],
        required_sources=("clap",),
        feature_mask=np.ones(4, dtype=np.float32),
        features_blend=0.42,
        folder_boost=6.0,
        folder_depth_boost=3.0,
    )
    infos = [{"path": f"p{i}"} for i in range(6)]
    inputs, meta = b.prepare_inputs(
        infos, object(), params, {"approx", "none", "centroids"}
    )
    assert [c[1] for c in calls] == ["centroids", "approx", "full"]
    assert [c[0] for c in calls] == [True, True, False]
    assert set(inputs) == {"centroids", "approx", "none"}
    assert inputs["centroids"].prepare_warm is False
    assert inputs["approx"].prepare_warm is True
    assert inputs["none"].prepare_warm is True
    assert meta["basis_modes"] == ["centroids", "approx", "none"]
    assert meta["prepare_modes"]["centroids"]["prepare_warm"] is False
    assert meta["weighted_prepare_s"] is None  # no full case, no legacy SVD

    calls.clear()
    inputs, meta = b.prepare_inputs(infos, object(), params, {"full", "none"})
    # The legacy full basis is still prepared first, and only when requested.
    assert [c[1] for c in calls] == ["full", "full"]
    assert inputs["full"].prepare_warm is False
    assert inputs["full"].basis_estimate_s is not None
    assert meta["weighted_prepare_s"] is not None

    calls.clear()
    inputs, meta = b.prepare_inputs(
        infos, object(), params, {"full"}, need_unweighted_control=True
    )
    # A shared-reference run exposes the warm unweighted control even though no
    # case asked for ``none``.
    assert [c[1] for c in calls] == ["full", "full"]
    assert inputs["none"].prepare_warm is True
    assert inputs["none"].basis_mode == "none"
    assert meta["prepare_modes"]["none"]["prepare_warm"] is True


def _score_recorder(monkeypatch):
    seen = []

    def fake_recall(reference, coords, **kwargs):
        seen.append(("recall", np.asarray(reference)))
        return 0.5

    def fake_spearman(reference, coords, **kwargs):
        seen.append(("spearman", np.asarray(reference)))
        return 0.5

    monkeypatch.setattr(b, "neighbourhood_recall", fake_recall)
    monkeypatch.setattr(b, "broad_distance_spearman", fake_spearman)
    return seen


def test_shared_metric_reference_scores_every_case_same_unweighted_pca(monkeypatch):
    cases = [
        b.CaseSpec("centroids", basis_mode="centroids"),
        b.CaseSpec("approx", basis_mode="approx"),
    ]
    inputs = {
        "centroids": _fake_input(20, basis_mode="centroids"),
        "approx": _fake_input(20, basis_mode="approx"),
        "none": _fake_input(20, folders=False),
    }
    seen = _score_recorder(monkeypatch)
    results = b.run_benchmark(
        cases,
        [2],
        inputs,
        repeats=1,
        fit=lambda reduced, dims, spec: b.FitOutcome(
            coords=np.zeros((len(reduced), dims), dtype=np.float32),
            stages={"tsne_final": 0.1},
            threadpool_info=[],
            wall=0.1,
        ),
        metric_reference="unweighted",
    )
    reference = inputs["none"].reduced
    assert len(seen) == 6  # 3 metrics × 2 cases
    for _name, ref in seen:
        assert ref is reference
    assert results["cases"]["centroids"]["2"]["metric_reference"] == "unweighted"
    assert any(
        "same unweighted PCA" in note
        for note in results["cases"]["centroids"]["2"]["notes"]
    )


def test_shared_metric_reference_rejects_a_different_cohort(monkeypatch):
    cases = [b.CaseSpec("centroids", basis_mode="centroids")]
    inputs = {
        "centroids": _fake_input(20, basis_mode="centroids"),
        "none": _fake_input(19, folders=False),
    }
    with pytest.raises(ValueError, match="different track cohort"):
        b.run_benchmark(
            cases,
            [2],
            inputs,
            repeats=1,
            fit=lambda reduced, dims, spec: b.FitOutcome(
                coords=np.zeros((len(reduced), dims), dtype=np.float32),
                stages={},
                threadpool_info=[],
                wall=0.1,
            ),
            metric_reference="unweighted",
        )


def test_own_metric_reference_remains_the_default(monkeypatch):
    cases = [b.CaseSpec("centroids", basis_mode="centroids")]
    inputs = {"centroids": _fake_input(20, basis_mode="centroids")}
    seen = _score_recorder(monkeypatch)
    results = b.run_benchmark(
        cases,
        [2],
        inputs,
        repeats=1,
        fit=lambda reduced, dims, spec: b.FitOutcome(
            coords=np.zeros((len(reduced), dims), dtype=np.float32),
            stages={},
            threadpool_info=[],
            wall=0.1,
        ),
    )
    assert all(ref is inputs["centroids"].reduced for _n, ref in seen)
    assert results["cases"]["centroids"]["2"]["metric_reference"] == "own"


def test_data_driven_folders_apply_without_clap_for_genre(monkeypatch):
    import backend.embeddings.layout as layout

    class Cache:
        def __init__(self, maps):
            self._maps = maps

        def get_all(self, source, fingerprints, version=0):
            del version
            return {fp: self._maps[source][fp] for fp in fingerprints}

    fps = [f"fp{i}" for i in range(8)]
    rng = np.random.default_rng(3)
    maps = {
        "maest_logits": {fp: rng.standard_normal(6).astype(np.float32) for fp in fps},
        "maest": {fp: rng.standard_normal(4).astype(np.float32) for fp in fps},
    }
    paths = [
        f"{'alpha' if i < 4 else 'beta'}/t{i}.mp3" for i in range(8)
    ]
    infos = [{"path": p, "fingerprint": fp} for p, fp in zip(paths, fps)]
    monkeypatch.setattr(layout, "is_model_ready", lambda: False)
    inp = layout.prepare_projection_input(
        infos,
        Cache(maps),
        sources=("maest_logits", "maest"),
        source_weights={"maest_logits": 0.4, "maest": 0.6, "clap": 0.0},
        required_sources=("maest_logits", "maest"),
        scale_folders=True,
    )
    assert inp.tsne_config != inp.comp_config  # folder weighting was applied
    assert not np.array_equal(inp.mat_proj, inp.mat)


def test_perform_fit_forwards_recipe_final_budget(monkeypatch):
    seen = {}

    def fake_fit_tsne(reduced, *, dims, tsne_n_iter, tsne_early_exaggeration_iter, **kwargs):
        seen.update(
            dims=dims,
            n_iter=tsne_n_iter,
            early=tsne_early_exaggeration_iter,
            kwargs=kwargs,
        )
        return object(), np.zeros((len(reduced), dims), dtype=np.float32)

    monkeypatch.setattr(b, "_fit_tsne", fake_fit_tsne)
    spec = b.CaseSpec("native", tsne_n_iter=50, tsne_early_exaggeration_iter=0)
    out = b._perform_fit(np.zeros((8, 3), dtype=np.float32), 2, spec)
    assert out.coords.shape == (8, 2)
    assert seen["n_iter"] == 50
    assert seen["early"] == 0
    assert seen["kwargs"]["n_jobs"] == -1


# ── folder on/off comparison gallery ─────────────────────────────────────────


def _gallery_params():
    from types import SimpleNamespace

    return SimpleNamespace(
        sources=("maest_logits", "maest"),
        source_weights={"maest_logits": 0.4, "maest": 0.6, "clap": 0.0},
        required_sources=("maest_logits", "maest"),
        feature_mask=np.ones(4, dtype=np.float32),
        features_blend=0.42,
        folder_boost=6.0,
        folder_depth_boost=3.0,
        tsne_n_iter=50,
        tsne_early_exaggeration_iter=0,
    )


class _VersionOnlyCache:
    """Only the metadata read used by ``_source_versions``; no vector caches."""

    def get_meta(self, key):
        return None


def _gallery_prepared(n=8):
    rng = np.random.default_rng(7)
    paths = [
        f"{'lib' if i % 2 == 0 else 'ext'}/t{i}.mp3" for i in range(n)
    ]

    def cap(dims, fp, prepare, mode="full", exag=None):
        return b.GalleryCapture(
            coords=rng.standard_normal((n, dims)).astype(np.float32),
            paths=list(paths),
            input_fingerprint=fp,
            timings={"prepare": prepare, "pca": 0.25, "fit": 1.5},
            basis_mode=mode,
            final_exaggeration=exag,
        )

    return {
        ("native", 2): cap(2, "fp-on", 2.0),
        ("native", 3): cap(3, "fp-on", 2.0),
        ("folders-off", 2): cap(2, "fp-off", 0.5, mode="none"),
        ("folders-off", 3): cap(3, "fp-off", 0.5, mode="none"),
    }, paths


def _write_gallery(tmp_path, prepared, params=None, subdir="folder-separation", **kwargs):
    from backend.embeddings.layout_evaluation import load_recipe_json, manifest_dict

    out = tmp_path / "layout-evaluation" / subdir
    b.write_comparison_gallery(
        prepared,
        params or _gallery_params(),
        output_dir=str(out),
        feature_cache=_VersionOnlyCache(),
        seed=42,
        recipe_id="genre_full",
        **kwargs,
    )
    return out, manifest_dict(str(out)), load_recipe_json


def test_gallery_writes_paired_2d_3d_on_off(tmp_path):
    prepared, paths = _gallery_prepared()
    out, manifest, load = _write_gallery(tmp_path, prepared)

    assert sorted(p.name for p in out.iterdir()) == [
        "folder_off.json",
        "folder_off_3d.json",
        "folder_on.json",
        "folder_on_3d.json",
        "manifest.json",
    ]
    assert manifest["schema_version"] == 1
    assert manifest["reference_id"] == "folder_on"
    assert manifest["reference_ids"] == {"2": "folder_on", "3": "folder_on_3d"}
    assert list(manifest["recipes"]) == [
        "folder_on",
        "folder_off",
        "folder_on_3d",
        "folder_off_3d",
    ]
    # One control dataset fingerprint on every artifact: unweighted input.
    assert manifest["dataset_fingerprint"] == "fp-off"
    for rid, group, dims in (
        ("folder_on", "folder_on", 2),
        ("folder_off", "folder_off", 2),
        ("folder_on_3d", "folder_on", 3),
        ("folder_off_3d", "folder_off", 3),
    ):
        entry = manifest["recipes"][rid]
        assert entry["dims"] == dims and entry["group"] == group
        art = load(str(out), rid)
        assert art is not None
        assert art["dataset_fingerprint"] == "fp-off"
        assert art["dims"] == dims and art["group"] == group
        assert [p["path"] for p in art["positions"]] == paths
        # The external-root cohort is included in every artifact, not filtered.
        assert any(p["path"].startswith("ext/") for p in art["positions"])
        assert art["recipe"]["scale_folders"] is (group == "folder_on")
        if dims == 2:
            assert all("z" not in p for p in art["positions"])
        else:
            assert all(np.isfinite(p["z"]) for p in art["positions"])
    assert load(str(out), "folder_on")["input_fingerprint"] == "fp-on"
    assert load(str(out), "folder_off")["input_fingerprint"] == "fp-off"
    assert load(str(out), "folder_on")["aligned_to"] is None
    assert load(str(out), "folder_off")["aligned_to"] == "folder_on"
    assert load(str(out), "folder_on_3d")["aligned_to"] is None
    assert load(str(out), "folder_off_3d")["aligned_to"] == "folder_on_3d"
    # The on/off maps are different revisions despite sharing the dataset fp.
    assert len({manifest["recipes"][r]["revision"] for r in manifest["recipes"]}) == 4
    # Production/folder metadata and the warm-prepare estimate are labelled.
    assert manifest["folder_weighting"] == {
        "scale_folders": True,
        "folder_boost": 6.0,
        "folder_depth_boost": 3.0,
        "reference": "folder_on",
        "control": "folder_off",
    }
    assert manifest["production"]["recipe"] == "genre_full"
    assert manifest["production"]["recipe_version"] == b.PROJECTION_RECIPE_VERSION == 10
    assert manifest["production"]["tsne_n_iter"] == 50
    assert manifest["seed"] == 42
    assert manifest["fit_seed"] == b.FIT_SEED == 42
    # Shared prepare/PCA once per basis mode; fits/alignment once per artifact.
    assert manifest["timings"]["gather"] == pytest.approx(2.0 + 0.5)
    assert manifest["timings"]["pca"] == pytest.approx(0.25 + 0.25)
    assert manifest["timings"]["tsne"] == pytest.approx(1.5 * 4)
    components = {
        k: v for k, v in manifest["timings"].items() if k != "total"
    }
    assert manifest["timings"]["total"] == pytest.approx(sum(components.values()))
    # ``None`` means production, whose canonical final exaggeration is 1.25.
    assert b._effective_final_exaggeration(None) == pytest.approx(1.25)
    assert b._effective_final_exaggeration(1.0) == pytest.approx(1.0)
    cases_by_case = {c["case"]: c for c in manifest["gallery"]["cases"]}
    assert cases_by_case["native"]["final_exaggeration"] is None
    assert cases_by_case["native"]["final_exaggeration_effective"] == pytest.approx(1.25)
    assert cases_by_case["folders-off"]["final_exaggeration_effective"] == pytest.approx(1.25)
    assert load(str(out), "folder_on")["recipe"]["final_exaggeration_effective"] == pytest.approx(1.25)
    assert manifest["prepare_estimate"]["weighted_prepare_s"] == 2.0
    assert manifest["prepare_estimate"]["unweighted_prepare_warm_s"] == 0.5
    assert "warm unweighted" in manifest["prepare_estimate"]["label"]
    # First measured fit timings are what the manifest/artifact records.
    assert manifest["recipes"]["folder_on"]["timings"]["tsne"] == 1.5
    # Cohort selectors: production default all / no limit.
    assert manifest["dataset"]["scope"] == "all"
    assert manifest["dataset"]["limit"] is None


def test_gallery_cli_seed_is_separate_from_production_fit_seed(tmp_path):
    from backend.embeddings.layout_evaluation import manifest_dict

    prepared, _paths = _gallery_prepared()
    out = tmp_path / "layout-evaluation" / "folder-separation"
    b.write_comparison_gallery(
        prepared,
        _gallery_params(),
        output_dir=str(out),
        feature_cache=_VersionOnlyCache(),
        seed=7,
    )
    manifest = manifest_dict(str(out))
    assert manifest["seed"] == 7
    assert manifest["fit_seed"] == b.FIT_SEED == 42


def test_cheap_gallery_writes_six_groups_twelve_artifacts(tmp_path):
    from backend.embeddings.layout_evaluation import (
        load_recipe_json,
        manifest_dict,
    )

    n = 8
    rng = np.random.default_rng(11)
    paths = [f"lib/t{i}.mp3" for i in range(n)]
    modes = {
        "folders-off": ("none", None),
        "centroids": ("centroids", None),
        "approx": ("approx", None),
        "off-exag15": ("none", 1.5),
        "off-exag2": ("none", 2.0),
        "centroids-exag15": ("centroids", 1.5),
    }
    prepared = {}
    for case, (mode, exag) in modes.items():
        for d in (2, 3):
            prepared[(case, d)] = b.GalleryCapture(
                coords=rng.standard_normal((n, d)).astype(np.float32),
                paths=list(paths),
                input_fingerprint=f"fp-{case}",
                timings={"prepare": 1.0 if mode != "none" else 0.2,
                         "pca": 0.25, "fit": 1.5},
                basis_mode=mode,
                final_exaggeration=exag,
            )

    out = tmp_path / "layout-evaluation" / "folder-cheap"
    b.write_comparison_gallery(
        prepared,
        _gallery_params(),
        output_dir=str(out),
        feature_cache=_VersionOnlyCache(),
        cases=list(modes),
    )
    manifest = manifest_dict(str(out))
    assert manifest["status"] == "complete"
    assert len(manifest["recipes"]) == 12
    assert sorted({r["group"] for r in manifest["recipes"].values()}) == sorted(
        b.GALLERY_IDS[c] for c in modes
    )
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{b.GALLERY_IDS[c]}{suffix}.json" for c in modes for suffix in ("", "_3d")]
        + ["manifest.json"]
    )
    # The unweighted baseline is the alignment reference and shared control.
    assert manifest["reference_id"] == "folder_off"
    assert manifest["reference_ids"] == {
        "2": "folder_off",
        "3": "folder_off_3d",
    }
    assert manifest["gallery"]["reference_case"] == "folders-off"
    assert manifest["gallery"]["control_case"] == "folders-off"
    assert manifest["dataset_fingerprint"] == "fp-folders-off"
    assert manifest["gallery"]["groups"] == [
        b.GALLERY_IDS[c] for c in modes
    ]
    params_by_case = {c["case"]: c for c in manifest["gallery"]["cases"]}
    assert params_by_case["off-exag15"]["final_exaggeration"] == 1.5
    assert params_by_case["off-exag15"]["final_exaggeration_effective"] == 1.5
    assert params_by_case["centroids"]["final_exaggeration"] is None
    assert params_by_case["centroids"]["final_exaggeration_effective"] == pytest.approx(1.25)
    assert params_by_case["centroids"]["folder_basis_mode"] == "centroids"
    assert params_by_case["folders-off"]["folder_basis_mode"] == "none"
    # Every non-reference map aligns onto the baseline of its own dimension.
    for case in modes:
        case_id = b.GALLERY_IDS[case]
        for suffix, dims, ref in (
            ("", 2, "folder_off"),
            ("_3d", 3, "folder_off_3d"),
        ):
            rid = f"{case_id}{suffix}"
            art = load_recipe_json(str(out), rid)
            assert art is not None
            assert art["dims"] == dims
            assert art["group"] == case_id
            assert art["aligned_to"] == (None if case == "folders-off" else ref)
            assert art["recipe"]["folder_basis_mode"] == modes[case][0]
            assert art["recipe"]["final_exaggeration"] == modes[case][1]
            assert [p["path"] for p in art["positions"]] == paths
            if dims == 3:
                assert all(np.isfinite(p["z"]) for p in art["positions"])
    assert len({e["revision"] for e in manifest["recipes"].values()}) == 12
    # One prepare per distinct basis mode: none (shared by three cases),
    # centroids (shared by two) and approx.
    assert manifest["timings"]["gather"] == pytest.approx(0.2 + 1.0 + 1.0)
    assert manifest["prepare_estimate"]["basis_estimate_s"] is None
    assert "weighted_prepare_s" not in manifest["prepare_estimate"]

    from flask import Flask

    from backend.factory import _register_error_handlers
    from backend.routes.api_layout_evaluation import (
        create_api_layout_evaluation_blueprint,
    )

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(
        create_api_layout_evaluation_blueprint(
            str(tmp_path / "layout-evaluation")
        )
    )
    client = app.test_client()
    r = client.get("/api/layout-evaluation?set=folder-cheap")
    assert r.status_code == 200
    assert len(r.get_json()["manifest"]["recipes"]) == 12
    r = client.get("/api/layout-evaluation/centroids_exag15_3d?set=folder-cheap")
    assert r.status_code == 200
    assert r.get_json()["dims"] == 3
    assert all(np.isfinite(p["z"]) for p in r.get_json()["positions"])


def test_gallery_explicit_reference_can_move_to_folders_off(tmp_path):
    prepared, _paths = _gallery_prepared()
    out = tmp_path / "layout-evaluation" / "folder-separation"
    b.write_comparison_gallery(
        prepared,
        _gallery_params(),
        output_dir=str(out),
        feature_cache=_VersionOnlyCache(),
        reference="folders-off",
    )
    from backend.embeddings.layout_evaluation import load_recipe_json

    assert load_recipe_json(str(out), "folder_on")["aligned_to"] == "folder_off"
    assert load_recipe_json(str(out), "folder_off")["aligned_to"] is None
    from backend.embeddings.layout_evaluation import manifest_dict

    manifest = manifest_dict(str(out))
    assert manifest["folder_weighting"]["reference"] == "folder_off"
    assert manifest["folder_weighting"]["control"] == "folder_off"
    # Native + unweighted are distinct preparations, even when the reference is
    # the control; the gather total must not count the control twice.
    assert manifest["timings"]["gather"] == pytest.approx(2.0 + 0.5)
    assert manifest["prepare_estimate"]["basis_estimate_s"] == pytest.approx(1.5)
    assert manifest["prepare_estimate"]["weighted_prepare_s"] == pytest.approx(2.0)


def test_gallery_records_cohort_scope_and_limit(tmp_path):
    prepared, _paths = _gallery_prepared()
    _out, manifest, _load = _write_gallery(
        tmp_path, prepared, scope="library", limit=123
    )
    assert manifest["dataset"]["scope"] == "library"
    assert manifest["dataset"]["limit"] == 123


def test_gallery_writes_empty_in_progress_manifest_before_artifacts(
    tmp_path, monkeypatch
):
    real_write = b._atomic_write_json
    interrupted = False

    def fail_on_off(path, obj):
        nonlocal interrupted
        if path.endswith("folder_off.json"):
            interrupted = True
            raise RuntimeError("simulated interruption")
        real_write(path, obj)

    monkeypatch.setattr(b, "_atomic_write_json", fail_on_off)
    prepared, _paths = _gallery_prepared()
    out = tmp_path / "layout-evaluation" / "folder-separation"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        b.write_comparison_gallery(
            prepared,
            _gallery_params(),
            output_dir=str(out),
            feature_cache=_VersionOnlyCache(),
            seed=42,
        )
    assert interrupted
    manifest = json.loads((out / "manifest.json").read_text())
    # The live manifest never lists the partially-written run's recipes.
    assert manifest["status"] == "in_progress"
    assert manifest["recipes"] == {}
    assert manifest["dataset"]["scope"] == "all"
    assert manifest["dataset"]["limit"] is None


def test_gallery_rejects_divergent_cohorts(tmp_path):
    prepared, _paths = _gallery_prepared()
    prepared[("folders-off", 3)] = b.GalleryCapture(
        coords=prepared[("folders-off", 3)].coords,
        paths=prepared[("folders-off", 3)].paths[:-1],
        input_fingerprint="fp-off",
        timings=prepared[("folders-off", 3)].timings,
    )
    with pytest.raises(ValueError, match="diverged"):
        _write_gallery(tmp_path, prepared)
    assert not (tmp_path / "layout-evaluation" / "folder-separation").exists() or not list(
        (tmp_path / "layout-evaluation" / "folder-separation").iterdir()
    )


def test_gallery_requires_all_four_case_dimension_fits(tmp_path):
    prepared, _paths = _gallery_prepared()
    del prepared[("folders-off", 3)]
    with pytest.raises(ValueError, match="missing"):
        _write_gallery(tmp_path, prepared)


def test_gallery_writer_does_not_gather_fit_or_read_vector_caches(tmp_path, monkeypatch):
    import backend.embeddings.layout as layout

    def boom(*_args, **_kwargs):
        raise AssertionError("gallery serialisation must not gather or fit")

    monkeypatch.setattr(layout, "prepare_projection_input", boom)
    monkeypatch.setattr(b, "_fit_tsne", boom)
    prepared, _paths = _gallery_prepared()

    class Guarded:
        def get_meta(self, _key):
            return None

        def __getattr__(self, name):
            raise AssertionError(f"gallery read a vector cache ({name})")

    out = tmp_path / "layout-evaluation" / "folder-separation"
    b.write_comparison_gallery(
        prepared,
        _gallery_params(),
        output_dir=str(out),
        feature_cache=Guarded(),
        seed=42,
    )
    # Only the four artifacts + manifest under the requested gallery dir.
    assert sorted(p.name for p in out.iterdir()) == [
        "folder_off.json",
        "folder_off_3d.json",
        "folder_on.json",
        "folder_on_3d.json",
        "manifest.json",
    ]
    assert sorted(p.name for p in (tmp_path / "layout-evaluation").iterdir()) == [
        "folder-separation"
    ]


def test_gallery_preflight_rejects_invalid_combinations():
    native = b.CaseSpec("native")
    off = b.CaseSpec("folders-off", basis_mode="none")
    b._validate_gallery_request([native, off], [2, 3], None)
    b._validate_gallery_request(
        [b.CaseSpec(name) for name in b.FOLDER_CHEAP_CASES], [2, 3], None
    )
    with pytest.raises(ValueError, match="at least two"):
        b._validate_gallery_request([native], [2, 3], None)
    with pytest.raises(ValueError, match="does not know"):
        b._validate_gallery_request(
            [native, b.CaseSpec("threads4", threads=4)], [2, 3], None
        )
    with pytest.raises(ValueError, match="requires --dims 2,3"):
        b._validate_gallery_request([native, off], [2], None)
    with pytest.raises(ValueError, match="--sizes is not supported"):
        b._validate_gallery_request([native, off], [2, 3], "100,200")


def test_gallery_route_reads_named_set_and_rejects_missing_z(tmp_path):
    from flask import Flask

    from backend.embeddings.layout_evaluation import manifest_dict
    from backend.factory import _register_error_handlers
    from backend.routes.api_layout_evaluation import (
        create_api_layout_evaluation_blueprint,
    )

    prepared, _paths = _gallery_prepared()
    out, _manifest, _load = _write_gallery(tmp_path, prepared)
    base = tmp_path / "layout-evaluation"

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(create_api_layout_evaluation_blueprint(str(base)))
    client = app.test_client()

    r = client.get("/api/layout-evaluation?set=folder-separation")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == [
        "folder_off",
        "folder_off_3d",
        "folder_on",
        "folder_on_3d",
    ]
    r = client.get("/api/layout-evaluation/folder_on_3d?set=folder-separation")
    assert r.status_code == 200
    assert r.get_json()["dims"] == 3
    # Not served from the base set: the named set is fixed, not a fallback.
    assert client.get("/api/layout-evaluation/folder_on_3d").status_code == 404

    # A group/dims mismatch is rejected by the loader, not just by the client.
    path = out / "folder_off.json"
    data = json.loads(path.read_text())
    data["group"] = "folder_on"
    path.write_text(json.dumps(data))
    assert _load(str(out), "folder_off") is None

    path = out / "folder_on_3d.json"
    data = json.loads(path.read_text())
    for pos in data["positions"]:
        pos.pop("z")
    path.write_text(json.dumps(data))
    assert _load(str(out), "folder_on_3d") is None
    # The atomic write last means the manifest was still a complete schema-1 doc.
    assert manifest_dict(str(out))["schema_version"] == 1


def test_run_benchmark_captures_first_fit_without_an_extra_fit():
    cases = [b.CaseSpec("native"), b.CaseSpec("folders-off", basis_mode="none")]
    inputs = {
        "full": _fake_input(20),
        "none": _fake_input(20, folders=False),
    }
    calls: list[tuple[str, int]] = []
    captured: dict[tuple[str, int], b.GalleryCapture] = {}

    def fake_fit(reduced, dims, spec):
        calls.append((spec.name, dims))
        return b.FitOutcome(
            coords=np.full((len(reduced), dims), float(len(calls)), dtype=np.float32),
            stages={"tsne_final": 0.2},
            threadpool_info=[],
            wall=0.2,
        )

    def capture(name, dim, coords, inp, outcome, spec):
        captured[(name, dim)] = b.GalleryCapture(
            coords=np.array(coords, copy=True),
            paths=list(inp.paths),
            input_fingerprint=inp.fingerprint,
            timings={"prepare": inp.prepare_s, "pca": inp.pca_s, "fit": outcome.wall},
            basis_mode=spec.basis_mode,
            final_exaggeration=spec.final_exaggeration,
        )

    b.run_benchmark(
        cases,
        [2, 3],
        inputs,
        repeats=2,
        fit=fake_fit,
        score=False,
        on_first_fit=capture,
    )
    assert len(calls) == 4 * 3  # warmup + 2 measured rounds, no extra fit
    assert set(captured) == {
        ("native", 2),
        ("native", 3),
        ("folders-off", 2),
        ("folders-off", 3),
    }
    # Captured coords are the first measured fit (round 0 after warmup), copied.
    assert sorted(c.coords[0, 0] for c in captured.values()) == [5.0, 6.0, 7.0, 8.0]
