"""Unit tests for embedding-space projection helpers (composite vectors,
folder hierarchy, parsers).  Avoids CLAP model load.
"""

import numpy as np
import pytest

import backend.embeddings.layout as layout
import backend.embeddings.lens_gauge as lens_gauge
import backend.embeddings.semantic_weighting as semantic_weighting
from backend.embeddings.layout import (
    _gather_composite_vecs,
    _get_prepared_source_row,
    compute_projection,
    parse_audio_feature_mask,
    parse_features_blend,
    parse_folder_boost,
    parse_folder_depth_boost,
)
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.semantic_weighting import (
    _apply_low_rank_semantic_weight,
    _build_folder_directions,
    _expand_folder_nodes,
    _folder_orthogonal_pc_directions,
    _folder_seeds_from_track_paths,
    _semantic_basis_param_key,
)
from backend.embeddings.librosa_audio_features import AUDIO_FEATURE_DIM


class _StubFeatureCache:
    """Minimal cache implementing the generic composite gather lookup."""

    def __init__(
        self,
        clap: dict[str, np.ndarray] | None = None,
        feats: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._maps = {
            "clap": clap or {},
            "features": feats or {},
        }

    def get_all(
        self, source: str, fingerprints: list[str], version: int = 0,
    ) -> dict[str, np.ndarray]:
        del version
        m = self._maps.get(source, {})
        return {fp: m[fp] for fp in fingerprints if fp in m}


def test_folder_seeds_and_expand_align_with_explicit():
    paths = ["alpha/beta/t1.mp3", "alpha/gamma/t2.mp3", "other/delta/x.mp3"]
    derived = _folder_seeds_from_track_paths(paths)
    explicit = ["alpha/beta", "alpha/gamma", "other/delta"]
    assert set(derived) == set(explicit)
    assert _expand_folder_nodes(derived) == _expand_folder_nodes(explicit)


def test_parse_audio_feature_mask():
    m = parse_audio_feature_mask("101011")
    assert m.shape == (AUDIO_FEATURE_DIM,)
    assert float(m[0]) == 1.0 and float(m[1]) == 0.0 and float(m[5]) == 1.0
    full = parse_audio_feature_mask(None)
    assert np.allclose(full, 1.0)


def test_parse_numeric_helpers_clip():
    assert parse_features_blend("bad", 0.42) == 0.42
    assert parse_features_blend("2.0", 0.42) == 1.0
    assert parse_folder_boost("99", 3.0) == 12.0
    assert parse_folder_depth_boost("0.5", 1.5) == 1.0


def test_prepared_source_row_cache_hits_on_repeat():
    layout._source_row_cache_clear()
    raw = np.array([3.0, 4.0], dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)
    a = _get_prepared_source_row("clap", "fp-row", raw, mask, 0.42)
    b = _get_prepared_source_row("clap", "fp-row", raw, mask, 0.42)
    assert a is b


# ── prepared-row cache byte accounting ───────────────────────────────


def _row_cost(v: np.ndarray) -> int:
    return int(v.nbytes) + layout._SOURCE_ROW_ENTRY_OVERHEAD


def test_source_row_cache_accounts_bytes_after_puts():
    layout._source_row_cache_clear()
    layout._source_row_cache_put(("clap", "fp1", 1), np.ones(32, dtype=np.float32))
    assert layout._source_row_cache_bytes == _row_cost(np.ones(32, dtype=np.float32))
    layout._source_row_cache_put(("clap", "fp2", 1), np.ones(64, dtype=np.float32))
    assert layout._source_row_cache_bytes == _row_cost(
        np.ones(32, dtype=np.float32)
    ) + _row_cost(np.ones(64, dtype=np.float32))


def test_source_row_cache_overwrite_does_not_double_count():
    layout._source_row_cache_clear()
    key = ("clap", "fp1", 1)
    layout._source_row_cache_put(key, np.ones(32, dtype=np.float32))
    layout._source_row_cache_put(key, np.ones(64, dtype=np.float32))
    assert len(layout._source_row_cache) == 1
    assert layout._source_row_cache_bytes == _row_cost(np.ones(64, dtype=np.float32))


def test_source_row_cache_evicts_lru_by_bytes(monkeypatch):
    layout._source_row_cache_clear()
    # Budget sized for exactly two entries: the third put evicts the oldest.
    budget = 2 * _row_cost(np.ones(64, dtype=np.float32))
    monkeypatch.setattr(layout, "_SOURCE_ROW_CACHE_BYTES_MAX", budget)
    layout._source_row_cache_put(("a", "fp", 1), np.ones(32, dtype=np.float32))
    layout._source_row_cache_put(("b", "fp", 1), np.ones(64, dtype=np.float32))
    layout._source_row_cache_put(("c", "fp", 1), np.ones(64, dtype=np.float32))
    assert set(layout._source_row_cache) == {("b", "fp", 1), ("c", "fp", 1)}
    assert layout._source_row_cache_bytes == budget


def test_source_row_cache_get_refreshes_recency(monkeypatch):
    layout._source_row_cache_clear()
    budget = 2 * _row_cost(np.ones(64, dtype=np.float32))
    monkeypatch.setattr(layout, "_SOURCE_ROW_CACHE_BYTES_MAX", budget)
    k_a = ("a", "fp", 1)
    k_b = ("b", "fp", 1)
    layout._source_row_cache_put(k_a, np.ones(64, dtype=np.float32))
    layout._source_row_cache_put(k_b, np.ones(32, dtype=np.float32))
    # Touching A demotes B to least-recently-used, so the next put evicts B.
    assert layout._source_row_cache_get(k_a) is not None
    layout._source_row_cache_put(("c", "fp", 1), np.ones(64, dtype=np.float32))
    assert k_a in layout._source_row_cache
    assert k_b not in layout._source_row_cache


def test_source_row_cache_clear_resets_bytes():
    layout._source_row_cache_put(("clap", "fp1", 1), np.ones(32, dtype=np.float32))
    assert layout._source_row_cache_bytes > 0
    layout._source_row_cache_clear()
    assert len(layout._source_row_cache) == 0
    assert layout._source_row_cache_bytes == 0


# ── per-source version snapshot (the cold-regather overhead fix) ───────


class _CountingSupervisedCache:
    """Stub cache with supervised rows plus a count of ``get_meta`` reads.

    Mirrors the real FeatureCache surface used by ``supervised.cache_version``
    and the composite gather's generic ``get_all`` lookup.

    The supervised rows are *version-aware*: the map ``get_all`` returns depends
    on the requested version (the training-run id), so a retrain
    (``set_run_id``) yields observably different rows the moment the version
    bumps — it is never the same stale map under a new key.
    """

    def __init__(self, run_id: int = 5) -> None:
        self._run_id = run_id
        self.get_meta_count = 0
        self._maps: dict[str, dict[str, np.ndarray]] = {}

    def set_run_id(self, run_id: int) -> None:
        self._run_id = run_id

    def get_meta(self, key: str):
        self.get_meta_count += 1
        return self._run_id if key == "supervised_run_id" else None

    def _supervised_map(self, version: int) -> dict[str, np.ndarray]:
        # Each training run writes its own supervised rows: shift the cached raw
        # rows by the run id so a version bump changes the *direction*, not just
        # a scale (a scalar scale would survive row normalisation).
        run = self._maps.get("supervised", {})
        shift = float(version)
        return {fp: v + shift for fp, v in run.items()}

    def get_all(
        self, source: str, fingerprints: list[str], version: int = 0,
    ) -> dict[str, np.ndarray]:
        if source == "supervised":
            m = self._supervised_map(version)
            return {fp: m[fp] for fp in fingerprints if fp in m}
        m = self._maps.get(source, {})
        return {fp: m[fp] for fp in fingerprints if fp in m}


def test_source_version_static_branch_makes_no_metadata_read():
    cache = _CountingSupervisedCache(run_id=9)
    assert layout._source_version("clap", cache) == layout.EMBEDDING_VERSION
    assert layout._source_version("features", cache) == layout._static_source_version(
        "features"
    )
    assert cache.get_meta_count == 0
    # The supervised branch is the only one that reads metadata.
    layout._source_version("supervised", cache)
    assert cache.get_meta_count == 1


def test_gather_reads_supervised_version_once_not_per_row():
    """A many-row mixed-source gather must snapshot the supervised run id exactly
    once: one metadata read regardless of row count."""
    layout._composite_cache.clear()
    n = 200
    cache = _CountingSupervisedCache(run_id=7)
    clap, maest, sup, infos = {}, {}, {}, []
    for i in range(n):
        fp = f"f{i}"
        clap[fp] = np.array([1.0, float(i)], dtype=np.float32)
        maest[fp] = np.ones(4, dtype=np.float32)
        sup[fp] = np.array([0.0, 1.0], dtype=np.float32)
        infos.append({"path": f"p/t{i}.mp3", "fingerprint": fp})
    cache._maps = {"clap": clap, "maest": maest, "supervised": sup}
    cache.get_meta_count = 0

    _, m, _, _, _, _, _ = _gather_composite_vecs(
        infos, cache, sources=("clap", "maest", "supervised"),
    )
    assert m.shape[0] == n
    assert cache.get_meta_count == 1

    # A second gather re-reads the (now possibly changed) version: the version is
    # never permanently cached, so a retrain invalidates the next gather.
    _gather_composite_vecs(infos, cache, sources=("clap", "maest", "supervised"))
    assert cache.get_meta_count == 2


def test_gather_invalidates_composite_cache_on_retrain():
    """A retrain changes the supervised version, so cache keys change and the
    composite matrix is recomputed — and holds the *new* supervised rows, not a
    stale reuse of the pre-retrain values."""
    layout._composite_cache.clear()
    layout._source_row_cache_clear()
    cache = _CountingSupervisedCache(run_id=1)
    fp = "fp"
    clap = np.array([3.0, 4.0], dtype=np.float32)
    sup = np.array([1.0, 2.0], dtype=np.float32)
    infos = [{"path": "a/t.mp3", "fingerprint": fp}]
    cache._maps = {"clap": {fp: clap}, "supervised": {fp: sup}}

    def gather():
        return _gather_composite_vecs(
            infos, cache, sources=("clap", "supervised"),
        )

    _, m1, _, _, k1, _, _ = gather()
    # Same version again → composite cache hit, identical matrix object.
    _, m2, _, _, k2, _, _ = gather()
    assert k1 == k2
    assert m1 is m2

    # Retrain → version bumps → key and matrix change, no stale reuse.
    cache.set_run_id(2)
    _, m3, _, _, k3, _, _ = gather()
    assert k3 != k1
    assert m3 is not m1

    # The recomputed matrix holds the *new* supervised rows (version 2), not the
    # stale version-1 values from before the retrain. The composite row is
    # re-derived from the freshly gathered map and matches the expected new
    # vector exactly, while the pre-retrain matrix holds a different direction.
    new_sup = (sup + 2.0) / np.linalg.norm(sup + 2.0)
    stale_sup = (sup + 1.0) / np.linalg.norm(sup + 1.0)
    assert np.allclose(m3[0][2:], new_sup, atol=1e-6)      # supervised block = new values
    assert not np.allclose(m3[0][2:], stale_sup)           # ...not the stale ones
    expected_row = np.concatenate([clap / np.linalg.norm(clap), new_sup])
    assert np.allclose(m3[0], expected_row, atol=1e-6)
    # The matrix norm is *not* stale either: re-fit from the new rows reproduces
    # the recomputed norm, proving it was rebuilt rather than reused.
    assert abs(float(np.linalg.norm(m3)) - float(np.linalg.norm(expected_row))) < 1e-6

    # And the new version is now the cached one.
    _, m4, _, _, k4, _, _ = gather()
    assert k3 == k4
    assert m3 is m4


def test_prepared_supervised_row_explicit_zero_version_bypasses_meta():
    """``source_version=0`` is a real version (no trained head), so a direct
    prepared-row call must use it as-is without a metadata read."""
    layout._source_row_cache_clear()
    cache = _CountingSupervisedCache(run_id=3)
    raw = np.array([1.0, 2.0], dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)
    v = _get_prepared_source_row(
        "supervised", "fp-zero", raw, mask, 0.42, cache, source_version=0,
    )
    assert cache.get_meta_count == 0
    assert np.allclose(v, raw / np.linalg.norm(raw))


def test_fetch_source_map_explicit_zero_version_is_not_reresolved():
    """``source_version=0`` is a real version (no trained head), not the None
    sentinel — it must be passed straight to the getter with no metadata read."""
    cache = _CountingSupervisedCache(run_id=3)
    seen: list[int] = []

    def _get_all(source, fingerprints, version=0):
        del source, fingerprints
        seen.append(version)
        return {}

    cache.get_all = _get_all
    layout._fetch_source_map("supervised", cache, ["fp"], source_version=0)
    assert seen == [0]
    assert cache.get_meta_count == 0


def test_fetch_source_map_unknown_source_raises():
    cache = _StubFeatureCache()
    with pytest.raises(ValueError):
        layout._fetch_source_map("bogus", cache, [])


def test_gather_unknown_source_raises():
    cache = _StubFeatureCache()
    with pytest.raises(ValueError):
        _gather_composite_vecs([], cache, sources=("bogus",))


def test_prepared_source_row_exact_normalisation():
    layout._source_row_cache_clear()
    raw = np.array([3.0, 4.0], dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)
    v = _get_prepared_source_row("clap", "fp-norm", raw, mask, 0.42)
    assert np.allclose(v, np.array([0.6, 0.8], dtype=np.float32))
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_gather_zero_rows_for_missing_optional_source():
    """An optional source a track lacks contributes a zero row; partial libraries
    still render instead of hiding the track until every model has run."""
    layout._composite_cache.clear()
    fp_present = "present"
    fp_missing = "missing"
    clap = np.array([3.0, 4.0], dtype=np.float32)
    feat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    infos = [
        {"path": "a/t_present.mp3", "fingerprint": fp_present},
        {"path": "b/t_missing.mp3", "fingerprint": fp_missing},
    ]
    cache = _StubFeatureCache(
        clap={fp_present: clap, fp_missing: clap}, feats={fp_present: feat}
    )
    _, mat, _, _, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        required_sources=("clap",),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=1.0,
    )
    assert mat.shape == (2, 2 + AUDIO_FEATURE_DIM)
    assert not np.all(mat[0, 2:] == 0.0)   # present track has a features row
    assert np.all(mat[1, 2:] == 0.0)       # missing track contributes a zero row


def test_composite_tier_cache_reuses_matrix():
    layout._composite_cache.clear()
    fp = "fpz"
    clap = np.array([1.0, 0.0], dtype=np.float32)
    feat = np.ones(6, dtype=np.float32)
    infos = [{"path": "a/z.mp3", "fingerprint": fp}]
    cache = _StubFeatureCache(clap={fp: clap}, feats={fp: feat})
    _, m1, _, _, k1, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=0.5,
    )
    _, m2, _, _, k2, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=0.5,
    )
    assert k1 == k2
    assert m1 is m2


def test_gather_composite_masks_and_blend():
    fp = "fp1"
    clap = np.array([3.0, 4.0], dtype=np.float32)
    feat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    infos = [{"path": "a/t.mp3", "fingerprint": fp}]
    cache = _StubFeatureCache(clap={fp: clap}, feats={fp: feat})
    mask = parse_audio_feature_mask("100000")

    _, mat_mask, _, _, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=mask,
        features_blend=1.0,
    )
    _, mat_no_mask, _, _, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=1.0,
    )
    assert mat_mask.shape == mat_no_mask.shape == (1, 8)
    assert not np.allclose(mat_mask, mat_no_mask)

    _, mat_weak, _, _, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=0.2,
    )
    _, mat_full, _, _, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=1.0,
    )
    assert np.linalg.norm(mat_weak[0, 2:]) < np.linalg.norm(mat_full[0, 2:])


def test_low_rank_semantic_weight_responds_to_depth_sliders():
    rng = np.random.default_rng(7)
    n, d, k = 24, 10, 5
    mat = rng.standard_normal((n, d)).astype(np.float32)
    T_n = rng.standard_normal((k, d)).astype(np.float32)
    T_n = T_n / np.linalg.norm(T_n, axis=1, keepdims=True)
    exp = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    o1 = _apply_low_rank_semantic_weight(mat, T_n, exp, 1.0, 1.1)
    o2 = _apply_low_rank_semantic_weight(mat, T_n, exp, 1.0, 2.4)
    assert not np.allclose(o1, o2)


def test_low_rank_boost_2_4_is_uniform_40_percent_of_6_across_depths():
    rng = np.random.default_rng(21)
    n, d, k = 24, 10, 6
    mat = rng.standard_normal((n, d)).astype(np.float32)
    T_n = rng.standard_normal((k, d)).astype(np.float32)
    T_n = T_n / np.linalg.norm(T_n, axis=1, keepdims=True)
    exp = np.array([0, 1, 2, 3, 0, 2], dtype=np.int32)
    old = _apply_low_rank_semantic_weight(mat, T_n, exp, 6.0, 3.0)
    new = _apply_low_rank_semantic_weight(mat, T_n, exp, 2.4, 3.0)
    expected = mat + 0.4 * (old - mat)
    np.testing.assert_allclose(new, expected, rtol=1e-5, atol=1e-6)


def test_low_rank_matches_dense_W():
    rng = np.random.default_rng(2)
    n, d, k = 12, 6, 4
    mat = rng.standard_normal((n, d)).astype(np.float32)
    T_n = rng.standard_normal((k, d)).astype(np.float32)
    T_n = T_n / np.linalg.norm(T_n, axis=1, keepdims=True)
    exp = np.array([0, 1, 2, 1], dtype=np.int32)
    boost, depth_boost = 1.35, 1.8
    w = (boost * (depth_boost ** exp.astype(np.float64))).astype(np.float32)
    d_mat = np.eye(d, dtype=np.float32) + T_n.T @ (w[:, None] * T_n)
    dense = (mat @ d_mat).astype(np.float32)
    lr = _apply_low_rank_semantic_weight(mat, T_n, exp, boost, depth_boost)
    assert np.allclose(dense, lr, rtol=1e-4, atol=1e-5)


def test_build_folder_directions_returns_exponents():
    d = 6
    n = 8
    mat = np.zeros((n, d), dtype=np.float32)
    mat[:4, 0] = 1.0
    mat[4:, 0] = -1.0
    paths = [f"p/a/b/c1/t{i}.mp3" for i in range(4)]
    paths += [f"p/a/b/c2/t{i}.mp3" for i in range(4, 8)]
    seeds = _folder_seeds_from_track_paths(paths)
    dirs, exps, _fb = _build_folder_directions(mat, paths, seeds)
    assert dirs is not None and exps is not None
    assert len(exps) == len(dirs)
    assert exps.dtype == np.int32
    assert np.all(exps >= 0)


def test_build_folder_directions_adds_orthogonal_pca_for_large_folders():
    """Residual PCA adds rows beyond hierarchical when a folder has ≥8 tracks."""
    d = 4
    paths = [f"m/gen/folk/t{i}.mp3" for i in range(8)]
    paths += [f"m/gen/rock/t{i}.mp3" for i in range(4)]
    mat = np.zeros((12, d), dtype=np.float32)
    mat[:4, 0] = 10.0
    mat[4:8, 1] = 10.0
    mat[8:, 2] = 8.0
    seeds = _folder_seeds_from_track_paths(paths)
    dirs, exps, _fb = _build_folder_directions(mat, paths, seeds)
    assert dirs is not None and exps is not None
    assert len(exps) == len(dirs)
    hier_only = _build_folder_directions(
        mat, paths, seeds, min_tracks_pca=10_000,
    )
    assert hier_only[0] is not None
    assert dirs.shape[0] > hier_only[0].shape[0]
    hier_folk = dirs[0]
    folk_pc = dirs[1]
    assert abs(float(np.dot(folk_pc, hier_folk))) < 0.05


def test_centroids_basis_mode_skips_pc_svd_and_keeps_hierarchy(monkeypatch):
    d = 4
    paths = [f"m/gen/folk/t{i}.mp3" for i in range(8)]
    paths += [f"m/gen/rock/t{i}.mp3" for i in range(4)]
    mat = np.zeros((12, d), dtype=np.float32)
    mat[:4, 0] = 10.0
    mat[4:8, 1] = 10.0
    mat[8:, 2] = 8.0
    seeds = _folder_seeds_from_track_paths(paths)

    def _fail_svd(*_args, **_kwargs):
        raise AssertionError("centroids mode must not run the within-folder SVD")

    monkeypatch.setattr(semantic_weighting, "svd_singular", _fail_svd)
    dirs, exps, _fb = _build_folder_directions(
        mat, paths, seeds, basis_mode="centroids",
    )
    hier = _build_folder_directions(mat, paths, seeds, max_pca_components=0)
    assert dirs is not None and exps is not None
    np.testing.assert_array_equal(dirs, hier[0])
    np.testing.assert_array_equal(exps, hier[1])


def test_full_basis_mode_is_the_default():
    d = 4
    paths = [f"m/gen/folk/t{i}.mp3" for i in range(8)]
    paths += [f"m/gen/rock/t{i}.mp3" for i in range(4)]
    mat = np.zeros((12, d), dtype=np.float32)
    mat[:4, 0] = 10.0
    mat[4:8, 1] = 10.0
    mat[8:, 2] = 8.0
    seeds = _folder_seeds_from_track_paths(paths)
    implicit = _build_folder_directions(mat, paths, seeds)
    explicit = _build_folder_directions(mat, paths, seeds, basis_mode="full")
    np.testing.assert_array_equal(implicit[0], explicit[0])
    np.testing.assert_array_equal(implicit[1], explicit[1])


def test_approx_pc_directions_match_full_subspace_for_clear_spectral_gap():
    d = 6
    n = 8
    block = np.zeros((n, d), dtype=np.float32)
    block[:, 1] = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32) * 10.0
    block[:, 2] = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32) * 3.0
    d_unit = np.zeros(d, dtype=np.float32)
    d_unit[0] = 1.0

    full = _folder_orthogonal_pc_directions(
        block, d_unit, max_components=2, min_second_sv_ratio=0.18,
        basis_mode="full",
    )
    approx = _folder_orthogonal_pc_directions(
        block, d_unit, max_components=2, min_second_sv_ratio=0.18,
        basis_mode="approx",
    )
    assert len(full) == len(approx) == 2
    for v in approx:
        assert v.shape == (d,)
        assert np.isfinite(v).all()
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
        assert abs(float(v @ d_unit)) < 1e-5
    # Clear spectral gap: the seeded randomized solver recovers the same
    # 2-D subspace, not necessarily byte-identical vectors.
    cos = np.abs(np.asarray(approx) @ np.asarray(full).T)
    assert np.all(cos.max(axis=1) > 0.999)


def test_semantic_basis_key_namespaces_only_nondefault_modes():
    full = _semantic_basis_param_key(None, ["a/b"], "full")
    assert full == _semantic_basis_param_key(None, ["a/b"])
    assert len({
        full,
        _semantic_basis_param_key(None, ["a/b"], "centroids"),
        _semantic_basis_param_key(None, ["a/b"], "approx"),
    }) == 3


def test_build_folder_directions_rejects_unknown_basis_mode():
    with pytest.raises(ValueError, match="unknown folder basis mode"):
        _build_folder_directions(
            np.zeros((4, 3), dtype=np.float32),
            ["a/t1.mp3", "a/t2.mp3", "b/t3.mp3", "b/t4.mp3"],
            ["a", "b"],
            basis_mode="bogus",
        )


def test_prepare_projection_input_namespaces_basis_mode_and_validates(monkeypatch):
    layout._semantic_basis_cache.clear()
    monkeypatch.setattr(layout, "is_model_ready", lambda: True)
    vecs = {
        "f0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "f1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "f2": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "f3": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    paths = ["a/t0.mp3", "a/t1.mp3", "b/t2.mp3", "b/t3.mp3"]
    infos = [{"path": p, "fingerprint": f"f{i}"} for i, p in enumerate(paths)]
    cache = _StubFeatureCache(clap=vecs)

    def _prep(mode=None):
        kwargs = {"scale_folders": True, "sources": ("clap",)}
        if mode is not None:
            kwargs["folder_basis_mode"] = mode
        return layout.prepare_projection_input(infos, cache, **kwargs)

    implicit = _prep()
    full = _prep("full")
    centroids = _prep("centroids")
    approx = _prep("approx")
    assert implicit.tsne_config == full.tsne_config
    assert "fbmcentroids" in centroids.tsne_config
    assert "fbmapprox" in approx.tsne_config
    assert len({implicit.tsne_config, centroids.tsne_config,
                approx.tsne_config}) == 3
    assert implicit.comp_config == centroids.comp_config == approx.comp_config
    assert len(layout._semantic_basis_cache) == 3

    with pytest.raises(ValueError, match="unknown folder basis mode"):
        _prep("bogus")


def test_prepare_projection_input_applies_folder_weighting_without_clap_or_model(monkeypatch):
    """A no-CLAP recipe needs no text model for data-driven folder directions."""
    layout._semantic_basis_cache.clear()
    monkeypatch.setattr(layout, "is_model_ready", lambda: False)
    rng = np.random.default_rng(8)
    feats = {f"f{i}": rng.standard_normal(6).astype(np.float32) for i in range(8)}
    paths = [
        f"{'alpha' if i < 4 else 'beta'}/t{i}.mp3" for i in range(8)
    ]
    infos = [{"path": p, "fingerprint": f"f{i}"} for i, p in enumerate(paths)]
    inp = layout.prepare_projection_input(
        infos,
        _StubFeatureCache(feats=feats),
        sources=("features",),
        feature_mask=parse_audio_feature_mask("111111"),
        required_sources=("features",),
        scale_folders=True,
    )
    assert inp.tsne_config != inp.comp_config  # folder weighting was applied
    assert not np.array_equal(inp.mat_proj, inp.mat)


def test_prepare_projection_input_requires_model_for_clap_or_tags(monkeypatch):
    """A CLAP column or explicit tags still need the model for text directions."""
    layout._semantic_basis_cache.clear()
    monkeypatch.setattr(layout, "is_model_ready", lambda: False)
    rng = np.random.default_rng(10)
    clap = {f"f{i}": rng.standard_normal(4).astype(np.float32) for i in range(8)}
    paths = [
        f"{'alpha' if i < 4 else 'beta'}/t{i}.mp3" for i in range(8)
    ]
    infos = [{"path": p, "fingerprint": f"f{i}"} for i, p in enumerate(paths)]
    cache = _StubFeatureCache(clap=clap)
    with_clap = layout.prepare_projection_input(
        infos, cache, sources=("clap",), scale_folders=True,
    )
    assert with_clap.tsne_config == with_clap.comp_config

    feats = {f"f{i}": rng.standard_normal(6).astype(np.float32) for i in range(8)}
    tagged = layout.prepare_projection_input(
        infos, _StubFeatureCache(feats=feats), sources=("features",),
        feature_mask=parse_audio_feature_mask("111111"),
        required_sources=("features",), scale_folders=True,
        context_tags=["techno"],
    )
    assert tagged.tsne_config == tagged.comp_config


def test_compute_projection_uses_centroid_basis_and_canonical_boost(monkeypatch):
    """Production compute selects the centroid basis, canonical boost and no PC SVD."""
    from backend.embeddings import semantic_weighting
    from backend.embeddings.projection_config import (
        FROZEN_FOLDER_BASIS_MODE,
        FROZEN_FOLDER_BOOST,
        FROZEN_FOLDER_DEPTH_BOOST,
    )

    layout._semantic_basis_cache.clear()
    monkeypatch.setattr(layout, "is_model_ready", lambda: False)

    def _pc_svd_must_not_run(*_args, **_kwargs):
        raise AssertionError("production centroid basis must skip within-folder SVD")

    monkeypatch.setattr(
        semantic_weighting, "_folder_orthogonal_pc_directions",
        _pc_svd_must_not_run,
    )
    rng = np.random.default_rng(9)
    feats = {f"f{i}": rng.standard_normal(6).astype(np.float32) for i in range(8)}
    paths = [
        f"{'alpha' if i < 4 else 'beta'}/t{i}.mp3" for i in range(8)
    ]
    infos = [{"path": p, "fingerprint": f"f{i}"} for i, p in enumerate(paths)]
    cache = _StubFeatureCache(feats=feats)
    seen: dict = {}
    real_prepare = layout.prepare_projection_input

    def spy(*args, **kwargs):
        seen.update(kwargs)
        inp = real_prepare(*args, **kwargs)
        seen["inp"] = inp
        return inp

    monkeypatch.setattr(layout, "prepare_projection_input", spy)
    out = compute_projection(
        infos,
        cache,
        method="pca",
        scale_folders=True,
        sources=("features",),
        feature_mask=parse_audio_feature_mask("111111"),
        required_sources=("features",),
    )
    assert len(out) == 8
    assert (FROZEN_FOLDER_BASIS_MODE, FROZEN_FOLDER_BOOST,
            FROZEN_FOLDER_DEPTH_BOOST) == ("centroids", 2.4, 3.0)
    assert seen["folder_basis_mode"] == "centroids"
    assert seen["folder_boost"] == pytest.approx(2.4)
    assert seen["folder_depth_boost"] == pytest.approx(3.0)
    assert "fbmcentroids" in seen["inp"].tsne_config


def test_compute_projection_pca_without_semantics():
    fp = "x"
    infos = [
        {"path": "a/t0.mp3", "fingerprint": fp},
        {"path": "b/t1.mp3", "fingerprint": fp},
    ]
    cache = _StubFeatureCache(clap={fp: np.ones(3, dtype=np.float32)})
    out = compute_projection(
        infos,
        cache,
        method="pca",
        scale_folders=False,
        sources=("clap",),
    )
    assert len(out) == 2
    assert all(0 <= o["x"] <= 1 and 0 <= o["y"] <= 1 for o in out)


def test_compute_projection_all_zero_weights_is_safe():
    """All-zero weights leave no feature block. The HTTP routes reject that vector,
    but the function must not index an empty matrix if it is ever reached."""
    fp = "x"
    infos = [
        {"path": "a/t0.mp3", "fingerprint": fp},
        {"path": "b/t1.mp3", "fingerprint": fp},
    ]
    cache = _StubFeatureCache(clap={fp: np.ones(3, dtype=np.float32)})
    out = compute_projection(
        infos,
        cache,
        method="pca",
        scale_folders=False,
        sources=("clap",),
        source_weights={"clap": 0.0},
    )
    assert len(out) == 2
    assert all(o["x"] == 0.5 and o["y"] == 0.5 for o in out)


def test_pca_ref_cache_is_bounded_and_counted():
    from backend.embeddings import projection_backends as pb

    pb.reset_pca_ref_cache()
    rng = np.random.default_rng(0)
    for i in range(pb._PCA_REF_CACHE_MAX + 3):
        paths = [f"view{i}/t{j}.mp3" for j in range(3)]
        mat = rng.standard_normal((3, 5)).astype(np.float32)
        pb._project_pca(mat, paths)

    assert len(pb._pca_ref_cache) <= pb._PCA_REF_CACHE_MAX
    stats = pb.pca_ref_stats()
    assert stats["evict"] >= 3
    assert stats["miss"] >= pb._PCA_REF_CACHE_MAX + 3


def test_pca_ref_aligns_revisited_view():
    from backend.embeddings import projection_backends as pb

    pb.reset_pca_ref_cache()
    rng = np.random.default_rng(1)
    paths_a = ["a/t0.mp3", "a/t1.mp3", "a/t2.mp3"]
    mat_a = rng.standard_normal((3, 5)).astype(np.float32)
    first = pb._project_pca(mat_a, paths_a).copy()

    paths_b = ["b/t0.mp3", "b/t1.mp3", "b/t2.mp3"]
    mat_b = rng.standard_normal((3, 5)).astype(np.float32)
    pb._project_pca(mat_b, paths_b)

    again = pb._project_pca(mat_a, paths_a)
    assert np.allclose(first, again)
    assert pb.pca_ref_stats()["hit"] == 1


def _two_folder_library(seed: int = 11):
    """A CLAP-only library with two well-populated folders."""
    rng = np.random.default_rng(seed)
    clap: dict[str, np.ndarray] = {}
    infos: list[dict] = []
    for folder in ("lib/Techno", "lib/House"):
        for i in range(10):
            fp = f"{folder}-{i}"
            clap[fp] = rng.standard_normal(8).astype(np.float32)
            infos.append({"path": f"{folder}/t{i}.mp3", "fingerprint": fp})
    return _StubFeatureCache(clap=clap), clap, infos


def test_fit_reuses_chart_when_a_track_joins_an_existing_folder(monkeypatch):
    """Adding a track to an existing folder must transform, not refit.

    The fit is keyed on the folder *set*, so the chart frame is stable and the
    new track is placed into it. This is the "growing library" case that used to
    refit every lens because the key carried the full path set.
    """
    from backend.embeddings import projection_backends as pb

    monkeypatch.setattr(layout, "is_model_ready", lambda: True)
    pb.reset_tsne_fit_cache()
    cache, clap, infos = _two_folder_library()

    def project(rows: list[dict]):
        return compute_projection(
            rows, cache, method="tsne", scale_folders=True, sources=("clap",),
        )

    project(infos)
    assert pb.tsne_fit_stats()["fit"] == 1

    extra_fp = "lib/Techno-extra"
    clap[extra_fp] = np.random.default_rng(3).standard_normal(8).astype(np.float32)
    project(infos + [{"path": "lib/Techno/extra.mp3", "fingerprint": extra_fp}])

    stats = pb.tsne_fit_stats()
    assert stats["transform"] == 1
    assert stats["fit"] == 1


def test_fit_refits_when_a_new_folder_appears(monkeypatch):
    """A new folder is a genuinely new direction, so the chart refits."""
    from backend.embeddings import projection_backends as pb

    monkeypatch.setattr(layout, "is_model_ready", lambda: True)
    pb.reset_tsne_fit_cache()
    cache, clap, infos = _two_folder_library(seed=12)

    def project(rows: list[dict]):
        return compute_projection(
            rows, cache, method="tsne", scale_folders=True, sources=("clap",),
        )

    project(infos)
    assert pb.tsne_fit_stats()["fit"] == 1

    rng = np.random.default_rng(4)
    for i in range(10):
        fp = f"lib/Ambient-{i}"
        clap[fp] = rng.standard_normal(8).astype(np.float32)
        infos.append({"path": f"lib/Ambient/t{i}.mp3", "fingerprint": fp})
    project(infos)

    assert pb.tsne_fit_stats()["fit"] == 2


def test_weight_jacobian_predicts_small_weight_steps():
    """The Jacobian is a first-order fit: a small step must point its way.

    This is the property the live source-weight drag relies on — the sign and
    direction must be right even though the magnitude is only approximate.
    """
    rng = np.random.default_rng(3)
    n = 40
    clap, feats, infos = {}, {}, []
    for i in range(n):
        fp = f"j{i}"
        clap[fp] = rng.standard_normal(8).astype(np.float32)
        feats[fp] = rng.standard_normal(6).astype(np.float32)
        infos.append({"path": f"a/b/t{i}.mp3", "fingerprint": fp})
    cache = _StubFeatureCache(clap=clap, feats=feats)
    weights = {"clap": 0.4, "features": 0.6}
    mask = parse_audio_feature_mask("111111")

    def project(w: dict[str, float]):
        return compute_projection(
            infos, cache, method="pca", sources=("clap", "features"),
            feature_mask=mask, features_blend=0.5, source_weights=w,
        )

    base = project(weights)
    _, jac = compute_projection(
        infos, cache, method="pca", sources=("clap", "features"),
        feature_mask=mask, features_blend=0.5, source_weights=weights,
        with_jacobian=True,
    )
    assert set(jac) == {"clap", "features"}
    assert jac["clap"].shape == (n, 2)

    delta = 1e-3
    stepped = project({"clap": 0.4 + delta, "features": 0.6})
    actual = np.array(
        [[p["x"], p["y"]] for p in stepped], dtype=np.float64
    ) - np.array([[p["x"], p["y"]] for p in base], dtype=np.float64)
    predicted = jac["clap"].astype(np.float64) * delta

    flat_a, flat_p = actual.ravel(), predicted.ravel()
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_p)
    assert denom > 1e-9
    cosine = float(flat_a @ flat_p) / denom
    assert cosine > 0.5


def test_weight_jacobian_one_sided_step_for_zero_weight_source():
    fp = "z"
    infos = [{"path": f"a/t{i}.mp3", "fingerprint": fp} for i in range(3)]
    cache = _StubFeatureCache(
        clap={fp: np.ones(3, dtype=np.float32)},
        feats={fp: np.ones(6, dtype=np.float32)},
    )
    _, jac = compute_projection(
        infos, cache, method="pca", sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"), features_blend=0.5,
        source_weights={"clap": 1.0, "features": 0.0}, with_jacobian=True,
    )
    # A zero-weight source that still has a block available has a one-sided
    # derivative (the rate of raising it from 0) — the slider preview needs it.
    assert "features" in jac
    assert "clap" in jac
    assert jac["features"].shape == (3, 2)


def test_weight_jacobian_skips_source_with_no_block():
    """A source with no cached rows at all has no derivative, even stepped +eps."""
    fp = "noblock"
    infos = [{"path": f"nb/t{i}.mp3", "fingerprint": fp} for i in range(3)]
    # ``clap`` has rows, ``features`` has none. Only ``clap`` is required, so the
    # tracks still qualify and ``features`` stays an absent (skipped) source.
    cache = _StubFeatureCache(clap={fp: np.ones(3, dtype=np.float32)})
    _, jac = compute_projection(
        infos, cache, method="pca", sources=("clap", "features"),
        required_sources=("clap",),
        feature_mask=parse_audio_feature_mask("111111"), features_blend=0.5,
        source_weights={"clap": 1.0, "features": 0.0}, with_jacobian=True,
    )
    # ``features`` has no block available, so it is skipped (not a zero entry).
    assert "features" not in jac
    assert "clap" in jac


def test_weight_jacobian_reports_layout_and_per_source_progress():
    rng = np.random.default_rng(9)
    n = 12
    clap, feats, infos = {}, {}, []
    for i in range(n):
        fp = f"p{i}"
        clap[fp] = rng.standard_normal(4).astype(np.float32)
        feats[fp] = rng.standard_normal(6).astype(np.float32)
        infos.append({"path": f"a/t{i}.mp3", "fingerprint": fp})
    cache = _StubFeatureCache(clap=clap, feats=feats)
    seen: list[tuple[str, int, int]] = []
    compute_projection(
        infos, cache, method="pca", sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"), features_blend=0.5,
        source_weights={"clap": 0.5, "features": 0.5}, with_jacobian=True,
        on_progress=lambda phase, done, total: seen.append((phase, done, total)),
    )
    # The base layout lands first, then the Jacobian reports per source.
    assert ("layout", 0, 0) in seen
    assert seen[-1] == ("jacobian", 2, 2)


def _coords(positions: list[dict]) -> np.ndarray:
    return np.asarray([[p["x"], p["y"]] for p in positions], dtype=np.float64)


def test_jacobian_request_does_not_return_the_cached_positions_list(monkeypatch):
    """A warm revision + ``with_jacobian`` must still return ``(positions, jac)``.

    The early cache return used to hand back the bare list, so the route's tuple
    unpack 500'd on an already-warm layout.
    """
    rng = np.random.default_rng(21)
    n = 12
    cache = _StubFeatureCache()
    infos = []
    for i in range(n):
        fp = f"c{i}"
        cache._maps["clap"][fp] = rng.standard_normal(4).astype(np.float32)
        infos.append({"path": f"a/t{i}.mp3", "fingerprint": fp})
    monkeypatch.setattr(
        layout,
        "_project_tsne",
        lambda mat, **_k: rng.standard_normal((mat.shape[0], 2)),
    )
    layout._revision_layout_cache.clear()
    layout._revision_layout_cache["rev-warm"] = [
        {"path": t["path"], "x": 0.0, "y": 0.0} for t in infos
    ]
    try:
        positions, jac = compute_projection(
            infos,
            cache,
            method="tsne",
            sources=("clap",),
            source_weights={"clap": 1.0},
            layout_revision="rev-warm",
            with_jacobian=True,
        )
    finally:
        layout._revision_layout_cache.clear()
    assert len(positions) == n
    assert isinstance(jac, dict)
    assert "clap" in jac


def test_gauge_scope_rekeys_coordinates_but_not_the_tsne_fit(monkeypatch):
    """Changing the configured roots must change the coordinate revision, while
    the fitted t-SNE embedding stays keyed on the vector config alone."""
    rng = np.random.default_rng(5)
    n = 20
    clap, feats, infos = {}, {}, []
    for i in range(n):
        fp = f"s{i}"
        clap[fp] = rng.standard_normal(8).astype(np.float32)
        feats[fp] = rng.standard_normal(6).astype(np.float32)
        infos.append({"path": f"a/t{i}.mp3", "fingerprint": fp})
    cache = _StubFeatureCache(clap=clap, feats=feats)
    paths = [t["path"] for t in infos]

    fit_keys: list[str] = []

    def fake_tsne(_mat, *, paths, config_key, **_kw):
        fit_keys.append(config_key)
        return np.linspace(-1.0, 1.0, 2 * len(paths)).reshape(len(paths), 2)

    monkeypatch.setattr(layout, "_project_tsne", fake_tsne)
    revisions: list[str] = []
    lens_gauge.reset_gauge_cache()
    try:
        for scope in ("library:a", "library:a|b"):
            revision = compute_layout_revision(
                eligible_paths=paths, cache_versions=(1,), gauge_scope=scope
            )
            revisions.append(revision)
            compute_projection(
                infos,
                cache,
                method="tsne",
                sources=("clap", "features"),
                source_weights={"clap": 0.5, "features": 0.5},
                layout_revision=revision,
                gauge_scope=scope,
            )
    finally:
        lens_gauge.reset_gauge_cache()
        layout._revision_layout_cache.clear()

    assert revisions[0] != revisions[1]
    assert fit_keys and len(set(fit_keys)) == 1


def test_weight_jacobian_is_in_the_served_gauge_frame():
    """The fallback derivative must carry the gauge's rotation and scale.

    A pre-seeded (non-anchor) reference rotates and scales the raw layout. The
    raw base and a raw one-sided step come from fresh scopes (each is its own
    anchor, so neither is gauged). The expected derivative is the raw step
    difference mapped by the *same* base transform the served layout uses. The
    old ungauged difference matched the raw step, not what the user sees.
    """
    rng = np.random.default_rng(3)
    n = 40
    clap, feats, infos = {}, {}, []
    for i in range(n):
        fp = f"g{i}"
        clap[fp] = rng.standard_normal(8).astype(np.float32)
        feats[fp] = rng.standard_normal(6).astype(np.float32)
        infos.append({"path": f"a/b/t{i}.mp3", "fingerprint": fp})
    cache = _StubFeatureCache(clap=clap, feats=feats)
    weights = {"clap": 0.4, "features": 0.6}
    delta = 1e-3
    mask = parse_audio_feature_mask("111111")

    def project(w: dict[str, float], scope: str, **kw):
        return compute_projection(
            infos,
            cache,
            method="pca",
            sources=("clap", "features"),
            feature_mask=mask,
            features_blend=0.5,
            source_weights=w,
            gauge_scope=scope,
            **kw,
        )

    lens_gauge.reset_gauge_cache()
    try:
        raw_base = _coords(project(weights, "raw-base"))
        raw_step = _coords(
            project({"clap": 0.4 + delta, "features": 0.6}, "raw-step")
        )
        t = np.deg2rad(37.0)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ref = layout._normalise_coords(
            1.7 * (rot @ raw_base.T).T + np.array([3.0, -2.0])
        )
        paths = [t["path"] for t in infos]
        family = layout._gauge_family("test", paths, cache, 2)
        assert lens_gauge.seed_reference(family, paths, ref)

        served_base = _coords(project(weights, "test"))
        # The served map keeps the reference orientation but is stretched per axis
        # to fill the view, so it is close, not exactly equal.
        assert np.allclose(np.ptp(served_base, axis=0), 1.0)
        assert np.corrcoef(served_base[:, 0], ref[:, 0])[0, 1] > 0.999
        assert np.corrcoef(served_base[:, 1], ref[:, 1])[0, 1] > 0.999
        _served, jac = project(weights, "test", with_jacobian=True)
    finally:
        lens_gauge.reset_gauge_cache()

    # The served frame is orientation + per-axis display stretch, so the raw ->
    # served map is affine, not a similarity.
    design = np.column_stack([raw_base, np.ones(len(raw_base))])
    solution, *_ = np.linalg.lstsq(design, served_base, rcond=None)
    affine = np.asarray(solution, dtype=np.float64).T
    expected = (
        raw_step @ affine[:, :2].T + affine[:, 2] - served_base
    ) / delta
    np.testing.assert_allclose(
        jac["clap"].astype(np.float64), expected, rtol=2e-3, atol=2e-3
    )


def _pearson(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum()) / float(np.linalg.norm(a) * np.linalg.norm(b))


def test_compute_projection_dims3_keeps_xy_orientation_and_fills_volume():
    """The final served 3-D frame keeps the head-on orientation but is stretched
    per axis to fill the unit volume; an exact x/y overlay is not required."""
    from backend.embeddings import lens_gauge, projection_backends

    lens_gauge.reset_gauge_cache()
    projection_backends.reset_pca_ref_cache()
    rng = np.random.default_rng(4)
    n = 40
    clap, infos = {}, []
    # A dominant 2-D structure keeps the PCA plane (and the head-on fit) exact.
    core = rng.standard_normal((n, 2))
    noise = 0.05 * rng.standard_normal((n, 3))
    rows = np.hstack([core, noise]).astype(np.float32)
    for i in range(n):
        fp = f"g{i}"
        clap[fp] = rows[i]
        infos.append({"path": f"a/t{i}.mp3", "fingerprint": fp})
    cache = _StubFeatureCache(clap=clap)

    ref = compute_projection(
        infos, cache, method="pca", sources=("clap",), dims=2,
    )
    # Rotate + reflect the 2-D layout into a deliberately different frame.
    c, s = np.cos(np.radians(37.0)), np.sin(np.radians(37.0))
    rot = np.array([[c, -s], [s, c]]) @ np.array([[1.0, 0.0], [0.0, -1.0]])
    align_ref = []
    for p in ref:
        xy = rot @ np.array([p["x"], p["y"]])
        align_ref.append({"path": p["path"], "x": float(xy[0]), "y": float(xy[1])})

    out = compute_projection(
        infos, cache, method="pca", sources=("clap",), dims=3, align_ref=align_ref,
    )
    by_path = {p["path"]: p for p in out}
    assert len(out) == n
    xs = [by_path[p["path"]]["x"] for p in align_ref]
    ys = [by_path[p["path"]]["y"] for p in align_ref]
    assert _pearson(xs, [p["x"] for p in align_ref]) > 0.99
    assert _pearson(ys, [p["y"] for p in align_ref]) > 0.99
    for axis in ("x", "y", "z"):
        values = np.asarray([p[axis] for p in out])
        assert np.isclose(values.min(), 0.0)
        assert np.isclose(values.max(), 1.0)
