"""Unit tests for embedding-space projection helpers (composite vectors,
folder hierarchy, parsers).  Avoids CLAP model load.
"""

import numpy as np

from backend.audio_features import AUDIO_FEATURE_DIM
from backend import embeddings
from backend.embeddings import (
    _apply_low_rank_semantic_weight,
    _build_folder_directions,
    _expand_folder_nodes,
    _folder_seeds_from_track_paths,
    _gather_composite_vecs,
    _get_prepared_source_row,
    compute_projection,
    parse_audio_feature_mask,
    parse_features_blend,
    parse_folder_boost,
    parse_folder_depth_boost,
)


class _StubFeatureCache:
    """Minimal cache implementing composite gather lookups."""

    def __init__(
        self,
        clap: dict[str, np.ndarray] | None = None,
        effnet: dict[str, np.ndarray] | None = None,
        feats: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._clap = clap or {}
        self._effnet = effnet or {}
        self._feat = feats or {}

    def get_all_embeddings(
        self, fingerprints: list[str], version: int = 0,
    ) -> dict[str, np.ndarray]:
        del version
        return {fp: self._clap[fp] for fp in fingerprints if fp in self._clap}

    def get_all_effnet_embeddings(
        self, fingerprints: list[str], version: int = 0,
    ) -> dict[str, np.ndarray]:
        del version
        return {fp: self._effnet[fp] for fp in fingerprints if fp in self._effnet}

    def get_all_audio_features(
        self, fingerprints: list[str], version: int = 0,
    ) -> dict[str, np.ndarray]:
        del version
        return {fp: self._feat[fp] for fp in fingerprints if fp in self._feat}


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
    embeddings._source_row_cache.clear()
    raw = np.array([3.0, 4.0], dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)
    a = _get_prepared_source_row("clap", "fp-row", raw, mask, 0.42)
    b = _get_prepared_source_row("clap", "fp-row", raw, mask, 0.42)
    assert a is b


def test_composite_tier_cache_reuses_matrix():
    embeddings._composite_cache.clear()
    fp = "fpz"
    clap = np.array([1.0, 0.0], dtype=np.float32)
    feat = np.ones(6, dtype=np.float32)
    infos = [{"path": "a/z.mp3", "fingerprint": fp}]
    cache = _StubFeatureCache(clap={fp: clap}, feats={fp: feat})
    _, m1, _, _, k1 = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=0.5,
    )
    _, m2, _, _, k2 = _gather_composite_vecs(
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

    _, mat_mask, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=mask,
        features_blend=1.0,
    )
    _, mat_no_mask, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=1.0,
    )
    assert mat_mask.shape == mat_no_mask.shape == (1, 8)
    assert not np.allclose(mat_mask, mat_no_mask)

    _, mat_weak, _, _, _ = _gather_composite_vecs(
        infos,
        cache,
        sources=("clap", "features"),
        feature_mask=parse_audio_feature_mask("111111"),
        features_blend=0.2,
    )
    _, mat_full, _, _, _ = _gather_composite_vecs(
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
