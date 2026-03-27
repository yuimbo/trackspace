"""Shared embedding-space layout: composite vectors, semantic weighting, 2-D projection.

CLAP model I/O lives in `clap.py`. EffNet and librosa descriptors live in
`effnet` / `librosa_audio_features`). This module concatenates cached source rows, applies
folder/tag re-weighting (CLAP text directions in the CLAP column span), and runs
PCA / UMAP / t-SNE. Re-exported from ``backend.embeddings``.
"""

import hashlib
import importlib.util
import logging
import os
import platform
import sys
import time
from collections import OrderedDict
import numpy as np

from .clap import (
    CLAP_SR,
    EMBEDDING_VERSION,
    batch_ensure_embeddings,
    generate_embedding,
    generate_text_embeddings,
    inference_lock as _inference_lock,
    is_model_ready,
    load_model,
    _load_audio_segments,
)
from .layout_revision import FOLDER_SEMANTIC_BASIS_VERSION

log = logging.getLogger(__name__)

# Within-folder PCA on deviations orthogonal to the hierarchical contrast axis.
FOLDER_PCA_MIN_TRACKS = 8
FOLDER_PCA_MAX_COMPONENTS = 2
FOLDER_PCA_SECOND_SV_RATIO = 0.18

# Tiered projection LRU (composite gather → semantic direction basis → UMAP/t-SNE).
_COMPOSITE_CACHE_MAX = 10
_SEMANTIC_BASIS_CACHE_MAX = 10
_REVISION_LAYOUT_CACHE_MAX = 16
_composite_cache: OrderedDict[str, tuple[list[str], np.ndarray, int, int]] = (
    OrderedDict()
)
_semantic_basis_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
    OrderedDict()
)
_revision_layout_cache: OrderedDict[str, list[dict]] = OrderedDict()

_SOURCE_ROW_CACHE_MAX = 24_000
_source_row_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()


def _source_row_cache_get(key: tuple) -> np.ndarray | None:
    if key not in _source_row_cache:
        return None
    _source_row_cache.move_to_end(key)
    return _source_row_cache[key]


def _source_row_cache_put(key: tuple, value: np.ndarray) -> None:
    _source_row_cache[key] = value
    _source_row_cache.move_to_end(key)
    while len(_source_row_cache) > _SOURCE_ROW_CACHE_MAX:
        _source_row_cache.popitem(last=False)


def _tier_cache_get(cache: OrderedDict, key: str):
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _tier_cache_put(cache: OrderedDict, key: str, value, max_size: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _source_versions_token() -> str:
    from .effnet import EFFNET_VERSION
    from .librosa_audio_features import FEATURES_VERSION

    return f"C{EMBEDDING_VERSION}E{EFFNET_VERSION}F{FEATURES_VERSION}"


def _composite_param_key(
    paths: list[str],
    sources: tuple[str, ...],
    mask: np.ndarray,
    blend: float,
) -> str:
    ph = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:24]
    return (
        f"{ph}|{_source_versions_token()}|{','.join(sources)}|"
        f"{mask.tobytes().hex()}|{blend:.6g}"
    )


def _get_prepared_source_row(
    source: str,
    fp: str,
    raw: np.ndarray,
    feature_mask: np.ndarray,
    features_blend: float,
) -> np.ndarray:
    """Return one track row for *source* with the same preprocessing as composite gather."""
    from .effnet import EFFNET_VERSION
    from .librosa_audio_features import FEATURES_VERSION

    raw = np.asarray(raw, dtype=np.float32).reshape(-1)

    if source == "clap":
        key = ("clap", fp, EMBEDDING_VERSION)
        hit = _source_row_cache_get(key)
        if hit is not None:
            return hit
        n = float(np.linalg.norm(raw))
        if n < 1e-9:
            n = 1.0
        v = (raw / n).astype(np.float32)
        _source_row_cache_put(key, v)
        return v

    if source == "effnet":
        key = ("effnet", fp, EFFNET_VERSION)
        hit = _source_row_cache_get(key)
        if hit is not None:
            return hit
        n = float(np.linalg.norm(raw))
        if n < 1e-9:
            n = 1.0
        v = (raw / n).astype(np.float32)
        _source_row_cache_put(key, v)
        return v

    if source == "features":
        b = float(np.clip(features_blend, 0.05, 1.0))
        key = ("features", fp, FEATURES_VERSION, feature_mask.tobytes(), round(b, 9))
        hit = _source_row_cache_get(key)
        if hit is not None:
            return hit
        v = raw * feature_mask
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            n = 1.0
        v = (v / n * b).astype(np.float32)
        _source_row_cache_put(key, v)
        return v

    raise ValueError(f"unknown source {source!r}")


def _semantic_basis_param_key(
    context_tags: list[str] | None,
    folder_seeds: list[str] | None,
) -> str:
    tags = ",".join(sorted(context_tags)) if context_tags else ""
    if folder_seeds:
        fp = ",".join(sorted(folder_seeds))
    else:
        fp = "-"
    return f"{tags}|{fp}|fsbv{FOLDER_SEMANTIC_BASIS_VERSION}"


def _folder_seeds_from_track_paths(paths: list[str]) -> list[str]:
    s: set[str] = set()
    for p in paths:
        sep = p.rfind("/")
        tf = p[:sep] if sep >= 0 else ""
        if tf:
            s.add(tf)
    return sorted(s)


def _expand_folder_nodes(folder_seeds: list[str]) -> set[str]:
    all_nodes: set[str] = {""}
    for f in folder_seeds:
        parts = [p for p in f.split("/") if p]
        for depth in range(1, len(parts) + 1):
            all_nodes.add("/".join(parts[:depth]))
    return all_nodes


def _folder_orthogonal_pc_directions(
    mat_block: np.ndarray,
    d_unit: np.ndarray,
    *,
    max_components: int,
    min_second_sv_ratio: float,
) -> list[np.ndarray]:
    n = mat_block.shape[0]
    if n < 4 or max_components < 1:
        return []

    c = mat_block.mean(axis=0)
    x_dev = mat_block - c
    proj = x_dev @ d_unit
    x_orth = x_dev - np.outer(proj, d_unit.astype(np.float64))

    _, s, vt = np.linalg.svd(x_orth.astype(np.float64), full_matrices=False)
    if s.size == 0 or float(s[0]) < 1e-12:
        return []

    d64 = d_unit.astype(np.float64)
    out: list[np.ndarray] = []
    lim = min(max_components, int(vt.shape[0]))
    for i in range(lim):
        if i > 0 and float(s[i]) < min_second_sv_ratio * float(s[0]):
            break
        v = np.asarray(vt[i], dtype=np.float64)
        v = v - float(v @ d64) * d64
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            continue
        out.append((v / nv).astype(np.float32))
    return out


def _build_folder_directions(
    mat: np.ndarray,
    paths: list[str],
    folder_seeds: list[str],
    min_tracks: int = 2,
    min_tracks_pca: int = FOLDER_PCA_MIN_TRACKS,
    max_pca_components: int = FOLDER_PCA_MAX_COMPONENTS,
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, int]]]:
    if not folder_seeds:
        return None, None, []

    all_nodes = _expand_folder_nodes(folder_seeds)

    track_folders: list[str] = []
    for p in paths:
        sep = p.rfind("/")
        track_folders.append(p[:sep] if sep >= 0 else "")

    node_indices: dict[str, list[int]] = {n: [] for n in all_nodes}
    for i, tf in enumerate(track_folders):
        node_indices[""].append(i)
        parts = [p for p in tf.split("/") if p]
        for depth in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:depth])
            if ancestor in node_indices:
                node_indices[ancestor].append(i)

    centroids: dict[str, np.ndarray | None] = {}
    for n, idx_list in node_indices.items():
        centroids[n] = mat[idx_list].mean(axis=0) if idx_list else None

    dirs: list[np.ndarray] = []
    exps: list[int] = []
    fallbacks: list[tuple[str, int]] = []

    for node in sorted(all_nodes - {""}):
        parts = node.split("/")
        parent = "/".join(parts[:-1])
        depth = len(parts)
        exp = depth - 1

        c = centroids.get(node)
        pc = centroids.get(parent)
        if c is None or pc is None:
            continue

        if len(node_indices[node]) < min_tracks:
            fallbacks.append((parts[-1], exp))
            continue

        d = c - pc
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue

        d_unit = (d / norm).astype(np.float32)
        dirs.append(d_unit)
        exps.append(exp)

        idx_list = node_indices[node]
        if len(idx_list) >= min_tracks_pca and max_pca_components > 0:
            block = mat[idx_list]
            extras = _folder_orthogonal_pc_directions(
                block,
                d_unit,
                max_components=max_pca_components,
                min_second_sv_ratio=FOLDER_PCA_SECOND_SV_RATIO,
            )
            for v_pc in extras:
                dirs.append(v_pc)
                exps.append(exp)

    if dirs:
        log.debug(
            "folder directions: %d data-driven rows, %d text-fallback",
            len(dirs), len(fallbacks),
        )
        return (
            np.stack(dirs),
            np.array(exps, dtype=np.int32),
            fallbacks,
        )
    return None, None, fallbacks


def _unit_clap_text_dirs_to_full(
    unit_rows: np.ndarray,
    d_total: int,
    clap_lo: int,
    clap_hi: int,
) -> np.ndarray:
    if clap_lo >= clap_hi:
        raise ValueError("clap column span must be non-empty")
    slot_w = clap_hi - clap_lo
    k, d_in = unit_rows.shape
    out = np.zeros((k, d_total), dtype=np.float32)
    if d_in <= slot_w:
        out[:, clap_lo : clap_lo + d_in] = unit_rows
    else:
        out[:, clap_lo:clap_hi] = unit_rows[:, :slot_w]
    return out


def _apply_low_rank_semantic_weight(
    mat: np.ndarray,
    T_n: np.ndarray,
    exp: np.ndarray,
    boost: float,
    depth_boost: float,
) -> np.ndarray:
    if T_n.shape[0] == 0:
        return mat
    gamma = np.power(
        float(depth_boost), exp.astype(np.float64),
    ).astype(np.float32)
    w = (gamma * float(boost)).astype(np.float32)
    g = mat @ T_n.T
    return (mat + (g * w) @ T_n).astype(np.float32)


def _build_semantic_basis(
    mat: np.ndarray,
    paths: list[str],
    context_tags: list[str] | None,
    folder_seeds: list[str] | None,
    clap_lo: int,
    clap_hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    d_total = mat.shape[1]
    have_clap_slot = clap_lo < clap_hi
    all_dirs: list[np.ndarray] = []
    all_exp: list[np.ndarray] = []

    if folder_seeds:
        fd, f_exp, fallbacks = _build_folder_directions(mat, paths, folder_seeds)
        if fd is not None and f_exp is not None:
            all_dirs.append(fd)
            all_exp.append(f_exp)

        if fallbacks and have_clap_slot:
            texts = [t for t, _ in fallbacks]
            fb_exp = np.array([e for _, e in fallbacks], dtype=np.int32)
            text_vecs = generate_text_embeddings(texts)
            if text_vecs is not None and text_vecs.shape[0] > 0:
                norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                full = _unit_clap_text_dirs_to_full(
                    text_vecs / norms, d_total, clap_lo, clap_hi,
                )
                all_dirs.append(full)
                all_exp.append(fb_exp)

    if context_tags and have_clap_slot:
        tag_vecs = generate_text_embeddings(context_tags)
        if tag_vecs is not None and tag_vecs.shape[0] > 0:
            norms = np.linalg.norm(tag_vecs, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            full = _unit_clap_text_dirs_to_full(
                tag_vecs / norms, d_total, clap_lo, clap_hi,
            )
            all_dirs.append(full)
            all_exp.append(np.zeros(len(context_tags), dtype=np.int32))

    if not all_dirs:
        return (
            np.empty((0, d_total), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    T_n = np.concatenate(all_dirs, axis=0).astype(np.float32)
    exp = np.concatenate(all_exp)
    return T_n, exp


def _apply_semantic_weighting(
    mat: np.ndarray,
    paths: list[str],
    context_tags: list[str] | None = None,
    folder_seeds: list[str] | None = None,
    boost: float = 3.0,
    depth_boost: float = 1.5,
    clap_lo: int = 0,
    clap_hi: int = 0,
) -> np.ndarray:
    T_n, exp = _build_semantic_basis(
        mat, paths, context_tags, folder_seeds, clap_lo, clap_hi,
    )
    if T_n.shape[0] == 0:
        return mat
    return _apply_low_rank_semantic_weight(mat, T_n, exp, boost, depth_boost)


def _gather_vecs(
    track_infos: list[dict],
    feature_cache,
    version: int,
) -> tuple[list[str], list[np.ndarray]]:
    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]
    embeddings_map = feature_cache.get_all_embeddings(fp_set, version=version)

    paths: list[str] = []
    vecs: list[np.ndarray] = []
    for t in track_infos:
        fp = t.get("fingerprint")
        if fp and fp in embeddings_map:
            paths.append(t["path"])
            vecs.append(embeddings_map[fp])
    return paths, vecs


def _gather_composite_vecs(
    track_infos: list[dict],
    feature_cache,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
) -> tuple[list[str], np.ndarray, int, int, str]:
    from .effnet import EFFNET_VERSION
    from .librosa_audio_features import AUDIO_FEATURE_DIM, FEATURES_VERSION

    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]

    source_maps: dict[str, dict[str, np.ndarray]] = {}
    if "clap" in sources:
        source_maps["clap"] = feature_cache.get_all_embeddings(fp_set, version=EMBEDDING_VERSION)
    if "effnet" in sources:
        source_maps["effnet"] = feature_cache.get_all_effnet_embeddings(fp_set, version=EFFNET_VERSION)
    if "features" in sources:
        source_maps["features"] = feature_cache.get_all_audio_features(fp_set, version=FEATURES_VERSION)

    paths: list[str] = []
    fp_order: list[str] = []
    for t in track_infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        if all(fp in source_maps.get(s, {}) for s in sources):
            paths.append(t["path"])
            fp_order.append(fp)

    if not paths:
        return paths, np.empty((0, 0), dtype=np.float32), 0, 0, ""

    if feature_mask is None:
        mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)
    else:
        mask = np.asarray(feature_mask, dtype=np.float32).reshape(-1)
        if mask.size != AUDIO_FEATURE_DIM:
            mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)

    blend = float(np.clip(features_blend, 0.05, 1.0))
    comp_key = _composite_param_key(paths, sources, mask, blend)
    cached = _tier_cache_get(_composite_cache, comp_key)
    if cached is not None:
        _p, _mat, _cd, _clo = cached
        log.debug("projection tier: composite cache hit")
        return paths, _mat, _cd, _clo, comp_key

    blocks: list[np.ndarray] = []
    clap_dim = 0
    clap_col_lo = 0
    col = 0
    for s in sources:
        smap = source_maps[s]
        block = np.stack(
            [
                _get_prepared_source_row(s, fp, smap[fp], mask, blend)
                for fp in fp_order
            ],
            axis=0,
        )
        if s == "clap":
            clap_dim = block.shape[1]
            clap_col_lo = col
        blocks.append(block)
        col += block.shape[1]

    mat = np.concatenate(blocks, axis=1).astype(np.float32)
    _tier_cache_put(
        _composite_cache,
        comp_key,
        (list(paths), mat, clap_dim, clap_col_lo),
        _COMPOSITE_CACHE_MAX,
    )
    return paths, mat, clap_dim, clap_col_lo, comp_key


def _normalise_coords(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    return (coords - mins) / ranges


_pca_ref_paths: list[str] = []
_pca_ref_coords: np.ndarray | None = None


def _project_pca(mat: np.ndarray, paths: list[str] | None = None) -> np.ndarray:
    global _pca_ref_paths, _pca_ref_coords

    centred = mat - mat.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ Vt[:2].T

    ref = _pca_ref_coords
    ref_paths = _pca_ref_paths
    if paths is not None and ref is not None and len(ref_paths) >= 2:
        ref_map = {p: i for i, p in enumerate(ref_paths)}
        idx_new, idx_ref = [], []
        for i, p in enumerate(paths):
            if p in ref_map:
                idx_new.append(i)
                idx_ref.append(ref_map[p])
        if len(idx_new) >= 2:
            A = coords[idx_new]
            B = ref[idx_ref]
            A_mu, B_mu = A.mean(axis=0), B.mean(axis=0)
            U, _, Vt2 = np.linalg.svd((A - A_mu).T @ (B - B_mu))
            R = U @ Vt2
            coords = (coords - A_mu) @ R + B_mu

    if paths is not None:
        _pca_ref_paths = list(paths)
        _pca_ref_coords = coords.copy()
    return coords


_cuml_projection_enabled: bool | None = None
_mlx_vis_projection_enabled: bool | None = None


def reset_projection_backend_cache() -> None:
    global _cuml_projection_enabled, _mlx_vis_projection_enabled
    _cuml_projection_enabled = None
    _mlx_vis_projection_enabled = None


def _use_cuml_projection() -> bool:
    global _cuml_projection_enabled
    if _cuml_projection_enabled is not None:
        return _cuml_projection_enabled
    raw = os.environ.get("TRACKSPACE_CUML", "1").strip().lower()
    if raw in ("0", "false", "no"):
        _cuml_projection_enabled = False
        return False
    try:
        import cupy as cp
    except ImportError:
        _cuml_projection_enabled = False
        return False
    try:
        if not cp.cuda.is_available():
            _cuml_projection_enabled = False
            return False
    except Exception:
        _cuml_projection_enabled = False
        return False
    try:
        import cuml  # noqa: F401
    except ImportError:
        _cuml_projection_enabled = False
        return False
    _cuml_projection_enabled = True
    log.info("UMAP/t-SNE: using RAPIDS cuML (CUDA)")
    return True


def _use_mlx_vis_projection() -> bool:
    global _mlx_vis_projection_enabled
    if _mlx_vis_projection_enabled is not None:
        return _mlx_vis_projection_enabled
    raw = os.environ.get("TRACKSPACE_MLX_VIS", "1").strip().lower()
    if raw in ("0", "false", "no"):
        _mlx_vis_projection_enabled = False
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        _mlx_vis_projection_enabled = False
        return False
    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec(
        "mlx_vis",
    ) is None:
        _mlx_vis_projection_enabled = False
        return False
    _mlx_vis_projection_enabled = True
    log.info("UMAP/t-SNE: using mlx-vis (Metal)")
    return True


def _mlx_tsne_pca_dim(n_features: int) -> int | None:
    if n_features < 2:
        return None
    return min(50, n_features - 1)


def _mlx_tsne_max_points() -> int:
    raw = os.environ.get("TRACKSPACE_MLX_TSNE_MAX_POINTS", "10000").strip()
    try:
        v = int(raw)
    except ValueError:
        return 10_000
    return v


def _project_umap(mat: np.ndarray) -> np.ndarray:
    global _cuml_projection_enabled, _mlx_vis_projection_enabled
    if _use_cuml_projection():
        try:
            from cuml.manifold import UMAP

            n_neighbors = min(15, len(mat) - 1)
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
                output_type="numpy",
            )
            return np.asarray(
                reducer.fit_transform(np.asarray(mat, dtype=np.float32)),
                dtype=np.float32,
            )
        except Exception as e:
            log.warning("cuML UMAP failed; using umap-learn: %s", e)
            _cuml_projection_enabled = False
    if _use_mlx_vis_projection():
        try:
            from mlx_vis._umap.umap import UMAP

            n_neighbors = min(15, len(mat) - 1)
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
                normalize=False,
            )
            x = np.asarray(mat, dtype=np.float32)
            with _inference_lock:
                y = reducer.fit_transform(x)
            return np.asarray(y, dtype=np.float32)
        except Exception as e:
            log.warning("mlx-vis UMAP failed; using umap-learn: %s", e)
            _mlx_vis_projection_enabled = False
    import umap  # lazy — heavy import
    n_neighbors = min(15, len(mat) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    return reducer.fit_transform(mat)


def _project_tsne(mat: np.ndarray) -> np.ndarray:
    global _cuml_projection_enabled, _mlx_vis_projection_enabled
    if _use_cuml_projection():
        try:
            from cuml.manifold import TSNE

            perplexity = min(30.0, max(5.0, (len(mat) - 1) / 3.0))
            reducer = TSNE(
                n_components=2,
                perplexity=float(perplexity),
                init="pca",
                random_state=42,
                method="fft",
                learning_rate_method="adaptive",
                output_type="numpy",
            )
            return np.asarray(
                reducer.fit_transform(np.asarray(mat, dtype=np.float32)),
                dtype=np.float32,
            )
        except Exception as e:
            log.warning("cuML t-SNE failed; using scikit-learn: %s", e)
            _cuml_projection_enabled = False
    cap = _mlx_tsne_max_points()
    if _use_mlx_vis_projection() and (cap <= 0 or len(mat) <= cap):
        pca_dim = _mlx_tsne_pca_dim(int(np.asarray(mat).shape[1]))
        if pca_dim is not None:
            try:
                from mlx_vis._tsne.tsne import TSNE

                n = len(mat)
                perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
                reducer = TSNE(
                    n_components=2,
                    perplexity=float(perplexity),
                    random_state=42,
                    pca_dim=pca_dim,
                    normalize=False,
                )
                x = np.asarray(mat, dtype=np.float32)
                with _inference_lock:
                    y = reducer.fit_transform(x)
                return np.asarray(y, dtype=np.float32)
            except Exception as e:
                log.warning("mlx-vis t-SNE failed; using scikit-learn: %s", e)
                _mlx_vis_projection_enabled = False
    from sklearn.manifold import TSNE  # lazy — heavy import
    perplexity = min(30.0, max(5.0, (len(mat) - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    return reducer.fit_transform(mat)


def parse_audio_feature_mask(raw: str | None) -> np.ndarray:
    from .librosa_audio_features import AUDIO_FEATURE_DIM

    if not raw or not str(raw).strip():
        return np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)
    s = str(raw).strip().replace(",", "")
    if len(s) != AUDIO_FEATURE_DIM or any(c not in "01" for c in s):
        return np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)
    return np.array([float(int(c)) for c in s], dtype=np.float32)


def parse_features_blend(raw: str | None, default: float = 0.42) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(np.clip(float(raw), 0.05, 1.0))
    except ValueError:
        return default


def parse_folder_boost(raw: str | None, default: float = 3.0) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(np.clip(float(raw), 0.0, 12.0))
    except ValueError:
        return default


def parse_folder_depth_boost(raw: str | None, default: float = 1.5) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(np.clip(float(raw), 1.0, 3.0))
    except ValueError:
        return default


def compute_projection(
    track_infos: list[dict],
    feature_cache,
    version: int = EMBEDDING_VERSION,
    method: str = "umap",
    context_tags: list[str] | None = None,
    context_folders: list[str] | None = None,
    scale_folders: bool = False,
    folder_boost: float = 3.0,
    folder_depth_boost: float = 1.5,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    layout_revision: str | None = None,
) -> list[dict]:
    if layout_revision and method in ("umap", "tsne"):
        hit = _tier_cache_get(_revision_layout_cache, layout_revision)
        if hit is not None:
            log.debug("projection: revision cache hit (%s)", method)
            return hit

    t0 = time.perf_counter()
    paths, mat, clap_dim, clap_col_lo, comp_key = _gather_composite_vecs(
        track_infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
    )
    t_after_gather = time.perf_counter()

    if len(paths) < 2:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "compute_projection method=%s n=%d gather=%.1fms (short-circuit)",
                method,
                len(paths),
                (t_after_gather - t0) * 1000,
            )
        return [{"path": p, "x": 0.5, "y": 0.5} for p in paths]

    if scale_folders:
        folder_seeds = (
            list(context_folders) if context_folders else _folder_seeds_from_track_paths(paths)
        )
    else:
        folder_seeds = None

    needs_weight = is_model_ready() and (bool(context_tags) or bool(folder_seeds))

    t_before_weight = time.perf_counter()
    mat_proj = mat
    if needs_weight:
        basis_key = f"{comp_key}|{_semantic_basis_param_key(context_tags, folder_seeds)}"
        basis = _tier_cache_get(_semantic_basis_cache, basis_key)
        if basis is None:
            clap_hi = clap_col_lo + clap_dim
            tags_for_weight = context_tags if clap_dim > 0 else None
            T_n, exp = _build_semantic_basis(
                mat,
                paths,
                tags_for_weight,
                folder_seeds,
                clap_col_lo,
                clap_hi,
            )
            _tier_cache_put(
                _semantic_basis_cache,
                basis_key,
                (T_n, exp),
                _SEMANTIC_BASIS_CACHE_MAX,
            )
        else:
            T_n, exp = basis
            log.debug("projection tier: semantic basis cache hit")
        mat_proj = _apply_low_rank_semantic_weight(
            mat, T_n, exp, folder_boost, folder_depth_boost,
        )
    t_after_weight = time.perf_counter()

    t_before_proj = time.perf_counter()
    if method == "pca":
        coords = _project_pca(mat_proj, paths)
    elif method == "tsne":
        coords = _project_tsne(mat_proj)
    else:
        coords = _project_umap(mat_proj)
    t_after_proj = time.perf_counter()

    t_before_norm = time.perf_counter()
    normed = _normalise_coords(coords)
    t_end = time.perf_counter()

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "compute_projection method=%s n=%d gather=%.1fms weight=%.1fms "
            "project=%.1fms norm=%.1fms total=%.1fms",
            method,
            len(paths),
            (t_after_gather - t0) * 1000,
            (t_after_weight - t_before_weight) * 1000,
            (t_after_proj - t_before_proj) * 1000,
            (t_end - t_before_norm) * 1000,
            (t_end - t0) * 1000,
        )

    result = [
        {"path": paths[i], "x": float(normed[i, 0]), "y": float(normed[i, 1])}
        for i in range(len(paths))
    ]

    if layout_revision and method in ("umap", "tsne"):
        _tier_cache_put(
            _revision_layout_cache, layout_revision, result, _REVISION_LAYOUT_CACHE_MAX,
        )

    return result


compute_umap = compute_projection
