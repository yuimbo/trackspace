"""2-D projection backends: PCA preview and cuML / mlx-vis / scikit-learn t-SNE.

Extracted from ``layout``. The PCA reference-alignment state lives here too; it
keeps successive PCA previews in a stable orientation so dots do not spin between
incremental refreshes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import platform
import sys
from collections import OrderedDict

import numpy as np

from .clap import inference_lock as _inference_lock

log = logging.getLogger(__name__)

#: Bounded PCA reference-alignment LRU. Each entry is one (paths, coords)
#: snapshot; revisiting an earlier view re-aligns to that view's own
#: orientation instead of the last one computed.
_PCA_REF_CACHE_MAX = 4
_pca_ref_cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
_pca_ref_stats = {"hit": 0, "miss": 0, "evict": 0}


def _pca_ref_key(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:24]


def pca_ref_stats() -> dict[str, int]:
    """Hit / miss / eviction counters for the PCA reference-alignment cache."""
    return dict(_pca_ref_stats)


def reset_pca_ref_cache() -> None:
    """Drop all PCA reference state and zero the counters (tests, config changes)."""
    _pca_ref_cache.clear()
    for key in _pca_ref_stats:
        _pca_ref_stats[key] = 0


def _pca_ref_lookup(paths: list[str]) -> tuple[list[str], np.ndarray] | None:
    """Reference to align against: exact view if cached, else the most recent."""
    key = _pca_ref_key(paths)
    entry = _pca_ref_cache.get(key)
    if entry is not None:
        _pca_ref_cache.move_to_end(key)
        _pca_ref_stats["hit"] += 1
        return entry
    _pca_ref_stats["miss"] += 1
    if not _pca_ref_cache:
        return None
    return next(reversed(_pca_ref_cache.values()))


def _pca_ref_store(paths: list[str], coords: np.ndarray) -> None:
    key = _pca_ref_key(paths)
    _pca_ref_cache[key] = (list(paths), coords.copy())
    _pca_ref_cache.move_to_end(key)
    while len(_pca_ref_cache) > _PCA_REF_CACHE_MAX:
        _pca_ref_cache.popitem(last=False)
        _pca_ref_stats["evict"] += 1


def _project_pca(mat: np.ndarray, paths: list[str] | None = None) -> np.ndarray:
    centred = mat - mat.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ Vt[:2].T

    if paths is not None:
        ref_entry = _pca_ref_lookup(paths)
        if ref_entry is not None:
            ref_paths, ref = ref_entry
            if len(ref_paths) >= 2:
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
        _pca_ref_store(paths, coords)
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
    log.info("t-SNE: using RAPIDS cuML (CUDA)")
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
    log.info("t-SNE: using mlx-vis (Metal)")
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
