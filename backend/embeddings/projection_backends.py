"""2-D projection backends: PCA preview and cuML / mlx-vis / scikit-learn t-SNE.

Extracted from ``layout``. The PCA reference-alignment state lives here too; it
keeps successive PCA previews in a stable orientation so dots do not spin between
incremental refreshes.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys

import numpy as np

from .clap import inference_lock as _inference_lock

log = logging.getLogger(__name__)

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
