"""Projection backends: PCA preview and cuML / mlx-vis / scikit-learn t-SNE.

Extracted from ``layout``. The PCA reference-alignment state lives here too; it
keeps successive PCA previews in a stable orientation so dots do not spin between
incremental refreshes. The t-SNE backends are 2-D by default with an opt-in 3-D
path (openTSNE or scikit-learn only); the 3-D namespaces its fit cache and
revision independently so 2-D and 3-D layouts never collide.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pickle
import platform
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .clap import inference_lock as _inference_lock
from .linalg_utils import apply_procrustes, procrustes, svd_singular
from .projection_config import (
    OPENTSNE_EARLY_EXAGGERATION_ITER,
    OPENTSNE_N_ITER,
    PROJECTION_RECIPE_VERSION,
)

log = logging.getLogger(__name__)

#: Bounded PCA reference-alignment LRU. Each entry is one (paths, coords)
#: snapshot; revisiting an earlier view re-aligns to that view's own
#: orientation instead of the last one computed.
_PCA_REF_CACHE_MAX = 4
_pca_ref_cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
_pca_ref_stats = {"hit": 0, "miss": 0, "evict": 0}


def _pca_ref_key(paths: list[str]) -> str:
    """Track-set digest for the PCA reference cache (shared with the layout cache)."""
    from .layout import track_set_key  # local: layout imports this module

    return track_set_key(paths)


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


def _project_pca(mat: np.ndarray, paths: list[str] | None = None, dims: int = 2) -> np.ndarray:
    centred = mat - mat.mean(axis=0)
    _, _, Vt = svd_singular(centred)
    coords = centred @ Vt[:dims].T

    if paths is not None and dims == 2:
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
                    # Rotation only: a PCA layout's scale is its own, so the fit
                    # aligns orientation and ignores the similarity's scale.
                    rot, _scale, a_mu, b_mu = procrustes(
                        coords[idx_new], ref[idx_ref]
                    )
                    coords = apply_procrustes(coords, rot, 1.0, a_mu, b_mu)
        _pca_ref_store(paths, coords)
    return coords


_cuml_projection_enabled: bool | None = None
_mlx_vis_projection_enabled: bool | None = None
_opentsne_projection_enabled: bool | None = None


def reset_projection_backend_cache() -> None:
    global _cuml_projection_enabled, _mlx_vis_projection_enabled
    global _opentsne_projection_enabled
    _cuml_projection_enabled = None
    _mlx_vis_projection_enabled = None
    _opentsne_projection_enabled = None


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


def _sklearn_tsne_pca_dim(n_features: int) -> int | None:
    """PCA width to pre-reduce to before the scikit-learn t-SNE fallback.

    Raw composite vectors are ~2,000-D; Barnes-Hut t-SNE on that directly is
    minutes-slow at library scale, long enough that the real layout never lands and
    the map stays on the PCA preview. The openTSNE path already reduces to 50, so
    mirror it here. Set ``TRACKSPACE_SKLEARN_TSNE_PCA_DIM=0`` to disable.
    """
    raw = os.environ.get("TRACKSPACE_SKLEARN_TSNE_PCA_DIM", "50").strip()
    try:
        dim = int(raw)
    except ValueError:
        dim = 50
    if dim <= 0 or n_features < 2:
        return None
    return max(2, min(dim, n_features - 1))


def _project_tsne_fallback(mat: np.ndarray, dims: int = 2) -> np.ndarray:
    # 3-D is openTSNE/sklearn only: cuML and mlx-vis are 2-D-only (and the mlx-vis
    # 3-D path is unverified), so go straight to the sklearn Barnes-Hut fallback.
    if dims == 3:
        return _sklearn_tsne(mat, n_components=3)
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
                # Pass the *raw* composite: mlx-vis runs its own PCA and its
                # branch raises `UnboundLocalError` on an already-reduced float32
                # matrix, so pre-reducing here would break the fallback.
                x = np.asarray(mat, dtype=np.float32)
                with _inference_lock:
                    y = reducer.fit_transform(x)
                return np.asarray(y, dtype=np.float32)
            except Exception as e:
                log.warning("mlx-vis t-SNE failed; using scikit-learn: %s", e)
                _mlx_vis_projection_enabled = False
    return _sklearn_tsne(mat, n_components=2)


def _sklearn_tsne(mat: np.ndarray, n_components: int) -> np.ndarray:
    """scikit-learn t-SNE after the same PCA pre-reduction as the openTSNE path.

    Raw composite vectors are ~2,000-D; Barnes-Hut t-SNE on that directly is
    minutes-slow at library scale, so the matrix is reduced first
    (``TRACKSPACE_SKLEARN_TSNE_PCA_DIM``, default 50; 0 disables). Used by the
    2-D fallback chain and the whole 3-D path.
    """
    from sklearn.manifold import TSNE  # lazy — heavy import
    pca_dim = _sklearn_tsne_pca_dim(int(np.asarray(mat).shape[1]))
    reduced = mat if pca_dim is None else _pca_reduce(mat, pca_dim)[0]
    perplexity = min(30.0, max(5.0, (len(reduced) - 1) / 3.0))
    kwargs: dict = {
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": 42,
    }
    if n_components == 3:
        # sklearn would auto-select the exact method for 3-D; Barnes-Hut is the
        # fast path the 2-D fallback uses, so pin it explicitly for 3-D too.
        kwargs["method"] = "barnes_hut"
    reducer = TSNE(**kwargs)
    return reducer.fit_transform(reduced)


# ─── openTSNE incremental t-SNE (prototype) ───────────────────
#
# openTSNE can place *new* points into an already-fitted embedding
# (``TSNEEmbedding.transform``) instead of refitting the whole library. That
# makes the common update — a generation batch finishes and only a few dozen
# tracks are new — cost seconds rather than a full refit.
#
# On by default: it is the only backend that can *place* new tracks into an
# already-fitted chart; see the projection section of ``audio_pipeline.md`` for
# the neighbourhood-recall measurements. Falls back to the cuML / mlx-vis /
# sklearn chain when unavailable, above the point cap, or on any error.
# CPU-only, so it needs no ``inference_lock`` (unlike mlx-vis).

#: Fitted embeddings kept in memory, one per vector config. Five projection
#: lenses share one config per lens, so the cap must exceed the lens count
#: (``projection_config.PROJECTION_LENSES``) or a full lens cycle evicts the
#: oldest fit and refits it on the next pass.
_OPENTSNE_FIT_CACHE_MAX = 8
_tsne_fit_cache: OrderedDict[str, "_TsneFit"] = OrderedDict()
_tsne_fit_stats = {"fit": 0, "transform": 0, "reuse": 0, "evict": 0}
#: The fit cache is mutated from Flask request threads and the layout-prefetch
#: worker, so an unguarded `move_to_end` / eviction could corrupt it.
_fit_cache_lock = threading.RLock()


@dataclass
class _TsneFit:
    """A fitted openTSNE embedding plus the PCA basis that feeds it."""

    paths: list[str]
    coords: np.ndarray  # coords (2-D or 3-D) aligned with ``paths``
    pca: object  # fitted sklearn PCA, for projecting new rows
    embedding: object  # fitted openTSNE TSNEEmbedding, for .transform()


def reset_tsne_fit_cache() -> None:
    """Drop fitted embeddings and zero the counters (tests, config changes)."""
    with _fit_cache_lock:
        _tsne_fit_cache.clear()
    for key in _tsne_fit_stats:
        _tsne_fit_stats[key] = 0


def tsne_fit_stats() -> dict[str, int]:
    """Fit / transform / reuse / eviction counters for the openTSNE cache."""
    return dict(_tsne_fit_stats)


def _load_persisted_fit(feature_cache: object, key: str) -> "_TsneFit | None":
    """Read the fitted embedding persisted for *key*, or ``None``.

    The in-process fit cache dies with the process, so without this a restart
    refits every lens (~20 s each) even though the layout coordinates are
    already cached. An unreadable or older blob reads as a miss: a library
    upgrade simply refits rather than loading an incompatible pickle.
    """
    getter = getattr(feature_cache, "get_tsne_fit", None)
    if getter is None:
        return None
    blob = getter(key)
    if not blob:
        return None
    try:
        fit = pickle.loads(blob)
    except Exception:
        log.warning("openTSNE fit: unreadable persisted fit for %s", key)
        return None
    return fit if isinstance(fit, _TsneFit) else None


def _store_persisted_fit(feature_cache: object, key: str, fit: "_TsneFit") -> None:
    """Persist *fit* under *key* (best-effort; the blob is opaque to the cache)."""
    putter = getattr(feature_cache, "put_tsne_fit", None)
    if putter is None:
        return
    try:
        putter(key, pickle.dumps(fit, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        log.debug("openTSNE fit: could not persist", exc_info=True)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _use_opentsne() -> bool:
    global _opentsne_projection_enabled
    if _opentsne_projection_enabled is not None:
        return _opentsne_projection_enabled
    if not _env_flag("TRACKSPACE_OPENTSNE", "1"):
        _opentsne_projection_enabled = False
        return False
    if importlib.util.find_spec("openTSNE") is None:
        _opentsne_projection_enabled = False
        return False
    _opentsne_projection_enabled = True
    log.info("t-SNE: using openTSNE (incremental)")
    return True


def _opentsne_pca_dim(n_features: int) -> int:
    raw = os.environ.get("TRACKSPACE_OPENTSNE_PCA_DIM", "50").strip()
    try:
        dim = int(raw)
    except ValueError:
        dim = 50
    return max(2, min(dim, max(2, n_features - 1)))


def _opentsne_max_points() -> int:
    raw = os.environ.get("TRACKSPACE_OPENTSNE_MAX_POINTS", "20000").strip()
    try:
        return int(raw)
    except ValueError:
        return 20_000


def _opentsne_refit_fraction() -> float:
    raw = os.environ.get("TRACKSPACE_OPENTSNE_REFIT_FRACTION", "0.3").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.3


def _tsne_perplexity(n: int) -> float:
    return min(30.0, max(5.0, (n - 1) / 3.0))


def _pca_reduce(mat: np.ndarray, dim: int) -> tuple[np.ndarray, object]:
    """Fit PCA to *dim* components; return (reduced, fitted PCA)."""
    from sklearn.decomposition import PCA  # lazy — heavy import

    n_samples, n_features = mat.shape
    k = max(1, min(dim, n_features - 1, n_samples - 1))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=42)
    return pca.fit_transform(mat), pca


def _fit_tsne(reduced: np.ndarray, dims: int = 2) -> tuple[object, np.ndarray]:
    from openTSNE import TSNE  # lazy — heavy import

    kwargs: dict = {
        "n_components": dims,
        "perplexity": _tsne_perplexity(len(reduced)),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": False,
        "n_iter": OPENTSNE_N_ITER,
        "early_exaggeration_iter": OPENTSNE_EARLY_EXAGGERATION_ITER,
    }
    if dims == 3:
        # openTSNE auto-selects negative_gradient_method, but the fft it picks
        # above 10k points is 2-D only — pin Barnes-Hut for 3-D explicitly.
        kwargs["negative_gradient_method"] = "bh"
    tsne = TSNE(**kwargs)
    embedding = tsne.fit(reduced)
    return embedding, np.asarray(embedding, dtype=np.float32)


def _fit_cache_key(config_key: str, dims: int = 2) -> str:
    """Namespace the fit cache so a recipe change cannot reuse a stale fit.

    Bumping ``PROJECTION_RECIPE_VERSION`` changes this suffix, so an embedding
    fitted under an older recipe — in memory or persisted on disk — is a miss
    and refits instead of being reused. A 3-D fit appends ``|dims=3`` so a 2-D
    and a 3-D embedding for the same vector config never share a key; 2-D keys
    are byte-identical to before the dims parameter existed.
    """
    key = f"{config_key}|opentsne_recipe={PROJECTION_RECIPE_VERSION}"
    return key if dims == 2 else f"{key}|dims=3"


def _fit_cache_get(key: str) -> "_TsneFit | None":
    with _fit_cache_lock:
        entry = _tsne_fit_cache.get(key)
        if entry is not None:
            _tsne_fit_cache.move_to_end(key)
        return entry


def _fit_cache_put(key: str, fit: "_TsneFit") -> None:
    with _fit_cache_lock:
        _tsne_fit_cache[key] = fit
        _tsne_fit_cache.move_to_end(key)
        while len(_tsne_fit_cache) > _OPENTSNE_FIT_CACHE_MAX:
            _tsne_fit_cache.popitem(last=False)
            _tsne_fit_stats["evict"] += 1


def _project_tsne_incremental(
    mat: np.ndarray,
    paths: list[str],
    config_key: str,
    *,
    cache_fit: bool = True,
    feature_cache: object = None,
    dims: int = 2,
) -> np.ndarray:
    """Fit once per vector config, then transform only newly added tracks.

    *cache_fit* is False for the throwaway fits of the weight Jacobian, whose
    one-off configs would otherwise evict the live fit the map needs. The fit is
    also persisted under *config_key* (via *feature_cache*) so a restart can
    transform rather than refit; only the config decides that key, so adding tracks
    still reuses it.
    """
    fit_key = _fit_cache_key(config_key, dims)
    fit = _fit_cache_get(fit_key)
    if fit is None and cache_fit:
        fit = _load_persisted_fit(feature_cache, fit_key)
        if fit is not None:
            _fit_cache_put(fit_key, fit)
    if fit is not None:
        index = {p: i for i, p in enumerate(fit.paths)}
        new_rows = [i for i, p in enumerate(paths) if p not in index]
        if not new_rows:
            # Same track set (e.g. a source toggle changed nothing here):
            # slice the cached coordinates, no compute at all.
            _tsne_fit_stats["reuse"] += 1
            return np.asarray([fit.coords[index[p]] for p in paths], dtype=np.float32)
        if len(new_rows) <= _opentsne_refit_fraction() * max(1, len(fit.paths)):
            coords = np.empty((len(paths), dims), dtype=np.float32)
            for i, p in enumerate(paths):
                known = index.get(p)
                if known is not None:
                    coords[i] = fit.coords[known]
            new_reduced = fit.pca.transform(mat[new_rows])
            partial = fit.embedding.transform(new_reduced)
            coords[new_rows] = np.asarray(partial, dtype=np.float32)
            # ``transform`` recentres the fitted embedding in place; re-read it so
            # the cached reference frame stays consistent for the next update.
            fit.coords = np.asarray(fit.embedding, dtype=np.float32)
            _tsne_fit_stats["transform"] += 1
            return coords

    reduced, pca = _pca_reduce(mat, _opentsne_pca_dim(mat.shape[1]))
    embedding, coords = _fit_tsne(reduced, dims=dims)
    if cache_fit:
        fit = _TsneFit(list(paths), coords, pca, embedding)
        _fit_cache_put(fit_key, fit)
        _store_persisted_fit(feature_cache, fit_key, fit)
    _tsne_fit_stats["fit"] += 1
    return coords


def _project_tsne(
    mat: np.ndarray,
    paths: list[str] | None = None,
    config_key: str | None = None,
    *,
    cache_fit: bool = True,
    feature_cache: object = None,
    dims: int = 2,
) -> np.ndarray:
    if paths and config_key and _use_opentsne():
        cap = _opentsne_max_points()
        if cap <= 0 or len(mat) <= cap:
            try:
                return _project_tsne_incremental(
                    mat,
                    paths,
                    config_key,
                    cache_fit=cache_fit,
                    feature_cache=feature_cache,
                    dims=dims,
                )
            except Exception as e:
                log.warning("openTSNE failed; using fitted-backend chain: %s", e)
    return _project_tsne_fallback(mat, dims=dims)
