"""Shared embedding-space layout: composite vectors and 2-D projection.

CLAP model I/O lives in `clap.py`. Librosa descriptors live in
`librosa_audio_features.py`. This module concatenates cached source rows
and orchestrates projection; semantic re-weighting lives in `semantic_weighting` and
the PCA / t-SNE backends in `projection_backends`. Re-exported from
``backend.embeddings``.
"""

import hashlib
import logging
import time
from collections import OrderedDict
import numpy as np

from .clap import (
    CLAP_SR,
    EMBEDDING_VERSION,
    batch_ensure_embeddings,
    generate_embedding,
    generate_text_embeddings,
    is_model_ready,
    load_model,
    _load_audio_segments,
)
from .projection_backends import (
    _project_pca,
    _project_tsne,
    reset_projection_backend_cache,
)
from .semantic_weighting import (
    _apply_low_rank_semantic_weight,
    _build_semantic_basis,
    _folder_seeds_from_track_paths,
    _semantic_basis_param_key,
)

log = logging.getLogger(__name__)

# Tiered projection LRU (composite gather → semantic direction basis → t-SNE).
_COMPOSITE_CACHE_MAX = 24
_SEMANTIC_BASIS_CACHE_MAX = 16
_REVISION_LAYOUT_CACHE_MAX = 32
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


def _supervised_cache_version(feature_cache) -> int:
    """Effective cache version for the ``supervised`` source.

    Supervised rows are versioned by the *training-run id* rather than a static
    recipe constant, because a retrain rewrites every row in place. ``0`` means no
    head has been trained, so the source is absent.
    """
    if feature_cache is None or not hasattr(feature_cache, "get_meta"):
        return 0
    from .supervised import cache_version

    return cache_version(feature_cache)


def _source_versions_token(feature_cache=None) -> str:
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_VERSION
    from .rhythm_features import RHYTHM_VERSION

    return (
        f"C{EMBEDDING_VERSION}F{FEATURES_VERSION}"
        f"M{MAEST_VERSION}R{RHYTHM_VERSION}"
        f"S{_supervised_cache_version(feature_cache)}"
    )


def _composite_config_token(
    sources: tuple[str, ...],
    mask: np.ndarray,
    blend: float,
    weights: dict[str, float] | None = None,
    versions_token: str = "",
) -> str:
    """Vector-config identity, independent of the *set* of tracks.

    The openTSNE fit cache keys on this: adding tracks must not invalidate a
    fitted embedding, while changing sources / mask / blend / weights must.
    """
    wk = (
        ",".join(f"{s}:{float(weights.get(s, 1.0)):.6g}" for s in sources)
        if weights
        else "-"
    )
    return (
        f"{versions_token}|{','.join(sources)}|"
        f"{mask.tobytes().hex()}|{blend:.6g}|{wk}"
    )


def _composite_param_key(
    paths: list[str],
    sources: tuple[str, ...],
    mask: np.ndarray,
    blend: float,
    weights: dict[str, float] | None = None,
    versions_token: str = "",
) -> str:
    ph = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:24]
    return f"{ph}|{_composite_config_token(sources, mask, blend, weights, versions_token)}"


def _source_version(source: str, feature_cache=None) -> int:
    """Cache version for one source (used in row-cache keys and reads).

    For ``supervised`` this is the current training-run id, so a retrain
    invalidates the in-process composite and row caches.
    """
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_VERSION
    from .rhythm_features import RHYTHM_VERSION

    return {
        "clap": EMBEDDING_VERSION,
        "features": FEATURES_VERSION,
        "maest": MAEST_VERSION,
        "maest_logits": MAEST_VERSION,
        "rhythm": RHYTHM_VERSION,
        "supervised": _supervised_cache_version(feature_cache),
    }[source]


def _get_prepared_source_row(
    source: str,
    fp: str,
    raw: np.ndarray,
    feature_mask: np.ndarray,
    features_blend: float,
    feature_cache=None,
) -> np.ndarray:
    """Return one track row for *source* with the same preprocessing as composite gather.

    Every source is L2-normalised so a high-dimensional block (768-D MAEST)
    cannot dominate a low-dimensional one (14-D rhythm) through sheer length.
    Relative influence is then set explicitly by the per-source weights applied
    in :func:`_gather_composite_vecs`, not by dimensionality.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)

    if source == "features":
        b = float(np.clip(features_blend, 0.05, 1.0))
        key = (
            "features",
            fp,
            _source_version(source, feature_cache),
            feature_mask.tobytes(),
            round(b, 9),
        )
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

    try:
        version = _source_version(source, feature_cache)
    except KeyError:
        raise ValueError(f"unknown source {source!r}") from None

    key = (source, fp, version)
    hit = _source_row_cache_get(key)
    if hit is not None:
        return hit
    n = float(np.linalg.norm(raw))
    if n < 1e-9:
        n = 1.0
    v = (raw / n).astype(np.float32)
    _source_row_cache_put(key, v)
    return v


def _gather_vecs(
    track_infos: list[dict],
    feature_cache,
    version: int,
) -> tuple[list[str], list[np.ndarray]]:
    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]
    embeddings_map = feature_cache.get_all("clap", fp_set, version=version)

    paths: list[str] = []
    vecs: list[np.ndarray] = []
    for t in track_infos:
        fp = t.get("fingerprint")
        if fp and fp in embeddings_map:
            paths.append(t["path"])
            vecs.append(embeddings_map[fp])
    return paths, vecs


def _fetch_source_map(
    source: str, feature_cache, fp_set: list[str]
) -> dict[str, np.ndarray]:
    """One batched cache read for *source*, keyed by its own version constant.

    A cache object without the generic getter (older cache, or a test stub) yields
    an empty map rather than raising — that source is simply absent for this
    projection.
    """
    try:
        version = _source_version(source, feature_cache)
    except KeyError:
        raise ValueError(f"unknown source {source!r}") from None
    getter = getattr(feature_cache, "get_all", None)
    if getter is None:
        return {}
    return getter(source, fp_set, version=version)


def _gather_composite_vecs(
    track_infos: list[dict],
    feature_cache,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    source_weights: dict[str, float] | None = None,
    required_sources: tuple[str, ...] | None = None,
) -> tuple[list[str], np.ndarray, int, int, str, str]:
    """Stack cached per-source rows into one weighted matrix.

    A track is placed on the map when it has every *required_sources* entry;
    optional sources it lacks contribute a zero row, so partially analysed
    libraries still render instead of hiding tracks until every model has run.

    Per-source weights are applied as ``sqrt(w)`` because squared Euclidean
    distance is additive across the concatenated blocks — scaling a block by
    ``sqrt(w)`` makes it contribute exactly ``w ×`` its distance (plan §8).
    """
    from .librosa_audio_features import AUDIO_FEATURE_DIM

    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]

    source_maps: dict[str, dict[str, np.ndarray]] = {
        s: _fetch_source_map(s, feature_cache, fp_set) for s in sources
    }

    required = tuple(required_sources) if required_sources else tuple(sources)
    # Only require sources that are actually enabled for this projection.
    required = tuple(s for s in required if s in sources)

    paths: list[str] = []
    fp_order: list[str] = []
    for t in track_infos:
        fp = t.get("fingerprint")
        if not fp:
            continue
        if all(fp in source_maps.get(s, {}) for s in required):
            paths.append(t["path"])
            fp_order.append(fp)

    if not paths:
        return paths, np.empty((0, 0), dtype=np.float32), 0, 0, "", ""

    if feature_mask is None:
        mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)
    else:
        mask = np.asarray(feature_mask, dtype=np.float32).reshape(-1)
        if mask.size != AUDIO_FEATURE_DIM:
            mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)

    blend = float(np.clip(features_blend, 0.05, 1.0))
    weights = dict(source_weights or {})
    versions_token = _source_versions_token(feature_cache)
    comp_key = _composite_param_key(
        paths, sources, mask, blend, weights, versions_token
    )
    comp_config = _composite_config_token(
        sources, mask, blend, weights, versions_token
    )
    cached = _tier_cache_get(_composite_cache, comp_key)
    if cached is not None:
        _p, _mat, _cd, _clo = cached
        log.debug("projection tier: composite cache hit")
        return paths, _mat, _cd, _clo, comp_key, comp_config

    blocks: list[np.ndarray] = []
    clap_dim = 0
    clap_col_lo = 0
    col = 0
    for s in sources:
        smap = source_maps.get(s, {})
        if not smap:
            continue
        weight = float(weights.get(s, 1.0)) if weights else 1.0
        if weight <= 0.0:
            continue

        dim = _source_block_dim(s, smap)
        if dim <= 0:
            continue
        zero = np.zeros(dim, dtype=np.float32)
        rows = []
        for fp in fp_order:
            raw = smap.get(fp)
            if raw is None:
                rows.append(zero)
            else:
                rows.append(
                    _get_prepared_source_row(
                        s, fp, raw, mask, blend, feature_cache
                    )
                )
        block = np.stack(rows, axis=0)
        if weight != 1.0:
            block = (block * float(np.sqrt(weight))).astype(np.float32)

        if s == "clap":
            clap_dim = block.shape[1]
            clap_col_lo = col
        blocks.append(block)
        col += block.shape[1]

    if not blocks:
        return paths, np.empty((0, 0), dtype=np.float32), 0, 0, "", comp_config

    mat = np.concatenate(blocks, axis=1).astype(np.float32)
    _tier_cache_put(
        _composite_cache,
        comp_key,
        (list(paths), mat, clap_dim, clap_col_lo),
        _COMPOSITE_CACHE_MAX,
    )
    return paths, mat, clap_dim, clap_col_lo, comp_key, comp_config


def _source_block_dim(source: str, smap: dict[str, np.ndarray]) -> int:
    """Width of one source block, taken from the first cached row."""
    for v in smap.values():
        if v is not None:
            return int(np.asarray(v).reshape(-1).size)
    return 0


def _normalise_coords(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    return (coords - mins) / ranges


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
    method: str = "tsne",
    context_tags: list[str] | None = None,
    context_folders: list[str] | None = None,
    scale_folders: bool = False,
    folder_boost: float = 3.0,
    folder_depth_boost: float = 1.5,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    layout_revision: str | None = None,
    source_weights: dict[str, float] | None = None,
    required_sources: tuple[str, ...] | None = None,
) -> list[dict]:
    if layout_revision and method == "tsne":
        hit = _tier_cache_get(_revision_layout_cache, layout_revision)
        if hit is not None:
            log.debug("projection: revision cache hit (tsne)")
            return hit

    t0 = time.perf_counter()
    paths, mat, clap_dim, clap_col_lo, comp_key, comp_config = _gather_composite_vecs(
        track_infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        source_weights=source_weights,
        required_sources=required_sources,
    )
    t_after_gather = time.perf_counter()

    # `mat.shape[0] < 2` guards the degenerate all-zero-weight case, where every
    # source block is dropped and no row can be projected. The HTTP routes reject
    # that vector before reaching here; this keeps the function safe regardless.
    if len(paths) < 2 or mat.shape[0] < 2:
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
    basis_key = ""
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
    else:
        if method != "tsne":
            log.debug("unknown projection method %r; using t-SNE", method)
        # The fit cache keys on the vector config (not the track set), so an
        # openTSNE embedding fitted for the same sources/mask/weighting is reused
        # and only newly added tracks are transformed into it.
        tsne_config = comp_config
        if needs_weight:
            tsne_config = (
                f"{comp_config}|{basis_key}"
                f"|{folder_boost:.6g}|{folder_depth_boost:.6g}"
            )
        coords = _project_tsne(mat_proj, paths=paths, config_key=tsne_config)
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

    if layout_revision and method == "tsne":
        _tier_cache_put(
            _revision_layout_cache, layout_revision, result, _REVISION_LAYOUT_CACHE_MAX,
        )

    return result
