"""Shared embedding-space layout: composite vectors and projection.

CLAP model I/O lives in `clap.py`. Librosa descriptors live in
`librosa_audio_features.py`. This module concatenates cached source rows
and orchestrates projection; semantic re-weighting lives in `semantic_weighting` and
the PCA / t-SNE backends in `projection_backends`. Re-exported from
``backend.embeddings``.
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

from .clap import (
    CLAP_SR,
    EMBEDDING_VERSION,
    batch_ensure_embeddings,
    generate_embedding,
    generate_text_embeddings,
    is_model_ready,
    load_error,
    load_model,
    _load_audio_segments,
)
from .feature_encoding import FEATURE_ENCODING_VERSION, encode_feature_vector
from .layout_revision import GAUGE_3D_VERSION
from .lens_gauge import (
    GAUGE_VERSION2,
    MIN_SHARED,
    gauge_layout,
    gauge_layout_3d,
    has_reference,
    seed_reference,
)
from .linalg_utils import matched_indices
from .projection_backends import (
    _project_pca,
    _project_tsne,
    reset_projection_backend_cache,
)
from .projection_config import (
    FROZEN_FOLDER_BASIS_MODE,
    FROZEN_FOLDER_BOOST,
    FROZEN_FOLDER_DEPTH_BOOST,
    recipe_for,
    tsne_iteration_suffix,
)
from .semantic_weighting import (
    _apply_low_rank_semantic_weight,
    _build_semantic_basis,
    _folder_seeds_from_track_paths,
    _semantic_basis_param_key,
    validate_folder_basis_mode,
)

log = logging.getLogger(__name__)

# Tiered projection LRU (composite gather → semantic direction basis → t-SNE).
_COMPOSITE_CACHE_MAX = 24
#: Byte budget for the composite matrices. A combined-lens matrix is ~101 MB at
#: 14k tracks, so 24 entries could reach ~2.4 GB; evict by total bytes too.
_COMPOSITE_CACHE_BYTES_MAX = 512 * 1024 * 1024
_SEMANTIC_BASIS_CACHE_MAX = 16
_REVISION_LAYOUT_CACHE_MAX = 32
_composite_cache: OrderedDict[str, tuple[list[str], np.ndarray, int, int]] = (
    OrderedDict()
)
_semantic_basis_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
    OrderedDict()
)
_revision_layout_cache: OrderedDict[str, list[dict]] = OrderedDict()
_composite_cache_bytes = 0

#: Byte budget for the prepared source-row LRU. The working set is ~42k rows
#: (7.1k tracks × 6 sources), so an entry cap thrashes a full pass and a weight
#: change recomputes every row; a byte budget keeps the whole set resident.
_SOURCE_ROW_CACHE_BYTES_MAX = 256 * 1024 * 1024
#: Per-entry Python/ndarray object overhead, so tiny rows (rhythm 56 B,
#: features 24 B) cannot grow the entry count without bound.
_SOURCE_ROW_ENTRY_OVERHEAD = 112
_source_row_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_source_row_cache_bytes = 0

#: Guards the module-level tier caches. They are mutated from Flask request
#: threads *and* the layout-prefetch worker, so an unguarded OrderedDict could
#: be corrupted by a concurrent `move_to_end`/eviction.
_cache_lock = threading.RLock()


def _source_row_cache_get(key: tuple) -> np.ndarray | None:
    with _cache_lock:
        if key not in _source_row_cache:
            return None
        _source_row_cache.move_to_end(key)
        return _source_row_cache[key]


def _source_row_cache_put(key: tuple, value: np.ndarray) -> None:
    """Byte-budgeted put for the prepared source-row LRU."""
    global _source_row_cache_bytes
    with _cache_lock:
        old = _source_row_cache.get(key)
        if old is not None:
            _source_row_cache_bytes -= int(old.nbytes) + _SOURCE_ROW_ENTRY_OVERHEAD
        _source_row_cache[key] = value
        _source_row_cache.move_to_end(key)
        _source_row_cache_bytes += int(value.nbytes) + _SOURCE_ROW_ENTRY_OVERHEAD
        while (
            _source_row_cache_bytes > _SOURCE_ROW_CACHE_BYTES_MAX
            and len(_source_row_cache) > 1
        ):
            _k, old_v = _source_row_cache.popitem(last=False)
            _source_row_cache_bytes -= int(old_v.nbytes) + _SOURCE_ROW_ENTRY_OVERHEAD


def _source_row_cache_clear() -> None:
    """Empty the prepared-row cache and reset its byte counter."""
    global _source_row_cache_bytes
    with _cache_lock:
        _source_row_cache.clear()
        _source_row_cache_bytes = 0


def _tier_cache_get(cache: OrderedDict, key: str):
    with _cache_lock:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _tier_cache_put(cache: OrderedDict, key: str, value, max_size: int) -> None:
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def _composite_cache_put(key: str, value) -> None:
    """Byte-budgeted put for the composite matrix tier cache."""
    global _composite_cache_bytes
    with _cache_lock:
        old = _composite_cache.get(key)
        if old is not None:
            _composite_cache_bytes -= int(old[1].nbytes)
        _composite_cache[key] = value
        _composite_cache.move_to_end(key)
        _composite_cache_bytes += int(value[1].nbytes)
        while (
            len(_composite_cache) > _COMPOSITE_CACHE_MAX
            or (
                _composite_cache_bytes > _COMPOSITE_CACHE_BYTES_MAX
                and len(_composite_cache) > 1
            )
        ):
            _k, old_v = _composite_cache.popitem(last=False)
            _composite_cache_bytes -= int(old_v[1].nbytes)


#: Metadata prefix for the persisted gauge anchor of a stable family. The value is
#: the layout revision whose persisted row is that family's reference frame.
_GAUGE_ANCHOR_META_PREFIX = "gauge_anchor"


def _gauge_scope_for(paths: list[str], gauge_scope: str | None) -> str:
    """The stable scope that identifies a gauge family.

    An explicit *gauge_scope* is used verbatim: the route, status and prefetch pass
    the configured library roots as ``library:<id>|<id>``. That is the only scope
    that stays stable when the configured roots change independently of the visible
    tracks, and it is part of the coordinate revision, so it must come from the
    configuration rather than be re-derived here.

    The fallback infers a scope from the paths, deliberately *not* from the exact
    track set: a folder subset and the whole library slice must share one frame.
    Virtual paths encode their root as the first component, which is stable under
    subsetting. Absolute paths have no root id — their first component is empty for
    every volume, so keying on it would put unrelated libraries in one frame — and
    fall back to their common ancestor directory instead. That ancestor narrows as
    the path set narrows, so without configured roots a folder subset may fork the
    family rather than join a foreign one; callers with configured roots must pass
    the explicit scope to keep subsets in one frame.
    """
    if gauge_scope is not None:
        return str(gauge_scope)
    roots = {
        str(p).split("/", 1)[0] for p in paths if p and not os.path.isabs(str(p))
    }
    absolute_dirs = sorted({
        os.path.dirname(str(p)) for p in paths if os.path.isabs(str(p))
    })
    if absolute_dirs:
        try:
            roots.add("abs:" + os.path.commonpath(absolute_dirs))
        except ValueError:
            # Mixed drives/anchors: no trustworthy common prefix exists.
            roots.add("abs:")
    return "\n".join(sorted(roots))


def _gauge_family(
    gauge_scope: str | None,
    paths: list[str],
    feature_cache,
    dims: int,
) -> str:
    """Stable family key shared by every lens over the same library scope.

    The source-version part uses the supervised-source generations (not the
    lens's own subset), so switching between ``clap`` and ``genre_full`` does not
    fork the family. 2-D and 3-D have separate namespaces; each carries the gauge
    semantics generation so a version bump never reuses an old frame.
    """
    token = _source_versions_token(
        feature_cache, sources=("maest_layer7", "mert_l19")
    )
    namespace = (
        f"2d{GAUGE_VERSION2}"
        if dims == 2
        else f"3d{GAUGE_VERSION2}.{GAUGE_3D_VERSION}"
    )
    return f"{namespace}|{_gauge_scope_for(paths, gauge_scope)}|{token}"


def _positions_array(
    positions: list[dict], dims: int
) -> tuple[list[str], np.ndarray] | None:
    """Paths plus a float64 coordinate matrix from cached positions, or ``None``."""
    if not positions:
        return None
    if dims == 3 and not all("z" in p for p in positions):
        return None
    paths = [str(p["path"]) for p in positions]
    try:
        coords = np.asarray(
            [
                [float(p["x"]), float(p["y"])]
                + ([float(p["z"])] if dims == 3 else [])
                for p in positions
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if coords.shape != (len(paths), dims) or not np.all(np.isfinite(coords)):
        return None
    return paths, coords


def _anchor_meta_key(family: str) -> str:
    return f"{_GAUGE_ANCHOR_META_PREFIX}:{family}"


def _ensure_gauge_reference(
    family: str,
    feature_cache,
    *,
    positions: list[dict] | None = None,
    dims: int = 2,
) -> None:
    """Restore *family*'s reference frame before aligning a layout.

    The persisted anchor (metadata → layout row) wins; if it is missing or its row
    was deleted, the just-loaded *positions* seed the in-memory reference so the
    family stays coherent until the next store replaces the pointer. Seeding never
    overwrites a live in-memory reference.
    """
    if has_reference(family):
        return
    get_meta = getattr(feature_cache, "get_meta", None)
    get_layout = getattr(feature_cache, "get_layout", None)
    revision = get_meta(_anchor_meta_key(family)) if get_meta is not None else None
    if revision and get_layout is not None:
        loaded = _positions_array(get_layout(revision) or [], dims)
        if loaded is not None:
            seed_reference(family, *loaded)
            return
    if positions:
        loaded = _positions_array(positions, dims)
        if loaded is not None:
            seed_reference(family, *loaded)


def _store_gauge_anchor(
    family: str,
    feature_cache,
    revision: str | None,
    positions: list[dict],
    dims: int,
) -> None:
    """Point *family*'s metadata at *revision* when no usable anchor exists.

    Only a complete, fittable layout (>= ``MIN_SHARED`` tracks, finite coordinates
    of the right shape) may anchor. An existing pointer whose layout row is gone is
    replaced; a live one is kept, so the first valid layout stays the reference
    across restarts. A malformed row is skipped rather than persisted as the frame.
    """
    if not revision:
        return
    loaded = _positions_array(positions, dims)
    if loaded is None or len(loaded[0]) < MIN_SHARED:
        return
    get_meta = getattr(feature_cache, "get_meta", None)
    set_meta = getattr(feature_cache, "set_meta", None)
    get_layout = getattr(feature_cache, "get_layout", None)
    if set_meta is None:
        return
    key = _anchor_meta_key(family)
    current = get_meta(key) if get_meta is not None else None
    if current:
        if get_layout is None:
            return
        # A deleted cache row must not strand the family on a missing anchor.
        if get_layout(current):
            return
    set_meta(key, revision)


def revision_layout_cached(
    revision: str | None,
    feature_cache=None,
    *,
    gauge_scope: str | None = None,
    dims: int = 2,
) -> list[dict] | None:
    """The cached t-SNE layout for *revision*, or ``None``.

    In-process LRU first, then the persisted copy: the LRU dies with the process,
    so without the disk tier a restart re-fits every lens (~20 s each). Lets a
    caller short-circuit before taking the projection slot: an already-warmed lens
    must not wait behind a background fit for another lens.

    A hit also restores (or seeds) the stable family's gauge reference, so a
    restart that serves a persisted layout still aligns the *next* cold lens. The
    hit's coordinates are returned unchanged — they are already in the family
    frame and must never be re-normalised.
    """
    if not revision:
        return None
    hit = _tier_cache_get(_revision_layout_cache, revision)
    if hit is None:
        getter = getattr(feature_cache, "get_layout", None)
        if getter is None:
            return None
        hit = getter(revision)
        if not hit:
            return None
        _tier_cache_put(
            _revision_layout_cache, revision, hit, _REVISION_LAYOUT_CACHE_MAX
        )
    paths = [str(p["path"]) for p in hit]
    family = _gauge_family(gauge_scope, paths, feature_cache, dims)
    _ensure_gauge_reference(
        family, feature_cache, positions=hit, dims=dims
    )
    _store_gauge_anchor(family, feature_cache, revision, hit, dims)
    return hit


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


def _source_versions_token(
    feature_cache=None,
    *,
    supervised_version: int | None = None,
    sources: tuple[str, ...] | None = None,
) -> str:
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_LAYER7_VERSION, MAEST_VERSION
    from .mert import MERT_VERSION
    from .rhythm_features import RHYTHM_VERSION

    if supervised_version is None:
        supervised_version = _supervised_cache_version(feature_cache)

    token = (
        f"C{EMBEDDING_VERSION}F{FEATURES_VERSION}"
        f"M{MAEST_VERSION}R{RHYTHM_VERSION}"
        f"S{supervised_version}"
    )
    # A composite that consumes the opt-in genre_full blocks must also key on their
    # cache generations, or a MERT/layer-7 version bump would not invalidate it. The
    # default lens never includes them, so its token is byte-identical to before.
    if sources and ("maest_layer7" in sources or "mert_l19" in sources):
        token += f"L{MAEST_LAYER7_VERSION}E{MERT_VERSION}"
    return token


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


def track_set_key(paths: list[str], *, sort: bool = False) -> str:
    """Digest of the *track set* alone, independent of any lens configuration.

    Shared by the composite cache key, the lens gauge, and the landmark placement
    cache: they must key on the tracks alone, never on anything that varies per lens
    (sources, mask, weights) or per call (the order the paths arrive in).
    """
    if sort:
        paths = sorted(paths)
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:24]


def _composite_param_key(
    paths: list[str],
    sources: tuple[str, ...],
    mask: np.ndarray,
    blend: float,
    weights: dict[str, float] | None = None,
    versions_token: str = "",
) -> str:
    ph = track_set_key(paths)
    return f"{ph}|{_composite_config_token(sources, mask, blend, weights, versions_token)}"


def _static_source_version(source: str) -> int:
    """Cache version for a source with a fixed recipe constant.

    ``supervised`` is excluded: its version is the *training-run id*, which is
    dynamic and therefore resolved separately (see :func:`_source_version`).
    """
    from .librosa_audio_features import FEATURES_VERSION
    from .maest import MAEST_LAYER7_VERSION, MAEST_VERSION
    from .mert import MERT_VERSION
    from .rhythm_features import RHYTHM_VERSION

    return {
        "clap": EMBEDDING_VERSION,
        "features": FEATURES_VERSION,
        "maest": MAEST_VERSION,
        "maest_logits": MAEST_VERSION,
        "maest_layer7": MAEST_LAYER7_VERSION,
        "mert_l19": MERT_VERSION,
        "rhythm": RHYTHM_VERSION,
    }[source]


def _source_version(source: str, feature_cache=None) -> int:
    """Cache version for one source (used in row-cache keys and reads).

    For ``supervised`` this is the current training-run id, so a retrain
    invalidates the in-process composite and row caches. Only the supervised
    branch reads the cache (a metadata query); static sources never do.
    """
    if source == "supervised":
        return _supervised_cache_version(feature_cache)
    return _static_source_version(source)


def _get_prepared_source_row(
    source: str,
    fp: str,
    raw: np.ndarray,
    feature_mask: np.ndarray,
    features_blend: float,
    feature_cache=None,
    *,
    source_version: int | None = None,
) -> np.ndarray:
    """Return one track row for *source* with the same preprocessing as composite gather.

    Every source is L2-normalised so a high-dimensional block (768-D MAEST)
    cannot dominate a low-dimensional one (14-D rhythm) through sheer length.
    Relative influence is then set explicitly by the per-source weights applied
    in :func:`_gather_composite_vecs`, not by dimensionality.

    ``source_version`` is the already-snapped cache version for *source*; when
    omitted it is resolved here (the direct-call path).
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)

    if source == "features":
        b = float(np.clip(features_blend, 0.05, 1.0))
        if source_version is None:
            source_version = _source_version(source, feature_cache)
        key = (
            "features",
            fp,
            source_version,
            feature_mask.tobytes(),
            round(b, 9),
            FEATURE_ENCODING_VERSION,
        )
        hit = _source_row_cache_get(key)
        if hit is not None:
            return hit
        # Re-encode the cached raw row into the perceptual space before masking;
        # a row that cannot be encoded contributes a zero block, like a missing one.
        encoded = encode_feature_vector(raw)
        if encoded is None:
            encoded = np.zeros_like(raw)
        v = encoded * feature_mask
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            n = 1.0
        v = (v / n * b).astype(np.float32)
        _source_row_cache_put(key, v)
        return v

    if source_version is None:
        try:
            source_version = _source_version(source, feature_cache)
        except KeyError:
            raise ValueError(f"unknown source {source!r}") from None

    key = (source, fp, source_version)
    hit = _source_row_cache_get(key)
    if hit is not None:
        return hit
    n = float(np.linalg.norm(raw))
    if n < 1e-9:
        n = 1.0
    v = (raw / n).astype(np.float32)
    _source_row_cache_put(key, v)
    return v


def _fetch_source_map(
    source: str, feature_cache, fp_set: list[str], *, source_version: int | None = None
) -> dict[str, np.ndarray]:
    """One batched cache read for *source*, keyed by its own version constant.

    A cache object without the generic getter (older cache, or a test stub) yields
    an empty map rather than raising — that source is simply absent for this
    projection. ``source_version`` is the already-snapped version for *source*;
    when omitted it is resolved here (the direct-call path).
    """
    if source_version is None:
        try:
            source_version = _source_version(source, feature_cache)
        except KeyError:
            raise ValueError(f"unknown source {source!r}") from None
    getter = getattr(feature_cache, "get_all", None)
    if getter is None:
        return {}
    return getter(source, fp_set, version=source_version)


def _gather_composite_vecs(
    track_infos: list[dict],
    feature_cache,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    source_weights: dict[str, float] | None = None,
    required_sources: tuple[str, ...] | None = None,
) -> tuple[list[str], np.ndarray, int, int, str, str, dict[str, tuple[int, int]]]:
    """Stack cached per-source rows into one weighted matrix.

    A track is placed on the map when it has every *required_sources* entry;
    optional sources it lacks contribute a zero row, so partially analysed
    libraries still render instead of hiding tracks until every model has run.

    Per-source weights are applied as ``sqrt(w)`` because squared Euclidean
    distance is additive across the concatenated blocks — scaling a block by
    ``sqrt(w)`` makes it contribute exactly ``w ×`` its distance (plan §8).

    Also returns each source's ``(lo, hi)`` column span in the matrix, which the
    weight Jacobian differentiates against.
    """
    from .librosa_audio_features import AUDIO_FEATURE_DIM

    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]

    # Snapshot the supervised training-run id exactly once per gather. The token,
    # every supervised fetch and every row key reuse this same value, so a retrain
    # invalidates all composite/row caches coherently and every part of this
    # gather stays version-consistent under that single snapshot.
    supervised_version = _supervised_cache_version(feature_cache)
    source_versions: dict[str, int] = {}
    source_maps: dict[str, dict[str, np.ndarray]] = {}
    for s in sources:
        if s == "supervised":
            v = supervised_version
        else:
            try:
                v = _static_source_version(s)
            except KeyError:
                raise ValueError(f"unknown source {s!r}") from None
        source_versions[s] = v
        source_maps[s] = _fetch_source_map(
            s, feature_cache, fp_set, source_version=v
        )

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
        return paths, np.empty((0, 0), dtype=np.float32), 0, 0, "", "", {}

    if feature_mask is None:
        mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)
    else:
        mask = np.asarray(feature_mask, dtype=np.float32).reshape(-1)
        if mask.size != AUDIO_FEATURE_DIM:
            mask = np.ones(AUDIO_FEATURE_DIM, dtype=np.float32)

    blend = float(np.clip(features_blend, 0.05, 1.0))
    weights = dict(source_weights or {})
    versions_token = _source_versions_token(
        feature_cache, supervised_version=supervised_version, sources=sources
    )
    comp_key = _composite_param_key(
        paths, sources, mask, blend, weights, versions_token
    )
    comp_config = _composite_config_token(
        sources, mask, blend, weights, versions_token
    )
    cached = _tier_cache_get(_composite_cache, comp_key)
    if cached is not None:
        _p, _mat, _cd, _clo, _spans = cached
        log.debug("projection tier: composite cache hit")
        return paths, _mat, _cd, _clo, comp_key, comp_config, _spans

    blocks: list[np.ndarray] = []
    clap_dim = 0
    clap_col_lo = 0
    col = 0
    spans: dict[str, tuple[int, int]] = {}
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
                        s, fp, raw, mask, blend, feature_cache,
                        source_version=source_versions[s],
                    )
                )
        block = np.stack(rows, axis=0)
        if weight != 1.0:
            block = (block * float(np.sqrt(weight))).astype(np.float32)

        if s == "clap":
            clap_dim = block.shape[1]
            clap_col_lo = col
        spans[s] = (col, col + block.shape[1])
        blocks.append(block)
        col += block.shape[1]

    if not blocks:
        return paths, np.empty((0, 0), dtype=np.float32), 0, 0, "", comp_config, {}

    mat = np.concatenate(blocks, axis=1).astype(np.float32)
    _composite_cache_put(
        comp_key, (list(paths), mat, clap_dim, clap_col_lo, spans)
    )
    return paths, mat, clap_dim, clap_col_lo, comp_key, comp_config, spans


@dataclass
class ProjectionInput:
    """Composite matrix plus the semantic weighting applied to it.

    Produced by :func:`prepare_projection_input` and consumed by both the
    interactive projection and the offline layout-evaluation generator, so the
    two derive their projection input from exactly the same rules.
    """

    paths: list[str]
    mat: np.ndarray
    clap_dim: int
    clap_col_lo: int
    comp_key: str
    comp_config: str
    spans: dict[str, tuple[int, int]]
    mat_proj: np.ndarray
    tsne_config: str


def prepare_projection_input(
    track_infos: list[dict],
    feature_cache,
    *,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    source_weights: dict[str, float] | None = None,
    required_sources: tuple[str, ...] | None = None,
    scale_folders: bool = False,
    context_tags: list[str] | None = None,
    context_folders: list[str] | None = None,
    folder_boost: float = 3.0,
    folder_depth_boost: float = 1.5,
    folder_basis_mode: str = "full",
) -> ProjectionInput:
    """Gather the composite matrix and apply any semantic weighting.

    Builds the same composite matrix (per-source weighted blocks, required-source
    gating) and then re-projects the CLAP column span along semantic folder/tag
    directions when a model is ready and supervision is supplied. ``scale_folders``
    derives the folder seeds from the track paths when no explicit context folders
    are given. The defaults mirror the frozen evaluation path (full basis, 3.0 /
    1.5); production passes its canonical params. ``scale_folders=False`` and empty
    context mean no supervision, which is the evaluation default.

    The folder-weighting gate is structural: data-driven folder directions need
    no CLAP, and a recipe with no CLAP column has no text fallbacks, so
    folder-only weighting applies whenever there are no tags and no CLAP slot —
    regardless of whether the text tower has loaded. Tags and a CLAP column
    still require the model.

    ``tsne_config`` is the effective vector-config key the production t-SNE fit
    cache keys on (``comp_config`` plus the semantic-basis params when weighting
    is active). The evaluation generator ignores it for its own uncached fits but
    it is returned so callers that do use the production cache key identically.

    ``folder_basis_mode`` defaults to ``"full"`` for the frozen evaluation and
    benchmark paths; production passes ``FROZEN_FOLDER_BASIS_MODE``
    (``"centroids"``) explicitly from ``compute_projection``. ``"centroids"``
    keeps only the hierarchical contrast directions and skips the within-folder
    PC SVD; ``"approx"`` solves those PCs with a seeded randomized SVD.
    Non-default modes are namespaced into the semantic-basis key, never exposed
    by the API, and never aliased onto a cached full basis.

    A degenerate input — fewer than two tracks, or a matrix with fewer than two
    rows — returns the unweighted bundle with no semantic inference, since there
    is nothing meaningful to project or re-weight.
    """
    validate_folder_basis_mode(folder_basis_mode)
    paths, mat, clap_dim, clap_col_lo, comp_key, comp_config, spans = (
        _gather_composite_vecs(
            track_infos,
            feature_cache,
            sources=sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            source_weights=source_weights,
            required_sources=required_sources,
        )
    )

    if len(paths) < 2 or mat.shape[0] < 2:
        return ProjectionInput(
            paths, mat, clap_dim, clap_col_lo, comp_key, comp_config, spans,
            mat, comp_config,
        )

    if scale_folders:
        folder_seeds = (
            list(context_folders)
            if context_folders
            else _folder_seeds_from_track_paths(paths)
        )
    else:
        folder_seeds = None

    # CLAP is only needed when a text direction can actually enter the basis:
    # explicit tags, or a CLAP column that folder text-fallbacks could fill.
    # Data-driven folder directions are model-free, so a recipe with no CLAP slot
    # and no tags applies them even while the text tower has not loaded.
    allow_without_model = not context_tags and clap_dim <= 0
    needs_weight = (bool(context_tags) or bool(folder_seeds)) and (
        is_model_ready() or allow_without_model
    )
    mat_proj = mat
    basis_key = ""
    if needs_weight:
        # Key the semantic basis on the vector *config* and the folder *set*, never
        # the full path set: the frame is defined by the folders, so a track added
        # to an existing folder must reuse the fitted chart (and be transformed into
        # it) rather than refit the whole map. A new folder is a genuinely new
        # direction, so it does refit.
        basis_key = (
            f"{comp_config}"
            f"|{_semantic_basis_param_key(context_tags, folder_seeds, folder_basis_mode)}"
        )
        basis = _tier_cache_get(_semantic_basis_cache, basis_key)
        if basis is None:
            clap_hi = clap_col_lo + clap_dim
            tags_for_weight = context_tags if clap_dim > 0 else None
            T_n, exp = _build_semantic_basis(
                mat, paths, tags_for_weight, folder_seeds, clap_col_lo, clap_hi,
                folder_basis_mode,
            )
            _tier_cache_put(
                _semantic_basis_cache, basis_key, (T_n, exp),
                _SEMANTIC_BASIS_CACHE_MAX,
            )
        else:
            T_n, exp = basis
        mat_proj = _apply_low_rank_semantic_weight(
            mat, T_n, exp, folder_boost, folder_depth_boost,
        )

    tsne_config = comp_config
    if needs_weight:
        tsne_config = (
            f"{comp_config}|{basis_key}"
            f"|{folder_boost:.6g}|{folder_depth_boost:.6g}"
        )
    return ProjectionInput(
        paths, mat, clap_dim, clap_col_lo, comp_key, comp_config, spans,
        mat_proj, tsne_config,
    )


def _source_block_dim(source: str, smap: dict[str, np.ndarray]) -> int:
    """Width of one source block, taken from the first cached row."""
    for v in smap.values():
        if v is not None:
            return int(np.asarray(v).reshape(-1).size)
    return 0


def _normalise_coords(coords: np.ndarray) -> np.ndarray:
    """Stretch every axis independently so each spans ``[0, 1]``.

    The served map must fill the view on every axis, so an elongated cloud
    becomes square instead of leaving most of one axis empty. That per-axis
    stretch is intentionally anisotropic: it is a display convention, not a
    structure claim, and the cross-lens gauge preserves orientation rather than
    exact distances or a 3-D head-on overlay. A degenerate axis (zero or
    non-finite span) is centred at 0.5 instead of dividing by zero.
    """
    out = np.array(coords, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[0] == 0:
        return coords
    mins = out.min(axis=0)
    maxs = out.max(axis=0)
    span = maxs - mins
    degenerate = ~np.isfinite(span) | (span < 1e-9)
    safe_span = np.where(degenerate, 1.0, span)
    out = (out - mins) / safe_span
    out[:, degenerate] = 0.5
    return out


def _weight_jacobian(
    project_fn: Callable[..., tuple],
    base_weights: dict[str, float],
    sources: tuple[str, ...],
    *,
    delta: float = 1e-3,
    on_progress: Callable[[int, int], None] | None = None,
    alignment_reference: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    """Finite-difference ``d(2-D coords) / d(source weight)`` at *base_weights*.

    *project_fn(weights, cache_fit=…)* returns ``(result, normed, spans)`` for one
    weight vector — :func:`compute_projection`'s own inner pipeline. One
    projection is run per source, so the caller must compute this once (on drag
    start) and apply it linearly for the small deltas of the drag; it is never a
    substitute for the full recompute on release.

    The perturbation fits are thrown away (*cache_fit=False*): their one-off configs
    would otherwise evict the live fit the base layout reuses. Both the base and
    the perturbations are *ungauged* (``gauge=False``) so the finite difference is
    taken in one shared coordinate frame. *on_progress* is called as
    ``(done, total)`` per source, for a determinate progress UI.

    *alignment_reference* is the **served** base layout (already gauged and
    per-axis normalised for display). The ungauged base is mapped onto it with a
    least-squares 2-D affine fit on the shared paths, and that one transform is
    applied to every perturbation before differencing. Translation cancels; the
    derivative carries the gauge's rotation plus the deliberate per-axis display
    stretch, so a drag preview moves in the frame the user sees. Perturbations are
    never gauged individually — that would refit each one and both pollute the
    family anchor and make the difference depend on the fit.

    A source at weight 0 still has a meaningful one-sided derivative: raising it is
    the slider's action, so we step it *up* by ``+eps`` and divide by that same
    ``eps`` (never a negative or two-sided step). A source with no block available
    at all (no cached rows for any track) is skipped — stepping it cannot move the
    chart, so it has no derivative to report.
    """
    base_result, base_normed, _spans = project_fn(base_weights, gauge=False)
    out: dict[str, np.ndarray] = {}
    if base_normed.shape[0] < 2:
        return out
    transform = _gauge_transform_for(base_result, base_normed, alignment_reference)
    aligned_base = (
        base_normed
        if transform is None
        else _apply_gauge_transform(base_normed, transform)
    )
    total = len(sources)
    done = 0
    for s in sources:
        done += 1
        weight = float(base_weights.get(s, 1.0))
        # One-sided positive step: raise *s* by +delta (never a two-sided or
        # negative step), so a zero-weight source's derivative is the rate of
        # raising it from 0.
        stepped = dict(base_weights)
        stepped[s] = weight + float(delta)
        _res, moved, moved_spans = project_fn(
            stepped, cache_fit=False, gauge=False
        )
        # Skip only a source with no block available: its +delta projection still
        # carries no span for it, so stepping cannot change the chart.
        if s not in moved_spans:
            if on_progress is not None:
                on_progress(done, total)
            continue
        if moved.shape == base_normed.shape:
            aligned_moved = (
                moved
                if transform is None
                else _apply_gauge_transform(moved, transform)
            )
            out[s] = ((aligned_moved - aligned_base) / float(delta)).astype(np.float32)
        if on_progress is not None:
            on_progress(done, total)
    return out


def _gauge_transform_for(
    base_result: list[dict],
    base_normed: np.ndarray,
    alignment_reference: list[dict] | None,
) -> np.ndarray | None:
    """Least-squares affine map from the ungauged base onto the served base.

    A similarity fit cannot express the served frame: the final display
    normalisation stretches each axis independently after the orientation fit, so
    the map from the raw normalised base to the served coordinates is affine (a
    rotation followed by an axis-wise stretch). Returns a ``2x3`` matrix, or
    ``None`` when the fit is underdetermined, degenerate or non-finite.
    """
    if alignment_reference is None or base_normed.ndim != 2:
        return None
    if base_normed.shape[1] != 2:
        return None
    ref = _positions_array(alignment_reference, 2)
    if ref is None:
        return None
    ref_paths, ref_coords = ref
    base_paths = [str(p["path"]) for p in base_result]
    mine, theirs = matched_indices(base_paths, ref_paths)
    if len(mine) < 3:
        return None
    src = base_normed[mine]
    dst = ref_coords[theirs]
    design = np.column_stack([src, np.ones(len(src))])
    try:
        if np.linalg.matrix_rank(design) < 3:
            return None
        solution, *_ = np.linalg.lstsq(design, dst, rcond=None)
    except np.linalg.LinAlgError:
        log.debug("jacobian gauge transform failed; leaving the derivative ungauged", exc_info=True)
        return None
    transform = np.asarray(solution, dtype=np.float64).T
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _apply_gauge_transform(coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map ungauged coordinates into the served base frame (affine ``2x3``)."""
    return coords @ transform[:, :2].T + transform[:, 2]



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
    folder_boost: float = FROZEN_FOLDER_BOOST,
    folder_depth_boost: float = FROZEN_FOLDER_DEPTH_BOOST,
    sources: tuple[str, ...] = ("clap",),
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    layout_revision: str | None = None,
    source_weights: dict[str, float] | None = None,
    required_sources: tuple[str, ...] | None = None,
    with_jacobian: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
    dims: int = 2,
    align_ref: list[dict] | None = None,
    recipe_id: str | None = None,
    tsne_n_iter: int | None = None,
    tsne_early_exaggeration_iter: int | None = None,
    gauge_scope: str | None = None,
) -> list[dict] | tuple[list[dict], dict[str, np.ndarray]]:
    if layout_revision and method == "tsne":
        hit = revision_layout_cached(
            layout_revision,
            feature_cache,
            gauge_scope=gauge_scope,
            dims=dims,
        )
        if hit is not None and not with_jacobian:
            log.debug("projection: revision cache hit (tsne)")
            return hit

    weights0 = dict(source_weights or {})

    # Effective openTSNE budget: the recipe default unless the request overrode it.
    # The deviation suffix keys the fit cache so two budgets never share a fit; the
    # default recipe at its default 50/0 yields an empty suffix (byte-identical).
    _recipe = recipe_for(recipe_id)
    eff_n_iter = int(tsne_n_iter if tsne_n_iter is not None else _recipe.tsne_n_iter)
    eff_early = int(
        tsne_early_exaggeration_iter
        if tsne_early_exaggeration_iter is not None
        else _recipe.tsne_early_exaggeration_iter
    )
    tsne_suffix = tsne_iteration_suffix(eff_n_iter, eff_early, _recipe)

    # The 3-D gauge pins the head-on (x,y) of a 3-D layout to this lens's own
    # 2-D layout (see `lens_gauge`); build the reference arrays once, not per
    # projection call.
    ref_paths: list[str] | None = None
    ref_xy: np.ndarray | None = None
    if align_ref is not None:
        ref_paths = [p["path"] for p in align_ref]
        ref_xy = np.asarray(
            [[p["x"], p["y"]] for p in align_ref], dtype=np.float64
        )

    # The gauge family key depends only on the scope, the track set and ``dims``,
    # all of which every pass over this request shares, so it is derived at most
    # once — and never for the ungauged Jacobian passes.
    gauge_family: str | None = None

    def _project_at(
        w: dict[str, float], *, cache_fit: bool = True, gauge: bool = True
    ):
        """Gather → weight → project → normalise for one weight vector.

        Returns ``(result, normed, spans)``. The composite gather is keyed on
        the weights, so a perturbation re-runs only the projection.

        *gauge* is off for the Jacobian difference pass: the base and every
        perturbation must share one frame, so both stay ungauged rather than
        aligning a perturbation (which would register a reference and change
        the frame between difference terms).
        """
        nonlocal gauge_family
        inp = prepare_projection_input(
            track_infos,
            feature_cache,
            sources=sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            source_weights=w,
            required_sources=required_sources,
            scale_folders=scale_folders,
            context_tags=context_tags,
            context_folders=context_folders,
            folder_boost=folder_boost,
            folder_depth_boost=folder_depth_boost,
            folder_basis_mode=FROZEN_FOLDER_BASIS_MODE,
        )
        # `mat.shape[0] < 2` guards the degenerate all-zero-weight case, where
        # every block is dropped. The HTTP routes reject that vector before
        # reaching here; this keeps the function safe regardless.
        if len(inp.paths) < 2 or inp.mat.shape[0] < 2:
            flat = [
                {"path": p, "x": 0.5, "y": 0.5, **({"z": 0.5} if dims == 3 else {})}
                for p in inp.paths
            ]
            return flat, np.empty((0, dims), dtype=np.float32), inp.spans

        if method == "pca":
            coords = _project_pca(inp.mat_proj, inp.paths, dims=dims)
        else:
            if method != "tsne":
                log.debug("unknown projection method %r; using t-SNE", method)
            # The fit cache keys on the vector config (not the track set), so an
            # openTSNE embedding fitted for the same sources/mask/weighting is
            # reused and only newly added tracks are transformed into it.
            coords = _project_tsne(
                inp.mat_proj,
                paths=inp.paths,
                config_key=inp.tsne_config,
                cache_fit=cache_fit,
                feature_cache=feature_cache,
                dims=dims,
                tsne_n_iter=eff_n_iter,
                tsne_early_exaggeration_iter=eff_early,
                tsne_key_suffix=tsne_suffix,
            )

        # Normalise per axis *before* gauging so the orientation fit sees a
        # comparable shape, then align to the family's reference. The gauge applies
        # rotation/reflection/translation only; uniform scale is ignored. Rotation
        # changes the axis extents, so the final served coordinate is normalised
        # per axis again below to fill every axis of the view.
        normed = _normalise_coords(coords)
        if gauge and cache_fit and (
            dims == 2 or (dims == 3 and align_ref is not None)
        ):
            if gauge_family is None:
                gauge_family = _gauge_family(
                    gauge_scope, inp.paths, feature_cache, dims
                )
            # One frame per stable family (library scope + cache generations),
            # shared by every lens and folder subset. A persisted anchor is
            # restored first so a cold start keeps the pre-restart frame.
            _ensure_gauge_reference(gauge_family, feature_cache, dims=dims)
            if dims == 2:
                normed = gauge_layout(gauge_family, inp.paths, normed)
            else:
                normed = gauge_layout_3d(
                    gauge_family,
                    inp.paths,
                    normed,
                    ref_paths,
                    ref_xy,
                )
        # Final display geometry: stretch each axis independently so the map fills
        # the view box. This is deliberate for both 2-D and 3-D — the x/y overlay
        # with the 2-D reference is only approximate after it, by design.
        normed = _normalise_coords(normed)
        result = [
            {
                "path": inp.paths[i],
                "x": float(normed[i, 0]),
                "y": float(normed[i, 1]),
                **({"z": float(normed[i, 2])} if dims == 3 else {}),
            }
            for i in range(len(inp.paths))
        ]
        return result, normed, inp.spans

    # The base layout is the whole of the *visible* work, so it is reported
    # first and on its own; the Jacobian that follows is a separate, slower pass
    # whose per-source progress the frontend renders in the background.
    if on_progress is not None:
        on_progress("layout", 0, 0)
    result, _normed, _spans = _project_at(weights0)

    if layout_revision and method == "tsne":
        _tier_cache_put(
            _revision_layout_cache, layout_revision, result, _REVISION_LAYOUT_CACHE_MAX,
        )
        putter = getattr(feature_cache, "put_layout", None)
        if putter is not None:
            putter(layout_revision, result)
        # First complete layout for a family becomes its persisted anchor; a tiny
        # folder never does. The pointer is only written when absent/unusable.
        family = gauge_family
        if family is None:
            family = _gauge_family(
                gauge_scope, [p["path"] for p in result], feature_cache, dims
            )
        _store_gauge_anchor(family, feature_cache, layout_revision, result, dims)

    if with_jacobian:
        jacobian = _weight_jacobian(
            _project_at,
            weights0,
            sources,
            on_progress=(
                (lambda done, total: on_progress("jacobian", done, total))
                if on_progress is not None
                else None
            ),
            # 2-D only: the derivative must live in the frame the user sees. A
            # cached layout hit skipped the early return above so this pass still
            # runs; a tiny/no-chart layout yields an empty Jacobian (not an error).
            alignment_reference=result if dims == 2 else None,
        )
        return result, jacobian
    return result
