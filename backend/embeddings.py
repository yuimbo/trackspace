"""CLAP audio embedding generation and UMAP projection.

Uses ``laion/larger_clap_music`` for content-based audio embeddings.
The model is loaded eagerly at import time (called once at server startup).

Embeddings are generated from multiple segments of each track and averaged
to produce a more robust representation.  Cached in the FeatureCache keyed
by chromaprint fingerprint + EMBEDDING_VERSION.
"""

import hashlib
import importlib.util
import logging
import os
import platform
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from backend.audio_features import load_resilient_audio_segments
from backend.layout_revision import FOLDER_SEMANTIC_BASIS_VERSION

log = logging.getLogger(__name__)

# Within-folder PCA on deviations orthogonal to the hierarchical contrast axis.
FOLDER_PCA_MIN_TRACKS = 8
FOLDER_PCA_MAX_COMPONENTS = 2
FOLDER_PCA_SECOND_SV_RATIO = 0.18

CLAP_MODEL_ID = "laion/larger_clap_music"
CLAP_SR = 48000
SEGMENT_SECONDS = 15
NUM_SEGMENTS = 3
# Bump this when the embedding strategy changes to auto-invalidate stale cache.
EMBEDDING_VERSION = 2

# Tiered projection LRU (composite gather → semantic direction basis → UMAP/t-SNE).
# Semantic *weights* (folder contrast + depth emphasis) are applied via a cheap
# low-rank multiply once ``T_n``/exponents are cached — no full D×D matrix.
# PCA bypasses layout cache (fast; global Procrustes reference is per-process).
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

# Prepared per-source row vectors (L2-normalised CLAP/EffNet; masked+norm+blend features).
# Lets us switch source *combinations* without re-running per-row normalisation or
# re-querying FeatureCache rows that are still in this LRU.
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
    from backend.audio_features import EFFNET_VERSION, FEATURES_VERSION

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
    """Return one track row for *source* with the same preprocessing as composite gather.

    Rows are cached per (fingerprint, source, version[, feature mask+blend]) so
    toggling source layers reuses work across different ``sources`` tuples.
    """
    from backend.audio_features import EFFNET_VERSION, FEATURES_VERSION

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
    """Folders + tags only — independent of contrast slider values."""
    tags = ",".join(sorted(context_tags)) if context_tags else ""
    if folder_seeds:
        fp = ",".join(sorted(folder_seeds))
    else:
        fp = "-"
    return f"{tags}|{fp}|fsbv{FOLDER_SEMANTIC_BASIS_VERSION}"


def _semantic_param_key(
    context_tags: list[str] | None,
    folder_seeds: list[str] | None,
    folder_boost: float,
    folder_depth_boost: float,
) -> str:
    return (
        f"{_semantic_basis_param_key(context_tags, folder_seeds)}|"
        f"{folder_boost:.6g}|{folder_depth_boost:.6g}"
    )


_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_model_lock = threading.Lock()
# Serialises CLAP on MPS/CUDA *and* mlx-vis on Metal: concurrent Metal command
# encoders from PyTorch and MLX on different threads trigger IOGPU failures.
_inference_lock = threading.Lock()
_loaded = False


def is_model_ready() -> bool:
    return _loaded


def load_model() -> None:
    """Eagerly load the CLAP model and processor into GPU/CPU memory."""
    global _model, _processor, _loaded
    if _loaded:
        return
    with _model_lock:
        if _loaded:
            return
        token = os.environ.get("HF_TOKEN")
        log.info("Loading CLAP model %s …", CLAP_MODEL_ID)
        _processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, token=token)
        _model = ClapModel.from_pretrained(CLAP_MODEL_ID, token=token)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
            log.info("Using CUDA acceleration")
        elif torch.backends.mps.is_available():
            _model = _model.to("mps")
            log.info("Using MPS acceleration (Apple Silicon)")
        else:
            log.info("Using CPU inference")
        _loaded = True
        log.info("CLAP model loaded.")


def _load_audio_segments(path: str) -> tuple[list[np.ndarray], str | None]:
    """Load up to NUM_SEGMENTS segments from different parts of the track.

    Short tracks (≤ 1.5× SEGMENT_SECONDS) return a single whole-file segment.
    Longer tracks return segments centred at 20%, 50%, and 80% through the file.

    Returns ``(segments, user_warning)`` when metadata vs decoded length disagrees.
    """
    try:
        return load_resilient_audio_segments(
            path, sr=CLAP_SR, segment_seconds=SEGMENT_SECONDS,
        )
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return [], None


def _embed_single(audio: np.ndarray) -> np.ndarray | None:
    """Run CLAP inference on a single audio waveform.

    Serialised via ``_inference_lock`` — the CLAP model (especially on MPS)
    is not safe for concurrent forward passes from multiple threads.
    """
    try:
        with _inference_lock:
            inputs = _processor(
                audio=audio,
                sampling_rate=CLAP_SR,
                return_tensors="pt",
            )
            device = next(_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = _model.get_audio_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state[:, 0]
        else:
            tensor = outputs

        return tensor.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        log.warning("CLAP segment inference failed: %s", e)
        return None


def generate_text_embeddings(texts: list[str]) -> np.ndarray | None:
    """Generate CLAP text embeddings for a list of strings.

    Returns an (N, D) float32 array, or None if the model isn't ready.
    Used to derive semantic directions from tag names and folder names.
    """
    if not _loaded or not texts:
        return None
    try:
        with _inference_lock:
            inputs = _processor(text=texts, return_tensors="pt", padding=True)
            device = next(_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state[:, 0]
        else:
            tensor = outputs
        return tensor.cpu().numpy().astype(np.float32)
    except Exception as e:
        log.warning("CLAP text embedding failed: %s", e)
        return None


def _folder_seeds_from_track_paths(paths: list[str]) -> list[str]:
    """Unique immediate parent folders of tracks (root excluded)."""
    s: set[str] = set()
    for p in paths:
        sep = p.rfind("/")
        tf = p[:sep] if sep >= 0 else ""
        if tf:
            s.add(tf)
    return sorted(s)


def _expand_folder_nodes(folder_seeds: list[str]) -> set[str]:
    """All folder path strings: seeds plus every ancestor segment."""
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
    """Principal directions of within-folder spread after removing the axis *d_unit*.

    *mat_block* is ``(n, D)`` rows for one folder; *d_unit* is the hierarchical
    ``normalize(c_child - c_parent)`` for that folder.  Returns 0–*max_components*
    unit vectors orthogonal to *d_unit* (and mutually orthogonal).
    """
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
    """Compute hierarchical contrast directions from actual track embeddings.

    For each folder node the primary direction is
    ``centroid(all tracks under node) − centroid(all tracks under parent)``.
    When a node has at least *min_tracks_pca* tracks, up to *max_pca_components*
    extra directions come from PCA on within-folder deviations after removing
    that hierarchical axis (orthogonal residual spread).

    Shared directions among siblings are naturally absorbed by the parent;
    each child retains only its unique contrast.  Later, semantic weights use
    ``folder_boost × folder_depth_boost ** exponent`` where *exponent* is
    ``depth − 1`` for that node (see *exponents*) — the same exponent is used
    for a node's hierarchical row and its PCA rows.

    *folder_seeds* lists folder paths (typically parents of tracks); parent
    nodes are inferred so the client need not enumerate the whole tree.

    Returns *(directions, exponents, text_fallbacks)* where *exponents* is
    ``(K,)`` int (``depth - 1`` per row, same order as *directions*), and
    *text_fallbacks* is ``(leaf_name, exponent)`` for CLAP-text fallback
    directions.
    """
    if not folder_seeds:
        return None, None, []

    all_nodes = _expand_folder_nodes(folder_seeds)

    # Map each embedding row to its direct folder.
    track_folders: list[str] = []
    for p in paths:
        sep = p.rfind("/")
        track_folders.append(p[:sep] if sep >= 0 else "")

    # Accumulate track indices per node (includes all descendants).
    node_indices: dict[str, list[int]] = {n: [] for n in all_nodes}
    for i, tf in enumerate(track_folders):
        node_indices[""].append(i)
        parts = [p for p in tf.split("/") if p]
        for depth in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:depth])
            if ancestor in node_indices:
                node_indices[ancestor].append(i)

    # Centroid per node.
    centroids: dict[str, np.ndarray | None] = {}
    for n, idx_list in node_indices.items():
        centroids[n] = mat[idx_list].mean(axis=0) if idx_list else None

    # Contrast directions.
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
    """Place CLAP text direction rows (K, d_txt) into the CLAP column span."""
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
    """Apply ``mat @ (I + T_n.T diag(w) T_n)`` with ``w_i = boost * depth_boost**exp_i``.

    *exp* holds per-row integer exponents (``depth - 1`` for folders; ``0`` for tags).
    Never materialises the D×D matrix.
    """
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
    """Return ``(T_n, exp)`` for low-rank semantic weighting (boost-agnostic)."""
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
    """Re-weight embedding space using data-driven folder directions and text
    tag directions.

    Folders use *hierarchical centroid decomposition*: each folder's primary
    direction is its centroid minus its parent's centroid.  Large folders also
    contribute PCA axes of deviation orthogonal to that contrast (within-folder
    spread).  Deeper levels receive increasing boost so that subtle differences
    are amplified.  Folders with fewer than 2 embedded tracks fall back to CLAP
    text embedding of the folder name.

    Tags always use CLAP text embedding (they are labels, not track collections).

    Uses a low-rank multiply equivalent to ``mat @ (I + T_n.T diag(w) T_n)``.
    """
    T_n, exp = _build_semantic_basis(
        mat, paths, context_tags, folder_seeds, clap_lo, clap_hi,
    )
    if T_n.shape[0] == 0:
        return mat
    return _apply_low_rank_semantic_weight(mat, T_n, exp, boost, depth_boost)


def generate_embedding(
    path: str,
    *,
    on_decode_warning: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """Generate a CLAP embedding for a single audio file.

    Loads multiple segments and averages their embeddings for a more robust
    representation.  Returns a 1-D float32 numpy array, or None on failure.

    *on_decode_warning* — invoked once per file when decoded audio is much shorter
    than container metadata (damaged / mis-tagged files).
    """
    if not _loaded:
        load_model()

    segments, decode_warn = _load_audio_segments(path)
    if decode_warn and on_decode_warning:
        on_decode_warning(decode_warn)
    if not segments:
        return None

    segment_embs: list[np.ndarray] = []
    for audio in segments:
        emb = _embed_single(audio)
        if emb is not None:
            segment_embs.append(emb)

    if not segment_embs:
        return None

    return np.mean(segment_embs, axis=0).astype(np.float32)


def batch_ensure_embeddings(
    track_infos: list[dict],
    music_root: str,
    feature_cache,
    version: int = EMBEDDING_VERSION,
) -> dict[str, bool]:
    """Ensure every track with a fingerprint has a CLAP embedding.

    *track_infos* is a list of dicts with at least ``path`` (relative) and
    ``fingerprint`` keys — the same shape returned by _cached_read_all +
    the track-list builder in app.py.

    Returns ``{rel_path: True/False}`` indicating whether an embedding
    exists (either already cached or freshly generated).
    """
    result: dict[str, bool] = {}
    to_generate: list[tuple[str, str]] = []  # (abs_path, fingerprint)

    fps = [t["fingerprint"] for t in track_infos if t.get("fingerprint")]
    existing = feature_cache.get_all_embeddings(fps, version=version)

    for t in track_infos:
        fp = t.get("fingerprint")
        if not fp:
            result[t["path"]] = False
            continue
        if fp in existing:
            result[t["path"]] = True
            continue
        to_generate.append((os.path.join(music_root, t["path"]), fp))
        result[t["path"]] = False

    for abs_path, fp in to_generate:
        emb = generate_embedding(abs_path)
        if emb is not None:
            feature_cache.put_embedding(fp, emb, version=version)
            rel = os.path.relpath(abs_path, music_root)
            result[rel] = True

    return result


def _gather_vecs(
    track_infos: list[dict],
    feature_cache,
    version: int,
) -> tuple[list[str], list[np.ndarray]]:
    """Collect embedding vectors for tracks that have them in the cache."""
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
    """Collect and concatenate vectors from multiple enabled sources.

    Each source block is L2-normalized per row before concatenation so that
    sources with different native dimensionalities contribute equally.  The
    audio-features block is then scaled by *features_blend* (< 1 weakens its
    influence versus CLAP / EffNet).  *feature_mask* zeros individual audio
    dimensions before normalization (length ``AUDIO_FEATURE_DIM``).

    Normalised rows are LRU-cached per fingerprint and source so changing the
    enabled *sources* tuple reuses prepared CLAP/EffNet/feature rows.

    Returns *(paths, matrix, clap_dim, clap_col_lo, composite_cache_key)*.
    *composite_cache_key* is ``\"\"`` when *paths* is empty.
    """
    from backend.audio_features import EFFNET_VERSION, FEATURES_VERSION, AUDIO_FEATURE_DIM

    fps = [t.get("fingerprint") for t in track_infos]
    fp_set = [fp for fp in fps if fp]

    # Pre-fetch all requested source maps keyed by fingerprint.
    source_maps: dict[str, dict[str, np.ndarray]] = {}
    if "clap" in sources:
        source_maps["clap"] = feature_cache.get_all_embeddings(fp_set, version=EMBEDDING_VERSION)
    if "effnet" in sources:
        source_maps["effnet"] = feature_cache.get_all_effnet_embeddings(fp_set, version=EFFNET_VERSION)
    if "features" in sources:
        source_maps["features"] = feature_cache.get_all_audio_features(fp_set, version=FEATURES_VERSION)

    # Determine which tracks have data for ALL enabled sources.
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

    # Build per-source matrices, normalize, and concatenate.
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
    """Normalise an (N, 2) array to [0, 1] per axis."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    return (coords - mins) / ranges


_pca_ref_paths: list[str] = []
_pca_ref_coords: np.ndarray | None = None


def _project_pca(mat: np.ndarray, paths: list[str] | None = None) -> np.ndarray:
    """Fast 2D projection via PCA (numpy SVD), Procrustes-stabilised.

    Successive calls align the new projection to the previous one using
    orthogonal Procrustes on shared points, preventing the random axis
    flips / 90° rotations that bare SVD produces when the data changes.
    """
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


# RAPIDS cuML manifolds use NVIDIA CUDA only (not Apple MPS). CLAP uses MPS on
# Apple Silicon when ``torch.cuda`` is false; install cuML per requirements-cuda.txt
# on Linux/WSL + NVIDIA to accelerate layout. ``TRACKSPACE_CUML=0`` forces CPU.
_cuml_projection_enabled: bool | None = None

# mlx-vis runs UMAP/t-SNE on Metal (Apple Silicon). Optional; see requirements.txt.
# ``TRACKSPACE_MLX_VIS=0`` forces CPU after cuML check.
_mlx_vis_projection_enabled: bool | None = None


def reset_projection_backend_cache() -> None:
    """Clear lazy cuML / mlx-vis detection (e.g. after changing env vars or in tests)."""
    global _cuml_projection_enabled, _mlx_vis_projection_enabled
    _cuml_projection_enabled = None
    _mlx_vis_projection_enabled = None


def _use_cuml_projection() -> bool:
    """True when cuML should run UMAP/t-SNE on a CUDA device."""
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
    """True when mlx-vis should run UMAP/t-SNE on Apple Metal (arm64 macOS)."""
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
    """mlx-vis TSNE requires ``n_features > pca_dim`` for its PCA prep branch."""
    if n_features < 2:
        return None
    return min(50, n_features - 1)


def _mlx_tsne_max_points() -> int:
    """Row count above which mlx-vis t-SNE is skipped in favour of scikit-learn.

    Profiled on Apple Silicon: **openTSNE** (FIt-SNE-style FFT and Barnes–Hut) is
    CPU-only (no MPS) and was slower than both mlx-vis and sklearn for *n* from
    hundreds through ~12k with typical embedding widths. **mlx-vis** beat sklearn
    up to roughly 10k points then sklearn's Barnes–Hut overtook; this cap avoids
    that regression while keeping Metal for normal library sizes.

    Set ``TRACKSPACE_MLX_TSNE_MAX_POINTS=0`` to always try mlx-vis (previous behaviour).
    """
    raw = os.environ.get("TRACKSPACE_MLX_TSNE_MAX_POINTS", "10000").strip()
    try:
        v = int(raw)
    except ValueError:
        return 10_000
    return v


def _project_umap(mat: np.ndarray) -> np.ndarray:
    """Full UMAP 2D projection. Higher quality but slower."""
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
    """t-SNE 2D projection.  Better at preserving local cluster structure.

    CUDA: RAPIDS cuML ``method="fft"`` when available.  Apple Silicon: mlx-vis
    on Metal up to :func:`_mlx_tsne_max_points`, then scikit-learn (see env var
    documented there).  Otherwise scikit-learn.
    """
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
    """Six ``0``/``1`` characters: tempo, key_cos, key_sin, mode, energy, dance."""
    from backend.audio_features import AUDIO_FEATURE_DIM

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
    """Project cached embeddings to 2D positions.

    *method* is ``"umap"`` (default), ``"tsne"``, or ``"pca"`` (instant,
    good enough for live intermediate updates during generation).

    *sources* selects which data to concatenate: any subset of
    ``("clap", "effnet", "features")``.  Each source block is
    L2-normalized before concatenation.

    *context_tags* — when provided the embedding space is re-weighted before
    projection.  Text-based directions apply only to the CLAP sub-space
    (requires CLAP in *sources*).

    Folder centroid decomposition runs when *scale_folders* is True.  Folder
    nodes are inferred from track paths; optional *context_folders* supplies
    an explicit seed list instead of deriving from paths.

    *layout_revision* — when provided (from ``compute_layout_revision``),
    the result is cached under this key.  Subsequent calls with the same
    revision return the cached layout without recomputing.

    Returns ``[{"path": rel, "x": float, "y": float}, ...]`` where x/y
    are in [0, 1].  Tracks without data for every enabled source are omitted.

    **Caching:** A deterministic ``layout_revision`` (SHA-256 of sorted
    paths + all projection parameters) is the primary layout cache key.
    Composite matrices and semantic-basis directions are still cached in
    their own LRU tiers (keyed by path-order-dependent hashes) to avoid
    redundant vector gathering and ``T_n`` construction on revision misses.
    PCA skips layout caching (fast; global Procrustes reference is per-process).
    """
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

    needs_weight = _loaded and (bool(context_tags) or bool(folder_seeds))

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


# Keep old name as alias for backward compat
compute_umap = compute_projection
