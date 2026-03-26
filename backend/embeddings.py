"""CLAP audio embedding generation and UMAP projection.

Uses ``laion/larger_clap_music`` for content-based audio embeddings.
The model is loaded eagerly at import time (called once at server startup).

Embeddings are generated from multiple segments of each track and averaged
to produce a more robust representation.  Cached in the FeatureCache keyed
by chromaprint fingerprint + EMBEDDING_VERSION.
"""

import os
import logging
import threading

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

log = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/larger_clap_music"
CLAP_SR = 48000
SEGMENT_SECONDS = 15
NUM_SEGMENTS = 3
# Bump this when the embedding strategy changes to auto-invalidate stale cache.
EMBEDDING_VERSION = 2

_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_model_lock = threading.Lock()
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


def _load_audio_segments(path: str) -> list[np.ndarray]:
    """Load up to NUM_SEGMENTS segments from different parts of the track.

    Short tracks (≤ 1.5× SEGMENT_SECONDS) return a single whole-file segment.
    Longer tracks return segments centred at 20%, 50%, and 80% through the file.
    """
    try:
        duration = librosa.get_duration(path=path)
        if duration <= 0:
            return []

        if duration <= SEGMENT_SECONDS * 1.5:
            audio, _ = librosa.load(path, sr=CLAP_SR, mono=True)
            return [audio] if len(audio) > 0 else []

        positions = [0.2, 0.5, 0.8]
        segments: list[np.ndarray] = []
        for pos in positions:
            centre = duration * pos
            offset = max(0.0, centre - SEGMENT_SECONDS / 2)
            offset = min(offset, max(0.0, duration - SEGMENT_SECONDS))
            audio, _ = librosa.load(
                path, sr=CLAP_SR, mono=True,
                offset=offset, duration=SEGMENT_SECONDS,
            )
            if len(audio) > 0:
                segments.append(audio)
        return segments
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return []


def _embed_single(audio: np.ndarray) -> np.ndarray | None:
    """Run CLAP inference on a single audio waveform."""
    try:
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


def _build_weighted_context(
    tag_names: list[str],
    folder_paths: list[str],
    boost: float = 3.0,
    depth_decay: float = 0.25,
) -> list[tuple[str, float]]:
    """Combine tag names and folder path segments into (text, weight) pairs.

    - Tags receive full *boost*.
    - Folder paths are split into segments.  A segment at depth *d* (1-based)
      receives ``boost * depth_decay ** (d - 1)``.  If the same segment name
      appears at multiple depths, the shallowest (highest weight) wins.
    """
    items: dict[str, float] = {}

    for tag in tag_names:
        items[tag] = max(items.get(tag, 0.0), boost)

    for fpath in folder_paths:
        parts = [p for p in fpath.split("/") if p]
        for depth_idx, segment in enumerate(parts):
            w = boost * (depth_decay ** depth_idx)
            items[segment] = max(items.get(segment, 0.0), w)

    return [(text, weight) for text, weight in items.items() if weight > 1e-6]


def _apply_semantic_weighting(
    mat: np.ndarray,
    context_tags: list[str] | None = None,
    context_folders: list[str] | None = None,
    boost: float = 3.0,
) -> np.ndarray:
    """Re-scale embedding space to emphasise directions aligned with user context.

    Uses CLAP's text encoder to embed tag names / folder names, then linearly
    boosts the component of each audio embedding along those semantic directions.

    Each context item *i* has its own weight *w_i*:

        W = I + T_n^T  diag(w)  T_n

    Tag names receive full *boost*.  Folder segments are depth-scaled: each
    subsequent subfolder level multiplies the weight by 0.25 (configurable via
    ``depth_decay`` in ``_build_weighted_context``).
    """
    items = _build_weighted_context(
        context_tags or [], context_folders or [], boost=boost,
    )
    if not items:
        return mat

    texts = [t for t, _ in items]
    weights = np.array([w for _, w in items], dtype=np.float32)

    text_vecs = generate_text_embeddings(texts)
    if text_vecs is None or text_vecs.shape[0] == 0:
        return mat

    norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    T_n = text_vecs / norms  # (K, D)

    # W = I + T_n^T @ diag(weights) @ T_n   (D×D)
    W = np.eye(mat.shape[1], dtype=np.float32) + T_n.T @ (weights[:, None] * T_n)
    return (mat @ W).astype(np.float32)


def generate_embedding(path: str) -> np.ndarray | None:
    """Generate a CLAP embedding for a single audio file.

    Loads multiple segments and averages their embeddings for a more robust
    representation.  Returns a 1-D float32 numpy array, or None on failure.
    """
    if not _loaded:
        load_model()

    segments = _load_audio_segments(path)
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


def _project_umap(mat: np.ndarray) -> np.ndarray:
    """Full UMAP 2D projection. Higher quality but slower."""
    import umap  # lazy — heavy import
    n_neighbors = min(15, len(mat) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1)
    return reducer.fit_transform(mat)


def _project_tsne(mat: np.ndarray) -> np.ndarray:
    """t-SNE 2D projection.  Better at preserving local cluster structure."""
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


def compute_projection(
    track_infos: list[dict],
    feature_cache,
    version: int = EMBEDDING_VERSION,
    method: str = "umap",
    context_tags: list[str] | None = None,
    context_folders: list[str] | None = None,
) -> list[dict]:
    """Project cached embeddings to 2D positions.

    *method* is ``"umap"`` (default), ``"tsne"``, or ``"pca"`` (instant,
    good enough for live intermediate updates during generation).

    *context_tags* / *context_folders* — when provided the embedding space is
    re-weighted via CLAP text embeddings.  Tag names get full boost; folder
    segments are depth-scaled (each subfolder level × 0.25).

    Returns ``[{"path": rel, "x": float, "y": float}, ...]`` where x/y
    are in [0, 1].  Tracks without embeddings are omitted.
    """
    paths, vecs = _gather_vecs(track_infos, feature_cache, version)

    if len(vecs) < 2:
        return [{"path": p, "x": 0.5, "y": 0.5} for p in paths]

    mat = np.stack(vecs)

    if (context_tags or context_folders) and _loaded:
        mat = _apply_semantic_weighting(mat, context_tags, context_folders)

    if method == "pca":
        coords = _project_pca(mat, paths)
    elif method == "tsne":
        coords = _project_tsne(mat)
    else:
        coords = _project_umap(mat)

    normed = _normalise_coords(coords)

    return [
        {"path": paths[i], "x": float(normed[i, 0]), "y": float(normed[i, 1])}
        for i in range(len(paths))
    ]


# Keep old name as alias for backward compat
compute_umap = compute_projection
