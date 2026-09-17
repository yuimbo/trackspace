"""Laion CLAP (``larger_clap_music_and_speech``) — audio + text embeddings for Trackspace.

Segment decode uses `load_resilient_audio_segments` from `audio_decode`.
`inference_lock` is the process-wide GPU/accelerator serialisation point: it is
shared with `layout` (mlx-vis on Metal), `maest` (its transformer), and the CLAP
text encoder, so no two frameworks dispatch to the accelerator concurrently — see
`audio_pipeline.md`.

Audio embeddings are produced by `analyze_batch`, which decodes per track and then
runs *one* forward pass per batch of segments (the feature extractor pads each
segment independently, so variable-length inputs batch safely). This mirrors
`maest.analyze_batch` and cuts lock acquisitions from three per track to one per
batch.

**Do not switch back to ``laion/larger_clap_music``.** That HF conversion is
degenerate: its text and audio towers both collapse to near-constant vectors
(text-vs-text cosine ≈ 0.999, audio-vs-audio ≈ 0.95–0.996, text-vs-audio ≈ 0.00),
so the canonical HF pipeline returns a uniform softmax and both semantic weighting and
text search are inert. Verified across CPU/MPS and transformers 4.57/5.3, with exact
weight loading — the checkpoint itself is broken (HF discussion #2 on that repo, open
since Dec 2023). ``larger_clap_music_and_speech`` is the music-trained checkpoint
whose conversion works.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from .audio_decode import load_resilient_audio_segments

log = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48000
SEGMENT_SECONDS = 15
NUM_SEGMENTS = 3
# Bump when CLAP chunking / pooling strategy changes (FeatureCache clap column).
# 3: switched from the degenerate ``laion/larger_clap_music`` conversion to
#    ``laion/larger_clap_music_and_speech`` (see the module docstring).
EMBEDDING_VERSION = 3

_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_model_lock = threading.Lock()
# Serialises CLAP on MPS/CUDA *and* mlx-vis on Metal from `layout` module.
inference_lock = threading.Lock()
_loaded = False
_load_failed: str | None = None


def is_model_ready() -> bool:
    return _loaded


def load_error() -> str | None:
    """Reason CLAP is unavailable, or ``None`` when it loaded (or is untried)."""
    return _load_failed


def load_model() -> None:
    """Load the CLAP model and processor into GPU/CPU memory (never raises)."""
    global _model, _processor, _loaded, _load_failed
    if _loaded:
        return
    with _model_lock:
        if _loaded:
            return
        try:
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
            _load_failed = None
            log.info("CLAP model loaded.")
        except Exception as e:
            _load_failed = f"{type(e).__name__}: {e}"
            log.exception("CLAP model load failed")


def _load_audio_segments(path: str) -> tuple[list[np.ndarray], str | None]:
    """Load segments for CLAP — see `NUM_SEGMENTS` / `SEGMENT_SECONDS`."""
    try:
        return load_resilient_audio_segments(
            path, sr=CLAP_SR, segment_seconds=SEGMENT_SECONDS,
        )
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return [], None


def _extract_tensor(outputs) -> "torch.Tensor":
    """Pooler output when present, else the CLS token, else the raw output."""
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0]
    return outputs


def analyze_batch(
    paths: list[str],
    *,
    on_decode_warning: Callable[[str, str], None] | None = None,
    max_batch: int = 12,
) -> dict[str, np.ndarray | None]:
    """Embed several tracks, batching segment inference across tracks.

    Decoding happens per track on the calling thread; inference is grouped into
    batches of at most *max_batch* segments so the accelerator is entered once per
    batch instead of once per segment. The feature extractor pads/truncates each
    segment independently, so variable-length segments batch safely.
    """
    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    if not paths:
        return results

    if not _loaded:
        load_model()
    if not _loaded:
        return results

    # ── decode all segments, remembering which track each belongs to ──
    flat: list[np.ndarray] = []
    owner: list[str] = []
    for p in paths:
        segments, warn = _load_audio_segments(p)
        if warn and on_decode_warning:
            on_decode_warning(p, warn)
        for seg in segments[:NUM_SEGMENTS]:
            if seg.size == 0:
                continue
            flat.append(np.asarray(seg, dtype=np.float32))
            owner.append(p)

    if not flat:
        return results

    per_track: dict[str, list[np.ndarray]] = {p: [] for p in paths}

    for start in range(0, len(flat), max_batch):
        chunk = flat[start : start + max_batch]
        chunk_owner = owner[start : start + max_batch]
        try:
            with inference_lock:
                inputs = _processor(
                    audio=chunk,
                    sampling_rate=CLAP_SR,
                    return_tensors="pt",
                )
                device = next(_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = _model.get_audio_features(**inputs)
                tensor = _extract_tensor(outputs)
                emb = tensor.detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning("CLAP batch inference failed (%d segments): %s", len(chunk), e)
            continue

        for i, p in enumerate(chunk_owner):
            per_track[p].append(emb[i])

    for p in paths:
        seg_embs = per_track.get(p) or []
        if seg_embs:
            results[p] = np.mean(seg_embs, axis=0).astype(np.float32)

    return results


def generate_text_embeddings(texts: list[str]) -> np.ndarray | None:
    """CLAP text encoder — tag/folder semantic directions in layout code."""
    if not _loaded or not texts:
        return None
    try:
        with inference_lock:
            inputs = _processor(text=texts, return_tensors="pt", padding=True)
            device = next(_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = _model.get_text_features(**inputs)
            tensor = _extract_tensor(outputs)
            return tensor.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        log.warning("CLAP text embedding failed: %s", e)
        return None


def generate_embedding(
    path: str,
    *,
    on_decode_warning: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """Mean-pooled multi-segment CLAP audio embedding, or None on failure."""
    def _warn(_p: str, message: str) -> None:
        on_decode_warning(message)

    results = analyze_batch(
        [path], on_decode_warning=_warn if on_decode_warning else None
    )
    return results.get(path)


def batch_ensure_embeddings(
    track_infos: list[dict],
    music_root: str,
    feature_cache,
    version: int = EMBEDDING_VERSION,
) -> dict[str, bool]:
    """CLAP-only cache fill helper (HTTP batch path uses `build_generation_work`)."""
    result: dict[str, bool] = {}
    to_generate: list[tuple[str, str]] = []

    fps = [t["fingerprint"] for t in track_infos if t.get("fingerprint")]
    existing = feature_cache.get_all("clap", fps, version=version)

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
            feature_cache.put("clap", fp, emb, version=version)
            rel = os.path.relpath(abs_path, music_root)
            result[rel] = True

    return result
