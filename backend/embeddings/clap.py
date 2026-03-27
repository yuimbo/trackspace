"""Laion CLAP (``larger_clap_music``) — audio + text embeddings for Trackspace.

Segment decode uses `load_resilient_audio_segments` from `audio_decode`.
`inference_lock` is shared with `layout` so mlx-vis (Metal) never runs
concurrently with PyTorch CLAP on another thread — see `AGENTS.md`.
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

CLAP_MODEL_ID = "laion/larger_clap_music"
CLAP_SR = 48000
SEGMENT_SECONDS = 15
NUM_SEGMENTS = 3
# Bump when CLAP chunking / pooling strategy changes (FeatureCache clap column).
EMBEDDING_VERSION = 2

_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_model_lock = threading.Lock()
# Serialises CLAP on MPS/CUDA *and* mlx-vis on Metal from `layout` module.
inference_lock = threading.Lock()
_loaded = False


def is_model_ready() -> bool:
    return _loaded


def load_model() -> None:
    """Load the CLAP model and processor into GPU/CPU memory."""
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
    """Load segments for CLAP — see `NUM_SEGMENTS` / `SEGMENT_SECONDS`."""
    try:
        return load_resilient_audio_segments(
            path, sr=CLAP_SR, segment_seconds=SEGMENT_SECONDS,
        )
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return [], None


def _embed_single(audio: np.ndarray) -> np.ndarray | None:
    try:
        with inference_lock:
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
    """CLAP text encoder — tag/folder semantic directions in layout code."""
    if not _loaded or not texts:
        return None
    try:
        with inference_lock:
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


def generate_embedding(
    path: str,
    *,
    on_decode_warning: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """Mean-pooled multi-segment CLAP audio embedding, or None on failure."""
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
    """CLAP-only cache fill helper (HTTP batch path uses `build_generation_work`)."""
    result: dict[str, bool] = {}
    to_generate: list[tuple[str, str]] = []

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
