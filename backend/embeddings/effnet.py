"""Discogs-EffNet ONNX embeddings (~1280-D) with Essentia-compatible mel preprocessing."""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request

import librosa
import numpy as np

from .audio_decode import load_resilient_audio_segments

log = logging.getLogger(__name__)

EFFNET_VERSION = 3

_EFFNET_SR = 16000
_EFFNET_N_FFT = 512
_EFFNET_HOP = 256
_EFFNET_N_MELS = 96
_EFFNET_PATCH_FRAMES = 128
_EFFNET_PATCH_HOP = 128
_EFFNET_BATCH_CAP = 64
_EFFNET_SEGMENT_SECONDS = 15

_effnet_session = None
_effnet_lock = threading.Lock()
_effnet_loaded = False

_EFFNET_MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    "discogs-effnet-bsdynamic-1.onnx"
)
_EFFNET_MODEL_NAME = "discogs-effnet-bsdynamic-1.onnx"
_EFFNET_MODEL_MIN_BYTES = 10_000_000

_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_DELAY = 5


def _model_dir() -> str:
    d = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "models",
    )
    os.makedirs(d, exist_ok=True)
    return d


def _ensure_effnet_model_file() -> str:
    path = os.path.join(_model_dir(), _EFFNET_MODEL_NAME)
    if os.path.isfile(path):
        if os.path.getsize(path) >= _EFFNET_MODEL_MIN_BYTES:
            return path
        log.warning(
            "Removing truncated model file (%d bytes): %s",
            os.path.getsize(path), path,
        )
        os.remove(path)

    tmp = path + ".tmp"
    last_err: Exception | None = None

    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        try:
            log.info(
                "Downloading %s (attempt %d/%d) …",
                _EFFNET_MODEL_URL, attempt, _DOWNLOAD_MAX_RETRIES,
            )
            urllib.request.urlretrieve(_EFFNET_MODEL_URL, tmp)

            if os.path.getsize(tmp) < _EFFNET_MODEL_MIN_BYTES:
                raise OSError(
                    f"Downloaded file too small ({os.path.getsize(tmp)} bytes), "
                    f"expected ≥{_EFFNET_MODEL_MIN_BYTES}",
                )

            os.rename(tmp, path)
            log.info("Download complete (%d bytes).", os.path.getsize(path))
            return path

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = e
            log.warning("Download attempt %d failed: %s", attempt, e)
            if os.path.isfile(tmp):
                os.remove(tmp)
            if attempt < _DOWNLOAD_MAX_RETRIES:
                delay = _DOWNLOAD_RETRY_DELAY * (2 ** (attempt - 1))
                log.info("Retrying in %ds …", delay)
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to download EffNet model after {_DOWNLOAD_MAX_RETRIES} attempts: {last_err}",
    )


def is_effnet_ready() -> bool:
    return _effnet_loaded


def load_effnet() -> None:
    global _effnet_session, _effnet_loaded
    if _effnet_loaded:
        return
    with _effnet_lock:
        if _effnet_loaded:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            log.warning("onnxruntime not installed — EffNet embeddings disabled")
            return
        onnx_path = _ensure_effnet_model_file()
        log.info("Loading EffNet ONNX model %s …", onnx_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        _effnet_session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers,
        )
        _effnet_loaded = True
        log.info(
            "EffNet ONNX model loaded (providers: %s).",
            _effnet_session.get_providers(),
        )


def compute_mel_spectrogram(audio: np.ndarray, sr: int = _EFFNET_SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=_EFFNET_N_FFT, hop_length=_EFFNET_HOP, n_mels=_EFFNET_N_MELS,
    )
    return np.log10(10000 * mel + 1).T.astype(np.float32)


def patch_mel_spectrogram(
    mel: np.ndarray,
    patch_frames: int = _EFFNET_PATCH_FRAMES,
    patch_hop: int = _EFFNET_PATCH_HOP,
) -> np.ndarray:
    n_frames = mel.shape[0]
    if n_frames < patch_frames:
        padded = np.zeros((patch_frames, mel.shape[1]), dtype=mel.dtype)
        padded[:n_frames] = mel
        return padded[np.newaxis]

    starts = list(range(0, n_frames - patch_frames + 1, patch_hop))
    if not starts:
        starts = [0]
    patches = np.stack([mel[s:s + patch_frames] for s in starts])
    return patches


def _load_effnet_segments(path: str) -> tuple[list[np.ndarray], str | None]:
    try:
        return load_resilient_audio_segments(
            path, sr=_EFFNET_SR, segment_seconds=_EFFNET_SEGMENT_SECONDS,
        )
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return [], None


def _run_effnet_batched(patches: np.ndarray) -> np.ndarray:
    assert _effnet_session is not None
    input_name = _effnet_session.get_inputs()[0].name
    output_name = _effnet_session.get_outputs()[0].name

    if len(patches) <= _EFFNET_BATCH_CAP:
        return _effnet_session.run([output_name], {input_name: patches})[0]

    results = []
    for i in range(0, len(patches), _EFFNET_BATCH_CAP):
        batch = patches[i:i + _EFFNET_BATCH_CAP]
        results.append(_effnet_session.run([output_name], {input_name: batch})[0])
    return np.concatenate(results, axis=0)


def _preprocess_effnet(path: str) -> tuple[np.ndarray | None, str | None]:
    try:
        segments, warn = _load_effnet_segments(path)
        if not segments:
            return None, warn
        all_patches: list[np.ndarray] = []
        for audio in segments:
            mel = compute_mel_spectrogram(audio)
            all_patches.append(patch_mel_spectrogram(mel))
        return np.concatenate(all_patches, axis=0), warn
    except Exception as e:
        log.warning("EffNet preprocessing failed for %s: %s", path, e)
        return None, None


def generate_effnet_embedding(path: str) -> np.ndarray | None:
    if not _effnet_loaded:
        load_effnet()
    if not _effnet_loaded or _effnet_session is None:
        return None

    patches, _warn = _preprocess_effnet(path)
    if patches is None:
        return None
    try:
        activations = _run_effnet_batched(patches)
        if activations is None or len(activations) == 0:
            return None
        return np.mean(activations, axis=0).astype(np.float32)
    except Exception as e:
        log.warning("EffNet embedding failed for %s: %s", path, e)
        return None


def generate_effnet_embeddings_batch(
    paths: list[str],
    decode_warnings: dict[str, str] | None = None,
) -> dict[str, np.ndarray | None]:
    if not paths:
        return {}
    if not _effnet_loaded:
        load_effnet()
    if not _effnet_loaded or _effnet_session is None:
        return {p: None for p in paths}

    valid_paths: list[str] = []
    all_patches: list[np.ndarray] = []
    patch_counts: list[int] = []
    for path in paths:
        patches, dwarn = _preprocess_effnet(path)
        if dwarn and decode_warnings is not None:
            decode_warnings[path] = dwarn
        if patches is not None and len(patches) > 0:
            valid_paths.append(path)
            all_patches.append(patches)
            patch_counts.append(len(patches))

    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    if not all_patches:
        return results

    try:
        combined = np.concatenate(all_patches, axis=0)
        activations = _run_effnet_batched(combined)
        offset = 0
        for path, count in zip(valid_paths, patch_counts):
            track_acts = activations[offset:offset + count]
            results[path] = np.mean(track_acts, axis=0).astype(np.float32)
            offset += count
    except Exception as e:
        log.warning("Batched EffNet inference failed: %s", e)

    return results
