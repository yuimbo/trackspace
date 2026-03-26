"""Audio feature extraction and Discogs-EffNet embeddings (ONNX Runtime).

Two capabilities:

1. **Discogs-EffNet embeddings** — penultimate-layer activations from the
   ``discogs-effnet-bsdynamic-1`` ONNX model (~1280 D).  Uses ``onnxruntime``
   for inference and ``librosa`` for mel-spectrogram preprocessing.

2. **Audio features** — lightweight descriptors computed with librosa:
   tempo (BPM), key (circle-of-fifths 2D encoding), mode, energy, danceability.
   Stored as a compact float32 vector (6 D).

Key encoding: the circle of fifths is mapped to a unit circle so that
harmonically close keys are geometrically close:

    fifths_pos:  C=0  G=1  D=2  A=3  E=4  B=5  F#=6  Db=7  Ab=8  Eb=9  Bb=10  F=11
    key_cos = cos(2π · pos / 12)
    key_sin = sin(2π · pos / 12)
    mode    = 1.0 (major) | 0.0 (minor)

Mel-spectrogram recipe (matches Essentia's TensorflowInputMusiCNN):
    sr=16000, n_fft=512, hop_length=256, n_mels=96
    compression: np.log10(10000 * mel + 1)
    Confirmed equivalent by Essentia maintainers:
    https://github.com/MTG/essentia/issues/1471
"""

import logging
import math
import os
import threading
import urllib.request

import librosa
import numpy as np

from backend.decode_stderr import librosa_get_duration, librosa_load

log = logging.getLogger(__name__)

# ── Versioning ────────────────────────────────────────────────
EFFNET_VERSION = 3
FEATURES_VERSION = 3

# Full ``extract_audio_features`` vector layout (see docstring above).
AUDIO_FEATURE_DIM = 6

# ── EffNet model state ────────────────────────────────────────
_effnet_session = None
_effnet_lock = threading.Lock()
_effnet_loaded = False

_EFFNET_SR = 16000
_EFFNET_N_FFT = 512
_EFFNET_HOP = 256
_EFFNET_N_MELS = 96
_EFFNET_PATCH_FRAMES = 128
_EFFNET_PATCH_HOP = 128  # no overlap — halves patch count, negligible quality impact
_EFFNET_BATCH_CAP = 64   # max patches per ONNX inference call
_EFFNET_SEGMENT_SECONDS = 15
_EFFNET_NUM_SEGMENTS = 3

_EFFNET_MODEL_URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx"
_EFFNET_MODEL_NAME = "discogs-effnet-bsdynamic-1.onnx"
_EFFNET_MODEL_MIN_BYTES = 10_000_000  # ~18 MB expected; anything under 10 MB is truncated


def _model_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models")
    os.makedirs(d, exist_ok=True)
    return d


_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_DELAY = 5  # seconds


def _ensure_effnet_model_file() -> str:
    """Download the EffNet ONNX model if it isn't already cached in data/models/.

    Detects and removes partial/truncated downloads by checking file size.
    Retries on transient HTTP errors (5xx, timeouts) up to 3 times with
    exponential back-off.
    """
    import time

    path = os.path.join(_model_dir(), _EFFNET_MODEL_NAME)
    if os.path.isfile(path):
        if os.path.getsize(path) >= _EFFNET_MODEL_MIN_BYTES:
            return path
        log.warning("Removing truncated model file (%d bytes): %s",
                    os.path.getsize(path), path)
        os.remove(path)

    tmp = path + ".tmp"
    last_err: Exception | None = None

    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        try:
            log.info("Downloading %s (attempt %d/%d) …",
                     _EFFNET_MODEL_URL, attempt, _DOWNLOAD_MAX_RETRIES)
            urllib.request.urlretrieve(_EFFNET_MODEL_URL, tmp)

            if os.path.getsize(tmp) < _EFFNET_MODEL_MIN_BYTES:
                raise OSError(
                    f"Downloaded file too small ({os.path.getsize(tmp)} bytes), "
                    f"expected ≥{_EFFNET_MODEL_MIN_BYTES}"
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
        f"Failed to download EffNet model after {_DOWNLOAD_MAX_RETRIES} attempts: {last_err}"
    )


def is_effnet_ready() -> bool:
    return _effnet_loaded


def load_effnet() -> None:
    """Load the Discogs-EffNet ONNX model into an inference session."""
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
        _effnet_session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        _effnet_loaded = True
        log.info("EffNet ONNX model loaded (providers: %s).",
                 _effnet_session.get_providers())


# ── Mel-spectrogram preprocessing ─────────────────────────────

def compute_mel_spectrogram(audio: np.ndarray, sr: int = _EFFNET_SR) -> np.ndarray:
    """Compute log-compressed mel-spectrogram matching Essentia's TensorflowInputMusiCNN.

    Returns array of shape ``(frames, 96)``.
    """
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
    """Slice a ``(frames, n_mels)`` mel-spectrogram into overlapping patches.

    Returns ``(N, patch_frames, n_mels)`` suitable for EffNet input.
    """
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


def _load_effnet_segments(path: str) -> list[np.ndarray]:
    """Load up to _EFFNET_NUM_SEGMENTS segments for EffNet processing.

    Short tracks (<=1.5x segment length) return a single whole-file segment.
    Longer tracks return segments centred at 20%, 50%, and 80%.
    """
    try:
        duration = librosa_get_duration(path=path)
        if duration <= 0:
            return []

        if duration <= _EFFNET_SEGMENT_SECONDS * 1.5:
            audio, _ = librosa_load(path, sr=_EFFNET_SR, mono=True)
            return [audio] if len(audio) > 0 else []

        positions = [0.2, 0.5, 0.8]
        segments: list[np.ndarray] = []
        for pos in positions:
            centre = duration * pos
            offset = max(0.0, centre - _EFFNET_SEGMENT_SECONDS / 2)
            offset = min(offset, max(0.0, duration - _EFFNET_SEGMENT_SECONDS))
            audio, _ = librosa_load(
                path, sr=_EFFNET_SR, mono=True,
                offset=offset, duration=_EFFNET_SEGMENT_SECONDS,
            )
            if len(audio) > 0:
                segments.append(audio)
        return segments
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return []


def _run_effnet_batched(patches: np.ndarray) -> np.ndarray:
    """Run EffNet inference in capped batches to limit memory pressure."""
    input_name = _effnet_session.get_inputs()[0].name
    output_name = _effnet_session.get_outputs()[0].name

    if len(patches) <= _EFFNET_BATCH_CAP:
        return _effnet_session.run([output_name], {input_name: patches})[0]

    results = []
    for i in range(0, len(patches), _EFFNET_BATCH_CAP):
        batch = patches[i:i + _EFFNET_BATCH_CAP]
        results.append(_effnet_session.run([output_name], {input_name: batch})[0])
    return np.concatenate(results, axis=0)


def _preprocess_effnet(path: str) -> np.ndarray | None:
    """Load audio and compute mel patches — no ONNX inference.

    Returns patches array of shape ``(N, 128, 96)`` or None on failure.
    """
    try:
        segments = _load_effnet_segments(path)
        if not segments:
            return None
        all_patches: list[np.ndarray] = []
        for audio in segments:
            mel = compute_mel_spectrogram(audio)
            all_patches.append(patch_mel_spectrogram(mel))
        return np.concatenate(all_patches, axis=0)
    except Exception as e:
        log.warning("EffNet preprocessing failed for %s: %s", path, e)
        return None


def generate_effnet_embedding(path: str) -> np.ndarray | None:
    """Generate a Discogs-EffNet embedding for a single audio file.

    Loads multiple segments (like CLAP), computes mel patches for each,
    runs batched ONNX inference, and averages all activations.
    Returns a 1-D float32 array, or None on failure.
    """
    if not _effnet_loaded:
        load_effnet()
    if not _effnet_loaded or _effnet_session is None:
        return None

    patches = _preprocess_effnet(path)
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
) -> dict[str, np.ndarray | None]:
    """Generate EffNet embeddings for multiple files with combined ONNX inference.

    Preprocessing (audio load + mel + patch) runs sequentially — libsndfile's
    MP3 decoder is not thread-safe and crashes under concurrent access.
    The speedup comes from concatenating patches from all tracks into fewer,
    larger ONNX inference calls.
    """
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
        patches = _preprocess_effnet(path)
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


# ── Audio features (librosa, no ML model) ─────────────────────

# Circle-of-fifths position for each pitch class (C, C#, D, … B).
_KEY_TO_FIFTHS: dict[str, int] = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
    "F#": 6, "Gb": 6,
    "C#": 7, "Db": 7,
    "G#": 8, "Ab": 8,
    "D#": 9, "Eb": 9,
    "A#": 10, "Bb": 10,
    "F": 11,
}

# Krumhansl-Schmuckler key profiles (correlation templates).
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

_BPM_LO = 60.0
_BPM_HI = 200.0
_FEATURE_SR = 22050
_FEATURE_EXCERPT_SECONDS = 20.0
_FEATURE_HOP = 1024


def _load_feature_audio_excerpt(path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Load a bounded excerpt for lightweight feature extraction.

    For long tracks, decode a centered excerpt to avoid full-file decode cost.
    This keeps runtime predictable while preserving enough rhythmic/harmonic
    content for coarse descriptors.
    """
    try:
        duration = librosa_get_duration(path=path)
        if duration <= 0:
            return None, None

        if duration <= _FEATURE_EXCERPT_SECONDS * 1.25:
            audio, sr = librosa_load(path, sr=_FEATURE_SR, mono=True)
            return (audio, sr) if len(audio) > 0 else (None, None)

        offset = max(0.0, (duration - _FEATURE_EXCERPT_SECONDS) * 0.5)
        audio, sr = librosa_load(
            path,
            sr=_FEATURE_SR,
            mono=True,
            offset=offset,
            duration=_FEATURE_EXCERPT_SECONDS,
        )
        return (audio, sr) if len(audio) > 0 else (None, None)
    except Exception as e:
        log.warning("Audio excerpt load failed for %s: %s", path, e)
        return None, None


def detect_key(audio: np.ndarray, sr: int = 44100) -> tuple[str, str]:
    """Detect musical key using Krumhansl-Schmuckler on chroma features.

    Returns ``(key_name, scale)`` where *key_name* is one of the 12 pitch
    classes (sharps only, e.g. ``"C#"`` not ``"Db"``) and *scale* is
    ``"major"`` or ``"minor"``.
    """
    # STFT chroma is much cheaper than CQT and sufficient for coarse key hints.
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=_FEATURE_HOP)
    mean_chroma = chroma.mean(axis=1)  # (12,)

    best_corr = -np.inf
    best_key = "C"
    best_scale = "major"

    for shift in range(12):
        rotated = np.roll(mean_chroma, -shift)
        corr_maj = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
        corr_min = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = _PITCH_CLASSES[shift]
            best_scale = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = _PITCH_CLASSES[shift]
            best_scale = "minor"

    return best_key, best_scale


def _compute_danceability(
    audio: np.ndarray,
    sr: int = 44100,
    onset_env: np.ndarray | None = None,
    tempo_bpm: float | None = None,
    hop_length: int = _FEATURE_HOP,
) -> float:
    """Estimate danceability from onset strength regularity.

    Measures how periodic/regular the rhythmic onsets are by computing the
    autocorrelation of the onset strength envelope and comparing the peak
    at the dominant tempo lag to the overall mean.
    """
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    if len(onset_env) < 4:
        return 0.0

    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    if len(ac) < 2:
        return 0.0

    ac = ac / (ac[0] + 1e-9)

    if tempo_bpm is None:
        tempo_bpm = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if hasattr(tempo_bpm, "__len__"):
            tempo_bpm = float(tempo_bpm[0])

    # Convert BPM to lag in onset-envelope frames.
    frames_per_sec = sr / hop_length
    if tempo_bpm > 0:
        lag = int(round(60.0 * frames_per_sec / tempo_bpm))
    else:
        lag = 0

    if 0 < lag < len(ac):
        peak = ac[lag]
    else:
        peak = float(np.max(ac[1:])) if len(ac) > 1 else 0.0

    return float(np.clip(peak, 0.0, 1.0))


def extract_audio_features(path: str) -> np.ndarray | None:
    """Extract a compact feature vector from an audio file.

    Returns float32 array of shape (6,):
        [tempo_norm, key_cos, key_sin, mode, energy_norm, danceability]
    or None on failure.
    """
    try:
        audio, sr = _load_feature_audio_excerpt(path)
        if audio is None or sr is None or len(audio) == 0:
            return None

        # Tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=_FEATURE_HOP)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0])
        tempo_norm = float(np.clip((tempo - _BPM_LO) / (_BPM_HI - _BPM_LO), 0.0, 1.0))

        # Key → circle-of-fifths unit circle
        key_str, scale_str = detect_key(audio, sr)
        fifths_pos = _KEY_TO_FIFTHS.get(key_str, 0)
        angle = 2.0 * math.pi * fifths_pos / 12.0
        key_cos = math.cos(angle)
        key_sin = math.sin(angle)
        mode = 1.0 if scale_str == "major" else 0.0

        # Energy (RMS, log-scaled, normalized to ~[0,1])
        rms = librosa.feature.rms(y=audio)
        mean_rms = float(rms.mean())
        energy_norm = float(np.clip(np.log1p(mean_rms * 100) / 5.0, 0.0, 1.0))

        # Danceability
        danceability = _compute_danceability(
            audio, sr, onset_env=onset_env, tempo_bpm=tempo, hop_length=_FEATURE_HOP
        )

        return np.array(
            [tempo_norm, key_cos, key_sin, mode, energy_norm, danceability],
            dtype=np.float32,
        )
    except Exception as e:
        log.warning("Audio feature extraction failed for %s: %s", path, e)
        return None


def generate_audio_features_batch(
    paths: list[str],
) -> dict[str, np.ndarray | None]:
    """Extract audio features for multiple paths.

    IMPORTANT: MP3 decode in the librosa/audioread stack is not reliably
    thread-safe in this app's mixed workload, so preprocessing must remain
    sequential to avoid intermittent native crashes.
    """
    if not paths:
        return {}

    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    for p in paths:
        results[p] = extract_audio_features(p)

    return results


def warmup_audio_features() -> None:
    """Pre-warm librosa kernels used by feature extraction.

    This reduces first-call latency spikes without touching user files.
    """
    try:
        seconds = 4
        n = int(_FEATURE_SR * seconds)
        t = np.linspace(0.0, seconds, num=n, endpoint=False, dtype=np.float32)
        audio = 0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
        _ = librosa.onset.onset_strength(y=audio, sr=_FEATURE_SR, hop_length=_FEATURE_HOP)
        _ = librosa.feature.tempo(y=audio, sr=_FEATURE_SR, hop_length=_FEATURE_HOP)
        _ = librosa.feature.chroma_stft(y=audio, sr=_FEATURE_SR, n_fft=2048, hop_length=_FEATURE_HOP)
        _ = librosa.feature.rms(y=audio)
        _ = detect_key(audio, _FEATURE_SR)
        _ = _compute_danceability(audio, _FEATURE_SR, hop_length=_FEATURE_HOP)
    except Exception as e:
        log.debug("Audio feature warmup skipped: %s", e)
