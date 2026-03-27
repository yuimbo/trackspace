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


def decode_reliability_user_message(meta_seconds: float, decoded_seconds: float) -> str:
    """Human-readable note when container metadata and decoded audio length disagree."""
    if not math.isfinite(meta_seconds) or meta_seconds <= 0:
        return (
            "Audio could not be decoded to match the file metadata; "
            "the file may be damaged — try re-encoding or replacing it."
        )
    return (
        "Metadata reports ~"
        f"{meta_seconds:.0f}s"
        " of audio but only ~"
        f"{decoded_seconds:.1f}s"
        " decoded — file may be damaged or mis-tagged; re-encode for reliable analysis."
    )


def _three_segments_from_samples(
    y: np.ndarray,
    *,
    segment_seconds: float,
    sr: int,
) -> list[np.ndarray]:
    """Take 20 % / 50 % / 80 % windows from an in-memory waveform (no MP3 seek)."""
    n = int(y.shape[0])
    if n <= 0:
        return []
    w = int(round(float(segment_seconds) * float(sr)))
    if w <= 0:
        return [y]
    total_dur = n / float(sr)
    if total_dur <= segment_seconds * 1.5:
        return [y]
    out: list[np.ndarray] = []
    for pos in (0.2, 0.5, 0.8):
        centre = int(round(pos * (n - 1)))
        start = max(0, centre - w // 2)
        end = min(n, start + w)
        start = max(0, end - w)
        chunk = y[start:end]
        if chunk.size > 0:
            out.append(chunk)
    return out


def _mono_full_short_metadata_decode(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    meta_short_max: float,
) -> tuple[bool, np.ndarray | None, int | None, str | None]:
    """If *meta_dur* is within the short-track budget, decode the whole file.

    Returns ``(False, None, None, None)`` when *meta_dur* > *meta_short_max* (caller
    should use long-track logic). Otherwise ``(True, audio, sr, warn)`` with
    *audio* ``None`` only when decode yielded no samples.
    """
    if meta_dur > meta_short_max:
        return False, None, None, None
    audio, sr_a = librosa_load(path, sr=sr, mono=True)
    if audio.size == 0:
        return True, None, None, None
    dec = float(audio.shape[0]) / float(sr_a)
    warn = None
    if meta_dur >= 2.0 and dec < meta_dur * 0.25:
        warn = decode_reliability_user_message(meta_dur, dec)
    return True, audio, sr_a, warn


def _probe_long_track_decode(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    segment_seconds: float,
) -> tuple[str, np.ndarray | None, int | None, str | None]:
    """Opening probe for files longer than the short-track cutoff.

    Returns a tag and payload:

    * ``"empty"`` — probe could not read audio.
    * ``"full"`` — metadata is unreliable vs probe; *audio* is a full decode.
    * ``"seek"`` — timed seeks are likely safe; other fields unused.
    """
    probe_len = min(segment_seconds, meta_dur)
    probe, sr_a = librosa_load(
        path, sr=sr, mono=True, offset=0.0, duration=probe_len,
    )
    if probe.size == 0:
        return "empty", None, None, None
    probe_dur = float(probe.shape[0]) / float(sr_a)
    if (
        probe_len >= 1.0
        and probe_dur < max(float(segment_seconds), probe_len) * 0.2
    ):
        audio, sr_a = librosa_load(path, sr=sr, mono=True)
        if audio.size == 0:
            return "empty", None, None, None
        tot_dur = float(audio.shape[0]) / float(sr_a)
        warn = decode_reliability_user_message(meta_dur, tot_dur)
        return "full", audio, sr_a, warn
    return "seek", None, None, None


def _try_librosa_seek_mono(
    path: str,
    *,
    sr: int,
    offset: float,
    read_dur: float,
) -> np.ndarray:
    """Partial decode; empty array on failure (librosa may raise on bad seeks)."""
    try:
        audio, _ = librosa_load(
            path, sr=sr, mono=True, offset=offset, duration=read_dur,
        )
        return audio
    except Exception:
        return np.array([], dtype=np.float32)


def _mono_centered_window_or_full(
    path: str,
    *,
    sr: int,
    meta_dur: float,
    window_seconds: float,
) -> tuple[np.ndarray | None, int | None, str | None]:
    """One window centred using metadata duration, or full decode if seek returns nothing.

    Used for lightweight feature extraction when a single excerpt is enough.
    """
    offset = max(0.0, (meta_dur - window_seconds) * 0.5)
    read_dur = min(window_seconds, max(0.0, meta_dur - offset))
    if read_dur < 0.05:
        return None, None, None
    try:
        audio, sr_o = librosa_load(
            path, sr=sr, mono=True, offset=offset, duration=read_dur,
        )
        partial_ok = True
    except Exception:
        audio, sr_o = librosa_load(path, sr=sr, mono=True)
        partial_ok = False
    if audio.size == 0:
        audio, sr_o = librosa_load(path, sr=sr, mono=True)
        if audio.size == 0:
            return None, None, None
        dec = float(audio.shape[0]) / float(sr_o)
        warn = (
            decode_reliability_user_message(meta_dur, dec)
            if meta_dur > dec + 1.0
            else None
        )
        return audio, sr_o, warn
    dec = float(audio.shape[0]) / float(sr_o)
    warn = None
    if partial_ok:
        # Bound excerpt: decoded length should match *read_dur*, not *meta_dur*.
        expected = min(float(read_dur), meta_dur)
        if expected >= 2.0 and dec < expected * 0.25:
            warn = decode_reliability_user_message(meta_dur, dec)
    else:
        if meta_dur > window_seconds * 1.25 and dec < meta_dur * 0.35:
            warn = decode_reliability_user_message(meta_dur, dec)
    return audio, sr_o, warn


def load_resilient_audio_segments(
    path: str,
    *,
    sr: int,
    segment_seconds: float,
) -> tuple[list[np.ndarray], str | None]:
    """Decode one or three segments; tolerate bad MP3 duration metadata and broken seeks.

    ``librosa.get_duration`` can report minutes from headers while audioread only
    yields a few hundred samples, and ``librosa.load(..., offset>0)`` may then
    raise ``ValueError: negative dimensions are not allowed``.

    Returns ``(segments, user_warning)``. *user_warning* is a short message when
    metadata and decoded length clearly disagree (for UI / SSE).
    """
    meta_dur = float(librosa_get_duration(path=path))
    if not math.isfinite(meta_dur) or meta_dur <= 0:
        return [], None

    done, audio, sr_a, warn = _mono_full_short_metadata_decode(
        path,
        sr=sr,
        meta_dur=meta_dur,
        meta_short_max=segment_seconds * 1.5,
    )
    if done:
        if audio is None:
            return [], None
        return [audio], warn

    kind, audio, sr_a, warn = _probe_long_track_decode(
        path, sr=sr, meta_dur=meta_dur, segment_seconds=segment_seconds,
    )
    if kind == "empty":
        return [], None
    if kind == "full":
        assert audio is not None and sr_a is not None
        tot_dur = float(audio.shape[0]) / float(sr_a)
        if tot_dur <= segment_seconds * 1.5:
            return [audio], warn
        return (
            _three_segments_from_samples(
                audio, segment_seconds=segment_seconds, sr=int(sr_a),
            ),
            warn,
        )

    segments: list[np.ndarray] = []
    for pos in (0.2, 0.5, 0.8):
        centre = meta_dur * pos
        offset = max(0.0, centre - segment_seconds / 2)
        offset = min(offset, max(0.0, meta_dur - segment_seconds))
        read_dur = min(segment_seconds, max(0.0, meta_dur - offset))
        if read_dur < 0.05:
            continue
        audio = _try_librosa_seek_mono(
            path, sr=sr, offset=offset, read_dur=read_dur,
        )
        if audio.size > 0:
            segments.append(audio)

    if len(segments) >= 2:
        return segments, None

    audio, sr_a = librosa_load(path, sr=sr, mono=True)
    if audio.size == 0:
        return [], None
    tot_dur = float(audio.shape[0]) / float(sr_a)
    warn = None
    if meta_dur > segment_seconds * 1.5 and tot_dur < meta_dur * 0.35:
        warn = decode_reliability_user_message(meta_dur, tot_dur)
    if tot_dur <= segment_seconds * 1.5:
        return [audio], warn
    return (
        _three_segments_from_samples(
            audio, segment_seconds=segment_seconds, sr=int(sr_a),
        ),
        warn,
    )


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


def _load_effnet_segments(path: str) -> tuple[list[np.ndarray], str | None]:
    """Load up to _EFFNET_NUM_SEGMENTS segments for EffNet processing.

    Short tracks (<=1.5x segment length) return a single whole-file segment.
    Longer tracks return segments centred at 20%, 50%, and 80%.
    """
    try:
        return load_resilient_audio_segments(
            path, sr=_EFFNET_SR, segment_seconds=_EFFNET_SEGMENT_SECONDS,
        )
    except Exception as e:
        log.warning("Failed to load audio %s: %s", path, e)
        return [], None


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


def _preprocess_effnet(path: str) -> tuple[np.ndarray | None, str | None]:
    """Load audio and compute mel patches — no ONNX inference.

    Returns ``(patches | None, decode_warning | None)``; patches have shape
    ``(N, 128, 96)`` on success.
    """
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
    """Generate a Discogs-EffNet embedding for a single audio file.

    Loads multiple segments (like CLAP), computes mel patches for each,
    runs batched ONNX inference, and averages all activations.
    Returns a 1-D float32 array, or None on failure.
    """
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
    """Generate EffNet embeddings for multiple files with combined ONNX inference.

    Preprocessing (audio load + mel + patch) runs sequentially — libsndfile's
    MP3 decoder is not thread-safe and crashes under concurrent access.
    The speedup comes from concatenating patches from all tracks into fewer,
    larger ONNX inference calls.

    If *decode_warnings* is provided, maps absolute *path* → user-facing warning
    when metadata vs decoded audio disagrees (same path may appear in other stages).
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


def _load_feature_audio_excerpt(
    path: str,
) -> tuple[np.ndarray | None, int | None, str | None]:
    """Load a bounded excerpt for lightweight feature extraction.

    For long tracks, decode a centered excerpt to avoid full-file decode cost.
    This keeps runtime predictable while preserving enough rhythmic/harmonic
    content for coarse descriptors.

    Shares probe / short-track logic with :func:`load_resilient_audio_segments`;
    centred-window loading uses :func:`_mono_centered_window_or_full`.

    Returns ``(audio, sr, user_warning)``.
    """
    try:
        meta_dur = float(librosa_get_duration(path=path))
        if not math.isfinite(meta_dur) or meta_dur <= 0:
            return None, None, None

        done, audio, sr, warn = _mono_full_short_metadata_decode(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            meta_short_max=_FEATURE_EXCERPT_SECONDS * 1.25,
        )
        if done:
            if audio is None:
                return None, None, None
            return audio, sr, warn

        kind, audio, sr, warn = _probe_long_track_decode(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            segment_seconds=_FEATURE_EXCERPT_SECONDS,
        )
        if kind == "empty":
            return None, None, None
        if kind == "full":
            if audio is None:
                return None, None, None
            return audio, sr, warn

        return _mono_centered_window_or_full(
            path,
            sr=_FEATURE_SR,
            meta_dur=meta_dur,
            window_seconds=_FEATURE_EXCERPT_SECONDS,
        )
    except Exception as e:
        log.warning("Audio excerpt load failed for %s: %s", path, e)
        return None, None, None


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


def extract_audio_features_and_warning(
    path: str,
) -> tuple[np.ndarray | None, str | None]:
    """Like :func:`extract_audio_features` but also returns a user-facing decode note."""
    try:
        audio, sr, warn = _load_feature_audio_excerpt(path)
        if audio is None or sr is None or len(audio) == 0:
            return None, warn

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

        return (
            np.array(
                [tempo_norm, key_cos, key_sin, mode, energy_norm, danceability],
                dtype=np.float32,
            ),
            warn,
        )
    except Exception as e:
        log.warning("Audio feature extraction failed for %s: %s", path, e)
        return None, None


def extract_audio_features(path: str) -> np.ndarray | None:
    """Extract a compact feature vector from an audio file.

    Returns float32 array of shape (6,):
        [tempo_norm, key_cos, key_sin, mode, energy_norm, danceability]
    or None on failure.
    """
    vec, _warn = extract_audio_features_and_warning(path)
    return vec


# Circle-of-fifths index 0..11 — same layout as ``_KEY_TO_FIFTHS`` values.
_FIFTHS_POS_NAMES = (
    "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F",
)


def audio_features_display_bpm_key(vec: np.ndarray | None) -> tuple[int | None, str | None]:
    """Decode a cached 6-D feature vector to rounded BPM and key label (e.g. ``Am``)."""
    if vec is None or int(vec.size) < AUDIO_FEATURE_DIM:
        return None, None
    tnorm = float(np.clip(float(vec[0]), 0.0, 1.0))
    bpm_f = tnorm * (_BPM_HI - _BPM_LO) + _BPM_LO
    bpm = int(round(bpm_f))
    key_cos = float(vec[1])
    key_sin = float(vec[2])
    mode_major = float(vec[3]) >= 0.5
    angle = math.atan2(key_sin, key_cos)
    fifths = int(round(12.0 * angle / (2.0 * math.pi))) % 12
    name = _FIFTHS_POS_NAMES[fifths]
    key_label = name if mode_major else f"{name}m"
    return bpm, key_label


def generate_audio_features_batch(
    paths: list[str],
    decode_warnings: dict[str, str] | None = None,
) -> dict[str, np.ndarray | None]:
    """Extract audio features for multiple paths.

    IMPORTANT: MP3 decode in the librosa/audioread stack is not reliably
    thread-safe in this app's mixed workload, so preprocessing must remain
    sequential to avoid intermittent native crashes.

    If *decode_warnings* is provided, fills ``abs_path -> message`` for unreliable decodes.
    """
    if not paths:
        return {}

    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    for p in paths:
        vec, w = extract_audio_features_and_warning(p)
        results[p] = vec
        if w and decode_warnings is not None:
            decode_warnings[p] = w

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
