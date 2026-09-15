"""MAEST (`discogs-maest-30s-pw-129e-519l`) — genre-aware embeddings + 519 style logits.

Primary genre model per the microgenre attack plan: trained on a Discogs
style taxonomy, so its 519 logits already describe each track as a fuzzy point
in genre space. Both outputs are cached — the logits are frequently *more*
useful for microgenre clustering than the hidden embedding.

Two design points worth knowing:

**Mel on CPU, transformer on GPU.** ``torchaudio``'s STFT window stays on CPU
under MPS (``stft input and window must be on the same device``), so the mel
front end is always run on CPU and only the transformer is moved to the
accelerator. This is not a workaround for a bug in our code — it is how the
model must be driven on Apple Silicon.

**Excerpt-level vectors are pooled, not concatenated.** Three 30 s excerpts at
20/50/80 % are embedded as one batch and mean-pooled (plan §2/§4). Per-excerpt
vectors are kept in the returned :class:`MaestAnalysis` so callers can later
detect genre-spanning tracks without re-running inference.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio_decode import load_resilient_audio_segments

log = logging.getLogger(__name__)

MAEST_ARCH = "discogs-maest-30s-pw-129e-519l"
MAEST_SR = 16000
#: Plan §2 — three 30 s excerpts instead of whole tracks.
MAEST_SEGMENT_SECONDS = 30
MAEST_NUM_SEGMENTS = 3

MAEST_EMBED_DIM = 768
MAEST_LOGIT_DIM = 519

#: Bump when excerpt strategy, pooling, or the checkpoint changes.
MAEST_VERSION = 1

_model = None
_mel = None
_device = "cpu"
_model_lock = threading.Lock()
#: Serialises MAEST GPU work against CLAP/mlx-vis — see AGENTS.md.
inference_lock = threading.Lock()
_loaded = False
_load_failed: str | None = None


@dataclass(frozen=True)
class MaestAnalysis:
    """Pooled track-level MAEST outputs plus the per-excerpt vectors."""

    embedding: np.ndarray  # (768,)
    logits: np.ndarray  # (519,)
    excerpt_embeddings: np.ndarray  # (n_excerpts, 768)
    num_excerpts: int


def is_model_ready() -> bool:
    return _loaded


def load_error() -> str | None:
    """Reason MAEST is unavailable, or ``None`` when it loaded (or is untried)."""
    return _load_failed


def load_model() -> None:
    """Load MAEST once; safe to call from several threads."""
    global _model, _mel, _device, _loaded, _load_failed
    if _loaded:
        return
    with _model_lock:
        if _loaded:
            return
        try:
            import torch
            from maest_infer import get_maest
        except ImportError as e:
            _load_failed = f"maest-infer not installed: {e}"
            log.warning("MAEST unavailable — %s", _load_failed)
            return

        try:
            log.info("Loading MAEST model %s …", MAEST_ARCH)
            model = get_maest(arch=MAEST_ARCH)
            model.eval()
            model.init_melspectrogram()

            if torch.cuda.is_available():
                _device = "cuda"
            elif torch.backends.mps.is_available():
                _device = "mps"
            else:
                _device = "cpu"
            _model = model.to(_device)

            # The mel front end must run on CPU: torchaudio's STFT requires the
            # input and the window buffer on the same device, and `model.to()`
            # would have dragged the window onto MPS. Detach it from the model
            # and force it back to CPU, then clear the attribute so a stray
            # waveform-input call can never re-enter the GPU mel path.
            _mel = model.melspectrogram.to("cpu")
            model.melspectrogram = None
            _loaded = True
            _load_failed = None
            log.info("MAEST model loaded (device=%s).", _device)
        except Exception as e:
            _load_failed = f"{type(e).__name__}: {e}"
            log.exception("MAEST model load failed")


def _load_excerpts(path: str) -> tuple[list[np.ndarray], str | None]:
    """Decode up to three 30 s excerpts at 20/50/80 % of the track."""
    try:
        return load_resilient_audio_segments(
            path, sr=MAEST_SR, segment_seconds=MAEST_SEGMENT_SECONDS
        )
    except Exception as e:
        log.warning("MAEST: failed to load audio %s: %s", path, e)
        return [], None


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Force an excerpt to exactly *target_len* samples so a batch can stack."""
    n = int(audio.shape[0])
    if n == target_len:
        return audio
    if n > target_len:
        return audio[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = audio
    return out


def analyze(
    path: str,
    *,
    on_decode_warning: Callable[[str], None] | None = None,
) -> MaestAnalysis | None:
    """Run MAEST on one track, returning pooled embedding + logits."""
    results = analyze_batch([path], on_decode_warning=_single_warn(on_decode_warning))
    return results.get(path)


def _single_warn(
    cb: Callable[[str], None] | None,
) -> Callable[[str, str], None] | None:
    if cb is None:
        return None

    def _fn(_path: str, message: str) -> None:
        cb(message)

    return _fn


def analyze_batch(
    paths: list[str],
    *,
    on_decode_warning: Callable[[str, str], None] | None = None,
    max_batch: int = 12,
) -> dict[str, MaestAnalysis | None]:
    """Analyse several tracks, batching excerpt inference across tracks.

    Decoding happens per track on the calling thread; inference is grouped into
    batches of at most *max_batch* excerpts so GPU memory stays bounded while
    still amortising kernel-launch cost (measured ~9x faster than per-excerpt
    calls on MPS).
    """
    results: dict[str, MaestAnalysis | None] = {p: None for p in paths}
    if not paths:
        return results

    if not _loaded:
        load_model()
    if not _loaded:
        return results

    import torch

    target_len = MAEST_SR * MAEST_SEGMENT_SECONDS

    # ── decode all excerpts, remembering which track each belongs to ──
    flat: list[np.ndarray] = []
    owner: list[str] = []
    for p in paths:
        segments, warn = _load_excerpts(p)
        if warn and on_decode_warning:
            on_decode_warning(p, warn)
        for seg in segments[:MAEST_NUM_SEGMENTS]:
            if seg.size == 0:
                continue
            flat.append(_pad_or_trim(np.asarray(seg, dtype=np.float32), target_len))
            owner.append(p)

    if not flat:
        return results

    per_track_emb: dict[str, list[np.ndarray]] = {p: [] for p in paths}
    per_track_logits: dict[str, list[np.ndarray]] = {p: [] for p in paths}

    for start in range(0, len(flat), max_batch):
        chunk = flat[start : start + max_batch]
        chunk_owner = owner[start : start + max_batch]
        try:
            wave = torch.from_numpy(np.stack(chunk, axis=0))
            # Mel on CPU by design; only the transformer runs on the accelerator.
            mel = _mel(wave)
            with inference_lock:
                mel_dev = mel.to(_device)
                with torch.no_grad():
                    logits, emb = _model(mel_dev, melspectrogram_input=True)
                logits_np = logits.detach().to("cpu").numpy().astype(np.float32)
                emb_np = emb.detach().to("cpu").numpy().astype(np.float32)
        except Exception as e:
            log.warning("MAEST batch inference failed (%d excerpts): %s", len(chunk), e)
            continue

        for i, p in enumerate(chunk_owner):
            per_track_emb[p].append(emb_np[i])
            per_track_logits[p].append(logits_np[i])

    for p in paths:
        embs = per_track_emb.get(p) or []
        lgts = per_track_logits.get(p) or []
        if not embs or not lgts:
            continue
        stacked = np.stack(embs, axis=0).astype(np.float32)
        results[p] = MaestAnalysis(
            embedding=stacked.mean(axis=0).astype(np.float32),
            logits=np.stack(lgts, axis=0).mean(axis=0).astype(np.float32),
            excerpt_embeddings=stacked,
            num_excerpts=len(embs),
        )

    return results


_labels_cache: list[str] | None = None


def discogs_style_labels() -> list[str]:
    """The 519 Discogs ``Genre---Style`` label names, or ``[]`` if unavailable."""
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache
    try:
        from maest_infer.discogs_labels import discogs_519labels

        _labels_cache = list(discogs_519labels)
    except Exception as e:
        log.debug("MAEST label table unavailable: %s", e)
        _labels_cache = []
    return _labels_cache


def top_styles(logits: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """Highest-scoring Discogs style labels for one logit vector.

    Used for cluster naming and track inspection (plan §16). Returns an empty
    list if the label table is unavailable rather than raising.
    """
    labels = discogs_style_labels()
    if not labels:
        return []

    v = np.asarray(logits, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return []
    n = min(len(labels), v.size)
    idx = np.argsort(-v[:n])[: max(1, k)]
    return [(str(labels[int(i)]), float(v[int(i)])) for i in idx]


def warmup() -> None:
    """Run one tiny forward pass so the first real batch is not paying init cost."""
    if not _loaded:
        load_model()
    if not _loaded:
        return
    try:
        import torch

        silence = np.zeros(MAEST_SR * MAEST_SEGMENT_SECONDS, dtype=np.float32)
        wave = torch.from_numpy(silence[None, :])
        mel = _mel(wave)
        with inference_lock:
            with torch.no_grad():
                _model(mel.to(_device), melspectrogram_input=True)
    except Exception as e:
        log.debug("MAEST warmup skipped: %s", e)
