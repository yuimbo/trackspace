"""MERT (`m-a-p/MERT-v2-30s`) — independent 1024-D audio encoder, layer-19 only.

An independent music audio encoder (not a MAEST extension) whose L19 embedding is
the ``genre_full`` recipe's MERT block (weight .15). This module emits exactly one
output — the pooled layer-19 vector — and keeps only that layer (never an L23 that
nothing consumes).

**Layer 19 is captured with a scoped hook per homogeneous length subgroup.** The
pinned ``forward`` encodes each distinct waveform length separately (``_encode``
per unique ``attention_mask`` length) and assembles the padded per-layer states
into the full batch (forward lines 343-356). To avoid materialising the 24-state
tuple, we bucket the already-decoded inputs by raw sample length — the same key
``forward`` buckets on (``attention_mask.sum``, line 334) — run one forward per
homogeneous subgroup with ``output_hidden_states=False`` and a scoped hook on
``layers[18]`` (which fires exactly once per subgroup, since ``_encode`` loops the
layers once), then assemble L19 zero-padded exactly as the pinned forward does.
Index 18 is one-based block 19: ``_encode`` appends one state per layer
(``hidden_states[0]`` = ``layers[0]`` = block 1, config ``num_hidden_layers=24``).

**Licence and code gates.** MERT weights are CC-BY-NC 4.0 and the repo ships
custom modeling code, so nothing loads or runs until the caller explicitly accepts
the non-commercial licence AND ``trust_remote_code``. Weights are never
auto-downloaded by default (``local_files_only``); ``allow_download`` is an
explicit, separate opt-in. Accelerator dispatch at load, warmup, and inference all
run under the shared ``clap.inference_lock`` — there is no new accelerator lock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from .audio_decode import load_resilient_audio_segments
from .clap import inference_lock

log = logging.getLogger(__name__)

MERT_REPO = "m-a-p/MERT-v2-30s"
#: Pinned 40-hex commit (never a branch). Custom modeling code is loaded verbatim
#: from this revision only.
MERT_REVISION = "0c0c8c6f6e04dcc39d96f94e0be9978a74848906"
MERT_SR = 24000
MERT_SEGMENT_SECONDS = 30
MERT_NUM_SEGMENTS = 3

MERT_EMBED_DIM = 1024
#: ``hidden_states[18]`` == ``layers[18]`` == one-based block 19: the pinned
#: ``_encode`` appends one state per layer, so ``hidden_states[0]`` is block 1 and
#: block N is ``hidden_states[N-1]`` (config ``num_hidden_layers=24``).
MERT_L19_INDEX = 18
MERT_L19_BLOCK = MERT_L19_INDEX + 1  # 19

#: Bump when MERT excerpt strategy, pooling, or checkpoint changes.
MERT_VERSION = 1

_model = None
_extractor = None
_device = "cpu"
_model_lock = threading.Lock()
_loaded = False
_load_failed: str | None = None

#: Explicit security gates. All default off: MERT must never load or run without them.
_accept_noncommercial = False
_trust_remote_code = False
_allow_download = False


def is_model_ready() -> bool:
    return _loaded


def load_error() -> str | None:
    """Reason MERT is unavailable, or ``None`` when it loaded (or is untried)."""
    return _load_failed


def set_licence_acceptance(accepted: bool) -> None:
    """Explicitly accept the CC-BY-NC 4.0 licence for the MERT weights."""
    global _accept_noncommercial
    _accept_noncommercial = bool(accepted)


def set_trust_remote_code(trusted: bool) -> None:
    """Allow MERT's repo-specific custom modeling code, pinned to MERT_REVISION."""
    global _trust_remote_code
    _trust_remote_code = bool(trusted)


def set_allow_download(allowed: bool) -> None:
    """Allow fetching uncached weights; default keeps ``local_files_only``."""
    global _allow_download
    _allow_download = bool(allowed)


def licence_acceptance() -> bool:
    return _accept_noncommercial


def load_model() -> None:
    """Load MERT once, after the explicit licence/trust gates; never raises."""
    global _model, _extractor, _device, _loaded, _load_failed
    if _loaded:
        return
    with _model_lock:
        if _loaded:
            return
        # Gates are checked before any transformers import so a non-accepted server
        # never pays the model-load import cost or touches the HF cache.
        if not _accept_noncommercial:
            _load_failed = (
                "MERT weights are CC-BY-NC 4.0; the licence must be accepted "
                "before loading"
            )
            log.warning("MERT unavailable — %s", _load_failed)
            return
        if not _trust_remote_code:
            _load_failed = (
                "MERT ships repo-specific modeling code; pass --trust-remote-code "
                "(pinned to the exact MERT_REVISION)"
            )
            log.warning("MERT unavailable — %s", _load_failed)
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as e:
            _load_failed = f"transformers not installed: {e}"
            log.warning("MERT unavailable — %s", _load_failed)
            return

        try:
            log.info("Loading MERT %s @ %s …", MERT_REPO, MERT_REVISION)
            kw = dict(
                revision=MERT_REVISION,
                trust_remote_code=_trust_remote_code,
                local_files_only=not _allow_download,
            )
            extractor = AutoFeatureExtractor.from_pretrained(MERT_REPO, **kw)
            model = AutoModel.from_pretrained(MERT_REPO, **kw)
            model.eval()
            import torch

            if torch.cuda.is_available():
                _device = "cuda"
            elif torch.backends.mps.is_available():
                _device = "mps"
            else:
                _device = "cpu"
            # Weight dispatch to the accelerator is itself a device op; run it under
            # the shared inference lock so it cannot race mlx-vis/CLAP/MAEST.
            with inference_lock:
                _model = model.to(_device)
            _extractor = extractor
            _loaded = True
            _load_failed = None
            log.info("MERT model loaded (device=%s).", _device)
        except Exception as e:
            _load_failed = f"{type(e).__name__}: {e}"
            log.exception("MERT model load failed")


def mask_pool(hidden, mask) -> np.ndarray:
    """Masked temporal mean: average *hidden* frames where *mask* is 1.

    Ported from the genre-space extractor (``backend/experiments/genre_space/
    extractors.py``) so production pooling is byte-identical to the benchmark.
    Accepts numpy arrays or torch tensors, batched (B, T, D) + (B, T) or single
    (T, D) + (T,). Raises ValueError on shape mismatch, non-binary or all-zero
    masks, or non-finite hidden values.
    """
    import torch

    if isinstance(hidden, torch.Tensor):
        hidden = hidden.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    h, m = np.asarray(hidden), np.asarray(mask)
    if h.ndim == 3 and m.ndim == 2:
        if h.shape[:2] != m.shape:
            raise ValueError(f"hidden {h.shape} and mask {m.shape} temporal dims differ")
        valid = m.sum(axis=1, keepdims=True)
        pooled = (h * m[..., None]).sum(axis=1) / np.maximum(valid, 1.0)
    elif h.ndim == 2 and m.ndim == 1:
        if h.shape[0] != m.shape[0]:
            raise ValueError(f"hidden {h.shape} and mask {m.shape} temporal dims differ")
        valid = m.sum()
        pooled = (h * m[:, None]).sum(axis=0) / max(valid, 1.0)
    else:
        raise ValueError(f"unsupported shapes: hidden {h.shape}, mask {m.shape}")
    if not np.isin(m, (0, 1)).all():
        raise ValueError("mask must be binary (0/1)")
    if not np.isfinite(h).all():
        raise ValueError("hidden contains non-finite values")
    if np.any(valid == 0):
        raise ValueError("mask marks no valid frame for at least one excerpt")
    return np.asarray(pooled, dtype=np.float32)


def _run_forward_l19(model, input_dev, attn_dev):
    """One homogeneous-subgroup forward; return its layer-19 tensor and feature mask.

    Registers a scoped forward hook on ``model.layers[MERT_L19_INDEX]`` and runs
    the forward with ``output_hidden_states=False`` so only this one layer's output
    is retained (no 24-state tuple). A homogeneous group encodes exactly once (the
    pinned ``_encode`` loops the layers once per ``forward``), so the hook must
    fire exactly once; any other count means the layer layout changed and we raise
    loudly rather than silently pooling garbage. The hook and its guard are removed
    in a ``finally`` even if the forward raises.
    """
    import torch

    captured = {"l19": None, "calls": 0}

    def _hook(_module, _args, output):
        captured["calls"] += 1
        if output.ndim != 3 or output.shape[-1] != MERT_EMBED_DIM:
            raise RuntimeError(
                f"MERT L19 hook saw unexpected block output shape {tuple(output.shape)}; "
                "the encoder layer layout changed"
            )
        captured["l19"] = output

    if len(model.layers) <= MERT_L19_INDEX:
        raise ValueError(
            f"MERT has {len(model.layers)} layers; need more than {MERT_L19_INDEX} "
            f"(hidden_states[{MERT_L19_INDEX}])"
        )
    layer = model.layers[MERT_L19_INDEX]
    handle = layer.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            out = model(
                input_values=input_dev,
                attention_mask=attn_dev,
                output_hidden_states=False,
            )
        mask = out.feature_attention_mask
    finally:
        handle.remove()
    if captured["calls"] != 1:
        raise RuntimeError(
            f"MERT L19 hook fired {captured['calls']} times for one homogeneous "
            "group; the layer layout changed"
        )
    if mask is None:
        raise ValueError("MERT output missing feature_attention_mask")
    l19 = captured["l19"]
    if l19 is None:
        raise RuntimeError("MERT L19 hook produced no output")
    return l19, mask


def _extract_l19(model, input_dev, attn_dev, batch_len) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch L19 + feature mask, assembled from per-length forward passes.

    The pinned ``forward`` encodes each distinct raw sample length separately
    (``_encode`` per unique ``attention_mask.sum``) and assembles the padded
    per-layer states into the full batch. We bucket the already-decoded inputs by
    that same raw-length key, run one ``output_hidden_states=False`` forward per
    homogeneous subgroup with a scoped hook on ``layers[18]``, then zero-pad and
    ``index_copy`` each subgroup's L19 into the full batch exactly as the pinned
    forward does — retaining a single 1024-D state instead of the 24-tuple.
    Validates the layer count, the assembled batch/embed dims, and that the mask
    matches the L19 temporal dims.

    Runs forwards sequentially under the caller's ``clap.inference_lock``.
    """
    import torch
    from torch.nn import functional as F

    ratio = int(model.config.inputs_to_logits_ratio)
    hidden_size = int(model.config.hidden_size)
    if attn_dev is None:
        lengths = torch.full(
            (batch_len,), int(input_dev.shape[1]), dtype=torch.long, device=input_dev.device
        )
    else:
        lengths = attn_dev.bool().sum(dim=1)
    if lengths.shape[0] != batch_len:
        raise ValueError(
            f"MERT batch mismatch: {lengths.shape[0]} input rows vs batch_len {batch_len}"
        )
    feature_lengths = lengths // ratio
    maximum = int(feature_lengths.max().item())
    feature_mask = torch.arange(maximum, device=input_dev.device)[None, :] < feature_lengths[:, None]

    l19_full = None
    for length in torch.unique(lengths, sorted=True).tolist():
        indices = torch.where(lengths == length)[0]
        group_in = input_dev.index_select(0, indices)[:, :length]
        group_attn = None if attn_dev is None else attn_dev.index_select(0, indices)[:, :length]
        group_l19, _group_mask = _run_forward_l19(model, group_in, group_attn)
        feat = int(group_l19.shape[1])
        if l19_full is None:
            l19_full = group_l19.new_zeros(batch_len, maximum, hidden_size)
        l19_full = l19_full.index_copy(0, indices, F.pad(group_l19, (0, 0, 0, maximum - feat)))

    if (
        l19_full.ndim != 3
        or l19_full.shape[0] != batch_len
        or l19_full.shape[-1] != MERT_EMBED_DIM
    ):
        raise ValueError(
            f"MERT L19 unexpected shape {tuple(l19_full.shape)}; "
            f"expected ({batch_len}, T, {MERT_EMBED_DIM})"
        )
    if tuple(feature_mask.shape) != tuple(l19_full.shape[:2]):
        raise ValueError(
            f"MERT mask {tuple(feature_mask.shape)} and L19 {tuple(l19_full.shape)} "
            "temporal dims differ"
        )
    return l19_full.detach().to("cpu"), feature_mask.detach().to("cpu")


def _load_excerpts(path: str) -> tuple[list[np.ndarray], str | None]:
    """Decode up to three 30 s excerpts at 20/50/80 % (MERT sample rate)."""
    try:
        return load_resilient_audio_segments(
            path, sr=MERT_SR, segment_seconds=MERT_SEGMENT_SECONDS
        )
    except Exception as e:
        log.warning("MERT: failed to load audio %s: %s", path, e)
        return [], None


def analyze_batch(
    paths: list[str],
    *,
    on_decode_warning: Callable[[str, str], None] | None = None,
    max_batch: int = 12,
) -> dict[str, np.ndarray | None]:
    """Pooled MERT layer-19 vector per track, or ``None`` when a track failed."""
    results: dict[str, np.ndarray | None] = {p: None for p in paths}
    if not paths:
        return results

    if not _loaded:
        load_model()
    if not _loaded:
        return results

    # ── decode all excerpts, remembering which track each belongs to ──
    flat: list[np.ndarray] = []
    owner: list[str] = []
    for p in paths:
        segments, warn = _load_excerpts(p)
        if warn and on_decode_warning:
            on_decode_warning(p, warn)
        for seg in segments[:MERT_NUM_SEGMENTS]:
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
            # Feature extraction stays on CPU (returns float tensors); the
            # transformer forward runs under the shared accelerator lock.
            feats = _extractor(
                chunk,
                sampling_rate=MERT_SR,
                return_attention_mask=True,
                padding=True,
                do_normalize=False,
                return_tensors="pt",
            )
            input_values = feats["input_values"]
            attention_mask = (
                feats["attention_mask"]
                if feats.get("attention_mask") is not None
                else None
            )
            with inference_lock:
                input_dev = input_values.to(_device)
                attn_dev = (
                    attention_mask.to(_device)
                    if attention_mask is not None
                    else None
                )
                # L19 + mask are copied to CPU while still holding the lock.
                l19, mask = _extract_l19(_model, input_dev, attn_dev, len(chunk))
            l19_np = l19.numpy().astype(np.float32)
            mask_np = mask.numpy()
            pooled = np.stack(
                [
                    mask_pool(l19_np[i], mask_np[i])
                    for i in range(len(chunk))
                ],
                axis=0,
            )
        except Exception as e:
            log.warning("MERT batch inference failed (%d excerpts): %s", len(chunk), e)
            continue

        for i, p in enumerate(chunk_owner):
            per_track[p].append(pooled[i])

    for p in paths:
        vecs = per_track.get(p) or []
        if not vecs:
            continue
        pooled_vec = np.stack(vecs, axis=0).mean(axis=0).astype(np.float32)
        if not np.isfinite(pooled_vec).all() or np.all(pooled_vec == 0.0):
            continue
        results[p] = pooled_vec

    return results


def warmup() -> None:
    """Run one tiny forward so the first real batch is not paying init cost."""
    if not _loaded:
        load_model()
    if not _loaded:
        return
    try:
        import torch

        silence = np.zeros(MERT_SR * MERT_SEGMENT_SECONDS, dtype=np.float32)
        feats = _extractor(
            [silence],
            sampling_rate=MERT_SR,
            return_attention_mask=True,
            padding=True,
            do_normalize=False,
            return_tensors="pt",
        )
        with inference_lock:
            with torch.inference_mode():
                _model(
                    input_values=feats["input_values"].to(_device),
                    attention_mask=(
                        feats["attention_mask"].to(_device)
                        if feats.get("attention_mask") is not None
                        else None
                    ),
                )
    except Exception as e:
        log.debug("MERT warmup skipped: %s", e)
