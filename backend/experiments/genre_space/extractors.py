"""Genre-space feature extractors: frozen-window MAEST and MERT extraction to NPZ.

One extractor per model (``model="maest" | "mert"``), both driven by the same
*frozen* per-track windows from the dataset manifest, decoded at the model's
native sample rate (MAEST 16 kHz, MERT 24 kHz) via ``backend.decode_stderr``.
The manifest is the single source of truth for which windows are analysed: an
empty/truncated decode or a stat mismatch fails that track, it never silently
substitutes a different window.

Outputs (each written atomically via a same-directory temp file, never
overwriting an existing artifact):

* ``<out>/<model>.npz``  -- fingerprint-keyed feature arrays
* ``<out>/<model>.json`` -- full extraction metadata (also the return value)

MAEST (``discogs-maest-30s-pw-129e-519l``): ``maest_logits`` (N, 519) head
logits, ``maest`` (N, 768) final embedding = mean(cls, dist) before the head,
and ``maest_layer7`` (N, 2304) the *native* maest_infer seventh block
(``transformer_block=6``) as cat(cls, dist, patch-mean). Windows are
zero-padded to 30 s (the model's fixed input); inference runs two forward
passes per track (final + block 7). The mel front end stays on CPU and the
transformer runs under the shared ``backend.embeddings.clap.inference_lock``;
device work is synchronised before the inference timer stops.

MERT (``m-a-p/MERT-v2-30s``, pinned 40-hex ``revision``): ``mert_l19`` /
``mert_l23`` (N, 1024) = ``hidden_states[18]`` / ``[22]`` pooled with a masked
temporal mean over ``feature_attention_mask`` (feature extractor at 24 kHz,
``padding=True``, ``return_attention_mask=True``, ``do_normalize=False``).

Safety: nothing downloads and no repo-specific code runs by default. MAEST
requires the checkpoint to already be in the torch.hub cache unless
``allow_download=True``; the metadata records the checkpoint's actual streaming
SHA-256 (hashed after load, never trusted from a registry). MERT loads
repo-specific custom modeling code, so ``trust_remote_code=True`` is a separate
explicit flag; ``allow_download`` only controls fetching uncached weights
(``local_files_only = not allow_download``). Per-track failures are recorded in
``failures``/``coverage``; they never crash the batch.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

MAEST_ARCH = "discogs-maest-30s-pw-129e-519l"
#: ``configs.default_cfgs`` key in the installed maest_infer (underscore form).
MAEST_CFG_KEY = "discogs_maest_30s_pw_129e_519l"
MAEST_SR = 16000
MAEST_WINDOW_SECONDS = 30.0
MAEST_EMBED_DIM = 768
MAEST_LOGIT_DIM = 519
#: Zero-based transformer block of the one-based 7th layer.
MAEST_LAYER7_BLOCK = 6
MAEST_LAYER7_DIM = 768 * 3  # native cat(cls, dist, patch-mean), not Essentia's 768

MERT_REPO = "m-a-p/MERT-v2-30s"
MERT_SR = 24000
MERT_EMBED_DIM = 1024
#: Model card: hidden_states[0] = block 1, so L19/L23 are indices 18/22.
MERT_L19_INDEX = 18
MERT_L23_INDEX = 22

MAX_TRACK_WINDOWS = 3  # NPZ excerpt schema is (N, 3, D); dataset caps manifests at 3
MIN_DECODE_FRACTION = 0.95  # gross-truncation guard for frozen-window decode
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_MAEST_KEYMAP = {
    "logits": ("maest_logits", "maest_logits_excerpts"),
    "emb": ("maest", "maest_excerpts"),
    "layer7": ("maest_layer7", "maest_layer7_excerpts"),
}
_MERT_KEYMAP = {
    "mert_l19": ("mert_l19", "mert_l19_excerpts"),
    "mert_l23": ("mert_l23", "mert_l23_excerpts"),
}
_WINDOW_POLICY = {
    "maest": "exact frozen manifest windows decoded at 16 kHz; each zero-padded to 30 s (model fixed input)",
    "mert": "exact frozen manifest windows decoded at 24 kHz; valid lengths kept, extractor pads right, feature_attention_mask excludes padding",
}


def _progress(on_progress: Callable[[str], None] | None, msg: str) -> None:
    if on_progress is not None:
        on_progress(msg)
    else:
        print(msg, flush=True)


def _pick_device(device: str) -> str:
    if device != "auto":
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'mps'; got {device!r}")
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync_device(device: str) -> None:
    """Block until device work is done so inference timers measure real elapsed time."""
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _stat_error(track: dict) -> str | None:
    try:
        st = os.stat(track["path"])
    except OSError as e:
        return f"stat failed: {e}"
    if st.st_mtime_ns != track["mtime_ns"] or st.st_size != track["size"]:
        return (
            f"file changed since manifest (mtime_ns {st.st_mtime_ns} != {track['mtime_ns']}, "
            f"size {st.st_size} != {track['size']})"
        )
    return None


def _decode_windows(path: str, sr: int, windows: list[dict]) -> tuple[list[np.ndarray], float, str | None]:
    """Decode the frozen manifest windows at *sr*; returns (audios, seconds, error)."""
    from backend.decode_stderr import librosa_load

    audios: list[np.ndarray] = []
    t0 = time.monotonic()
    try:
        for w in windows:
            audio, _sr = librosa_load(
                path, sr=sr, mono=True, offset=float(w["start"]), duration=float(w["duration"])
            )
            a = np.asarray(audio, dtype=np.float32)
            if a.size == 0:
                return [], time.monotonic() - t0, f"empty decode for window {w['start']}+{w['duration']}s"
            expected = float(w["duration"]) * sr
            if a.shape[0] < MIN_DECODE_FRACTION * expected:
                return [], time.monotonic() - t0, (
                    f"truncated decode for window {w['start']}+{w['duration']}s: "
                    f"{a.shape[0]}/{expected:.0f} samples (<{MIN_DECODE_FRACTION:.0%} of expected)"
                )
            audios.append(a)
    except Exception as e:
        return [], time.monotonic() - t0, f"decode failed: {type(e).__name__}: {e}"
    return audios, time.monotonic() - t0, None


def _pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    n = audio.shape[0]
    if n == target:
        return audio
    if n > target:  # manifest caps duration at 30 s; only guards rounding
        return audio[:target]
    out = np.zeros(target, dtype=np.float32)
    out[:n] = audio
    return out


def _maest_checkpoint() -> tuple[str, str | None, Path]:
    """(url, registry sha256-or-None, torch.hub cache path) for the frozen checkpoint."""
    from maest_infer import configs
    from torch.hub import get_dir

    url = configs.default_cfgs[MAEST_CFG_KEY]["url"]
    cache_path = Path(get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    sha: str | None = None
    reg = Path(configs.__file__).resolve().parent / "data" / "checkpoints.json"
    try:
        for entry in json.loads(reg.read_text()).get("checkpoints", {}).values():
            if entry.get("url") == url:
                sha = entry.get("sha256")
                break
    except Exception:
        sha = None
    return url, sha, cache_path


def _load_maest(device: str, allow_download: bool) -> tuple[Any, Any, dict]:
    """Load a private MAEST instance; mel front end pinned to CPU as in production."""
    from backend.embeddings import maest as prod_maest
    from maest_infer import get_maest

    if (prod_maest.MAEST_ARCH != MAEST_ARCH or prod_maest.MAEST_SR != MAEST_SR
            or prod_maest.MAEST_LOGIT_DIM != MAEST_LOGIT_DIM):
        raise RuntimeError("genre-space MAEST constants drifted from backend.embeddings.maest")
    url, sha_expected, cache_path = _maest_checkpoint()
    if not cache_path.is_file() and not allow_download:
        raise RuntimeError(
            f"MAEST checkpoint not in the torch.hub cache ({cache_path}, from {url}). "
            "Pass allow_download=True to fetch it (torch.hub downloads to the cache)."
        )
    model = get_maest(arch=MAEST_ARCH)
    model.eval()
    model.init_melspectrogram()
    model = model.to(device)
    mel = model.melspectrogram.to("cpu")  # torchaudio STFT window must stay on CPU (MPS)
    model.melspectrogram = None
    sha_actual = _sha256_file(cache_path) if cache_path.is_file() else None
    return model, mel, {
        "arch": MAEST_ARCH,
        "url": url,
        "sha256": sha_actual,  # actual streaming hash of the cache file, verified after load
        "sha256_expected": sha_expected,  # registry value, informational only
        "cache_path": str(cache_path),
    }


def _mert_revision_cached(revision: str) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE  # single source for the HF cache dir

    cache = Path(HF_HUB_CACHE)
    return (cache / f"models--{MERT_REPO.replace('/', '--')}" / "snapshots" / revision).is_dir()


def _load_mert(device: str, revision: str | None, allow_download: bool, trust_remote_code: bool) -> tuple[Any, Any, dict]:
    if revision is None or not _REVISION_RE.fullmatch(revision):
        raise ValueError("MERT requires revision=<40 hex char commit> (a pinned commit, not a branch)")
    if not trust_remote_code:
        raise RuntimeError(
            "MERT loads repo-specific custom modeling code from m-a-p/MERT-v2-30s; "
            "pass trust_remote_code=True to acknowledge it (code was reviewed at the pinned revision)."
        )
    if not allow_download and not _mert_revision_cached(revision):
        raise RuntimeError(
            f"MERT revision {revision} is not in the local HF cache; pass allow_download=True "
            "to fetch it, or pin a cached revision."
        )
    from transformers import AutoFeatureExtractor, AutoModel

    kw = dict(revision=revision, trust_remote_code=trust_remote_code, local_files_only=not allow_download)
    extractor = AutoFeatureExtractor.from_pretrained(MERT_REPO, **kw)
    model = AutoModel.from_pretrained(MERT_REPO, **kw)
    model.eval()
    model = model.to(device)
    return model, extractor, {
        "repo": MERT_REPO,
        "revision": revision,
        "trusted_remote_code": True,
        "custom_code_acknowledged": True,
        "local_files_only": not allow_download,
    }


def mask_pool(hidden, mask) -> np.ndarray:
    """Masked temporal mean: average *hidden* frames where *mask* is 1 (padded frames excluded).

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


def _assemble(per_window: dict[str, np.ndarray], pad_to: int) -> dict | None:
    """Window arrays -> pooled + zero-padded excerpts + explicit count.

    Returns None when a pooled vector is all-zero or non-finite (invalid).
    """
    n = next(iter(per_window.values())).shape[0]
    pooled: dict[str, np.ndarray] = {}
    excerpts: dict[str, np.ndarray] = {}
    for name, arr in per_window.items():
        p = arr.mean(axis=0).astype(np.float32)
        if not np.isfinite(p).all() or np.all(p == 0.0):
            return None
        pooled[name] = p
        e = np.zeros((pad_to,) + tuple(arr.shape[1:]), dtype=np.float32)
        e[:n] = arr
        excerpts[name] = e
    return {"pooled": pooled, "excerpts": excerpts, "count": int(n)}


def _maest_track(model, mel, device, path, windows):
    audios, decode_s, err = _decode_windows(path, MAEST_SR, windows)
    if err:
        return None, err, decode_s, 0.0
    target = int(MAEST_WINDOW_SECONDS * MAEST_SR)
    batch = np.stack([_pad_or_trim(a, target) for a in audios]).astype(np.float32)
    import torch
    from backend.embeddings.clap import inference_lock

    t1 = time.monotonic()
    try:
        wave = torch.from_numpy(batch)
        mel_in = mel(wave)  # CPU mel by design (torchaudio STFT window stays on CPU)
        with inference_lock:
            with torch.inference_mode():
                mel_dev = mel_in.to(device)
                logits, emb = model(mel_dev, melspectrogram_input=True)
                _, layer7 = model(mel_dev, melspectrogram_input=True, transformer_block=MAEST_LAYER7_BLOCK)
                # Copy to CPU while still holding the accelerator lock, then sync
                # so the inference timer measures real elapsed device time.
                logits = logits.detach().to("cpu")
                emb = emb.detach().to("cpu")
                layer7 = layer7.detach().to("cpu")
            _sync_device(device)
    except Exception as e:
        return None, f"inference failed: {type(e).__name__}: {e}", decode_s, time.monotonic() - t1
    infer_s = time.monotonic() - t1
    out = {}
    for name, t, dim in (
        ("logits", logits, MAEST_LOGIT_DIM),
        ("emb", emb, MAEST_EMBED_DIM),
        ("layer7", layer7, MAEST_LAYER7_DIM),
    ):
        arr = t.numpy().astype(np.float32)
        if arr.shape != (len(audios), dim) or not np.isfinite(arr).all():
            return None, f"invalid {name} output shape {arr.shape}", decode_s, infer_s
        out[name] = arr
    return out, None, decode_s, infer_s


def _mert_track(model, extractor, device, path, windows):
    audios, decode_s, err = _decode_windows(path, MERT_SR, windows)
    if err:
        return None, err, decode_s, 0.0
    import torch
    from backend.embeddings.clap import inference_lock

    t1 = time.monotonic()
    try:
        feats = extractor(
            list(audios),
            sampling_rate=MERT_SR,
            return_attention_mask=True,
            padding=True,
            do_normalize=False,
            return_tensors="pt",
        )
        input_values = feats["input_values"].to(device)
        attention_mask = feats["attention_mask"].to(device) if feats.get("attention_mask") is not None else None
        with inference_lock:
            with torch.inference_mode():
                out = model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
                if len(out.hidden_states) <= MERT_L23_INDEX:
                    raise ValueError(
                        f"model returned {len(out.hidden_states)} hidden states; need more than {MERT_L23_INDEX}"
                    )
                mask = out.feature_attention_mask
                if mask is None:
                    raise ValueError("model output missing feature_attention_mask")
                # CPU copies under the accelerator lock; sync before the timer stops.
                l19 = out.hidden_states[MERT_L19_INDEX].detach().to("cpu")
                l23 = out.hidden_states[MERT_L23_INDEX].detach().to("cpu")
                mask = mask.detach().to("cpu")
            _sync_device(device)
    except Exception as e:
        return None, f"inference failed: {type(e).__name__}: {e}", decode_s, time.monotonic() - t1
    infer_s = time.monotonic() - t1
    result = {}
    for label, hs in (("mert_l19", l19), ("mert_l23", l23)):
        if hs.ndim != 3 or hs.shape[0] != len(audios) or hs.shape[-1] != MERT_EMBED_DIM:
            return None, f"invalid {label} shape {tuple(hs.shape)}", decode_s, infer_s
        result[label] = np.stack([mask_pool(hs[i], mask[i]) for i in range(len(audios))], axis=0).astype(np.float32)
    return result, None, decode_s, infer_s


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(target: Path, write_fn) -> None:
    tmp: str | None = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        os.close(fd)
        with open(tmp, "wb") as f:
            write_fn(f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _atomic_write_npz(target: Path, arrays: dict) -> str:
    _atomic_write(target, lambda f: np.savez_compressed(f, **arrays))
    return _sha256_file(target)


def _peak_rss() -> dict:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    raw = int(ru.ru_maxrss)
    if sys.platform == "darwin":
        return {"peak_rss_bytes": raw, "ru_maxrss_raw": raw, "unit": "bytes", "note": "macOS ru_maxrss is in bytes"}
    return {"peak_rss_bytes": raw * 1024, "ru_maxrss_raw": raw, "unit": "bytes", "note": "Linux ru_maxrss is in KiB"}


def _device_memory(device: str) -> dict | None:
    if device == "cpu":
        return None
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return {
            "kind": "cuda_peak_allocated_bytes",
            "bytes": int(torch.cuda.max_memory_allocated(device)),
            "semantics": "peak allocator usage since start, sampled at end (torch.cuda.max_memory_allocated)",
        }
    if device == "mps" and torch.backends.mps.is_available():
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "semantics": "mps.current_allocated_memory = allocator current usage (not peak); driver_allocated_memory = driver-reserved incl. cached; sampled at end",
        }
    return None


def _software_versions() -> dict:
    import importlib.metadata

    out = {"python": platform.python_version(), "platform": sys.platform}
    for name in ("numpy", "torch", "torchaudio", "transformers", "maest-infer", "librosa", "soundfile"):
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = "unavailable"
    return out


def extract(
    manifest: dict,
    output: Path,
    *,
    model: str = "maest",
    revision: str | None = None,
    device: str = "auto",
    allow_download: bool = False,
    trust_remote_code: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Extract frozen-window features for every manifest track into *output*.

    Validates the manifest with :func:`dataset.validate_manifest` and rejects a
    ``dataset_id`` that does not match the canonical hash — both before anything
    is loaded or created. Returns the metadata dict (also written to
    ``<output>/<model>.json``); feature arrays go to ``<output>/<model>.npz``.
    Existing artifacts are refused, never overwritten. Per-track
    decode/inference/stat failures are recorded in ``failures``/``coverage``.
    """
    wall0 = time.monotonic()
    model = str(model).lower().strip()
    if model not in ("maest", "mert"):
        raise ValueError("model must be 'maest' or 'mert'")
    from . import dataset

    problems = dataset.validate_manifest(manifest)
    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))
    dataset_id = dataset.dataset_id(manifest)
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError(
            "manifest dataset_id does not match the canonical manifest ID hash "
            "(the manifest was edited or is stale)"
        )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{model}.npz"
    json_path = out / f"{model}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing artifact(s): {npz_path}, {json_path}. "
            "Use a fresh output directory (run dirs)."
        )
    dev = _pick_device(device)

    if model == "maest":
        m, mel, ckpt = _load_maest(dev, allow_download)
        keymap = _MAEST_KEYMAP
        track_fn = lambda t: _maest_track(m, mel, dev, t["path"], t["windows"])  # noqa: E731
        layers = {
            "maest_logits": "final head logits (519 Discogs genre/style classes)",
            "maest": "final transformer embedding: mean(cls, dist) before the head (768)",
            "maest_layer7": (
                "native maest_infer seventh block (one-based; transformer_block=6 zero-based) as "
                "cat(cls, dist, patch-mean) -> 2304. NOT the Essentia "
                "discogs-maest-30s-pw-519l-2.json layer7 (per-token [1, n_tokens, 768])"
            ),
        }
        notes = ["two forward passes per track (final + block 7)", "windows zero-padded to 30 s (model fixed input)"]
    else:
        m, ext, ckpt = _load_mert(dev, revision, allow_download, trust_remote_code)
        keymap = _MERT_KEYMAP
        track_fn = lambda t: _mert_track(m, ext, dev, t["path"], t["windows"])  # noqa: E731
        layers = {
            "mert_l19": "hidden_states[18] = block 19 (one-based; card: hidden_states[0] = block 1) -> 1024",
            "mert_l23": "hidden_states[22] = block 23 (one-based) -> 1024",
        }
        notes = [
            "windows keep valid lengths; extractor pads right (padding=True, return_attention_mask=True, "
            "do_normalize=False); feature_attention_mask excludes padded frames",
            "MERT MelFrontend uses torchaudio STFT buffers; MPS may need device='cpu' (device errors are reported, never retried)",
        ]

    arrays: dict[str, list[np.ndarray]] = {key: [] for pair in keymap.values() for key in pair}
    arrays["excerpt_counts"] = []
    ok_fps: list[str] = []
    failures: list[dict] = []
    timing: dict[str, dict[str, float]] = {}
    n = len(manifest["tracks"])
    for i, track in enumerate(manifest["tracks"]):
        fp = track["fp"]
        _progress(on_progress, f"[{i + 1}/{n}] {fp}")
        err = _stat_error(track)
        stage = "stat"
        per_window = None
        decode_s = infer_s = 0.0
        if err is None:
            try:
                per_window, err, decode_s, infer_s = track_fn(track)
            except Exception as e:
                per_window, err, decode_s, infer_s = None, f"{type(e).__name__}: {e}", 0.0, 0.0
            if err is not None:
                stage = "decode" if ("decode" in err or "window" in err) else "inference"
        if err is not None:
            failures.append({"fp": fp, "path": track["path"], "stage": stage, "error": err})
            _progress(on_progress, f"[{i + 1}/{n}] {fp} FAILED ({stage}): {err}")
            continue
        assert per_window is not None
        assembled = _assemble(per_window, MAX_TRACK_WINDOWS)
        if assembled is None:
            failures.append({
                "fp": fp, "path": track["path"], "stage": "pool",
                "error": "pooled vector is all-zero or non-finite",
            })
            _progress(on_progress, f"[{i + 1}/{n}] {fp} FAILED (pool): zero pooled vector")
            continue
        ok_fps.append(fp)
        timing[fp] = {"decode_s": round(decode_s, 4), "inference_s": round(infer_s, 4)}
        for name, (pool_key, exc_key) in keymap.items():
            arrays[pool_key].append(assembled["pooled"][name])
            arrays[exc_key].append(assembled["excerpts"][name])
        arrays["excerpt_counts"].append(assembled["count"])

    if not ok_fps:
        raise RuntimeError(f"no tracks extracted successfully ({len(failures)} failed): {failures[:2]}")

    npz_arrays: dict[str, np.ndarray] = {"fingerprints": np.array(ok_fps, dtype=f"<U{max(len(f) for f in ok_fps)}")}
    for key, lst in arrays.items():
        if key == "excerpt_counts":
            npz_arrays[key] = np.array(lst, dtype=np.int64)
        else:
            npz_arrays[key] = np.stack(lst, axis=0).astype(np.float32)

    meta = {
        "dataset_id": dataset_id,
        "manifest_dataset_id": manifest.get("dataset_id"),
        "model": model,
        "sample_rate_hz": MAEST_SR if model == "maest" else MERT_SR,
        "window_policy": _WINDOW_POLICY[model],
        "checkpoint": ckpt,
        "layers": layers,
        "pooling": {
            "method": "mean over valid windows; MERT layers also masked temporal mean over feature_attention_mask",
            "excerpt_slots": MAX_TRACK_WINDOWS,
            "missing_slots": "zero-padded in excerpt arrays; excerpt_counts records the real count",
            "invalid": "all-zero/non-finite pooled vector -> track failure",
        },
        "dtype": "float32",
        "software": _software_versions(),
        "device": dev,
        "device_memory": _device_memory(dev),
        "wall_seconds": round(time.monotonic() - wall0, 4),
        "per_track_timing": timing,
        "coverage": {"total": n, "ok": len(ok_fps), "failed": len(failures)},
        "failures": failures,
        "rss": _peak_rss(),
        "notes": notes,
    }
    json.dumps(meta, allow_nan=False)  # validate JSON-safety before any artifact is written
    artifact_sha = _atomic_write_npz(npz_path, npz_arrays)
    meta["artifact"] = {"npz": npz_path.name, "sha256": artifact_sha, "size_bytes": int(npz_path.stat().st_size)}
    _atomic_write(json_path, lambda f: f.write((json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")))
    _progress(on_progress, f"wrote {npz_path.name}: {len(ok_fps)}/{n} tracks")
    return meta
