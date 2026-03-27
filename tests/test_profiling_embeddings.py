"""Lightweight profiling tests for embedding/feature extraction methods.

Run:
  .venv/bin/python -m pytest tests/test_profiling_embeddings.py -v -s

Projection CPU vs GPU (cuML / mlx-vis)::

  TRACKSPACE_PROFILE_PROJECTION_N=800 \\
  TRACKSPACE_PROFILE_PROJECTION_D=128 \\
  TRACKSPACE_PROFILE_PROJECTION_RUNS=3 \\
  .venv/bin/python -m pytest tests/test_profiling_embeddings.py -v -s -k projection

These tests print elapsed timing metrics so methods can be compared quickly.
They also keep simple assertions so regressions fail fast.
"""

from __future__ import annotations

import os
import time
from statistics import mean

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_TRACK = os.path.join(FIXTURE_DIR, "test_track.mp3")
TEST_TRACK_HARMONIC = os.path.join(FIXTURE_DIR, "test_track_2.mp3")


def _timed_runs(fn, runs: int = 3, warmup: bool = False):
    if warmup:
        fn()
    samples: list[float] = []
    last_result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_result = fn()
        samples.append(time.perf_counter() - t0)
    return last_result, samples


def _print_stats(label: str, samples: list[float]) -> None:
    ms = [s * 1000.0 for s in samples]
    print(
        f"{label:<24} "
        f"mean={mean(ms):8.1f} ms  "
        f"min={min(ms):8.1f} ms  "
        f"max={max(ms):8.1f} ms  "
        f"runs={len(ms)}"
    )


@pytest.fixture(scope="module")
def effnet_loaded() -> bool:
    from backend.embeddings.effnet import is_effnet_ready, load_effnet

    load_effnet()
    return is_effnet_ready()


@pytest.fixture(scope="module")
def clap_loaded() -> bool:
    from backend.embeddings import is_model_ready, load_model

    try:
        load_model()
    except Exception:
        return False
    return is_model_ready()


def test_profile_audio_features():
    from backend.embeddings.librosa_audio_features import extract_audio_features

    result, samples = _timed_runs(lambda: extract_audio_features(TEST_TRACK), runs=5)
    _print_stats("audio_features", samples)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (6,)
    assert np.isfinite(result).all()


def test_profile_audio_features_batch_vs_sequential():
    from backend.embeddings.librosa_audio_features import (
        extract_audio_features,
        generate_audio_features_batch,
    )

    # Repeat fixtures to get a meaningful batch size for timing.
    paths = [TEST_TRACK, TEST_TRACK_HARMONIC] * 4

    def _run_seq():
        return {p: extract_audio_features(p) for p in paths}

    seq_result, seq_samples = _timed_runs(_run_seq, runs=3)
    _print_stats("audio_feat_seq_8", seq_samples)

    bat_result, bat_samples = _timed_runs(
        lambda: generate_audio_features_batch(paths),
        runs=3,
    )
    _print_stats("audio_feat_batch_8", bat_samples)

    seq_mean = mean(seq_samples)
    bat_mean = mean(bat_samples)
    speedup = (seq_mean / bat_mean) if bat_mean > 0 else 0.0
    print(f"{'audio_feat_batch_speedup':<24} x{speedup:0.2f}")

    assert isinstance(seq_result, dict)
    assert isinstance(bat_result, dict)
    assert set(bat_result.keys()) == set(paths)
    for val in bat_result.values():
        assert val is not None
        assert isinstance(val, np.ndarray)
        assert val.shape == (6,)
        assert np.isfinite(val).all()


def test_profile_effnet_single(effnet_loaded):
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.embeddings.effnet import generate_effnet_embedding

    result, samples = _timed_runs(
        lambda: generate_effnet_embedding(TEST_TRACK),
        runs=3,
    )
    _print_stats("effnet_single", samples)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape[0] > 0
    assert np.isfinite(result).all()


def test_profile_effnet_batch(effnet_loaded):
    if not effnet_loaded:
        pytest.skip("onnxruntime or EffNet model not available")

    from backend.embeddings.effnet import generate_effnet_embeddings_batch

    paths = [TEST_TRACK, TEST_TRACK_HARMONIC]
    result, samples = _timed_runs(
        lambda: generate_effnet_embeddings_batch(paths),
        runs=3,
    )
    _print_stats("effnet_batch_2tracks", samples)
    per_track_ms = (mean(samples) * 1000.0) / len(paths)
    print(f"{'effnet_batch_per_track':<24} mean={per_track_ms:8.1f} ms")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(paths)
    for emb in result.values():
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] > 0
        assert np.isfinite(emb).all()


def test_profile_clap_single(clap_loaded):
    if not clap_loaded:
        pytest.skip("CLAP model unavailable (check HF_TOKEN/network)")

    from backend.embeddings import generate_embedding

    result, samples = _timed_runs(
        lambda: generate_embedding(TEST_TRACK),
        runs=3,
    )
    _print_stats("clap_single", samples)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape[0] > 0
    assert np.isfinite(result).all()


def _cuda_cuml_available() -> bool:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        if not cp.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        import cuml  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True


def _mlx_vis_available() -> bool:
    import importlib.util
    import platform
    import sys

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_vis") is not None
    )


def _projection_matrix(n_rows: int, n_cols: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (x / norms).astype(np.float32)


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_profile_projection_cpu_vs_gpu(method: str, monkeypatch):
    """Compare CPU (umap-learn / sklearn) vs cuML or mlx-vis when available."""
    from backend.embeddings.layout import (
        _project_tsne,
        _project_umap,
        reset_projection_backend_cache,
    )

    n_env = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_N", "320"))
    d_env = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_D", "128"))
    n_rows = max(80, min(n_env, 10_000))
    n_cols = max(8, min(d_env, 2048))
    X = _projection_matrix(n_rows, n_cols)
    proj = _project_umap if method == "umap" else _project_tsne

    runs = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_RUNS", "3"))

    def run_cpu():
        monkeypatch.setenv("TRACKSPACE_CUML", "0")
        monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
        reset_projection_backend_cache()
        return proj(X.copy())

    # Warm CPU path too — first umap-learn call can trigger heavy Numba compilation.
    out_cpu, cpu_samples = _timed_runs(run_cpu, runs=runs, warmup=True)
    _print_stats(f"{method}_cpu", cpu_samples)
    assert out_cpu.shape == (n_rows, 2)
    assert np.isfinite(out_cpu).all()

    if _cuda_cuml_available():
        accel = "cuml"

        def run_gpu():
            monkeypatch.setenv("TRACKSPACE_CUML", "1")
            monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
            reset_projection_backend_cache()
            return proj(X.copy())

        out_gpu, gpu_samples = _timed_runs(run_gpu, runs=runs, warmup=True)
        _print_stats(f"{method}_cuml", gpu_samples)
    elif _mlx_vis_available():
        accel = "mlx_vis"

        def run_gpu():
            monkeypatch.setenv("TRACKSPACE_CUML", "0")
            monkeypatch.setenv("TRACKSPACE_MLX_VIS", "1")
            reset_projection_backend_cache()
            return proj(X.copy())

        out_gpu, gpu_samples = _timed_runs(run_gpu, runs=runs, warmup=True)
        _print_stats(f"{method}_mlx_vis", gpu_samples)
    else:
        print(
            f"{method}_gpu: skipped (no CUDA+cuML and no mlx-vis on this machine)"
        )
        return

    assert out_gpu.shape == (n_rows, 2)
    assert np.isfinite(out_gpu).all()
    gpu_mean = mean(gpu_samples)
    cpu_mean = mean(cpu_samples)
    if gpu_mean > 0:
        speedup = cpu_mean / gpu_mean
        print(
            f"{method}_speedup_{accel:<8} "
            f"x{speedup:0.2f}  (CPU {cpu_mean*1000:.1f} ms vs "
            f"GPU {gpu_mean*1000:.1f} ms)"
        )


def test_profile_projection_cpu_only_quick(monkeypatch):
    """Small CPU UMAP timing — runs everywhere."""
    from backend.embeddings.layout import _project_umap, reset_projection_backend_cache

    monkeypatch.setenv("TRACKSPACE_CUML", "0")
    monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
    reset_projection_backend_cache()

    X = _projection_matrix(48, 16)
    out, samples = _timed_runs(lambda: _project_umap(X.copy()), runs=2)
    _print_stats("umap_cpu_quick", samples)

    assert out.shape == (48, 2)
    assert np.isfinite(out).all()
