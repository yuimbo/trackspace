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


def test_profile_projection_cpu_vs_gpu(monkeypatch):
    """Compare CPU (sklearn) vs cuML or mlx-vis t-SNE when available."""
    from backend.embeddings.projection_backends import (
        _project_tsne,
        reset_projection_backend_cache,
    )

    n_env = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_N", "320"))
    d_env = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_D", "128"))
    n_rows = max(80, min(n_env, 10_000))
    n_cols = max(8, min(d_env, 2048))
    X = _projection_matrix(n_rows, n_cols)

    runs = int(os.environ.get("TRACKSPACE_PROFILE_PROJECTION_RUNS", "3"))

    def run_cpu():
        monkeypatch.setenv("TRACKSPACE_CUML", "0")
        monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
        reset_projection_backend_cache()
        return _project_tsne(X.copy())

    out_cpu, cpu_samples = _timed_runs(run_cpu, runs=runs, warmup=True)
    _print_stats("tsne_cpu", cpu_samples)
    assert out_cpu.shape == (n_rows, 2)
    assert np.isfinite(out_cpu).all()

    if _cuda_cuml_available():
        accel = "cuml"

        def run_gpu():
            monkeypatch.setenv("TRACKSPACE_CUML", "1")
            monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
            reset_projection_backend_cache()
            return _project_tsne(X.copy())

        out_gpu, gpu_samples = _timed_runs(run_gpu, runs=runs, warmup=True)
        _print_stats("tsne_cuml", gpu_samples)
    elif _mlx_vis_available():
        accel = "mlx_vis"

        def run_gpu():
            monkeypatch.setenv("TRACKSPACE_CUML", "0")
            monkeypatch.setenv("TRACKSPACE_MLX_VIS", "1")
            reset_projection_backend_cache()
            return _project_tsne(X.copy())

        out_gpu, gpu_samples = _timed_runs(run_gpu, runs=runs, warmup=True)
        _print_stats("tsne_mlx_vis", gpu_samples)
    else:
        print("tsne_gpu: skipped (no CUDA+cuML and no mlx-vis on this machine)")
        return

    assert out_gpu.shape == (n_rows, 2)
    assert np.isfinite(out_gpu).all()
    gpu_mean = mean(gpu_samples)
    cpu_mean = mean(cpu_samples)
    if gpu_mean > 0:
        speedup = cpu_mean / gpu_mean
        print(
            f"tsne_speedup_{accel:<8} "
            f"x{speedup:0.2f}  (CPU {cpu_mean*1000:.1f} ms vs "
            f"GPU {gpu_mean*1000:.1f} ms)"
        )


def test_profile_projection_cpu_only_quick(monkeypatch):
    """Small CPU t-SNE timing — runs everywhere."""
    from backend.embeddings.projection_backends import (
        _project_tsne,
        reset_projection_backend_cache,
    )

    monkeypatch.setenv("TRACKSPACE_CUML", "0")
    monkeypatch.setenv("TRACKSPACE_MLX_VIS", "0")
    reset_projection_backend_cache()

    X = _projection_matrix(48, 16)
    out, samples = _timed_runs(lambda: _project_tsne(X.copy()), runs=2)
    _print_stats("tsne_cpu_quick", samples)

    assert out.shape == (48, 2)
    assert np.isfinite(out).all()


def test_profile_incremental_tsne(monkeypatch):
    """openTSNE cold fit vs transforming newly added tracks.

    Reports the two numbers the incremental path is about: the one-off cost of
    fitting an embedding, and the cost of placing ``ADDED`` new tracks into it.
    Run with ``-s`` to see them; raise ``N`` toward the real library size
    (~9.5k) for a representative measurement::

      TRACKSPACE_PROFILE_OPENTSNE_N=9500 TRACKSPACE_PROFILE_OPENTSNE_D=50 \\
      TRACKSPACE_PROFILE_OPENTSNE_ADDED=500 \\
      .venv/bin/python -m pytest tests/test_profiling_embeddings.py -v -s \\
        -k incremental_tsne
    """
    pytest.importorskip("openTSNE")
    from backend.embeddings.projection_backends import (
        _project_tsne,
        reset_projection_backend_cache,
        reset_tsne_fit_cache,
        tsne_fit_stats,
    )

    n = int(os.environ.get("TRACKSPACE_PROFILE_OPENTSNE_N", "2000"))
    d = int(os.environ.get("TRACKSPACE_PROFILE_OPENTSNE_D", "50"))
    added = int(os.environ.get("TRACKSPACE_PROFILE_OPENTSNE_ADDED", "200"))

    monkeypatch.setenv("TRACKSPACE_OPENTSNE", "1")
    reset_projection_backend_cache()
    reset_tsne_fit_cache()

    X = _projection_matrix(n, d)
    paths = [f"t{i}.mp3" for i in range(n)]
    _, fit_samples = _timed_runs(
        lambda: _project_tsne(X, paths=paths, config_key="bench"), runs=1,
    )
    _print_stats("tsne_opentsne_fit", fit_samples)

    X2 = _projection_matrix(n + added, d, seed=7)
    paths2 = paths + [f"n{i}.mp3" for i in range(added)]
    out, transform_samples = _timed_runs(
        lambda: _project_tsne(X2, paths=paths2, config_key="bench"), runs=1,
    )
    _print_stats(f"tsne_opentsne_+{added}", transform_samples)

    assert out.shape == (n + added, 2)
    stats = tsne_fit_stats()
    assert stats["fit"] == 1
    assert stats["transform"] == 1


def test_profile_layout_fidelity():
    """Neighbourhood recall of the incremental placement vs a full fit.

    Reports the fidelity cost of the incremental path on synthetic data: the base
    layout's own recall@k, the placed points' recall, and (for reference) a
    subsample-and-interpolate baseline. Run with ``-s`` and raise ``N`` toward the
    real library for a representative measurement::

      TRACKSPACE_PROFILE_FIDELITY_N=8000 \\
      .venv/bin/python -m pytest tests/test_profiling_embeddings.py -v -s \\
        -k layout_fidelity
    """
    pytest.importorskip("openTSNE")
    from backend.embeddings.layout_fidelity import neighbourhood_recall
    from backend.embeddings.projection_backends import (
        _opentsne_pca_dim,
        _pca_reduce,
        _tsne_perplexity,
    )

    n = int(os.environ.get("TRACKSPACE_PROFILE_FIDELITY_N", "2000"))
    d = int(os.environ.get("TRACKSPACE_PROFILE_FIDELITY_D", "50"))
    k = int(os.environ.get("TRACKSPACE_PROFILE_FIDELITY_K", "10"))

    from sklearn.manifold import TSNE as SklearnTSNE
    from openTSNE import TSNE as OpenTSNE

    rng = np.random.default_rng(0)
    hi = rng.standard_normal((n, d)).astype(np.float32)

    base = np.asarray(
        SklearnTSNE(
            n_components=2,
            perplexity=_tsne_perplexity(n),
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(hi),
        dtype=np.float32,
    )
    print(
        f"recall@{k}_full_fit          "
        f"{neighbourhood_recall(hi, base, k=k, sample=2000):.3f}"
    )

    # Fit on 95% and place the remaining 5% with openTSNE's .transform — the
    # incremental path the map takes when a few tracks are added.
    hold = rng.choice(n, max(1, n // 20), replace=False)
    mask = np.ones(n, bool)
    mask[hold] = False
    reduced, pca = _pca_reduce(hi[mask], _opentsne_pca_dim(d))
    emb = OpenTSNE(
        n_components=2,
        perplexity=_tsne_perplexity(int(mask.sum())),
        random_state=42,
        n_jobs=-1,
        verbose=False,
    ).fit(reduced)
    placed = np.empty((n, 2), dtype=np.float32)
    placed[mask] = np.asarray(emb, dtype=np.float32)
    placed[hold] = np.asarray(
        emb.transform(pca.transform(hi[hold])), dtype=np.float32
    )
    print(
        f"recall@{k}_placed            "
        f"{neighbourhood_recall(hi, placed, k=k, sample=2000):.3f}"
    )

    assert placed.shape == (n, 2)
    assert np.isfinite(placed).all()
