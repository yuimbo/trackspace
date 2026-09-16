"""Measure the ``features`` stage throughput as a function of worker count.

The stage is CPU-bound (librosa classical 6-D + rhythm 14-D, plus madmom tempo) and
independent per track, so it should scale with workers. It does **not** scale with
*threads*: madmom's BLSTM forward pass is GIL-bound, so more features threads are
slower (measured 0.8 s → 10.5 s for one track at 1 → 4 threads). The handler
therefore runs each track in a **process** pool, and this script measures that: it
drives the real handler over a deterministic sample of the real library with N pool
processes and reports wall-clock / tracks-per-minute for each N.

Non-destructive: fingerprints are read from the real track cache, but feature rows and
queue state live in a scratch temp dir, so ``data/audio_features.db`` is never
touched. It also asserts every sampled track ended up cached — a write silently dropped
by SQLite contention would show up as a missing row.

Run from the repo root::

    .venv/bin/python tests/profile_features_workers.py --root /path/to/music
    .venv/bin/python tests/profile_features_workers.py --root /path/to/music \
        --workers 1,2,4,8 --sample 48
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import audio_features  # noqa: E402
from backend.cache import TrackCache  # noqa: E402
from backend.embeddings import rhythm_features as rhythm_mod  # noqa: E402
from backend.embeddings.feature_cache import FeatureCache  # noqa: E402
from backend.embeddings.librosa_audio_features import FEATURES_VERSION  # noqa: E402
from backend.embeddings.rhythm_features import RHYTHM_VERSION  # noqa: E402
from backend.fingerprint import compute_fingerprint  # noqa: E402
from backend.jobs import KIND_FEATURES, JobQueue, JobWorker  # noqa: E402
from backend.services.analysis_pipeline import (  # noqa: E402
    AnalysisDeps,
    make_features_handler,
)
from backend.tags import read_all  # noqa: E402


def _pick_sample(root: str, count: int) -> list[str]:
    """Evenly spaced sample of mp3s across the sorted library (one per folder bias)."""
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.lower().endswith(".mp3"):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    if len(paths) <= count:
        return paths
    step = len(paths) / float(count)
    return [paths[int(i * step)] for i in range(count)]


def _cached_read_all(track_cache: TrackCache, path: str) -> dict:
    """app.py's ``_cached_read_all``, without the Flask state closure."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return read_all(path)
    cached = track_cache.get(path, mtime)
    if cached is not None and cached.get("fingerprint"):
        return cached
    data = cached if cached is not None else read_all(path)
    if not data.get("fingerprint"):
        data["fingerprint"] = compute_fingerprint(path)
    if data.get("fingerprint") or cached is None:
        track_cache.put(path, mtime, data)
    return data


def _measure(root: str, sample_paths: list[str], workers: int, data_dir: str) -> dict:
    """Run the features stage over *sample_paths* with *workers* and time the drain."""
    track_cache = TrackCache(os.path.join(data_dir, "track_cache.db"))
    with tempfile.TemporaryDirectory(prefix="trackspace-features-profile-") as scratch:
        feature_cache = FeatureCache(os.path.join(scratch, "features.db"))
        queue = JobQueue(os.path.join(scratch, "jobs.db"), max_attempts=1)

        deps = AnalysisDeps(
            feature_cache=feature_cache,
            track_cache=track_cache,
            cached_read_all=lambda p: _cached_read_all(track_cache, p),
            resolve_virtual_path=lambda p: p,
            virtual_from_abs=lambda p: p,
            list_mp3s=lambda folder, recursive: list(sample_paths),
            maest_mod=None,
            rhythm_mod=rhythm_mod,
            audio_features_mod=audio_features,
            maest_version=0,
            rhythm_version=RHYTHM_VERSION,
            features_version=FEATURES_VERSION,
        )
        # The pool size is the measured variable: the handler runs each track in
        # its own process and the single worker only leases and writes.
        handler = make_features_handler(deps, pool_size=workers)

        queue.enqueue_many(
            KIND_FEATURES, [(p, {"path": p}) for p in sample_paths]
        )

        worker = JobWorker(
            queue, KIND_FEATURES, handler, batch_size=max(2, workers), idle_sleep=0.02
        )
        started = time.perf_counter()
        worker.start()
        try:
            while queue.stats(KIND_FEATURES).outstanding > 0:
                time.sleep(0.05)
        finally:
            worker.stop()
            if worker._thread is not None:
                worker._thread.join(timeout=30.0)
        elapsed = time.perf_counter() - started

        stats = queue.stats(KIND_FEATURES)
        cached_features = feature_cache.count("features", version=FEATURES_VERSION)
        cached_rhythm = feature_cache.count("rhythm", version=RHYTHM_VERSION)

    return {
        "workers": workers,
        "tracks": len(sample_paths),
        "seconds": elapsed,
        "tracks_per_min": (len(sample_paths) / elapsed * 60.0) if elapsed > 0 else 0.0,
        "done": stats.done,
        "failed": stats.failed,
        "cached_features": cached_features,
        "cached_rhythm": cached_rhythm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Library root to sample from")
    parser.add_argument(
        "--workers", default="1,2,4,8", help="Comma-separated pool process counts"
    )
    parser.add_argument("--sample", type=int, default=32, help="Tracks per run")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Real data dir holding track_cache.db (default: repo data/)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    data_dir = args.data_dir or os.environ.get("TRACKSPACE_DATA_DIR") or str(ROOT / "data")
    counts = [int(c) for c in str(args.workers).split(",") if c.strip()]
    sample = _pick_sample(root, args.sample)
    if not sample:
        print(f"no mp3s under {root}", file=sys.stderr)
        return 2

    print(f"root={root}")
    print(f"sample={len(sample)} tracks  data_dir={data_dir}")
    print("warming feature pipeline (main thread)…")
    audio_features.warmup_audio_features()
    rhythm_mod.warmup_rhythm_features()

    print(f"{'procs':>8} {'seconds':>10} {'tracks/min':>12} "
          f"{'done':>6} {'failed':>7} {'feat':>6} {'rhythm':>7}")
    for n in counts:
        r = _measure(root, sample, n, data_dir)
        print(
            f"{r['workers']:>8} {r['seconds']:>10.1f} {r['tracks_per_min']:>12.1f} "
            f"{r['done']:>6} {r['failed']:>7} {r['cached_features']:>6} "
            f"{r['cached_rhythm']:>7}"
        )
        if r["done"] + r["failed"] != len(sample):
            print("  WARNING: not every job reached a terminal state")
        if r["cached_features"] < r["done"] or r["cached_rhythm"] < r["done"]:
            print("  WARNING: cache rows missing — a write may have been dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
