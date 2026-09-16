"""Per-stage throughput harness for the whole analysis pipeline.

Reports **tracks/min** and **s/track** for each stage (scan / features /
structure / maest / clap / supervise / cluster) over a deterministic sample of the
real library, so every change in the *Analysis performance* section of `TODO.md` is
comparable. It drives the **real** handlers through the durable job queue, exactly as
production does — not a re-implementation of the stages.

Stages run in pipeline order and share one feature cache per repeat, so the
whole-library stages (`supervise`, `cluster`) see the per-track rows produced earlier
in the same repeat. A repeat rebuilds both scratch caches, which forces every stage to
redo its work; the `features` process pool and the MAEST/CLAP models are built once and
survive across repeats, so repeat 2+ is steady-state. Both every run and the best run
per stage are reported.

**Run this on an idle machine.** The stages are CPU/GPU-bound, so a concurrent build,
video call, or second analysis inflates the numbers and makes the run-to-run spread
larger than the effect you are trying to measure (observed: one stage swinging 26–52 s
under load). Use the best run, not the first, when comparing changes.

Non-destructive: every fingerprint, feature row, and queue state lives in scratch temp
dirs, so ``data/`` is never written. The MAEST/CLAP models are read from the usual
caches but never re-written.

Run from the repo root::

    .venv/bin/python tests/profile_pipeline.py --root /path/to/music
    .venv/bin/python tests/profile_pipeline.py --root /path/to/music \\
        --sample 48 --repeat 2 --stages scan,features,structure,maest,clap
    .venv/bin/python tests/profile_pipeline.py --root /path/to/music \\
        --sample 96 --stages all --repeat 1

Sweep one stage's parameter by running it alone, e.g. the MAEST batch size::

    for b in 1 2 4 8 16; do
      .venv/bin/python tests/profile_pipeline.py --root /path/to/music \\
        --sample 48 --stages maest --maest-batch $b --repeat 3
    done
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import audio_features  # noqa: E402
from backend.cache import TrackCache  # noqa: E402
from backend.embeddings import clap as clap_mod  # noqa: E402
from backend.embeddings import maest as maest_mod  # noqa: E402
from backend.embeddings import rhythm_features as rhythm_mod  # noqa: E402
from backend.embeddings.feature_cache import FeatureCache  # noqa: E402
from backend.embeddings.librosa_audio_features import FEATURES_VERSION  # noqa: E402
from backend.embeddings.maest import MAEST_VERSION  # noqa: E402
from backend.embeddings.rhythm_features import RHYTHM_VERSION  # noqa: E402
from backend.embeddings.structure import STRUCTURE_VERSION  # noqa: E402
from backend.fingerprint import compute_fingerprint  # noqa: E402
from backend.jobs import (  # noqa: E402
    KIND_CLAP,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_SCAN,
    KIND_STRUCTURE,
    KIND_SUPERVISED,
    JobQueue,
    JobWorker,
)
from backend.services.analysis_pipeline import (  # noqa: E402
    AnalysisDeps,
    ClusterStore,
    make_clap_handler,
    make_cluster_handler,
    make_features_handler,
    make_maest_handler,
    make_scan_handler,
    make_structure_handler,
    make_supervised_handler,
)
from backend.tags import read_all  # noqa: E402

#: Per-track stages, in pipeline order.
PER_TRACK_STAGES: tuple[str, ...] = (
    "scan",
    "features",
    "structure",
    "maest",
    "clap",
)
#: Whole-library singleton stages (need the sample's MAEST rows from earlier stages).
WHOLE_LIBRARY_STAGES: tuple[str, ...] = ("supervise", "cluster")
ALL_STAGES: tuple[str, ...] = PER_TRACK_STAGES + WHOLE_LIBRARY_STAGES

_STAGE_KIND: dict[str, str] = {
    "scan": KIND_SCAN,
    "features": KIND_FEATURES,
    "structure": KIND_STRUCTURE,
    "maest": KIND_MAEST,
    "clap": KIND_CLAP,
    "supervise": KIND_SUPERVISED,
    "cluster": KIND_CLUSTER,
}


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


def _make_deps(scratch: str) -> AnalysisDeps:
    """AnalysisDeps over scratch caches; the caches are swapped per repeat."""
    deps = AnalysisDeps(
        feature_cache=None,
        track_cache=None,
        cached_read_all=lambda p: _cached_read_all(deps.track_cache, p),
        resolve_virtual_path=lambda p: p,
        virtual_from_abs=lambda p: p,
        list_mp3s=lambda folder, recursive: [],
        maest_mod=maest_mod,
        rhythm_mod=rhythm_mod,
        audio_features_mod=audio_features,
        maest_version=MAEST_VERSION,
        rhythm_version=RHYTHM_VERSION,
        features_version=FEATURES_VERSION,
        clap_mod=clap_mod,
        clap_version=clap_mod.EMBEDDING_VERSION,
        structure_version=STRUCTURE_VERSION,
        clap_enabled=True,
        data_dir=os.path.join(scratch, "head"),
    )
    os.makedirs(deps.data_dir, exist_ok=True)
    return deps


def _warm_models(stages: list[str]) -> None:
    """Load the models a requested stage needs once, outside the timed region."""
    if {"maest", "supervise", "cluster"} & set(stages):
        maest_mod.load_model()
        if not maest_mod.is_model_ready():
            print(f"  MAEST unavailable: {maest_mod.load_error()}")
    if "clap" in stages:
        clap_mod.load_model()
        if not clap_mod.is_model_ready():
            print(f"  CLAP unavailable: {clap_mod.load_error()}")


def _handler_for(
    stage: str,
    deps: AnalysisDeps,
    store: ClusterStore,
    sample: list[str],
    *,
    features_workers: int,
    structure_workers: int,
    maest_batch: int,
    clap_batch: int,
) -> tuple[Any, int]:
    """The real production handler for *stage*, plus the worker's claim batch size."""
    if stage == "scan":
        return make_scan_handler(deps), 8
    if stage == "features":
        return (
            make_features_handler(deps, pool_size=features_workers),
            max(2, features_workers),
        )
    if stage == "structure":
        return (
            make_structure_handler(deps, pool_size=structure_workers),
            max(2, structure_workers),
        )
    if stage == "maest":
        return make_maest_handler(deps, batch_size=maest_batch), maest_batch
    if stage == "clap":
        return make_clap_handler(deps, batch_size=clap_batch), clap_batch

    def _build_infos(_folder: str, _recursive: bool) -> list[dict]:
        infos: list[dict] = []
        for p in sample:
            data = _cached_read_all(deps.track_cache, p)
            fp = data.get("fingerprint")
            if isinstance(fp, str) and fp:
                infos.append(
                    {"path": p, "fingerprint": fp, "title": data.get("title") or ""}
                )
        return infos

    if stage == "supervise":
        return make_supervised_handler(deps, build_track_infos=_build_infos), 1
    if stage == "cluster":
        return make_cluster_handler(deps, store, build_track_infos=_build_infos), 1
    raise ValueError(f"unknown stage: {stage}")


def _drain(
    stage: str,
    deps: AnalysisDeps,
    scratch: str,
    sample: list[str],
    *,
    run: int,
    handler: Any,
    claim_batch: int,
) -> dict:
    """Enqueue and drain one stage, timing the whole drain."""
    kind = _STAGE_KIND[stage]
    # A fresh queue per (stage, run): a reused queue would still hold the previous
    # run's jobs as ``done`` and the drain would finish instantly doing no work.
    queue = JobQueue(os.path.join(scratch, f"jobs-{stage}-{run}.db"), max_attempts=1)
    if stage in PER_TRACK_STAGES:
        queue.enqueue_many(kind, [(p, {"path": p}) for p in sample])
    else:
        queue.enqueue_many(kind, [("library", {"reason": "profile"})])

    worker = JobWorker(queue, kind, handler, batch_size=claim_batch, idle_sleep=0.02)
    started = time.perf_counter()
    worker.start()
    try:
        while queue.stats(kind).outstanding > 0:
            time.sleep(0.02)
    finally:
        worker.stop()
        if worker._thread is not None:
            worker._thread.join(timeout=120.0)
    elapsed = time.perf_counter() - started

    stats = queue.stats(kind)
    return {
        "stage": stage,
        "run": run + 1,
        "tracks": len(sample),
        "seconds": elapsed,
        "done": stats.done,
        "failed": stats.failed,
        "note": _note(queue.progress(kind)),
    }


def _note(progress: dict | None) -> str:
    """A one-line reason for a no-op or failure (whole-library stages skip often)."""
    detail = (progress or {}).get("detail") or {}
    if detail.get("skipped"):
        return f"skipped: {detail.get('reason', '')}"
    if detail.get("error"):
        return f"error: {detail['error']}"
    return ""


def _measure(
    stages: list[str],
    sample: list[str],
    *,
    repeat: int,
    features_workers: int,
    structure_workers: int,
    maest_batch: int,
    clap_batch: int,
) -> list[dict]:
    """Run the stage sequence *repeat* times, sharing caches within a repeat."""
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="trackspace-profile-") as scratch:
        deps = _make_deps(scratch)
        store = ClusterStore()
        # Handlers are built once so the features process pool and the models survive
        # across repeats; only the scratch caches are rebuilt, forcing real work.
        handlers = {
            stage: _handler_for(
                stage,
                deps,
                store,
                sample,
                features_workers=features_workers,
                structure_workers=structure_workers,
                maest_batch=maest_batch,
                clap_batch=clap_batch,
            )
            for stage in stages
        }
        for i in range(repeat):
            deps.track_cache = TrackCache(os.path.join(scratch, f"track-{i}.db"))
            deps.feature_cache = FeatureCache(os.path.join(scratch, f"feat-{i}.db"))
            # ``scan`` pays the fingerprint cost; every other stage gets them
            # pre-warmed (untimed) so it is not charged for the scan.
            if "scan" not in stages:
                for p in sample:
                    _cached_read_all(deps.track_cache, p)
            for stage in stages:
                handler, claim_batch = handlers[stage]
                rows.append(
                    _drain(
                        stage,
                        deps,
                        scratch,
                        sample,
                        run=i,
                        handler=handler,
                        claim_batch=claim_batch,
                    )
                )
    return rows


def _rate(row: dict) -> float:
    secs = row["seconds"]
    return row["tracks"] / secs * 60.0 if secs > 0 else 0.0


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'stage':>10} {'run':>4} {'tracks':>7} {'seconds':>10} "
        f"{'s/track':>9} {'tracks/min':>11} {'done':>6} {'failed':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        tracks = r["tracks"]
        secs = r["seconds"]
        per_track = secs / tracks if tracks else 0.0
        note = f"  {r['note']}" if r.get("note") else ""
        print(
            f"{r['stage']:>10} {r['run']:>4} {tracks:>7} {secs:>10.1f} "
            f"{per_track:>9.3f} {_rate(r):>11.1f} {r['done']:>6} {r['failed']:>7}"
            f"{note}"
        )
        expected = tracks if r["stage"] in PER_TRACK_STAGES else 1
        if r["done"] + r["failed"] != expected:
            print("  WARNING: not every job reached a terminal state")

    # Best (min seconds) run per stage: the number to compare across changes.
    print()
    best_header = f"{'stage':>10} {'best s/track':>13} {'best tracks/min':>16}"
    print(best_header)
    print("-" * len(best_header))
    for stage in dict.fromkeys(r["stage"] for r in rows):
        stage_rows = [r for r in rows if r["stage"] == stage]
        best = min(stage_rows, key=lambda r: r["seconds"])
        tracks = best["tracks"] or 1
        print(
            f"{stage:>10} {best['seconds'] / tracks:>13.3f} {_rate(best):>16.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Library root to sample from")
    parser.add_argument("--sample", type=int, default=48, help="Tracks per run")
    parser.add_argument(
        "--stages",
        default=",".join(PER_TRACK_STAGES),
        help=(
            "Comma-separated stages, or 'all' for every stage "
            f"(known: {', '.join(ALL_STAGES)})"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Runs with fresh scratch caches (later runs are steady-state)",
    )
    parser.add_argument(
        "--features-workers",
        type=int,
        default=4,
        help="Processes in the features stage pool",
    )
    parser.add_argument(
        "--structure-workers",
        type=int,
        default=4,
        help="Threads in the structure stage pool",
    )
    parser.add_argument("--maest-batch", type=int, default=4, help="MAEST tracks per batch")
    parser.add_argument("--clap-batch", type=int, default=4, help="CLAP tracks per batch")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    stages = (
        list(ALL_STAGES)
        if args.stages.strip().lower() == "all"
        else [s.strip() for s in args.stages.split(",") if s.strip()]
    )
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    # Whole-library stages need the sample's MAEST rows, so run maest before them.
    stages.sort(key=ALL_STAGES.index)

    sample = _pick_sample(root, args.sample)
    if not sample:
        print(f"no mp3s under {root}", file=sys.stderr)
        return 2

    print(f"root={root}")
    print(f"sample={len(sample)} tracks  repeat={max(1, args.repeat)}")
    print(
        f"features_workers={args.features_workers} "
        f"structure_workers={args.structure_workers} "
        f"maest_batch={args.maest_batch} clap_batch={args.clap_batch}"
    )
    print("warming feature pipeline (main thread)…")
    audio_features.warmup_audio_features()
    rhythm_mod.warmup_rhythm_features()
    print("warming models…")
    _warm_models(stages)

    rows = _measure(
        stages,
        sample,
        repeat=max(1, args.repeat),
        features_workers=max(1, args.features_workers),
        structure_workers=max(1, args.structure_workers),
        maest_batch=max(1, args.maest_batch),
        clap_batch=max(1, args.clap_batch),
    )
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
