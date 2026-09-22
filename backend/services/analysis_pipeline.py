"""Binds job kinds to compute stages and owns the cached clustering result.

This is the seam between :mod:`backend.jobs` (generic durable queue) and
:mod:`backend.embeddings` (the models). Each ``_handle_*`` function is a
:data:`~backend.jobs.worker.JobHandler`: it receives a batch of claimed jobs and
returns one :class:`~backend.jobs.worker.JobResult` per job id.

Handler contract worth remembering when adding a stage:

* **Never raise for a single bad track.** Return ``JobResult.failure(...)`` so
  the rest of the batch still commits and the queue owns the retry decision.
* **Use ``retry=False`` for permanent errors** (missing file, no fingerprint) so
  a dead track does not consume three attempts and three model runs.
* **Batch at the GPU boundary only.** MAEST batches inference across tracks;
  the CPU stages stay per-track so one slow decode cannot stall a whole batch.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.embeddings import clustering as clustering_mod
from backend.embeddings import structure as structure_mod
from backend.embeddings import supervised as supervised_mod
from backend.embeddings.clustering import ClusterResult, ProgressFn
from backend.embeddings.projection_config import SUPERVISED_INPUT_WEIGHTS
from backend.jobs import (
    Job,
    JobQueue,
    JobResult,
    KIND_CLAP,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_LAYOUT,
    KIND_MAEST,
    KIND_SCAN,
    KIND_STRUCTURE,
    KIND_SUPERVISED,
)
from backend.services import features_worker

log = logging.getLogger(__name__)


@dataclass
class AnalysisDeps:
    """Everything the handlers need, injected so they stay testable."""

    feature_cache: Any
    track_cache: Any
    cached_read_all: Callable[[str], dict[str, Any]]
    resolve_virtual_path: Callable[[str], str]
    virtual_from_abs: Callable[[str], str]
    list_mp3s: Callable[[str, bool], list[str]]
    maest_mod: Any
    rhythm_mod: Any
    audio_features_mod: Any
    maest_version: int
    rhythm_version: int
    features_version: int
    clap_mod: Any = None
    clap_version: int = 0
    #: Coarse track-structure (intro/drop/break) cache version.
    structure_version: int = 0
    #: False when the server was started with ``--no-clap`` — the CLAP stage is
    #: then never enqueued, so a disabled server cannot accumulate pending jobs
    #: behind a model that will never load.
    clap_enabled: bool = True
    #: Root of the runtime ``data/`` dir — where the supervised head lives.
    data_dir: str = ""
    emit_decode_warning: Callable[[str, str], None] | None = None


class ClusterStore:
    """Holds the most recent :class:`ClusterResult` plus a path→cluster index.

    Kept in memory rather than SQLite: a full re-cluster of the real library
    takes a few seconds, so persistence would buy little and add an
    invalidation problem. The revision guards staleness.

    The store also owns the *requested* source weights (``None`` = frozen
    defaults). Weights are a setting, not a per-job argument, because re-clusters are
    also triggered automatically when an analysis stage drains; a per-job payload
    would let those automatic runs silently revert to the defaults.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._result: ClusterResult | None = None
        self._computed_at: float = 0.0
        self._weights: dict[str, float] | None = None

    def set(self, result: ClusterResult | None) -> None:
        with self._lock:
            self._result = result
            self._computed_at = time.time()

    def get(self) -> ClusterResult | None:
        with self._lock:
            return self._result

    def revision(self) -> str | None:
        with self._lock:
            return self._result.revision if self._result else None

    def computed_at(self) -> float:
        with self._lock:
            return self._computed_at

    def set_weights(self, weights: dict[str, float] | None) -> None:
        """Set the requested source weights (``None`` restores the defaults)."""
        with self._lock:
            self._weights = dict(weights) if weights else None

    def weights(self) -> dict[str, float] | None:
        with self._lock:
            return dict(self._weights) if self._weights else None

    def assignments_for(self, resolution: float) -> dict[str, int]:
        """path → cluster id at the nearest available resolution."""
        with self._lock:
            res = self._result
        if res is None:
            return {}
        level = res.level(resolution)
        if level is None:
            return {}
        return {p: int(level.labels[i]) for i, p in enumerate(res.paths)}

    def summary(self) -> dict[str, Any]:
        requested = self.weights()
        with self._lock:
            res = self._result
            at = self._computed_at
        if res is None:
            return {"available": False, "requested_weights": requested}
        return {
            "available": True,
            "computed_at": at,
            "requested_weights": requested,
            **res.as_dict(),
        }


# ──────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────


def _fingerprint_for(deps: AnalysisDeps, virtual_path: str) -> tuple[str | None, str | None]:
    """(fingerprint, abs_path) for a virtual path, or (None, None) if unusable."""
    try:
        abs_path = deps.resolve_virtual_path(virtual_path)
    except Exception as e:
        log.debug("resolve failed for %s: %s", virtual_path, e)
        return None, None
    if not abs_path or not os.path.isfile(abs_path):
        return None, None
    try:
        info = deps.cached_read_all(abs_path)
    except Exception as e:
        log.debug("read failed for %s: %s", abs_path, e)
        return None, abs_path
    fp = info.get("fingerprint")
    return (fp if isinstance(fp, str) and fp else None), abs_path


def make_scan_handler(deps: AnalysisDeps) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Read ID3 tags + compute the Chromaprint fingerprint for each track."""

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        for job in jobs:
            vpath = job.path
            if not vpath:
                out[job.id] = JobResult.failure("job payload has no path", retry=False)
                continue
            try:
                abs_path = deps.resolve_virtual_path(vpath)
            except Exception as e:
                out[job.id] = JobResult.failure(f"unresolvable path: {e}", retry=False)
                continue
            if not abs_path or not os.path.isfile(abs_path):
                out[job.id] = JobResult.failure("file not found", retry=False)
                continue
            try:
                info = deps.cached_read_all(abs_path)
            except Exception as e:
                out[job.id] = JobResult.failure(f"{type(e).__name__}: {e}")
                continue
            out[job.id] = JobResult.success(
                fingerprint=bool(info.get("fingerprint")),
                title=(info.get("title") or "").strip() or None,
            )
        return out

    return handle


def make_features_handler(
    deps: AnalysisDeps,
    *,
    pool_size: int = 1,
    pool_factory: Callable[[int], Any] | None = None,
    compute_fn: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Classical 6-D descriptors + the 14-D rhythm vector, both cached by fingerprint.

    madmom's BLSTM forward pass is GIL-bound, so a *thread* pool cannot run this
    stage in parallel — more features threads are slower, not faster (measured 0.8 s
    at 1 thread vs 10.5 s at 4 for one track). With ``pool_size > 1`` the
    per-track decode+compute runs in a **process** pool instead, which scales
    near-linearly; the parent keeps the cache writes and the warning relay.

    ``pool_factory`` / ``compute_fn`` exist so tests can inject fakes without
    spawning real processes.
    """

    def _compute_in_process(
        abs_path: str, *, want_features: bool, want_rhythm: bool
    ) -> dict[str, Any]:
        """Serial fallback using the injected modules (also what tests fake)."""
        out: dict[str, Any] = {
            "features": None, "rhythm": None, "warnings": {}, "error": None,
        }
        try:
            if want_features:
                warns: dict[str, str] = {}
                out["features"] = deps.audio_features_mod.generate_audio_features_batch(
                    [abs_path], decode_warnings=warns
                ).get(abs_path)
                out["warnings"].update(warns)
            if want_rhythm:
                warns = {}
                out["rhythm"] = deps.rhythm_mod.generate_rhythm_features_batch(
                    [abs_path], decode_warnings=warns
                ).get(abs_path)
                out["warnings"].update(warns)
        except Exception as e:  # noqa: BLE001 — reported per job, never raised
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    compute = compute_fn or (
        features_worker.compute_track_features
        if pool_size > 1
        else _compute_in_process
    )

    executor: Any = None
    executor_lock = threading.Lock()

    def _get_executor() -> Any:
        nonlocal executor
        if pool_size <= 1:
            return None
        with executor_lock:
            if executor is None:
                executor = (pool_factory or _default_pool_factory)(pool_size)
            return executor

    def _reset_executor() -> None:
        """Drop a broken pool so the next batch rebuilds it."""
        nonlocal executor
        with executor_lock:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    log.debug("features pool shutdown failed", exc_info=True)
                executor = None

    def _store(
        job: Job,
        abs_path: str,
        fp: str,
        want_features: bool,
        want_rhythm: bool,
        res: dict[str, Any],
    ) -> JobResult:
        if res.get("error"):
            return JobResult.failure(str(res["error"]))
        produced: list[str] = []
        failed: list[str] = []
        if want_features:
            vec = res.get("features")
            if vec is not None:
                deps.feature_cache.put(
                    "features", fp, vec, version=deps.features_version
                )
                produced.append("features")
            else:
                failed.append("features")
        if want_rhythm:
            vec = res.get("rhythm")
            if vec is not None:
                deps.feature_cache.put("rhythm", fp, vec, version=deps.rhythm_version)
                produced.append("rhythm")
            else:
                failed.append("rhythm")
        _relay_warnings(deps, res.get("warnings") or {})
        if failed and not produced:
            return JobResult.failure(
                f"extraction failed: {', '.join(failed)}", sources=failed
            )
        return JobResult.success(sources=produced, partial=failed or None)

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        targets: list[tuple[Job, str, str, bool, bool]] = []
        for job in jobs:
            vpath = job.path
            if not vpath:
                out[job.id] = JobResult.failure("job payload has no path", retry=False)
                continue
            fp, abs_path = _fingerprint_for(deps, vpath)
            if abs_path is None:
                out[job.id] = JobResult.failure("file not found", retry=False)
                continue
            if not fp:
                out[job.id] = JobResult.failure("no fingerprint (is fpcalc installed?)")
                continue
            want_features = not deps.feature_cache.has(
                "features", fp, version=deps.features_version
            )
            want_rhythm = not deps.feature_cache.has(
                "rhythm", fp, version=deps.rhythm_version
            )
            if not want_features and not want_rhythm:
                out[job.id] = JobResult.success(cached=True)
                continue
            targets.append((job, abs_path, fp, want_features, want_rhythm))

        if not targets:
            return out

        executor = _get_executor()
        if executor is None:
            results = {
                abs_path: compute(abs_path, want_features=wf, want_rhythm=wr)
                for _, abs_path, _, wf, wr in targets
            }
        else:
            try:
                futures = {
                    executor.submit(
                        compute, abs_path, want_features=wf, want_rhythm=wr
                    ): abs_path
                    for _, abs_path, _, wf, wr in targets
                }
                results = {}
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except Exception:
                log.exception("features worker pool failed")
                _reset_executor()
                for job, *_rest in targets:
                    out[job.id] = JobResult.failure("features worker pool failed")
                return out

        for job, abs_path, fp, wf, wr in targets:
            out[job.id] = _store(job, abs_path, fp, wf, wr, results.get(abs_path) or {})
        return out

    return handle


def _default_pool_factory(pool_size: int) -> Any:
    """A spawn-based process pool, warmed once per child.

    ``spawn`` (not ``fork``) is required: the parent is multi-threaded (Flask,
    workers, watchdog) and forking it would hand children locks held by other
    threads. ``warm`` builds each child's madmom processors on start, so the first
    track in a child does not pay the model-load cost.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    executor = ProcessPoolExecutor(
        max_workers=pool_size,
        mp_context=mp.get_context("spawn"),
        initializer=features_worker.warm,
    )
    atexit.register(executor.shutdown, wait=False, cancel_futures=True)
    return executor


def _default_thread_pool_factory(pool_size: int) -> Any:
    """A plain thread pool.

    Structure detection decodes with ``soundfile``, whose reads release the GIL, so
    threads scale (measured ~2.1x at 4 workers) without a spawned child re-importing
    the whole backend for a few hundred milliseconds of work per track.
    """
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="structure")
    atexit.register(executor.shutdown, wait=False, cancel_futures=True)
    return executor


def make_structure_handler(
    deps: AnalysisDeps,
    *,
    pool_size: int = 1,
    executor_factory: Callable[[int], Any] | None = None,
    compute_fn: Callable[[str], dict[str, Any]] | None = None,
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Intro / drop / break timestamps from a whole-track loudness envelope.

    One source, one per-track job. Decode dominates the cost and its reads release
    the GIL, so the per-track work runs in a **thread** pool (no spawned child pays
    the backend re-import the ``features`` process pool does). ``compute_fn`` /
    ``executor_factory`` exist so tests can inject fakes.
    """

    compute = compute_fn or structure_mod.compute_structure

    executor: Any = None
    executor_lock = threading.Lock()

    def _get_executor() -> Any:
        nonlocal executor
        if pool_size <= 1:
            return None
        with executor_lock:
            if executor is None:
                executor = (executor_factory or _default_thread_pool_factory)(pool_size)
            return executor

    def _reset_executor() -> None:
        """Drop a broken pool so the next batch rebuilds it."""
        nonlocal executor
        with executor_lock:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    log.debug("structure pool shutdown failed", exc_info=True)
                executor = None

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        targets: list[tuple[Job, str, str]] = []  # (job, abs_path, fingerprint)

        for job in jobs:
            vpath = job.path
            if not vpath:
                out[job.id] = JobResult.failure("job payload has no path", retry=False)
                continue
            fp, abs_path = _fingerprint_for(deps, vpath)
            if abs_path is None:
                out[job.id] = JobResult.failure("file not found", retry=False)
                continue
            if not fp:
                out[job.id] = JobResult.failure("no fingerprint (is fpcalc installed?)")
                continue
            if deps.feature_cache.has(
                "structure", fp, version=deps.structure_version
            ):
                out[job.id] = JobResult.success(cached=True)
                continue
            targets.append((job, abs_path, fp))

        if not targets:
            return out

        executor = _get_executor()
        if executor is None:
            results = {abs_path: compute(abs_path) for _, abs_path, _ in targets}
        else:
            try:
                futures = {
                    executor.submit(compute, abs_path): abs_path
                    for _, abs_path, _ in targets
                }
                results = {}
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except Exception:
                log.exception("structure worker pool failed")
                _reset_executor()
                for job, *_rest in targets:
                    out[job.id] = JobResult.failure("structure worker pool failed")
                return out

        for job, abs_path, fp in targets:
            res = results.get(abs_path) or {}
            if res.get("error"):
                out[job.id] = JobResult.failure(str(res["error"]))
                continue
            blob = res.get("structure")
            if blob is None:
                out[job.id] = JobResult.failure("structure produced no result")
                continue
            try:
                deps.feature_cache.put(
                    "structure", fp, blob, version=deps.structure_version
                )
            except Exception as e:
                out[job.id] = JobResult.failure(f"cache write failed: {e}")
                continue
            out[job.id] = JobResult.success()
        return out

    return handle


def make_maest_handler(
    deps: AnalysisDeps, *, batch_size: int = 4
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """MAEST embedding + 519 style logits, batched across the claimed jobs."""

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        targets: list[tuple[Job, str, str]] = []  # (job, abs_path, fingerprint)

        for job in jobs:
            vpath = job.path
            if not vpath:
                out[job.id] = JobResult.failure("job payload has no path", retry=False)
                continue
            fp, abs_path = _fingerprint_for(deps, vpath)
            if abs_path is None:
                out[job.id] = JobResult.failure("file not found", retry=False)
                continue
            if not fp:
                out[job.id] = JobResult.failure("no fingerprint (is fpcalc installed?)")
                continue
            # A row counts as cached only when both the pooled outputs *and* the
            # per-excerpt vectors are present: excerpt vectors were added after the
            # pooled pair shipped, so pre-existing rows must be re-analysed once.
            if deps.feature_cache.has(
                "maest", fp, version=deps.maest_version
            ) and deps.feature_cache.has(
                "maest_excerpts", fp, version=deps.maest_version
            ):
                out[job.id] = JobResult.success(cached=True)
                continue
            targets.append((job, abs_path, fp))

        if not targets:
            return out

        if not deps.maest_mod.is_model_ready():
            deps.maest_mod.load_model()
        if not deps.maest_mod.is_model_ready():
            err = deps.maest_mod.load_error() or "MAEST model unavailable"
            for job, _, _ in targets:
                out[job.id] = JobResult.failure(err)
            return out

        warns: dict[str, str] = {}

        def _warn(path: str, message: str) -> None:
            warns[path] = message

        results = deps.maest_mod.analyze_batch(
            [abs_path for _, abs_path, _ in targets],
            on_decode_warning=_warn,
            max_batch=max(1, batch_size * 3),
        )
        _relay_warnings(deps, warns)

        for job, abs_path, fp in targets:
            analysis = results.get(abs_path)
            if analysis is None:
                out[job.id] = JobResult.failure("MAEST produced no embedding")
                continue
            try:
                deps.feature_cache.put_maest(
                    fp,
                    analysis.embedding,
                    analysis.logits,
                    excerpts=analysis.excerpt_embeddings,
                    version=deps.maest_version,
                )
            except Exception as e:
                out[job.id] = JobResult.failure(f"cache write failed: {e}")
                continue
            out[job.id] = JobResult.success(excerpts=analysis.num_excerpts)
        return out

    return handle


def make_clap_handler(
    deps: AnalysisDeps, *, batch_size: int = 4
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """CLAP audio embedding, batched across the claimed jobs.

    CLAP is optional but not ornamental: its audio block is the column span the
    semantic weighting in ``layout`` projects tag text directions onto, so without it
    tag-based weighting is inert (see ``audio_pipeline.md``).
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        targets: list[tuple[Job, str, str]] = []  # (job, abs_path, fingerprint)

        for job in jobs:
            vpath = job.path
            if not vpath:
                out[job.id] = JobResult.failure("job payload has no path", retry=False)
                continue
            fp, abs_path = _fingerprint_for(deps, vpath)
            if abs_path is None:
                out[job.id] = JobResult.failure("file not found", retry=False)
                continue
            if not fp:
                out[job.id] = JobResult.failure("no fingerprint (is fpcalc installed?)")
                continue
            if deps.feature_cache.has("clap", fp, version=deps.clap_version) and (
                deps.feature_cache.has("clap_segments", fp, version=deps.clap_version)
            ):
                out[job.id] = JobResult.success(cached=True)
                continue
            targets.append((job, abs_path, fp))

        if not targets:
            return out

        if not deps.clap_mod.is_model_ready():
            deps.clap_mod.load_model()
        if not deps.clap_mod.is_model_ready():
            err = deps.clap_mod.load_error() or "CLAP model unavailable"
            for job, _, _ in targets:
                out[job.id] = JobResult.failure(err)
            return out

        warns: dict[str, str] = {}

        def _warn(path: str, message: str) -> None:
            warns[path] = message

        results = deps.clap_mod.analyze_batch_segments(
            [abs_path for _, abs_path, _ in targets],
            on_decode_warning=_warn,
            max_batch=max(1, batch_size * deps.clap_mod.NUM_SEGMENTS),
        )
        _relay_warnings(deps, warns)

        for job, abs_path, fp in targets:
            entry = results.get(abs_path)
            if entry is None:
                out[job.id] = JobResult.failure("CLAP produced no embedding")
                continue
            seg_vecs, seg_positions = entry
            mean = np.mean(seg_vecs, axis=0).astype(np.float32)
            try:
                deps.feature_cache.put("clap", fp, mean, version=deps.clap_version)
                deps.feature_cache.put(
                    "clap_segments", fp, seg_vecs, version=deps.clap_version
                )
                deps.feature_cache.put(
                    "clap_segment_pos", fp, seg_positions, version=deps.clap_version
                )
            except Exception as e:
                out[job.id] = JobResult.failure(f"cache write failed: {e}")
                continue
            out[job.id] = JobResult.success()
        return out

    return handle


def _relay_warnings(deps: AnalysisDeps, warns: dict[str, str]) -> None:
    if not warns or deps.emit_decode_warning is None:
        return
    for abs_path, message in warns.items():
        try:
            deps.emit_decode_warning(deps.virtual_from_abs(abs_path), message)
        except Exception:
            log.debug("decode warning relay failed for %s", abs_path)


def make_cluster_handler(
    deps: AnalysisDeps,
    store: ClusterStore,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
    emit: ProgressFn | None = None,
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Re-cluster the whole library.

    Cluster jobs are library-wide singletons: the queue's ``(kind, dedupe_key)``
    uniqueness means many "library changed" signals collapse into at most one
    pending re-cluster, which is exactly the desired coalescing behaviour.

    Source weights come from *store* (the current requested setting), falling back to
    the *weights* argument fixed at wiring time. Reading them at run time — rather
    than from the job payload — keeps user-chosen weights applied to the automatic
    re-clusters that follow an analysis drain.

    *emit* receives ``(phase, done, total)`` phase updates for the status UI.
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        if not jobs:
            return out

        effective_weights = store.weights() or weights
        try:
            result = run_clustering(
                deps,
                build_track_infos=build_track_infos,
                weights=effective_weights,
                on_progress=emit,
            )
        except Exception as e:
            log.exception("Clustering failed")
            for job in jobs:
                out[job.id] = JobResult.failure(f"{type(e).__name__}: {e}")
            return out

        store.set(result)
        detail: dict[str, Any] = (
            {
                "n_tracks": result.n_tracks,
                "levels": len(result.levels),
                "revision": result.revision[:12],
            }
            if result
            else {"n_tracks": 0, "levels": 0}
        )
        for job in jobs:
            out[job.id] = JobResult.success(**detail)
        return out

    return handle


def make_layout_handler(
    prefetch: Callable[[str], int],
    *,
    emit: ProgressFn | None = None,
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Warm one projection lens's 2-D layout per job.

    Library-wide but idempotent: a lens whose revision is already cached is a hit
    inside ``compute_projection``, so re-running the prefetch after every cluster
    pass costs a revision check, not a refit. *prefetch* returns the number of
    tracks placed and is wired to the projection stack in ``app.py``.
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        total = len(jobs)
        for i, job in enumerate(jobs):
            lens_id = str(job.payload.get("lens") or job.dedupe_key)
            try:
                n = int(prefetch(lens_id))
            except Exception as e:
                log.exception("Layout prefetch failed for lens %s", lens_id)
                out[job.id] = JobResult.failure(f"{type(e).__name__}: {e}")
                continue
            out[job.id] = JobResult.success(lens=lens_id, n_tracks=n)
            if emit is not None:
                emit(f"layout {lens_id}", i + 1, total)
        return out

    return handle


def gather_source_blocks(
    deps: AnalysisDeps,
    infos: list[dict[str, Any]],
    *,
    clap_version: int,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray | None]:
    """Assemble aligned per-source matrices for the tracks that have MAEST data.

    MAEST is required — it is the primary genre model, and a track without it
    has no meaningful position in the genre space. Optional sources are filled
    with zero rows where missing so every block keeps the same row order as
    *paths*; a zero row is neutral after per-block standardisation.
    """
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    if not fps:
        return [], {}, None

    fc = deps.feature_cache
    maest_map = fc.get_all("maest", fps, version=deps.maest_version)
    logits_map = fc.get_all("maest_logits", fps, version=deps.maest_version)
    rhythm_map = fc.get_all("rhythm", fps, version=deps.rhythm_version)
    feat_map = fc.get_all("features", fps, version=deps.features_version)
    clap_map = fc.get_all("clap", fps, version=clap_version)
    supervised_map = fc.get_all(
        "supervised", fps, version=supervised_mod.cache_version(fc)
    )

    paths: list[str] = []
    picked: list[str] = []
    for t in infos:
        fp = t.get("fingerprint")
        if fp and fp in maest_map and fp in logits_map:
            paths.append(t["path"])
            picked.append(fp)

    if not paths:
        return [], {}, None

    def _stack(source_map: dict[str, np.ndarray], name: str) -> np.ndarray | None:
        present = [source_map.get(fp) for fp in picked]
        dims = {int(v.size) for v in present if v is not None}
        if not dims:
            return None
        if len(dims) > 1:
            log.warning("%s: inconsistent vector sizes %s; skipping source", name, dims)
            return None
        dim = dims.pop()
        rows = [
            (v.astype(np.float32).reshape(-1) if v is not None else np.zeros(dim, np.float32))
            for v in present
        ]
        return np.stack(rows, axis=0).astype(np.float32)

    blocks: dict[str, np.ndarray] = {}
    for name, source_map in (
        ("maest", maest_map),
        ("maest_logits", logits_map),
        ("rhythm", rhythm_map),
        ("features", feat_map),
        ("clap", clap_map),
        ("supervised", supervised_map),
    ):
        block = _stack(source_map, name)
        if block is not None:
            blocks[name] = block

    return paths, blocks, blocks.get("maest_logits")


def run_clustering(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
    on_progress: ProgressFn | None = None,
) -> ClusterResult | None:
    """Gather cached vectors for the whole library and run the clustering pipeline."""

    def _progress(phase: str, done: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(phase, done, total)

    _progress("vectors")
    infos = build_track_infos("", True)
    # Give newly analysed tracks their supervised row before gathering, so this
    # pass already clusters them in the learned space. Best-effort: a projection
    # failure must not take clustering down with it.
    try:
        project_supervised_missing(deps, infos)
    except Exception:
        log.exception("Supervised projection failed; clustering without new rows")
    paths, blocks, logits = gather_source_blocks(
        deps,
        infos,
        clap_version=deps.clap_version,
    )
    if not paths:
        log.info("Clustering: no tracks with MAEST data yet")
        return None

    return clustering_mod.cluster_tracks(
        paths,
        blocks,
        weights=weights,
        style_logits=logits,
        style_label_names=deps.maest_mod.discogs_style_labels(),
        cache_versions=(
            deps.maest_version,
            deps.rhythm_version,
            deps.features_version,
            supervised_mod.cache_version(deps.feature_cache),
        ),
        on_progress=on_progress,
    )


# ──────────────────────────────────────────────────────────────
# Weak supervision (folder-supervised projection head)
# ──────────────────────────────────────────────────────────────


def _mining_vectors(
    blocks: dict[str, np.ndarray], excerpts: list[np.ndarray | None]
) -> np.ndarray | None:
    """Per-track 768-D MAEST vectors for hard-negative mining.

    Excerpt means live in the same space as the pooled MAEST block, so a track
    with excerpts is represented by its excerpt centroid and one without falls back
    to its pooled embedding — one consistent space either way.
    """
    base = blocks.get("maest")
    if base is None:
        return None
    vecs = np.asarray(base, dtype=np.float32).copy()
    for i, exc in enumerate(excerpts):
        if i >= vecs.shape[0]:
            break
        if exc is not None and exc.shape[0] > 0:
            vecs[i] = exc.mean(axis=0)
    return vecs


def _excerpts_aligned(
    deps: AnalysisDeps, infos: list[dict[str, Any]], paths: list[str]
) -> list[np.ndarray | None]:
    fp_by_path = {t["path"]: t.get("fingerprint") for t in infos}
    fps = [fp_by_path.get(p) for p in paths]
    exc_map = deps.feature_cache.get_all(
        "maest_excerpts", [fp for fp in fps if fp], version=deps.maest_version
    )
    return [
        supervised_mod.decode_excerpts(exc_map[fp]) if fp in exc_map else None
        for fp in fps
    ]


#: Sources the supervised head consumes: a track needs all of them before it can be
#: placed, and an imprint must analyse the target folder for exactly these. ``features``
#: is deliberately absent — its weight is 0 in ``SUPERVISED_INPUT_WEIGHTS``.
SUPERVISED_INPUT_SOURCES: tuple[str, ...] = (
    "maest",
    "maest_logits",
    "rhythm",
    "clap",
)


def _source_version(deps: AnalysisDeps, source: str) -> int:
    if source == "clap":
        return deps.clap_version
    if source == "rhythm":
        return deps.rhythm_version
    return deps.maest_version


def _current_source_versions(deps: AnalysisDeps) -> dict[str, int]:
    """Encoder versions a freshly trained head records in its meta.

    One owner for the source→version map so training and import compare against
    exactly the same set (a mismatch on any of these makes a head stale).
    """
    return {
        "clap": deps.clap_version,
        "features": deps.features_version,
        "maest": deps.maest_version,
        "rhythm": deps.rhythm_version,
    }


def supervised_imprint_status(
    deps: AnalysisDeps, infos: list[dict[str, Any]]
) -> dict[str, Any]:
    """How many of *infos* are fully analysed for a folder projection.

    Readiness means every input source the head consumes is cached, because
    :func:`build_input_matrix` zero-fills a missing source: a half-analysed track
    would still be placed, just wrongly. The count drives the imprint progress UI.
    """
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    total = len(infos)
    if not fps:
        return {"tracks": total, "analyzed": 0, "pending": total, "ready": False}
    have = set(fps)
    for source in SUPERVISED_INPUT_SOURCES:
        have &= set(
            deps.feature_cache.get_all(
                source, list(have), version=_source_version(deps, source)
            )
        )
    return {
        "tracks": total,
        "analyzed": len(have),
        "pending": total - len(have),
        "ready": total > 0 and len(have) == total,
    }


def _gather_selected_infos(
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    folders: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Track infos for the union of *folders* (recursive), or the whole library.

    Folders are virtual paths (a root id or a subtree). Overlapping selections
    are deduped by path so a track under two selected folders is counted once.
    """
    if not folders:
        return build_track_infos("", True)
    by_path: dict[str, dict[str, Any]] = {}
    for folder in folders:
        for info in build_track_infos(str(folder), True):
            by_path.setdefault(str(info.get("path", "")), info)
    return sorted(by_path.values(), key=lambda t: t["path"])


def train_supervised(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    folders: Sequence[str] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Train the folder-supervised head and (re)write every supervised row.

    *folders* restricts the training labels to those virtual folders (a root id or
    a subtree); ``None`` or empty trains on the whole library. Tracks outside the
    selection still get a ``supervised`` row from the trained head — the selection
    chooses what the head *learns* from, not what it is applied to.

    Returns handler detail: training stats on success, or ``{"skipped": True,
    "reason": ...}`` when there are too few labelled folders. Exceptions
    propagate: the handler maps them to a failed ``JobResult`` so the queue owns
    the retry decision.
    """

    def _progress(phase: str, done: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(phase, done, total)

    infos = _gather_selected_infos(build_track_infos, folders)
    _progress("gather", 0, len(infos))
    paths, blocks, _logits = gather_source_blocks(
        deps, infos, clap_version=deps.clap_version
    )
    if not paths:
        log.info("Supervised training skipped: no tracks with MAEST data")
        return {"skipped": True, "reason": "no tracks with MAEST data yet"}
    _progress("gather", len(paths), len(infos))

    labels, class_names = supervised_mod.derive_folder_labels(paths)
    if len(class_names) < 2:
        log.info(
            "Supervised training skipped: %d folder(s) with >= %d tracks",
            len(class_names),
            supervised_mod.MIN_FOLDER_LABEL_SIZE,
        )
        return {
            "skipped": True,
            "reason": (
                f"need at least 2 folders with "
                f"{supervised_mod.MIN_FOLDER_LABEL_SIZE}+ tracks "
                f"(found {len(class_names)})"
            ),
        }

    mat, used = supervised_mod.build_input_matrix(blocks, SUPERVISED_INPUT_WEIGHTS)
    if mat.size == 0 or mat.shape[0] != len(paths):
        log.warning("Supervised training skipped: input matrix is not aligned")
        return {"skipped": True, "reason": "input matrix is not aligned"}

    excerpts = _excerpts_aligned(deps, infos, paths)
    confidence = supervised_mod.excerpt_confidence(excerpts)
    _progress("mining")
    mining = _mining_vectors(blocks, excerpts)
    hard_pairs = (
        supervised_mod.mine_hard_negatives(mining, labels)
        if mining is not None
        else None
    )

    result = supervised_mod.train_head(
        mat,
        labels,
        sample_weights=confidence,
        hard_pairs=hard_pairs,
        on_epoch=lambda e, n: _progress("training", e, n),
    )

    projected = supervised_mod.project_from_state(
        result["state_dict"],
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        mat=mat,
    )
    # Per-folder centroids in the projected space, stored with the head so a
    # nearest-centroid folder assignment (and the export bundle) needs no retrain.
    centroids, _centroid_counts = supervised_mod.compute_centroids(
        projected, labels, len(class_names)
    )
    # Per-class cosine floor for the fan-out gate: a track is only placed when it
    # is at least as close to a learned folder as that folder's own tracks are.
    thresholds = supervised_mod.compute_match_thresholds(
        projected, labels, centroids, len(class_names)
    )

    run_id = supervised_mod.new_run_id(deps.data_dir)
    meta = supervised_mod.build_meta(
        run_id=run_id,
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        input_sources=used,
        input_weights=SUPERVISED_INPUT_WEIGHTS,
        class_names=class_names,
        train_folders=[str(f) for f in folders] if folders else [],
        source_versions=_current_source_versions(deps),
        class_match_thresholds=thresholds.tolist(),
        stats=result["stats"],
    )
    supervised_mod.save_head(
        deps.data_dir, result["state_dict"], meta, centroids=centroids
    )
    deps.feature_cache.set_meta(supervised_mod.RUN_ID_META_KEY, run_id)

    _progress("projecting", 0, len(paths))
    fp_by_path = {t["path"]: t.get("fingerprint") for t in infos}
    written = 0
    for i, path in enumerate(paths):
        fp = fp_by_path.get(path)
        if not fp:
            continue
        deps.feature_cache.put("supervised", fp, projected[i], version=run_id)
        written += 1
        if written % 500 == 0:
            _progress("projecting", written, len(paths))
    _progress("projecting", written, len(paths))

    stats = dict(result["stats"])
    stats.update(
        {
            "run_id": run_id,
            "rows_written": written,
            "input_dim": result["in_dim"],
            "folders": len(class_names),
        }
    )
    log.info(
        "Supervised head run %d: %d folders, %d labelled tracks, %d rows written",
        run_id,
        len(class_names),
        int((labels >= 0).sum()),
        written,
    )
    return stats


def project_supervised_missing(
    deps: AnalysisDeps, infos: list[dict[str, Any]]
) -> int:
    """Project tracks missing a supervised row through the current head.

    A cheap CPU matmul, so it runs at the top of every cluster pass: tracks
    analysed since the last training run get their row without a retrain.
    """
    head = supervised_mod.load_head(deps.data_dir)
    if head is None:
        return 0
    meta = head["meta"]
    version = int(meta.get("run_id", 0))
    if version <= 0:
        return 0

    paths, blocks, _logits = gather_source_blocks(
        deps, infos, clap_version=deps.clap_version
    )
    if not paths:
        return 0

    fp_by_path = {t["path"]: t.get("fingerprint") for t in infos}
    fps = [fp_by_path.get(p) for p in paths]
    have = set(
        deps.feature_cache.get_all(
            "supervised", [fp for fp in fps if fp], version=version
        )
    )
    todo = [i for i, fp in enumerate(fps) if fp and fp not in have]
    if not todo:
        return 0

    mat, used = supervised_mod.build_input_matrix(blocks, SUPERVISED_INPUT_WEIGHTS)
    if used != list(meta.get("input_sources", [])) or mat.shape[1] != int(
        meta.get("in_dim", 0)
    ):
        log.warning(
            "Supervised projection skipped: input space changed (used=%s meta=%s dim=%d/%s)",
            used,
            meta.get("input_sources"),
            mat.shape[1],
            meta.get("in_dim"),
        )
        return 0

    z = supervised_mod.project(head, mat[todo])
    written = 0
    for k, i in enumerate(todo):
        fp = fps[i]
        if not fp:
            continue
        deps.feature_cache.put("supervised", fp, z[k], version=version)
        written += 1
    if written:
        log.info(
            "Supervised: projected %d new track(s) with run %d", written, version
        )
    return written


def make_supervised_handler(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    emit: ProgressFn | None = None,
    default_folders: Callable[[], Sequence[str]] | None = None,
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Train the supervised head once for the whole library (singleton job).

    *default_folders* supplies the label sources when the job carries no explicit
    selection — the app wires the library roots here, so external roots are not
    trained on unless the user checks them explicitly.
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        if not jobs:
            return out
        # The folder selection rides on the singleton job's payload. A coalesced
        # re-request keeps the pending row (and its selection), so the UI disables
        # the Train button while a run is outstanding rather than relying on a
        # second click to change it.
        folders: list[str] | None = None
        for job in jobs:
            raw = job.payload.get("folders")
            if isinstance(raw, list) and raw:
                folders = [str(x) for x in raw]
                break
        if folders is None and default_folders is not None:
            folders = [str(f) for f in default_folders()] or None
        try:
            stats = train_supervised(
                deps,
                build_track_infos=build_track_infos,
                folders=folders,
                on_progress=emit,
            )
        except Exception as e:
            log.exception("Supervised training failed")
            for job in jobs:
                out[job.id] = JobResult.failure(f"{type(e).__name__}: {e}")
            return out
        # A skip is a *successful* no-op, not a failure: the auto-trigger fires
        # on every MAEST drain while no head exists, so it must stay re-runnable
        # until the library has enough labelled folders.
        for job in jobs:
            out[job.id] = JobResult.success(**stats)
        return out

    return handle


# ──────────────────────────────────────────────────────────────
# Head bundles: export / import / compare, and folder fan-out
# ──────────────────────────────────────────────────────────────


def supervised_head_info(deps: AnalysisDeps) -> dict[str, Any]:
    """Metadata for the installed head, for the UI's head controls."""
    head = supervised_mod.load_head(deps.data_dir)
    if head is None:
        return {"available": False}
    meta = head["meta"]
    payload = head["payload"]
    return {
        "available": True,
        "run_id": meta.get("run_id"),
        "trained_at": meta.get("trained_at"),
        "imported_at": meta.get("imported_at"),
        "n_classes": len(meta.get("class_names", [])),
        "in_dim": meta.get("in_dim"),
        "input_sources": list(meta.get("input_sources", [])),
        "train_folders": list(meta.get("train_folders", [])),
        "source_versions": dict(meta.get("source_versions", {})),
        "match_percentile": meta.get("match_percentile"),
        "has_match_thresholds": bool(meta.get("class_match_thresholds")),
        "has_centroids": payload.get("centroids") is not None,
    }


def export_supervised_head(deps: AnalysisDeps) -> dict | None:
    """Portable bundle of the installed head, or ``None`` when there is none."""
    return supervised_mod.head_bundle(deps.data_dir)


def import_supervised_head(
    deps: AnalysisDeps, bundle: dict, *, allow_mismatch: bool = False
) -> dict:
    """Install a bundle under a fresh local run id and re-project the library.

    The new run id makes every cached ``supervised`` row stale, so the next
    cluster pass re-projects the whole library through the imported head.

    A head trained on a different encoder version would produce rows in the wrong
    space, so the import is refused with
    :class:`~backend.embeddings.supervised.HeadVersionMismatch` unless
    *allow_mismatch* is set (the user has explicitly accepted it).
    """
    mismatches = supervised_mod.bundle_version_mismatches(
        bundle, _current_source_versions(deps)
    )
    if mismatches and not allow_mismatch:
        raise supervised_mod.HeadVersionMismatch(mismatches)
    meta = supervised_mod.install_head_bundle(deps.data_dir, bundle)
    deps.feature_cache.set_meta(
        supervised_mod.RUN_ID_META_KEY, int(meta["run_id"])
    )
    return meta


@dataclass(frozen=True)
class _HeadParts:
    """The installed head and the pieces its consumers match tracks against."""

    head: dict[str, Any] | None
    meta: dict[str, Any]
    class_names: list[str]
    centroids: np.ndarray | None
    thresholds: np.ndarray | None


def _load_head_parts(deps: AnalysisDeps) -> _HeadParts:
    """Load the installed head, its class names, centroids, and match thresholds.

    A missing head or centroids comes back as ``None`` rather than an error: each
    caller has its own message and availability semantics.
    """
    head = supervised_mod.load_head(deps.data_dir)
    if head is None:
        return _HeadParts(None, {}, [], None, None)
    meta = head["meta"]
    class_names = list(meta.get("class_names", []))
    raw_thresholds = meta.get("class_match_thresholds")
    thresholds = (
        np.asarray(raw_thresholds, dtype=np.float32)
        if isinstance(raw_thresholds, (list, tuple)) and raw_thresholds
        else None
    )
    return _HeadParts(
        head,
        meta,
        class_names,
        supervised_mod.head_centroids(deps.data_dir, head),
        thresholds,
    )


def supervised_predictions(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Nearest learned folder per track, from the installed head's centroids.

    Reads the cached ``supervised`` rows (no projection, no encoder work) and
    matches each against the head's per-folder centroids. ``assignments`` maps every
    track that has a row to its nearest folder; ``matched`` is the subset that also
    clears its class's training-distance floor — the same rule a fan-out uses to
    decide whether a track is close enough to a learned folder to be placed in it.
    """
    parts = _load_head_parts(deps)
    if parts.head is None or parts.centroids is None:
        return {"available": False, "assignments": {}, "matched": []}
    if not parts.class_names or parts.centroids.shape[0] != len(parts.class_names):
        return {"available": False, "assignments": {}, "matched": []}
    run_id = int(parts.meta.get("run_id", 0))

    infos = build_track_infos("", True)
    rows = deps.feature_cache.get_all(
        "supervised",
        [str(t["fingerprint"]) for t in infos if t.get("fingerprint")],
        version=run_id,
    )
    paths: list[str] = []
    vecs: list[np.ndarray] = []
    for t in infos:
        fp = t.get("fingerprint")
        vec = rows.get(str(fp)) if fp else None
        if vec is None:
            continue
        paths.append(str(t["path"]))
        vecs.append(vec)
    if not paths:
        return {"available": True, "assignments": {}, "matched": []}

    idx, sim = supervised_mod.predict_folders(parts.centroids, np.stack(vecs))
    matched = supervised_mod.matches_thresholds(idx, sim, parts.thresholds)
    assignments: dict[str, str] = {}
    matched_paths: list[str] = []
    for i, path in enumerate(paths):
        c = int(idx[i])
        if c < 0 or c >= len(parts.class_names):
            continue
        assignments[path] = parts.class_names[c]
        if bool(matched[i]):
            matched_paths.append(path)
    return {
        "available": True,
        "assignments": assignments,
        "matched": matched_paths,
    }


def _unique_target(path: str) -> str:
    """A path that does not clobber an existing file, ``" (n)"``-suffixed."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{base} ({n}){ext}"):
        n += 1
    return f"{base} ({n}){ext}"


def _sweep_empty_dirs(root_abs: str) -> int:
    """Remove empty directories below *root_abs*, deepest first.

    A recursive fan-out lifts every nested file into a learned folder, leaving the
    old subdirectories empty. ``os.rmdir`` only succeeds on an empty directory, so
    walking deepest-first also removes chains that held nothing but other empty
    folders. Never removes *root_abs* itself; returns how many directories went away.
    """
    root_abs = os.path.abspath(root_abs)
    dirs: list[str] = []
    for dirpath, dirnames, _files in os.walk(root_abs):
        for name in dirnames:
            dirs.append(os.path.join(dirpath, name))
    removed = 0
    for path in sorted(dirs, key=lambda p: p.count(os.sep), reverse=True):
        try:
            os.rmdir(path)
            removed += 1
        except OSError:
            continue
    return removed


def fanout_supervised(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
    folder_vpath: str,
    recursive: bool = False,
    apply: bool = False,
    generic_unmatched: bool = False,
) -> dict[str, Any]:
    """Sort the tracks in *folder_vpath* into the head's folder subpaths.

    The plan is always computed; *apply* is the only step that touches disk, so
    the caller can review the moves first. A track already in its predicted folder
    is skipped, a class that is the training root itself maps to no subfolder and
    is skipped, and a name collision gets a `` (n)`` suffix rather than
    overwriting a file.

    A track only moves when it is *close enough* to a learned folder: its winning
    similarity must clear that class's training-distance floor (see
    :func:`supervised_mod.compute_match_thresholds`). With *generic_unmatched*,
    a track that resembles no learned folder is filed under its top Discogs label
    as ``Generic/<Genre>/<Style>`` instead; without it (or when the label table is
    unavailable) the track is reported as ``unmatched`` and left where it is.

    With *recursive*, nested files are lifted into the learned folders directly
    below the target root, their pre-move relative folder is remembered per
    fingerprint, and the emptied old directories are swept afterwards. On *apply*,
    every move is journalled before it happens so
    :func:`undo_fanout_supervised` can restore the previous layout.
    """
    parts = _load_head_parts(deps)
    if parts.head is None:
        return {"ok": False, "error": "no trained head to fan out with"}
    if parts.centroids is None:
        return {"ok": False, "error": "head has no folder centroids; retrain it"}
    if parts.centroids.shape[0] != len(parts.class_names):
        return {"ok": False, "error": "head centroids do not match its folders"}
    head, meta, class_names = parts.head, parts.meta, parts.class_names
    centroids, thresholds = parts.centroids, parts.thresholds

    infos = build_track_infos(folder_vpath, recursive)
    if not infos:
        return {
            "ok": True,
            "folder": folder_vpath,
            "tracks": 0,
            "moved": 0,
            "generic": 0,
            "unmatched": 0,
            "applied": bool(apply),
            "plan": [],
            "swept_dirs": 0,
        }

    paths, blocks, logits = gather_source_blocks(
        deps, infos, clap_version=deps.clap_version
    )
    mat, used = supervised_mod.build_input_matrix(blocks, SUPERVISED_INPUT_WEIGHTS)
    if (
        not paths
        or used != list(meta.get("input_sources", []))
        or mat.shape[1] != int(meta.get("in_dim", 0))
    ):
        return {
            "ok": False,
            "error": "input space does not match the head; retrain it here first",
        }

    projected = supervised_mod.project(head, mat)
    idx, sim = supervised_mod.predict_folders(centroids, projected)
    matched = supervised_mod.matches_thresholds(idx, sim, thresholds)
    root_abs = deps.resolve_virtual_path(folder_vpath)
    fp_by_path = {
        str(t["path"]): str(t["fingerprint"])
        for t in infos
        if t.get("fingerprint")
    }

    plan: list[dict[str, Any]] = []
    moved = 0
    generic = 0
    unmatched = 0
    errors: list[str] = []
    batch_id = 0

    def _generic_bucket(i: int) -> tuple[str, str] | None:
        """``(subpath, label)`` from a track's top Discogs style, or ``None``."""
        if logits is None or i >= logits.shape[0]:
            return None
        styles = deps.maest_mod.top_styles(logits[i], k=1)
        if not styles:
            return None
        label = str(styles[0][0])
        sub = supervised_mod.generic_subpath(label)
        return (sub, label) if sub else None

    def _place(vpath: str, src: str, sub: str, entry: dict[str, Any]) -> None:
        """Plan or apply one move into ``root_abs/sub``; updates the counters."""
        nonlocal moved, batch_id
        target = os.path.normpath(os.path.join(root_abs, sub, os.path.basename(src)))
        entry["target"] = deps.virtual_from_abs(target)
        if target == src:
            entry["status"] = "in-place"
            return
        if not apply:
            entry["status"] = "planned"
            return
        row_id = 0
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            target = _unique_target(target)
            if batch_id == 0:
                batch_id = deps.feature_cache.journal_begin()
            row_id = deps.feature_cache.journal_add(batch_id, src, target)
            shutil.move(src, target)
            fp = fp_by_path.get(vpath)
            if recursive and fp:
                rel = os.path.relpath(src, root_abs)
                origin = os.path.dirname(rel)
                if origin and origin != ".":
                    deps.feature_cache.record_origin(fp, origin)
            moved += 1
            entry["target"] = deps.virtual_from_abs(target)
            entry["status"] = "moved"
        except OSError as e:
            deps.feature_cache.journal_remove(row_id)
            entry["status"] = "error"
            errors.append(f"{vpath}: {e}")

    for i, vpath in enumerate(paths):
        c = int(idx[i])
        src = deps.resolve_virtual_path(vpath)
        folder_name = class_names[c] if 0 <= c < len(class_names) else ""
        sub = supervised_mod.folder_subpath(folder_name) if c >= 0 else ""
        if bool(matched[i]) and sub:
            entry = {
                "path": vpath,
                "folder": folder_name,
                "similarity": float(sim[i]),
            }
            _place(vpath, src, sub, entry)
            plan.append(entry)
            continue
        bucket = _generic_bucket(i) if generic_unmatched else None
        if bucket is not None:
            generic += 1
            generic_sub, label = bucket
            entry = {
                "path": vpath,
                "folder": generic_sub,
                "label": label,
                "similarity": float(sim[i]),
                "generic": True,
            }
            _place(vpath, src, generic_sub, entry)
            plan.append(entry)
            continue
        # Too far from every learned folder to belong to one: leave it in place
        # and report why, rather than force it into the nearest.
        unmatched += 1
        plan.append(
            {
                "path": vpath,
                "folder": folder_name,
                "target": vpath,
                "similarity": float(sim[i]),
                "threshold": (
                    float(thresholds[c])
                    if thresholds is not None and 0 <= c < thresholds.size
                    else None
                ),
                "status": "unmatched",
            }
        )

    swept = _sweep_empty_dirs(root_abs) if apply and recursive else 0
    return {
        "ok": True,
        "folder": folder_vpath,
        "tracks": len(paths),
        "moved": moved,
        "generic": generic,
        "unmatched": unmatched,
        "applied": bool(apply),
        "errors": errors,
        "plan": plan,
        "swept_dirs": swept,
    }


def undo_fanout_supervised(deps: AnalysisDeps) -> dict[str, Any]:
    """Reverse the most recent applied fan-out batch from the move journal.

    Each moved file is restored to its original path, which also recovers the name
    a collision suffix had changed. A target that is gone (already moved again) is
    skipped, and an original path that is occupied is reported as an error rather
    than clobbered. The batch is marked undone either way so a partial failure
    does not strand the journal.
    """
    batch = deps.feature_cache.latest_move_batch()
    if batch is None or not batch["moves"]:
        return {"ok": False, "error": "no fan-out moves to undo"}

    restored = 0
    skipped = 0
    errors: list[str] = []
    for move in reversed(batch["moves"]):
        source, target = move["source"], move["target"]
        if not os.path.exists(target):
            skipped += 1
            continue
        if os.path.exists(source):
            errors.append(f"{target}: {source} already exists")
            continue
        try:
            os.makedirs(os.path.dirname(source), exist_ok=True)
            shutil.move(target, source)
            restored += 1
        except OSError as e:
            errors.append(f"{target}: {e}")
    deps.feature_cache.mark_batch_undone(batch["batch_id"])
    return {
        "ok": True,
        "batch_id": batch["batch_id"],
        "total": len(batch["moves"]),
        "restored": restored,
        "skipped": skipped,
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────
# Enqueue helpers
# ──────────────────────────────────────────────────────────────


def enqueue_library_work(
    queue: JobQueue,
    infos: list[dict[str, Any]],
    *,
    deps: AnalysisDeps,
    kinds: Sequence[str] = (KIND_FEATURES, KIND_MAEST),
    priority_paths: Sequence[str] = (),
    force: bool = False,
    revive_failed: bool = False,
) -> dict[str, int]:
    """Queue per-track analysis for tracks that still need it.

    Work is filtered against the cache *before* enqueueing so the queue reflects
    outstanding work rather than the whole library — which is what makes the
    progress numbers meaningful.

    A track that is in that filtered list always gets a runnable job: a *done*
    queue row for it is revived, because the cache — not the queue — is the source
    of truth. Without this a source-version bump (which blanks the stale rows on
    load) would never re-queue its backfill: the cache says the work is missing
    while the queue still calls it done. *force* only widens the cache filter to
    include already-cached tracks.

    *revive_failed* additionally revives terminally-failed rows. Automatic
    keep-warm enqueues leave it off so a genuinely broken file is not retried on
    every call; explicit retriggers and the version-purge backfill set it.
    """
    fc = deps.feature_cache
    queued: dict[str, int] = {}

    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    have_maest = (
        set(fc.get_all("maest", fps, version=deps.maest_version))
        if KIND_MAEST in kinds
        else set()
    )
    # Excerpt vectors shipped after the pooled MAEST pair, so a track can be
    # "analysed" yet still lack them; the queue re-runs it once to backfill.
    have_excerpts = (
        set(fc.get_all("maest_excerpts", fps, version=deps.maest_version))
        if KIND_MAEST in kinds
        else set()
    )
    have_rhythm = (
        set(fc.get_all("rhythm", fps, version=deps.rhythm_version))
        if KIND_FEATURES in kinds
        else set()
    )
    have_feats = (
        set(fc.get_all("features", fps, version=deps.features_version))
        if KIND_FEATURES in kinds
        else set()
    )
    have_clap = (
        set(fc.get_all("clap", fps, version=deps.clap_version))
        if KIND_CLAP in kinds
        else set()
    )
    # Segment vectors shipped after the pooled CLAP mean, so a track can be
    # "analysed" yet still lack them; the queue re-runs it once to backfill.
    have_clap_segments = (
        set(fc.get_all("clap_segments", fps, version=deps.clap_version))
        if KIND_CLAP in kinds
        else set()
    )
    have_structure = (
        set(fc.get_all("structure", fps, version=deps.structure_version))
        if KIND_STRUCTURE in kinds
        else set()
    )

    if KIND_SCAN in kinds:
        items = [(t["path"], {"path": t["path"]}) for t in infos]
        queued[KIND_SCAN] = queue.enqueue_many(
            KIND_SCAN, items, priority=50, requeue_terminal=force
        )

    if KIND_FEATURES in kinds:
        items = [
            (t["path"], {"path": t["path"]})
            for t in infos
            if t.get("fingerprint")
            and (
                force
                or t["fingerprint"] not in have_rhythm
                or t["fingerprint"] not in have_feats
            )
        ]
        queued[KIND_FEATURES] = queue.enqueue_many(
            KIND_FEATURES,
            items,
            priority=100,
            requeue_terminal=True,
            requeue_failed=revive_failed,
        )

    if KIND_MAEST in kinds:
        items = [
            (t["path"], {"path": t["path"]})
            for t in infos
            if t.get("fingerprint")
            and (
                force
                or t["fingerprint"] not in have_maest
                or t["fingerprint"] not in have_excerpts
            )
        ]
        queued[KIND_MAEST] = queue.enqueue_many(
            KIND_MAEST,
            items,
            priority=100,
            requeue_terminal=True,
            requeue_failed=revive_failed,
        )

    if KIND_STRUCTURE in kinds:
        items = [
            (t["path"], {"path": t["path"]})
            for t in infos
            if t.get("fingerprint")
            and (force or t["fingerprint"] not in have_structure)
        ]
        queued[KIND_STRUCTURE] = queue.enqueue_many(
            KIND_STRUCTURE,
            items,
            priority=105,
            requeue_terminal=True,
            requeue_failed=revive_failed,
        )

    # CLAP is skipped entirely when the server runs with --no-clap: the worker
    # gates on model readiness, so enqueueing would leave jobs pending forever.
    if KIND_CLAP in kinds and deps.clap_enabled:
        items = [
            (t["path"], {"path": t["path"]})
            for t in infos
            if t.get("fingerprint")
            and (
                force
                or t["fingerprint"] not in have_clap
                or t["fingerprint"] not in have_clap_segments
            )
        ]
        queued[KIND_CLAP] = queue.enqueue_many(
            KIND_CLAP,
            items,
            priority=110,
            requeue_terminal=True,
            requeue_failed=revive_failed,
        )

    if priority_paths:
        wanted = list(priority_paths)
        for kind in (KIND_FEATURES, KIND_STRUCTURE, KIND_MAEST, KIND_CLAP):
            if kind in kinds and (kind != KIND_CLAP or deps.clap_enabled):
                queue.prioritize(kind, wanted, priority=0)

    return queued


def enqueue_clustering(queue: JobQueue, *, reason: str = "library-change") -> int:
    """Request one library-wide re-cluster.

    All pending requests collapse onto the single dedupe key ``"library"``, so
    a burst of filesystem events cannot queue a burst of expensive re-clusters.
    """
    return queue.enqueue_many(
        KIND_CLUSTER,
        [("library", {"reason": reason})],
        priority=200,
        requeue_terminal=True,
    )


def enqueue_layouts(
    queue: JobQueue,
    lens_ids: Sequence[str],
    *,
    reason: str = "cluster-drained",
) -> int:
    """Request one layout-prefetch job per lens.

    ``(kind, dedupe_key)`` uniqueness collapses repeats, so a burst of drains
    cannot queue a burst of fits; ``requeue_terminal`` revives the completed rows
    so the next drain re-checks the revision.
    """
    return queue.enqueue_many(
        KIND_LAYOUT,
        [(lens_id, {"lens": lens_id, "reason": reason}) for lens_id in lens_ids],
        priority=300,
        requeue_terminal=True,
    )


def enqueue_supervised(
    queue: JobQueue,
    *,
    reason: str = "manual",
    folders: Sequence[str] | None = None,
) -> int:
    """Request one supervised-head training run.

    Like clustering it is a singleton on the ``"library"`` dedupe key, because a
    retrain is whole-head by nature and repeated requests must coalesce. *folders*
    restricts the training labels to those virtual folders; the payload carries the
    selection so the handler knows what to train on.
    """
    payload: dict[str, Any] = {"reason": reason}
    if folders:
        payload["folders"] = [str(f) for f in folders]
    return queue.enqueue_many(
        KIND_SUPERVISED,
        [("library", payload)],
        priority=150,
        requeue_terminal=True,
    )
