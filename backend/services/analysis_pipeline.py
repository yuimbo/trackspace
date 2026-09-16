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
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.embeddings import clustering as clustering_mod
from backend.embeddings import supervised as supervised_mod
from backend.embeddings.clustering import ClusterResult
from backend.embeddings.projection_config import SUPERVISED_INPUT_WEIGHTS
from backend.jobs import (
    Job,
    JobQueue,
    JobResult,
    KIND_CLAP,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_SCAN,
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
            if deps.feature_cache.has("clap", fp, version=deps.clap_version):
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

        results = deps.clap_mod.analyze_batch(
            [abs_path for _, abs_path, _ in targets],
            on_decode_warning=_warn,
            max_batch=max(1, batch_size * deps.clap_mod.NUM_SEGMENTS),
        )
        _relay_warnings(deps, warns)

        for job, abs_path, fp in targets:
            emb = results.get(abs_path)
            if emb is None:
                out[job.id] = JobResult.failure("CLAP produced no embedding")
                continue
            try:
                deps.feature_cache.put("clap", fp, emb, version=deps.clap_version)
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
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Re-cluster the whole library.

    Cluster jobs are library-wide singletons: the queue's ``(kind, dedupe_key)``
    uniqueness means many "library changed" signals collapse into at most one
    pending re-cluster, which is exactly the desired coalescing behaviour.

    Source weights come from *store* (the current requested setting), falling back to
    the *weights* argument fixed at wiring time. Reading them at run time — rather
    than from the job payload — keeps user-chosen weights applied to the automatic
    re-clusters that follow an analysis drain.
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        if not jobs:
            return out

        effective_weights = store.weights() or weights
        try:
            result = run_clustering(
                deps, build_track_infos=build_track_infos, weights=effective_weights
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
) -> ClusterResult | None:
    """Gather cached vectors for the whole library and run the clustering pipeline."""
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


def train_supervised(
    deps: AnalysisDeps,
    *,
    build_track_infos: Callable[[str, bool], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Train the folder-supervised head and (re)write every supervised row.

    Returns handler detail: training stats on success, or ``{"skipped": True,
    "reason": ...}`` when there are too few labelled folders. Exceptions
    propagate: the handler maps them to a failed ``JobResult`` so the queue owns
    the retry decision.
    """
    infos = build_track_infos("", True)
    paths, blocks, _logits = gather_source_blocks(
        deps, infos, clap_version=deps.clap_version
    )
    if not paths:
        log.info("Supervised training skipped: no tracks with MAEST data")
        return {"skipped": True, "reason": "no tracks with MAEST data yet"}

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
    mining = _mining_vectors(blocks, excerpts)
    hard_pairs = (
        supervised_mod.mine_hard_negatives(mining, labels)
        if mining is not None
        else None
    )

    result = supervised_mod.train_head(
        mat, labels, sample_weights=confidence, hard_pairs=hard_pairs
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
        source_versions={
            "clap": deps.clap_version,
            "features": deps.features_version,
            "maest": deps.maest_version,
            "rhythm": deps.rhythm_version,
        },
        stats=result["stats"],
    )
    supervised_mod.save_head(deps.data_dir, result["state_dict"], meta)
    deps.feature_cache.set_meta(supervised_mod.RUN_ID_META_KEY, run_id)

    projected = supervised_mod.project_from_state(
        result["state_dict"],
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        mat=mat,
    )
    fp_by_path = {t["path"]: t.get("fingerprint") for t in infos}
    written = 0
    for i, path in enumerate(paths):
        fp = fp_by_path.get(path)
        if not fp:
            continue
        deps.feature_cache.put("supervised", fp, projected[i], version=run_id)
        written += 1

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
) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Train the supervised head once for the whole library (singleton job)."""

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        if not jobs:
            return out
        try:
            stats = train_supervised(deps, build_track_infos=build_track_infos)
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
) -> dict[str, int]:
    """Queue per-track analysis for tracks that still need it.

    Work is filtered against the cache *before* enqueueing so the queue reflects
    outstanding work rather than the whole library — which is what makes the
    progress numbers meaningful.
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
            KIND_FEATURES, items, priority=100, requeue_terminal=force
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
            KIND_MAEST, items, priority=100, requeue_terminal=force
        )

    # CLAP is skipped entirely when the server runs with --no-clap: the worker
    # gates on model readiness, so enqueueing would leave jobs pending forever.
    if KIND_CLAP in kinds and deps.clap_enabled:
        items = [
            (t["path"], {"path": t["path"]})
            for t in infos
            if t.get("fingerprint")
            and (force or t["fingerprint"] not in have_clap)
        ]
        queued[KIND_CLAP] = queue.enqueue_many(
            KIND_CLAP, items, priority=110, requeue_terminal=force
        )

    if priority_paths:
        wanted = list(priority_paths)
        for kind in (KIND_FEATURES, KIND_MAEST, KIND_CLAP):
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


def enqueue_supervised(queue: JobQueue, *, reason: str = "manual") -> int:
    """Request one library-wide supervised-head training run.

    Like clustering it is a singleton on the ``"library"`` dedupe key: retraining
    is whole-library by nature, so repeated requests must coalesce.
    """
    return queue.enqueue_many(
        KIND_SUPERVISED,
        [("library", {"reason": reason})],
        priority=150,
        requeue_terminal=True,
    )
