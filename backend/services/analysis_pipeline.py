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

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.embeddings import clustering as clustering_mod
from backend.embeddings.clustering import ClusterResult
from backend.jobs import Job, JobQueue, JobResult, KIND_CLUSTER, KIND_FEATURES, KIND_MAEST, KIND_SCAN

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
    clap_version: int = 0
    effnet_version: int = 0
    emit_decode_warning: Callable[[str, str], None] | None = None


class ClusterStore:
    """Holds the most recent :class:`ClusterResult` plus a path→cluster index.

    Kept in memory rather than SQLite: a full re-cluster of the real library
    takes a few seconds, so persistence would buy little and add an
    invalidation problem. The revision guards staleness.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._result: ClusterResult | None = None
        self._computed_at: float = 0.0

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
        with self._lock:
            res = self._result
            at = self._computed_at
        if res is None:
            return {"available": False}
        return {"available": True, "computed_at": at, **res.as_dict()}


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


def make_features_handler(deps: AnalysisDeps) -> Callable[[Sequence[Job]], dict[int, JobResult]]:
    """Classical 6-D descriptors + the 14-D rhythm vector, both cached by fingerprint."""

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
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

            produced: list[str] = []
            failed: list[str] = []

            if not deps.feature_cache.has_audio_features(fp, version=deps.features_version):
                warns: dict[str, str] = {}
                vec = deps.audio_features_mod.generate_audio_features_batch(
                    [abs_path], decode_warnings=warns
                ).get(abs_path)
                _relay_warnings(deps, warns)
                if vec is not None:
                    deps.feature_cache.put_audio_features(
                        fp, vec, version=deps.features_version
                    )
                    produced.append("features")
                else:
                    failed.append("features")

            if not deps.feature_cache.has_rhythm_features(fp, version=deps.rhythm_version):
                warns = {}
                vec = deps.rhythm_mod.generate_rhythm_features_batch(
                    [abs_path], decode_warnings=warns
                ).get(abs_path)
                _relay_warnings(deps, warns)
                if vec is not None:
                    deps.feature_cache.put_rhythm_features(
                        fp, vec, version=deps.rhythm_version
                    )
                    produced.append("rhythm")
                else:
                    failed.append("rhythm")

            if failed and not produced:
                out[job.id] = JobResult.failure(
                    f"extraction failed: {', '.join(failed)}", sources=failed
                )
            else:
                out[job.id] = JobResult.success(sources=produced, partial=failed or None)
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
            if deps.feature_cache.has_maest_embedding(fp, version=deps.maest_version):
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
                    fp, analysis.embedding, analysis.logits, version=deps.maest_version
                )
            except Exception as e:
                out[job.id] = JobResult.failure(f"cache write failed: {e}")
                continue
            out[job.id] = JobResult.success(excerpts=analysis.num_excerpts)
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
    """

    def handle(jobs: Sequence[Job]) -> dict[int, JobResult]:
        out: dict[int, JobResult] = {}
        if not jobs:
            return out

        try:
            result = run_clustering(
                deps, build_track_infos=build_track_infos, weights=weights
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
    effnet_version: int,
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
    maest_map = fc.get_all_maest_embeddings(fps, version=deps.maest_version)
    logits_map = fc.get_all_maest_logits(fps, version=deps.maest_version)
    rhythm_map = fc.get_all_rhythm_features(fps, version=deps.rhythm_version)
    feat_map = fc.get_all_audio_features(fps, version=deps.features_version)
    clap_map = fc.get_all_embeddings(fps, version=clap_version)
    effnet_map = fc.get_all_effnet_embeddings(fps, version=effnet_version)

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
        ("effnet", effnet_map),
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
    paths, blocks, logits = gather_source_blocks(
        deps,
        infos,
        clap_version=deps.clap_version,
        effnet_version=deps.effnet_version,
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
        ),
    )


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
        set(fc.get_all_maest_embeddings(fps, version=deps.maest_version))
        if KIND_MAEST in kinds
        else set()
    )
    have_rhythm = (
        set(fc.get_all_rhythm_features(fps, version=deps.rhythm_version))
        if KIND_FEATURES in kinds
        else set()
    )
    have_feats = (
        set(fc.get_all_audio_features(fps, version=deps.features_version))
        if KIND_FEATURES in kinds
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
            if t.get("fingerprint") and (force or t["fingerprint"] not in have_maest)
        ]
        queued[KIND_MAEST] = queue.enqueue_many(
            KIND_MAEST, items, priority=100, requeue_terminal=force
        )

    if priority_paths:
        wanted = list(priority_paths)
        for kind in (KIND_FEATURES, KIND_MAEST):
            if kind in kinds:
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
