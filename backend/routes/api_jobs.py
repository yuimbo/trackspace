"""HTTP surface for the analysis queue and clustering results.

The queue is the single source of truth for "what is the server doing" — these
endpoints read aggregate SQL counts rather than in-process counters, so the
numbers stay correct across restarts and cannot drift from reality.
"""

from __future__ import annotations

import json as _json
import logging
import queue as _queue
from collections.abc import Callable
from typing import Any

from flask import Blueprint, jsonify, request

from backend.embeddings.clustering import DEFAULT_WEIGHTS
from backend.jobs import (
    JOB_KINDS,
    KIND_CLAP,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_SUPERVISED,
    kind_label,
)
from backend.services.request_validation import (
    one_of,
    opt_bool,
    opt_int,
    opt_str,
    opt_str_list,
    opt_weight_map,
    parse_json_body,
)

log = logging.getLogger(__name__)


def create_api_jobs_blueprint(
    *,
    job_queue: Any,
    workers: Any,
    clusters: Any,
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
    enqueue_library_work: Callable[..., dict[str, int]],
    enqueue_clustering: Callable[..., int],
    enqueue_supervised: Callable[..., int],
    analysis_deps: Any,
    broadcaster: Any,
    sse_response: Callable[[Any], Any],
    maest_mod: Any,
) -> Blueprint:
    bp = Blueprint("api_jobs", __name__)

    def _queue_payload() -> dict[str, Any]:
        by_kind = job_queue.stats_by_kind()
        kinds: dict[str, Any] = {}
        for kind in JOB_KINDS:
            st = by_kind.get(kind)
            counts = (st.as_dict() if st else {
                "pending": 0, "running": 0, "done": 0, "failed": 0,
                "total": 0, "finished": 0, "outstanding": 0,
            })
            entry: dict[str, Any] = dict(counts)
            entry["label"] = kind_label(kind)
            progress = job_queue.progress(kind)
            if progress:
                entry["last"] = progress
            stage = job_queue.stage_progress(kind)
            if stage:
                entry["stage"] = stage
            kinds[kind] = entry

        outstanding = sum(k["outstanding"] for k in kinds.values())
        total = sum(k["total"] for k in kinds.values())
        finished = sum(k["finished"] for k in kinds.values())
        return {
            "kinds": kinds,
            "order": list(JOB_KINDS),
            "outstanding": outstanding,
            "total": total,
            "finished": finished,
            "active": outstanding > 0,
            "epoch": job_queue.epoch(),
        }

    @bp.route("/api/jobs/status")
    def api_jobs_status():
        payload = _queue_payload()
        payload["models"] = {
            "maest_ready": maest_mod.is_model_ready(),
            "maest_error": maest_mod.load_error(),
        }
        payload["clusters"] = {
            "available": clusters.get() is not None,
            "revision": clusters.revision(),
        }
        return jsonify(payload)

    @bp.route("/api/jobs/failures")
    def api_jobs_failures():
        kind = request.args.get("kind") or None
        if kind is not None:
            one_of(kind, JOB_KINDS, field="kind")
        limit = opt_int(request.args, "limit", 50, lo=1, hi=200)
        return jsonify({"failures": job_queue.failures(kind, limit=limit)})

    @bp.route("/api/jobs/enqueue", methods=["POST"])
    def api_jobs_enqueue():
        """Queue outstanding analysis for the library (idempotent)."""
        data = parse_json_body(required=False)
        folder = opt_str(data, "folder", "")
        recursive = opt_bool(data, "recursive", True)
        force = opt_bool(data, "force", False)
        priority_paths = opt_str_list(data, "priority_paths")
        kinds = opt_str_list(data, "kinds", [KIND_FEATURES, KIND_MAEST, KIND_CLAP])
        for kind in kinds:
            one_of(kind, JOB_KINDS, field="kinds")

        infos = build_track_infos_fn(folder, recursive)
        queued = enqueue_library_work(
            job_queue,
            infos,
            deps=analysis_deps,
            kinds=tuple(kinds),
            priority_paths=priority_paths,
            force=force,
        )
        workers.wake()
        return jsonify({"ok": True, "queued": queued, "queue": _queue_payload()})

    @bp.route("/api/jobs/retry", methods=["POST"])
    def api_jobs_retry():
        data = parse_json_body(required=False)
        kind = data.get("kind") or None
        if kind is not None:
            one_of(kind, JOB_KINDS, field="kind")
        n = job_queue.retry_failed(kind)
        workers.wake(kind)
        return jsonify({"ok": True, "retried": n, "queue": _queue_payload()})

    @bp.route("/api/jobs/stream")
    def api_jobs_stream():
        """SSE feed of queue progress.

        Always emits an initial snapshot so a client connecting mid-run (or
        after a reload) renders correct state immediately rather than waiting
        for the next job to finish.
        """
        q = broadcaster.subscribe(maxsize=4000)

        def generate():
            try:
                snapshot = _json.dumps({"type": "snapshot", **_queue_payload()})
                yield f"event: snapshot\ndata: {snapshot}\n\n"
                while True:
                    try:
                        event = q.get(timeout=25)
                    except _queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    etype = event.get("type", "progress")
                    yield f"event: {etype}\ndata: {_json.dumps(event)}\n\n"
            finally:
                broadcaster.unsubscribe(q)

        return sse_response(generate())

    # ── clustering ────────────────────────────────────────────

    @bp.route("/api/clusters")
    def api_clusters():
        """Cluster assignments at one resolution, plus the level summary."""
        result = clusters.get()
        if result is None:
            return jsonify({"available": False, "assignments": {}, "levels": []})

        raw = request.args.get("resolution")
        try:
            resolution = float(raw) if raw is not None else None
        except ValueError:
            resolution = None

        level = (
            result.level(resolution)
            if resolution is not None
            else result.levels[len(result.levels) // 2]
        )
        if level is None:
            return jsonify({"available": False, "assignments": {}, "levels": []})

        assignments = {p: int(level.labels[i]) for i, p in enumerate(result.paths)}
        return jsonify({
            "available": True,
            "revision": result.revision,
            "resolution": level.resolution,
            "resolutions": [lv.resolution for lv in result.levels],
            "n_clusters": level.n_clusters,
            "assignments": assignments,
            "names": {str(k): v for k, v in level.names.items()},
            "sizes": {str(k): v for k, v in level.sizes.items()},
            "modularity": level.modularity,
            "levels": [lv.as_dict() for lv in result.levels],
            "weights": result.weights,
            "sources_used": result.sources_used,
        })

    @bp.route("/api/clusters/summary")
    def api_clusters_summary():
        return jsonify(clusters.summary())

    @bp.route("/api/clusters/rebuild", methods=["POST"])
    def api_clusters_rebuild():
        """Queue a re-cluster, optionally changing the source weights.

        An absent ``weights`` field keeps the current setting (defaults on a fresh
        server), so the plain "rebuild" button stays a re-run rather than a reset.
        A supplied map is validated, stored, and applied by the worker when it runs.
        """
        data = parse_json_body(required=False)
        weights = opt_weight_map(
            data, "weights", allowed=DEFAULT_WEIGHTS, defaults=DEFAULT_WEIGHTS
        )
        if weights is not None:
            clusters.set_weights(weights)
        n = enqueue_clustering(job_queue, reason="manual")
        workers.wake(KIND_CLUSTER)
        return jsonify({"ok": True, "queued": n, "weights": clusters.weights()})

    @bp.route("/api/supervised/train", methods=["POST"])
    def api_supervised_train():
        """Queue one folder-supervised head training run.

        Library-wide and idempotent: the job dedupes on ``"library"``, so a
        double-click cannot queue two training runs. Skipped (too few labelled
        folders) is reported by the job's own detail, not here.
        """
        n = enqueue_supervised(job_queue, reason="manual")
        workers.wake(KIND_SUPERVISED)
        return jsonify({"ok": True, "queued": n})

    @bp.route("/api/clusters/track")
    def api_clusters_track():
        """Per-track cluster membership and top Discogs styles (inspection aid)."""
        path = request.args.get("path", "")
        result = clusters.get()
        if not path or result is None:
            return jsonify({"available": False})
        try:
            idx = result.paths.index(path)
        except ValueError:
            return jsonify({"available": False})

        levels = []
        for lv in result.levels:
            cid = int(lv.labels[idx])
            levels.append({
                "resolution": lv.resolution,
                "cluster": cid,
                "name": lv.names.get(cid, ""),
                "size": lv.sizes.get(cid, 0),
            })
        return jsonify({"available": True, "path": path, "levels": levels})

    return bp
