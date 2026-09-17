from __future__ import annotations

import json as _json
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol

from flask import Blueprint, jsonify, request

from backend.embeddings import landmark_projection as landmark_mod
from backend.embeddings import supervised as supervised_mod
from backend.embeddings.projection_config import (
    FROZEN_SOURCE_WEIGHTS,
    REQUIRED_SOURCES,
)
from backend.jobs import (
    KIND_CLAP,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_STRUCTURE,
)
from backend.services.embedding_status_cache import EmbeddingStatusCache
from backend.services.request_validation import (
    one_of,
    opt_bool,
    opt_str,
    opt_str_list,
    parse_json_body,
)

log = logging.getLogger(__name__)

#: Per-track stages the coverage panel may retrigger individually. The
#: library-wide tail stages (supervise/cluster) are triggered by their own routes.
RETRIGGER_STAGES: tuple[str, ...] = (
    KIND_FEATURES,
    KIND_STRUCTURE,
    KIND_MAEST,
    KIND_CLAP,
)


class _Broadcaster(Protocol):
    def subscribe(self, *, maxsize: int = 2000) -> queue.Queue[dict[str, Any]]: ...
    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None: ...


def create_api_embeddings_blueprint(
    *,
    embeddings_mod: Any,
    embedding_version: int,
    features_version: int,
    feature_cache: Any,
    source_versions_cls: type,
    batch_fetch_maps: Callable[..., Any],
    coverage_payload: Callable[..., Any],
    layout_revision_for_projection: Callable[..., str | None],
    status_cache: EmbeddingStatusCache,
    library_mtime_signature: Callable[[str, bool], str],
    track_infos_cached: Callable[[str, bool, str], list[dict[str, Any]]],
    projection_query_fingerprint: Callable[[Any], str],
    parse_projection_params_fn: Callable[..., Any],
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
    resolve_virtual_path: Callable[[str], str],
    virtual_from_abs_path: Callable[[str], str],
    broadcast_embed_event: Callable[[dict[str, Any]], None],
    emit_decode_warning_once: Callable[[str, str, set[str]], None],
    broadcaster: _Broadcaster,
    sse_response: Callable[[Any], Any],
    enqueue_analysis: Callable[..., dict[str, int]],
    job_queue: Any,
    maest_mod: Any,
    rhythm_version: int,
    structure_version: int,
) -> Blueprint:
    bp = Blueprint("api_embeddings", __name__)

    @bp.route("/api/embeddings/stream")
    def api_embeddings_stream():
        q = broadcaster.subscribe(maxsize=2000)

        def generate():
            try:
                # "Is work running" is now a queue question, not a process flag.
                if not _analysis_progress()[0]:
                    yield "event: done\ndata: {}\n\n"
                    return
                while True:
                    try:
                        event = q.get(timeout=30)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    etype = event.get("type", "progress")
                    if etype == "drained":
                        # Workers drain per kind, so the first drained event
                        # (e.g. features) must not end the stream while MAEST
                        # is still running. Only a fully idle queue is terminal.
                        if _analysis_progress()[0]:
                            continue
                        yield "event: done\ndata: {}\n\n"
                        break
                    if etype == "progress":
                        # Per-job worker events carry no queue totals; merge in
                        # the aggregate the frontend renders as "done/total".
                        _running, progress = _analysis_progress()
                        if progress:
                            event = {**event, **progress}
                    yield f"event: {etype}\ndata: {_json.dumps(event)}\n\n"
            finally:
                broadcaster.unsubscribe(q)

        return sse_response(generate())

    def _versions() -> Any:
        return source_versions_cls(
            embedding_version,
            features_version,
            maest_mod.MAEST_VERSION,
            rhythm_version,
            supervised_mod.cache_version(feature_cache),
            structure_version,
        )

    def _version_tuple(versions: Any) -> tuple[int, ...]:
        """Cache generations that feed the layout revision hash."""
        return (
            versions.clap,
            versions.features,
            versions.maest,
            versions.rhythm,
            versions.supervised,
        )

    def _analysis_progress() -> tuple[bool, dict[str, int] | None]:
        """Generation state derived from the queue, not an in-process flag.

        Counting rows means the answer survives restarts and cannot get stuck
        ``True`` if a worker dies mid-batch.
        """
        by_kind = job_queue.stats_by_kind()
        done = 0
        total = 0
        for kind in (KIND_FEATURES, KIND_MAEST, KIND_CLAP):
            st = by_kind.get(kind)
            if st is None:
                continue
            done += st.finished
            total += st.total
        running = any(
            (by_kind.get(k).outstanding if by_kind.get(k) else 0) > 0
            for k in (KIND_FEATURES, KIND_MAEST, KIND_CLAP)
        )
        return running, ({"done": done, "total": total} if total else None)

    @bp.route("/api/embeddings/status")
    def api_embeddings_status():
        if request.args.get("models_only") == "1":
            running, progress = _analysis_progress()
            return jsonify({
                "model_ready": embeddings_mod.is_model_ready(),
                "maest_model_ready": maest_mod.is_model_ready(),
                "generating": running,
                "progress": progress if running else None,
            })

        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "1") == "1"
        lib_sig = library_mtime_signature(rel, recursive)
        infos = track_infos_cached(rel, recursive, lib_sig)

        versions = _versions()
        write_epoch = feature_cache.write_epoch()
        proj_fp = projection_query_fingerprint(request)

        running, progress = _analysis_progress()

        def _compute() -> tuple[dict[str, Any], str | None]:
            fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
            maps = batch_fetch_maps(feature_cache, fps, versions)
            proj = parse_projection_params_fn(request, default_when_no_method=False)
            return (
                coverage_payload(infos, maps, versions),
                layout_revision_for_projection(
                    infos, maps, proj, cache_versions=_version_tuple(versions)
                ),
            )

        layout_revision: str | None = None
        if running:
            # Coverage moves constantly during a run; caching it would serve
            # numbers that are already wrong by the time they are rendered.
            base, layout_revision = _compute()
        else:
            cov_key = (lib_sig, write_epoch, proj_fp)
            cached_cov = status_cache.get_status_coverage(cov_key)
            if cached_cov is not None:
                base, layout_revision = cached_cov
            else:
                base, layout_revision = _compute()
                status_cache.put_status_coverage(
                    cov_key, base=base, layout_revision=layout_revision
                )

        payload = {
            **base,
            "model_ready": embeddings_mod.is_model_ready(),
            "maest_model_ready": maest_mod.is_model_ready(),
            "maest_error": maest_mod.load_error(),
            "generating": running,
            "progress": progress if running else None,
        }
        if layout_revision is not None:
            payload["layout_revision"] = layout_revision
        return jsonify(payload)

    @bp.route("/api/embeddings/generate", methods=["POST"])
    def api_embeddings_generate():
        """Queue outstanding analysis work.

        Kept for backward compatibility with the existing frontend call site;
        the actual execution is owned by the durable job queue, so this is now
        an idempotent "make sure this work is scheduled" request rather than a
        "start a background thread" one. Repeated calls are harmless.

        ``stages`` optionally restricts the request to specific per-track stages
        (the coverage panel's per-stage retrigger); it defaults to every stage.
        """
        data = parse_json_body(required=False)
        folder = opt_str(data, "folder", "")
        recursive = opt_bool(data, "recursive", True)
        priority_paths = opt_str_list(data, "priority_paths")
        stages = opt_str_list(data, "stages")
        for stage in stages:
            one_of(stage, RETRIGGER_STAGES, field="stages")

        infos = build_track_infos_fn(folder, recursive)
        # ``paths`` restricts the request to specific tracks (the broken-files view's
        # per-row retry); an empty list means the whole folder.
        paths = opt_str_list(data, "paths")
        if paths:
            wanted = set(paths)
            infos = [t for t in infos if t.get("path") in wanted]
        queued = enqueue_analysis(
            infos=infos,
            priority_paths=priority_paths,
            force=opt_bool(data, "force", False),
            kinds=tuple(stages) or None,
        )
        return jsonify({"ok": True, "queued": queued})

    @bp.route("/api/analysis/problems")
    def api_analysis_problems():
        """Tracks that cannot be analysed: no fingerprint, or a terminal job failure.

        The coverage panel shows only *counts*; this is the actionable list behind them,
        so a damaged file can be found, re-encoded, or deleted.
        """
        infos = build_track_infos_fn("", True)
        unfingerprinted = [t["path"] for t in infos if not t.get("fingerprint")]
        by_path: dict[str, dict[str, Any]] = {}
        for f in job_queue.failures(None, limit=1000):
            entry = by_path.setdefault(
                f["key"], {"path": f["key"], "kinds": [], "errors": []}
            )
            if f["kind"] not in entry["kinds"]:
                entry["kinds"].append(f["kind"])
            if f.get("error") and f["error"] not in entry["errors"]:
                entry["errors"].append(f["error"])
        return jsonify(
            {
                "ok": True,
                "unfingerprinted": unfingerprinted,
                "failed": list(by_path.values()),
            }
        )

    @bp.route("/api/embeddings/projection")
    @bp.route("/api/embeddings/umap")  # backward compat alias
    def api_embeddings_projection():
        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "1") == "1"
        infos = build_track_infos_fn(rel, recursive)

        proj = parse_projection_params_fn(request, default_when_no_method=True)

        fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
        pversions = _versions()
        maps = batch_fetch_maps(feature_cache, fps, pversions)
        revision = layout_revision_for_projection(
            infos, maps, proj, cache_versions=_version_tuple(pversions)
        )

        with_jacobian = request.args.get("jacobian", "") == "1"

        def _layout_progress(phase: str, done: int = 0, total: int = 0) -> None:
            # The layout is a plain request, not a queued job, so its phases ride
            # the embed stream rather than the queue's stage_progress.
            broadcast_embed_event({
                "type": "layout_progress",
                "phase": phase,
                "done": int(done),
                "total": int(total),
            })

        weights0 = dict(proj.source_weights or FROZEN_SOURCE_WEIGHTS)
        positions = embeddings_mod.compute_projection(
            infos,
            feature_cache,
            version=embedding_version,
            method=proj.method,
            context_tags=proj.tag_names,
            context_folders=proj.explicit_folders if proj.explicit_folders else None,
            scale_folders=proj.scale_folders,
            folder_boost=proj.folder_boost,
            folder_depth_boost=proj.folder_depth_boost,
            sources=proj.sources,
            feature_mask=proj.feature_mask,
            features_blend=proj.features_blend,
            layout_revision=revision,
            source_weights=proj.source_weights or FROZEN_SOURCE_WEIGHTS,
            required_sources=REQUIRED_SOURCES,
            on_progress=_layout_progress,
        )

        if not with_jacobian:
            return jsonify({"positions": positions, "revision": revision})

        # The base layout is fetched and drawn first (above); the Jacobian is a
        # separate, slower pass. The landmark extension differentiates a *fixed*,
        # smooth function of the weights, so its derivative is exact rather than
        # refit noise (an independent refit lands in a different local optimum).
        _layout_progress("chart", 0, 0)
        model = landmark_mod.landmark_model_for(
            revision,
            infos,
            feature_cache,
            positions,
            sources=proj.sources,
            feature_mask=proj.feature_mask,
            features_blend=proj.features_blend,
            required_sources=REQUIRED_SOURCES,
            weights0=weights0,
        )
        if model is None:
            # Too few tracks to fit a chart: fall back to the refit finite
            # difference (correct for PCA, noisy for t-SNE).
            _positions, jacobian = embeddings_mod.compute_projection(
                infos,
                feature_cache,
                version=embedding_version,
                method=proj.method,
                context_tags=proj.tag_names,
                context_folders=proj.explicit_folders if proj.explicit_folders else None,
                scale_folders=proj.scale_folders,
                folder_boost=proj.folder_boost,
                folder_depth_boost=proj.folder_depth_boost,
                sources=proj.sources,
                feature_mask=proj.feature_mask,
                features_blend=proj.features_blend,
                layout_revision=revision,
                source_weights=proj.source_weights or FROZEN_SOURCE_WEIGHTS,
                required_sources=REQUIRED_SOURCES,
                with_jacobian=True,
            )
        else:
            _layout_progress("jacobian", 0, 0)
            jacobian = model.jacobian(weights0)
        return jsonify({
            "positions": positions,
            "revision": revision,
            "weights": weights0,
            "jacobian": {
                s: [[float(dx), float(dy)] for dx, dy in arr]
                for s, arr in jacobian.items()
            },
        })

    return bp
