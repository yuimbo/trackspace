from __future__ import annotations

import json as _json
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol

from flask import Blueprint, jsonify, request

from backend.embeddings.projection_config import (
    FROZEN_SOURCE_WEIGHTS,
    REQUIRED_SOURCES,
)
from backend.jobs import KIND_FEATURES, KIND_MAEST
from backend.services.embedding_status_cache import EmbeddingStatusCache

log = logging.getLogger(__name__)


class _Broadcaster(Protocol):
    def subscribe(self, *, maxsize: int = 2000) -> queue.Queue[dict[str, Any]]: ...
    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None: ...


def create_api_embeddings_blueprint(
    *,
    embeddings_mod: Any,
    audio_features_mod: Any,
    embedding_version: int,
    effnet_version: int,
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
                    yield f"event: {etype}\ndata: {_json.dumps(event)}\n\n"
                    if etype in ("done", "error"):
                        break
            finally:
                broadcaster.unsubscribe(q)

        return sse_response(generate())

    def _versions() -> Any:
        return source_versions_cls(
            embedding_version,
            effnet_version,
            features_version,
            maest_mod.MAEST_VERSION,
            rhythm_version,
        )

    def _version_tuple(versions: Any) -> tuple[int, ...]:
        """Cache generations that feed the layout revision hash."""
        return (
            versions.clap,
            versions.effnet,
            versions.features,
            versions.maest,
            versions.rhythm,
        )

    def _analysis_progress() -> tuple[bool, dict[str, int] | None]:
        """Generation state derived from the queue, not an in-process flag.

        Counting rows means the answer survives restarts and cannot get stuck
        ``True`` if a worker dies mid-batch.
        """
        by_kind = job_queue.stats_by_kind()
        done = 0
        total = 0
        for kind in (KIND_FEATURES, KIND_MAEST):
            st = by_kind.get(kind)
            if st is None:
                continue
            done += st.finished
            total += st.total
        running = any(
            (by_kind.get(k).outstanding if by_kind.get(k) else 0) > 0
            for k in (KIND_FEATURES, KIND_MAEST)
        )
        return running, ({"done": done, "total": total} if total else None)

    @bp.route("/api/embeddings/status")
    def api_embeddings_status():
        if request.args.get("models_only") == "1":
            running, progress = _analysis_progress()
            return jsonify({
                "model_ready": embeddings_mod.is_model_ready(),
                "effnet_model_ready": audio_features_mod.is_effnet_ready(),
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
            "effnet_model_ready": audio_features_mod.is_effnet_ready(),
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
        """
        data = request.get_json(silent=True) or {}
        folder = data.get("folder", "")
        recursive = bool(data.get("recursive", True))
        priority_paths: list[str] = data.get("priority_paths", []) or []

        infos = build_track_infos_fn(folder, recursive)
        queued = enqueue_analysis(
            infos=infos,
            priority_paths=priority_paths,
            force=bool(data.get("force", False)),
        )
        return jsonify({"ok": True, "queued": queued})

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
            source_weights=FROZEN_SOURCE_WEIGHTS,
            required_sources=REQUIRED_SOURCES,
        )

        return jsonify({"positions": positions, "revision": revision})

    return bp
