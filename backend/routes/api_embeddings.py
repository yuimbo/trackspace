from __future__ import annotations

import json as _json
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol

from flask import Blueprint, jsonify, request

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
    build_generation_work: Callable[..., Any],
    coverage_payload: Callable[..., Any],
    layout_revision_for_projection: Callable[..., str | None],
    embed_lock: threading.Lock,
    embed_status: dict[str, Any],
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
    is_generating: Callable[[], bool],
    sse_response: Callable[[Any], Any],
) -> Blueprint:
    bp = Blueprint("api_embeddings", __name__)

    @bp.route("/api/embeddings/stream")
    def api_embeddings_stream():
        q = broadcaster.subscribe(maxsize=2000)

        def generate():
            try:
                if not is_generating():
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

    @bp.route("/api/embeddings/status")
    def api_embeddings_status():
        if request.args.get("models_only") == "1":
            with embed_lock:
                running = embed_status["running"]
                progress_done = embed_status["done"]
                progress_total = embed_status["total"]
            return jsonify({
                "model_ready": embeddings_mod.is_model_ready(),
                "effnet_model_ready": audio_features_mod.is_effnet_ready(),
                "generating": running,
                "progress": {"done": progress_done, "total": progress_total} if running else None,
            })

        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "1") == "1"
        lib_sig = library_mtime_signature(rel, recursive)
        infos = track_infos_cached(rel, recursive, lib_sig)

        versions = source_versions_cls(embedding_version, effnet_version, features_version)
        write_epoch = feature_cache.write_epoch()
        proj_fp = projection_query_fingerprint(request)

        with embed_lock:
            running = embed_status["running"]
            progress_done = embed_status["done"]
            progress_total = embed_status["total"]

        layout_revision: str | None = None
        if running:
            fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
            maps = batch_fetch_maps(feature_cache, fps, versions)
            base = coverage_payload(infos, maps, versions)
            proj = parse_projection_params_fn(request, default_when_no_method=False)
            layout_revision = layout_revision_for_projection(
                infos,
                maps,
                proj,
                cache_versions=(versions.clap, versions.effnet, versions.features),
            )
        else:
            cov_key = (lib_sig, write_epoch, proj_fp)
            cached_cov = status_cache.get_status_coverage(cov_key)
            if cached_cov is not None:
                base, layout_revision = cached_cov
            else:
                base = None
            if base is None:
                fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
                maps = batch_fetch_maps(feature_cache, fps, versions)
                base = coverage_payload(infos, maps, versions)
                proj = parse_projection_params_fn(request, default_when_no_method=False)
                layout_revision = layout_revision_for_projection(
                    infos,
                    maps,
                    proj,
                    cache_versions=(versions.clap, versions.effnet, versions.features),
                )
                status_cache.put_status_coverage(
                    cov_key,
                    base=base,
                    layout_revision=layout_revision,
                )

        payload = {
            **base,
            "model_ready": embeddings_mod.is_model_ready(),
            "effnet_model_ready": audio_features_mod.is_effnet_ready(),
            "generating": running,
            "progress": {"done": progress_done, "total": progress_total} if running else None,
        }
        if layout_revision is not None:
            payload["layout_revision"] = layout_revision
        return jsonify(payload)

    @bp.route("/api/embeddings/generate", methods=["POST"])
    def api_embeddings_generate():
        data = request.get_json(force=True)
        rel = data.get("folder", "")
        recursive = data.get("recursive", True)
        priority_paths: list[str] = data.get("priority_paths", [])
        sources: list[str] = data.get("sources", ["clap"])
        folder = rel

        if "clap" in sources and not embeddings_mod.is_model_ready():
            return jsonify({"ok": False, "error": "CLAP model still loading, try again shortly"})
        if "effnet" in sources and not audio_features_mod.is_effnet_ready():
            return jsonify({"ok": False, "error": "EffNet model still loading, try again shortly"})

        with embed_lock:
            if embed_status["running"]:
                return jsonify({"ok": False, "error": "Generation already in progress"})
            embed_status["running"] = True
            embed_status["done"] = 0
            embed_status["total"] = 0
            embed_status["error"] = None

        def _run():
            try:
                decode_warn_seen: set[str] = set()

                def _note_decode(rel_path: str, msg: str) -> None:
                    emit_decode_warning_once(rel_path, msg, decode_warn_seen)

                infos = build_track_infos_fn(folder, recursive)
                vers = source_versions_cls(embedding_version, effnet_version, features_version)
                work = build_generation_work(infos, feature_cache, sources, vers)

                if priority_paths:
                    pset = set(priority_paths)
                    deprioritized = any(w[0]["path"] not in pset for w in work)
                    if deprioritized:
                        work.sort(key=lambda w: (0 if w[0]["path"] in pset else 1, w[0]["path"]))
                    else:
                        work.sort(key=lambda w: w[0]["path"])
                else:
                    work.sort(key=lambda w: w[0]["path"])

                with embed_lock:
                    embed_status["total"] = len(work)

                _EFFNET_CHUNK = 8
                done = 0
                for chunk_start in range(0, max(len(work), 1), _EFFNET_CHUNK):
                    chunk = work[chunk_start:chunk_start + _EFFNET_CHUNK]
                    if not chunk:
                        break

                    effnet_paths = [
                        resolve_virtual_path(t["path"])
                        for t, needed in chunk if "effnet" in needed
                    ]
                    effnet_decode_warns: dict[str, str] = {}
                    effnet_results = (
                        audio_features_mod.generate_effnet_embeddings_batch(
                            effnet_paths,
                            decode_warnings=effnet_decode_warns,
                        )
                        if effnet_paths
                        else {}
                    )
                    for abs_p, msg in effnet_decode_warns.items():
                        _note_decode(virtual_from_abs_path(abs_p), msg)
                    feature_paths = [
                        resolve_virtual_path(t["path"])
                        for t, needed in chunk if "features" in needed
                    ]
                    feat_decode_warns: dict[str, str] = {}
                    feature_results = (
                        audio_features_mod.generate_audio_features_batch(
                            feature_paths,
                            decode_warnings=feat_decode_warns,
                        )
                        if feature_paths
                        else {}
                    )
                    for abs_p, msg in feat_decode_warns.items():
                        _note_decode(virtual_from_abs_path(abs_p), msg)

                    for t, needed in chunk:
                        abs_path = resolve_virtual_path(t["path"])
                        fp = t["fingerprint"]
                        ok = True
                        failures: list[str] = []

                        if "clap" in needed:
                            track_rel = t["path"]
                            emb = embeddings_mod.generate_embedding(
                                abs_path,
                                on_decode_warning=lambda m, rp=track_rel: _note_decode(rp, m),
                            )
                            if emb is not None:
                                feature_cache.put_embedding(fp, emb, version=embedding_version)
                            else:
                                ok = False
                                failures.append("clap")

                        if "effnet" in needed:
                            emb = effnet_results.get(abs_path)
                            if emb is not None:
                                feature_cache.put_effnet_embedding(fp, emb, version=effnet_version)
                            else:
                                ok = False
                                failures.append("effnet")

                        if "features" in needed:
                            feat = feature_results.get(abs_path)
                            if feat is not None:
                                feature_cache.put_audio_features(fp, feat, version=features_version)
                            else:
                                ok = False
                                failures.append("features")

                        done += 1
                        with embed_lock:
                            embed_status["done"] = done
                        prog_evt: dict[str, Any] = {
                            "type": "progress",
                            "path": t["path"],
                            "ok": ok,
                            "done": done,
                            "total": len(work),
                        }
                        if failures:
                            prog_evt["failures"] = failures
                            log.warning(
                                "Embedding incomplete for %s (%s)",
                                t["path"],
                                ", ".join(failures),
                            )
                        broadcast_embed_event(prog_evt)

            except Exception as e:
                log.exception("Embedding generation failed")
                with embed_lock:
                    embed_status["error"] = str(e)
                broadcast_embed_event({"type": "error", "error": str(e)})
            finally:
                with embed_lock:
                    embed_status["running"] = False
                broadcast_embed_event({"type": "done"})

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True})

    @bp.route("/api/embeddings/projection")
    @bp.route("/api/embeddings/umap")  # backward compat alias
    def api_embeddings_projection():
        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "1") == "1"
        infos = build_track_infos_fn(rel, recursive)

        proj = parse_projection_params_fn(request, default_when_no_method=True)

        fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
        pversions = source_versions_cls(embedding_version, effnet_version, features_version)
        maps = batch_fetch_maps(feature_cache, fps, pversions)
        revision = layout_revision_for_projection(
            infos,
            maps,
            proj,
            cache_versions=(pversions.clap, pversions.effnet, pversions.features),
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
        )

        return jsonify({"positions": positions, "revision": revision})

    return bp
