from __future__ import annotations

import json as _json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

from flask import Blueprint, jsonify, request


def create_api_library_blueprint(
    *,
    list_mp3s: Callable[[str, bool], list[str]],
    cached_read_all: Callable[[str], dict[str, Any]],
    executor: Executor,
    feature_cache: Any,
    features_version: int,
    audio_features_mod: Any,
    read_paths_parallel: Callable[..., dict[str, dict[str, Any]]],
    iter_parallel_reads: Callable[..., Any],
    track_dict_from_read_service: Callable[..., dict[str, Any]],
    build_track_list_service: Callable[..., list[dict[str, Any]]],
    emit_scan_progress_events: Callable[..., None],
    sse_response: Callable[[Any], Any],
    virtual_from_abs: Callable[[str], str],
) -> Blueprint:
    bp = Blueprint("api_library", __name__)

    def _track_dict_from_read(
        path: str,
        info: dict[str, Any],
        *,
        feat_by_fp: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return track_dict_from_read_service(
            path,
            info,
            virtual_from_abs=virtual_from_abs,
            display_bpm_key=audio_features_mod.audio_features_display_bpm_key,
            get_audio_features=lambda fp: feature_cache.get_audio_features(fp, features_version),
            feat_by_fp=feat_by_fp,
        )

    def _build_track_list(paths: list[str], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return build_track_list_service(
            paths,
            results,
            track_dict_builder=_track_dict_from_read,
            get_all_audio_features=lambda fps: feature_cache.get_all_audio_features(fps, features_version),
        )

    @bp.route("/api/tracks")
    def api_tracks():
        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "0") == "1"
        paths = list_mp3s(rel, recursive)
        results = read_paths_parallel(paths, read_fn=cached_read_all, executor=executor)
        return jsonify(_build_track_list(paths, results))

    @bp.route("/api/library/stream")
    def api_library_stream():
        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "0") == "1"
        paths = list_mp3s(rel, recursive)
        total = len(paths)
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=total + 100)

        def _scan() -> None:
            emit_scan_progress_events(
                paths,
                read_fn=cached_read_all,
                executor=executor,
                track_from_read=_track_dict_from_read,
                emit=q.put_nowait,
            )

        threading.Thread(target=_scan, daemon=True).start()

        def generate():
            while True:
                try:
                    event = q.get(timeout=60)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                etype = event.get("type", "progress")
                yield f"event: {etype}\ndata: {_json.dumps(event)}\n\n"
                if etype == "done":
                    break

        return sse_response(generate())

    @bp.route("/api/tags")
    def api_tags():
        rel = request.args.get("folder", "")
        recursive = request.args.get("recursive", "1") == "1"
        paths = list_mp3s(rel, recursive)
        names: set[str] = set()
        for _, info in iter_parallel_reads(paths, read_fn=cached_read_all, executor=executor):
            names.update(info.get("tags", {}).keys())
        return jsonify(sorted(names))

    return bp
