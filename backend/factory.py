"""Flask application factory — construct `app` and register domain blueprints."""

from __future__ import annotations

import threading
from collections.abc import Callable, MutableMapping
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from flask import Flask

from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.routes.api_folders import create_api_folders_blueprint
from backend.routes.api_library import create_api_library_blueprint
from backend.routes.api_tags import create_api_tags_blueprint
from backend.routes.static_audio import create_static_audio_blueprint


@dataclass
class TrackspaceBlueprintDeps:
    """Dependencies for registering all HTTP blueprints (passed from `app` wiring)."""

    # Embeddings
    embeddings_mod: Any
    audio_features_mod: Any
    embedding_version: int
    effnet_version: int
    features_version: int
    feature_cache: Any
    source_versions_cls: type
    batch_fetch_maps: Callable[..., Any]
    build_generation_work: Callable[..., Any]
    coverage_payload: Callable[..., Any]
    layout_revision_for_projection: Callable[..., Any]
    embed_lock: threading.Lock
    embed_status: dict[str, Any]
    status_cache: Any
    library_mtime_signature: Callable[[str, bool], str]
    track_infos_cached: Callable[[str, bool, str], list[dict[str, Any]]]
    projection_query_fingerprint: Callable[[Any], str]
    parse_projection_params_fn: Callable[..., Any]
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]]
    broadcast_embed_event: Callable[[dict[str, Any]], None]
    emit_decode_warning_once: Callable[[str, str, set[str]], None]
    broadcaster: Any
    is_generating: Callable[[], bool]
    sse_response: Callable[[Any], Any]
    # Library
    list_mp3s: Callable[[str, bool], list[str]]
    cached_read_all: Callable[[str], dict[str, Any]]
    executor: Executor
    read_paths_parallel: Callable[..., Any]
    iter_parallel_reads: Callable[..., Any]
    track_dict_from_read_service: Callable[..., Any]
    build_track_list_service: Callable[..., Any]
    emit_scan_progress_events: Callable[..., None]
    # Folders
    roots: MutableMapping[str, str]
    folder_tree: Callable[..., dict[str, Any]]
    add_root: Callable[[str], tuple[bool, str]]
    remove_root: Callable[[str], bool]
    schedule_root_watch: Callable[[str], None]
    unschedule_root_watch: Callable[[str], None]
    save_roots_state: Callable[[], None]
    track_cache: Any
    dir_color_fn: Callable[[str], str]
    # Path helpers (shared)
    resolve_virtual_path: Callable[[str], str]
    virtual_from_abs: Callable[[str], str]
    # Static / audio
    dist_dir: str


def create_app(*, template_folder: str, deps: TrackspaceBlueprintDeps) -> Flask:
    app = Flask(__name__, static_folder=None, template_folder=template_folder)
    app.register_blueprint(
        create_api_embeddings_blueprint(
            embeddings_mod=deps.embeddings_mod,
            audio_features_mod=deps.audio_features_mod,
            embedding_version=deps.embedding_version,
            effnet_version=deps.effnet_version,
            features_version=deps.features_version,
            feature_cache=deps.feature_cache,
            source_versions_cls=deps.source_versions_cls,
            batch_fetch_maps=deps.batch_fetch_maps,
            build_generation_work=deps.build_generation_work,
            coverage_payload=deps.coverage_payload,
            layout_revision_for_projection=deps.layout_revision_for_projection,
            embed_lock=deps.embed_lock,
            embed_status=deps.embed_status,
            status_cache=deps.status_cache,
            library_mtime_signature=deps.library_mtime_signature,
            track_infos_cached=deps.track_infos_cached,
            projection_query_fingerprint=deps.projection_query_fingerprint,
            parse_projection_params_fn=deps.parse_projection_params_fn,
            build_track_infos_fn=deps.build_track_infos_fn,
            resolve_virtual_path=deps.resolve_virtual_path,
            virtual_from_abs_path=deps.virtual_from_abs,
            broadcast_embed_event=deps.broadcast_embed_event,
            emit_decode_warning_once=deps.emit_decode_warning_once,
            broadcaster=deps.broadcaster,
            is_generating=deps.is_generating,
            sse_response=deps.sse_response,
        )
    )
    app.register_blueprint(
        create_api_library_blueprint(
            list_mp3s=deps.list_mp3s,
            cached_read_all=deps.cached_read_all,
            executor=deps.executor,
            feature_cache=deps.feature_cache,
            features_version=deps.features_version,
            audio_features_mod=deps.audio_features_mod,
            read_paths_parallel=deps.read_paths_parallel,
            iter_parallel_reads=deps.iter_parallel_reads,
            track_dict_from_read_service=deps.track_dict_from_read_service,
            build_track_list_service=deps.build_track_list_service,
            emit_scan_progress_events=deps.emit_scan_progress_events,
            sse_response=deps.sse_response,
            virtual_from_abs=deps.virtual_from_abs,
        )
    )
    app.register_blueprint(
        create_api_folders_blueprint(
            roots=deps.roots,
            folder_tree=deps.folder_tree,
            resolve_virtual_path=deps.resolve_virtual_path,
            virtual_from_abs=deps.virtual_from_abs,
            add_root=deps.add_root,
            remove_root=deps.remove_root,
            schedule_root_watch=deps.schedule_root_watch,
            unschedule_root_watch=deps.unschedule_root_watch,
            save_roots_state=deps.save_roots_state,
            track_cache=deps.track_cache,
            dir_color_fn=deps.dir_color_fn,
        )
    )
    app.register_blueprint(
        create_api_tags_blueprint(
            list_mp3s=deps.list_mp3s,
            resolve_virtual_path=deps.resolve_virtual_path,
        )
    )
    app.register_blueprint(
        create_static_audio_blueprint(
            dist_dir=deps.dist_dir,
            resolve_virtual_path=deps.resolve_virtual_path,
        )
    )
    return app
