#!/usr/bin/env python3
"""Trackspace – organise music in a linear tag space.  Flask backend."""

import hashlib
import os
import logging
import argparse
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from backend.process_limits import raise_nofile_limit

raise_nofile_limit()

from watchdog.observers import Observer

from backend.tags import read_all
from backend.fingerprint import compute_fingerprint
from backend import audio_features
from backend import embeddings
from backend.embeddings import maest as maest_mod
from backend.embeddings import rhythm_features as rhythm_mod
from backend.embeddings.clap import EMBEDDING_VERSION
from backend.embeddings.effnet import EFFNET_VERSION
from backend.embeddings.librosa_audio_features import FEATURES_VERSION
from backend.embeddings.maest import MAEST_VERSION
from backend.embeddings.rhythm_features import RHYTHM_VERSION
from backend.embeddings.coverage import (
    SourceVersions,
    batch_fetch_maps,
    coverage_payload,
)
from backend.jobs import (
    JobWorker,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_MAEST,
    KIND_SCAN,
)
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_clustering,
    enqueue_library_work,
    make_cluster_handler,
    make_features_handler,
    make_maest_handler,
    make_scan_handler,
)
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.library_scan import emit_scan_progress_events
from backend.services.roots_registry import (
    add_root as roots_add,
    load_roots_state as roots_load,
    remove_root as roots_remove,
    save_roots_state as roots_save,
)
from backend.services.parallel_reads import iter_parallel_reads, read_paths_parallel
from backend.services.projection_request import (
    ProjectionParams,
    embedding_projection_query_fingerprint,
    parse_projection_params,
)
from backend.services.track_payload import (
    build_track_list as build_track_list_service,
    track_dict_from_read as track_dict_from_read_service,
)
from backend.services.virtual_paths import (
    is_mp3 as is_mp3_service,
    list_mp3s as list_mp3s_service,
    resolve_virtual_path as resolve_virtual_path_service,
    root_and_rel_from_virtual as root_and_rel_from_virtual_service,
    virtual_from_abs as virtual_from_abs_service,
)
from backend.factory import TrackspaceBlueprintDeps, create_app
from backend.fs_watch import FilesystemWatchDeps, TrackspaceFilesystemHandler
from backend.sse import sse_response
from backend.trackspace_state import PKG_DIR, TrackspaceState

log = logging.getLogger(__name__)

state = TrackspaceState.create()


def _purge_stale_embedding_cache_rows() -> dict[str, int]:
    """NULL out cached blobs older than current version constants.

    Runs once when ``backend.app`` is imported so stale rows are cleared before
    any HTTP handler runs (not only when ``main()`` is invoked).
    """
    fc = state.feature_cache
    return {
        "clap": fc.purge_old_versions(EMBEDDING_VERSION),
        "effnet": fc.purge_old_effnet_versions(EFFNET_VERSION),
        "features": fc.purge_old_feature_versions(FEATURES_VERSION),
        "maest": fc.purge_old_maest_versions(MAEST_VERSION),
        "rhythm": fc.purge_old_rhythm_versions(RHYTHM_VERSION),
    }


_startup_purges = _purge_stale_embedding_cache_rows()


def _library_mtime_signature(folder_vpath: str, recursive: bool) -> str:
    """Cheap fingerprint of mp3 set under one virtual folder (or all roots)."""
    h = hashlib.sha256()
    for p in _list_mp3s(folder_vpath, recursive):
        rel = _virtual_from_abs(p)
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(b"\0")
        try:
            h.update(str(os.path.getmtime(p)).encode("ascii", errors="ignore"))
        except OSError:
            h.update(b"0")
        h.update(b"\0")
    return h.hexdigest()


def _embedding_projection_query_fingerprint(req) -> str:
    return embedding_projection_query_fingerprint(req)


def _track_infos_cached(folder_abs: str, recursive: bool, lib_sig: str) -> list[dict]:
    key = (folder_abs, recursive, lib_sig)
    cached = state.status_cache.get_track_infos(key)
    if cached is not None:
        return cached
    infos = _build_track_infos(folder_abs, recursive)
    state.status_cache.put_track_infos(key, infos)
    return infos


def _broadcast_embed_event(event: dict) -> None:
    state.embed_broadcaster.publish(event)


def _emit_decode_warning_once(rel_path: str, message: str, seen_paths: set[str]) -> None:
    """Notify SSE subscribers once per relative path per generation run."""
    if not message or rel_path in seen_paths:
        return
    seen_paths.add(rel_path)
    _broadcast_embed_event({
        "type": "decode_warning",
        "path": rel_path,
        "message": message,
    })

def _add_root(abs_path: str) -> tuple[bool, str]:
    return roots_add(abs_path, state.roots)


def _remove_root(root_id: str) -> bool:
    return roots_remove(root_id, state.roots)


def _schedule_root_watch(root_id: str) -> None:
    """Start watchdog monitoring for one root when observer is active."""
    if state.observer is None:
        return
    root_abs = state.roots.get(root_id)
    if not root_abs or root_id in state.root_watches:
        return
    try:
        state.root_watches[root_id] = state.observer.schedule(
            TrackspaceFilesystemHandler(FILESYSTEM_WATCH_DEPS),
            root_abs,
            recursive=True,
        )
    except Exception:
        log.exception("Watchdog: failed to schedule root watch for %s (%s)", root_id, root_abs)


def _unschedule_root_watch(root_id: str) -> None:
    """Stop watchdog monitoring for one root when observer is active."""
    watch = state.root_watches.pop(root_id, None)
    if state.observer is None or watch is None:
        return
    try:
        state.observer.unschedule(watch)
    except Exception:
        log.exception("Watchdog: failed to unschedule root watch for %s", root_id)


def _save_roots_state() -> None:
    roots_save(state.roots, state.roots_state_path)


def _load_roots_state(default_root: str | None) -> None:
    roots_load(state.roots, state.roots_state_path, default_root=default_root)


def _root_and_rel_from_virtual(vpath: str) -> tuple[str, str]:
    return root_and_rel_from_virtual_service(vpath, state.roots)


def _virtual_from_abs(abs_path: str) -> str:
    return virtual_from_abs_service(abs_path, state.roots)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(rel: str) -> str:
    return resolve_virtual_path_service(rel, state.roots)


def _is_mp3(name: str) -> bool:
    return is_mp3_service(name)


def _list_mp3s_under_abs(folder: str, recursive: bool = False) -> list[str]:
    """Return absolute paths of mp3 files in one absolute folder."""
    results = []
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if _is_mp3(f):
                    results.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and _is_mp3(f):
                results.append(full)
    return sorted(results)


def _list_mp3s(folder_vpath: str, recursive: bool = False) -> list[str]:
    return list_mp3s_service(folder_vpath, state.roots, recursive)


def _folder_tree(root: str, root_id: str, rel_prefix: str = "") -> dict:
    """Return a nested dict representing the subfolder tree under one root."""
    name = os.path.basename(root.rstrip(os.sep)) or root
    children = []
    try:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                child_rel = entry if not rel_prefix else f"{rel_prefix}/{entry}"
                children.append(_folder_tree(full, root_id, child_rel))
    except PermissionError:
        pass
    vpath = root_id if not rel_prefix else f"{root_id}/{rel_prefix}"
    return {"name": name, "path": vpath, "children": children, "root_id": root_id}


def _dir_color(folder_path: str) -> str:
    """Deterministic folder swatch colour (matches frontend dirColor())."""
    if not folder_path:
        return "hsl(350,60%,55%)"
    h = 0
    for ch in folder_path:
        h = (h * 31 + ord(ch)) & 0x3FFFF
    return f"hsl({h % 360},65%,60%)"


def _cached_read_all(path: str) -> dict:
    """Read ID3 tags + metadata + fingerprint for *path*, consulting cache first.

    Cache key is (path, mtime).  Any write to the file changes mtime, so stale
    entries are never returned — no explicit invalidation is needed.
    The chromaprint fingerprint is computed once and stored alongside tags.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return read_all(path)

    cached = state.track_cache.get(path, mtime)
    if cached is not None and cached.get("fingerprint"):
        return cached

    data = cached if cached is not None else read_all(path)
    if not data.get("fingerprint"):
        data["fingerprint"] = compute_fingerprint(path)
    if data.get("fingerprint") or cached is None:
        state.track_cache.put(path, mtime, data)
    return data


def _queue_track_analysis(rel_path: str) -> None:
    """Queue analysis for one track discovered by the filesystem watcher."""
    info = {"path": rel_path, "fingerprint": None}
    try:
        info["fingerprint"] = _cached_read_all(_resolve(rel_path)).get("fingerprint")
    except Exception:
        log.debug("watch enqueue: could not read %s", rel_path)
        return
    if not info["fingerprint"]:
        return
    _enqueue_analysis(infos=[info])


FILESYSTEM_WATCH_DEPS = FilesystemWatchDeps(
    track_cache=state.track_cache,
    cached_read_all=_cached_read_all,
    virtual_from_abs=_virtual_from_abs,
    is_mp3_path=_is_mp3,
    enqueue_track=_queue_track_analysis,
)


# ---------------------------------------------------------------------------
# API – Embeddings (helpers + blueprint)
# ---------------------------------------------------------------------------


def _parse_projection_params(
    req,
    *,
    default_when_no_method: bool = False,
) -> ProjectionParams | None:
    return parse_projection_params(req, default_when_no_method=default_when_no_method)


def _build_track_infos(folder_vpath: str, recursive: bool) -> list[dict]:
    """Build lightweight track info dicts with path + fingerprint for embedding ops."""
    paths = _list_mp3s(folder_vpath, recursive)
    infos = []
    for p, info in iter_parallel_reads(
        paths, read_fn=_cached_read_all, executor=state.thread_pool,
    ):
        rel_path = _virtual_from_abs(p)
        infos.append({
            "path": rel_path,
            "fingerprint": info.get("fingerprint"),
        })
    infos.sort(key=lambda t: t["path"])
    return infos


# ---------------------------------------------------------------------------
# Analysis pipeline — durable queue + background workers
# ---------------------------------------------------------------------------

ANALYSIS_DEPS = AnalysisDeps(
    feature_cache=state.feature_cache,
    track_cache=state.track_cache,
    cached_read_all=_cached_read_all,
    resolve_virtual_path=_resolve,
    virtual_from_abs=_virtual_from_abs,
    list_mp3s=_list_mp3s,
    maest_mod=maest_mod,
    rhythm_mod=rhythm_mod,
    audio_features_mod=audio_features,
    maest_version=MAEST_VERSION,
    rhythm_version=RHYTHM_VERSION,
    features_version=FEATURES_VERSION,
    clap_version=EMBEDDING_VERSION,
    effnet_version=EFFNET_VERSION,
    emit_decode_warning=lambda rel, msg: _broadcast_embed_event(
        {"type": "decode_warning", "path": rel, "message": msg}
    ),
)


def _on_job_event(event: dict) -> None:
    """Relay worker events to SSE subscribers."""
    state.embed_broadcaster.publish(event)


def _after_cluster_inputs_changed(event: dict) -> None:
    """Queue a re-cluster once a per-track stage finishes its backlog.

    Clustering is library-wide, so it is pointless to run it per track — it is
    triggered when a stage drains, and the queue's dedupe key collapses repeats.
    """
    if event.get("type") == "drained":
        enqueue_clustering(state.job_queue, reason=f"{event.get('kind')}-drained")
        state.workers.wake(KIND_CLUSTER)


def _job_event_sink(event: dict) -> None:
    _on_job_event(event)
    _after_cluster_inputs_changed(event)


def _enqueue_analysis(
    *,
    infos: list[dict],
    priority_paths: list[str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Queue per-track analysis and wake the workers."""
    queued = enqueue_library_work(
        state.job_queue,
        infos,
        deps=ANALYSIS_DEPS,
        kinds=(KIND_FEATURES, KIND_MAEST),
        priority_paths=priority_paths or [],
        force=force,
    )
    state.workers.wake()
    return queued


def _register_workers() -> None:
    """Create one worker per pipeline stage (threads start in ``main()``)."""
    state.workers.add(
        JobWorker(
            state.job_queue,
            KIND_SCAN,
            make_scan_handler(ANALYSIS_DEPS),
            batch_size=8,
            on_event=_job_event_sink,
        )
    )
    state.workers.add(
        JobWorker(
            state.job_queue,
            KIND_FEATURES,
            make_features_handler(ANALYSIS_DEPS),
            batch_size=2,
            on_event=_job_event_sink,
        )
    )
    state.workers.add(
        JobWorker(
            state.job_queue,
            KIND_MAEST,
            make_maest_handler(ANALYSIS_DEPS, batch_size=4),
            batch_size=4,
            on_event=_job_event_sink,
            # Idle until the model has finished loading rather than failing
            # every claimed job during the first minute after startup.
            gate=maest_mod.is_model_ready,
        )
    )
    state.workers.add(
        JobWorker(
            state.job_queue,
            KIND_CLUSTER,
            make_cluster_handler(
                ANALYSIS_DEPS,
                state.clusters,
                build_track_infos=_build_track_infos,
            ),
            batch_size=1,
            on_event=_on_job_event,
        )
    )


_register_workers()


app = create_app(
    template_folder=os.path.join(PKG_DIR, "templates"),
    deps=TrackspaceBlueprintDeps(
        embeddings_mod=embeddings,
        audio_features_mod=audio_features,
        embedding_version=EMBEDDING_VERSION,
        effnet_version=EFFNET_VERSION,
        features_version=FEATURES_VERSION,
        feature_cache=state.feature_cache,
        source_versions_cls=SourceVersions,
        batch_fetch_maps=batch_fetch_maps,
        coverage_payload=coverage_payload,
        layout_revision_for_projection=layout_revision_for_projection,
        status_cache=state.status_cache,
        library_mtime_signature=_library_mtime_signature,
        track_infos_cached=_track_infos_cached,
        projection_query_fingerprint=_embedding_projection_query_fingerprint,
        parse_projection_params_fn=_parse_projection_params,
        build_track_infos_fn=_build_track_infos,
        broadcast_embed_event=_broadcast_embed_event,
        emit_decode_warning_once=_emit_decode_warning_once,
        broadcaster=state.embed_broadcaster,
        sse_response=sse_response,
        list_mp3s=_list_mp3s,
        cached_read_all=_cached_read_all,
        executor=state.thread_pool,
        read_paths_parallel=read_paths_parallel,
        iter_parallel_reads=iter_parallel_reads,
        track_dict_from_read_service=track_dict_from_read_service,
        build_track_list_service=build_track_list_service,
        emit_scan_progress_events=emit_scan_progress_events,
        roots=state.roots,
        folder_tree=_folder_tree,
        add_root=_add_root,
        remove_root=_remove_root,
        schedule_root_watch=_schedule_root_watch,
        unschedule_root_watch=_unschedule_root_watch,
        save_roots_state=_save_roots_state,
        track_cache=state.track_cache,
        dir_color_fn=_dir_color,
        resolve_virtual_path=_resolve,
        virtual_from_abs=_virtual_from_abs,
        dist_dir=state.dist_dir,
        job_queue=state.job_queue,
        workers=state.workers,
        clusters=state.clusters,
        enqueue_library_work=enqueue_library_work,
        enqueue_clustering=enqueue_clustering,
        enqueue_analysis=_enqueue_analysis,
        analysis_deps=ANALYSIS_DEPS,
        maest_mod=maest_mod,
        rhythm_version=RHYTHM_VERSION,
    ),
)
app.extensions["trackspace_state"] = state


# ---------------------------------------------------------------------------
# Background maintenance
# ---------------------------------------------------------------------------

def _start_gc_thread(interval: int = 600) -> None:
    """Prune TrackCache entries whose files no longer exist, every *interval* seconds."""
    def _loop() -> None:
        while True:
            time.sleep(interval)
            pruned = state.track_cache.prune_missing()
            if pruned:
                log.info("GC: pruned %d stale cache entr%s", pruned, "y" if pruned == 1 else "ies")
    threading.Thread(target=_loop, daemon=True, name="trackspace-gc").start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trackspace server")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Optional initial music root (otherwise use persisted roots only; no implicit cwd)",
    )
    parser.add_argument("-p", "--port", type=int, default=5111)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-clap", action="store_true",
                        help="Skip eager CLAP model loading at startup")
    parser.add_argument("--no-effnet", action="store_true",
                        help="Skip eager EffNet model loading at startup")
    parser.add_argument("--no-maest", action="store_true",
                        help="Skip eager MAEST model loading at startup")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    cli_root = os.path.abspath(args.root) if args.root else None
    state.music_root = cli_root or ""
    _load_roots_state(cli_root)
    if state.roots:
        print(f"Trackspace serving roots: {', '.join(state.roots.values())}")
    else:
        print("Trackspace: no library roots — add folders in the sidebar (or pass a path on the CLI)")

    from backend.fingerprint import _FPCALC
    if _FPCALC:
        print(f"fpcalc: {_FPCALC}")
    else:
        print("WARNING: fpcalc not found — fingerprinting disabled. Install: brew install chromaprint")

    state.observer = Observer()
    state.observer.daemon = True
    state.observer.start()
    for rid in list(state.roots.keys()):
        _schedule_root_watch(rid)

    _start_gc_thread()

    pruned = state.track_cache.prune_missing()
    if pruned:
        print(f"Cache: pruned {pruned} stale entr{'y' if pruned == 1 else 'ies'}")
    _version_labels = {
        "clap": EMBEDDING_VERSION,
        "effnet": EFFNET_VERSION,
        "features": FEATURES_VERSION,
        "maest": MAEST_VERSION,
        "rhythm": RHYTHM_VERSION,
    }
    for name, n in _startup_purges.items():
        if n:
            print(
                f"{name}: purged {n} stale v<{_version_labels[name]} "
                f"entr{'y' if n == 1 else 'ies'} (on load)"
            )
    print(
        f"Versions: CLAP={EMBEDDING_VERSION} EffNet={EFFNET_VERSION} "
        f"Features={FEATURES_VERSION} MAEST={MAEST_VERSION} Rhythm={RHYTHM_VERSION}"
    )

    # Any jobs left 'running' by a previous process are reclaimed so the queue
    # resumes instead of stalling on leases owned by a dead worker.
    reclaimed = state.job_queue.reclaim_expired()
    if reclaimed:
        print(f"Queue: reclaimed {reclaimed} interrupted job(s) from a previous run")
    q_stats = state.job_queue.stats()
    if q_stats.outstanding:
        print(f"Queue: resuming {q_stats.outstanding} outstanding job(s)")

    state.workers.start_all()

    if not args.no_clap:
        def _bg_load_clap():
            print("Loading CLAP model in background (this may take a moment on first run)…")
            embeddings.load_model()
            print("CLAP model ready.")
        threading.Thread(target=_bg_load_clap, daemon=True).start()
    if not args.no_effnet:
        def _bg_load_effnet():
            print("Loading EffNet model in background…")
            audio_features.load_effnet()
            if audio_features.is_effnet_ready():
                print("EffNet model ready.")
            else:
                print("EffNet model not available (essentia-tensorflow may not be installed).")
        threading.Thread(target=_bg_load_effnet, daemon=True).start()
    if not args.no_maest:
        def _bg_load_maest():
            print("Loading MAEST model in background…")
            maest_mod.load_model()
            if maest_mod.is_model_ready():
                maest_mod.warmup()
                print("MAEST model ready.")
            else:
                print(f"MAEST not available: {maest_mod.load_error()}")
            # Workers gate on model readiness; nudge them once it flips.
            state.workers.wake(KIND_MAEST)
        threading.Thread(target=_bg_load_maest, daemon=True).start()

    def _bg_warm_audio_features():
        print("Warming audio feature pipeline…")
        audio_features.warmup_audio_features()
        rhythm_mod.warmup_rhythm_features()
        print("Audio feature pipeline warm.")
    threading.Thread(target=_bg_warm_audio_features, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
