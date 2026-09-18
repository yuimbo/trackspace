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

from flask import Flask

from backend.process_limits import raise_nofile_limit

raise_nofile_limit()

from watchdog.observers import Observer

from backend.tags import read_all
from backend.fingerprint import compute_fingerprint
from backend import audio_features
from backend import embeddings
from backend.embeddings import clap as clap_mod
from backend.embeddings import maest as maest_mod
from backend.embeddings import rhythm_features as rhythm_mod
from backend.embeddings import supervised as supervised_mod
from backend.embeddings.clap import EMBEDDING_VERSION
from backend.embeddings.librosa_audio_features import FEATURES_VERSION
from backend.embeddings.maest import MAEST_VERSION
from backend.embeddings.rhythm_features import RHYTHM_VERSION
from backend.embeddings.structure import STRUCTURE_VERSION
from backend.embeddings.coverage import (
    SourceVersions,
    batch_fetch_maps,
    coverage_payload,
)
from backend.jobs import (
    JobWorker,
    KIND_CLAP,
    KIND_CLUSTER,
    KIND_FEATURES,
    KIND_LAYOUT,
    KIND_MAEST,
    KIND_SCAN,
    KIND_STRUCTURE,
    KIND_SUPERVISED,
)
from backend.embeddings.projection_config import (
    PROJECTION_LENS_IDS,
    PROJECTION_LENSES,
)
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_clustering,
    enqueue_layouts,
    enqueue_library_work,
    enqueue_supervised,
    make_clap_handler,
    make_cluster_handler,
    make_features_handler,
    make_layout_handler,
    make_maest_handler,
    make_scan_handler,
    make_structure_handler,
    make_supervised_handler,
)
from backend.services.dir_colors import dir_color
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.lens_layout import compute_lens_projection
from backend.services.library_scan import emit_scan_progress_events
from backend.services.roots_registry import (
    add_root as roots_add,
    library_root_ids as roots_library_ids,
    load_roots_state as roots_load,
    remove_root as roots_remove,
    save_roots_state as roots_save,
    set_root_role as roots_set_role,
)
from backend.services.parallel_reads import iter_parallel_reads, read_paths_parallel
from backend.services.projection_request import (
    ProjectionParams,
    embedding_projection_query_fingerprint,
    parse_projection_params,
    projection_params_for_lens,
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

#: Versioned cache sources and the constant that owns each version. One map so the
#: startup purge and its report can never disagree about which sources exist.
SOURCE_VERSIONS: dict[str, int] = {
    "clap": EMBEDDING_VERSION,
    "clap_segments": EMBEDDING_VERSION,
    "clap_segment_pos": EMBEDDING_VERSION,
    "features": FEATURES_VERSION,
    "maest": MAEST_VERSION,
    "rhythm": RHYTHM_VERSION,
    "structure": STRUCTURE_VERSION,
}

#: Which job kind re-runs each versioned cache source, so a startup purge can
#: enqueue exactly the backfill it created. Several sources share one stage (CLAP
#: writes its pooled vector, segment vectors and positions in one pass).
PURGED_SOURCE_KINDS: dict[str, str] = {
    "clap": KIND_CLAP,
    "clap_segments": KIND_CLAP,
    "clap_segment_pos": KIND_CLAP,
    "features": KIND_FEATURES,
    "maest": KIND_MAEST,
    "rhythm": KIND_FEATURES,
    "structure": KIND_STRUCTURE,
}

#: Processes used to drain the CPU-bound ``features`` stage. madmom's BLSTM
#: forward pass is GIL-bound, so this stage scales across *processes*, not threads
#: (see ``backend/services/features_worker.py``). Measured on a 64-track real
#: sample: 51 → 134 tracks/min at 1 → 4 processes, with 8 adding little more.
#: Override with ``TRACKSPACE_FEATURES_WORKERS``.
DEFAULT_FEATURES_WORKERS = 4


#: Structure detection decodes with ``soundfile``, whose reads release the GIL, so
#: threads scale (measured ~2.1x at 4 workers on a real sample) without the spawn
#: cost the features process pool pays. Override with ``TRACKSPACE_STRUCTURE_WORKERS``.
DEFAULT_STRUCTURE_WORKERS = 4


def _features_worker_count() -> int:
    """``TRACKSPACE_FEATURES_WORKERS`` if valid, else the measured default."""
    raw = (os.environ.get("TRACKSPACE_FEATURES_WORKERS") or "").strip()
    if not raw:
        return DEFAULT_FEATURES_WORKERS
    try:
        n = int(raw)
    except ValueError:
        log.warning(
            "TRACKSPACE_FEATURES_WORKERS=%r is not an integer; using %d",
            raw,
            DEFAULT_FEATURES_WORKERS,
        )
        return DEFAULT_FEATURES_WORKERS
    if n < 1:
        log.warning(
            "TRACKSPACE_FEATURES_WORKERS=%d < 1; using %d",
            n,
            DEFAULT_FEATURES_WORKERS,
        )
        return DEFAULT_FEATURES_WORKERS
    return n


def _structure_worker_count() -> int:
    """``TRACKSPACE_STRUCTURE_WORKERS`` if valid, else the measured default."""
    raw = (os.environ.get("TRACKSPACE_STRUCTURE_WORKERS") or "").strip()
    if not raw:
        return DEFAULT_STRUCTURE_WORKERS
    try:
        n = int(raw)
    except ValueError:
        log.warning(
            "TRACKSPACE_STRUCTURE_WORKERS=%r is not an integer; using %d",
            raw,
            DEFAULT_STRUCTURE_WORKERS,
        )
        return DEFAULT_STRUCTURE_WORKERS
    if n < 1:
        log.warning(
            "TRACKSPACE_STRUCTURE_WORKERS=%d < 1; using %d",
            n,
            DEFAULT_STRUCTURE_WORKERS,
        )
        return DEFAULT_STRUCTURE_WORKERS
    return n


def _purge_stale_embedding_cache_rows(ts: TrackspaceState) -> dict[str, int]:
    """NULL out cached blobs older than current version constants.

    Called once per process before any HTTP handler serves requests. Importing this
    module must not touch the filesystem, so this is not run at import time.
    """
    fc = ts.feature_cache
    return {
        src: fc.purge_old_versions(src, ver)
        for src, ver in SOURCE_VERSIONS.items()
    }


def _kinds_for_purged_sources(purged: dict[str, int]) -> tuple[str, ...]:
    """Job kinds that rewrite the cache sources a purge just blanked, deduped.

    Several sources share one stage, so the map is folded in stage order; a source
    with nothing purged (or one not in the map) contributes no work.
    """
    return tuple(
        dict.fromkeys(
            PURGED_SOURCE_KINDS[src]
            for src, n in purged.items()
            if n and src in PURGED_SOURCE_KINDS
        )
    )


def build_app(state: TrackspaceState, *, clap_enabled: bool = True) -> Flask:
    """Wire helpers, workers, and blueprints around *state*; return the app.

    ``main()`` builds the state after CLI parsing and calls this, so importing
    ``backend.app`` has no side effects (no caches, thread pool, or purge) and
    tests can inject a scratch state.

    *clap_enabled* mirrors ``--no-clap``: when False the CLAP stage is never
    enqueued, so a disabled server cannot accumulate jobs that will never run.
    """
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
        state.root_roles.pop(root_id, None)
        return roots_remove(root_id, state.roots)


    def _set_root_role(root_id: str, role: str) -> None:
        roots_set_role(root_id, role, state.root_roles)


    def _library_root_ids() -> list[str]:
        """Virtual ids of roots that are part of "my library" (not external).

        The supervised head learns from these by default, so download/reference
        roots do not pollute the folder labels; they still get predictions.
        """
        return roots_library_ids(state.roots, state.root_roles)


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
        roots_save(
            state.roots,
            state.roots_state_path,
            temp_roots=state.temp_roots,
            root_roles=state.root_roles,
        )


    def _load_roots_state(default_root: str | None) -> None:
        roots_load(
            state.roots,
            state.roots_state_path,
            default_root=default_root,
            temp_roots=state.temp_roots,
            root_roles=state.root_roles,
        )


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


    def _list_mp3s(
        folder_vpath: str, recursive: bool = False, *, root_ids=None
    ) -> list[str]:
        return list_mp3s_service(folder_vpath, state.roots, recursive, root_ids=root_ids)


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


    def _build_track_infos(
        folder_vpath: str, recursive: bool, *, include_external: bool = True
    ) -> list[dict]:
        """Build lightweight track info dicts with path + fingerprint for embedding ops.

        *include_external* False scopes an all-roots walk to the library roots
        only, so external tracks do not enter the default projection (they are
        still analysed; the External toggle requests them explicitly).
        """
        root_ids = None if include_external else _library_root_ids()
        paths = _list_mp3s(folder_vpath, recursive, root_ids=root_ids)
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
    # Layout prefetch — warm every lens's projection in the durable queue
    # ---------------------------------------------------------------------------

    #: One-entry cache so the five lens jobs of a prefetch batch share one walk.
    _prefetch_infos_cache: dict[str, list[dict]] = {}


    def _prefetch_infos() -> list[dict]:
        sig = _library_mtime_signature("", True)
        cached = _prefetch_infos_cache.get(sig)
        if cached is not None:
            return cached
        infos = _build_track_infos("", True, include_external=False)
        _prefetch_infos_cache.clear()
        _prefetch_infos_cache[sig] = infos
        return infos


    def _prefetch_lens(lens_id: str) -> int:
        """Warm one lens's projection so a later switch is a cache hit.

        Runs through the durable ``layout`` queue after the cluster pass. The
        revision is computed exactly as the HTTP route does, so the warmed entry is
        the one the client will request; an already-current lens is a cache hit
        inside ``compute_projection`` and costs no fit.
        """
        lens = next((l for l in PROJECTION_LENSES if l.id == lens_id), None)
        if lens is None:
            raise ValueError(f"unknown lens {lens_id!r}")
        proj = projection_params_for_lens(lens.sources, lens.weights)
        positions, _revision = compute_lens_projection(
            _prefetch_infos(),
            state.feature_cache,
            proj,
            embeddings_mod=embeddings,
            batch_fetch_maps=batch_fetch_maps,
            layout_revision_for_projection=layout_revision_for_projection,
            source_versions=SourceVersions(
                clap=EMBEDDING_VERSION,
                features=FEATURES_VERSION,
                maest=MAEST_VERSION,
                rhythm=RHYTHM_VERSION,
                supervised=supervised_mod.cache_version(state.feature_cache),
                structure=STRUCTURE_VERSION,
            ),
            embedding_version=EMBEDDING_VERSION,
            user=False,
        )
        return len(positions)


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
        clap_mod=clap_mod,
        rhythm_mod=rhythm_mod,
        audio_features_mod=audio_features,
        maest_version=MAEST_VERSION,
        rhythm_version=RHYTHM_VERSION,
        features_version=FEATURES_VERSION,
        clap_version=EMBEDDING_VERSION,
        structure_version=STRUCTURE_VERSION,
        clap_enabled=clap_enabled,
        data_dir=state.data_dir,
        emit_decode_warning=lambda rel, msg: _broadcast_embed_event(
            {"type": "decode_warning", "path": rel, "message": msg}
        ),
    )


    def _on_job_event(event: dict) -> None:
        """Relay worker events to SSE subscribers."""
        state.embed_broadcaster.publish(event)


    def _after_cluster_inputs_changed(event: dict) -> None:
        """Queue library-wide follow-up work once a per-track stage drains.

        Clustering is library-wide, so it is pointless to run it per track — it is
        triggered when a stage drains, and the queue's dedupe key collapses repeats.

        The first full MAEST pass also auto-queues one supervised training run, but
        only while no head exists: after that a retrain is an explicit user action
        (folder reorganisation is not visible to the watcher).

        Once the cluster pass lands, the per-lens layouts are warmed so switching a
        lens is instant. This returns instead of re-enqueuing clustering: a cluster
        drain must not feed another cluster pass.
        """
        if event.get("type") != "drained":
            return
        kind = event.get("kind")
        if kind == KIND_CLUSTER:
            enqueue_layouts(
                state.job_queue, PROJECTION_LENS_IDS, reason="cluster-drained"
            )
            state.workers.wake(KIND_LAYOUT)
            return
        if (
            kind == KIND_MAEST
            and supervised_mod.current_run_id(state.data_dir) == 0
        ):
            enqueue_supervised(state.job_queue, reason="first-maest-drain")
            state.workers.wake(KIND_SUPERVISED)
        enqueue_clustering(state.job_queue, reason=f"{kind}-drained")
        state.workers.wake(KIND_CLUSTER)


    def _job_event_sink(event: dict) -> None:
        _on_job_event(event)
        _after_cluster_inputs_changed(event)


    def _stage_emit(kind: str):
        """Phase-progress sink for a library-wide singleton stage.

        Records the phase on the queue (so a status snapshot mid-run is correct)
        and publishes it on the same SSE channel as the worker events.
        """

        def _emit(phase: str, done: int = 0, total: int = 0) -> None:
            state.job_queue.note_stage_progress(
                kind, phase=phase, done=done, total=total
            )
            _job_event_sink(
                {
                    "type": "stage_progress",
                    "kind": kind,
                    "phase": phase,
                    "done": int(done),
                    "total": int(total),
                }
            )

        return _emit


    def _enqueue_analysis(
        *,
        infos: list[dict],
        priority_paths: list[str] | None = None,
        force: bool = False,
        kinds: tuple[str, ...] | None = None,
        revive_failed: bool = False,
    ) -> dict[str, int]:
        """Queue per-track analysis and wake the workers.

        *kinds* restricts the request to specific stages (the coverage panel's
        per-stage retrigger); ``None`` means every per-track stage.
        *revive_failed* also revives terminally-failed jobs — explicit user
        retriggers and the version-purge backfill set it; the frontend's automatic
        keep-warm generate does not.
        """
        queued = enqueue_library_work(
            state.job_queue,
            infos,
            deps=ANALYSIS_DEPS,
            kinds=kinds or (KIND_FEATURES, KIND_STRUCTURE, KIND_MAEST, KIND_CLAP),
            priority_paths=priority_paths or [],
            force=force,
            revive_failed=revive_failed,
        )
        state.workers.wake()
        return queued


    def _backfill_purged_analysis(purged: dict[str, int]) -> dict[str, int]:
        """Re-queue the per-track work a startup version purge just blanked.

        The purge blanks stale cache rows, but the durable queue still holds their
        jobs in a terminal state, so an ordinary enqueue would skip them. Scoping the
        backfill to the sources that were *actually* purged keeps a standing schema
        backlog (e.g. CLAP segment vectors added later) out of an unrelated restart
        — that one is retriggered explicitly from the coverage panel. Once the
        purged work drains this enqueues nothing.
        """
        kinds = _kinds_for_purged_sources(purged)
        if not kinds:
            return {}
        return _enqueue_analysis(
            infos=_build_track_infos("", True), kinds=kinds, revive_failed=True
        )


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
        features_workers = _features_worker_count()
        state.workers.add(
            JobWorker(
                state.job_queue,
                KIND_FEATURES,
                make_features_handler(ANALYSIS_DEPS, pool_size=features_workers),
                # Claim enough to keep every pool process busy in one batch.
                batch_size=max(2, features_workers),
                on_event=_job_event_sink,
            )
        )
        structure_workers = _structure_worker_count()
        state.workers.add(
            JobWorker(
                state.job_queue,
                KIND_STRUCTURE,
                make_structure_handler(ANALYSIS_DEPS, pool_size=structure_workers),
                # Claim enough to keep every pool thread busy in one batch.
                batch_size=max(2, structure_workers),
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
                KIND_CLAP,
                make_clap_handler(ANALYSIS_DEPS, batch_size=4),
                batch_size=4,
                on_event=_job_event_sink,
                # Idle until the model has finished loading rather than failing
                # every claimed job during the first minute after startup.
                gate=clap_mod.is_model_ready,
            )
        )
        state.workers.add(
            JobWorker(
                state.job_queue,
                KIND_SUPERVISED,
                make_supervised_handler(
                    ANALYSIS_DEPS,
                    build_track_infos=_build_track_infos,
                    emit=_stage_emit(KIND_SUPERVISED),
                    # Labels come from "my library" roots only by default; external
                    # roots still receive predictions from the trained head.
                    default_folders=_library_root_ids,
                ),
                batch_size=1,
                # Drain hook: a completed training run rewrites the supervised
                # rows, so the cluster pass must follow to use them.
                on_event=_job_event_sink,
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
                    emit=_stage_emit(KIND_CLUSTER),
                ),
                batch_size=1,
                # Drain hook: a completed cluster pass is the last consumer of
                # the cached vectors, so it warms the per-lens layouts.
                on_event=_job_event_sink,
            )
        )
        state.workers.add(
            JobWorker(
                state.job_queue,
                KIND_LAYOUT,
                make_layout_handler(
                    _prefetch_lens,
                    emit=_stage_emit(KIND_LAYOUT),
                ),
                batch_size=1,
                # No drain hook: a layout drain must not re-enqueue clustering.
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
            library_root_ids=_library_root_ids,
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
            temp_roots=state.temp_roots,
            root_roles=state.root_roles,
            folder_tree=_folder_tree,
            add_root=_add_root,
            remove_root=_remove_root,
            set_root_role=_set_root_role,
            schedule_root_watch=_schedule_root_watch,
            unschedule_root_watch=_unschedule_root_watch,
            save_roots_state=_save_roots_state,
            track_cache=state.track_cache,
            dir_color_fn=dir_color,
            resolve_virtual_path=_resolve,
            virtual_from_abs=_virtual_from_abs,
            dist_dir=state.dist_dir,
            job_queue=state.job_queue,
            workers=state.workers,
            clusters=state.clusters,
            similarity_index=state.similarity,
            enqueue_library_work=enqueue_library_work,
            enqueue_clustering=enqueue_clustering,
            enqueue_supervised=enqueue_supervised,
            enqueue_analysis=_enqueue_analysis,
            analysis_deps=ANALYSIS_DEPS,
            maest_mod=maest_mod,
            rhythm_version=RHYTHM_VERSION,
            structure_version=STRUCTURE_VERSION,
        ),
    )
    app.extensions["trackspace_state"] = state
    # Lifecycle hooks ``main()`` runs after CLI parsing. They close over the
    # state and watch deps, so expose them rather than leaking a module global.
    app.extensions["trackspace_lifecycle"] = {
        "load_roots_state": _load_roots_state,
        "schedule_root_watch": _schedule_root_watch,
        "backfill_purged_analysis": _backfill_purged_analysis,
    }
    return app


# ---------------------------------------------------------------------------
# Background maintenance
# ---------------------------------------------------------------------------

def _start_gc_thread(state: TrackspaceState, interval: int = 600) -> None:
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
    parser.add_argument("--no-maest", action="store_true",
                        help="Skip eager MAEST model loading at startup")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Build the process state *after* CLI parsing (not at import time) so that
    # importing ``backend.app`` has no side effects, and purge stale cache rows
    # before the app starts serving.
    state = TrackspaceState.create()
    app = build_app(state, clap_enabled=not args.no_clap)
    lifecycle = app.extensions["trackspace_lifecycle"]
    startup_purges = _purge_stale_embedding_cache_rows(state)

    cli_root = os.path.abspath(args.root) if args.root else None
    state.music_root = cli_root or ""
    lifecycle["load_roots_state"](cli_root)
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
        lifecycle["schedule_root_watch"](rid)

    _start_gc_thread(state)

    pruned = state.track_cache.prune_missing()
    if pruned:
        print(f"Cache: pruned {pruned} stale entr{'y' if pruned == 1 else 'ies'}")
    for name, n in startup_purges.items():
        if n:
            print(
                f"{name}: purged {n} stale v<{SOURCE_VERSIONS.get(name, '?')} "
                f"entr{'y' if n == 1 else 'ies'} (on load)"
            )
    print(
        f"Versions: CLAP={EMBEDDING_VERSION} "
        f"Features={FEATURES_VERSION} MAEST={MAEST_VERSION} "
        f"Rhythm={RHYTHM_VERSION} Structure={STRUCTURE_VERSION}"
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

    # The cluster store is in-memory, so a restart with an already-analysed library
    # has no drain event to trigger a re-cluster and `/api/clusters` would report
    # nothing until the user pressed "Rebuild clusters". Self-heal one run here; the
    # queue's single "library" dedupe key collapses it with any drain or manual
    # request, and a library with no MAEST rows is a cheap no-op.
    enqueue_clustering(state.job_queue, reason="startup")
    state.workers.wake(KIND_CLUSTER)

    def _bg_backfill_analysis():
        """Self-heal a source-version bump: re-queue what the purge just blanked."""
        try:
            queued = lifecycle["backfill_purged_analysis"](startup_purges)
            outstanding = sum(int(v) for v in queued.values())
            if outstanding:
                detail = ", ".join(f"{k}={v}" for k, v in sorted(queued.items()))
                print(f"Backfill: queued {outstanding} purged per-track job(s) ({detail})")
        except Exception:
            log.exception("startup analysis backfill failed")
    threading.Thread(
        target=_bg_backfill_analysis, daemon=True, name="trackspace-backfill"
    ).start()

    if not args.no_clap:
        def _bg_load_clap():
            print("Loading CLAP model in background (this may take a moment on first run)…")
            embeddings.load_model()
            if clap_mod.is_model_ready():
                print("CLAP model ready.")
            else:
                print(f"CLAP not available: {clap_mod.load_error()}")
            # Workers gate on model readiness; nudge them once it flips.
            state.workers.wake(KIND_CLAP)
        threading.Thread(target=_bg_load_clap, daemon=True).start()
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
