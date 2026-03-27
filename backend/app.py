#!/usr/bin/env python3
"""Trackspace – organise music in a linear tag space.  Flask backend."""

import hashlib
import os
import sys
import subprocess
import json as _json
import logging
import argparse
import threading
import queue
import time
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from backend.process_limits import raise_nofile_limit

raise_nofile_limit()

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    send_from_directory,
    render_template,
    abort,
)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent, DirMovedEvent

from backend.tags import write_tag, delete_tag, rename_tag, read_all
from backend.cache import TrackCache
from backend.fingerprint import compute_fingerprint
from backend.feature_cache import FeatureCache
from backend import embeddings
from backend import audio_features
from backend.embeddings import EMBEDDING_VERSION
from backend.audio_features import EFFNET_VERSION, FEATURES_VERSION
from backend.embedding_coverage import (
    SourceVersions,
    batch_fetch_maps,
    build_generation_work,
    coverage_payload,
)
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.embedding_broadcaster import EmbeddingBroadcaster
from backend.services.embedding_status_cache import EmbeddingStatusCache
from backend.services.library_scan import emit_scan_progress_events
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
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.sse import sse_response

log = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)

_THREAD_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)
_CACHE = TrackCache(os.path.join(_DATA_DIR, "track_cache.db"))
_FEATURES = FeatureCache(os.path.join(_DATA_DIR, "audio_features.db"))

class _FilesystemHandler(FileSystemEventHandler):
    """React to filesystem changes under MUSIC_ROOT.

    * Moved files/folders  → remap TrackCache keys.
    * New MP3 files        → pre-warm tag cache and (if CLAP is ready)
                             generate a CLAP embedding in the background.
    """

    # How long to wait after a FileCreatedEvent before reading, to let
    # large file copies finish writing.  We poll size until stable.
    _SETTLE_SECS = 3.0
    _SETTLE_POLLS = 3

    def on_moved(self, event) -> None:
        if isinstance(event, DirMovedEvent):
            old_prefix = event.src_path.rstrip(os.sep) + os.sep
            new_prefix = event.dest_path.rstrip(os.sep) + os.sep
            _CACHE.remap_prefix(old_prefix, new_prefix)
        elif isinstance(event, FileMovedEvent):
            _CACHE.remap(event.src_path, event.dest_path)
            # Treat intra-library moves of MP3s the same as new arrivals so
            # embeddings are generated if they were missing at the source.
            if _is_mp3(event.dest_path):
                threading.Thread(
                    target=self._ingest_new,
                    args=(event.dest_path,),
                    daemon=True,
                ).start()

    def on_created(self, event) -> None:
        if isinstance(event, FileCreatedEvent) and _is_mp3(event.src_path):
            threading.Thread(
                target=self._ingest_new,
                args=(event.src_path,),
                daemon=True,
            ).start()

    def _ingest_new(self, abs_path: str) -> None:
        """Wait for the file to finish writing, read tags, then embed."""
        # Poll until file size is stable (handles slow copies).
        prev_size = -1
        for _ in range(self._SETTLE_POLLS):
            time.sleep(self._SETTLE_SECS / self._SETTLE_POLLS)
            try:
                cur_size = os.path.getsize(abs_path)
            except OSError:
                return  # file disappeared
            if cur_size == prev_size:
                break
            prev_size = cur_size

        if not os.path.isfile(abs_path):
            return

        try:
            data = _cached_read_all(abs_path)
        except Exception:
            log.exception("Watchdog: failed to read new file %s", abs_path)
            return

        rel_path = _virtual_from_abs(abs_path)
        log.info("Watchdog: ingested %s", rel_path)

        fp = data.get("fingerprint")
        if not fp or not embeddings.is_model_ready():
            return
        if _FEATURES.has_embedding(fp, version=EMBEDDING_VERSION):
            return

        with _embed_lock:
            if _embed_status["running"]:
                # A bulk generation is in flight; the new file will be
                # included when the user next triggers embedding mode.
                return
            _embed_status.update({"running": True, "done": 0, "total": 1, "error": None})

        def _run() -> None:
            try:
                warn_seen: set[str] = set()
                emb = embeddings.generate_embedding(
                    abs_path,
                    on_decode_warning=lambda m: _emit_decode_warning_once(
                        rel_path, m, warn_seen,
                    ),
                )
                ok = emb is not None
                if ok:
                    _FEATURES.put_embedding(fp, emb, version=EMBEDDING_VERSION)
                evt: dict = {
                    "type": "progress",
                    "path": rel_path,
                    "ok": ok,
                    "done": 1,
                    "total": 1,
                }
                if not ok:
                    evt["failures"] = ["clap"]
                    log.warning("Watchdog: CLAP embedding failed for %s", rel_path)
                _broadcast_embed_event(evt)
                log.info("Watchdog: embedding %s for %s", "ok" if ok else "failed", rel_path)
            except Exception as e:
                log.exception("Watchdog: embedding generation failed for %s", rel_path)
                with _embed_lock:
                    _embed_status["error"] = str(e)
                _broadcast_embed_event({"type": "error", "error": str(e)})
            finally:
                with _embed_lock:
                    _embed_status["running"] = False
                _broadcast_embed_event({"type": "done"})

        threading.Thread(target=_run, daemon=True).start()


_embed_status: dict = {"running": False, "done": 0, "total": 0, "error": None}
_embed_lock = threading.Lock()

# SSE broadcast: each connected EventSource client gets its own Queue.
_embed_broadcaster = EmbeddingBroadcaster()

# Embedding /status caches (see api_embeddings_status).
_status_cache = EmbeddingStatusCache(track_infos_max=8, status_coverage_max=48)


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
    cached = _status_cache.get_track_infos(key)
    if cached is not None:
        return cached
    infos = _build_track_infos(folder_abs, recursive)
    _status_cache.put_track_infos(key, infos)
    return infos


def _broadcast_embed_event(event: dict) -> None:
    _embed_broadcaster.publish(event)


def _embed_is_running() -> bool:
    with _embed_lock:
        return bool(_embed_status["running"])


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

DIST_DIR = os.path.join(_PROJECT_DIR, "frontend", "dist")

app = Flask(__name__, static_folder=None, template_folder=os.path.join(_PKG_DIR, "templates"))

MUSIC_ROOT: str = ""  # initial CLI root
ROOTS: "OrderedDict[str, str]" = OrderedDict()
_ROOTS_STATE_PATH = os.path.join(_DATA_DIR, "roots.json")
_OBSERVER: Observer | None = None
_ROOT_WATCHES: dict[str, object] = {}


def _stable_root_id(abs_path: str) -> str:
    base = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:8]
    rid = base
    i = 1
    while rid in ROOTS and ROOTS[rid] != abs_path:
        rid = f"{base[:6]}{i:02d}"
        i += 1
    return rid


def _add_root(abs_path: str) -> tuple[bool, str]:
    """Insert a root folder and merge redundant descendants."""
    p = os.path.abspath(abs_path)
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    for rid, root_abs in list(ROOTS.items()):
        if p == root_abs or p.startswith(root_abs.rstrip(os.sep) + os.sep):
            return False, rid
    for rid, root_abs in list(ROOTS.items()):
        if root_abs.startswith(p.rstrip(os.sep) + os.sep):
            del ROOTS[rid]
    rid = _stable_root_id(p)
    ROOTS[rid] = p
    return True, rid


def _remove_root(root_id: str) -> bool:
    if root_id in ROOTS:
        del ROOTS[root_id]
        return True
    return False


def _schedule_root_watch(root_id: str) -> None:
    """Start watchdog monitoring for one root when observer is active."""
    if _OBSERVER is None:
        return
    root_abs = ROOTS.get(root_id)
    if not root_abs or root_id in _ROOT_WATCHES:
        return
    try:
        _ROOT_WATCHES[root_id] = _OBSERVER.schedule(_FilesystemHandler(), root_abs, recursive=True)
    except Exception:
        log.exception("Watchdog: failed to schedule root watch for %s (%s)", root_id, root_abs)


def _unschedule_root_watch(root_id: str) -> None:
    """Stop watchdog monitoring for one root when observer is active."""
    watch = _ROOT_WATCHES.pop(root_id, None)
    if _OBSERVER is None or watch is None:
        return
    try:
        _OBSERVER.unschedule(watch)
    except Exception:
        log.exception("Watchdog: failed to unschedule root watch for %s", root_id)


def _save_roots_state() -> None:
    """Persist active roots so they survive process restarts."""
    payload = {"roots": list(ROOTS.values())}
    with open(_ROOTS_STATE_PATH, "w", encoding="utf-8") as f:
        _json.dump(payload, f)


def _load_roots_state(default_root: str | None) -> None:
    """Load persisted roots.

    If the JSON is missing/empty and *default_root* is set, use that single root.
    If *default_root* is ``None`` (no CLI path / env), start with no roots until
    the user adds one in the UI — do **not** impute cwd.
    """
    ROOTS.clear()
    loaded_any = False
    try:
        with open(_ROOTS_STATE_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        for raw in data.get("roots", []):
            p = os.path.abspath(str(raw))
            if os.path.isdir(p):
                _add_root(p)
                loaded_any = True
    except Exception:
        loaded_any = False

    if not loaded_any and default_root is not None:
        _add_root(default_root)

    # Normalize persisted state (drops deleted/nonexistent paths, merged descendants).
    _save_roots_state()


def _root_and_rel_from_virtual(vpath: str) -> tuple[str, str]:
    return root_and_rel_from_virtual_service(vpath, ROOTS)


def _virtual_from_abs(abs_path: str) -> str:
    return virtual_from_abs_service(abs_path, ROOTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(rel: str) -> str:
    return resolve_virtual_path_service(rel, ROOTS)


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
    return list_mp3s_service(folder_vpath, ROOTS, recursive)


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

    cached = _CACHE.get(path, mtime)
    if cached is not None and cached.get("fingerprint"):
        return cached

    data = cached if cached is not None else read_all(path)
    if not data.get("fingerprint"):
        data["fingerprint"] = compute_fingerprint(path)
    if data.get("fingerprint") or cached is None:
        _CACHE.put(path, mtime, data)
    return data


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
    for p, info in iter_parallel_reads(paths, read_fn=_cached_read_all, executor=_THREAD_POOL):
        rel_path = _virtual_from_abs(p)
        infos.append({
            "path": rel_path,
            "fingerprint": info.get("fingerprint"),
        })
    infos.sort(key=lambda t: t["path"])
    return infos


app.register_blueprint(
    create_api_embeddings_blueprint(
        embeddings_mod=embeddings,
        audio_features_mod=audio_features,
        embedding_version=EMBEDDING_VERSION,
        effnet_version=EFFNET_VERSION,
        features_version=FEATURES_VERSION,
        feature_cache=_FEATURES,
        source_versions_cls=SourceVersions,
        batch_fetch_maps=batch_fetch_maps,
        build_generation_work=build_generation_work,
        coverage_payload=coverage_payload,
        layout_revision_for_projection=layout_revision_for_projection,
        embed_lock=_embed_lock,
        embed_status=_embed_status,
        status_cache=_status_cache,
        library_mtime_signature=_library_mtime_signature,
        track_infos_cached=_track_infos_cached,
        projection_query_fingerprint=_embedding_projection_query_fingerprint,
        parse_projection_params_fn=_parse_projection_params,
        build_track_infos_fn=_build_track_infos,
        resolve_virtual_path=_resolve,
        virtual_from_abs_path=_virtual_from_abs,
        broadcast_embed_event=_broadcast_embed_event,
        emit_decode_warning_once=_emit_decode_warning_once,
        broadcaster=_embed_broadcaster,
        is_generating=_embed_is_running,
        sse_response=sse_response,
    )
)


def _dir_color(folder_path: str) -> str:
    """Deterministic folder swatch colour (matches frontend dirColor())."""
    if not folder_path:
        return "hsl(350,60%,55%)"
    h = 0
    for ch in folder_path:
        h = (h * 31 + ord(ch)) & 0x3FFFF
    return f"hsl({h % 360},65%,60%)"


# ---------------------------------------------------------------------------
# HTMX partials
# ---------------------------------------------------------------------------

@app.route("/partials/folder-tree")
def partial_folder_tree():
    """HTML fragment for the folder sidebar (HTMX)."""
    active = request.args.get("active", "")
    active_folders = request.args.getlist("active_folders")
    if not active_folders:
        active_folders = [active if active != "" else "."]
    pending_rename = request.args.get("pending_rename", "")
    trees = [_folder_tree(root_abs, rid) for rid, root_abs in ROOTS.items()]
    return render_template(
        "partials/folder_tree.html",
        trees=trees,
        active_folders=active_folders,
        pending_rename=pending_rename,
        dir_color=_dir_color,
    )


# ---------------------------------------------------------------------------
# Page – serve Vite build (production)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if os.path.isdir(DIST_DIR):
        return send_from_directory(DIST_DIR, "index.html")
    return (
        "Frontend not built. Run <code>npm run build</code> in <code>frontend/</code>, "
        "or use the Vite dev server (<code>npm run dev</code>) for development."
    ), 404


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(DIST_DIR, "assets"), filename)


# ---------------------------------------------------------------------------
# API – Folders
# ---------------------------------------------------------------------------

@app.route("/api/folders")
def api_folders():
    """Return the folder tree for all roots (or a subpath)."""
    rel = request.args.get("root", "")
    if rel in ("", "."):
        return jsonify({"roots": [_folder_tree(root_abs, rid) for rid, root_abs in ROOTS.items()]})
    root_abs = _resolve(rel)
    parts = rel.split("/", 1)
    root_id = parts[0]
    rel_inside = parts[1] if len(parts) > 1 else ""
    return jsonify(_folder_tree(root_abs, root_id, rel_inside))


@app.route("/api/roots/add", methods=["POST"])
def api_roots_add():
    data = request.get_json(force=True)
    path = (data.get("path") or "").strip()
    if not path:
        abort(400)
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        return jsonify({"ok": False, "error": "Directory not found"})
    before_ids = set(ROOTS.keys())
    changed, root_id = _add_root(abs_path)
    after_ids = set(ROOTS.keys())
    for rid in sorted(before_ids - after_ids):
        _unschedule_root_watch(rid)
    for rid in sorted(after_ids - before_ids):
        _schedule_root_watch(rid)
    _save_roots_state()
    return jsonify({"ok": True, "changed": changed, "root_id": root_id})


@app.route("/api/roots/remove", methods=["POST"])
def api_roots_remove():
    data = request.get_json(force=True)
    root_id = (data.get("root_id") or "").strip()
    if not root_id:
        abort(400)
    if not _remove_root(root_id):
        return jsonify({"ok": False, "error": "Unknown root"})
    _unschedule_root_watch(root_id)
    _save_roots_state()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API – Tracks
# ---------------------------------------------------------------------------

@app.route("/api/tracks")
def api_tracks():
    """Return tracks (with tag data) for a folder.

    Query params:
        folder    – relative path (default: root)
        recursive – "1" to include subfolders
    """
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "0") == "1"
    paths = _list_mp3s(rel, recursive)

    # Read each file's ID3 header once (tags + metadata) in parallel threads.
    # _cached_read_all checks the in-memory LRU then SQLite before hitting disk.
    results = read_paths_parallel(paths, read_fn=_cached_read_all, executor=_THREAD_POOL)

    return jsonify(_build_track_list(paths, results))


def _track_dict_from_read(
    p: str,
    info: dict,
    *,
    feat_by_fp: dict[str, object] | None = None,
) -> dict:
    return track_dict_from_read_service(
        p,
        info,
        virtual_from_abs=_virtual_from_abs,
        display_bpm_key=audio_features.audio_features_display_bpm_key,
        get_audio_features=lambda fp: _FEATURES.get_audio_features(fp, FEATURES_VERSION),
        feat_by_fp=feat_by_fp,
    )


def _build_track_list(paths: list[str], results: dict[str, dict]) -> list[dict]:
    return build_track_list_service(
        paths,
        results,
        track_dict_builder=_track_dict_from_read,
        get_all_audio_features=lambda fps: _FEATURES.get_all_audio_features(fps, FEATURES_VERSION),
    )


@app.route("/api/library/stream")
def api_library_stream():
    """SSE endpoint — streams library scan progress then delivers the full track list.

    Query params:
        folder    – relative path (default: root)
        recursive – "1" to include subfolders

    Event types:
      ``progress`` — ``{"done": N, "total": N, "path": "rel/file.mp3"}``
      ``done``     — ``{"tracks": [...]}`` — full track list, same shape as /api/tracks
    """
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "0") == "1"
    paths = _list_mp3s(rel, recursive)
    total = len(paths)

    # Each SSE client gets its own queue; the scan thread fills it.
    q: queue.Queue[dict] = queue.Queue(maxsize=total + 100)

    def _scan() -> None:
        emit_scan_progress_events(
            paths,
            read_fn=_cached_read_all,
            executor=_THREAD_POOL,
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


# ---------------------------------------------------------------------------
# API – Tags
# ---------------------------------------------------------------------------

@app.route("/api/tags")
def api_tags():
    """Return all known tag names across currently visible tracks."""
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    paths = _list_mp3s(rel, recursive)
    # Use the same cached reads as api_tracks to avoid redundant disk I/O.
    names: set[str] = set()
    for _, info in iter_parallel_reads(paths, read_fn=_cached_read_all, executor=_THREAD_POOL):
        names.update(info.get("tags", {}).keys())
    return jsonify(sorted(names))


@app.route("/api/tracks/tags", methods=["POST"])
def api_update_tags():
    """Batch-update tag values.

    Expects JSON body:
        { "updates": [ {"path": "rel/file.mp3", "tag": "energy", "value": 0.7}, … ] }

    A null value deletes the tag.
    """
    data = request.get_json(force=True)
    updates = data.get("updates", [])
    for u in updates:
        abs_path = _resolve(u["path"])
        tagname = u["tag"]
        value = u.get("value")
        if value is None:
            delete_tag(abs_path, tagname)
        else:
            write_tag(abs_path, tagname, value)
    return jsonify({"ok": True, "count": len(updates)})


@app.route("/api/folders/reveal", methods=["POST"])
def api_reveal_folder():
    """Open a folder in the OS file manager (Finder on macOS, Explorer on Windows).

    Body: {"path": "rel/path/to/folder"}
    """
    data = request.get_json(force=True)
    abs_path = _resolve(data.get("path", ""))
    if not os.path.isdir(abs_path):
        abort(404)
    if sys.platform == "darwin":
        subprocess.Popen(["open", abs_path])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", abs_path])
    else:
        subprocess.Popen(["xdg-open", abs_path])
    return jsonify({"ok": True})


@app.route("/api/folders/create", methods=["POST"])
def api_create_folder():
    """Create a new subdirectory.

    Body: {"parent": "rel/path", "name": "new_dir"}
    """
    data = request.get_json(force=True)
    parent = _resolve(data.get("parent", ""))
    name = os.path.basename(data.get("name", "").strip())
    if not name:
        abort(400)
    new_dir = os.path.join(parent, name)
    if os.path.exists(new_dir):
        return jsonify({"ok": False, "error": "Already exists"})
    os.makedirs(new_dir)
    return jsonify({"ok": True, "path": _virtual_from_abs(new_dir)})


@app.route("/api/folders/rename", methods=["POST"])
def api_rename_folder():
    """Rename a folder (leaf name only).

    Body: {"path": "rel/path/to/folder", "name": "new_name"}
    """
    data = request.get_json(force=True)
    old_abs = _resolve(data.get("path", ""))
    new_name = os.path.basename(data.get("name", "").strip())
    if not new_name:
        abort(400)
    if not os.path.isdir(old_abs):
        abort(404)
    new_abs = os.path.join(os.path.dirname(old_abs), new_name)
    if os.path.exists(new_abs):
        return jsonify({"ok": False, "error": "Already exists"})
    os.rename(old_abs, new_abs)
    _CACHE.remap_prefix(old_abs + os.sep, new_abs + os.sep)
    return jsonify({"ok": True, "path": _virtual_from_abs(new_abs)})


@app.route("/api/tracks/move", methods=["POST"])
def api_move_tracks():
    """Move files to a different folder.

    Body: {"paths": ["rel/file.mp3", ...], "dest": "rel/dest/folder"}
    """
    data = request.get_json(force=True)
    dest_abs = _resolve(data.get("dest", ""))
    if not os.path.isdir(dest_abs):
        abort(400)
    moved, errors = 0, []
    for rel in data.get("paths", []):
        src = _resolve(rel)
        if not os.path.isfile(src):
            errors.append(f"Not found: {rel}")
            continue
        dst = os.path.join(dest_abs, os.path.basename(src))
        if src == dst:
            continue  # already in this folder — skip silently
        if os.path.exists(dst):
            errors.append(f"Already exists: {os.path.basename(src)}")
            continue
        try:
            os.rename(src, dst)
        except OSError:
            shutil.move(src, dst)
        _CACHE.remap(src, dst)
        moved += 1
    return jsonify({"ok": True, "moved": moved, "errors": errors})


@app.route("/api/tags/rename", methods=["POST"])
def api_rename_tag():
    """Rename a tag across all files in a folder.

    Body: {"old": "energy", "new": "vibe", "folder": "", "recursive": true}
    """
    data = request.get_json(force=True)
    folder = data.get("folder", "")
    recursive = data.get("recursive", True)
    paths = _list_mp3s(folder, recursive)
    count = sum(1 for p in paths if rename_tag(p, data["old"], data["new"]))
    return jsonify({"ok": True, "renamed": count})


@app.route("/api/tags/delete", methods=["POST"])
def api_delete_tag():
    """Delete a tag from all files in a folder.

    Body: {"name": "energy", "folder": "", "recursive": true}
    """
    data = request.get_json(force=True)
    folder = data.get("folder", "")
    recursive = data.get("recursive", True)
    paths = _list_mp3s(folder, recursive)
    for p in paths:
        delete_tag(p, data["name"])
    return jsonify({"ok": True, "files": len(paths)})


# ---------------------------------------------------------------------------
# API – Audio preview
# ---------------------------------------------------------------------------

@app.route("/api/audio/<path:relpath>")
def api_audio(relpath):
    """Stream an mp3 file for hover-preview playback."""
    abs_path = _resolve(relpath)
    if not os.path.isfile(abs_path):
        abort(404)
    return send_file(abs_path, mimetype="audio/mpeg")


# ---------------------------------------------------------------------------
# Background maintenance
# ---------------------------------------------------------------------------

def _start_gc_thread(interval: int = 600) -> None:
    """Prune TrackCache entries whose files no longer exist, every *interval* seconds."""
    def _loop() -> None:
        while True:
            time.sleep(interval)
            pruned = _CACHE.prune_missing()
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    global MUSIC_ROOT
    cli_root = os.path.abspath(args.root) if args.root else None
    MUSIC_ROOT = cli_root or ""
    _load_roots_state(cli_root)
    if ROOTS:
        print(f"Trackspace serving roots: {', '.join(ROOTS.values())}")
    else:
        print("Trackspace: no library roots — add folders in the sidebar (or pass a path on the CLI)")

    from backend.fingerprint import _FPCALC
    if _FPCALC:
        print(f"fpcalc: {_FPCALC}")
    else:
        print("WARNING: fpcalc not found — fingerprinting disabled. Install: brew install chromaprint")

    global _OBSERVER
    _OBSERVER = Observer()
    _OBSERVER.daemon = True
    _OBSERVER.start()
    for rid in list(ROOTS.keys()):
        _schedule_root_watch(rid)

    _start_gc_thread()

    pruned = _CACHE.prune_missing()
    if pruned:
        print(f"Cache: pruned {pruned} stale entr{'y' if pruned == 1 else 'ies'}")
    purged = _FEATURES.purge_old_versions(EMBEDDING_VERSION)
    if purged:
        print(f"CLAP: purged {purged} stale v<{EMBEDDING_VERSION} entr{'y' if purged == 1 else 'ies'}")
    purged_effnet = _FEATURES.purge_old_effnet_versions(EFFNET_VERSION)
    if purged_effnet:
        print(f"EffNet: purged {purged_effnet} stale v<{EFFNET_VERSION} entr{'y' if purged_effnet == 1 else 'ies'}")
    purged_feat = _FEATURES.purge_old_feature_versions(FEATURES_VERSION)
    if purged_feat:
        print(f"Features: purged {purged_feat} stale v<{FEATURES_VERSION} entr{'y' if purged_feat == 1 else 'ies'}")
    print(f"Versions: CLAP={EMBEDDING_VERSION} EffNet={EFFNET_VERSION} Features={FEATURES_VERSION}")
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
    def _bg_warm_audio_features():
        print("Warming audio feature pipeline…")
        audio_features.warmup_audio_features()
        print("Audio feature pipeline warm.")
    threading.Thread(target=_bg_warm_audio_features, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
