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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from backend.process_limits import raise_nofile_limit

raise_nofile_limit()

from flask import (
    Flask,
    Response,
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
    eligible_paths_for_projection,
)
from backend.layout_revision import compute_layout_revision

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

        rel_path = os.path.relpath(abs_path, MUSIC_ROOT)
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
_embed_subscribers: list[queue.Queue] = []

# Embedding /status caches (see api_embeddings_status).
_TRACK_INFOS_CACHE_MAX = 8
_track_infos_lru: OrderedDict[tuple[str, bool, str], list[dict]] = OrderedDict()
_STATUS_COVERAGE_CACHE_MAX = 48
_status_coverage_lru: OrderedDict[tuple[str, int, str], tuple[dict, str | None]] = OrderedDict()
_embed_status_cache_lock = threading.Lock()


def _library_mtime_signature(folder_abs: str, recursive: bool) -> str:
    """Cheap fingerprint of the mp3 set under *folder_abs* (path + mtime per file)."""
    h = hashlib.sha256()
    for p in _list_mp3s(folder_abs, recursive):
        rel = os.path.relpath(p, MUSIC_ROOT)
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(b"\0")
        try:
            h.update(str(os.path.getmtime(p)).encode("ascii", errors="ignore"))
        except OSError:
            h.update(b"0")
        h.update(b"\0")
    return h.hexdigest()


def _embedding_projection_query_fingerprint(req) -> str:
    """Stable hash of query args that affect coverage counts or layout_revision."""
    skip = frozenset({"folder", "recursive", "models_only"})
    parts: list[str] = []
    for k in sorted(req.args.keys()):
        if k in skip:
            continue
        parts.append(f"{k}={req.args.get(k, '')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _track_infos_cached(folder_abs: str, recursive: bool, lib_sig: str) -> list[dict]:
    key = (folder_abs, recursive, lib_sig)
    with _embed_status_cache_lock:
        if key in _track_infos_lru:
            _track_infos_lru.move_to_end(key)
            return _track_infos_lru[key]
    infos = _build_track_infos(folder_abs, recursive)
    with _embed_status_cache_lock:
        _track_infos_lru[key] = infos
        _track_infos_lru.move_to_end(key)
        while len(_track_infos_lru) > _TRACK_INFOS_CACHE_MAX:
            _track_infos_lru.popitem(last=False)
    return infos


def _broadcast_embed_event(event: dict) -> None:
    with _embed_lock:
        dead: list[queue.Queue] = []
        for q in _embed_subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _embed_subscribers.remove(q)


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

MUSIC_ROOT: str = ""  # set via CLI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(rel: str) -> str:
    """Resolve a client-supplied relative path against MUSIC_ROOT safely."""
    joined = os.path.normpath(os.path.join(MUSIC_ROOT, rel))
    if not joined.startswith(MUSIC_ROOT):
        abort(403)
    return joined


def _is_mp3(name: str) -> bool:
    return name.lower().endswith(".mp3")


def _list_mp3s(folder: str, recursive: bool = False) -> list[str]:
    """Return absolute paths of mp3 files in *folder*."""
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


def _folder_tree(root: str) -> dict:
    """Return a nested dict representing the subfolder tree under *root*."""
    name = os.path.basename(root) or root
    children = []
    try:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                children.append(_folder_tree(full))
    except PermissionError:
        pass
    return {"name": name, "path": os.path.relpath(root, MUSIC_ROOT), "children": children}


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
    tree = _folder_tree(MUSIC_ROOT)
    return render_template(
        "partials/folder_tree.html",
        tree=tree,
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
    """Return the folder tree under MUSIC_ROOT (or a subpath)."""
    rel = request.args.get("root", "")
    root = _resolve(rel)
    return jsonify(_folder_tree(root))


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
    folder = _resolve(rel)

    paths = _list_mp3s(folder, recursive)

    # Read each file's ID3 header once (tags + metadata) in parallel threads.
    # _cached_read_all checks the in-memory LRU then SQLite before hitting disk.
    results: dict[str, dict] = {}
    futures = {_THREAD_POOL.submit(_cached_read_all, p): p for p in paths}
    for fut in as_completed(futures):
        results[futures[fut]] = fut.result()

    return jsonify(_build_track_list(paths, results))


def _track_dict_from_read(
    p: str,
    info: dict,
    *,
    feat_by_fp: dict[str, object] | None = None,
) -> dict:
    """One API track object from a filesystem path and ``_cached_read_all`` payload."""
    rel_path = os.path.relpath(p, MUSIC_ROOT)
    folder_rel = os.path.relpath(os.path.dirname(p), MUSIC_ROOT)
    if folder_rel == ".":
        folder_rel = ""
    fp = info.get("fingerprint")
    bpm: int | None = None
    musical_key: str | None = None
    if isinstance(fp, str) and fp:
        arr = feat_by_fp.get(fp) if feat_by_fp is not None else _FEATURES.get_audio_features(
            fp, FEATURES_VERSION
        )
        if arr is not None:
            bpm, musical_key = audio_features.audio_features_display_bpm_key(arr)
    return {
        "path": rel_path,
        "filename": os.path.basename(p),
        "folder": folder_rel,
        "tags": info.get("tags", {}),
        "artist": info.get("artist", ""),
        "title": info.get("title", ""),
        "fingerprint": info.get("fingerprint"),
        "bpm": bpm,
        "musical_key": musical_key,
    }


def _build_track_list(paths: list[str], results: dict[str, dict]) -> list[dict]:
    """Convert path→data mapping into the track dicts the frontend expects."""
    fps_ordered: list[str] = []
    seen: set[str] = set()
    for p in paths:
        fp = results.get(p, {}).get("fingerprint")
        if not isinstance(fp, str) or not fp or fp in seen:
            continue
        seen.add(fp)
        fps_ordered.append(fp)
    feat_map = (
        _FEATURES.get_all_audio_features(fps_ordered, FEATURES_VERSION)
        if fps_ordered
        else {}
    )
    return [_track_dict_from_read(p, results.get(p, {}), feat_by_fp=feat_map) for p in paths]


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
    folder = _resolve(rel)
    paths = _list_mp3s(folder, recursive)
    total = len(paths)

    # Each SSE client gets its own queue; the scan thread fills it.
    q: queue.Queue[dict] = queue.Queue(maxsize=total + 100)

    def _scan() -> None:
        futures = {_THREAD_POOL.submit(_cached_read_all, p): p for p in paths}
        done = 0
        for fut in as_completed(futures):
            p = futures[fut]
            info = fut.result()
            done += 1
            q.put_nowait({
                "type": "progress",
                "done": done,
                "total": total,
                "track": _track_dict_from_read(p, info),
            })
        q.put_nowait({"type": "done"})

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

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# API – Tags
# ---------------------------------------------------------------------------

@app.route("/api/tags")
def api_tags():
    """Return all known tag names across currently visible tracks."""
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    folder = _resolve(rel)
    paths = _list_mp3s(folder, recursive)
    # Use the same cached reads as api_tracks to avoid redundant disk I/O.
    futures = {_THREAD_POOL.submit(_cached_read_all, p): p for p in paths}
    names: set[str] = set()
    for fut in as_completed(futures):
        names.update(fut.result().get("tags", {}).keys())
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
    return jsonify({"ok": True, "path": os.path.relpath(new_dir, MUSIC_ROOT)})


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
    return jsonify({"ok": True, "path": os.path.relpath(new_abs, MUSIC_ROOT)})


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
        os.rename(src, dst)
        _CACHE.remap(src, dst)
        moved += 1
    return jsonify({"ok": True, "moved": moved, "errors": errors})


@app.route("/api/tags/rename", methods=["POST"])
def api_rename_tag():
    """Rename a tag across all files in a folder.

    Body: {"old": "energy", "new": "vibe", "folder": "", "recursive": true}
    """
    data = request.get_json(force=True)
    folder = _resolve(data.get("folder", ""))
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
    folder = _resolve(data.get("folder", ""))
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
# API – Embeddings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProjectionParams:
    method: str
    tag_names: list[str]
    explicit_folders: list[str]
    scale_folders: bool
    feature_mask: object  # np.ndarray after parse
    features_blend: float
    folder_boost: float
    folder_depth_boost: float
    sources: tuple[str, ...]


def _parse_projection_params(
    req,
    *,
    default_when_no_method: bool = False,
) -> _ProjectionParams | None:
    """Parse projection query args shared by status + projection routes.

    For ``/api/embeddings/status``, *default_when_no_method* is False so a
    minimal poll URL omits ``layout_revision``.  For the projection route,
    pass ``default_when_no_method=True`` so ``method`` defaults to umap.
    """
    raw_method = req.args.get("method", "")
    if not raw_method and not default_when_no_method:
        return None
    method = raw_method or "umap"
    if method not in ("umap", "pca", "tsne"):
        method = "umap"

    raw_tags = req.args.get("context_tags", "")
    raw_folders = req.args.get("context_folders", "")
    raw_sources = req.args.get("sources", "clap")
    tag_names = [s.strip() for s in raw_tags.split(",") if s.strip()] if raw_tags else []
    explicit_folders = (
        [s.strip() for s in raw_folders.split(",") if s.strip()] if raw_folders else []
    )
    scale_folders = req.args.get("scale_folders", "") == "1"
    if not scale_folders and explicit_folders:
        scale_folders = True
    feature_mask = embeddings.parse_audio_feature_mask(req.args.get("feature_mask"))
    features_blend = embeddings.parse_features_blend(req.args.get("features_blend"))
    folder_boost = embeddings.parse_folder_boost(req.args.get("folder_boost"))
    folder_depth_boost = embeddings.parse_folder_depth_boost(
        req.args.get("folder_depth_boost"),
    )
    sources = tuple(
        s.strip()
        for s in raw_sources.split(",")
        if s.strip() in ("clap", "effnet", "features")
    )
    if not sources:
        sources = ("clap",)

    return _ProjectionParams(
        method=method,
        tag_names=tag_names,
        explicit_folders=explicit_folders,
        scale_folders=scale_folders,
        feature_mask=feature_mask,
        features_blend=features_blend,
        folder_boost=folder_boost,
        folder_depth_boost=folder_depth_boost,
        sources=sources,
    )


def _feature_mask_as_list(params: _ProjectionParams) -> list[float]:
    import numpy as np

    m = params.feature_mask
    arr = np.asarray(m, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr]


def _build_track_infos(folder_abs: str, recursive: bool) -> list[dict]:
    """Build lightweight track info dicts with path + fingerprint for embedding ops."""
    paths = _list_mp3s(folder_abs, recursive)
    futures = {_THREAD_POOL.submit(_cached_read_all, p): p for p in paths}
    infos = []
    for fut in as_completed(futures):
        p = futures[fut]
        info = fut.result()
        rel_path = os.path.relpath(p, MUSIC_ROOT)
        infos.append({
            "path": rel_path,
            "fingerprint": info.get("fingerprint"),
        })
    infos.sort(key=lambda t: t["path"])
    return infos


@app.route("/api/embeddings/status")
def api_embeddings_status():
    """Return embedding coverage stats for a folder.

    Query params:
        models_only=1 — skip library scan; only model readiness + generation flag
            (for lightweight polling).
        With projection parameters (``method``, ``sources``, …) the response
        includes ``layout_revision`` so the client can skip redundant layout fetches.

    When generation is idle, coverage + layout_revision are cached by library
    signature, feature-cache write epoch, and projection query fingerprint.
    """
    if request.args.get("models_only") == "1":
        with _embed_lock:
            running = _embed_status["running"]
            progress_done = _embed_status["done"]
            progress_total = _embed_status["total"]
        return jsonify({
            "model_ready": embeddings.is_model_ready(),
            "effnet_model_ready": audio_features.is_effnet_ready(),
            "generating": running,
            "progress": {"done": progress_done, "total": progress_total} if running else None,
        })

    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    folder = _resolve(rel)
    lib_sig = _library_mtime_signature(folder, recursive)
    infos = _track_infos_cached(folder, recursive, lib_sig)

    versions = SourceVersions(EMBEDDING_VERSION, EFFNET_VERSION, FEATURES_VERSION)
    write_epoch = _FEATURES.write_epoch()
    proj_fp = _embedding_projection_query_fingerprint(request)

    with _embed_lock:
        running = _embed_status["running"]
        progress_done = _embed_status["done"]
        progress_total = _embed_status["total"]

    layout_revision: str | None = None
    if running:
        fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
        maps = batch_fetch_maps(_FEATURES, fps, versions)
        base = coverage_payload(infos, maps, versions)
        proj = _parse_projection_params(request, default_when_no_method=False)
        if proj is not None:
            eligible = eligible_paths_for_projection(infos, maps, proj.sources)
            layout_revision = compute_layout_revision(
                eligible_paths=eligible,
                method=proj.method,
                sources=proj.sources,
                feature_mask=_feature_mask_as_list(proj),
                features_blend=proj.features_blend,
                folder_boost=proj.folder_boost,
                folder_depth_boost=proj.folder_depth_boost,
                context_tags=proj.tag_names,
                context_folders=proj.explicit_folders if proj.explicit_folders else None,
                scale_folders=proj.scale_folders,
                cache_versions=(versions.clap, versions.effnet, versions.features),
            )
    else:
        cov_key = (lib_sig, write_epoch, proj_fp)
        with _embed_status_cache_lock:
            if cov_key in _status_coverage_lru:
                base, layout_revision = _status_coverage_lru[cov_key]
                _status_coverage_lru.move_to_end(cov_key)
                base = dict(base)
            else:
                base = None
        if base is None:
            fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
            maps = batch_fetch_maps(_FEATURES, fps, versions)
            base = coverage_payload(infos, maps, versions)
            proj = _parse_projection_params(request, default_when_no_method=False)
            layout_revision = None
            if proj is not None:
                eligible = eligible_paths_for_projection(infos, maps, proj.sources)
                layout_revision = compute_layout_revision(
                    eligible_paths=eligible,
                    method=proj.method,
                    sources=proj.sources,
                    feature_mask=_feature_mask_as_list(proj),
                    features_blend=proj.features_blend,
                    folder_boost=proj.folder_boost,
                    folder_depth_boost=proj.folder_depth_boost,
                    context_tags=proj.tag_names,
                    context_folders=proj.explicit_folders if proj.explicit_folders else None,
                    scale_folders=proj.scale_folders,
                    cache_versions=(versions.clap, versions.effnet, versions.features),
                )
            with _embed_status_cache_lock:
                _status_coverage_lru[cov_key] = (dict(base), layout_revision)
                _status_coverage_lru.move_to_end(cov_key)
                while len(_status_coverage_lru) > _STATUS_COVERAGE_CACHE_MAX:
                    _status_coverage_lru.popitem(last=False)

    payload = {
        **base,
        "model_ready": embeddings.is_model_ready(),
        "effnet_model_ready": audio_features.is_effnet_ready(),
        "generating": running,
        "progress": {"done": progress_done, "total": progress_total} if running else None,
    }
    if layout_revision is not None:
        payload["layout_revision"] = layout_revision
    return jsonify(payload)


@app.route("/api/embeddings/generate", methods=["POST"])
def api_embeddings_generate():
    """Kick off background generation for one or more embedding sources.

    Body: {"folder": "", "recursive": true, "priority_paths": [...],
           "sources": ["clap", "effnet", "features"]}
    ``priority_paths`` is an optional list of relative track paths that should
    be generated first (e.g. tracks currently visible on screen).
    ``sources`` defaults to ``["clap"]`` for backward compatibility.
    """
    data = request.get_json(force=True)
    rel = data.get("folder", "")
    recursive = data.get("recursive", True)
    priority_paths: list[str] = data.get("priority_paths", [])
    sources: list[str] = data.get("sources", ["clap"])
    folder = _resolve(rel)

    # Validate that required models are ready.
    if "clap" in sources and not embeddings.is_model_ready():
        return jsonify({"ok": False, "error": "CLAP model still loading, try again shortly"})
    if "effnet" in sources and not audio_features.is_effnet_ready():
        return jsonify({"ok": False, "error": "EffNet model still loading, try again shortly"})

    with _embed_lock:
        if _embed_status["running"]:
            return jsonify({"ok": False, "error": "Generation already in progress"})
        _embed_status["running"] = True
        _embed_status["done"] = 0
        _embed_status["total"] = 0
        _embed_status["error"] = None

    def _run():
        try:
            decode_warn_seen: set[str] = set()

            def _note_decode(rel: str, msg: str) -> None:
                _emit_decode_warning_once(rel, msg, decode_warn_seen)

            infos = _build_track_infos(folder, recursive)
            vers = SourceVersions(EMBEDDING_VERSION, EFFNET_VERSION, FEATURES_VERSION)
            work = build_generation_work(infos, _FEATURES, sources, vers)

            # Deterministic order; optional tier for paths the client cares about first.
            # If every work item is "priority", the tier is a no-op vs path sort — skip
            # the extra pass so giant priority_paths payloads are unnecessary.
            if priority_paths:
                pset = set(priority_paths)
                deprioritized = any(w[0]["path"] not in pset for w in work)
                if deprioritized:
                    work.sort(key=lambda w: (0 if w[0]["path"] in pset else 1, w[0]["path"]))
                else:
                    work.sort(key=lambda w: w[0]["path"])
            else:
                work.sort(key=lambda w: w[0]["path"])

            with _embed_lock:
                _embed_status["total"] = len(work)

            _EFFNET_CHUNK = 8
            done = 0
            for chunk_start in range(0, max(len(work), 1), _EFFNET_CHUNK):
                chunk = work[chunk_start:chunk_start + _EFFNET_CHUNK]
                if not chunk:
                    break

                effnet_paths = [
                    os.path.join(MUSIC_ROOT, t["path"])
                    for t, needed in chunk if "effnet" in needed
                ]
                effnet_decode_warns: dict[str, str] = {}
                effnet_results = (
                    audio_features.generate_effnet_embeddings_batch(
                        effnet_paths, decode_warnings=effnet_decode_warns,
                    )
                    if effnet_paths else {}
                )
                for abs_p, msg in effnet_decode_warns.items():
                    _note_decode(os.path.relpath(abs_p, MUSIC_ROOT), msg)
                feature_paths = [
                    os.path.join(MUSIC_ROOT, t["path"])
                    for t, needed in chunk if "features" in needed
                ]
                feat_decode_warns: dict[str, str] = {}
                feature_results = (
                    audio_features.generate_audio_features_batch(
                        feature_paths, decode_warnings=feat_decode_warns,
                    )
                    if feature_paths else {}
                )
                for abs_p, msg in feat_decode_warns.items():
                    _note_decode(os.path.relpath(abs_p, MUSIC_ROOT), msg)

                for t, needed in chunk:
                    abs_path = os.path.join(MUSIC_ROOT, t["path"])
                    fp = t["fingerprint"]
                    ok = True
                    failures: list[str] = []

                    if "clap" in needed:
                        rel = t["path"]
                        emb = embeddings.generate_embedding(
                            abs_path,
                            on_decode_warning=lambda m, rp=rel: _note_decode(rp, m),
                        )
                        if emb is not None:
                            _FEATURES.put_embedding(fp, emb, version=EMBEDDING_VERSION)
                        else:
                            ok = False
                            failures.append("clap")

                    if "effnet" in needed:
                        emb = effnet_results.get(abs_path)
                        if emb is not None:
                            _FEATURES.put_effnet_embedding(fp, emb, version=EFFNET_VERSION)
                        else:
                            ok = False
                            failures.append("effnet")

                    if "features" in needed:
                        feat = feature_results.get(abs_path)
                        if feat is not None:
                            _FEATURES.put_audio_features(fp, feat, version=FEATURES_VERSION)
                        else:
                            ok = False
                            failures.append("features")

                    done += 1
                    with _embed_lock:
                        _embed_status["done"] = done
                    prog_evt: dict = {
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
                    _broadcast_embed_event(prog_evt)

        except Exception as e:
            log.exception("Embedding generation failed")
            with _embed_lock:
                _embed_status["error"] = str(e)
            _broadcast_embed_event({"type": "error", "error": str(e)})
        finally:
            with _embed_lock:
                _embed_status["running"] = False
            _broadcast_embed_event({"type": "done"})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/embeddings/stream")
def api_embeddings_stream():
    """SSE endpoint — streams embedding generation progress events.

    Event types:
      ``progress`` — ``{"path", "ok", "done", "total", "failures"?}``
      (*failures* lists sources that did not write cache: clap / effnet / features)
      ``decode_warning`` — ``{"path", "message"}`` metadata vs decode mismatch
      ``done``     — generation finished
      ``error``    — generation failed with message
    """
    q: queue.Queue[dict] = queue.Queue(maxsize=2000)
    with _embed_lock:
        _embed_subscribers.append(q)

    def generate():
        try:
            with _embed_lock:
                running = _embed_status["running"]
            if not running:
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
            with _embed_lock:
                if q in _embed_subscribers:
                    _embed_subscribers.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/embeddings/projection")
@app.route("/api/embeddings/umap")  # backward compat alias
def api_embeddings_projection():
    """Return 2D positions (PCA / t-SNE / UMAP) for the active embedding mix.

    Query params:
        folder              – relative path (default: root)
        recursive           – "1" to include subfolders
        method              – "umap" (default), "tsne", or "pca" (fast, for live updates)
        context_tags        – comma-separated tag names (full boost)
        scale_folders       – "1" to enable folder centroid re-weighting (hierarchy from paths)
        context_folders     – optional explicit folder seeds (legacy; implies scaling if set)
        folder_boost        – folder / tag contrast strength (default 3)
        folder_depth_boost  – multiply deeper folder contrasts (1=uniform, up to 3)
        sources             – comma-separated: clap, effnet, features
        feature_mask        – six 0/1 chars: tempo, key×2, mode, energy, dance
        features_blend      – scale of audio-feature block after norm (default 0.42)
    """
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    folder = _resolve(rel)
    infos = _build_track_infos(folder, recursive)

    proj = _parse_projection_params(request, default_when_no_method=True)

    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    pversions = SourceVersions(EMBEDDING_VERSION, EFFNET_VERSION, FEATURES_VERSION)
    maps = batch_fetch_maps(_FEATURES, fps, pversions)
    eligible = eligible_paths_for_projection(infos, maps, proj.sources)
    revision = compute_layout_revision(
        eligible_paths=eligible,
        method=proj.method,
        sources=proj.sources,
        feature_mask=_feature_mask_as_list(proj),
        features_blend=proj.features_blend,
        folder_boost=proj.folder_boost,
        folder_depth_boost=proj.folder_depth_boost,
        context_tags=proj.tag_names,
        context_folders=proj.explicit_folders if proj.explicit_folders else None,
        scale_folders=proj.scale_folders,
        cache_versions=(pversions.clap, pversions.effnet, pversions.features),
    )

    positions = embeddings.compute_projection(
        infos,
        _FEATURES,
        version=EMBEDDING_VERSION,
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
    parser.add_argument("root", nargs="?", default=".",
                        help="Root music directory to serve (default: cwd)")
    parser.add_argument("-p", "--port", type=int, default=5111)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-clap", action="store_true",
                        help="Skip eager CLAP model loading at startup")
    parser.add_argument("--no-effnet", action="store_true",
                        help="Skip eager EffNet model loading at startup")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    global MUSIC_ROOT
    MUSIC_ROOT = os.path.abspath(args.root)
    print(f"Trackspace serving: {MUSIC_ROOT}")

    from backend.fingerprint import _FPCALC
    if _FPCALC:
        print(f"fpcalc: {_FPCALC}")
    else:
        print("WARNING: fpcalc not found — fingerprinting disabled. Install: brew install chromaprint")

    _observer = Observer()
    _observer.schedule(_FilesystemHandler(), MUSIC_ROOT, recursive=True)
    _observer.daemon = True
    _observer.start()

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
