#!/usr/bin/env python3
"""Trackspace – organise music in a linear tag space.  Flask backend."""

import os
import sys
import subprocess
import json as _json
import logging
import argparse
import threading
import queue
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
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
from backend.embeddings import EMBEDDING_VERSION

load_dotenv()
log = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)

_THREAD_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)
_CACHE = TrackCache(os.path.join(_DATA_DIR, "track_cache.db"))
_FEATURES = FeatureCache(os.path.join(_DATA_DIR, "audio_features.db"))

# In-process LRU cache for projection results.
# Key: "{library_hash}|{method}|{context_tags}|{context_folders}"
# where library_hash is a SHA-256 of the sorted set of fingerprints that
# currently have embeddings.  The hash changes automatically whenever tracks
# are added, removed, or re-embedded, so stale entries are never returned.
# PCA projections are intentionally excluded (they are fast and their
# Procrustes-alignment reference state changes with each call).
_PROJECTION_CACHE_MAX = 8
_projection_cache: OrderedDict[str, list[dict]] = OrderedDict()

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
                emb = embeddings.generate_embedding(abs_path)
                ok = emb is not None
                if ok:
                    _FEATURES.put_embedding(fp, emb, version=EMBEDDING_VERSION)
                _broadcast_embed_event({
                    "type": "progress",
                    "path": rel_path,
                    "ok": ok,
                    "done": 1,
                    "total": 1,
                })
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
    pending_rename = request.args.get("pending_rename", "")
    tree = _folder_tree(MUSIC_ROOT)
    return render_template(
        "partials/folder_tree.html",
        tree=tree,
        active_folder=active,
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

    tracks = []
    for p in paths:
        rel_path = os.path.relpath(p, MUSIC_ROOT)
        folder_rel = os.path.relpath(os.path.dirname(p), MUSIC_ROOT)
        if folder_rel == ".":
            folder_rel = ""
        info = results.get(p, {})
        tracks.append({
            "path": rel_path,
            "filename": os.path.basename(p),
            "folder": folder_rel,
            "tags": info.get("tags", {}),
            "artist": info.get("artist", ""),
            "title": info.get("title", ""),
            "fingerprint": info.get("fingerprint"),
        })
    return jsonify(tracks)


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
    return infos


@app.route("/api/embeddings/status")
def api_embeddings_status():
    """Return embedding coverage stats for a folder."""
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    folder = _resolve(rel)
    infos = _build_track_infos(folder, recursive)

    total = len(infos)
    fingerprinted = sum(1 for t in infos if t.get("fingerprint"))
    fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
    existing = _FEATURES.get_all_embeddings(fps, version=EMBEDDING_VERSION)
    embedded = len(existing)

    with _embed_lock:
        running = _embed_status["running"]
        progress_done = _embed_status["done"]
        progress_total = _embed_status["total"]

    return jsonify({
        "total": total,
        "fingerprinted": fingerprinted,
        "embedded": embedded,
        "pending": fingerprinted - embedded,
        "model_ready": embeddings.is_model_ready(),
        "generating": running,
        "embedding_version": EMBEDDING_VERSION,
        "progress": {"done": progress_done, "total": progress_total} if running else None,
    })


@app.route("/api/embeddings/generate", methods=["POST"])
def api_embeddings_generate():
    """Kick off background CLAP embedding generation for tracks missing embeddings.

    Body: {"folder": "", "recursive": true, "priority_paths": [...]}
    ``priority_paths`` is an optional list of relative track paths that should
    be generated first (e.g. tracks currently visible on screen).
    """
    data = request.get_json(force=True)
    rel = data.get("folder", "")
    recursive = data.get("recursive", True)
    priority_paths: list[str] = data.get("priority_paths", [])
    folder = _resolve(rel)

    if not embeddings.is_model_ready():
        return jsonify({"ok": False, "error": "CLAP model still loading, try again shortly"})

    with _embed_lock:
        if _embed_status["running"]:
            return jsonify({"ok": False, "error": "Generation already in progress"})
        _embed_status["running"] = True
        _embed_status["done"] = 0
        _embed_status["total"] = 0
        _embed_status["error"] = None

    def _run():
        try:
            infos = _build_track_infos(folder, recursive)
            fps_needed: list[dict] = []
            for t in infos:
                fp = t.get("fingerprint")
                if fp and not _FEATURES.has_embedding(fp, version=EMBEDDING_VERSION):
                    fps_needed.append(t)

            # Prioritise tracks the user can currently see.
            if priority_paths:
                pset = set(priority_paths)
                fps_needed.sort(key=lambda t: (0 if t["path"] in pset else 1, t["path"]))

            with _embed_lock:
                _embed_status["total"] = len(fps_needed)

            done = 0
            for t in fps_needed:
                abs_path = os.path.join(MUSIC_ROOT, t["path"])
                emb = embeddings.generate_embedding(abs_path)
                ok = emb is not None
                if ok:
                    _FEATURES.put_embedding(t["fingerprint"], emb, version=EMBEDDING_VERSION)
                done += 1
                with _embed_lock:
                    _embed_status["done"] = done
                _broadcast_embed_event({
                    "type": "progress",
                    "path": t["path"],
                    "ok": ok,
                    "done": done,
                    "total": len(fps_needed),
                })

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
      ``progress`` — ``{"path", "ok", "done", "total"}``
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
                yield f"event: done\ndata: {{}}\n\n"
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


@app.route("/api/embeddings/umap")
def api_embeddings_umap():
    """Return 2D positions for tracks with embeddings.

    Query params:
        folder          – relative path (default: root)
        recursive       – "1" to include subfolders
        method          – "umap" (default), "tsne", or "pca" (fast, for live updates)
        context_tags    – comma-separated tag names (full boost)
        context_folders – comma-separated folder paths (depth-scaled boost)
    """
    rel = request.args.get("folder", "")
    recursive = request.args.get("recursive", "1") == "1"
    method = request.args.get("method", "umap")
    if method not in ("umap", "pca", "tsne"):
        method = "umap"
    raw_tags = request.args.get("context_tags", "")
    raw_folders = request.args.get("context_folders", "")
    tag_names = [s.strip() for s in raw_tags.split(",") if s.strip()] if raw_tags else []
    folder_paths = [s.strip() for s in raw_folders.split(",") if s.strip()] if raw_folders else []
    folder = _resolve(rel)
    infos = _build_track_infos(folder, recursive)

    # Build a cache key from the sorted set of fingerprints that currently have
    # embeddings.  The hash changes whenever the embedded library changes, so
    # stale results are never served.  PCA is excluded: it is fast and its
    # Procrustes reference state changes per call.
    cache_key: str | None = None
    if method != "pca":
        fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
        present_fps = sorted(_FEATURES.get_all_embeddings(fps, version=EMBEDDING_VERSION).keys())
        if present_fps:
            lib_hash = hashlib.sha256(",".join(present_fps).encode()).hexdigest()[:20]
            tags_part = ",".join(sorted(tag_names))
            folders_part = ",".join(sorted(folder_paths))
            cache_key = f"{lib_hash}|{method}|{tags_part}|{folders_part}"
            if cache_key in _projection_cache:
                _projection_cache.move_to_end(cache_key)
                log.debug("projection cache hit: %s", cache_key)
                return jsonify(_projection_cache[cache_key])

    positions = embeddings.compute_projection(
        infos, _FEATURES, version=EMBEDDING_VERSION, method=method,
        context_tags=tag_names, context_folders=folder_paths,
    )

    if cache_key is not None:
        _projection_cache[cache_key] = positions
        _projection_cache.move_to_end(cache_key)
        if len(_projection_cache) > _PROJECTION_CACHE_MAX:
            evicted = _projection_cache.popitem(last=False)
            log.debug("projection cache evicted: %s", evicted[0])

    return jsonify(positions)


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
        print(f"Embeddings: purged {purged} stale v<{EMBEDDING_VERSION} entr{'y' if purged == 1 else 'ies'}")
    print(f"Embedding version: {EMBEDDING_VERSION}")
    if not args.no_clap:
        def _bg_load():
            print("Loading CLAP model in background (this may take a moment on first run)…")
            embeddings.load_model()
            print("CLAP model ready.")
        threading.Thread(target=_bg_load, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
