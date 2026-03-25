#!/usr/bin/env python3
"""Trackspace – organise music in a linear tag space.  Flask backend."""

import os
import argparse

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    send_from_directory,
    render_template,
    abort,
)

from tags import write_tag, delete_tag, rename_tag, batch_read, collect_tag_names, read_metadata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")

app = Flask(__name__, static_folder=None, template_folder=os.path.join(BASE_DIR, "templates"))

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
    tag_data = batch_read(paths)

    tracks = []
    for p in paths:
        rel = os.path.relpath(p, MUSIC_ROOT)
        folder = os.path.relpath(os.path.dirname(p), MUSIC_ROOT)
        if folder == ".":
            folder = ""
        meta = read_metadata(p)
        tracks.append({
            "path": rel,
            "filename": os.path.basename(p),
            "folder": folder,
            "tags": tag_data.get(p, {}),
            "artist": meta.get("artist", ""),
            "title": meta.get("title", ""),
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
    return jsonify(collect_tag_names(paths))


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
        if os.path.exists(dst):
            errors.append(f"Already exists: {os.path.basename(src)}")
            continue
        os.rename(src, dst)
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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trackspace server")
    parser.add_argument("root", nargs="?", default=".",
                        help="Root music directory to serve (default: cwd)")
    parser.add_argument("-p", "--port", type=int, default=5111)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global MUSIC_ROOT
    MUSIC_ROOT = os.path.abspath(args.root)
    print(f"Trackspace serving: {MUSIC_ROOT}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
