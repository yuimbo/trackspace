from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, MutableMapping
from typing import Any

from flask import Blueprint, abort, jsonify, render_template, request


def create_api_folders_blueprint(
    *,
    roots: MutableMapping[str, str],
    folder_tree: Callable[..., dict[str, Any]],
    resolve_virtual_path: Callable[[str], str],
    virtual_from_abs: Callable[[str], str],
    add_root: Callable[[str], tuple[bool, str]],
    remove_root: Callable[[str], bool],
    schedule_root_watch: Callable[[str], None],
    unschedule_root_watch: Callable[[str], None],
    save_roots_state: Callable[[], None],
    track_cache: Any,
    dir_color_fn: Callable[[str], str],
) -> Blueprint:
    bp = Blueprint("api_folders", __name__)

    @bp.route("/partials/folder-tree")
    def partial_folder_tree():
        """HTML fragment for the folder sidebar (HTMX)."""
        active = request.args.get("active", "")
        active_folders = request.args.getlist("active_folders")
        if not active_folders:
            active_folders = [active if active != "" else "."]
        pending_rename = request.args.get("pending_rename", "")
        trees = [folder_tree(root_abs, rid) for rid, root_abs in roots.items()]
        return render_template(
            "partials/folder_tree.html",
            trees=trees,
            active_folders=active_folders,
            pending_rename=pending_rename,
            dir_color=dir_color_fn,
        )

    @bp.route("/api/folders")
    def api_folders():
        """Return the folder tree for all roots (or a subpath)."""
        rel = request.args.get("root", "")
        if rel in ("", "."):
            return jsonify({"roots": [folder_tree(root_abs, rid) for rid, root_abs in roots.items()]})
        root_abs = resolve_virtual_path(rel)
        parts = rel.split("/", 1)
        root_id = parts[0]
        rel_inside = parts[1] if len(parts) > 1 else ""
        return jsonify(folder_tree(root_abs, root_id, rel_inside))

    @bp.route("/api/roots/add", methods=["POST"])
    def api_roots_add():
        data = request.get_json(force=True)
        path = (data.get("path") or "").strip()
        if not path:
            abort(400)
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return jsonify({"ok": False, "error": "Directory not found"})
        before_ids = set(roots.keys())
        changed, root_id = add_root(abs_path)
        after_ids = set(roots.keys())
        for rid in sorted(before_ids - after_ids):
            unschedule_root_watch(rid)
        for rid in sorted(after_ids - before_ids):
            schedule_root_watch(rid)
        save_roots_state()
        return jsonify({"ok": True, "changed": changed, "root_id": root_id})

    @bp.route("/api/roots/remove", methods=["POST"])
    def api_roots_remove():
        data = request.get_json(force=True)
        root_id = (data.get("root_id") or "").strip()
        if not root_id:
            abort(400)
        if not remove_root(root_id):
            return jsonify({"ok": False, "error": "Unknown root"})
        unschedule_root_watch(root_id)
        save_roots_state()
        return jsonify({"ok": True})

    @bp.route("/api/folders/reveal", methods=["POST"])
    def api_reveal_folder():
        """Open a folder in the OS file manager (Finder on macOS, Explorer on Windows).

        Body: {"path": "rel/path/to/folder"}
        """
        data = request.get_json(force=True)
        abs_path = resolve_virtual_path(data.get("path", ""))
        if not os.path.isdir(abs_path):
            abort(404)
        if sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
        return jsonify({"ok": True})

    @bp.route("/api/folders/create", methods=["POST"])
    def api_create_folder():
        """Create a new subdirectory.

        Body: {"parent": "rel/path", "name": "new_dir"}
        """
        data = request.get_json(force=True)
        parent = resolve_virtual_path(data.get("parent", ""))
        name = os.path.basename(data.get("name", "").strip())
        if not name:
            abort(400)
        new_dir = os.path.join(parent, name)
        if os.path.exists(new_dir):
            return jsonify({"ok": False, "error": "Already exists"})
        os.makedirs(new_dir)
        return jsonify({"ok": True, "path": virtual_from_abs(new_dir)})

    @bp.route("/api/folders/rename", methods=["POST"])
    def api_rename_folder():
        """Rename a folder (leaf name only).

        Body: {"path": "rel/path/to/folder", "name": "new_name"}
        """
        data = request.get_json(force=True)
        old_abs = resolve_virtual_path(data.get("path", ""))
        new_name = os.path.basename(data.get("name", "").strip())
        if not new_name:
            abort(400)
        if not os.path.isdir(old_abs):
            abort(404)
        new_abs = os.path.join(os.path.dirname(old_abs), new_name)
        if os.path.exists(new_abs):
            return jsonify({"ok": False, "error": "Already exists"})
        os.rename(old_abs, new_abs)
        track_cache.remap_prefix(old_abs + os.sep, new_abs + os.sep)
        return jsonify({"ok": True, "path": virtual_from_abs(new_abs)})

    @bp.route("/api/tracks/move", methods=["POST"])
    def api_move_tracks():
        """Move files to a different folder.

        Body: {"paths": ["rel/file.mp3", ...], "dest": "rel/dest/folder"}
        """
        data = request.get_json(force=True)
        dest_abs = resolve_virtual_path(data.get("dest", ""))
        if not os.path.isdir(dest_abs):
            abort(400)
        moved, errors = 0, []
        for rel in data.get("paths", []):
            src = resolve_virtual_path(rel)
            if not os.path.isfile(src):
                errors.append(f"Not found: {rel}")
                continue
            dst = os.path.join(dest_abs, os.path.basename(src))
            if src == dst:
                continue
            if os.path.exists(dst):
                errors.append(f"Already exists: {os.path.basename(src)}")
                continue
            try:
                os.rename(src, dst)
            except OSError:
                shutil.move(src, dst)
            track_cache.remap(src, dst)
            moved += 1
        return jsonify({"ok": True, "moved": moved, "errors": errors})

    return bp
