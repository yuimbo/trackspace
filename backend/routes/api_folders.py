from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, MutableMapping
from typing import Any

from flask import Blueprint, abort, jsonify, render_template, request

from backend.services.request_validation import (
    ApiValidationError,
    one_of,
    opt_str,
    parse_json_body,
    require_str,
    require_str_list,
)
from backend.services.roots_registry import (
    ROLES,
    ROLE_LIBRARY,
    root_role,
)


def _safe_leaf_name(name: str) -> str:
    """Validate a user-supplied single path component.

    Rejects empty names, ``.`` / ``..``, and anything containing a path
    separator so a crafted name cannot escape its parent directory.
    """
    stripped = name.strip()
    if stripped in ("", ".", ".."):
        raise ApiValidationError("field 'name' must be a valid directory name")
    if "/" in stripped or "\\" in stripped or os.sep in stripped:
        raise ApiValidationError("field 'name' must not contain a path separator")
    return stripped


def create_api_folders_blueprint(
    *,
    roots: MutableMapping[str, str],
    root_roles: MutableMapping[str, str],
    folder_tree: Callable[..., dict[str, Any]],
    resolve_virtual_path: Callable[[str], str],
    virtual_from_abs: Callable[[str], str],
    add_root: Callable[[str], tuple[bool, str]],
    remove_root: Callable[[str], bool],
    set_root_role: Callable[[str, str], None],
    schedule_root_watch: Callable[[str], None],
    unschedule_root_watch: Callable[[str], None],
    save_roots_state: Callable[[], None],
    track_cache: Any,
    dir_color_fn: Callable[[str], str],
    dir_anchors_fn: Callable[[], dict[str, str]],
) -> Blueprint:
    bp = Blueprint("api_folders", __name__)

    def _roots_payload() -> list[dict[str, Any]]:
        """One tree per root, each tagged with its role for the sidebar sections."""
        return [
            {
                **folder_tree(root_abs, rid),
                "role": root_role(rid, root_roles),
                "abs_path": root_abs,
            }
            for rid, root_abs in roots.items()
        ]

    @bp.route("/partials/folder-tree")
    def partial_folder_tree():
        """HTML fragment for the folder sidebar (HTMX)."""
        active = request.args.get("active", "")
        active_folders = request.args.getlist("active_folders")
        if not active_folders:
            active_folders = [active if active != "" else "."]
        pending_rename = request.args.get("pending_rename", "")
        return render_template(
            "partials/folder_tree.html",
            trees=_roots_payload(),
            active_folders=active_folders,
            pending_rename=pending_rename,
            dir_color=dir_color_fn,
            dir_anchors=dir_anchors_fn(),
        )

    @bp.route("/api/folders")
    def api_folders():
        """Return the folder tree for all roots (or a subpath)."""
        rel = request.args.get("root", "")
        if rel in ("", "."):
            return jsonify(
                {"roots": _roots_payload(), "anchors": dir_anchors_fn()}
            )
        root_abs = resolve_virtual_path(rel)
        parts = rel.split("/", 1)
        root_id = parts[0]
        rel_inside = parts[1] if len(parts) > 1 else ""
        return jsonify(folder_tree(root_abs, root_id, rel_inside))

    @bp.route("/api/roots/add", methods=["POST"])
    def api_roots_add():
        data = parse_json_body()
        path = require_str(data, "path").strip()
        role = one_of(
            opt_str(data, "role", ROLE_LIBRARY).strip() or ROLE_LIBRARY,
            ROLES,
            field="role",
        )
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return jsonify({"ok": False, "error": "Directory not found"})
        before_ids = set(roots.keys())
        changed, root_id = add_root(abs_path)
        after_ids = set(roots.keys())
        for rid in sorted(before_ids - after_ids):
            unschedule_root_watch(rid)
            root_roles.pop(rid, None)
        for rid in sorted(after_ids - before_ids):
            schedule_root_watch(rid)
        if changed:
            set_root_role(root_id, role)
        save_roots_state()
        return jsonify(
            {
                "ok": True,
                "changed": changed,
                "root_id": root_id,
                "role": root_role(root_id, root_roles),
            }
        )

    @bp.route("/api/roots/remove", methods=["POST"])
    def api_roots_remove():
        data = parse_json_body()
        root_id = require_str(data, "root_id").strip()
        if not remove_root(root_id):
            return jsonify({"ok": False, "error": "Unknown root"})
        root_roles.pop(root_id, None)
        unschedule_root_watch(root_id)
        save_roots_state()
        return jsonify({"ok": True})

    @bp.route("/api/roots/role", methods=["POST"])
    def api_roots_role():
        """Set a root's role: ``library`` (default) or ``external``."""
        data = parse_json_body()
        root_id = require_str(data, "root_id").strip()
        role = one_of(require_str(data, "role").strip(), ROLES, field="role")
        if root_id not in roots:
            return jsonify({"ok": False, "error": "Unknown root"})
        set_root_role(root_id, role)
        save_roots_state()
        return jsonify({"ok": True, "root_id": root_id, "role": role})

    @bp.route("/api/folders/reveal", methods=["POST"])
    def api_reveal_folder():
        """Open a folder in the OS file manager (Finder on macOS, Explorer on Windows).

        Body: {"path": "rel/path/to/folder"}
        """
        data = parse_json_body()
        abs_path = resolve_virtual_path(opt_str(data, "path", ""))
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
        data = parse_json_body()
        parent = resolve_virtual_path(opt_str(data, "parent", ""))
        name = _safe_leaf_name(require_str(data, "name"))
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
        data = parse_json_body()
        old_abs = resolve_virtual_path(opt_str(data, "path", ""))
        new_name = _safe_leaf_name(require_str(data, "name"))
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
        data = parse_json_body()
        dest_abs = resolve_virtual_path(opt_str(data, "dest", ""))
        if not os.path.isdir(dest_abs):
            abort(400)
        paths = require_str_list(data, "paths")
        moved, errors = 0, []
        for rel in paths:
            try:
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
            except Exception as exc:
                errors.append(f"Failed: {rel} ({exc})")
        return jsonify({"ok": True, "moved": moved, "errors": errors})

    return bp
