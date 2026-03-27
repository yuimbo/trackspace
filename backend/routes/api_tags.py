from __future__ import annotations

from collections.abc import Callable

from flask import Blueprint, jsonify, request

from backend.tags import delete_tag, rename_tag, write_tag


def create_api_tags_blueprint(
    *,
    list_mp3s: Callable[[str, bool], list[str]],
    resolve_virtual_path: Callable[[str], str],
) -> Blueprint:
    bp = Blueprint("api_tags", __name__)

    @bp.route("/api/tracks/tags", methods=["POST"])
    def api_update_tags():
        """Batch-update tag values.

        Expects JSON body:
            { "updates": [ {"path": "rel/file.mp3", "tag": "energy", "value": 0.7}, … ] }

        A null value deletes the tag.
        """
        data = request.get_json(force=True)
        updates = data.get("updates", [])
        for u in updates:
            abs_path = resolve_virtual_path(u["path"])
            tagname = u["tag"]
            value = u.get("value")
            if value is None:
                delete_tag(abs_path, tagname)
            else:
                write_tag(abs_path, tagname, value)
        return jsonify({"ok": True, "count": len(updates)})

    @bp.route("/api/tags/rename", methods=["POST"])
    def api_rename_tag():
        """Rename a tag across all files in a folder.

        Body: {"old": "energy", "new": "vibe", "folder": "", "recursive": true}
        """
        data = request.get_json(force=True)
        folder = data.get("folder", "")
        recursive = data.get("recursive", True)
        paths = list_mp3s(folder, recursive)
        count = sum(1 for p in paths if rename_tag(p, data["old"], data["new"]))
        return jsonify({"ok": True, "renamed": count})

    @bp.route("/api/tags/delete", methods=["POST"])
    def api_delete_tag():
        """Delete a tag from all files in a folder.

        Body: {"name": "energy", "folder": "", "recursive": true}
        """
        data = request.get_json(force=True)
        folder = data.get("folder", "")
        recursive = data.get("recursive", True)
        paths = list_mp3s(folder, recursive)
        for p in paths:
            delete_tag(p, data["name"])
        return jsonify({"ok": True, "files": len(paths)})

    return bp
