from __future__ import annotations

from collections.abc import Callable

from flask import Blueprint, jsonify

from backend.services.request_validation import (
    opt_bool,
    opt_float,
    opt_str,
    parse_json_body,
    require_mapping_list,
    require_str,
)
from backend.tags import delete_tag, rename_tag, write_tag


def create_api_tags_blueprint(
    *,
    list_mp3s: Callable[[str, bool], list[str]],
    resolve_virtual_path: Callable[[str], str],
) -> Blueprint:
    bp = Blueprint("api_tags", __name__)

    def _validate_update(item: dict) -> tuple[str, str, float | None]:
        """Structurally validate one update; raises ApiValidationError."""
        rel = require_str(item, "path")
        tagname = require_str(item, "tag")
        value = opt_float(item, "value", default=None, lo=0.0, hi=1.0)
        return rel, tagname, value

    @bp.route("/api/tracks/tags", methods=["POST"])
    def api_update_tags():
        """Batch-update tag values.

        Expects JSON body:
            { "updates": [ {"path": "rel/file.mp3", "tag": "energy", "value": 0.7}, … ] }

        A null value deletes the tag. The whole payload is validated up front;
        per-file failures (missing file, unwritable ID3 frame) are collected in
        ``errors`` so one bad track does not abort the batch. Successful writes
        stay applied.
        """
        data = parse_json_body()
        updates = require_mapping_list(data, "updates")
        validated = [_validate_update(u) for u in updates]

        count = 0
        errors: list[dict[str, str]] = []
        for rel, tagname, value in validated:
            try:
                abs_path = resolve_virtual_path(rel)
                if value is None:
                    delete_tag(abs_path, tagname)
                else:
                    write_tag(abs_path, tagname, value)
                count += 1
            except Exception as exc:
                errors.append({"path": rel, "error": str(exc)})
        return jsonify({"ok": True, "count": count, "errors": errors})

    @bp.route("/api/tags/rename", methods=["POST"])
    def api_rename_tag():
        """Rename a tag across all files in a folder.

        Body: {"old": "energy", "new": "vibe", "folder": "", "recursive": true}
        """
        data = parse_json_body()
        old = require_str(data, "old")
        new = require_str(data, "new")
        folder = opt_str(data, "folder", "")
        recursive = opt_bool(data, "recursive", True)

        paths = list_mp3s(folder, recursive)
        count = 0
        errors: list[dict[str, str]] = []
        for p in paths:
            try:
                if rename_tag(p, old, new):
                    count += 1
            except Exception as exc:
                errors.append({"path": p, "error": str(exc)})
        return jsonify({"ok": True, "renamed": count, "errors": errors})

    @bp.route("/api/tags/delete", methods=["POST"])
    def api_delete_tag():
        """Delete a tag from all files in a folder.

        Body: {"name": "energy", "folder": "", "recursive": true}
        """
        data = parse_json_body()
        name = require_str(data, "name")
        folder = opt_str(data, "folder", "")
        recursive = opt_bool(data, "recursive", True)

        paths = list_mp3s(folder, recursive)
        errors: list[dict[str, str]] = []
        for p in paths:
            try:
                delete_tag(p, name)
            except Exception as exc:
                errors.append({"path": p, "error": str(exc)})
        return jsonify({"ok": True, "files": len(paths), "errors": errors})

    return bp
