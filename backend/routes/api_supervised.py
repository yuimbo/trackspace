"""HTTP surface for folder-supervised heads: info, export, import, compare, fan-out.

The head is a portable artifact (weights + meta + per-folder centroids), so these
routes let a user move an organization system between libraries. Two of them mutate
state — import overwrites the installed head and fan-out moves files on disk — so both
validate their input and report a client-safe JSON error rather than a 500. Fan-out is
dry-run by default; only ``apply`` touches disk.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from typing import Any

from flask import (
    Blueprint,
    after_this_request,
    jsonify,
    request,
    send_file,
)

from backend.embeddings import supervised as supervised_mod
from backend.services import analysis_pipeline as pipeline
from backend.services.request_validation import (
    ApiValidationError,
    opt_bool,
    opt_str,
    parse_json_body,
)

log = logging.getLogger(__name__)


def _read_upload(attr: str) -> tuple[dict, str]:
    """Save one uploaded bundle to a temp file and parse it.

    The caller owns the temp file; this returns ``(bundle, path)`` so the route can
    clean it up after the response has been sent.
    """
    upload = request.files.get(attr)
    if upload is None or not upload.filename:
        raise ApiValidationError("no head bundle file uploaded")
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    upload.save(path)
    return supervised_mod.read_head_bundle(path), path


def create_api_supervised_blueprint(
    *,
    analysis_deps: Any,
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
) -> Blueprint:
    bp = Blueprint("api_supervised", __name__)

    def _cleanup_after(path: str) -> None:
        @after_this_request
        def _remove(response):
            try:
                os.unlink(path)
            except OSError:
                pass
            return response

    @bp.route("/api/supervised/head")
    def api_supervised_head():
        """Metadata for the installed head, or ``{"available": false}``."""
        return jsonify(pipeline.supervised_head_info(analysis_deps))

    @bp.route("/api/supervised/export")
    def api_supervised_export():
        """Download the installed head as a portable bundle."""
        bundle = pipeline.export_supervised_head(analysis_deps)
        if bundle is None:
            return jsonify({"ok": False, "error": "no trained head to export"}), 404
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        supervised_mod.write_head_bundle(path, bundle)
        _cleanup_after(path)
        return send_file(
            path, as_attachment=True, download_name="trackspace_head.pt"
        )

    @bp.route("/api/supervised/import", methods=["POST"])
    def api_supervised_import():
        """Install an uploaded bundle under a fresh local run id.

        A bundle trained on different encoder versions is refused with a 409 and a
        ``source_mismatches`` map; the client can retry with ``force=1`` once the
        user has accepted the mismatch.
        """
        try:
            bundle, path = _read_upload("bundle")
        except ApiValidationError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        _cleanup_after(path)
        force = (request.form.get("force") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            pipeline.import_supervised_head(
                analysis_deps, bundle, allow_mismatch=force
            )
        except supervised_mod.HeadVersionMismatch as e:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": str(e),
                        "source_mismatches": e.mismatches,
                    }
                ),
                409,
            )
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "head": pipeline.supervised_head_info(analysis_deps)})

    @bp.route("/api/supervised/compare", methods=["POST"])
    def api_supervised_compare():
        """Compare the installed head with an uploaded bundle."""
        current = pipeline.export_supervised_head(analysis_deps)
        if current is None:
            return jsonify({"ok": False, "error": "no trained head to compare"}), 400
        try:
            other, path = _read_upload("bundle")
        except ApiValidationError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        _cleanup_after(path)
        return jsonify(
            {"ok": True, "compare": supervised_mod.compare_heads(current, other)}
        )

    @bp.route("/api/supervised/fanout", methods=["POST"])
    def api_supervised_fanout():
        """Sort a folder's tracks into the head's folder subpaths.

        Dry-run unless ``apply`` is set; an invalid virtual folder or an input
        space that no longer matches the head is a 400 with a client-safe message.
        """
        data = parse_json_body(required=False)
        folder = opt_str(data, "folder", "")
        if not folder:
            return jsonify({"ok": False, "error": "folder is required"}), 400
        recursive = opt_bool(data, "recursive", False)
        apply = opt_bool(data, "apply", False)
        generic = opt_bool(data, "generic", False)
        try:
            result = pipeline.fanout_supervised(
                analysis_deps,
                build_track_infos=build_track_infos_fn,
                folder_vpath=folder,
                recursive=recursive,
                apply=apply,
                generic_unmatched=generic,
            )
        except ApiValidationError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @bp.route("/api/supervised/fanout/undo")
    def api_supervised_fanout_undo_status():
        """Whether a fan-out batch is available to undo, and how many moves."""
        batch = analysis_deps.feature_cache.latest_move_batch()
        return jsonify(
            {
                "ok": True,
                "available": batch is not None and bool(batch["moves"]),
                "total": len(batch["moves"]) if batch else 0,
            }
        )

    @bp.route("/api/supervised/fanout/undo", methods=["POST"])
    def api_supervised_fanout_undo():
        """Reverse the most recent applied fan-out batch's moves on disk."""
        result = pipeline.undo_fanout_supervised(analysis_deps)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    return bp
