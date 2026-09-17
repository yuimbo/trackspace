"""HTTP surface for imprinting the head's folder structure on an external folder.

An *imprint* is a folder fan-out whose target is outside the library. The target
is registered as an **ephemeral root** — enough for the scan, analysis, and fan-out
paths to resolve it — and removed again once the user finishes. This module only
owns the target's lifetime and its analysis; the plan and the moves themselves are
the existing ``/api/supervised/fanout`` route.

Flow: ``analyze`` registers the root and queues analysis, ``status`` reports how many
tracks are ready, ``fanout`` plans/applies the moves, and ``finish`` drops the root.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, MutableMapping, MutableSet
from typing import Any

from flask import Blueprint, jsonify, request

from backend.jobs import KIND_CLAP, KIND_FEATURES, KIND_MAEST
from backend.services import analysis_pipeline as pipeline
from backend.services.request_validation import (
    opt_bool,
    opt_str,
    parse_json_body,
    require_str,
)

log = logging.getLogger(__name__)


def create_api_imprint_blueprint(
    *,
    analysis_deps: Any,
    roots: MutableMapping[str, str],
    temp_roots: MutableSet[str],
    add_root: Callable[[str], tuple[bool, str]],
    remove_root: Callable[[str], bool],
    schedule_root_watch: Callable[[str], None],
    unschedule_root_watch: Callable[[str], None],
    save_roots_state: Callable[[], None],
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
    enqueue_library_work_fn: Callable[..., dict[str, int]],
    virtual_from_abs: Callable[[str], str],
    job_queue: Any = None,
    workers: Any = None,
) -> Blueprint:
    bp = Blueprint("api_imprint", __name__)

    @bp.route("/api/supervised/imprint/analyze", methods=["POST"])
    def api_imprint_analyze():
        """Register an external folder and queue analysis for its tracks.

        A folder already covered by a root is used as-is (no ephemeral root is
        added), so the user can also imprint a subtree of the library. The response
        carries the track counts so the UI can show analysis progress before the
        fan-out plan is requested.
        """
        data = parse_json_body()
        raw_path = require_str(data, "path").strip()
        recursive = opt_bool(data, "recursive", True)
        if not getattr(analysis_deps, "clap_enabled", True):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "CLAP is disabled on this server, so the folder head "
                            "cannot place tracks."
                        ),
                    }
                ),
                400,
            )
        abs_path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.isdir(abs_path):
            return jsonify({"ok": False, "error": "Directory not found"}), 400
        # ``add_root`` drops redundant descendant roots, so registering a folder
        # that *contains* a library root would silently unregister that root.
        prefix = abs_path.rstrip(os.sep) + os.sep
        if any(
            root_abs != abs_path and root_abs.startswith(prefix)
            for root_abs in roots.values()
        ):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "That folder contains library roots; choose a folder "
                            "that does not contain your library."
                        ),
                    }
                ),
                400,
            )

        changed, root_id = add_root(abs_path)
        if changed:
            temp_roots.add(root_id)
            schedule_root_watch(root_id)
            save_roots_state()
        try:
            folder = virtual_from_abs(abs_path)
        except Exception:
            return jsonify({"ok": False, "error": "Folder is not covered by a root"}), 400

        infos = build_track_infos_fn(folder, recursive)
        queued = enqueue_library_work_fn(
            job_queue,
            infos,
            deps=analysis_deps,
            kinds=(KIND_FEATURES, KIND_MAEST, KIND_CLAP),
        )
        if workers is not None:
            workers.wake()
        status = pipeline.supervised_imprint_status(analysis_deps, infos)
        return jsonify(
            {
                "ok": True,
                "root_id": root_id,
                "folder": folder,
                "added": changed,
                "queued": queued,
                **status,
            }
        )

    @bp.route("/api/supervised/imprint/status")
    def api_imprint_status():
        """How many tracks under *folder* are ready to be placed."""
        folder = request.args.get("folder", "").strip()
        if not folder:
            return jsonify({"ok": False, "error": "folder is required"}), 400
        recursive = request.args.get("recursive", "1") == "1"
        infos = build_track_infos_fn(folder, recursive)
        return jsonify(
            {
                "ok": True,
                "folder": folder,
                **pipeline.supervised_imprint_status(analysis_deps, infos),
            }
        )

    @bp.route("/api/supervised/imprint/finish", methods=["POST"])
    def api_imprint_finish():
        """Remove an ephemeral imprint root once its moves are done.

        Only roots this module registered are removed: a target that was already
        covered by a library root must not be dropped.
        """
        data = parse_json_body(required=False)
        root_id = opt_str(data, "root_id", "").strip()
        if not root_id:
            return jsonify({"ok": False, "error": "root_id is required"}), 400
        if root_id not in temp_roots:
            return jsonify({"ok": True, "removed": False})
        temp_roots.discard(root_id)
        remove_root(root_id)
        unschedule_root_watch(root_id)
        save_roots_state()
        return jsonify({"ok": True, "removed": True})

    return bp
