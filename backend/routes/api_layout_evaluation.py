"""Read-only GET endpoints for the offline layout-evaluation artifacts.

Serves the precomputed maps the ``backend.precompute_layout_evaluation`` CLI
writes. GET only — there is no write/upload surface. All reads go through
:mod:`backend.embeddings.layout_evaluation`, which only lists recipes that the
current manifest marks ``complete`` (never stale artifacts from a previous
dataset), and which rejects traversal, malformed JSON, and non-finite
coordinates.
"""

from __future__ import annotations

import os
import re

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import NotFound

from backend.embeddings.layout_evaluation import (
    load_completed,
    load_recipe_json,
    manifest_dict,
)
from backend.services.request_validation import ApiValidationError, one_of

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

#: Named gallery sets. The value is the fixed subdirectory of the base
#: layout-evaluation dir the set's artifacts live in (``None`` = the base dir
#: itself). Only these names are ever resolved — never arbitrary path input.
SET_SUBDIRS: dict[str, str | None] = {
    "base": None,
    "budget": "budget",
    "scale": "scale",
    "early": "early",
    "mixes": "mixes",
    "ablation": "ablation",
}


def _resolve_dir(set_name: str | None, base_dir: str) -> str:
    """Resolve a ``set`` query param to a fixed dir under *base_dir*.

    ``None`` (no ``set``) keeps the base dir. Anything not on the explicit
    allowlist (:data:`SET_SUBDIRS`) is a 400, and a named set's dir is always a
    fixed child of the base — never derived from caller input.
    """
    if set_name is None:
        return base_dir
    one_of(set_name, SET_SUBDIRS, field="set")
    subdir = SET_SUBDIRS[set_name]
    return os.path.join(base_dir, subdir) if subdir else base_dir


def create_api_layout_evaluation_blueprint(layout_eval_base_dir: str) -> Blueprint:
    """Build the blueprint reading artifacts from *layout_eval_base_dir*.

    A ``?set=<name>`` query param selects one of the fixed named sets (e.g.
    ``budget`` under ``<base>/budget``); without it the base dir (the original
    eight-map gallery) is served.
    """
    bp = Blueprint("api_layout_evaluation", __name__)

    @bp.route("/api/layout-evaluation")
    def api_layout_evaluation_index():
        layout_eval_dir = _resolve_dir(request.args.get("set"), layout_eval_base_dir)
        manifest = manifest_dict(layout_eval_dir)
        if not manifest:
            raise NotFound(description="no layout-evaluation generation found")
        return jsonify({"ok": True, "manifest": manifest})

    @bp.route("/api/layout-evaluation/<path:recipe_id>")
    def api_layout_evaluation_recipe(recipe_id: str):
        if not _SAFE_ID.match(recipe_id):
            raise ApiValidationError(
                f"field 'recipe_id' must match [A-Za-z0-9_-]+ "
                f"(got {recipe_id!r})"
            )
        layout_eval_dir = _resolve_dir(request.args.get("set"), layout_eval_base_dir)
        completed = load_completed(layout_eval_dir)
        if recipe_id not in completed:
            raise NotFound(
                description=f"no completed layout-evaluation recipe {recipe_id!r}"
            )
        artifact = load_recipe_json(layout_eval_dir, recipe_id)
        if artifact is None:
            # The manifest lists it as complete but the artifact is missing,
            # corrupt, or has non-finite coordinates — treat as a 500 so the
            # client never receives a broken map.
            from flask import abort
            abort(500, description="layout-evaluation artifact is missing or invalid")
        return jsonify({"ok": True, **artifact})

    return bp
