"""Per-track similarity: nearest neighbours and the "anti-track".

Read-only over ``FeatureCache`` through the same weighted composite matrix that
clustering and the layout use, so a track's nearest/furthest neighbours mean what
the map means. The matrix is built lazily by :class:`SimilarityIndex` (cached by
clustering revision + weights + library mtime), never on a hot path: the endpoint
is only hit by an explicit inspect action.

Only ``path`` and ``score`` are returned — the frontend already holds every track's
artist/title/folder, so the response stays O(k).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from flask import Blueprint, jsonify, request

from backend.services.analysis_pipeline import gather_source_blocks
from backend.services.request_validation import (
    one_of,
    opt_int,
    require_str,
)
from backend.services.similarity import SimilarityIndex

log = logging.getLogger(__name__)

DEFAULT_K = 5
MAX_K = 25
MAX_PATH_LEN = 1024


def create_api_similar_blueprint(
    *,
    analysis_deps: Any,
    index: SimilarityIndex,
    clusters: Any,
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
    library_mtime_signature: Callable[[str, bool], str],
) -> Blueprint:
    bp = Blueprint("api_similar", __name__)

    @bp.route("/api/tracks/similar")
    def api_similar():
        path = require_str(request.args, "path", max_len=MAX_PATH_LEN)
        k = opt_int(request.args, "k", DEFAULT_K, lo=1, hi=MAX_K)
        assert k is not None  # a default was supplied
        mode = one_of(request.args.get("mode", "near"), ("near", "far"), field="mode")

        result = clusters.get()
        # The weights actually used by the current clusters are the weights the map
        # shows; fall back to the requested setting, then to the frozen defaults.
        weights = (
            result.weights
            if result is not None
            else clusters.weights()
        )
        key = (
            result.revision if result is not None else None,
            tuple(sorted((weights or {}).items())),
            library_mtime_signature("", True),
        )

        hit = index.lookup(key)
        if hit is None:
            infos = build_track_infos_fn("", True)
            paths, blocks, _logits = gather_source_blocks(
                analysis_deps, infos, clap_version=analysis_deps.clap_version
            )
            if not paths:
                return (
                    jsonify({"ok": False, "error": "No analysed tracks yet"}),
                    503,
                )
            hit = index.build(key, paths, blocks, weights)
        if hit is None:
            return jsonify({"ok": False, "error": "No usable embedding vectors"}), 503

        paths, matrix, row = hit
        idx = row.get(path)
        if idx is None:
            return (
                jsonify(
                    {"ok": False, "error": "This track has no cached embedding"}
                ),
                404,
            )

        results = index.rank(paths, matrix, idx, k, mode)
        return jsonify(
            {
                "ok": True,
                "path": path,
                "mode": mode,
                "n_tracks": len(paths),
                "results": results,
            }
        )

    return bp
