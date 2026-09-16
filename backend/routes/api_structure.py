"""Per-track structure (intro / drop / break) for the hover preview and tooltip.

Read-only over ``FeatureCache``: the ``structure`` blob is written by the queue's
``structure`` stage, and the genre-spanning flag is derived from the cached
``maest_excerpts`` blob. Only the requested paths are resolved (via the track cache),
so a hover is O(requested paths), not O(library) — no full-library scan per request.

Absent sections are stored as ``NaN`` and surface as ``null``. Paths with no cached
structure row are simply omitted from the response — the frontend falls back to a fixed
preview position until the background backfill reaches the track.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from flask import Blueprint, jsonify, request

from backend.embeddings import structure as structure_mod
from backend.embeddings import supervised as supervised_mod
from backend.services.request_validation import ApiValidationError

log = logging.getLogger(__name__)

#: Upper bound on paths per request, so one hover cannot resolve the whole library.
MAX_PATHS = 64


def create_api_structure_blueprint(
    *,
    feature_cache: Any,
    structure_version: int,
    maest_version: int,
    resolve_virtual_path: Callable[[str], str],
    cached_read_all: Callable[[str], dict[str, Any]],
) -> Blueprint:
    bp = Blueprint("api_structure", __name__)

    @bp.route("/api/tracks/structure")
    def api_structure():
        raw = request.args.getlist("paths")
        if len(raw) == 1 and "," in raw[0]:
            raw = raw[0].split(",")
        wanted = [p.strip() for p in raw if p.strip()]
        if not wanted:
            raise ApiValidationError("paths is required")
        if len(wanted) > MAX_PATHS:
            raise ApiValidationError(f"at most {MAX_PATHS} paths per request")

        by_path: dict[str, str] = {}
        fps: list[str] = []
        for path in wanted:
            try:
                abs_path = resolve_virtual_path(path)
                info = cached_read_all(abs_path) if abs_path else {}
            except Exception:
                continue
            fp = info.get("fingerprint")
            if isinstance(fp, str) and fp:
                by_path[path] = fp
                fps.append(fp)

        structures = feature_cache.get_all(
            "structure", fps, version=structure_version
        )
        excerpts = feature_cache.get_all(
            "maest_excerpts", fps, version=maest_version
        )

        tracks: list[dict[str, Any]] = []
        for path, fp in by_path.items():
            blob = structures.get(fp)
            if blob is None:
                continue
            entry: dict[str, Any] = {"path": path}
            entry.update(structure_mod.decode_structure(blob))
            exc = excerpts.get(fp)
            entry["genre_spanning"] = supervised_mod.is_genre_spanning(
                supervised_mod.decode_excerpts(exc) if exc is not None else None
            )
            tracks.append(entry)

        return jsonify({"ok": True, "tracks": tracks})

    return bp
