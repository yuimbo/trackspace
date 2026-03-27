from __future__ import annotations

import os
from collections.abc import Callable

from flask import Blueprint, abort, send_file, send_from_directory


def create_static_audio_blueprint(
    *,
    dist_dir: str,
    resolve_virtual_path: Callable[[str], str],
) -> Blueprint:
    bp = Blueprint("static_audio", __name__)

    @bp.route("/")
    def index():
        if os.path.isdir(dist_dir):
            return send_from_directory(dist_dir, "index.html")
        return (
            "Frontend not built. Run <code>npm run build</code> in <code>frontend/</code>, "
            "or use the Vite dev server (<code>npm run dev</code>) for development."
        ), 404

    @bp.route("/assets/<path:filename>")
    def serve_assets(filename: str):
        return send_from_directory(os.path.join(dist_dir, "assets"), filename)

    @bp.route("/api/audio/<path:relpath>")
    def api_audio(relpath: str):
        """Stream an mp3 file for hover-preview playback."""
        abs_path = resolve_virtual_path(relpath)
        if not os.path.isfile(abs_path):
            abort(404)
        return send_file(abs_path, mimetype="audio/mpeg")

    return bp
