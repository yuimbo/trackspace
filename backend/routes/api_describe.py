"""Zero-shot text descriptions for one track from its cached CLAP audio embedding.

Read-only over ``FeatureCache``: only the requested path is resolved (via the track
cache), and the audio vector is the pooled ``clap`` row the layout/search use. Tracks
cached before the pooled row existed fall back to the mean of their ``clap_segments``
rows. A track with neither is a 404, not a re-encode.

The text bank is encoded once per process by ``clap_describe.PromptBank``; the request
thread only scores the cached matrix. That scoring needs no accelerator, and the one-time
encode runs under ``clap.inference_lock`` inside ``generate_text_embeddings`` — the same
serialisation the search route uses, so this adds no new concurrency risk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request

from backend.embeddings.clap_describe import (
    AxisDirections,
    PromptBank,
    coordinates,
    rank_descriptions,
)
from backend.services.request_validation import opt_int, require_str

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 8
MAX_LIMIT = 20
DEFAULT_COORDINATES = 6
MAX_COORDINATES = 12
MAX_PATH_LEN = 1024
#: Cosine floor for a phrase to be worth showing. Below this the match is noise,
#: so the list is short rather than padded; the top phrase is always kept, so a
#: track with only weak matches still shows its best one.
MIN_SIMILARITY = 0.35


def _audio_vector(feature_cache: Any, fp: str, version: int) -> np.ndarray | None:
    """The track's pooled CLAP vector, falling back to its segment mean."""
    row = feature_cache.get_all("clap", [fp], version=version).get(fp)
    if row is not None and getattr(row, "size", 0):
        return np.asarray(row, dtype=np.float32).reshape(-1)
    seg = feature_cache.get_all("clap_segments", [fp], version=version).get(fp)
    if seg is not None and getattr(seg, "size", 0):
        mat = np.asarray(seg, dtype=np.float32)
        return mat.reshape(-1) if mat.ndim == 1 else mat.mean(axis=0)
    return None


def create_api_describe_blueprint(
    *,
    clap_mod: Any,
    feature_cache: Any,
    embedding_version: int,
    resolve_virtual_path: Callable[[str], str],
    cached_read_all: Callable[[str], dict[str, Any]],
    bank: PromptBank | None = None,
    axis_directions: AxisDirections | None = None,
) -> Blueprint:
    bp = Blueprint("api_describe", __name__)
    prompt_bank = bank or PromptBank()
    axes = axis_directions or AxisDirections()

    @bp.route("/api/tracks/describe")
    def api_describe():
        path = require_str(request.args, "path", max_len=MAX_PATH_LEN).strip()
        limit = opt_int(request.args, "limit", DEFAULT_LIMIT, lo=1, hi=MAX_LIMIT)
        assert limit is not None  # a default was supplied

        # Never load the model from a request: ``--no-clap`` must stay a no-op.
        if not clap_mod.is_model_ready():
            error = (
                clap_mod.load_error()
                or "CLAP model is still loading — try again shortly"
            )
            return jsonify({"ok": False, "error": error}), 503

        try:
            abs_path = resolve_virtual_path(path)
            info = cached_read_all(abs_path) if abs_path else {}
        except Exception:
            info = {}
        fp = info.get("fingerprint")
        if not isinstance(fp, str) or not fp:
            return jsonify({"ok": False, "error": "Unknown track"}), 404

        audio = _audio_vector(feature_cache, fp, embedding_version)
        if audio is None:
            return (
                jsonify(
                    {"ok": False, "error": "This track has no cached CLAP embedding"}
                ),
                404,
            )

        matrix = prompt_bank.matrix(clap_mod.generate_text_embeddings)
        if matrix is None:
            return jsonify({"ok": False, "error": "CLAP text encoder unavailable"}), 503

        results = [
            r
            for i, r in enumerate(
                rank_descriptions(audio, matrix, prompt_bank.prompts, limit)
            )
            if i == 0 or r["score"] >= MIN_SIMILARITY
        ]
        dirs = axes.directions(clap_mod.generate_text_embeddings)
        coords = (
            coordinates(audio, dirs, axes.axes, DEFAULT_COORDINATES)
            if dirs is not None
            else []
        )
        return jsonify(
            {
                "ok": True,
                "path": path,
                "results": results,
                "coordinates": coords,
                "n_descriptions": len(prompt_bank.prompts),
            }
        )

    return bp
