"""Semantic text search over the cached CLAP audio embeddings.

A free-text query is encoded with CLAP's text tower and ranked by cosine similarity
against every cached CLAP audio embedding — the same 512-D space the layout's
semantic weighting already projects tag/folder text into.

This is **read-only over `FeatureCache`**: nothing re-runs an encoder over the
library. The text forward pass runs on the request thread, guarded by
`clap.inference_lock` inside `generate_text_embeddings` — the exact path semantic
weighting already uses during a projection request, so it adds no new concurrency
risk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request

from backend.services.request_validation import opt_int, require_str

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MAX_QUERY_LEN = 200


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a floor, so a zero row cannot divide by zero."""
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    np.maximum(norms, 1e-9, out=norms)
    return mat / norms


def rank_by_similarity(
    query_vec: np.ndarray,
    paths: list[str],
    vecs: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank *paths* by cosine similarity to *query_vec*, descending.

    Both sides are L2-normalised first: cached CLAP audio rows are the mean of three
    segment vectors and are not unit-length. The sort is stable, so identical
    fingerprints (same audio, two paths) keep their input (path-sorted) order.
    """
    if not paths:
        return []
    q = _l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
    m = _l2_normalize(np.asarray(vecs, dtype=np.float32))
    scores = m @ q
    order = np.argsort(-scores, kind="stable")[:limit]
    return [{"path": paths[int(i)], "score": float(scores[int(i)])} for i in order]


def create_api_search_blueprint(
    *,
    clap_mod: Any,
    feature_cache: Any,
    embedding_version: int,
    build_track_infos_fn: Callable[[str, bool], list[dict[str, Any]]],
) -> Blueprint:
    bp = Blueprint("api_search", __name__)

    @bp.route("/api/search")
    def api_search():
        query = require_str(request.args, "q", max_len=MAX_QUERY_LEN).strip()
        limit = opt_int(request.args, "limit", DEFAULT_LIMIT, lo=1, hi=MAX_LIMIT)
        assert limit is not None  # a default was supplied

        # Never load the model from a request: ``--no-clap`` must stay a no-op.
        if not clap_mod.is_model_ready():
            error = clap_mod.load_error() or "CLAP model is still loading — try again shortly"
            return jsonify({"ok": False, "error": error}), 503

        text_vecs = clap_mod.generate_text_embeddings([query])
        if text_vecs is None or len(text_vecs) == 0:
            return jsonify({"ok": False, "error": "CLAP text encoder unavailable"}), 503

        infos = build_track_infos_fn("", True)
        fps = [t["fingerprint"] for t in infos if t.get("fingerprint")]
        emb_map = feature_cache.get_all("clap", fps, version=embedding_version)

        paths: list[str] = []
        vecs: list[np.ndarray] = []
        for t in infos:
            fp = t.get("fingerprint")
            if fp and fp in emb_map:
                paths.append(t["path"])
                vecs.append(emb_map[fp])

        results = rank_by_similarity(
            text_vecs[0], paths, np.asarray(vecs, dtype=np.float32), limit
        )
        return jsonify({
            "ok": True,
            "query": query,
            "results": results,
            "searched": len(paths),
            "total_tracks": len(infos),
        })

    return bp
