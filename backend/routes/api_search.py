"""Semantic text search over the cached CLAP audio embeddings.

A free-text query is encoded with CLAP's text tower and ranked by cosine similarity
against the cached per-segment CLAP audio vectors — the same 512-D space the layout's
semantic weighting already projects tag/folder text into. Each track scores on its
best-matching segment, and the hit carries that segment's centre as a ``section``
fraction so the preview can audition the relevant part of the track. Tracks analysed
before per-segment vectors existed fall back to the pooled mean (no section).

This is **read-only over `FeatureCache`**: nothing re-runs an encoder over the
library. The text forward pass runs on the request thread, guarded by
`clap.inference_lock` inside `generate_text_embeddings` — the exact path semantic
weighting already uses during a projection request, so it adds no new concurrency
risk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
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
    tracks: Sequence[tuple[str, np.ndarray, np.ndarray | None]],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank *tracks* by cosine similarity to *query_vec*, descending.

    Each entry is ``(path, vectors, positions)``: *vectors* is either a single
    mean vector of shape ``(dim,)`` or a matrix of per-segment vectors of shape
    ``(n, dim)``, and *positions* is the per-segment centre as a fraction of the
    track duration (``None`` for a mean vector). A track scores on its
    best-matching segment and the hit carries that segment's ``section`` fraction so
    the preview can audition the relevant part of the track.

    Both sides are L2-normalised first: cached CLAP audio rows are the mean of three
    segment vectors and are not unit-length. The sort is stable, so equal scores
    keep their input (path-sorted) order.
    """
    if not tracks:
        return []
    q = _l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
    scored: list[tuple[float, int, str, float | None]] = []
    for idx, (path, vectors, positions) in enumerate(tracks):
        m = np.asarray(vectors, dtype=np.float32)
        if m.ndim == 1:
            m = m.reshape(1, -1)
        sims = _l2_normalize(m) @ q
        pos: np.ndarray | None = None
        if positions is not None:
            p = np.asarray(positions, dtype=np.float32).reshape(-1)
            if p.size == sims.size:
                pos = p
                # Absent segments are stored as NaN; never let them win.
                sims = np.where(np.isfinite(p), sims, -np.inf)
        best = int(np.argmax(sims))
        section = (
            float(pos[best])
            if pos is not None and np.isfinite(pos[best])
            else None
        )
        scored.append((float(sims[best]), idx, path, section))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [
        {"path": path, "score": score, "section": section}
        for score, _idx, path, section in scored[:limit]
    ]


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
        seg_map = feature_cache.get_all(
            "clap_segments", fps, version=embedding_version
        )
        pos_map = feature_cache.get_all(
            "clap_segment_pos", fps, version=embedding_version
        )

        dim = int(np.asarray(text_vecs[0]).reshape(-1).size)
        tracks: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        for t in infos:
            fp = t.get("fingerprint")
            if not fp:
                continue
            seg = seg_map.get(fp)
            if seg is not None and seg.size:
                vecs = np.asarray(seg, dtype=np.float32).reshape(-1, dim)
                pos = pos_map.get(fp)
                positions = (
                    np.asarray(pos, dtype=np.float32).reshape(-1)
                    if pos is not None and pos.size == vecs.shape[0]
                    else None
                )
                tracks.append((t["path"], vecs, positions))
            elif fp in emb_map:
                tracks.append(
                    (t["path"], np.asarray(emb_map[fp], dtype=np.float32), None)
                )

        results = rank_by_similarity(text_vecs[0], tracks, limit)
        return jsonify({
            "ok": True,
            "query": query,
            "results": results,
            "searched": len(tracks),
            "total_tracks": len(infos),
        })

    return bp
