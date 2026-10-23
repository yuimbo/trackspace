"""Semantic folder colour anchors: which ancestor a folder is a variation of.

``dir_color`` paints a folder as a tint of an ancestor's hue — its *anchor*. The
default anchor is structural (the top-level folder), which assumes a curated taxonomy
where every subfolder is a variation of its playlist. Real libraries mix a coherent
genre tree (``GENRES/<genre>/<subgenre>``) with folders of unrelated tracks
(downloads, scratch), where siblings are not variations at all.

This module decides the anchor from the tracks themselves. A child is a variation of
its parent when the two folders' subtree centroids are close in the same weighted
composite space the map and the clustering use — a sub-genre. Otherwise the child
starts a new colour family. Folders too thin to measure fall back to the root role:
external roots are never variations (their contents are unrelated), library roots keep
the structural anchor, so an unanalysed library is coloured exactly as before.

Read-only over ``FeatureCache`` and cached by library signature, so the folder tree
and the canvas share one computation and a burst of redraws pays it once. The build
runs off the request thread and streams ``colors_progress`` phases, so a large
library's walk never stalls the library load and is visible while it runs.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import numpy as np

from backend.embeddings.clustering import build_weighted_matrix
from backend.embeddings.linalg_utils import l2_normalize
from backend.embeddings.projection_config import FROZEN_SOURCE_WEIGHTS
from backend.services.analysis_pipeline import gather_source_blocks
from backend.services.roots_registry import ROLE_EXTERNAL, root_role

log = logging.getLogger(__name__)

#: Fewest analysed tracks a folder (or its parent) needs before its tracks may
#: decide the anchor. Below this the root role decides instead.
MIN_ANCHOR_TRACKS = 4

#: A child counts as a variation of its parent when their subtree centroids are at
#: least this cosine-similar in the frozen composite space. Tuned so a sub-genre
#: stays in the family while an unrelated folder does not; the one knob to tune
#: against a real library.
VARIATION_MIN_COSINE = 0.6

#: Rebuild the cached map after this long even if the library signature is
#: unchanged, so a long-running server cannot serve a stale map forever.
CACHE_TTL_SECONDS = 300.0


def _folder_of(path: str) -> str:
    """Leaf folder of a track's virtual path (``Root/Sub/x.mp3`` → ``Root/Sub``)."""
    norm = str(path).replace("\\", "/").rstrip("/")
    if "/" not in norm:
        return ""
    return norm.rsplit("/", 1)[0]


def _ancestors(folder: str) -> list[str]:
    """*folder* and every ancestor up to the root, shortest first."""
    parts = folder.split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each row, keeping the shape (a 1-D row stays 1-D)."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        return l2_normalize(arr.reshape(1, -1))[0]
    return l2_normalize(arr)


def compute_folder_anchors(
    roots: Iterable[str],
    paths: Sequence[str],
    matrix: np.ndarray,
    *,
    external_root_ids: Collection[str] = (),
    min_tracks: int = MIN_ANCHOR_TRACKS,
    variation_min_cosine: float = VARIATION_MIN_COSINE,
) -> dict[str, str]:
    """Anchor every folder to the ancestor whose hue family it belongs to.

    *matrix* is the row-aligned weighted composite matrix (the map's space).
    A root is its own anchor and a top-level folder always starts its own family
    (a playlist, not a variation). Every deeper folder is walked shallow-first: it
    is a variation of its parent when the two subtree centroids are at least
    *variation_min_cosine* similar, and a variation inherits its parent's anchor.
    A folder too thin to measure falls back to the root role: external roots are
    distinct, library roots keep the structural anchor.
    """
    root_list = sorted({str(r) for r in roots if str(r)})
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != len(paths) or not root_list:
        return {}

    unit = _normalize_rows(m)
    folder_set = set(root_list)
    for p in paths:
        folder = _folder_of(str(p))
        if folder:
            folder_set.update(_ancestors(folder))

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for i, p in enumerate(paths):
        folder = _folder_of(str(p))
        if not folder:
            continue
        for anc in _ancestors(folder):
            if anc not in folder_set:
                continue
            if anc not in sums:
                sums[anc] = np.zeros(m.shape[1], dtype=np.float32)
            sums[anc] += unit[i]
            counts[anc] = counts.get(anc, 0) + 1
    centroids = {
        f: l2_normalize(vec.reshape(1, -1))[0] for f, vec in sums.items()
    }

    external = {str(r) for r in external_root_ids}
    anchors: dict[str, str] = {}
    for folder in sorted(folder_set, key=lambda f: (f.count("/"), f)):
        depth = folder.count("/")
        if depth <= 1:
            # A root and each top-level playlist is its own family.
            anchors[folder] = folder
            continue
        parent = folder.rsplit("/", 1)[0]
        parent_anchor = anchors.get(parent, parent)
        if (
            counts.get(folder, 0) >= min_tracks
            and counts.get(parent, 0) >= min_tracks
        ):
            sim = float(np.dot(centroids[folder], centroids[parent]))
            variation = sim >= variation_min_cosine
        else:
            # Too thin to measure: external roots are never variations (their
            # contents are unrelated), a library keeps the structural anchor.
            variation = folder.split("/", 1)[0] not in external
        anchors[folder] = parent_anchor if variation else folder
    return anchors


class FolderAnchors:
    """Background-built, library-signature-keyed folder anchor map.

    The map depends only on the analysed tracks and the root roles, so the key is
    ``(library signature, roots, external roots)``: an analysis finishing, a root
    being added or removed, or a role changing each rebuild it. The build runs on a
    daemon thread — the tree partial only ever reads the cache — so the library
    load never waits on a full-library gather. *weights* defaults to the frozen
    lens so the colour families do not move when the clustering weights are
    tweaked.
    """

    def __init__(
        self,
        deps: Any,
        *,
        roots: Any,
        root_roles: Any,
        build_track_infos: Callable[..., list[dict[str, Any]]],
        library_mtime_signature: Callable[[str, bool], str],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        min_tracks: int = MIN_ANCHOR_TRACKS,
        variation_min_cosine: float = VARIATION_MIN_COSINE,
    ) -> None:
        self._deps = deps
        self._roots = roots
        self._root_roles = root_roles
        self._build_track_infos = build_track_infos
        self._library_mtime_signature = library_mtime_signature
        self._on_progress = on_progress
        self._min_tracks = int(min_tracks)
        self._variation_min_cosine = float(variation_min_cosine)
        self._lock = threading.RLock()
        self._key: tuple[Any, ...] | None = None
        self._built_at = 0.0
        self._anchors: dict[str, str] = {}
        self._building = False
        self._thread: threading.Thread | None = None

    def _emit(self, phase: str, done: int = 0, total: int = 0) -> None:
        if self._on_progress is not None:
            self._on_progress(
                {"phase": phase, "done": int(done), "total": int(total)}
            )

    def get(self) -> dict[str, str]:
        """Current anchor map; never walks or blocks on the request thread.

        The folder-tree partial renders on the request thread while the library is
        still scanning, so a synchronous full-library walk here would stall the
        loading overlay at the last track. Within the TTL the cache is returned
        as-is; otherwise one background refresh recomputes the signature and
        rebuilds, and the next tree refresh (a drained run or the completed scan)
        picks up the result.
        """
        with self._lock:
            if (
                self._key is not None
                and time.time() - self._built_at < CACHE_TTL_SECONDS
            ):
                return self._anchors
            if self._building:
                return self._anchors
            self._building = True
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return self._anchors

    def wait(self, timeout: float | None = None) -> None:
        """Block until the in-flight refresh finishes (eager warmup / tests)."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _refresh(self) -> None:
        anchors = self._anchors
        try:
            key = (
                self._library_mtime_signature("", True),
                tuple(sorted(self._roots)),
                tuple(
                    sorted(
                        rid
                        for rid in self._roots
                        if root_role(rid, self._root_roles) == ROLE_EXTERNAL
                    )
                ),
            )
            with self._lock:
                unchanged = self._key == key
            if not unchanged:
                anchors = self._compute(key)
        except Exception:
            log.exception("Folder anchors: computation failed")
            key = self._key  # type: ignore[assignment]
        with self._lock:
            if key is not None:
                self._key = key
            self._built_at = time.time()
            self._anchors = anchors
            self._building = False

    def _compute(self, key: tuple[Any, ...]) -> dict[str, str]:
        try:
            self._emit("scanning")
            infos = self._build_track_infos(
                "", True, on_progress=lambda n: self._emit("scanning", n)
            )
            self._emit("reading", 0, 6)
            paths, blocks, _logits = gather_source_blocks(
                self._deps,
                infos,
                clap_version=self._deps.clap_version,
                on_progress=lambda done, total: self._emit("reading", done, total),
            )
            if not paths:
                return {}
            self._emit("matrix")
            matrix, _used, _effective = build_weighted_matrix(
                blocks, FROZEN_SOURCE_WEIGHTS
            )
            return compute_folder_anchors(
                self._roots,
                paths,
                matrix,
                external_root_ids=key[2],
                min_tracks=self._min_tracks,
                variation_min_cosine=self._variation_min_cosine,
            )
        except Exception:
            log.exception("Folder anchors: computation failed")
            return {}
        finally:
            self._emit("done")
