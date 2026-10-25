"""Gallery serialization for the genre-space maps: schema-1 manifest + recipes.

Co-locates everything the ``maps --gallery-output`` path needs that the maps
CLI otherwise does not: safe deterministic gallery ids, the gallery dataset
fingerprint (cohort fp/order + every source artifact checksum + projection
config), per-recipe revisions, persisted-roots → virtual-path mapping, and
atomic manifest/artifact writes.

The gallery is schema-1 compatible with the read-only layout-evaluation surface
(``layout_evaluation.load_recipe_json`` and the ``genre`` sets in
``api_layout_evaluation.SET_SUBDIRS``), so the existing browser blind-eval UI
renders it unchanged: manifest keys are the safe gallery ids, each recipe is a
``<safeid>.json`` artifact, and ``positions`` carry *virtual* paths (``rid/rel``)
resolved against the app's persisted roots — never absolute paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import numpy as np

#: Projection parameters the maps step always uses (one fixed PCA50 → openTSNE
#: representation per recipe; no new dimension-method sweep).
_PCA_DIM = 50
_EARLY_ITER = 250
_METHOD = "tsne"

#: Only ``[A-Za-z0-9_-]`` survive a gallery id (the API recipe allowlist).
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")

_SCHEMA_VERSION = 1


def safe_gallery_id(raw_rid: str) -> str:
    """Deterministic, file/API-safe gallery id for a representation id.

    Representation ids may carry ``+`` / ``@`` (e.g.
    ``maest_logits+maest@643-357``); the gallery manifest keys and artifact
    files must be ``[A-Za-z0-9_-]+`` to satisfy the route allowlist and file
    safety, so those characters are folded to ``_`` under a ``genre_`` prefix.
    Collision handling is :func:`_unique_safe_ids`.
    """
    slug = _SAFE_CHARS.sub("_", raw_rid).strip("_") or "rep"
    return f"genre_{slug}"


def _unique_safe_ids(rids: list[str]) -> dict[str, str]:
    """raw rid → unique safe id; disambiguates slug collisions with a short hash.

    The result is deterministic for a given recipe list (the only use: a fixed
    ``--recipes`` set); a collision only occurs when distinct ids slug to the
    same base.
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for r in rids:
        gid = safe_gallery_id(r)
        if gid in used:
            gid = f"{gid}-{hashlib.sha1(r.encode('utf-8')).hexdigest()[:8]}"
        used.add(gid)
        out[r] = gid
    return out


def _dataset_fingerprint(
    tracks: list[dict],
    meta: dict,
    *,
    seed: int,
    budget: int,
    n_jobs: int,
) -> str:
    """Gallery identity: cohort fp/order + every source artifact checksum + config.

    Hashing the intersection cohort's fingerprint order plus ALL source artifact
    sha256 (not just the manifest id) means a changed embedding source — even a
    re-extracted vector with the same cohort — invalidates the fingerprint and
    so the ratings/caches keyed on it. The projection config (seed/budget/…)
    is folded in too.
    """
    h = hashlib.sha256()
    for t in tracks:
        h.update(str(t["fp"]).encode("utf-8", "replace"))
        h.update(b"\0")
    for model in ("maest", "mert"):
        m = meta.get("models", {}).get(model, {})
        art = m.get("artifact") if isinstance(m, dict) else None
        sha = art.get("sha256") if isinstance(art, dict) else None
        h.update(str(sha or "").encode("utf-8"))
        h.update(b"\0")
    h.update(
        f"seed={seed};budget={budget};n_jobs={n_jobs};"
        f"method={_METHOD};pca_dim={_PCA_DIM};early={_EARLY_ITER}".encode("utf-8")
    )
    return h.hexdigest()


def _input_fingerprint(
    tracks: list[dict],
    composed: np.ndarray,
    weights: dict[str, float],
) -> str:
    """Per-recipe input identity: fp order + weight vector + composed matrix bytes."""
    h = hashlib.sha256()
    for t in tracks:
        h.update(str(t["fp"]).encode("utf-8", "replace"))
        h.update(b"\0")
    h.update(repr(sorted(weights.items())).encode("utf-8"))
    h.update(b"\0")
    h.update(np.ascontiguousarray(np.asarray(composed, dtype=np.float32)).tobytes())
    return h.hexdigest()


def _recipe_revision(
    dataset_fp: str,
    raw_rid: str,
    input_fp: str,
    weights: dict[str, float],
    *,
    seed: int,
    budget: int,
) -> str:
    """Per-recipe revision: weights + input hash + budget/seed/config."""
    h = hashlib.sha256(dataset_fp.encode("utf-8"))
    h.update(raw_rid.encode("utf-8"))
    h.update(b"\0")
    h.update(input_fp.encode("utf-8"))
    h.update(b"\0")
    h.update(repr(sorted(weights.items())).encode("utf-8"))
    h.update(b"\0")
    h.update(f"seed={seed};budget={budget}".encode("utf-8"))
    return h.hexdigest()


def load_roots(roots_file: str) -> dict[str, str]:
    """Persisted roots via ``load_roots_state`` (same stable ids as the app)."""
    from backend.services.roots_registry import load_roots_state

    roots: dict[str, str] = {}
    load_roots_state(roots, os.path.abspath(roots_file), default_root=None)
    if not roots:
        raise ValueError(
            f"no library roots persisted in {roots_file} (open the app once to register "
            "roots, or point --roots-file at a valid roots.json)"
        )
    return roots


def virtual_paths(tracks: list[dict], roots: dict[str, str]) -> list[str]:
    """Absolute track paths → virtual ``rid/rel`` paths under *roots*.

    Raises ``ValueError`` (before any gallery file is written or fitted) if any
    track lies outside every persisted root, so the UI never receives an absolute
    path.
    """
    from werkzeug.exceptions import BadRequest

    from backend.services.virtual_paths import virtual_from_abs

    out: list[str] = []
    for t in tracks:
        try:
            out.append(virtual_from_abs(t["path"], roots))
        except BadRequest:
            raise ValueError(
                f"track path is not under any persisted library root: {t['path']!r} "
                f"(persisted roots: {sorted(set(roots.values()))}); refusing to write a "
                "gallery whose positions would be absolute paths"
            ) from None
    return out


def _refuse_existing(paths: list[str]) -> None:
    for p in paths:
        if os.path.exists(p):
            raise ValueError(f"refusing to overwrite existing gallery file: {p}")


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


class Gallery:
    """State + writes for one ``--gallery-output`` run of ``maps``."""

    def __init__(
        self,
        tracks: list[dict],
        meta: dict,
        rids: list[str],
        *,
        seed: int,
        budget: int,
        n_jobs: int,
        gallery_dir: str,
        roots_file: str,
    ) -> None:
        self.tracks = tracks
        self.meta = meta
        self.rids = list(rids)
        self.seed = seed
        self.budget = budget
        self.n_jobs = n_jobs
        self.gallery_dir = os.path.abspath(gallery_dir)
        self.roots_file = roots_file
        self.roots: dict[str, str] = {}
        self.vpaths: list[str] = []
        self.safe_ids: dict[str, str] = {}
        self.dataset_fp = ""
        self.reference_id = ""
        self._sources: set[str] = set()
        self._entries: dict[str, dict] = {}
        self._artifacts: dict[str, dict] = {}

    def preflight(self) -> None:
        """Load roots + virtual paths and preflight every gallery file, pre-fit.

        Raises before any fitting or write on an unresolved root path, an empty
        roots file, or an existing output file (so a partial previous run is never
        silently extended).
        """
        self.roots = load_roots(self.roots_file)
        self.vpaths = virtual_paths(self.tracks, self.roots)
        self.safe_ids = _unique_safe_ids(self.rids)
        self.dataset_fp = _dataset_fingerprint(
            self.tracks, self.meta, seed=self.seed, budget=self.budget, n_jobs=self.n_jobs
        )
        self.reference_id = self.safe_ids[self.rids[0]]
        files = [os.path.join(self.gallery_dir, "manifest.json")]
        files += [
            os.path.join(self.gallery_dir, f"{self.safe_ids[r]}.json") for r in self.rids
        ]
        _refuse_existing(files)
        os.makedirs(self.gallery_dir, exist_ok=True)

    def add(
        self,
        raw_rid: str,
        *,
        weights: dict[str, float],
        composed: np.ndarray,
        coords: np.ndarray,
        aligned_to_raw: str | None,
        timings: dict,
        fidelity: dict,
        label: str,
    ) -> None:
        """Record one finished map's artifact + manifest entry (safe ids inside)."""
        safe = self.safe_ids[raw_rid]
        n = len(self.tracks)
        input_fp = _input_fingerprint(self.tracks, composed, weights)
        revision = _recipe_revision(
            self.dataset_fp, raw_rid, input_fp, weights, seed=self.seed, budget=self.budget
        )
        aligned_to = self.safe_ids[aligned_to_raw] if aligned_to_raw else None
        positions = [
            {
                "path": self.vpaths[i],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
            for i in range(n)
        ]
        actual_weights = {k: float(v) for k, v in sorted(weights.items())}
        self._sources.update(weights)
        self._artifacts[safe] = {
            "id": safe,
            "representation_id": raw_rid,
            "label": label,
            "revision": revision,
            "dataset_fingerprint": self.dataset_fp,
            "input_fingerprint": input_fp,
            "source_weights": actual_weights,
            "tracks": n,
            "aligned_to": aligned_to,
            "timings": timings,
            "positions": positions,
            "fidelity": fidelity,
            "source_metadata": self.meta,
        }
        self._entries[safe] = {
            "status": "complete",
            "label": label,
            "revision": revision,
            "input_fingerprint": input_fp,
            "tracks": n,
            "aligned_to": aligned_to,
            "timings": timings,
            "file": f"{safe}.json",
        }

    def write(self) -> dict:
        """Atomically write every recipe artifact then the manifest."""
        for safe, artifact in self._artifacts.items():
            _atomic_write_json(os.path.join(self.gallery_dir, f"{safe}.json"), artifact)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "status": "complete",
            "dataset_fingerprint": self.dataset_fp,
            "reference_id": self.reference_id,
            "sources": sorted(self._sources),
            "recipes": self._entries,
        }
        _atomic_write_json(os.path.join(self.gallery_dir, "manifest.json"), manifest)
        return manifest
