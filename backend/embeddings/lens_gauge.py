"""Put every lens's 2-D layout in one shared orientation (rotation + handedness).

t-SNE's objective is invariant to rotation, reflection and uniform scale, so two
lenses over the same library land in arbitrary, unrelated frames. Switching lens
then looks like the whole map flips, even though neighbourhoods are unchanged —
measured on the real library, 6 of 10 lens pairs differed by a *reflection*.

The fix is a Procrustes (Kabsch) fit against a reference lens, on the tracks the
two layouts share. Only the rotation (and reflection) is applied: uniform scale is
ignored, so a lens switch never resizes the map. The pipeline normalises each axis
to ``[0, 1]`` before this orientation fit and again after it, so the final served
map fills every axis of the view box. That second, per-axis normalisation is
deliberately anisotropic: it can stretch the short axis, so exact 2-D/3-D overlay
and distance ratios in the served frame are not preserved. What is preserved is
the thing the gauge exists for — the displayed orientation and handedness agree
across lenses.

Two alternatives were measured and rejected:

- An anchor-free canonical gauge (each lens rotated onto its own PCA axes with a
  skew-based sign rule) only reached 0.479 and made 4 of 10 pairs *worse* — the
  principal axes are not stable when the cloud is near-isotropic.
- Axis flip/swap only (no rotation) was worse than Procrustes on every lens.

The reference is whichever layout is registered first for a given *stable family*
(the library scope plus cache generations, not the exact track set); which lens
that is barely matters (mean RMSE 0.299-0.316 across all five choices), so it is
deliberately not configurable. Fitting is robust to partial overlap: a fit on 5%
of the tracks scored 0.3756 vs 0.3723 on all of them. A family is shared by a
folder subset and the whole library slice, so an anchor registered by one is
usable by the other. Layouts with fewer than :data:`MIN_SHARED` tracks never
become the anchor (a tiny folder would otherwise strand the rest of the family).

3-D layouts follow the same rule, head-on: a 3-D t-SNE layout is also invariant to
rotation/reflection, so its (x,y) is an arbitrary frame of the same lens's 2-D
layout. `gauge_layout_3d` fits the 3-D layout so its x,y match that 2-D layout (a
Procrustes fit onto the head-on projection, which pins the rotation up to the sign
of z), then fixes the z sign against the family's first-registered 3-D layout so
"up" is consistent across lenses. The caller then normalises all three axes for
the unit cube, so the served 3-D frame keeps the stable orientation but fills the
full volume instead of preserving an exact 2-D x/y overlay. A 3-D layout without a
2-D reference is left ungauged.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from .linalg_utils import (
    apply_procrustes,
    matched_indices,
    procrustes,
    svd_singular,
)

log = logging.getLogger(__name__)

#: Reference frames kept, keyed by track-set family. Small: one per library slice.
_GAUGE_CACHE_MAX = 8

#: Minimum shared tracks before a fit is trustworthy. The same floor gates
#: anchor registration so a tiny folder can never define a family's frame.
MIN_SHARED = 8

#: Gauge semantics generation. Bumped whenever normalisation or the family rule
#: changes the coordinates stored for a family; it is part of the layout revision
#: and the family key, so persisted layout rows from the old semantics never seed
#: a reference frame.
GAUGE_VERSION2 = 3

_gauge_cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
_gauge_lock = threading.RLock()
_gauge_stats = {"anchor": 0, "aligned": 0, "reflected": 0, "skipped": 0}


def gauge_stats() -> dict[str, int]:
    """Counters: anchors registered, layouts aligned, reflections, skips."""
    with _gauge_lock:
        return dict(_gauge_stats)


def reset_gauge_cache() -> None:
    """Drop all reference frames and zero the counters (tests, config changes)."""
    with _gauge_lock:
        _gauge_cache.clear()
        for k in _gauge_stats:
            _gauge_stats[k] = 0


def reference(family: str) -> tuple[list[str], np.ndarray] | None:
    """The registered reference frame for *family*, or ``None``."""
    with _gauge_lock:
        entry = _gauge_cache.get(family)
        if entry is None:
            return None
        _gauge_cache.move_to_end(family)
        paths, coords = entry
        return list(paths), np.array(coords, dtype=np.float64, copy=True)


def has_reference(family: str) -> bool:
    """Whether *family* already has a registered reference frame."""
    with _gauge_lock:
        return family in _gauge_cache


def _register(family: str, paths: list[str], coords: np.ndarray) -> bool:
    """Register *coords* as *family*'s frame when it has none; ``True`` if new.

    An existing entry has no effect beyond marking it recently used, so a live
    family is never re-anchored.
    """
    with _gauge_lock:
        if family in _gauge_cache:
            _gauge_cache.move_to_end(family)
            return False
        _gauge_cache[family] = (list(paths), np.asarray(coords, dtype=np.float64))
        _gauge_cache.move_to_end(family)
        while len(_gauge_cache) > _GAUGE_CACHE_MAX:
            _gauge_cache.popitem(last=False)
        _gauge_stats["anchor"] += 1
        return True


def seed_reference(family: str, paths: list[str], coords: np.ndarray) -> bool:
    """Register *coords* as *family*'s reference only when none exists yet.

    Used to restore a persisted anchor on a cold start. An existing in-memory
    reference always wins, so repeated calls never re-anchor a live family. A
    layout with fewer than :data:`MIN_SHARED` tracks is never an anchor. Returns
    ``True`` when the entry was newly registered.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if (
        len(paths) < MIN_SHARED
        or arr.ndim != 2
        or arr.shape[0] != len(paths)
        or arr.shape[1] not in (2, 3)
        or not np.all(np.isfinite(arr))
    ):
        return False
    return _register(family, paths, arr)


def align_to_reference(
    paths: list[str],
    coords: np.ndarray,
    ref_paths: list[str],
    ref_coords: np.ndarray,
) -> tuple[np.ndarray, bool] | None:
    """*coords* rotated/reflected into the reference frame, or ``None`` if not fittable.

    The fit uses only the tracks both layouts contain, then applies the resulting
    rotation and translation to every point. Uniform scale is deliberately ignored:
    the final per-axis normalisation owns the served extent, so scaling here would
    only add a redundant, cancellation-prone step. Returns ``(aligned, reflected)``.
    """
    mine, theirs = matched_indices(paths, ref_paths)
    if len(mine) < MIN_SHARED:
        return None
    rot, _scale, src_mu, dst_mu = procrustes(coords[mine], ref_coords[theirs])
    if not np.all(np.isfinite(rot)):
        return None
    aligned = apply_procrustes(coords, rot, 1.0, src_mu, dst_mu)
    if not np.all(np.isfinite(aligned)):
        return None
    return aligned, bool(np.linalg.det(rot) < 0)


def head_on_fit(
    paths: list[str],
    coords3: np.ndarray,
    ref_paths: list[str],
    ref_xy: np.ndarray,
) -> np.ndarray | None:
    """Similarity fit of a 3-D layout so its x,y match a 2-D reference.

    The fit uses only the tracks both layouts contain, then applies the resulting
    transform to every point of *coords3*: its x,y land on the reference and its
    z is the out-of-plane direction that completes the proper rotation. Returns
    ``None`` when the shared tracks are too few or the layout is degenerate
    (collinear, or a non-finite layout).

    The plane and scale come from the polar part of the least-squares map of the
    reference on the cloud (``b.T a (a.T a)⁻¹``): when the two layouts are exactly
    consistent this map is already ``scale × orthonormal``, so the polar part
    recovers the true similarity exactly. (The naive SVD of ``a.T b`` alone
    optimises the *unscaled* fit and is exact only for isotropic clouds.)
    """
    P = np.asarray(coords3, dtype=np.float64)
    Q = np.asarray(ref_xy, dtype=np.float64)
    if (
        P.ndim != 2
        or P.shape[1] != 3
        or Q.ndim != 2
        or Q.shape[1] != 2
        or len(paths) != P.shape[0]
        or len(ref_paths) != Q.shape[0]
    ):
        return None
    mine, theirs = matched_indices(paths, ref_paths)
    if len(mine) < MIN_SHARED:
        return None
    pm = P[mine]
    qm = Q[theirs]
    a = pm - pm.mean(0)  # n×3, the matched rows of the 3-D cloud
    b = qm - qm.mean(0)  # n×2, the matched rows of the 2-D reference
    U, S, Vt = svd_singular(a.T @ b)  # U 3×2, S (2,), Vt 2×2 (degeneracy probe)
    if S[0] <= 1e-12 or S[1] < 1e-9 * S[0] or not np.all(np.isfinite(S)):
        return None
    W = b.T @ a @ np.linalg.pinv(a.T @ a)  # 2×3 least-squares map
    U2, S2, Vt2 = svd_singular(W)  # U2 2×2, S2 (2,), Vt2 2×3
    A = U2 @ Vt2  # 2×3 orthonormal rows: the head-on rotation
    r3 = np.cross(A[0], A[1])  # out-of-plane direction completing the frame
    proj = a @ A.T
    denom = float((proj**2).sum())
    if (
        denom <= 1e-12
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(r3))
    ):
        return None
    scale = float((proj * b).sum() / denom)
    if not np.isfinite(scale) or scale <= 0:
        return None
    c = P - pm.mean(0)  # every row of the 3-D cloud, same origin as the fit
    out_xy = scale * (c @ A.T) + qm.mean(0)
    out_z = scale * (c @ r3)
    out = np.column_stack([out_xy, out_z])
    if not np.all(np.isfinite(out)):
        return None
    return out


def gauge_layout_3d(
    family: str,
    paths: list[str],
    coords3: np.ndarray,
    ref_paths: list[str],
    ref_xy: np.ndarray,
) -> np.ndarray:
    """Return *coords3* with x,y pinned to the lens's own 2-D layout.

    *family* is the family's full key (the caller namespaces 2-D and 3-D
    separately); the head-on fit fixes the rotation up to the sign of z, which is
    then pinned against the family's first-registered 3-D layout (a mirrored but
    equally valid view is flipped back). Any failure returns *coords3* unchanged —
    an unaligned map is a cosmetic regression, never an error.
    """
    if coords3.ndim != 2 or coords3.shape[1] != 3 or len(paths) != coords3.shape[0]:
        return coords3
    try:
        aligned = head_on_fit(paths, coords3, ref_paths, ref_xy)
    except Exception:
        log.debug("3-D lens gauge head-on fit failed; leaving layout unaligned", exc_info=True)
        aligned = None
    if aligned is None:
        with _gauge_lock:
            _gauge_stats["skipped"] += 1
        return coords3
    with _gauge_lock:
        if _register(family, paths, aligned):
            return aligned
        anchor_paths, anchor_coords = _gauge_cache[family]
    mine, theirs = matched_indices(paths, anchor_paths)
    reflected = False
    if len(mine) >= MIN_SHARED:
        z = aligned[mine, 2]
        za = anchor_coords[theirs, 2]
        s = float(((z - z.mean()) * (za - za.mean())).sum())
        if s < 0:
            aligned = aligned.copy()
            aligned[:, 2] = -aligned[:, 2]
            reflected = True
    with _gauge_lock:
        _gauge_stats["aligned"] += 1
        if reflected:
            _gauge_stats["reflected"] += 1
    return aligned


def gauge_layout(
    family: str, paths: list[str], coords: np.ndarray
) -> np.ndarray:
    """Return *coords* in the shared gauge for *family*, registering it if first.

    *family* identifies the comparable set of layouts (same library slice and cache
    generation); lenses within one family are made mutually consistent, and a
    different family gets its own anchor. Any failure returns *coords* unchanged —
    an unaligned map is a cosmetic regression, never an error.
    """
    if coords.ndim != 2 or coords.shape[1] != 2 or len(paths) != coords.shape[0]:
        return coords
    if len(paths) < MIN_SHARED:
        # A layout with too few tracks to fit against must not define the family
        # frame; it stays in its own frame until a family anchor exists.
        with _gauge_lock:
            _gauge_stats["skipped"] += 1
        return coords
    with _gauge_lock:
        if _register(family, paths, coords):
            return coords
        ref_paths, ref_coords = _gauge_cache[family]
    try:
        out = align_to_reference(
            paths, np.asarray(coords, dtype=np.float64), ref_paths, ref_coords
        )
    except Exception:
        log.debug("lens gauge alignment failed; leaving layout unrotated", exc_info=True)
        out = None
    if out is None:
        with _gauge_lock:
            _gauge_stats["skipped"] += 1
        return coords
    aligned, reflected = out
    with _gauge_lock:
        _gauge_stats["aligned"] += 1
        if reflected:
            _gauge_stats["reflected"] += 1
    return aligned
