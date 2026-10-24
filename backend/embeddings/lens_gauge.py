"""Put every lens's 2-D layout in one shared gauge (orientation + handedness).

t-SNE's objective is invariant to rotation, reflection and uniform scale, so two
lenses over the same library land in arbitrary, unrelated frames. Switching lens
then looks like the whole map flips, even though neighbourhoods are unchanged —
measured on the real library, 6 of 10 lens pairs differed by a *reflection*.

The fix is a Procrustes (Kabsch) similarity fit against a reference lens, on the
tracks the two layouts share. A similarity transform is shape-preserving: measured
distance-ratio spread 4e-16 and 100% nearest-neighbour preservation, so this only
changes the gauge, never the structure. Measured mean pairwise RMSE across the five
lenses: 0.518 raw -> 0.305 aligned.

Two alternatives were measured and rejected:

- An anchor-free canonical gauge (each lens rotated onto its own PCA axes with a
  skew-based sign rule) only reached 0.479 and made 4 of 10 pairs *worse* — the
  principal axes are not stable when the cloud is near-isotropic.
- Axis flip/swap only (no rotation) was worse than Procrustes on every lens.

The reference is whichever layout is registered first for a given track-set
"family"; which lens that is barely matters (mean RMSE 0.299-0.316 across all five
choices), so it is deliberately not configurable. Fitting is robust to partial
overlap: a fit on 5% of the tracks scored 0.3756 vs 0.3723 on all of them.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from .linalg_utils import apply_procrustes, matched_indices, procrustes

log = logging.getLogger(__name__)

#: Reference frames kept, keyed by track-set family. Small: one per library slice.
_GAUGE_CACHE_MAX = 8

#: Minimum shared tracks before a fit is trustworthy.
MIN_SHARED = 8

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


def align_to_reference(
    paths: list[str],
    coords: np.ndarray,
    ref_paths: list[str],
    ref_coords: np.ndarray,
) -> tuple[np.ndarray, bool] | None:
    """*coords* rotated/scaled into the reference frame, or ``None`` if not fittable.

    The fit uses only the tracks both layouts contain, then applies the resulting
    transform to every point. Returns ``(aligned, reflected)``.
    """
    mine, theirs = matched_indices(paths, ref_paths)
    if len(mine) < MIN_SHARED:
        return None
    rot, scale, src_mu, dst_mu = procrustes(coords[mine], ref_coords[theirs])
    if not np.all(np.isfinite(rot)) or not np.isfinite(scale) or scale <= 0:
        return None
    aligned = apply_procrustes(coords, rot, scale, src_mu, dst_mu)
    if not np.all(np.isfinite(aligned)):
        return None
    return aligned, bool(np.linalg.det(rot) < 0)


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
    with _gauge_lock:
        entry = _gauge_cache.get(family)
        if entry is None:
            _gauge_cache[family] = (list(paths), np.asarray(coords, dtype=np.float64))
            _gauge_cache.move_to_end(family)
            while len(_gauge_cache) > _GAUGE_CACHE_MAX:
                _gauge_cache.popitem(last=False)
            _gauge_stats["anchor"] += 1
            return coords
        _gauge_cache.move_to_end(family)
        ref_paths, ref_coords = entry
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
