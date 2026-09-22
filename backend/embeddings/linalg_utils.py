"""Linear-algebra helpers shared by the embedding pipeline.

`np.linalg.svd` drives LAPACK's ``gesdd``, which raises "SVD did not converge" on
some finite but numerically rank-deficient matrices — common here because a folder's
tracks can be nearly collinear in the composite space. scipy's ``gesvd`` driver
converges on the same input, so a non-convergence retries with it instead of losing
the whole request to a 500.

`l2_normalize` is the one row-wise normaliser every similarity path (search, CLAP,
clustering, the supervised head) shares.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg

log = logging.getLogger(__name__)


def l2_normalize(mat: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise *mat* along *axis*, with a floor so a zero row cannot divide by zero."""
    norms = np.linalg.norm(mat, axis=axis, keepdims=True)
    np.maximum(norms, 1e-9, out=norms)
    return mat / norms


def procrustes(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Best ``(rotation, scale, src_centroid, dst_centroid)`` mapping *src* onto *dst*.

    Reflections are allowed: a flipped layout is exactly the case a gauge fix
    corrects, so constraining ``det(R) = +1`` would leave the flip in place.
    """
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    a = src - src_mu
    b = dst - dst_mu
    u, s, vt = svd_singular(a.T @ b)
    rot = vt.T @ u.T
    denom = float((a**2).sum())
    scale = float(s.sum() / denom) if denom > 1e-12 else 1.0
    return rot, scale, src_mu, dst_mu


def apply_procrustes(
    coords: np.ndarray,
    rot: np.ndarray,
    scale: float,
    src_mu: np.ndarray,
    dst_mu: np.ndarray,
) -> np.ndarray:
    """Apply a :func:`procrustes` result to *coords* (column-vector form)."""
    return (scale * (rot @ (coords - src_mu).T)).T + dst_mu


def svd_singular(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, s, vt)`` for *mat*, robust to a ``gesdd`` non-convergence.

    Equivalent to ``np.linalg.svd(mat, full_matrices=False)``, with a scipy
    ``gesvd`` fallback when the default driver fails to converge.
    """
    try:
        return np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        log.debug("gesdd did not converge on %s; retrying with gesvd", mat.shape)
        return scipy.linalg.svd(mat, full_matrices=False, lapack_driver="gesvd")
