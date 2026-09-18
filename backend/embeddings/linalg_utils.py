"""Linear-algebra helpers shared by the embedding pipeline.

`np.linalg.svd` drives LAPACK's ``gesdd``, which raises "SVD did not converge" on
some finite but numerically rank-deficient matrices — common here because a folder's
tracks can be nearly collinear in the composite space. scipy's ``gesvd`` driver
converges on the same input, so a non-convergence retries with it instead of losing
the whole request to a 500.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg

log = logging.getLogger(__name__)


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
