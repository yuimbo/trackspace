"""Smooth landmark extension of the t-SNE chart, with an exact weight Jacobian.

An independent refit's finite difference is not a derivative: t-SNE is non-convex and
its init is data-dependent, so a small weight change lands in a different local
optimum (measured: PCA cos 0.9998, t-SNE cos −0.17). This module instead fits a
*fixed, smooth* function of the weights and differentiates it analytically.

Landmarks ``l`` are actual tracks. With their unweighted per-source blocks
``u_ls`` and cached squared distances

    Delta_ils = ||u_is - u_ls||^2

the extension is

    D2_il(w) = sum_s w_s * Delta_ils            (the same additive metric)
    a_il(w)  = softmax_l(-D2_il(w) / tau)        (normalised affinities)
    g_i(w)   = sum_l a_il(w) c_l                 (landmark coordinates)

``c_l`` is fitted once against the existing base layout. The preview is the
residual form

    y~_i(w) = y_i^0 + g_i(w) - g_i(w0)

which reproduces the base layout *exactly* at w = w0. The derivative is closed
form (differentiating the normalised exponentials):

    dg_i/dw_s = -(1/tau) * sum_l a_il * Delta_ils * (c_l - g_i)

It is exact for this nonlinear ``g`` (not a finite difference) and avoids
differentiating ``sqrt(w_s)``, because the squared distances are already affine in
``w``. Existing tracks never re-read their 1819-D features: everything is
``O(N*M*S)`` after the ``Delta`` cache. A frozen reference chart cannot
reorganize its own geometry to match a radically different metric, so insertion and
chart rebuilding stay different operations.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Default landmark count. 128 keeps the per-source Delta cache ~51 MB at 20k
#: tracks (the researcher's table: 512 landmarks would be ~205 MB).
DEFAULT_LANDMARKS = 128
#: tau as this quantile of the D2 distribution, so the softmax is neither
#: one-hot (hard, discontinuous) nor uniform (over-smoothed). Fixed per chart.
DEFAULT_TAU_QUANTILE = 0.5
#: Ridge for the landmark-coordinate fit (A c ~= y^0).
FIT_RIDGE = 1e-3


@dataclass
class LandmarkModel:
    """A fitted landmark extension: unweighted blocks, deltas and coefficients."""

    paths: list[str]
    sources: tuple[str, ...]
    #: ``source -> (N, M)`` squared distances ``||u_is - u_ls||^2``.
    delta: dict[str, np.ndarray]
    #: ``(M, 2)`` landmark coordinates fitted against the base layout.
    coeffs: np.ndarray
    #: ``(N, 2)`` base layout the coefficients were fitted against.
    base: np.ndarray
    #: Softmax bandwidth, fixed per chart.
    tau: float
    #: The weights the coefficients were fitted at.
    weights0: dict[str, float]

    def affinities(self, weights: dict[str, float]) -> np.ndarray:
        """``(N, M)`` row-normalised landmark affinities at *weights*."""
        first = self.delta[self.sources[0]]
        d2 = np.zeros(first.shape, dtype=np.float64)
        for s in self.sources:
            d2 += float(weights.get(s, 1.0)) * self.delta[s]
        z = -d2 / self.tau
        # Stability: subtract the row max before the softmax normaliser.
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def evaluate(self, weights: dict[str, float]) -> np.ndarray:
        """``(N, 2)`` extension values ``g_i(w)`` at *weights*."""
        return self.affinities(weights) @ self.coeffs

    def raw_jacobian(self, weights: dict[str, float]) -> dict[str, np.ndarray]:
        """Exact ``dg_i/dw_s`` as ``source -> (N, 2)``, at *weights*.

        Closed form: ``dg/dw_s = -(1/tau) sum_l a_il Delta_ils (c_l - g_i)``.
        This is the derivative of the *nonlinear* extension, so a first-order step
        ``g(w0) + J dw`` tracks the real ``g(w0 + dw)`` to second order.
        """
        a = self.affinities(weights)
        g = a @ self.coeffs
        out: dict[str, np.ndarray] = {}
        for s in self.sources:
            d = self.delta[s]
            weighted = a * d
            # sum_l a_il Delta_ils c_l  and  sum_l a_il Delta_ils
            p = weighted @ self.coeffs
            q = weighted.sum(axis=1)
            out[s] = (-(p - q[:, None] * g) / self.tau).astype(np.float32)
        return out

    def jacobian(self, weights: dict[str, float]) -> dict[str, np.ndarray]:
        """``raw_jacobian`` with the translation + uniform-zoom component removed.

        A Jacobian that only offsets or scales the whole map carries no shape
        information (the canvas refits the view anyway), so it reads as "nothing
        happened"; the returned field is the shape change.
        """
        return {
            s: _remove_gauge(arr, self.base)
            for s, arr in self.raw_jacobian(weights).items()
        }

    def preview(self, weights: dict[str, float]) -> dict[str, Any]:
        """Positions at *weights* plus the exact Jacobian at *weights*.

        The caller renders ``y0 + J dw`` (first-order) or ``y0 + g(w) - g(w0)``
        (exact for this ``g``); both equal ``y0`` at ``w = w0``.
        """
        g0 = self.evaluate(self.weights0)
        return {
            "paths": list(self.paths),
            "weights": dict(self.weights0),
            "g0": g0.astype(np.float32),
            "jacobian": self.jacobian(self.weights0),
        }


def _unweighted_composite(
    track_infos: list[dict],
    feature_cache: Any,
    *,
    sources: tuple[str, ...],
    feature_mask: np.ndarray | None,
    features_blend: float,
    required_sources: tuple[str, ...] | None,
) -> tuple[list[str], np.ndarray, dict[str, tuple[int, int]]]:
    """Gather the per-source blocks *without* source weights (all 1.0).

    The landmark distances are built on the unweighted ``u_is``; the weights enter
    only through the additive ``D2 = sum_s w_s Delta_s``, which is what makes the
    derivative affine in ``w`` and avoids the ``sqrt(w_s)`` singularity.
    """
    from .layout import _gather_composite_vecs

    unit = {s: 1.0 for s in sources}
    paths, mat, _clap_dim, _clap_lo, _key, _cfg, spans = _gather_composite_vecs(
        track_infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        source_weights=unit,
        required_sources=required_sources,
    )
    return paths, mat, spans


def _select_landmarks(mat_u: np.ndarray, n_landmarks: int, seed: int) -> np.ndarray:
    """Landmark row indices: the actual tracks nearest deterministic k-means centres.

    Landmarks are real cached rows, so ``u_ls`` is a genuine block; a centre need
    not be an actual track. Falls back to a strided subset if clustering fails.
    """
    n = int(mat_u.shape[0])
    m = int(min(max(1, n_landmarks), n))
    if m >= n:
        return np.arange(n)
    try:
        from sklearn.cluster import MiniBatchKMeans  # lazy — heavy import

        km = MiniBatchKMeans(
            n_clusters=m,
            random_state=int(seed),
            n_init=3,
            batch_size=max(256, n // 50),
        )
        labels = km.fit_predict(mat_u)
        centres = km.cluster_centers_
    except Exception as e:
        log.warning("landmark k-means failed (%s); using strided subset", e)
        return np.linspace(0, n - 1, m).astype(np.int64)
    out = np.empty(m, dtype=np.int64)
    for c in range(m):
        rows = np.where(labels == c)[0]
        if rows.size == 0:
            out[c] = 0
            continue
        d = ((mat_u[rows] - centres[c]) ** 2).sum(axis=1)
        out[c] = int(rows[int(np.argmin(d))])
    return out


def _source_deltas(
    mat_u: np.ndarray,
    spans: dict[str, tuple[int, int]],
    landmark_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """``source -> (N, M)`` squared distances to the landmarks, in float32."""
    out: dict[str, np.ndarray] = {}
    for s, (lo, hi) in spans.items():
        block = mat_u[:, lo:hi].astype(np.float64)
        sq = (block**2).sum(axis=1)
        lm = block[landmark_idx]
        gram = block @ lm.T
        out[s] = (sq[:, None] + sq[landmark_idx][None, :] - 2.0 * gram).astype(
            np.float32
        )
    return out


def _tau_from(delta: dict[str, np.ndarray], quantile: float) -> float:
    vals = np.concatenate([d.ravel() for d in delta.values()])
    return float(max(np.quantile(vals, float(quantile)), 1e-6))


def _remove_gauge(jac: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Project the translation + uniform-zoom component out of a movement field.

    A field that only translates or uniformly scales the whole map carries no shape
    information — the canvas refits the view regardless — so it reads as "nothing
    happened". Fit ``J_i ~= alpha + beta * r_i`` with ``r_i = base_i - centroid``
    (2 intercepts + one shared zoom) and keep the residual, which is the shape change.
    """
    n = int(base.shape[0])
    if n < 4:
        return jac
    r = base - base.mean(axis=0)
    design = np.zeros((2 * n, 3), dtype=np.float64)
    design[:n, 0] = 1.0
    design[:n, 2] = r[:, 0]
    design[n:, 1] = 1.0
    design[n:, 2] = r[:, 1]
    target = np.concatenate([jac[:, 0], jac[:, 1]]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = np.empty((n, 2), dtype=np.float64)
    fitted[:, 0] = coef[0] + coef[2] * r[:, 0]
    fitted[:, 1] = coef[1] + coef[2] * r[:, 1]
    return (jac - fitted).astype(np.float32)


def build_landmark_model(
    track_infos: list[dict],
    feature_cache: Any,
    base_positions: list[dict],
    *,
    sources: tuple[str, ...],
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    required_sources: tuple[str, ...] | None = None,
    weights0: dict[str, float] | None = None,
    n_landmarks: int = DEFAULT_LANDMARKS,
    tau_quantile: float = DEFAULT_TAU_QUANTILE,
    seed: int = 42,
) -> LandmarkModel | None:
    """Fit the landmark extension against the existing base 2-D layout.

    Returns ``None`` when there is nothing to fit (fewer than two eligible tracks).
    The caller caches the result per chart revision: refitting ``c`` per drag would
    be the "recompute the fitted quantity" mistake the derivative must avoid.
    """
    w0 = {s: float((weights0 or {}).get(s, 1.0)) for s in sources}
    paths, mat_u, spans = _unweighted_composite(
        track_infos,
        feature_cache,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        required_sources=required_sources,
    )
    if len(paths) < 2 or mat_u.shape[0] < 2:
        return None

    by_path = {p["path"]: p for p in base_positions}
    keep = [i for i, p in enumerate(paths) if p in by_path]
    if len(keep) < 2:
        return None
    paths = [paths[i] for i in keep]
    mat_u = mat_u[keep]

    landmark_idx = _select_landmarks(mat_u, int(n_landmarks), int(seed))
    delta = _source_deltas(mat_u, spans, landmark_idx)
    tau = _tau_from(delta, tau_quantile)

    a0 = _affinities_at(delta, sources, w0, tau)
    y0 = np.array(
        [[by_path[p]["x"], by_path[p]["y"]] for p in paths], dtype=np.float64
    )
    gram = a0.T @ a0 + FIT_RIDGE * np.eye(a0.shape[1])
    try:
        coeffs = np.linalg.solve(gram, a0.T @ y0)
    except np.linalg.LinAlgError:
        return None

    model = LandmarkModel(
        paths=paths,
        sources=tuple(sources),
        delta=delta,
        coeffs=coeffs,
        base=y0,
        tau=tau,
        weights0=w0,
    )
    log.info(
        "Landmark extension: %d tracks, %d landmarks, tau=%.4g",
        len(paths),
        int(landmark_idx.size),
        tau,
    )
    return model


def _affinities_at(
    delta: dict[str, np.ndarray],
    sources: tuple[str, ...],
    weights: dict[str, float],
    tau: float,
) -> np.ndarray:
    d2 = np.zeros(delta[sources[0]].shape, dtype=np.float64)
    for s in sources:
        d2 += float(weights.get(s, 1.0)) * delta[s]
    z = -d2 / tau
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


#: Bounded LRU of fitted charts, keyed by ``layout_revision``. The fitted ``c``
#: is a chart artifact: refitting it per drag would be the "recompute a fitted
#: quantity" mistake, and evicting it would silently change an "identical" request.
_MODEL_CACHE_MAX = 8
_model_cache: OrderedDict[str, LandmarkModel] = OrderedDict()


def reset_landmark_model_cache() -> None:
    """Drop fitted charts (tests, config changes)."""
    _model_cache.clear()


def landmark_model_for(
    layout_revision: str | None,
    track_infos: list[dict],
    feature_cache: Any,
    base_positions: list[dict],
    *,
    sources: tuple[str, ...],
    feature_mask: np.ndarray | None = None,
    features_blend: float = 0.42,
    required_sources: tuple[str, ...] | None = None,
    weights0: dict[str, float] | None = None,
    n_landmarks: int = DEFAULT_LANDMARKS,
) -> LandmarkModel | None:
    """Fit (or reuse) the chart for *layout_revision*; ``None`` when unfittable.

    Cached per chart revision, which already includes the weights, so a drag reuses
    the fit and only the ``O(N*M*S)`` evaluation runs.
    """
    if layout_revision:
        hit = _model_cache.get(layout_revision)
        if hit is not None:
            _model_cache.move_to_end(layout_revision)
            return hit
    model = build_landmark_model(
        track_infos,
        feature_cache,
        base_positions,
        sources=sources,
        feature_mask=feature_mask,
        features_blend=features_blend,
        required_sources=required_sources,
        weights0=weights0,
        n_landmarks=n_landmarks,
    )
    if model is not None and layout_revision:
        _model_cache[layout_revision] = model
        _model_cache.move_to_end(layout_revision)
        while len(_model_cache) > _MODEL_CACHE_MAX:
            _model_cache.popitem(last=False)
    return model
