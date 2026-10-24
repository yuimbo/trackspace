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
    #: ``(M, dims)`` landmark coordinates fitted against the base layout.
    coeffs: np.ndarray
    #: ``(N, dims)`` base layout the coefficients were fitted against.
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
        """``(N, dims)`` extension values ``g_i(w)`` at *weights*."""
        return self.affinities(weights) @ self.coeffs

    def raw_jacobian(self, weights: dict[str, float]) -> dict[str, np.ndarray]:
        """Exact ``dg_i/dw_s`` as ``source -> (N, dims)``, at *weights*.

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


def _landmark_rows(
    mat_u: np.ndarray,
    spans: dict[str, tuple[int, int]],
    landmark_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """``source -> (M, D_s)`` unweighted blocks for the landmark tracks.

    Fixed once per chart: the same rows serve both the fit and the out-of-sample
    extension, so an inserted track's distances are measured against the chart's own
    landmarks rather than its own rows.
    """
    return {
        s: mat_u[landmark_idx, lo:hi].astype(np.float64)
        for s, (lo, hi) in spans.items()
    }


def _deltas_to_landmark_rows(
    mat_u: np.ndarray,
    spans: dict[str, tuple[int, int]],
    landmark_rows: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """``source -> (N, M)`` squared distances from *mat_u* to fixed landmark rows."""
    out: dict[str, np.ndarray] = {}
    for s, (lo, hi) in spans.items():
        block = mat_u[:, lo:hi].astype(np.float64)
        lm = landmark_rows[s]
        sq = (block**2).sum(axis=1)
        sq_l = (lm**2).sum(axis=1)
        gram = block @ lm.T
        out[s] = (sq[:, None] + sq_l[None, :] - 2.0 * gram).astype(np.float32)
    return out


def _tau_from(delta: dict[str, np.ndarray], quantile: float) -> float:
    vals = np.concatenate([d.ravel() for d in delta.values()])
    return float(max(np.quantile(vals, float(quantile)), 1e-6))


def _remove_gauge(jac: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Project the gauge component out of a movement field.

    The gauge is a per-axis translation plus one shared uniform zoom — no rotation.
    A field that only translates or uniformly scales the whole map carries no shape
    information — the canvas refits the view regardless — so it reads as "nothing
    happened". Fit ``J_i ~= alpha_k + beta * r_i`` with ``r_i = base_i - centroid``
    (one intercept per axis + one shared zoom) and keep the residual, which is the
    shape change. ``base.shape[1]`` fixes the dimensionality (2 or 3).
    """
    n = int(base.shape[0])
    if n < 4:
        return jac
    dims = int(base.shape[1])
    r = base - base.mean(axis=0)
    design = np.zeros((dims * n, dims + 1), dtype=np.float64)
    for k in range(dims):
        design[k * n:(k + 1) * n, k] = 1.0
        design[k * n:(k + 1) * n, -1] = r[:, k]
    # Axis-major flatten (all x rows, then all y rows, …), matching the design's
    # row layout; for dims=2 this is exactly ``concatenate([jac[:, 0], jac[:, 1]])``.
    target = jac.ravel(order="F").astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = (design @ coef).reshape(dims, n).T
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
    place_infos: list[dict] | None = None,
) -> LandmarkModel | None:
    """Fit the landmark extension against the existing base layout (2-D or 3-D).

    Returns ``None`` when there is nothing to fit (fewer than two eligible tracks).
    The caller caches the result per chart revision: refitting ``c`` per drag would
    be the "recompute the fitted quantity" mistake the derivative must avoid.

    *place_infos* are extra tracks the chart *places* but does not fit: the
    landmarks, bandwidth and coefficients stay fixed by the base layout, and the
    extra tracks get a Nyström-style out-of-sample extension in the same frame.
    That is how external roots are drawn on a library-only chart without a refit:
    the fitted rows keep their true base positions and the extra rows are placed
    by the same frozen ``g``.
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
    fit_paths = [paths[i] for i in keep]
    fit_mat = mat_u[keep]

    landmark_idx = _select_landmarks(fit_mat, int(n_landmarks), int(seed))
    landmark_rows = _landmark_rows(fit_mat, spans, landmark_idx)
    fit_delta = _deltas_to_landmark_rows(fit_mat, spans, landmark_rows)
    tau = _tau_from(fit_delta, tau_quantile)
    # A source with no cached rows for any track contributes no span; keep only
    # the sources that actually have a block, or the affinity sum would index a
    # missing key (e.g. ``supervised`` before a head is trained).
    active_sources = tuple(s for s in sources if s in fit_delta)

    a0 = _affinities_at(fit_delta, active_sources, w0, tau)
    # The layout dimensionality comes from the base positions themselves: a "z"
    # key makes it a 3-D chart, otherwise 2-D.
    first = by_path[fit_paths[0]]
    keys = ("x", "y", "z") if "z" in first else ("x", "y")
    y0 = np.array(
        [[by_path[p][k] for k in keys] for p in fit_paths], dtype=np.float64
    )
    gram = a0.T @ a0 + FIT_RIDGE * np.eye(a0.shape[1])
    try:
        coeffs = np.linalg.solve(gram, a0.T @ y0)
    except np.linalg.LinAlgError:
        return None

    placed_paths: list[str] = []
    placed_delta: dict[str, np.ndarray] | None = None
    if place_infos:
        candidate_paths, candidate_mat, candidate_spans = _unweighted_composite(
            place_infos,
            feature_cache,
            sources=sources,
            feature_mask=feature_mask,
            features_blend=features_blend,
            required_sources=required_sources,
        )
        # A source block missing from the extra tracks would shift every later
        # column; drop the extension rather than measure the wrong blocks.
        if candidate_mat.shape[0] and candidate_spans == spans:
            placed_paths = candidate_paths
            placed_delta = _deltas_to_landmark_rows(
                candidate_mat, spans, landmark_rows
            )

    if placed_delta is not None:
        a_placed = _affinities_at(placed_delta, active_sources, w0, tau)
        g_placed = a_placed @ coeffs
        all_paths = fit_paths + placed_paths
        delta = {
            s: np.vstack([fit_delta[s], placed_delta[s]]) for s in active_sources
        }
        base = np.vstack([y0, g_placed])
    else:
        all_paths, delta, base = fit_paths, fit_delta, y0

    model = LandmarkModel(
        paths=all_paths,
        sources=active_sources,
        delta=delta,
        coeffs=coeffs,
        base=base,
        tau=tau,
        weights0=w0,
    )
    log.info(
        "Landmark extension: %d fitted, %d placed, %d landmarks, tau=%.4g",
        len(fit_paths),
        len(all_paths) - len(fit_paths),
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
    place_infos: list[dict] | None = None,
) -> LandmarkModel | None:
    """Fit (or reuse) the chart for *layout_revision*; ``None`` when unfittable.

    Cached per chart revision, which already includes the weights, so a drag reuses
    the fit and only the ``O(N*M*S)`` evaluation runs. When *place_infos* is
    given, the cache key also carries their path set: the same chart extends a
    different set of external tracks, and serving one set's coordinates for another
    would be wrong.
    """
    cache_key = layout_revision
    if layout_revision and place_infos:
        from .layout import track_set_key  # local: layout imports this module

        sig = track_set_key([t["path"] for t in place_infos], sort=True)
        cache_key = f"{layout_revision}|place:{sig}"
    if cache_key:
        hit = _model_cache.get(cache_key)
        if hit is not None:
            _model_cache.move_to_end(cache_key)
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
        place_infos=place_infos,
    )
    if model is not None and cache_key:
        _model_cache[cache_key] = model
        _model_cache.move_to_end(cache_key)
        while len(_model_cache) > _MODEL_CACHE_MAX:
            _model_cache.popitem(last=False)
    return model
