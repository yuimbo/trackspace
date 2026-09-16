"""Landmark extension: exact weight Jacobian, residual form, determinism.

The Jacobian is checked against *central differences of the same frozen ``g``*
(not against an independent refit, whose different local optimum is not a
derivative). A first-order method's residual must fall with the step size; a
refit-based finite difference does not.
"""

from __future__ import annotations

import numpy as np

from backend.embeddings.landmark_projection import (
    LandmarkModel,
    _remove_gauge,
    build_landmark_model,
)


def _model(seed: int = 0, n: int = 80, m: int = 8) -> LandmarkModel:
    rng = np.random.default_rng(seed)
    sources = ("a", "b", "c")
    delta = {
        s: (rng.random((n, m)).astype(np.float32) * 4.0) for s in sources
    }
    return LandmarkModel(
        paths=[f"t{i}.mp3" for i in range(n)],
        sources=sources,
        delta=delta,
        coeffs=rng.normal(size=(m, 2)),
        base=rng.random((n, 2)).astype(np.float32),
        tau=1.0,
        weights0={"a": 0.4, "b": 0.35, "c": 0.25},
    )


def test_raw_jacobian_matches_central_differences_of_g():
    """The closed form must equal a central difference of the *same* g."""
    model = _model()
    w0 = dict(model.weights0)
    jac = model.raw_jacobian(w0)
    assert set(jac) == set(model.sources)
    for s in model.sources:
        h = 1e-4
        up, dn = dict(w0), dict(w0)
        up[s] += h
        dn[s] -= h
        fd = (model.evaluate(up) - model.evaluate(dn)) / (2 * h)
        assert np.abs(jac[s] - fd).max() / (np.abs(fd).max() + 1e-12) < 1e-3


def test_remove_gauge_kills_translation_and_zoom():
    """A pure offset/zoom field carries no shape change and must vanish."""
    rng = np.random.default_rng(0)
    base = rng.random((50, 2))
    r = base - base.mean(axis=0)
    # alpha + beta * r  ==  translation + uniform zoom about the centroid.
    gauge = 0.7 + 0.4 * r
    assert np.abs(_remove_gauge(gauge, base)).max() < 1e-6
    # A genuine shape change survives.
    shape = rng.normal(size=(50, 2))
    kept = _remove_gauge(gauge + shape, base)
    assert np.abs(kept).max() > 0


def test_jacobian_is_the_gauge_projected_raw_field():
    model = _model(seed=5)
    w0 = dict(model.weights0)
    raw = model.raw_jacobian(w0)
    projected = model.jacobian(w0)
    for s in model.sources:
        np.testing.assert_allclose(
            projected[s], _remove_gauge(raw[s], model.base), atol=1e-6
        )


def test_first_order_residual_shrinks_with_the_step():
    """A smooth method's residual is second order, not refit noise."""
    model = _model(seed=1)
    w0 = dict(model.weights0)
    jac = model.jacobian(w0)
    g0 = model.evaluate(w0)
    rng = np.random.default_rng(7)
    direction = rng.normal(size=len(model.sources))

    errs: list[float] = []
    for h in (1e-1, 1e-2, 1e-3):
        w = {
            s: max(0.05, w0[s] + h * float(direction[i]))
            for i, s in enumerate(model.sources)
        }
        pred = g0 + sum(jac[s] * (w[s] - w0[s]) for s in model.sources)
        errs.append(float(np.abs(model.evaluate(w) - pred).max()))

    # Residual falls with the step; 10x smaller step -> ~100x smaller residual.
    assert errs[1] < errs[0] / 10
    assert errs[2] < errs[1] / 10
    assert errs[2] < 1e-4


def test_preview_reproduces_the_base_layout_at_w0():
    """The residual form gives y~_i(w0) = y_i^0 exactly."""
    model = _model(seed=2)
    p = model.preview(model.weights0)
    g0 = model.evaluate(model.weights0)
    np.testing.assert_allclose(p["g0"], g0, atol=1e-6)
    # At w = w0 the movement g(w0) - g(w0) is exactly zero, so the frontend's
    # y^0 + J*dw leaves the base layout untouched.
    assert set(p["jacobian"]) == set(model.sources)
    assert p["weights"] == model.weights0


def test_preview_is_deterministic():
    model = _model(seed=3)
    first = model.preview(model.weights0)
    second = model.preview(model.weights0)
    for s in model.sources:
        np.testing.assert_array_equal(first["jacobian"][s], second["jacobian"][s])


# ── pipeline build (stub cache, no audio) ──────────────────────


class _StubFeatureCache:
    def __init__(self, clap, feats):
        self._maps = {"clap": clap, "features": feats}

    def get_all(self, source, fingerprints, version=0):
        del version
        m = self._maps.get(source, {})
        return {fp: m[fp] for fp in fingerprints if fp in m}


def _infos(n: int = 40, dim: int = 5):
    rng = np.random.default_rng(4)
    clap, feats, infos = {}, {}, []
    for i in range(n):
        fp = f"fp{i}"
        clap[fp] = rng.standard_normal(6).astype(np.float32)
        feats[fp] = rng.standard_normal(6).astype(np.float32)
        infos.append({"path": f"a/b/t{i}.mp3", "fingerprint": fp})
    return _StubFeatureCache(clap, feats), infos


def test_build_returns_none_without_enough_tracks():
    cache, infos = _infos(n=1)
    assert (
        build_landmark_model(
            infos,
            cache,
            [{"path": "a/b/t0.mp3", "x": 0.5, "y": 0.5}],
            sources=("clap", "features"),
        )
        is None
    )


def test_build_fits_against_the_base_layout():
    cache, infos = _infos(n=40)
    # A chart is a smooth function of the composite, so make the synthetic base
    # layout one too: two normalised composite coordinates.
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    base = [
        {"path": t["path"], "x": float(clap[i, 0]), "y": float(clap[i, 1])}
        for i, t in enumerate(infos)
    ]
    y0 = np.array([[p["x"], p["y"]] for p in base])
    baseline = float(np.abs(y0 - y0.mean(axis=0)).mean())

    model = build_landmark_model(
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.5,
        n_landmarks=20,
    )
    assert model is not None
    assert model.delta["clap"].shape == (40, 20)
    assert model.coeffs.shape == (20, 2)
    assert model.tau > 0

    # The fitted extension reproduces the base chart better than the mean does.
    g0 = model.evaluate(model.weights0)
    assert np.abs(g0 - y0).mean() < 0.5 * baseline


def test_chart_is_reused_per_layout_revision():
    """Refitting the chart per drag would recompute a fitted quantity."""
    from backend.embeddings.landmark_projection import (
        landmark_model_for,
        reset_landmark_model_cache,
    )

    reset_landmark_model_cache()
    cache, infos = _infos(n=40)
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    base = [
        {"path": t["path"], "x": float(clap[i, 0]), "y": float(clap[i, 1])}
        for i, t in enumerate(infos)
    ]
    first = landmark_model_for(
        "rev-1",
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.5,
        n_landmarks=20,
    )
    second = landmark_model_for(
        "rev-1",
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.5,
        n_landmarks=20,
    )
    assert first is not None and first is second
    # A different chart revision is a different chart.
    third = landmark_model_for(
        "rev-2",
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.5,
        n_landmarks=20,
    )
    assert third is not None and third is not first
    reset_landmark_model_cache()
