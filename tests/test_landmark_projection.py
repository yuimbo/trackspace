"""Landmark extension: exact weight Jacobian, residual form, determinism.

The Jacobian is checked against *central differences of the same frozen ``g``*
(not against an independent refit, whose different local optimum is not a
derivative). A first-order method's residual must fall with the step size; a
refit-based finite difference does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings.landmark_projection import (
    LandmarkModel,
    _remove_gauge,
    build_landmark_model,
)


def _model(seed: int = 0, n: int = 80, m: int = 8, dims: int = 2) -> LandmarkModel:
    rng = np.random.default_rng(seed)
    sources = ("a", "b", "c")
    delta = {
        s: (rng.random((n, m)).astype(np.float32) * 4.0) for s in sources
    }
    return LandmarkModel(
        paths=[f"t{i}.mp3" for i in range(n)],
        sources=sources,
        delta=delta,
        coeffs=rng.normal(size=(m, dims)),
        base=rng.random((n, dims)).astype(np.float32),
        tau=1.0,
        weights0={"a": 0.4, "b": 0.35, "c": 0.25},
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_raw_jacobian_matches_central_differences_of_g(dims):
    """The closed form must equal a central difference of the *same* g."""
    model = _model(dims=dims)
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


def test_analytic_jacobian_at_zero_for_a_new_source():
    """A source at weight 0 (a genre override you can raise) still has a Jacobian.

    The analytic derivative of the frozen ``g`` is defined at w_s = 0 — it is not
    skipped like the finite-difference fallback used to. This is the column the
    weight slider previews when dragging a zero-weighted genre source up.
    """
    rng = np.random.default_rng(11)
    n, m, dims = 64, 10, 2
    sources = ("genre", "maest", "layer7", "mert", "rhythm", "clap", "supervised")
    delta = {s: (rng.random((n, m)).astype(np.float32) * 4.0) for s in sources}
    model = LandmarkModel(
        paths=[f"t{i}.mp3" for i in range(n)],
        sources=sources,
        delta=delta,
        coeffs=rng.normal(size=(m, dims)),
        base=rng.random((n, dims)).astype(np.float32),
        tau=1.0,
        weights0={
            "genre": 0.40, "maest": 0.20, "layer7": 0.25, "mert": 0.15,
            "rhythm": 0.0, "clap": 0.0, "supervised": 0.0,
        },
    )
    jac = model.jacobian(dict(model.weights0))
    # Every source — including the zero-weight ones — has a Jacobian column.
    assert set(jac) == set(sources)
    for s in ("rhythm", "clap", "supervised"):
        assert jac[s].shape == (n, dims)
        # A genuinely informative (non-gauge) field for a zero source: raising it
        # re-shapes the map, so its gauge-projected field is not all zero.
        assert np.abs(jac[s]).max() > 0


@pytest.mark.parametrize("dims", [2, 3])
def test_remove_gauge_kills_translation_and_zoom(dims):
    """A pure offset/zoom field carries no shape change and must vanish."""
    rng = np.random.default_rng(0)
    base = rng.random((50, dims))
    r = base - base.mean(axis=0)
    # alpha + beta * r  ==  per-axis translation + one shared zoom about the
    # centroid (the gauge the projector removes).
    gauge = 0.7 + 0.4 * r
    assert np.abs(_remove_gauge(gauge, base)).max() < 1e-6
    # A genuine shape change survives.
    shape = rng.normal(size=(50, dims))
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


@pytest.mark.parametrize("dims", [2, 3])
def test_preview_reproduces_the_base_layout_at_w0(dims):
    """The residual form gives y~_i(w0) = y_i^0 exactly."""
    model = _model(seed=2, dims=dims)
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


@pytest.mark.parametrize("dims", [2, 3])
def test_build_fits_against_the_base_layout(dims):
    cache, infos = _infos(n=40)
    # A chart is a smooth function of the composite, so make the synthetic base
    # layout one too: normalised composite coordinates (z only for dims=3).
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    keys = ("x", "y", "z")[:dims]
    base = [
        {"path": t["path"], **{k: float(clap[i, j]) for j, k in enumerate(keys)}}
        for i, t in enumerate(infos)
    ]
    y0 = np.array([[p[k] for k in keys] for p in base])
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
    assert model.coeffs.shape == (20, dims)
    assert model.base.shape == (40, dims)
    assert model.tau > 0

    # The fitted extension reproduces the base chart better than the mean does.
    g0 = model.evaluate(model.weights0)
    assert np.abs(g0 - y0).mean() < 0.5 * baseline


def test_build_tolerates_a_source_without_rows():
    """A source with no cached rows (e.g. an untrained head) must not break it."""
    cache, infos = _infos(n=40)
    del cache._maps["features"]
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    base = [
        {"path": t["path"], "x": float(clap[i, 0]), "y": float(clap[i, 1])}
        for i, t in enumerate(infos)
    ]

    model = build_landmark_model(
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.37,
        required_sources=("clap",),
        n_landmarks=20,
    )
    assert model is not None
    assert set(model.sources) == {"clap"}
    assert set(model.delta) == {"clap"}
    assert set(model.raw_jacobian(model.weights0)) == {"clap"}


def test_place_infos_do_not_move_the_base_chart():
    """Placing extra tracks must leave the fitted base chart bit-identical.

    This is the invariant behind "showing external roots never moves my library":
    the landmarks, bandwidth and coefficients come from the base layout, and the
    extra rows are only evaluated through the same frozen ``g``.
    """
    cache, infos = _infos(n=40)
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    base = [
        {"path": t["path"], "x": float(clap[i, 0]), "y": float(clap[i, 1])}
        for i, t in enumerate(infos)
    ]
    extra = [{"path": f"x/ext/t{i}.mp3", "fingerprint": f"xfp{i}"} for i in range(6)]
    rng = np.random.default_rng(11)
    for t in extra:
        cache._maps["clap"][t["fingerprint"]] = rng.standard_normal(6).astype(
            np.float32
        )
        cache._maps["features"][t["fingerprint"]] = rng.standard_normal(6).astype(
            np.float32
        )

    kwargs = {
        "sources": ("clap", "features"),
        "feature_mask": np.ones(6, dtype=np.float32),
        "features_blend": 0.5,
        "n_landmarks": 20,
    }
    plain = build_landmark_model(infos, cache, base, **kwargs)
    extended = build_landmark_model(infos, cache, base, place_infos=extra, **kwargs)
    assert plain is not None and extended is not None
    assert len(plain.paths) == len(infos)
    assert len(extended.paths) == len(infos) + len(extra)

    g_plain = plain.evaluate(plain.weights0)
    g_ext = extended.evaluate(extended.weights0)
    np.testing.assert_array_equal(g_ext[: len(infos)], g_plain)
    # The extra rows land in the chart's own frame (a convex combination of the
    # landmark coordinates), not at a fixed preview corner.
    placed = g_ext[len(infos):]
    assert np.isfinite(placed).all()
    assert ((placed >= -0.5) & (placed <= 1.5)).all()


def test_place_infos_without_the_required_source_are_dropped():
    """A placed set missing a source block cannot be measured; skip it whole."""
    cache, infos = _infos(n=40)
    clap = np.array([cache._maps["clap"][t["fingerprint"]] for t in infos])
    clap = clap / (np.linalg.norm(clap, axis=1, keepdims=True) + 1e-9)
    base = [
        {"path": t["path"], "x": float(clap[i, 0]), "y": float(clap[i, 1])}
        for i, t in enumerate(infos)
    ]
    # Only the clap row is present, so the features column would shift.
    extra = [{"path": "x/ext/t0.mp3", "fingerprint": "xfp0"}]
    cache._maps["clap"]["xfp0"] = np.ones(6, dtype=np.float32)

    model = build_landmark_model(
        infos,
        cache,
        base,
        sources=("clap", "features"),
        feature_mask=np.ones(6, dtype=np.float32),
        features_blend=0.5,
        n_landmarks=20,
        place_infos=extra,
    )
    assert model is not None
    assert len(model.paths) == len(infos)


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
