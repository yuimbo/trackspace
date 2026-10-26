"""Shared test helpers."""

import pytest


class CountingConn:
    """Delegating proxy over a ``ClosingConnection`` that counts ``execute``.

    Installed by monkeypatching ``cache._conn`` / ``queue._conn`` so tests can
    assert that a memo hit skipped the SQL round-trip. ``__enter__`` returns the
    proxy so the ``with self._conn() as conn: conn.execute(...)`` idiom is
    counted; ``__exit__`` keeps the ClosingConnection's commit-then-close
    semantics.
    """

    def __init__(self, inner, counter: list[int]) -> None:
        self._inner = inner  # ClosingConnection
        self._counter = counter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def execute(self, sql, params=()):
        self._counter[0] += 1
        return self._inner._conn.execute(sql, params)

    def __getattr__(self, name):
        # Delegate to the raw connection, like ClosingConnection.__enter__ does.
        return getattr(self._inner._conn, name)


def install_counting_conn(obj, counter: list[int]) -> None:
    """Route ``obj._conn`` through a :class:`CountingConn` counting into *counter*."""

    real = obj._conn

    def wrapped():
        return CountingConn(real(), counter)

    obj._conn = wrapped


# ── layout-evaluation gallery harness ────────────────────────────────────────
#
# One real generation per named gallery set is expensive, so the fake cache,
# the generation call, the Flask app and the default-eight fixture live here
# instead of being copied into every ``test_layout_evaluation_hierarchy_*``
# module.


def build_layout_cache(tmp_path, n, seed=0):
    """``FeatureCache`` plus ``infos`` for *n* synthetic four-source tracks."""
    import numpy as np

    from backend.embeddings.clap import EMBEDDING_VERSION
    from backend.embeddings.feature_cache import FeatureCache
    from backend.embeddings.maest import MAEST_VERSION
    from backend.embeddings.rhythm_features import RHYTHM_VERSION

    fc = FeatureCache(str(tmp_path / "audio_features.db"))
    rng = np.random.RandomState(seed)
    fps = [f"fp{i:03d}" for i in range(n)]
    for fp in fps:
        fc.put("maest_logits", fp, rng.standard_normal(519).astype(np.float32), MAEST_VERSION)
        fc.put("maest", fp, rng.standard_normal(768).astype(np.float32), MAEST_VERSION)
        fc.put("rhythm", fp, rng.standard_normal(14).astype(np.float32), RHYTHM_VERSION)
        fc.put("clap", fp, rng.standard_normal(512).astype(np.float32), EMBEDDING_VERSION)
    infos = [{"path": f"lib/{fp}.mp3", "fingerprint": fp} for fp in fps]
    return fc, infos


def fake_layout_fit(reduced, *, recipe, seed, n_jobs, init=None):
    """Deterministic stand-in for ``fit_opentsne`` honouring ``recipe.dims``."""
    import numpy as np

    if init is not None:
        return np.asarray(init, dtype=np.float32)
    dims = recipe.dims or 2
    return np.random.default_rng(seed).standard_normal(
        (len(reduced), dims)).astype(np.float32)


def run_layout_generation(feature_cache, infos, output_dir, recipe_ids, reference_id):
    """One ``run_generation`` over the frozen gallery inputs."""
    import backend.embeddings.layout_evaluation as le
    from backend.embeddings.projection_config import (
        FROZEN_FEATURE_MASK,
        FROZEN_SOURCES,
        FROZEN_SOURCE_WEIGHTS,
        REQUIRED_SOURCES,
    )

    return le.run_generation(
        feature_cache=feature_cache,
        infos=infos,
        output_dir=str(output_dir),
        sources=FROZEN_SOURCES,
        source_weights=FROZEN_SOURCE_WEIGHTS,
        required_sources=REQUIRED_SOURCES,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=0.42,
        seed=42,
        n_jobs=2,
        recipe_ids=recipe_ids,
        reference_id=reference_id,
    )


def make_layout_eval_app(layout_dir):
    """A minimal Flask app serving only the layout-evaluation blueprint."""
    from flask import Flask

    from backend.factory import _register_error_handlers
    from backend.routes.api_layout_evaluation import create_api_layout_evaluation_blueprint

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(create_api_layout_evaluation_blueprint(str(layout_dir)))
    return app


def recipe_field_diff(ref, variant) -> set[str]:
    """Names of the ``Recipe`` fields on which *variant* differs from *ref*."""
    from dataclasses import asdict

    d_ref, d_var = asdict(ref), asdict(variant)
    return {k for k in d_ref if d_ref[k] != d_var[k]}


@pytest.fixture
def layout_cache(tmp_path):
    """``build(n, seed=0) -> (FeatureCache, infos)`` under this test's tmp dir."""

    def build(n, seed=0):
        return build_layout_cache(tmp_path, n, seed)

    return build


@pytest.fixture
def layout_generate(monkeypatch):
    """``generate(fc, infos, out, recipe_ids, reference_id, *, fake=False)``."""

    def generate(fc, infos, out, recipe_ids, reference_id, *, fake=False):
        if fake:
            import backend.embeddings.layout_evaluation as le

            monkeypatch.setattr(le, "fit_opentsne", fake_layout_fit)
        return run_layout_generation(fc, infos, out, recipe_ids, reference_id)

    return generate


@pytest.fixture
def layout_app():
    """``app(layout_dir) -> Flask app`` with the layout-evaluation blueprint."""
    return make_layout_eval_app


@pytest.fixture(scope="module")
def default_gallery(tmp_path_factory):
    """The artifact dir of one real default-eight generation at 10 tracks."""
    root = tmp_path_factory.mktemp("default-gallery")
    fc, infos = build_layout_cache(root, 10)
    out = root / "layout-evaluation"
    run_layout_generation(fc, infos, out, None, "baseline")
    return out
