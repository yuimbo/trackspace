"""The cheap projection-settings read surface.

``GET /api/embeddings/projection-settings`` reuses the projection query parser and
the effective-weight helper, so the View options panel can describe every source
influence without scanning the library, loading a model or fitting a layout.
"""

from __future__ import annotations

import pytest
from flask import Flask

from backend.embeddings.coverage import CachedSourceMaps
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import FROZEN_SOURCES
from backend.factory import _register_error_handlers
from backend.routes.api_embeddings import create_api_embeddings_blueprint
from backend.services.layout_revision_for_status import (
    layout_revision_for_projection,
)
from backend.services.projection_request import (
    effective_source_weights,
    parse_projection_params,
)


class _MertStub:
    MERT_VERSION = 1

    def is_model_ready(self):
        return False

    def load_error(self):
        return None


def _boom(*_args, **_kwargs):
    raise AssertionError("projection-settings must not read tracks or compute")


def _client(*, genre_full_enabled: bool = True, mert_enabled: bool = False):
    bp = create_api_embeddings_blueprint(
        embeddings_mod=None,
        embedding_version=1,
        features_version=1,
        feature_cache=None,
        source_versions_cls=object,
        batch_fetch_maps=_boom,
        coverage_payload=_boom,
        layout_revision_for_projection=_boom,
        status_cache=None,
        library_mtime_signature=_boom,
        track_infos_cached=_boom,
        projection_query_fingerprint=_boom,
        parse_projection_params_fn=lambda req, **kw: parse_projection_params(
            req, **{**kw, "genre_full_enabled": genre_full_enabled}
        ),
        build_track_infos_fn=_boom,
        library_root_ids=lambda: [],
        resolve_virtual_path=lambda rel: rel,
        virtual_from_abs_path=lambda abs_path: abs_path,
        broadcast_embed_event=lambda _event: None,
        emit_decode_warning_once=lambda *_args: None,
        broadcaster=None,
        sse_response=None,
        enqueue_analysis=lambda **_kwargs: {},
        job_queue=None,
        maest_mod=None,
        mert_mod=_MertStub(),
        genre_full_enabled=genre_full_enabled,
        mert_enabled=mert_enabled,
        rhythm_version=1,
        structure_version=1,
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client()


def _source(payload: dict, source_id: str) -> dict:
    return next(row for row in payload["sources"] if row["id"] == source_id)


def test_genre_settings_show_zero_non_genre_weights_and_folder_basis():
    payload = _client().get(
        "/api/embeddings/projection-settings?recipe=genre_full"
    ).get_json()

    assert payload["ok"] is True
    assert payload["recipe"] == "genre_full"
    assert _source(payload, "maest_logits")["weight"] == pytest.approx(0.40)
    assert _source(payload, "maest")["weight"] == pytest.approx(0.20)
    assert _source(payload, "maest_layer7")["weight"] == pytest.approx(0.25)
    assert _source(payload, "mert_l19")["weight"] == pytest.approx(0.15)
    assert _source(payload, "rhythm")["weight"] == 0.0
    assert _source(payload, "clap")["weight"] == 0.0
    assert _source(payload, "supervised")["weight"] == 0.0
    assert _source(payload, "features")["weight"] == 0.0

    assert payload["folder_basis"]["mode"] == "centroids"
    assert payload["folder_basis"]["boost"] == pytest.approx(2.4)
    assert payload["folder_basis"]["depth_boost"] == pytest.approx(3.0)

    assert payload["weights_editable"] is True
    assert payload["weights_editable_keys"] == [
        "maest_logits",
        "maest",
        "maest_layer7",
        "mert_l19",
        "rhythm",
        "clap",
        "supervised",
    ]


def test_default_budget_and_weight_override_are_effective_without_track_reads():
    payload = _client().get(
        "/api/embeddings/projection-settings"
        "?w=rhythm:0.3&tsne_n_iter=123&tsne_early_exaggeration_iter=7"
    ).get_json()

    assert payload["budget"]["tsne_n_iter"] == 123
    assert payload["budget"]["tsne_early_exaggeration_iter"] == 7
    assert payload["budget"]["default_tsne_n_iter"] == 50
    assert payload["budget"]["default_tsne_early_exaggeration_iter"] == 0
    assert _source(payload, "rhythm")["weight"] == pytest.approx(0.3)
    assert _source(payload, "maest_logits")["weight"] == pytest.approx(0.45)


def test_genre_weights_reject_features_but_accept_seven_legal_keys():
    legal = _client().get(
        "/api/embeddings/projection-settings"
        "?recipe=genre_full&w=rhythm:0.3,maest:0.3"
    )
    assert legal.status_code == 200
    payload = legal.get_json()
    assert _source(payload, "rhythm")["weight"] == pytest.approx(0.3)
    assert _source(payload, "maest")["weight"] == pytest.approx(0.3)
    assert len(payload["weights_editable_keys"]) == 7
    assert "features" not in payload["weights_editable_keys"]

    rejected = _client().get(
        "/api/embeddings/projection-settings?recipe=genre_full&w=features:0.1"
    )
    assert rejected.status_code == 400
    assert rejected.get_json()["ok"] is False


def test_source_lens_is_fixed_to_weight_one_and_not_editable():
    payload = _client().get(
        "/api/embeddings/projection-settings?sources=clap"
    ).get_json()

    assert payload["recipe"] == "default"
    assert _source(payload, "clap")["weight"] == 1.0
    assert all(
        row["weight"] == 0.0 for row in payload["sources"] if row["id"] != "clap"
    )
    assert payload["weights_editable"] is False
    assert payload["weights_editable_keys"] == []
    assert _source(payload, "clap")["in_lens"] is True
    assert _source(payload, "maest_logits")["in_lens"] is False


def test_weight_for_inactive_source_is_masked_to_zero():
    payload = _client().get(
        "/api/embeddings/projection-settings?sources=clap&w=maest:0.45"
    ).get_json()

    # The gather only consumes the clap block; maest is not an influence.
    assert _source(payload, "maest")["weight"] == 0.0
    assert _source(payload, "maest_logits")["weight"] == 0.0
    assert _source(payload, "rhythm")["weight"] == 0.0
    # The active source keeps the value the real projection receives (the
    # parser's frozen default when ``w=`` omits it).
    assert _source(payload, "clap")["weight"] == pytest.approx(0.10)


def test_active_weight_override_is_preserved():
    payload = _client().get(
        "/api/embeddings/projection-settings?sources=clap&w=clap:0.4"
    ).get_json()

    assert _source(payload, "clap")["weight"] == pytest.approx(0.4)
    assert all(
        row["weight"] == 0.0 for row in payload["sources"] if row["id"] != "clap"
    )


def test_masked_all_zero_weight_is_reported_without_new_validation():
    """Masking can expose an all-zero effective lens; settings reports it as-is.

    The request parser still accepts the raw (inactive-positive) vector, so this
    slice must not invent a second validation rule.
    """
    response = _client().get(
        "/api/embeddings/projection-settings?sources=clap&w=clap:0"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert _source(payload, "clap")["weight"] == 0.0
    assert all(row["weight"] == 0.0 for row in payload["sources"])


@pytest.mark.parametrize(
    "query",
    ["?dims=4", "?recipe=bogus", "?method=bogus"],
)
def test_invalid_projection_query_is_json_400(query):
    response = _client().get("/api/embeddings/projection-settings" + query)
    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"]


def test_disabled_optional_model_blocks_are_annotated_truthfully():
    payload = _client(genre_full_enabled=False, mert_enabled=False).get(
        "/api/embeddings/projection-settings"
    ).get_json()

    layer7 = _source(payload, "maest_layer7")
    mert = _source(payload, "mert_l19")
    assert layer7["available"] is False
    assert mert["available"] is False

    # Genre connected but MERT opted out: layer-7 is available, MERT is not.
    partial = _client(genre_full_enabled=True, mert_enabled=False).get(
        "/api/embeddings/projection-settings?recipe=genre_full"
    ).get_json()
    assert _source(partial, "maest_layer7")["available"] is True
    assert _source(partial, "mert_l19")["available"] is False
    # The recipe's configured weight is still reported; availability is separate.
    assert _source(partial, "mert_l19")["weight"] == pytest.approx(0.15)


def test_disabled_genre_recipe_is_described_not_rejected():
    """Settings mirrors the status bootstrap: a disabled recipe stays readable.

    The projection route still rejects ``recipe=genre_full`` on a disabled server;
    only this read surface tolerates it so the panel can say *why* it is absent.
    """
    response = _client(genre_full_enabled=False, mert_enabled=False).get(
        "/api/embeddings/projection-settings?recipe=genre_full"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["recipe"] == "genre_full"
    assert _source(payload, "maest_layer7")["available"] is False
    assert _source(payload, "mert_l19")["available"] is False


class _Req:
    def __init__(self, args: dict[str, str]):
        self.args = args


def _revision(proj):
    maps = CachedSourceMaps(
        clap={"fp1": object()},
        features={},
        maest={"fp1": object()},
        maest_logits={"fp1": object()},
    )
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    return layout_revision_for_projection(
        infos, maps, proj, cache_versions=(2, 1, 1)
    )


def test_layout_revision_uses_masked_effective_weights():
    proj = parse_projection_params(_Req({"sources": "clap", "w": "maest:0.45"}))
    assert proj.source_weights is not None
    assert proj.source_weights["maest"] == pytest.approx(0.45)

    revision = _revision(proj)
    expected = compute_layout_revision(
        eligible_paths=["a.mp3"],
        cache_versions=(2, 1, 1),
        weights=effective_source_weights(proj),
        sources=("clap",),
    )
    raw = compute_layout_revision(
        eligible_paths=["a.mp3"],
        cache_versions=(2, 1, 1),
        weights=proj.source_weights,
        sources=("clap",),
    )
    assert revision == expected
    assert revision != raw


def test_default_layout_revision_keeps_its_none_weight_payload():
    proj = parse_projection_params(_Req({}))
    revision = _revision(proj)
    expected = compute_layout_revision(
        eligible_paths=["a.mp3"],
        cache_versions=(2, 1, 1),
        weights=None,
        sources=FROZEN_SOURCES,
    )
    assert revision == expected


def test_coarse_schedule_and_final_perplexity_are_described_not_counted():
    payload = _client().get(
        "/api/embeddings/projection-settings"
    ).get_json()

    coarse = payload["coarse_schedule"]
    assert coarse["fraction_count"] == 16
    assert coarse["min_fraction"] == pytest.approx(1 / 256)
    assert coarse["max_fraction"] == pytest.approx(1 / 16)
    assert coarse["iteration_budget"] == 16
    assert coarse["early_exaggeration_iter"] == 100
    assert coarse["perplexity_gamma"] == pytest.approx(4.0)
    # The textual rule must include openTSNE's small-level clip without scanning.
    assert "count - 1" in coarse["perplexity_rule"]
    assert "/ 3" in coarse["perplexity_rule"]

    assert payload["final_perplexity"]["rule"] == "min(30, max(5, (n - 1) / 3))"
    assert payload["final_perplexity"]["resolved"] is None
    assert payload["audio_features"]["blend"] == pytest.approx(0.42)
    assert payload["audio_features"]["dim"] == 6
    assert payload["budget"]["final_exaggeration"] == pytest.approx(1.25)
