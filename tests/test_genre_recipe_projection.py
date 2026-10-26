"""Opt-in genre_full recipe: layout revision, cache-version, and budget propagation.

The default recipe must stay byte-identical (same revision/cache keys as before
this slice); the opt-in genre_full recipe must key its own revision/fit on the
genre block cache generations and its 50/0 openTSNE budget (encoded
absolutely so a default never aliases a legacy budget), and a user-chosen
iteration budget must never share a fit with the recipe default.

Synthetic data only — no models, no inference, no cache writes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.embeddings.coverage import (
    CachedSourceMaps,
    SourceVersions,
    eligible_paths_for_projection,
)
from backend.embeddings.layout_revision import compute_layout_revision
from backend.embeddings.projection_config import (
    GENRE_FULL_RECIPE_WEIGHTS,
    GENRE_FULL_SOURCES,
    recipe_for,
    required_sources_for,
    tsne_iteration_suffix,
)
from backend.services.layout_revision_for_status import layout_revision_for_projection
from backend.services.lens_layout import versions_tuple
from backend.services.projection_request import parse_projection_params


class _Req:
    def __init__(self, args: dict[str, str]):
        self.args = args


_INFOS = [
    {"path": "a.mp3", "fingerprint": "f1"},
    {"path": "b.mp3", "fingerprint": "f2"},
]
_MAPS = CachedSourceMaps(
    clap={},
    features={},
    maest={"f1": True, "f2": True},
    maest_logits={"f1": True, "f2": True},
)
_CV = (1, 2, 3, 4, 5)


def _versions() -> SourceVersions:
    return SourceVersions(
        clap=1,
        features=2,
        maest=3,
        rhythm=4,
        supervised=5,
        structure=6,
        maest_layer7=7,
        mert_l19=8,
    )


def _revision(proj, versions=None) -> str:
    versions = versions or _versions()
    return layout_revision_for_projection(
        _INFOS, _MAPS, proj,
        cache_versions=versions_tuple(versions, getattr(proj, "recipe_id", None)),
    )


# ── default recipe byte-identity ────────────────────────────


def test_default_recipe_revision_is_byte_identical():
    """A default-request proj hashes exactly like the pre-recipe form.

    ``layout_revision_for_projection`` must not add recipe/tsne fields for the
    default recipe, or every existing cached layout would be invalidated.
    """
    proj = parse_projection_params(_Req({}))
    rev = _revision(proj)
    assert rev == compute_layout_revision(
        eligible_paths=eligible_paths_for_projection(
            _INFOS, _MAPS, proj.sources, proj.required_sources
        ),
        cache_versions=_CV,
        weights=proj.source_weights,
        sources=proj.sources,
    )
    # The default recipe's cache_versions are the original 5-tuple (no genre blocks).
    assert versions_tuple(_versions()) == _CV
    assert versions_tuple(_versions(), None) == _CV


def test_default_recipe_tsne_suffix_is_empty():
    recipe = recipe_for(None)
    assert (recipe.tsne_n_iter, recipe.tsne_early_exaggeration_iter) == (50, 0)
    assert tsne_iteration_suffix(50, 0, recipe) == ""
    # A deviating budget is encoded, so it refits rather than reusing the default.
    assert tsne_iteration_suffix(250, 150, recipe) == "n250|e150"


# ── genre_full recipe revision ──────────────────────────────


def test_genre_full_revision_includes_genre_cache_versions():
    """The genre recipe extends cache_versions with layer-7 + MERT generations."""
    proj = parse_projection_params(_Req({"recipe": "genre_full"}), genre_full_enabled=True)
    versions = _versions()
    assert versions_tuple(versions, "genre_full") == _CV + (7, 8)
    # A different MERT generation must produce a different genre revision.
    other = SimpleNamespace(**_versions().__dict__)
    other.mert_l19 = 99
    assert _revision(proj) != _revision(proj, other)


def test_genre_full_revision_differs_from_default():
    default_proj = parse_projection_params(_Req({}))
    genre_proj = parse_projection_params(_Req({"recipe": "genre_full"}), genre_full_enabled=True)
    assert _revision(genre_proj) != _revision(default_proj)


def test_genre_full_weights_are_exact_and_other_sources_zero():
    recipe = recipe_for("genre_full")
    assert recipe.sources == GENRE_FULL_SOURCES
    assert recipe.weights["maest_logits"] == pytest.approx(0.40)
    assert recipe.weights["maest"] == pytest.approx(0.20)
    assert recipe.weights["maest_layer7"] == pytest.approx(0.25)
    assert recipe.weights["mert_l19"] == pytest.approx(0.15)
    # A pure genre lens: non-genre sources are explicitly zero (no budget blending).
    for src in ("rhythm", "clap", "supervised", "features"):
        assert recipe.weights[src] == 0.0
    assert GENRE_FULL_RECIPE_WEIGHTS == recipe.weights


def test_genre_recipe_keeps_maest_pair_required_and_others_optional():
    """A track needs the MAEST pair to appear; layer-7/MERT are optional zero-fill."""
    required = required_sources_for(GENRE_FULL_SOURCES)
    assert required == ("maest_logits", "maest")
    assert "maest_layer7" not in required
    assert "mert_l19" not in required


# ── openTSNE budget propagation ─────────────────────────────


def test_tsne_budget_deviation_changes_the_layout_revision():
    default_proj = parse_projection_params(_Req({}))
    same_proj = parse_projection_params(_Req({}))
    fast_proj = parse_projection_params(_Req({"tsne_n_iter": "100"}))
    balanced_proj = parse_projection_params(_Req({"tsne_n_iter": "250", "tsne_early_exaggeration_iter": "150"}))
    # An omitted budget is the recipe default → same revision.
    assert _revision(same_proj) == _revision(default_proj)
    # Any explicit budget (even 100/100, above the 50/0 default) deviates → a
    # distinct revision (a different fit).
    assert _revision(fast_proj) != _revision(default_proj)
    assert _revision(balanced_proj) != _revision(default_proj)


def test_tsne_budget_deviation_changes_the_fit_cache_key():
    recipe = recipe_for(None)
    assert tsne_iteration_suffix(50, 0, recipe) == ""
    assert "n250" in tsne_iteration_suffix(250, 150, recipe)
    genre = recipe_for("genre_full")
    # The genre recipe encodes its effective budget *absolutely*, so its 50/0
    # default never aliases a legacy default fit under a bare key.
    assert tsne_iteration_suffix(50, 0, genre) == "n50|e0"
    assert tsne_iteration_suffix(500, 250, genre) == "n500|e250"
    assert tsne_iteration_suffix(50, 0, genre) != tsne_iteration_suffix(500, 250, genre)
    # A user override of the genre budget still encodes absolutely.
    assert tsne_iteration_suffix(250, 150, genre) == "n250|e150"


def test_early_exaggeration_zero_matches_the_default_budget():
    """``early=0`` is the default now, so it must not fork the revision.

    ``layout_revision_for_projection`` uses explicit ``is None`` checks, so an
    omitted early budget and an explicit 0 resolve to the same default; a non-zero
    early budget is still a deviation and gets its own revision.
    """
    default_proj = parse_projection_params(_Req({}))
    early0_proj = parse_projection_params(_Req({"tsne_early_exaggeration_iter": "0"}))
    assert early0_proj.tsne_early_exaggeration_iter == 0
    assert early0_proj.tsne_n_iter == 50  # n_iter unchanged by the early override
    assert _revision(early0_proj) == _revision(default_proj)
    # A non-zero deviation differs from the default.
    early150_proj = parse_projection_params(_Req({"tsne_early_exaggeration_iter": "150"}))
    assert _revision(early150_proj) != _revision(default_proj)
    assert _revision(early150_proj) == compute_layout_revision(
        eligible_paths=eligible_paths_for_projection(
            _INFOS, _MAPS, early150_proj.sources, early150_proj.required_sources
        ),
        cache_versions=_CV,
        weights=early150_proj.source_weights,
        sources=early150_proj.sources,
        tsne_early_exaggeration_iter=150,
    )


def test_genre_full_budget_propagates_to_effective_params():
    proj = parse_projection_params(_Req({"recipe": "genre_full"}), genre_full_enabled=True)
    # The effective budget the projection runs at is the recipe's 50/0 default.
    assert proj.tsne_n_iter == 50
    assert proj.tsne_early_exaggeration_iter == 0
    # The genre recipe encodes its effective budget absolutely, so the default
    # 50/0 and the gallery 500/250 override never share a revision.
    assert tsne_iteration_suffix(50, 0, recipe_for("genre_full")) == "n50|e0"
    default_proj = parse_projection_params(_Req({}))
    assert _revision(proj) != _revision(default_proj)


def test_genre_default_and_gallery_budget_revisions_differ():
    """The 50/0 default must never read the legacy 500/250 fit/layout: the
    genre recipe encodes its effective budget absolutely, so the two budgets fork
    their layout revisions (and fit keys)."""
    default_genre = parse_projection_params(
        _Req({"recipe": "genre_full"}), genre_full_enabled=True
    )
    gallery = parse_projection_params(
        _Req({
            "recipe": "genre_full",
            "tsne_n_iter": "500",
            "tsne_early_exaggeration_iter": "250",
        }),
        genre_full_enabled=True,
    )
    assert _revision(default_genre) != _revision(gallery)
