"""Genre-space scoring experiments: recipes, compose, score_neighbours, evaluate.

Synthetic data only — no models, no caches, no inference. Pins the recipe/id
contract, L2-per-block sqrt-weight composition, cosine ranking, the aggregates,
and the dev-only-selection PCA protocol.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from backend.experiments.genre_space import scoring as gs


def _track(fp, split, label="A", artist=None, release=None, path=None):
    return {
        "fingerprint": fp,
        "path": path or f"/{fp}.mp3",
        "label": label,
        "artist": artist,
        "release": release,
        "split": split,
    }


# ── recipes ──────────────────────────────────────────────────────────────────


def test_recipes_full_set_deterministic_safe_ids():
    avail = {"maest_logits", "maest", "maest_layer7", "mert_l19", "mert_l23"}
    r = gs.recipes(avail)
    assert list(r) == [
        "maest_logits",
        "maest_logits+maest@643-357",
        "maest_logits+maest_layer7@800-200",
        "maest_logits+maest_layer7@650-350",
        "maest_logits+maest_layer7@500-500",
        "mert_l19",
        "mert_l23",
        "maest_logits+mert_l19@800-200",
        "maest_logits+mert_l19@650-350",
        "maest_logits+mert_l19@500-500",
        "maest_logits+mert_l23@800-200",
        "maest_logits+mert_l23@650-350",
        "maest_logits+mert_l23@500-500",
    ]
    assert len(r) == 13
    assert r == gs.recipes(avail)  # deterministic across calls
    assert all(re.fullmatch(r"[a-z0-9_+@-]+", rid) for rid in r)


def test_recipes_normalized_maest_pair():
    r = gs.recipes({"maest_logits", "maest"})
    assert set(r) == {"maest_logits", "maest_logits+maest@643-357"}
    w = r["maest_logits+maest@643-357"]
    # Normalised from the production 0.45/0.25 pair (sums to 1, unlike production).
    assert w == pytest.approx({"maest_logits": 0.45 / 0.70, "maest": 0.25 / 0.70})
    assert sum(w.values()) == pytest.approx(1.0)


def test_recipes_hidden_share_sweeps():
    r = gs.recipes({"maest_logits", "maest_layer7", "mert_l19"})
    assert r["maest_logits+maest_layer7@800-200"] == pytest.approx(
        {"maest_logits": 0.8, "maest_layer7": 0.2}
    )
    assert r["maest_logits+maest_layer7@650-350"] == pytest.approx(
        {"maest_logits": 0.65, "maest_layer7": 0.35}
    )
    assert r["maest_logits+maest_layer7@500-500"] == pytest.approx(
        {"maest_logits": 0.5, "maest_layer7": 0.5}
    )
    assert r["maest_logits+mert_l19@800-200"] == pytest.approx(
        {"maest_logits": 0.8, "mert_l19": 0.2}
    )
    assert r["maest_logits+mert_l19@650-350"] == pytest.approx(
        {"maest_logits": 0.65, "mert_l19": 0.35}
    )
    assert r["maest_logits+mert_l19@500-500"] == pytest.approx(
        {"maest_logits": 0.5, "mert_l19": 0.5}
    )


def test_recipes_availability_filtering():
    assert gs.recipes(set()) == {}
    assert gs.recipes({"maest"}) == {}  # maest alone is not a recipe
    assert set(gs.recipes({"mert_l19"})) == {"mert_l19"}
    assert set(gs.recipes({"maest_logits", "mert_l23"})) == {
        "maest_logits",
        "mert_l23",
        "maest_logits+mert_l23@800-200",
        "maest_logits+mert_l23@650-350",
        "maest_logits+mert_l23@500-500",
    }
    # Unknown sources are ignored, not an error.
    assert gs.recipes({"clap", "maest_logits"}) == gs.recipes({"maest_logits"})


# ── directed recipe set ──────────────────────────────────────────────────────


def test_recipes_directed_exact_five_normalized_readable_ids():
    r = gs.recipes({"maest_logits", "maest", "maest_layer7", "mert_l19"},
                   recipe_set="directed")
    assert list(r) == [
        "genre_control",
        "genre_structure",
        "genre_without_layer7",
        "genre_without_mert",
        "genre_full",
    ]
    # Every weight vector is normalised to sum 1.
    for w in r.values():
        assert sum(w.values()) == pytest.approx(1.0)
    # The directed ids are readable and route/file safe verbatim (no +/@ slug).
    for rid in r:
        assert re.fullmatch(r"[a-z0-9_]+", rid)
    assert r["genre_control"] == pytest.approx({"maest_logits": 0.65, "maest": 0.35})
    assert r["genre_structure"] == pytest.approx({"maest_logits": 0.65, "maest_layer7": 0.35})
    assert r["genre_without_layer7"] == pytest.approx(
        {"maest_logits": 0.65, "maest": 0.20, "mert_l19": 0.15}
    )
    assert r["genre_without_mert"] == pytest.approx(
        {"maest_logits": 0.55, "maest": 0.20, "maest_layer7": 0.25}
    )
    assert r["genre_full"] == pytest.approx(
        {"maest_logits": 0.40, "maest": 0.20, "maest_layer7": 0.25, "mert_l19": 0.15}
    )
    # Deterministic across calls; the source ordering is the canonical one.
    assert r == gs.recipes({"maest_logits", "maest", "maest_layer7", "mert_l19"},
                           recipe_set="directed")


def test_recipes_directed_contrast_swaps():
    """The fixed contrasts are exactly the .25 and .15 logits swaps called out."""
    r = gs.recipes({"maest_logits", "maest", "maest_layer7", "mert_l19"},
                   recipe_set="directed")
    full, no_l7, no_mert = r["genre_full"], r["genre_without_layer7"], r["genre_without_mert"]
    # full vs without_layer7: .25 of the logits weight moves to layer7 (mert fixed).
    assert full["maest_logits"] == pytest.approx(no_l7["maest_logits"] - 0.25)
    assert full["maest_layer7"] == pytest.approx(no_l7.get("maest_layer7", 0) + 0.25)
    assert full["maest"] == pytest.approx(no_l7["maest"])
    assert full["mert_l19"] == pytest.approx(no_l7["mert_l19"])
    # full vs without_mert: .15 of the logits weight moves to mert_l19 (layer7 fixed).
    assert full["maest_logits"] == pytest.approx(no_mert["maest_logits"] - 0.15)
    assert full["mert_l19"] == pytest.approx(no_mert.get("mert_l19", 0) + 0.15)
    assert full["maest"] == pytest.approx(no_mert["maest"])
    assert full["maest_layer7"] == pytest.approx(no_mert["maest_layer7"])


def test_recipes_directed_fails_clear_if_any_source_missing():
    # All-or-nothing: missing any required source raises — never a partial 5-set,
    # because the preset only makes sense as a complete comparison.
    for missing in ({"maest_logits", "maest", "maest_layer7"},  # no mert_l19
                    {"maest_logits", "maest", "mert_l19"},      # no maest_layer7
                    {"maest", "maest_layer7", "mert_l19"},      # no maest_logits
                    {"maest_logits"},):                          # far short
        with pytest.raises(ValueError, match="directed recipe set requires every source"):
            gs.recipes(missing, recipe_set="directed")
    # A baseline recipe set over the same sources still returns recipes.
    assert gs.recipes({"maest_logits", "maest", "maest_layer7"}, recipe_set="baseline")


def test_recipes_unknown_recipe_set_raises():
    with pytest.raises(ValueError, match="unknown recipe_set"):
        gs.recipes({"maest_logits"}, recipe_set="nope")


def test_recipes_baseline_unchanged():
    avail = {"maest_logits", "maest", "maest_layer7", "mert_l19", "mert_l23"}
    # Explicit recipe_set="baseline" is identical to the (default) original call.
    assert gs.recipes(avail, recipe_set="baseline") == gs.recipes(avail)
    assert len(gs.recipes(avail, recipe_set="baseline")) == 13
    assert "genre_control" not in gs.recipes(avail, recipe_set="baseline")


def test_recipe_labels_cover_directed_ids_and_are_readable():
    assert set(gs.RECIPE_LABELS) == set(gs.DIRECTED_RECIPES) | set(gs.LOGITS_AB_RECIPES)
    assert gs.RECIPE_LABELS["genre_control"] == "Control: genre logits + final MAEST"
    assert gs.RECIPE_LABELS["genre_structure"] == "Structure: genre logits + MAEST layer7"
    assert gs.RECIPE_LABELS["genre_without_layer7"] == (
        "Without layer7: logits + final MAEST + MERT19"
    )
    assert gs.RECIPE_LABELS["genre_without_mert"] == (
        "Without MERT19: logits + final MAEST + layer7"
    )
    assert gs.RECIPE_LABELS["genre_full"] == (
        "Full: logits + final MAEST + layer7 + MERT19"
    )
    assert gs.RECIPE_LABELS["genre_without_logits"] == (
        "Without logits: final MAEST + layer7 + MERT19"
    )


def test_recipes_logits_ab_exact_two_normalized_derived_from_full():
    """logits-ab is exactly two options: genre_full (reused) + its without-logits ablation."""
    avail = {"maest_logits", "maest", "maest_layer7", "mert_l19"}
    r = gs.recipes(avail, recipe_set="logits-ab")
    assert list(r) == ["genre_full", "genre_without_logits"]
    # genre_full is copied verbatim from the directed constant — one source of truth.
    assert r["genre_full"] == gs.DIRECTED_RECIPES["genre_full"]
    # without_logits drops the logits source and renormalises the rest to sum 1
    # (0.20/.25/.15 of the full recipe -> 1/3, 5/12, .25 over a .60 total).
    assert r["genre_without_logits"] == pytest.approx(
        {"maest": 1 / 3, "maest_layer7": 5 / 12, "mert_l19": 0.25}
    )
    for w in r.values():
        assert sum(w.values()) == pytest.approx(1.0)
    for rid in r:
        assert re.fullmatch(r"[a-z0-9_]+", rid)
    # Deterministic across calls.
    assert r == gs.recipes(avail, recipe_set="logits-ab")


def test_recipes_logits_ab_derived_from_direct_full_not_hand_copied():
    """The without-logits weights are derived from DIRECTED_RECIPES['genre_full']."""
    full = gs.DIRECTED_RECIPES["genre_full"]
    sub = {s: w for s, w in full.items() if s != "maest_logits"}
    expected = {s: w / sum(sub.values()) for s, w in sub.items()}
    assert gs.LOGITS_AB_RECIPES["genre_without_logits"] == pytest.approx(expected)
    # The shared genre_full entry is literally the same weight vector, not a copy
    # that could drift independently.
    assert gs.LOGITS_AB_RECIPES["genre_full"] == full


def test_recipes_logits_ab_fails_clear_if_any_source_missing():
    # All-or-nothing like directed: missing the logits source (or any source the
    # full recipe needs) raises — never a partial 1-option set.
    for missing in ({"maest", "maest_layer7", "mert_l19"},  # no maest_logits
                    {"maest_logits", "maest", "maest_layer7"},  # no mert_l19
                    {"maest_logits"}):                          # far short
        with pytest.raises(ValueError, match="logits-ab recipe set requires every source"):
            gs.recipes(missing, recipe_set="logits-ab")


def test_recipes_logits_ab_leaves_directed_and_baseline_unchanged():
    avail = {"maest_logits", "maest", "maest_layer7", "mert_l19"}
    # Directed is still the exact five, unchanged.
    assert list(gs.recipes(avail, recipe_set="directed")) == list(gs.DIRECTED_RECIPES)
    # Baseline is untouched and contains none of the named preset ids.
    base = gs.recipes(set(avail) | {"mert_l23"}, recipe_set="baseline")
    assert len(base) == 13
    assert not (set(base) & set(gs.DIRECTED_RECIPES))
    assert not (set(base) & set(gs.LOGITS_AB_RECIPES))


def test_evaluate_directed_recipe_set():
    rng = np.random.default_rng(0)
    tracks = _random_tracks()
    n = len(tracks)
    vecs = {
        "maest_logits": rng.normal(size=(n, 6)),
        "maest": rng.normal(size=(n, 5)),
        "maest_layer7": rng.normal(size=(n, 4)),
        "mert_l19": rng.normal(size=(n, 4)),
    }
    rep = gs.evaluate(vecs, tracks, k=3, pca_dim=5, seed=1, recipe_set="directed")
    assert rep["recipes"] == list(gs.DIRECTED_RECIPES)
    # All five directed recipes are scored in both geometries.
    for rid in rep["recipes"]:
        geoms = rep["scores"][rid]["geometries"]
        assert set(geoms) == {"raw", "pca5"}
        for geom in ("raw", "pca5"):
            for split in ("dev", "heldout"):
                for mode in ("unrestricted", "exclude_related"):
                    assert rep["scores"][rid]["geometries"][geom][split][mode][
                        "precision"
                    ]["macro_per_label"] is not None
    # Each recipe records its own source weights.
    for rid, weights in gs.DIRECTED_RECIPES.items():
        assert rep["scores"][rid]["weights"] == weights


def test_evaluate_directed_fails_if_source_missing():
    tracks = _random_tracks()
    n = len(tracks)
    rng = np.random.default_rng(0)
    vecs = {
        "maest_logits": rng.normal(size=(n, 6)),
        "maest": rng.normal(size=(n, 5)),
        "maest_layer7": rng.normal(size=(n, 4)),
        # mert_l19 absent — directed needs it for genre_without_layer7 / genre_full.
    }
    with pytest.raises(ValueError, match="directed recipe set requires every source"):
        gs.evaluate(vecs, tracks, k=3, pca_dim=5, seed=1, recipe_set="directed")


# ── compose ──────────────────────────────────────────────────────────────────


def test_compose_l2_sqrt_weight_concat():
    a = np.array([[3.0, 4.0], [6.0, 8.0]])  # row norms 5 and 10
    b = np.array([[0.0, 1.0], [1.0, 0.0]])  # row norms 1 and 1
    out = gs.compose({"a": a, "b": b}, {"a": 1.0, "b": 4.0})
    # Weights renormalise to sum 1 → a: 0.2, b: 0.8; each block is L2-normalised
    # per row, sqrt-weight scaled, concatenated (higher weight first: a, b).
    expected = np.array(
        [
            [3 / 5 * 0.2**0.5, 4 / 5 * 0.2**0.5, 0.0, 0.8**0.5],
            [6 / 10 * 0.2**0.5, 8 / 10 * 0.2**0.5, 0.8**0.5, 0.0],
        ],
        dtype=np.float32,
    )
    assert out.dtype == np.float32
    assert np.allclose(out, expected)
    # Renormalised weights → the composed rows are unit-norm.
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_compose_normalises_weights():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [0.0, 2.0]])
    raw = gs.compose({"a": a, "b": b}, {"a": 0.45, "b": 0.25})
    normed = gs.compose({"a": a, "b": b}, {"a": 0.45 / 0.7, "b": 0.25 / 0.7})
    # Only relative influence matters: an unnormalised pair matches its
    # normalised form exactly.
    assert np.allclose(raw, normed)


def test_compose_deterministic_and_no_mutation():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    a0, b0 = a.copy(), b.copy()
    out1 = gs.compose({"a": a, "b": b}, {"a": 0.8, "b": 0.2})
    out2 = gs.compose({"a": a, "b": b}, {"a": 0.8, "b": 0.2})
    assert np.array_equal(out1, out2)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


@pytest.mark.parametrize(
    "vectors,weights",
    [
        ({"a": np.zeros((3, 2))}, {"a": 1.0}),  # zero rows
        ({"a": np.ones((3, 2)), "b": np.ones((4, 2))}, {"a": 1.0, "b": 1.0}),  # misaligned
        ({"a": np.ones((3,))}, {"a": 1.0}),  # 1-D
        ({"a": np.full((3, 2), np.nan)}, {"a": 1.0}),  # NaN
        ({"a": np.full((3, 2), np.inf)}, {"a": 1.0}),  # inf
        ({"a": np.ones((3, 2))}, {}),  # empty weights
        ({"a": np.ones((3, 2))}, {"a": -1.0}),  # negative weight
        ({"a": np.ones((3, 2))}, {"a": 0.0}),  # zero weight
        ({"a": np.ones((3, 2))}, {"a": np.nan}),  # NaN weight
        ({"a": np.ones((3, 2))}, {"b": 1.0}),  # unknown source
    ],
)
def test_compose_rejects_invalid_inputs(vectors, weights):
    with pytest.raises(ValueError):
        gs.compose(vectors, weights)


# ── score_neighbours ─────────────────────────────────────────────────────────


def test_score_neighbours_self_excluded_by_index_not_slicing():
    # Rows 0 and 1 are identical to the query (row 3): a "drop the first sorted
    # element" implementation would either keep self or drop a genuine tied
    # neighbour, because the query is not the lowest tied index.
    M = np.array(
        [
            [1.0, 0.0],  # 0 A1 dev
            [1.0, 0.0],  # 1 A2 dev (identical to A1 and the query)
            [0.0, 1.0],  # 2 B1 dev
            [1.0, 0.0],  # 3 q  heldout
            [0.9, 0.1],  # 4 B2 dev
        ]
    )
    tracks = [
        _track("A1", "dev", "A", "X", "R1"),
        _track("A2", "dev", "A", "Y", "R2"),
        _track("B1", "dev", "B", "Z", "R3"),
        _track("q", "heldout", "A", "W", "R4"),
        _track("B2", "dev", "B", "V", "R5"),
    ]
    res = gs.score_neighbours(M, tracks, k=3, query_split="heldout")
    row = res["per_query"][0]
    fps = [nb["fingerprint"] for nb in row["neighbours"]]
    assert fps == ["A1", "A2", "B2"]  # self absent; tied A1/A2 kept in index order
    assert row["effective_k"] == 3
    assert row["precision"] == pytest.approx(2 / 3)

    # k clamps per query to the candidate count.
    res2 = gs.score_neighbours(M, tracks, k=100, query_split="heldout")
    assert res2["counts"]["effective_k_max"] == 4
    assert res2["per_query"][0]["effective_k"] == 4
    assert len(res2["per_query"][0]["neighbours"]) == 4


def test_score_neighbours_tie_break_by_input_index():
    M = np.array(
        [
            [1.0, 0.0],  # 0 q
            [0.5, 0.8660254038],  # 1 c1
            [0.5, 0.8660254038],  # 2 c2 (identical to c1 → cosine tie)
            [0.5, -0.8660254038],  # 3 c3 (same cosine, different direction)
        ]
    )
    tracks = [
        _track("q", "heldout", "A"),
        _track("c1", "dev", "B"),
        _track("c2", "dev", "B"),
        _track("c3", "dev", "B"),
    ]
    res = gs.score_neighbours(M, tracks, k=2, query_split="heldout")
    nb = res["per_query"][0]["neighbours"]
    assert [x["fingerprint"] for x in nb] == ["c1", "c2"]
    assert [x["cosine"] for x in nb] == pytest.approx([0.5, 0.5])


def test_score_neighbours_precision_confusion_related():
    M = np.array(
        [
            [1.0, 0.0],  # 0 qA  heldout
            [1.0, 0.0],  # 1 A2  dev (identical → cosine ties with self)
            [0.0, 1.0],  # 2 B1  dev
            [0.0, 1.0],  # 3 B2  dev (identical to B1 → cosine tie)
        ]
    )
    tracks = [
        _track("qA", "heldout", "A", "artA", "relA", "/qA.mp3"),
        _track("A2", "dev", "A", "artA", "relA2", "/A2.mp3"),
        _track("B1", "dev", "B", "artB", "relB", "/B1.mp3"),
        _track("B2", "dev", "B", "artB", "relB2", "/B2.mp3"),
    ]
    res = gs.score_neighbours(M, tracks, k=2, query_split="heldout")
    assert res["query_split"] == "heldout"
    assert res["candidates"] == "all other tracks (both splits), self excluded by index"
    assert res["candidate_pool"] == {"dev": 3, "heldout": 1}
    assert res["counts"] == {
        "tracks": 4,
        "queries": 1,
        "queries_scored": 1,
        "queries_omitted_no_candidates": 0,
        "effective_k_min": 2,
        "effective_k_max": 2,
    }
    assert res["precision"]["micro"] == pytest.approx(0.5)
    assert res["precision"]["macro_per_label"] == pytest.approx(0.5)
    assert res["precision"]["per_label"] == {"A": pytest.approx(0.5)}
    assert res["precision"]["label_query_counts"] == {"A": 1}
    # Row-normalised retrieved-label fractions; the same label is included.
    assert res["confusion"] == {"A": {"A": 0.5, "B": 0.5}}
    assert res["related"]["same_artist"] == {
        "concentration": pytest.approx(0.5),
        "denominator_queries": 1,
        "metadata_coverage": 1.0,
    }
    assert res["related"]["same_release"] == {
        "concentration": pytest.approx(0.0),
        "denominator_queries": 1,
        "metadata_coverage": 1.0,
    }
    row = res["per_query"][0]
    assert set(row) == {"path", "fingerprint", "label", "effective_k", "precision", "neighbours"}
    assert row["path"] == "/qA.mp3"
    assert row["fingerprint"] == "qA"
    assert row["label"] == "A"
    nb = row["neighbours"]
    assert [x["fingerprint"] for x in nb] == ["A2", "B1"]
    assert set(nb[0]) == {"path", "fingerprint", "label", "cosine"}
    assert nb[0]["path"] == "/A2.mp3"
    assert nb[0]["label"] == "A"
    assert nb[0]["cosine"] == pytest.approx(1.0)
    assert nb[1]["cosine"] == pytest.approx(0.0)


def test_score_neighbours_exclude_related():
    # Orthonormal rows: every cosine ties at 0, so ranking is pure input order.
    M = np.eye(6)
    tracks = [
        _track("q", "heldout", "A", "X", "R", "/q.mp3"),
        _track("c1", "dev", "B", "X", "R1", "/c1.mp3"),  # same artist
        _track("c2", "dev", "B", "Y", "R", "/c2.mp3"),  # same release
        _track("c3", "dev", "B", "X", "R", "/c3.mp3"),  # same artist AND release
        _track("c4", "dev", "B", "Z", "R2", "/c4.mp3"),  # unrelated
        _track("c5", "dev", "B", None, None, "/c5.mp3"),  # missing metadata
    ]
    res = gs.score_neighbours(M, tracks, k=3, query_split="heldout", exclude_related=True)
    nb = res["per_query"][0]["neighbours"]
    assert [x["fingerprint"] for x in nb] == ["c4", "c5"]
    assert res["per_query"][0]["effective_k"] == 2
    # Without exclusion the same k returns the lower-indexed related tracks.
    res2 = gs.score_neighbours(M, tracks, k=3, query_split="heldout", exclude_related=False)
    assert [x["fingerprint"] for x in res2["per_query"][0]["neighbours"]] == ["c1", "c2", "c3"]


def test_score_neighbours_missing_metadata_never_equal():
    M = np.eye(3)
    tracks = [
        _track("q", "heldout", "A", None, None),
        _track("d1", "dev", "B", None, "R"),  # unknown artist vs unknown artist
        _track("d2", "dev", "B", None, None),  # both unknown
    ]
    res = gs.score_neighbours(M, tracks, k=2, query_split="heldout", exclude_related=True)
    # Unknown-vs-unknown is never a match, so nothing is excluded.
    assert [x["fingerprint"] for x in res["per_query"][0]["neighbours"]] == ["d1", "d2"]


def test_macro_precision_weights_labels_equally():
    # qA's neighbour is a same-label A track (precision 1.0); every B query's
    # nearest candidate is the lowest-indexed track, the A cluster (0.0).
    M = np.eye(8)
    M[1] = M[0]  # a1 identical to qA
    tracks = [
        _track("qA", "heldout", "A"),
        _track("a1", "dev", "A"),
        _track("b1", "heldout", "B"),
        _track("b2", "heldout", "B"),
        _track("b3", "heldout", "B"),
        _track("b4", "heldout", "B"),
        _track("b5", "heldout", "B"),
        _track("a2", "dev", "A"),
    ]
    res = gs.score_neighbours(M, tracks, k=1, query_split="heldout")
    assert res["precision"]["per_label"] == {"A": pytest.approx(1.0), "B": pytest.approx(0.0)}
    assert res["precision"]["macro_per_label"] == pytest.approx(0.5)
    assert res["precision"]["micro"] == pytest.approx(1 / 6)
    assert res["precision"]["label_query_counts"] == {"A": 1, "B": 5}


def test_score_neighbours_validation():
    tracks = [_track("t0", "dev", "A"), _track("t1", "dev", "A"), _track("t2", "dev", "A")]
    M = np.eye(3)
    with pytest.raises(ValueError):
        gs.score_neighbours(np.eye(4), tracks)  # row count mismatch
    with pytest.raises(ValueError):
        gs.score_neighbours(np.ones(3), tracks)  # 1-D matrix
    with pytest.raises(ValueError):
        gs.score_neighbours(np.full((3, 2), np.nan), tracks)  # non-finite
    bad = np.eye(3)
    bad[1] = 0.0
    with pytest.raises(ValueError):
        gs.score_neighbours(bad, tracks)  # zero row
    with pytest.raises(ValueError):
        gs.score_neighbours(M, [_track("t0", "dev", "A")])  # missing split key
    with pytest.raises(ValueError):
        gs.score_neighbours(M, [_track("t0", "dev", "")])  # empty label
    with pytest.raises(ValueError):
        gs.score_neighbours(M, [_track("t0", "train", "A")])  # invalid split
    with pytest.raises(ValueError):
        gs.score_neighbours(M, tracks, query_split="train")
    with pytest.raises(ValueError):
        gs.score_neighbours(M, tracks, k=0)


def test_score_neighbours_no_candidate_queries_omitted():
    M = np.array([[1.0, 0.0]])
    tracks = [_track("only", "heldout", "A")]
    res = gs.score_neighbours(M, tracks, k=3, query_split="heldout")
    assert res["counts"]["queries"] == 1
    assert res["counts"]["queries_scored"] == 0
    assert res["counts"]["queries_omitted_no_candidates"] == 1
    assert res["precision"]["micro"] is None
    assert res["precision"]["macro_per_label"] is None
    assert res["precision"]["per_label"] == {}
    assert res["confusion"] == {}
    assert res["per_query"] == []


# ── PCA dev-only fit contract ────────────────────────────────────────────────


def test_pca_dev_fit_contract():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(3)
    mat = rng.normal(size=(8, 5))
    dev_idx = [0, 1, 2, 3]
    before = mat.copy()
    proj, info = gs.pca_dev_fit_transform(mat, dev_idx, 3, seed=7)
    assert info["fit_on"] == "dev rows only"
    assert info["components"] == 3
    # Fitted only on the dev rows, then applied to every row — never a
    # fit_transform on the whole matrix.
    expected = (
        PCA(n_components=3, svd_solver="randomized", random_state=7)
        .fit(mat[dev_idx])
        .transform(mat)
    )
    assert np.allclose(proj, expected)
    full_fit = (
        PCA(n_components=3, svd_solver="randomized", random_state=7).fit(mat).transform(mat)
    )
    assert not np.allclose(proj, full_fit)  # a whole-matrix fit differs
    assert np.array_equal(mat, before)  # input untouched
    # Components capped to min(dim, n_dev - 1, n_features - 1).
    _, info2 = gs.pca_dev_fit_transform(mat, dev_idx, 50, seed=7)
    assert info2["components"] == 3  # min(50, 4 - 1, 5 - 1)
    with pytest.raises(ValueError):
        gs.pca_dev_fit_transform(mat, [0], 3, seed=7)  # fewer than 2 dev rows


# ── evaluate ─────────────────────────────────────────────────────────────────


def _random_tracks(n_dev=4, n_held=4, seed=0):
    rng = np.random.default_rng(seed)
    tracks = [
        _track(f"d{i}", "dev", "A" if i % 2 == 0 else "B", f"artist{i}", f"rel{i}")
        for i in range(n_dev)
    ] + [
        _track(f"h{i}", "heldout", "B" if i % 2 == 0 else "A", f"artist{i + n_dev}", f"rel{i + n_dev}")
        for i in range(n_held)
    ]
    return tracks


def test_evaluate_report_structure():
    rng = np.random.default_rng(0)
    tracks = _random_tracks()
    n = len(tracks)
    vecs = {"maest_logits": rng.normal(size=(n, 6)), "mert_l19": rng.normal(size=(n, 4))}
    rep = gs.evaluate(vecs, tracks, k=3, pca_dim=5, seed=1)

    assert rep["selection"] == "dev"
    assert rep["dataset"] == {
        "tracks": 8,
        "intersection_rows": 8,
        "dropped_rows": 0,
        "splits": {"dev": 4, "heldout": 4},
    }
    assert rep["recipes"] == [
        "maest_logits",
        "mert_l19",
        "maest_logits+mert_l19@800-200",
        "maest_logits+mert_l19@650-350",
        "maest_logits+mert_l19@500-500",
    ]
    assert rep["geometries"] == ["raw", "pca5"]

    for rid in rep["recipes"]:
        geoms = rep["scores"][rid]["geometries"]
        assert set(geoms) == {"raw", "pca5"}
        assert geoms["pca5"]["components"] >= 1
        assert geoms["pca5"]["fit_on"] == "dev rows only"
        for geom in ("raw", "pca5"):
            entry = geoms[geom]
            for split in ("dev", "heldout"):
                for mode in ("unrestricted", "exclude_related"):
                    s = entry[split][mode]
                    # Full score_neighbours reports are retained for CLI/CSV
                    # consumers — nothing is reduced to a summary here.
                    assert s["precision"]["macro_per_label"] is not None
                    assert s["precision"]["micro"] is not None
                    assert s["counts"]["queries_scored"] >= 1
                    assert "per_query" in s and "confusion" in s and "related" in s

    assert set(rep["selected"]) == {"maest_logits", "mert_l19"}
    for fam, sel in rep["selected"].items():
        assert sel["selection_split"] == "dev"
        assert sel["selection_scoring"] == "unrestricted"
        assert sel["recipe"] in rep["recipes"]
        assert sel["geometry"] in rep["geometries"]
        assert sel["dev"]["unrestricted"]["macro_per_label"] is not None
        assert sel["dev"]["exclude_related"]["macro_per_label"] is not None
        assert sel["heldout"]["unrestricted"]["macro_per_label"] is not None
        assert sel["heldout"]["exclude_related"]["macro_per_label"] is not None
    assert rep["selected_recipe_ids"] == [
        rep["selected"][fam]["recipe"] for fam in sorted(rep["selected"])
    ]
    assert rep["pca"]["fit_on"] == "dev rows only"


def test_evaluate_selection_uses_dev_unrestricted_only():
    # The alone recipe wins the dev *unrestricted* macro (1.0), but its dev
    # *exclude_related* macro (0.0) is worse than the mix recipes' (1.0) — so a
    # selection that peeked at the diagnostic would pick the wrong recipe.
    mert = np.eye(8)  # orthonormal → index-order neighbours for the alone recipe
    maest = np.zeros((8, 2))
    maest[0] = [1.0, 0.0]  # q1
    maest[3] = [0.99, 0.01]  # b1: nearest in maest space (label B)
    maest[5] = [0.9, 0.1]  # a3: next (label A)
    maest[[1, 2, 4, 6, 7]] = [0.0, 1.0]  # everything else: cos 0 with q1
    tracks = [
        _track("q1", "dev", "A", "X"),
        _track("m1", "heldout", "A", "X"),
        _track("m2", "heldout", "B", "Z"),
        _track("b1", "heldout", "B", "X"),
        _track("a2", "heldout", "A", "X"),
        _track("a3", "heldout", "A", "W"),
        _track("h6", "heldout", "B", "U"),
        _track("h7", "heldout", "A", "T"),
    ]
    rep = gs.evaluate({"maest_logits": maest, "mert_l19": mert}, tracks, k=1, seed=42)

    sel = rep["selected"]["mert_l19"]
    assert sel["recipe"] == "mert_l19"
    assert sel["dev"]["unrestricted"]["macro_per_label"] == pytest.approx(1.0)
    # The diagnostic disagrees with the selection → the selection cannot have
    # used it: the alone recipe's excluded score is 0.0 while the mix's is 1.0.
    alone_excl = rep["scores"]["mert_l19"]["geometries"]["raw"]["dev"]["exclude_related"][
        "precision"
    ]["macro_per_label"]
    mix_excl = rep["scores"]["maest_logits+mert_l19@800-200"]["geometries"]["raw"]["dev"][
        "exclude_related"
    ]["precision"]["macro_per_label"]
    assert alone_excl == pytest.approx(0.0)
    assert mix_excl == pytest.approx(1.0)
    assert rep["selected_recipe_ids"] == ["maest_logits", "mert_l19"]


def test_evaluate_selection_tie_stable_recipe_name():
    def simplex_rows(m):
        # All pairwise cosines equal (1/2): rankings are pure input order, so
        # every recipe ties on dev and the lexicographically smallest id wins.
        rows = []
        for i in range(m):
            v = np.zeros(m + 1)
            v[0] = 1.0
            v[i + 1] = 1.0
            rows.append(v)
        return np.asarray(rows, dtype=np.float64)

    m = 7
    tracks = [_track("d0", "dev", "A", "X", "R")] + [
        _track(f"h{i}", "heldout", "A" if i % 2 == 0 else "B", f"Y{i}", f"S{i}")
        for i in range(6)
    ]
    rows = simplex_rows(m)
    vecs = {"maest_logits": rows, "mert_l19": rows.copy(), "mert_l23": rows.copy()}
    rep = gs.evaluate(vecs, tracks, k=3, seed=42)

    # One dev row → the PCA geometry cannot fit for any recipe.
    for rid in rep["recipes"]:
        assert rep["scores"][rid]["geometries"]["pca50"]["status"] == "unavailable"
    sel = rep["selected"]["maest_logits"]
    assert {
        k: sel[k] for k in ("recipe", "geometry", "selection_split", "selection_scoring")
    } == {
        "recipe": "maest_logits",
        "geometry": "raw",
        "selection_split": "dev",
        "selection_scoring": "unrestricted",
    }
    assert rep["selected"]["mert_l19"]["recipe"] == "maest_logits+mert_l19@500-500"
    assert rep["selected"]["mert_l23"]["recipe"] == "maest_logits+mert_l23@500-500"
    assert rep["selected_recipe_ids"] == [
        "maest_logits",
        "maest_logits+mert_l19@500-500",
        "maest_logits+mert_l23@500-500",
    ]


def test_evaluate_selection_is_dev_argmax():
    rng = np.random.default_rng(11)
    tracks = _random_tracks(seed=11)
    n = len(tracks)
    vecs = {
        "maest_logits": rng.normal(size=(n, 6)),
        "maest": rng.normal(size=(n, 5)),
        "mert_l19": rng.normal(size=(n, 4)),
    }
    rep = gs.evaluate(vecs, tracks, k=2, pca_dim=4, seed=2)
    for fam, sel in rep["selected"].items():
        candidates = [
            (rid, geom)
            for rid in rep["recipes"]
            if rep["scores"][rid]["family"] == fam
            for geom in rep["geometries"]
            if "dev" in rep["scores"][rid]["geometries"][geom]
        ]
        assert candidates
        best = min(
            candidates,
            key=lambda p: (
                -rep["scores"][p[0]]["geometries"][p[1]]["dev"]["unrestricted"]["precision"][
                    "macro_per_label"
                ],
                p[0],
                p[1],
            ),
        )
        assert (sel["recipe"], sel["geometry"]) == best


def test_evaluate_intersection_drops_zero_rows_without_mutation():
    rng = np.random.default_rng(5)
    tracks = [_track(f"d{i}", "dev", "A", f"a{i}", f"r{i}") for i in range(3)] + [
        _track(f"h{i}", "heldout", "B", f"a{i + 3}", f"r{i + 3}") for i in range(4)
    ]
    n = 7
    ml = rng.normal(size=(n, 5))
    mt = rng.normal(size=(n, 4))
    mt[5] = 0.0  # heldout track h2 has no mert_l19 row
    ml_before, mt_before = ml.copy(), mt.copy()
    rep = gs.evaluate({"maest_logits": ml, "mert_l19": mt}, tracks, k=2, pca_dim=3, seed=3)
    assert rep["dataset"]["tracks"] == 6
    assert rep["dataset"]["dropped_rows"] == 1
    assert rep["dataset"]["splits"] == {"dev": 3, "heldout": 3}
    for rid in rep["recipes"]:
        held = rep["scores"][rid]["geometries"]["raw"]["heldout"]["unrestricted"]
        assert held["counts"]["queries_scored"] == 3
    assert np.array_equal(ml, ml_before) and np.array_equal(mt, mt_before)


def test_evaluate_progress_callback_reports_each_recipe_twice():
    from collections import Counter

    rng = np.random.default_rng(31)
    tracks = _random_tracks(seed=31)
    n = len(tracks)
    vecs = {
        "maest_logits": rng.normal(size=(n, 5)),
        "mert_l19": rng.normal(size=(n, 4)),
    }
    calls: list[str] = []
    rep = gs.evaluate(vecs, tracks, k=2, pca_dim=4, seed=1, on_progress=calls.append)
    counts = Counter(calls)
    # Each recipe id fires twice (before composing/PCA and after complete); the
    # payload is the bare id — no fabricated durations or other strings.
    assert set(counts) == set(rep["recipes"])
    assert all(count == 2 for count in counts.values())
    # No callback (the default) changes nothing and prints nothing.
    rep2 = gs.evaluate(vecs, tracks, k=2, pca_dim=4, seed=1)
    assert rep2 == rep


def test_evaluate_deterministic():
    rng = np.random.default_rng(21)
    tracks = _random_tracks(seed=21)
    n = len(tracks)
    vecs = {"maest_logits": rng.normal(size=(n, 5)), "mert_l23": rng.normal(size=(n, 4))}
    a = gs.evaluate(vecs, tracks, k=2, pca_dim=4, seed=42)
    b = gs.evaluate(vecs, tracks, k=2, pca_dim=4, seed=42)
    assert a == b


def test_evaluate_validation():
    tracks = [_track("d0", "dev", "A"), _track("h0", "heldout", "A")]
    with pytest.raises(ValueError):
        gs.evaluate({}, tracks)  # empty vectors
    with pytest.raises(ValueError):
        gs.evaluate({"clap": np.eye(2)}, tracks)  # no recipes for these sources
    with pytest.raises(ValueError):
        gs.evaluate({"maest_logits": np.eye(3)}, tracks)  # row count mismatch
    with pytest.raises(ValueError):
        gs.evaluate(  # a provided source no recipe uses is rejected, not ignored
            {"maest_logits": np.eye(2), "clap": np.eye(2)}, tracks
        )
    with pytest.raises(ValueError):
        gs.evaluate(  # no dev tracks
            {"maest_logits": np.eye(2)},
            [_track("h0", "heldout", "A"), _track("h1", "heldout", "A")],
        )
