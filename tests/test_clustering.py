"""Microgenre clustering pipeline: weighting, kNN graph, Leiden, naming.

The interesting properties here are statistical, so the tests build synthetic
libraries with known ground truth and assert that the pipeline recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import clustering as cl


def _synthetic_library(
    n_genres: int = 5,
    per_genre: int = 40,
    dim: int = 64,
    spread: float = 0.6,
    seed: int = 7,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(paths, feature matrix, true genre labels) with separable genres."""
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    rows: list[np.ndarray] = []
    truth: list[int] = []
    for g in range(n_genres):
        centre = rng.normal(0, 1, dim)
        for i in range(per_genre):
            paths.append(f"g{g}/track{i}.mp3")
            rows.append(centre + rng.normal(0, spread, dim))
            truth.append(g)
    return paths, np.asarray(rows, dtype=np.float32), np.asarray(truth)


def _ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def test_weighted_matrix_applies_sqrt_weights() -> None:
    a = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=(10, 8)).astype(np.float32)

    mat, used, effective = cl.build_weighted_matrix(
        {"maest_logits": a, "rhythm": b}, {"maest_logits": 0.75, "rhythm": 0.25}
    )
    assert used == ["maest_logits", "rhythm"]
    assert effective == {"maest_logits": 0.75, "rhythm": 0.25}
    assert mat.shape == (10, 16)

    # Each block is unit-norm before weighting, so its norm after weighting is
    # sqrt(w) — which is what makes squared distances combine as w * d^2.
    left = np.linalg.norm(mat[:, :8], axis=1)
    right = np.linalg.norm(mat[:, 8:], axis=1)
    assert np.allclose(left, np.sqrt(0.75), atol=1e-5)
    assert np.allclose(right, np.sqrt(0.25), atol=1e-5)


def test_zero_weight_sources_are_dropped() -> None:
    a = np.ones((4, 3), dtype=np.float32)
    mat, used, _ = cl.build_weighted_matrix(
        {"maest": a, "features": a}, {"maest": 1.0, "features": 0.0}
    )
    assert used == ["maest"]
    assert mat.shape[1] == 3


def test_build_weighted_matrix_handles_no_usable_sources() -> None:
    mat, used, effective = cl.build_weighted_matrix({}, {"maest": 1.0})
    assert mat.size == 0 and used == [] and effective == {}


def test_reduce_dimensions_caps_components() -> None:
    x = np.random.default_rng(3).normal(size=(50, 200)).astype(np.float32)
    assert cl.reduce_dimensions(x, 32).shape == (50, 32)
    # Never asks for more components than the data can support.
    assert cl.reduce_dimensions(x, 999).shape[1] <= 200


def test_knn_graph_is_connected_after_bridging() -> None:
    """Disconnected components make Leiden's resolution parameter inert."""
    rng = np.random.default_rng(5)
    # Two far-apart blobs: without bridging this graph has 2 components.
    blob_a = rng.normal(-50, 0.2, (30, 8))
    blob_b = rng.normal(50, 0.2, (30, 8))
    mat = np.vstack([blob_a, blob_b]).astype(np.float32)

    edges, weights = cl.build_knn_graph(mat, k=5, connect_components=True)
    assert len(edges) == len(weights)

    parent = list(range(len(mat)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    assert len({find(v) for v in range(len(mat))}) == 1


def test_knn_weights_are_non_negative() -> None:
    mat = np.random.default_rng(9).normal(size=(40, 16)).astype(np.float32)
    _, weights = cl.build_knn_graph(mat, k=6)
    assert all(w >= 0.0 for w in weights)


def test_cluster_tracks_recovers_known_genres() -> None:
    paths, mat, truth = _synthetic_library()
    result = cl.cluster_tracks(paths, {"maest_logits": mat}, weights={"maest_logits": 1.0})
    assert result is not None
    assert result.n_tracks == len(paths)

    best = max(_ari(truth, lv.labels) for lv in result.levels)
    assert best > 0.85, f"clustering failed to recover genres (best ARI {best:.2f})"


def test_cluster_tracks_returns_none_for_tiny_library() -> None:
    paths = [f"t{i}.mp3" for i in range(4)]
    mat = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    assert cl.cluster_tracks(paths, {"maest_logits": mat}) is None


def test_cluster_tracks_rejects_row_count_mismatch() -> None:
    paths = [f"t{i}.mp3" for i in range(20)]
    mat = np.random.default_rng(0).normal(size=(19, 8)).astype(np.float32)
    assert cl.cluster_tracks(paths, {"maest_logits": mat}) is None


def test_multiple_resolutions_produce_a_hierarchy() -> None:
    paths, mat, _ = _synthetic_library(n_genres=6, per_genre=40, spread=1.1, seed=11)
    result = cl.cluster_tracks(
        paths,
        {"maest_logits": mat},
        weights={"maest_logits": 1.0},
        resolutions=(0.15, 1.0, 6.0),
    )
    assert result is not None
    counts = [lv.n_clusters for lv in result.levels]
    # Finer resolution must never produce fewer clusters than coarser.
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0]


def test_revision_is_deterministic_and_config_sensitive() -> None:
    paths, mat, _ = _synthetic_library(n_genres=3, per_genre=15)
    blocks = {"maest_logits": mat}
    a = cl.cluster_tracks(paths, blocks, weights={"maest_logits": 1.0})
    b = cl.cluster_tracks(paths, blocks, weights={"maest_logits": 1.0})
    assert a is not None and b is not None
    assert a.revision == b.revision

    c = cl.cluster_tracks(paths, blocks, weights={"maest_logits": 0.5})
    assert c is not None
    assert c.revision != a.revision


def test_revision_ignores_input_ordering() -> None:
    """Row order must not change identity, or caches would miss spuriously."""
    paths = [f"g/{i}.mp3" for i in range(12)]
    rev_a = cl.compute_clustering_revision(
        paths, {"maest": 1.0}, ["maest"], [1.0], 64, 10
    )
    rev_b = cl.compute_clustering_revision(
        list(reversed(paths)), {"maest": 1.0}, ["maest"], [1.0], 64, 10
    )
    assert rev_a == rev_b


def test_name_clusters_uses_distinguishing_styles() -> None:
    """Names should reflect what separates a cluster, not the library average."""
    labels = np.array([0] * 10 + [1] * 10)
    logits = np.zeros((20, 3), dtype=np.float32)
    # Every track scores high on style 0 — so it must NOT become the name.
    logits[:, 0] = 10.0
    logits[:10, 1] = 5.0  # distinguishes cluster 0
    logits[10:, 2] = 5.0  # distinguishes cluster 1

    names, evidence = cl.name_clusters(
        labels,
        logits,
        ["Electronic---Common", "Electronic---Jungle", "Electronic---Gabber"],
        top_k=2,
    )
    assert "Jungle" in names[0]
    assert "Gabber" in names[1]
    assert "Common" not in names[0] and "Common" not in names[1]
    assert evidence[0][0][0] == "Electronic---Jungle"


def test_name_clusters_tolerates_missing_logits() -> None:
    labels = np.array([0, 0, 1, 1])
    assert cl.name_clusters(labels, None, ["a", "b"]) == ({}, {})
    assert cl.name_clusters(labels, np.zeros((4, 2), np.float32), None) == ({}, {})


def test_cluster_result_level_picks_nearest_resolution() -> None:
    paths, mat, _ = _synthetic_library(n_genres=3, per_genre=15)
    result = cl.cluster_tracks(
        paths, {"maest_logits": mat}, weights={"maest_logits": 1.0},
        resolutions=(0.2, 1.0, 5.0),
    )
    assert result is not None
    assert result.level(0.9).resolution == 1.0
    assert result.level(100.0).resolution == 5.0


def test_assignments_cover_every_path() -> None:
    paths, mat, _ = _synthetic_library(n_genres=3, per_genre=15)
    result = cl.cluster_tracks(paths, {"maest_logits": mat}, weights={"maest_logits": 1.0})
    assert result is not None
    assignments = result.assignments()
    assert len(assignments) == len(paths)
    assert {a.path for a in assignments} == set(paths)
    for a in assignments:
        assert set(a.by_resolution) == {lv.resolution for lv in result.levels}


@pytest.mark.skipif(not cl.leiden_available(), reason="leidenalg not installed")
def test_leiden_reports_modularity() -> None:
    paths, mat, _ = _synthetic_library(n_genres=4, per_genre=25)
    result = cl.cluster_tracks(paths, {"maest_logits": mat}, weights={"maest_logits": 1.0})
    assert result is not None
    assert any(lv.modularity is not None for lv in result.levels)
