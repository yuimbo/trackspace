"""Microgenre clustering: weighted block distance → cosine kNN graph → Leiden.

Implements sections 5, 6 and 8 of the microgenre attack plan. Three decisions
here are deliberate and easy to get wrong, so they are spelled out:

**Cluster in feature space, never in UMAP/t-SNE space.** 2-D projections
distort global geometry; they are for *looking at*, not for grouping by. The
pipeline is ``features → standardize → PCA(64–256) → cosine kNN → Leiden`` and
the 2-D layout is computed separately.

**Weighted blocks, not blind concatenation.** Each source (MAEST style logits,
MAEST embedding, rhythm, CLAP, EffNet, classical features) is L2-normalised
*within its block* and then scaled by ``sqrt(weight)``. Because squared
Euclidean distance decomposes as a sum over blocks, scaling a block by
``sqrt(w)`` makes it contribute exactly ``w ×`` its distance — so the familiar
weighted-distance formula from plan §8 falls out of ordinary PCA/kNN on the
stacked matrix, with no custom metric required.

**Multi-resolution, not one true granularity.** Leiden runs at several
resolutions to produce a hierarchy (broad genre → microgenre) instead of
pretending a single correct cut exists. Clusters are named from the MAEST
Discogs style logits of their members.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Bump when the clustering recipe changes (invalidates cached cluster runs).
CLUSTERING_VERSION = 1

#: Plan §8 starting weights. MAEST's genre/style space leads; rhythm is a
#: meaningful but minority voice; CLAP/EffNet are complementary texture.
DEFAULT_WEIGHTS: dict[str, float] = {
    "maest_logits": 0.45,
    "maest": 0.25,
    "rhythm": 0.15,
    "clap": 0.10,
    "effnet": 0.05,
    "features": 0.00,
}

#: Resolutions swept to build the hierarchy (coarse → fine).
#:
#: Calibrated against the real library (~9.5k cached CLAP vectors, k=20), which
#: yields roughly 2 / 3 / 6 / 12 / 25 / 60 clusters across this range — broad
#: genre families through to microgenres. Below ~0.1 the whole graph collapses
#: into a single cluster, so the sweep starts above that.
DEFAULT_RESOLUTIONS: tuple[float, ...] = (0.15, 0.4, 0.8, 1.5, 3.0, 6.0)

DEFAULT_PCA_DIM = 128
DEFAULT_KNN_K = 20
MIN_TRACKS_FOR_CLUSTERING = 8


@dataclass(frozen=True)
class ClusterAssignment:
    """One track's cluster id at each resolution."""

    path: str
    #: resolution → cluster id
    by_resolution: dict[float, int]


@dataclass
class ClusterLevel:
    """Clustering outcome at one Leiden resolution."""

    resolution: float
    labels: np.ndarray
    n_clusters: int
    sizes: dict[int, int]
    names: dict[int, str] = field(default_factory=dict)
    #: cluster id → ranked (style, score) evidence behind the name
    style_evidence: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    modularity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "n_clusters": self.n_clusters,
            "sizes": {str(k): v for k, v in self.sizes.items()},
            "names": {str(k): v for k, v in self.names.items()},
            "modularity": self.modularity,
        }


@dataclass
class ClusterResult:
    """Full multi-resolution clustering of one track set."""

    paths: list[str]
    levels: list[ClusterLevel]
    revision: str
    weights: dict[str, float]
    sources_used: list[str]
    n_tracks: int
    pca_dim: int
    knn_k: int

    def level(self, resolution: float) -> ClusterLevel | None:
        best: ClusterLevel | None = None
        for lv in self.levels:
            if best is None or abs(lv.resolution - resolution) < abs(
                best.resolution - resolution
            ):
                best = lv
        return best

    def assignments(self) -> list[ClusterAssignment]:
        out: list[ClusterAssignment] = []
        for i, p in enumerate(self.paths):
            out.append(
                ClusterAssignment(
                    path=p,
                    by_resolution={
                        lv.resolution: int(lv.labels[i]) for lv in self.levels
                    },
                )
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "n_tracks": self.n_tracks,
            "sources_used": self.sources_used,
            "weights": self.weights,
            "pca_dim": self.pca_dim,
            "knn_k": self.knn_k,
            "clustering_version": CLUSTERING_VERSION,
            "levels": [lv.as_dict() for lv in self.levels],
            "resolutions": [lv.resolution for lv in self.levels],
        }


# ──────────────────────────────────────────────────────────────
# Feature assembly (plan §8)
# ──────────────────────────────────────────────────────────────


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (mat / norms).astype(np.float32)


def _standardize(mat: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance per column, tolerant of constant columns."""
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return ((mat - mu) / sd).astype(np.float32)


def build_weighted_matrix(
    source_blocks: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Stack per-source blocks into one weighted matrix.

    Each block is standardized, L2-normalised per row, then multiplied by
    ``sqrt(weight)``. Squared Euclidean distance is additive over blocks, so
    this reproduces ``d² = Σ wᵢ · dᵢ²`` — plan §8's weighted distance — while
    letting ordinary PCA and kNN operate on a plain matrix.

    Sources with zero/absent weight, or with no rows, are skipped.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    used: list[str] = []
    blocks: list[np.ndarray] = []
    effective: dict[str, float] = {}

    for name, block in source_blocks.items():
        weight = float(w.get(name, 0.0))
        if weight <= 0.0 or block is None:
            continue
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        # Standardize first so a block with wild scale (raw logits) cannot
        # dominate purely through magnitude before normalisation.
        norm = _l2_normalize_rows(_standardize(arr))
        blocks.append(norm * float(np.sqrt(weight)))
        used.append(name)
        effective[name] = weight

    if not blocks:
        return np.empty((0, 0), dtype=np.float32), [], {}

    return np.concatenate(blocks, axis=1).astype(np.float32), used, effective


def reduce_dimensions(mat: np.ndarray, n_components: int = DEFAULT_PCA_DIM) -> np.ndarray:
    """PCA via SVD to at most *n_components* dims (plan §5)."""
    n_samples, n_features = mat.shape
    k = int(min(n_components, n_features, max(1, n_samples - 1)))
    if k >= n_features:
        return mat.astype(np.float32)
    centred = mat - mat.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        log.warning("PCA SVD did not converge; clustering on raw weighted matrix")
        return mat.astype(np.float32)
    return (centred @ vt[:k].T).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# kNN graph
# ──────────────────────────────────────────────────────────────


def build_knn_graph(
    mat: np.ndarray,
    k: int = DEFAULT_KNN_K,
    *,
    connect_components: bool = True,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Symmetric cosine kNN graph as (edges, weights).

    Cosine similarity is a dot product once rows are L2-normalised, so the whole
    similarity matrix is one GEMM. At ~7.4k tracks that is a 7400² float32
    matrix (~220 MB) — acceptable, and far simpler than an ANN index. Negative
    similarities are clamped to 0 because Leiden expects non-negative weights.

    When *connect_components* is set the graph is additionally bridged into a
    single connected component. This matters more than it looks: Leiden can
    never merge vertices that have no path between them, so a fragmented kNN
    graph pins the cluster count to the component count and makes the
    resolution parameter inert — the multi-resolution hierarchy silently
    collapses to one granularity. Bridges use each component pair's most
    similar cross-edge, so they follow real structure rather than inventing it.
    """
    n = int(mat.shape[0])
    if n < 2:
        return [], []

    k = int(max(1, min(k, n - 1)))
    x = _l2_normalize_rows(mat)
    sim = (x @ x.T).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)

    # argpartition gives the k largest per row in O(n) per row.
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    edge_w: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            if j == i:
                continue
            s = float(sim[i, j])
            if not np.isfinite(s):
                continue
            key = (i, j) if i < j else (j, i)
            # Mutual neighbours keep the stronger of the two observations.
            prev = edge_w.get(key)
            if prev is None or s > prev:
                edge_w[key] = s

    if not edge_w:
        return [], []

    if connect_components:
        _bridge_components(edge_w, sim, n)

    items = sorted(edge_w.items())
    edges = [e for e, _ in items]
    weights = [max(0.0, w) for _, w in items]
    return edges, weights


def _bridge_components(
    edge_w: dict[tuple[int, int], float], sim: np.ndarray, n: int
) -> None:
    """Link disconnected components via their strongest cross-pair edge.

    Mutates *edge_w* in place. Repeatedly finds the two components joined by
    the highest similarity and adds that single edge, until one component
    remains — a maximum-similarity spanning step over the component graph.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for i, j in edge_w:
        union(i, j)

    comps: dict[int, list[int]] = {}
    for v in range(n):
        comps.setdefault(find(v), []).append(v)
    if len(comps) < 2:
        return

    groups = list(comps.values())
    # Guard against the pathological case of thousands of singleton components:
    # the pairwise search below is O(C²) in component count.
    if len(groups) > 512:
        log.debug("kNN graph has %d components; skipping bridge step", len(groups))
        return

    while len(groups) > 1:
        best: tuple[float, int, int, int, int] | None = None
        for a in range(len(groups)):
            rows = np.asarray(groups[a], dtype=np.int64)
            for b in range(a + 1, len(groups)):
                cols = np.asarray(groups[b], dtype=np.int64)
                sub = sim[np.ix_(rows, cols)]
                flat = int(np.argmax(sub))
                s = float(sub.flat[flat])
                if not np.isfinite(s):
                    continue
                if best is None or s > best[0]:
                    ri, ci = divmod(flat, sub.shape[1])
                    best = (s, a, b, int(rows[ri]), int(cols[ci]))
        if best is None:
            return
        s, a, b, i, j = best
        key = (i, j) if i < j else (j, i)
        edge_w.setdefault(key, max(0.0, s))
        groups[a] = groups[a] + groups[b]
        groups.pop(b)


# ──────────────────────────────────────────────────────────────
# Leiden
# ──────────────────────────────────────────────────────────────


def leiden_available() -> bool:
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        return False
    return True


def _cluster_one_resolution(
    graph: Any, weights: list[float], resolution: float, seed: int = 42
) -> tuple[np.ndarray, float | None]:
    import leidenalg

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=float(resolution),
        seed=seed,
    )
    labels = np.asarray(partition.membership, dtype=np.int32)
    try:
        modularity = float(partition.modularity)
    except Exception:
        modularity = None
    return labels, modularity


def _fallback_cluster(mat: np.ndarray, resolution: float) -> tuple[np.ndarray, None]:
    """Agglomerative fallback when leidenalg/igraph are unavailable.

    Resolution is mapped to a cluster count so the multi-resolution UI keeps
    working (coarser resolution → fewer clusters), at lower quality.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = int(mat.shape[0])
    n_clusters = int(np.clip(round(resolution * 12), 2, max(2, n // 4)))
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return model.fit_predict(_l2_normalize_rows(mat)).astype(np.int32), None


# ──────────────────────────────────────────────────────────────
# Cluster naming from MAEST style logits (plan §6/§16)
# ──────────────────────────────────────────────────────────────


def name_clusters(
    labels: np.ndarray,
    logits: np.ndarray | None,
    label_names: Sequence[str] | None,
    *,
    top_k: int = 3,
) -> tuple[dict[int, str], dict[int, list[tuple[str, float]]]]:
    """Name each cluster by the Discogs styles its members score highest on.

    The score is a *contrast*: a cluster's mean logit for a style minus the
    library-wide mean. Without that subtraction every cluster in an electronic
    library is simply named "Electronic---Techno"; with it, each cluster is
    named by what makes it distinct from the rest of the library.
    """
    names: dict[int, str] = {}
    evidence: dict[int, list[tuple[str, float]]] = {}
    if logits is None or label_names is None or logits.size == 0:
        return names, evidence

    lg = np.asarray(logits, dtype=np.float32)
    if lg.ndim != 2 or lg.shape[0] != labels.shape[0]:
        return names, evidence

    n_labels = min(len(label_names), lg.shape[1])
    global_mean = lg[:, :n_labels].mean(axis=0)

    for cid in sorted({int(c) for c in labels}):
        member_mask = labels == cid
        if not member_mask.any():
            continue
        contrast = lg[member_mask, :n_labels].mean(axis=0) - global_mean
        order = np.argsort(-contrast)[:top_k]
        ranked = [(str(label_names[int(i)]), float(contrast[int(i)])) for i in order]
        evidence[cid] = ranked
        # "Electronic---Drum n Bass" → "Drum n Bass"; keep the leaf style only.
        leaves = [nm.split("---")[-1] for nm, _ in ranked[: max(1, top_k - 1)]]
        names[cid] = " / ".join(dict.fromkeys(leaves))
    return names, evidence


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def compute_clustering_revision(
    paths: Sequence[str],
    weights: dict[str, float],
    sources: Sequence[str],
    resolutions: Sequence[float],
    pca_dim: int,
    knn_k: int,
    cache_versions: Sequence[int] = (),
) -> str:
    """Deterministic id for one clustering configuration + track set."""
    h = hashlib.sha256()
    h.update(f"cv{CLUSTERING_VERSION}".encode())
    for p in sorted(paths):
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\0")
    for s in sorted(sources):
        h.update(f"{s}={weights.get(s, 0.0):.6g};".encode())
    h.update(",".join(f"{r:.4g}" for r in resolutions).encode())
    h.update(f"|pca{pca_dim}|k{knn_k}|".encode())
    h.update(",".join(str(int(v)) for v in cache_versions).encode())
    return h.hexdigest()


def cluster_tracks(
    paths: list[str],
    source_blocks: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
    resolutions: Sequence[float] = DEFAULT_RESOLUTIONS,
    pca_dim: int = DEFAULT_PCA_DIM,
    knn_k: int = DEFAULT_KNN_K,
    style_logits: np.ndarray | None = None,
    style_label_names: Sequence[str] | None = None,
    cache_versions: Sequence[int] = (),
) -> ClusterResult | None:
    """Run the full pipeline: weight → PCA → cosine kNN → multi-resolution Leiden.

    Returns ``None`` when there is too little data to cluster meaningfully.
    """
    n = len(paths)
    if n < MIN_TRACKS_FOR_CLUSTERING:
        log.info("Clustering skipped: only %d track(s)", n)
        return None

    mat, used, effective = build_weighted_matrix(source_blocks, weights)
    if mat.size == 0:
        log.warning("Clustering skipped: no usable feature sources")
        return None
    if mat.shape[0] != n:
        log.error(
            "Clustering aborted: %d feature rows for %d paths", mat.shape[0], n
        )
        return None

    reduced = reduce_dimensions(mat, pca_dim)
    edges, edge_weights = build_knn_graph(reduced, knn_k)

    revision = compute_clustering_revision(
        paths, effective, used, resolutions, pca_dim, knn_k, cache_versions
    )

    graph = None
    if edges and leiden_available():
        try:
            import igraph as ig

            graph = ig.Graph(n=n, edges=edges)
            graph.es["weight"] = edge_weights
        except Exception as e:
            log.warning("igraph construction failed (%s); using fallback clustering", e)
            graph = None

    levels: list[ClusterLevel] = []
    for res in resolutions:
        try:
            if graph is not None:
                labels, modularity = _cluster_one_resolution(graph, edge_weights, float(res))
            else:
                labels, modularity = _fallback_cluster(reduced, float(res))
        except Exception as e:
            log.warning("Clustering failed at resolution %.3g: %s", res, e)
            continue

        uniq, counts = np.unique(labels, return_counts=True)
        names, evidence = name_clusters(labels, style_logits, style_label_names)
        levels.append(
            ClusterLevel(
                resolution=float(res),
                labels=labels,
                n_clusters=int(uniq.size),
                sizes={int(c): int(k) for c, k in zip(uniq, counts)},
                names=names,
                style_evidence=evidence,
                modularity=modularity,
            )
        )

    if not levels:
        return None

    log.info(
        "Clustered %d tracks over %d resolution(s): %s",
        n,
        len(levels),
        ", ".join(f"{lv.resolution:g}→{lv.n_clusters}" for lv in levels),
    )

    return ClusterResult(
        paths=list(paths),
        levels=levels,
        revision=revision,
        weights=effective,
        sources_used=used,
        n_tracks=n,
        pca_dim=int(reduced.shape[1]),
        knn_k=int(knn_k),
    )
