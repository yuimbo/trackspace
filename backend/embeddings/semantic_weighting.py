"""Semantic re-weighting of the embedding space before 2-D projection.

Folders are manual curation, so their tracks are the ground truth: each folder
node contributes a contrast direction from hierarchical centroid decomposition
(child centroid minus parent centroid). Tags are labels rather than track collections,
so they are projected into the CLAP text column span instead.

Extracted from ``layout`` — see audio_pipeline.md for the design rules.
"""

from __future__ import annotations

import logging

import numpy as np

from .clap import generate_text_embeddings
from .layout_revision import FOLDER_SEMANTIC_BASIS_VERSION
from .linalg_utils import svd_singular

log = logging.getLogger(__name__)

# Within-folder PCA on deviations orthogonal to the hierarchical contrast axis.
FOLDER_PCA_MIN_TRACKS = 8
FOLDER_PCA_MAX_COMPONENTS = 2
FOLDER_PCA_SECOND_SV_RATIO = 0.18

#: Folder-basis solver modes. ``full`` is the production default; the others are
#: benchmark-only seams and are never exposed by the HTTP API or recipe config.
FOLDER_BASIS_MODES = ("full", "centroids", "approx")


def validate_folder_basis_mode(mode: str) -> str:
    if mode not in FOLDER_BASIS_MODES:
        raise ValueError(
            f"unknown folder basis mode {mode!r} "
            f"(allowed: {', '.join(FOLDER_BASIS_MODES)})"
        )
    return mode


def _semantic_basis_param_key(
    context_tags: list[str] | None,
    folder_seeds: list[str] | None,
    folder_basis_mode: str = "full",
) -> str:
    tags = ",".join(sorted(context_tags)) if context_tags else ""
    if folder_seeds:
        fp = ",".join(sorted(folder_seeds))
    else:
        fp = "-"
    key = f"{tags}|{fp}|fsbv{FOLDER_SEMANTIC_BASIS_VERSION}"
    # Default full keeps the legacy key byte-identical; non-default modes are
    # namespaced so a cached full basis can never alias a centroid/approx one.
    if folder_basis_mode != "full":
        key += f"|fbm{folder_basis_mode}"
    return key


def _folder_seeds_from_track_paths(paths: list[str]) -> list[str]:
    s: set[str] = set()
    for p in paths:
        sep = p.rfind("/")
        tf = p[:sep] if sep >= 0 else ""
        if tf:
            s.add(tf)
    return sorted(s)


def _expand_folder_nodes(folder_seeds: list[str]) -> set[str]:
    all_nodes: set[str] = {""}
    for f in folder_seeds:
        parts = [p for p in f.split("/") if p]
        for depth in range(1, len(parts) + 1):
            all_nodes.add("/".join(parts[:depth]))
    return all_nodes


def _folder_orthogonal_pc_directions(
    mat_block: np.ndarray,
    d_unit: np.ndarray,
    *,
    max_components: int,
    min_second_sv_ratio: float,
    basis_mode: str = "full",
) -> list[np.ndarray]:
    validate_folder_basis_mode(basis_mode)
    n = mat_block.shape[0]
    if n < 4 or max_components < 1:
        return []

    c = mat_block.mean(axis=0)
    x_dev = mat_block - c
    proj = x_dev @ d_unit
    x_orth = x_dev - np.outer(proj, d_unit.astype(np.float64))

    if not np.isfinite(x_orth).all():
        log.debug("folder PC directions: non-finite block %s; skipping", mat_block.shape)
        return []

    x64 = x_orth.astype(np.float64)
    if basis_mode == "approx":
        from sklearn.utils.extmath import randomized_svd

        n_components = min(2, x64.shape[0], x64.shape[1])
        _, s, vt = randomized_svd(
            x64,
            n_components=n_components,
            n_iter=5,
            n_oversamples=10,
            random_state=42,
        )
    else:
        _, s, vt = svd_singular(x64)
    if s.size == 0 or float(s[0]) < 1e-12:
        return []

    d64 = d_unit.astype(np.float64)
    out: list[np.ndarray] = []
    lim = min(max_components, int(vt.shape[0]))
    for i in range(lim):
        if i > 0 and float(s[i]) < min_second_sv_ratio * float(s[0]):
            break
        v = np.asarray(vt[i], dtype=np.float64)
        v = v - float(v @ d64) * d64
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            continue
        out.append((v / nv).astype(np.float32))
    return out


def _build_folder_directions(
    mat: np.ndarray,
    paths: list[str],
    folder_seeds: list[str],
    min_tracks: int = 2,
    min_tracks_pca: int = FOLDER_PCA_MIN_TRACKS,
    max_pca_components: int = FOLDER_PCA_MAX_COMPONENTS,
    basis_mode: str = "full",
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, int]]]:
    validate_folder_basis_mode(basis_mode)
    if not folder_seeds:
        return None, None, []

    all_nodes = _expand_folder_nodes(folder_seeds)

    track_folders: list[str] = []
    for p in paths:
        sep = p.rfind("/")
        track_folders.append(p[:sep] if sep >= 0 else "")

    node_indices: dict[str, list[int]] = {n: [] for n in all_nodes}
    for i, tf in enumerate(track_folders):
        node_indices[""].append(i)
        parts = [p for p in tf.split("/") if p]
        for depth in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:depth])
            if ancestor in node_indices:
                node_indices[ancestor].append(i)

    centroids: dict[str, np.ndarray | None] = {}
    for n, idx_list in node_indices.items():
        centroids[n] = mat[idx_list].mean(axis=0) if idx_list else None

    dirs: list[np.ndarray] = []
    exps: list[int] = []
    fallbacks: list[tuple[str, int]] = []

    for node in sorted(all_nodes - {""}):
        parts = node.split("/")
        parent = "/".join(parts[:-1])
        depth = len(parts)
        exp = depth - 1

        c = centroids.get(node)
        pc = centroids.get(parent)
        if c is None or pc is None:
            continue

        if len(node_indices[node]) < min_tracks:
            fallbacks.append((parts[-1], exp))
            continue

        d = c - pc
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue

        d_unit = (d / norm).astype(np.float32)
        dirs.append(d_unit)
        exps.append(exp)

        idx_list = node_indices[node]
        if (
            basis_mode != "centroids"
            and len(idx_list) >= min_tracks_pca
            and max_pca_components > 0
        ):
            block = mat[idx_list]
            extras = _folder_orthogonal_pc_directions(
                block,
                d_unit,
                max_components=max_pca_components,
                min_second_sv_ratio=FOLDER_PCA_SECOND_SV_RATIO,
                basis_mode=basis_mode,
            )
            for v_pc in extras:
                dirs.append(v_pc)
                exps.append(exp)

    if dirs:
        log.debug(
            "folder directions: %d data-driven rows, %d text-fallback",
            len(dirs), len(fallbacks),
        )
        return (
            np.stack(dirs),
            np.array(exps, dtype=np.int32),
            fallbacks,
        )
    return None, None, fallbacks


def _unit_clap_text_dirs_to_full(
    unit_rows: np.ndarray,
    d_total: int,
    clap_lo: int,
    clap_hi: int,
) -> np.ndarray:
    if clap_lo >= clap_hi:
        raise ValueError("clap column span must be non-empty")
    slot_w = clap_hi - clap_lo
    k, d_in = unit_rows.shape
    out = np.zeros((k, d_total), dtype=np.float32)
    if d_in <= slot_w:
        out[:, clap_lo : clap_lo + d_in] = unit_rows
    else:
        out[:, clap_lo:clap_hi] = unit_rows[:, :slot_w]
    return out


def _apply_low_rank_semantic_weight(
    mat: np.ndarray,
    T_n: np.ndarray,
    exp: np.ndarray,
    boost: float,
    depth_boost: float,
) -> np.ndarray:
    if T_n.shape[0] == 0:
        return mat
    gamma = np.power(
        float(depth_boost), exp.astype(np.float64),
    ).astype(np.float32)
    w = (gamma * float(boost)).astype(np.float32)
    g = mat @ T_n.T
    return (mat + (g * w) @ T_n).astype(np.float32)


def _build_semantic_basis(
    mat: np.ndarray,
    paths: list[str],
    context_tags: list[str] | None,
    folder_seeds: list[str] | None,
    clap_lo: int,
    clap_hi: int,
    folder_basis_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    validate_folder_basis_mode(folder_basis_mode)
    d_total = mat.shape[1]
    have_clap_slot = clap_lo < clap_hi
    all_dirs: list[np.ndarray] = []
    all_exp: list[np.ndarray] = []

    if folder_seeds:
        fd, f_exp, fallbacks = _build_folder_directions(
            mat, paths, folder_seeds, basis_mode=folder_basis_mode,
        )
        if fd is not None and f_exp is not None:
            all_dirs.append(fd)
            all_exp.append(f_exp)

        if fallbacks and have_clap_slot:
            texts = [t for t, _ in fallbacks]
            fb_exp = np.array([e for _, e in fallbacks], dtype=np.int32)
            text_vecs = generate_text_embeddings(texts)
            if text_vecs is not None and text_vecs.shape[0] > 0:
                norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                full = _unit_clap_text_dirs_to_full(
                    text_vecs / norms, d_total, clap_lo, clap_hi,
                )
                all_dirs.append(full)
                all_exp.append(fb_exp)

    if context_tags and have_clap_slot:
        tag_vecs = generate_text_embeddings(context_tags)
        if tag_vecs is not None and tag_vecs.shape[0] > 0:
            norms = np.linalg.norm(tag_vecs, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            full = _unit_clap_text_dirs_to_full(
                tag_vecs / norms, d_total, clap_lo, clap_hi,
            )
            all_dirs.append(full)
            all_exp.append(np.zeros(len(context_tags), dtype=np.int32))

    if not all_dirs:
        return (
            np.empty((0, d_total), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    T_n = np.concatenate(all_dirs, axis=0).astype(np.float32)
    exp = np.concatenate(all_exp)
    return T_n, exp
