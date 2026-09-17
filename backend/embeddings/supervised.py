"""Weak supervision: a folder-labelled projection head over cached vectors.

Folders are the user's manual curation, so they are noisy but real labels. This
module trains a small MLP that maps the composite of cached source blocks into a
128-D space where same-folder tracks sit together, then exposes that space as an
ordinary source block (``supervised``) so the 2-D map and the clustering can use it
like any other signal. Nothing here re-runs an encoder — training reads only the
cache.

Design points that matter:

**Stateless input recipe.** The head's input is the per-source blocks L2-normalised
*per row* and scaled by ``sqrt(weight)`` (the same algebra as the clustering
pipeline), with **no column standardisation**. Column statistics depend on the current
track set, so a stateful recipe would make a single new track's projection drift as
the library grows; the stateless form projects one track identically whether it is alone
or among 9.5k others.

**CPU-only training.** At library scale the head is a few seconds of CPU matmuls, which
deliberately keeps it off MPS/Metal: the GPU paths are serialised by
``clap.inference_lock`` (see ``audio_pipeline.md``), and a training run holding that
lock would stall all embedding generation for no benefit.

**Excerpts refine the supervision.** The per-excerpt MAEST vectors are used for (a)
mining the hardest cross-folder negative pairs — the genre-straddling neighbours that
a track-level loss would happily ignore — and (b) down-weighting tracks whose excerpts
disagree, which are noisier folder labels.

**The run id is part of cache identity.** A retrain rewrites every ``supervised`` row at
the same ``SUPERVISED_VERSION``, so the *run id* — not the version constant — is what
layout and cluster revisions must include to stop serving stale coordinates.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .maest import MAEST_EMBED_DIM
from .projection_config import FROZEN_SOURCE_WEIGHTS, FROZEN_SOURCES

log = logging.getLogger(__name__)

#: Bump when the training recipe (loss, architecture, input assembly) changes.
SUPERVISED_VERSION = 1

#: Output width of the learned block.
SUPERVISED_DIM = 128

#: Folders with fewer tracks cannot provide a reliable class.
MIN_FOLDER_LABEL_SIZE = 8

DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUT_DIM = SUPERVISED_DIM
DEFAULT_EPOCHS = 40
DEFAULT_PATIENCE = 8
DEFAULT_LR = 1e-3
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SEED = 42
#: Class-balanced batch sampling: classes per batch × samples per class.
DEFAULT_BATCH_CLASSES = 32
DEFAULT_SAMPLES_PER_CLASS = 8
VAL_FRACTION = 0.15
#: Hard-negative mining depth and the margin the repulsion term enforces.
DEFAULT_HARD_NEG_K = 3
HARD_NEG_MARGIN = 0.3
HARD_NEG_WEIGHT = 0.5
#: Excerpt dispersion → label-confidence decay. Deliberately gentle: normal
#: within-track variation must not erase the supervision signal, only clearly
#: genre-straddling tracks (dispersion ≳ 0.3) are strongly down-weighted.
DISPERSION_ALPHA = 2.0

#: Excerpt dispersion at/above which a track is reported as *genre-spanning* in the
#: UI. Same measure as the confidence decay above (1 − mean pairwise cosine), but a
#: separate knob: the threshold is a display decision, not a loss weight.
SPANNING_DISPERSION_THRESHOLD = 0.25

#: Per-class match floor for folder prediction, as a percentile of the training
#: rows' cosine to their own centroid. A fan-out only places a track whose winning
#: similarity clears its class's floor, so a track that resembles none of the learned
#: folders is left alone rather than forced into the nearest one.
DEFAULT_MATCH_PERCENTILE = 5.0

#: Sentinel match floor for a class with no training rows: cosine lives in
#: ``[-1, 1]``, so ``-2`` can never gate a track (i.e. "always match").
MATCH_ALWAYS = -2.0

HEAD_FILENAME = "supervised_head.pt"
META_FILENAME = "supervised_head.json"

#: Portable head-bundle format written by export and read by import. A bundle is
#: self-describing so importing a head trained on another library can verify it
#: before overwriting anything (see :func:`read_head_bundle`).
HEAD_BUNDLE_FORMAT = "trackspace-head"
HEAD_BUNDLE_VERSION = 1

#: FeatureCache meta key holding the current training-run id. Readers that only
#: hold the cache (layout, clustering) resolve the supervised cache version from it.
RUN_ID_META_KEY = "supervised_run_id"


def cache_version(feature_cache) -> int:
    """Effective cache version for supervised rows: the training-run id.

    ``0`` means no head has been trained, which makes the source absent. Kept
    here (not in ``feature_cache``) so the key name has one owner.
    """
    try:
        raw = feature_cache.get_meta(RUN_ID_META_KEY)
    except Exception:
        return 0
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


# ──────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────


def folder_label_for_path(path: str) -> str:
    """Leaf folder of a virtual path (``Root/Techno/x.mp3`` → ``Root/Techno``)."""
    norm = str(path).replace("\\", "/").rstrip("/")
    if "/" not in norm:
        return ""
    return norm.rsplit("/", 1)[0]


def derive_folder_labels(
    paths: list[str], *, min_size: int = MIN_FOLDER_LABEL_SIZE
) -> tuple[np.ndarray, list[str]]:
    """Map each path to a folder class index; ``-1`` for unlabelled tracks.

    Classes are the full folder paths (root included) with at least *min_size*
    tracks — smaller folders are noise. Names are sorted so the mapping is stable
    across runs, which matters because the class list is recorded in the head meta.
    """
    folders = [folder_label_for_path(p) for p in paths]
    counts = Counter(f for f in folders if f)
    classes = sorted(c for c, n in counts.items() if n >= int(min_size))
    index = {c: i for i, c in enumerate(classes)}
    labels = np.array([index.get(f, -1) for f in folders], dtype=np.int64)
    return labels, classes


# ──────────────────────────────────────────────────────────────
# Input assembly
# ──────────────────────────────────────────────────────────────


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (mat / norms).astype(np.float32)


def build_input_matrix(
    source_blocks: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack per-source blocks into the head's input, in canonical source order.

    Each block is L2-normalised *per row* and scaled by ``sqrt(weight)`` — the same
    weighted-distance algebra as clustering, minus the dataset-dependent column
    standardisation (see the module docstring). Returns ``(matrix, used_sources)``;
    callers record ``used_sources`` so projection reproduces the exact same input.
    """
    w = dict(FROZEN_SOURCE_WEIGHTS if weights is None else weights)
    used: list[str] = []
    blocks: list[np.ndarray] = []
    for name in FROZEN_SOURCES:
        weight = float(w.get(name, 0.0))
        block = source_blocks.get(name)
        if weight <= 0.0 or block is None:
            continue
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        scaled = _l2_normalize_rows(arr) * float(np.sqrt(weight))
        blocks.append(scaled.astype(np.float32))
        used.append(name)
    if not blocks:
        return np.empty((0, 0), dtype=np.float32), []
    return np.concatenate(blocks, axis=1).astype(np.float32), used


# ──────────────────────────────────────────────────────────────
# Excerpt mining
# ──────────────────────────────────────────────────────────────


def decode_excerpts(blob: np.ndarray) -> np.ndarray:
    """Reshape a flat cached excerpt blob to ``(n_excerpts, MAEST_EMBED_DIM)``."""
    flat = np.asarray(blob, dtype=np.float32).reshape(-1)
    if flat.size == 0 or flat.size % MAEST_EMBED_DIM:
        return np.empty((0, MAEST_EMBED_DIM), dtype=np.float32)
    return flat.reshape(-1, MAEST_EMBED_DIM)


def excerpt_confidence(
    excerpts_per_track: Sequence[np.ndarray | None], *, alpha: float = DISPERSION_ALPHA
) -> np.ndarray:
    """Per-track label confidence from intra-track excerpt dispersion.

    A track whose excerpts point in wildly different directions (intro vs drop) is a
    poor single-label example, so its loss contribution is decayed. Tracks with a
    single excerpt carry no dispersion signal and stay at full confidence.
    """
    out = np.ones(len(excerpts_per_track), dtype=np.float32)
    for i, exc in enumerate(excerpts_per_track):
        dispersion = excerpt_dispersion(exc)
        if dispersion is None:
            continue
        out[i] = float(np.exp(-float(alpha) * dispersion))
    return out


def excerpt_dispersion(excerpts: np.ndarray | None) -> float | None:
    """Intra-track excerpt dispersion (``1 − mean pairwise cosine``), or ``None``.

    ``None`` when there are fewer than two usable excerpts — a single excerpt
    carries no dispersion signal, so callers must not read "0.0" as agreement.
    """
    if excerpts is None:
        return None
    arr = np.asarray(excerpts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    norm = _l2_normalize_rows(arr)
    sim = norm @ norm.T
    m = int(sim.shape[0])
    iu = np.triu_indices(m, k=1)
    mean_cos = float(np.clip(sim[iu].mean(), -1.0, 1.0))
    return max(0.0, 1.0 - mean_cos)


def is_genre_spanning(
    excerpts: np.ndarray | None,
    *,
    threshold: float = SPANNING_DISPERSION_THRESHOLD,
) -> bool:
    """True when a track's excerpts disagree enough to look intro-vs-drop.

    The user-facing counterpart of the confidence decay in
    :func:`excerpt_confidence`: the same dispersion measure, reported as a flag.
    Tracks with fewer than two excerpts are never flagged.
    """
    dispersion = excerpt_dispersion(excerpts)
    return dispersion is not None and dispersion >= threshold


def mine_hard_negatives(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = DEFAULT_HARD_NEG_K,
    chunk: int = 1024,
) -> np.ndarray:
    """Hardest cross-label neighbours as ``(m, 2)`` index pairs.

    *vectors* are per-track pooled vectors (one row per labelled track). For each
    track the top-*k* most cosine-similar tracks of a *different* class are
    returned; those are exactly the pairs a plain contrastive denominator treats as
    already-resolved and therefore under-penalises. Chunked so the similarity
    matrix never materialises at library scale.
    """
    x = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(x.shape[0])
    if n < 2 or x.ndim != 2 or x.shape[0] != y.shape[0]:
        return np.empty((0, 2), dtype=np.int64)

    k = int(max(1, min(k, n - 1)))
    xn = _l2_normalize_rows(x)
    pairs: set[tuple[int, int]] = set()
    step = int(max(1, chunk))
    for start in range(0, n, step):
        stop = min(n, start + step)
        sim = xn[start:stop] @ xn.T
        # Same-label entries (self included) are never candidates.
        sim[y[start:stop, None] == y[None, :]] = -np.inf
        kk = min(k, int(sim.shape[1] - 1))
        if kk <= 0:
            continue
        idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        for r in range(idx.shape[0]):
            i = start + r
            for j in idx[r]:
                j = int(j)
                if not np.isfinite(sim[r, j]):
                    continue
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def _hard_adjacency(hard_pairs: np.ndarray | None) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    if hard_pairs is None or int(np.asarray(hard_pairs).shape[0]) == 0:
        return adj
    for a, b in np.asarray(hard_pairs, dtype=np.int64):
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    return adj


def _batch_hard_pairs(adj: dict[int, set[int]], rows: np.ndarray):
    """Batch-local hard pairs as a torch ``(p, 2)`` index tensor, or ``None``.

    Filtering through the adjacency sets is O(sum of degrees in the batch), not
    O(total pairs) per batch.
    """
    if not adj:
        return None
    row_set = {int(r) for r in rows.tolist()}
    position = {int(r): i for i, r in enumerate(rows.tolist())}
    local: list[tuple[int, int]] = []
    for r in row_set:
        for partner in adj.get(r, ()):  # type: ignore[union-attr]
            if partner in row_set and r < partner:
                local.append((position[r], position[partner]))
    if not local:
        return None
    import torch

    return torch.tensor(local, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────


def _build_model(in_dim: int, hidden_dim: int, out_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _supcon_loss(
    z,
    labels,
    sample_weights=None,
    hard_index_pairs=None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    hard_weight: float = HARD_NEG_WEIGHT,
):
    """Supervised contrastive loss plus a hard-negative repulsion term.

    ``z`` is L2-normalised; positives are all same-label rows in the batch. The
    repulsion term pushes mined cross-folder pairs below ``HARD_NEG_MARGIN`` so the
    hardest confusable neighbours are explicitly separated.
    """
    import torch

    n = int(z.shape[0])
    if n < 2:
        return z.sum() * 0.0

    logits = (z @ z.T) / float(temperature)
    # Stability: subtract the row max before the softmax normaliser.
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_mask = (labels[:, None] == labels[None, :]) & ~torch.eye(
        n, dtype=torch.bool, device=z.device
    )
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    per_anchor = -(pos_mask * log_prob).sum(dim=1) / pos_count

    if sample_weights is not None:
        loss = (per_anchor * sample_weights).sum() / sample_weights.sum().clamp(
            min=1e-6
        )
    else:
        loss = per_anchor.mean()

    if hard_index_pairs is not None and int(hard_index_pairs.shape[0]) > 0:
        hard_index_pairs = hard_index_pairs.to(z.device)
        cos = (z[hard_index_pairs[:, 0]] * z[hard_index_pairs[:, 1]]).sum(dim=1)
        hard = torch.clamp(cos - HARD_NEG_MARGIN, min=0.0).mean()
        loss = loss + float(hard_weight) * hard

    return loss


def _class_balanced_batches(labels, rng, batch_classes, samples_per_class):
    classes = np.unique(labels)
    by_class = {int(c): np.where(labels == c)[0] for c in classes}
    order = rng.permutation(classes)
    for start in range(0, len(order), int(batch_classes)):
        group = order[start : start + int(batch_classes)]
        picked: list[int] = []
        for c in group:
            members = by_class[int(c)]
            replace = len(members) < int(samples_per_class)
            picked.extend(
                rng.choice(members, size=int(samples_per_class), replace=replace).tolist()
            )
        if picked:
            yield np.array(picked, dtype=np.int64)


def _split_per_class(y: np.ndarray, rng, val_fraction: float):
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = int(round(float(val_fraction) * idx.size))
        # Keep at least one training row per class so SupCon always has an anchor.
        n_val = min(n_val, max(0, idx.size - 1))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    return train.astype(np.int64), val.astype(np.int64)


def train_head(
    mat: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    hard_pairs: np.ndarray | None = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    out_dim: int = DEFAULT_OUT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    temperature: float = DEFAULT_TEMPERATURE,
    batch_classes: int = DEFAULT_BATCH_CLASSES,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    val_fraction: float = VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    on_epoch: Callable[[int, int], None] | None = None,
) -> dict:
    """Train the head; returns ``{"state_dict", "in_dim", "hidden_dim", "out_dim", "stats"}``.

    Only labelled rows (``labels >= 0``) are used. The validation split is taken
    per class so every class keeps at least one training example; early stopping on
    validation loss returns the best weights, not the last ones.

    *on_epoch* is called as ``(epoch_number, max_epochs)`` after each epoch
    (1-based), for a progress UI.
    """
    import torch

    x = np.asarray(mat, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("input matrix and labels are misaligned")

    labelled = np.where(y >= 0)[0]
    if labelled.size < 2 or np.unique(y[labelled]).size < 2:
        raise ValueError("not enough labelled tracks in at least two folders")
    x = x[labelled]
    y = y[labelled]
    if sample_weights is not None:
        w_all = np.asarray(sample_weights, dtype=np.float32).reshape(-1)[labelled]
    else:
        w_all = np.ones(labelled.size, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    train_idx, val_idx = _split_per_class(y, rng, val_fraction)
    if train_idx.size == 0:
        raise ValueError("no training rows after the validation split")

    # Remap mined pair indices (labelled-row space) and keep them as an adjacency.
    hard_adj: dict[int, set[int]] = {}
    if hard_pairs is not None and int(np.asarray(hard_pairs).shape[0]) > 0:
        global_to_local = {int(g): i for i, g in enumerate(labelled.tolist())}
        remapped: list[tuple[int, int]] = []
        for a, b in np.asarray(hard_pairs, dtype=np.int64):
            la = global_to_local.get(int(a))
            lb = global_to_local.get(int(b))
            if la is not None and lb is not None:
                remapped.append((la, lb))
        hard_adj = _hard_adjacency(
            np.array(remapped, dtype=np.int64) if remapped else None
        )

    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    model = _build_model(int(x.shape[1]), int(hidden_dim), int(out_dim)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(lr))

    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    wt = torch.from_numpy(w_all).to(device)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(int(max(1, epochs))):
        model.train()
        batch_losses: list[float] = []
        for batch in _class_balanced_batches(
            y[train_idx], rng, batch_classes, samples_per_class
        ):
            rows = train_idx[batch]
            z = torch.nn.functional.normalize(model(xt[rows]), dim=1)
            loss = _supcon_loss(
                z,
                yt[rows],
                wt[rows],
                _batch_hard_pairs(hard_adj, rows),
                temperature=temperature,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.detach()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        model.eval()
        with torch.inference_mode():
            z_val = torch.nn.functional.normalize(model(xt[val_idx]), dim=1)
            val_loss = float(
                _supcon_loss(
                    z_val,
                    yt[val_idx],
                    wt[val_idx],
                    _batch_hard_pairs(hard_adj, val_idx),
                    temperature=temperature,
                )
            )
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if on_epoch is not None:
            try:
                on_epoch(epoch + 1, int(max(1, epochs)))
            except Exception:
                log.debug("supervised epoch callback failed", exc_info=True)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break

    return {
        "state_dict": best_state,
        "in_dim": int(x.shape[1]),
        "hidden_dim": int(hidden_dim),
        "out_dim": int(out_dim),
        "stats": {
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_classes": int(np.unique(y).size),
            "n_labelled": int(labelled.size),
            "epochs_run": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "hard_pairs": int(sum(len(v) for v in hard_adj.values()) // 2),
            "history": history,
        },
    }


# ──────────────────────────────────────────────────────────────
# Persistence + projection
# ──────────────────────────────────────────────────────────────


def head_path(data_dir: str) -> str:
    return os.path.join(data_dir, HEAD_FILENAME)


def meta_path(data_dir: str) -> str:
    return os.path.join(data_dir, META_FILENAME)


def save_head(
    data_dir: str,
    state_dict: dict,
    meta: dict,
    centroids: np.ndarray | None = None,
) -> None:
    """Persist weights + a JSON sidecar; the sidecar alone carries the run id.

    *centroids*, when given, is the ``(n_classes, out_dim)`` per-folder centroid
    matrix aligned with ``meta["class_names"]`` — the data nearest-centroid folder
    assignment and the export bundle need.
    """
    import torch

    os.makedirs(data_dir, exist_ok=True)
    payload: dict = {
        "state_dict": state_dict,
        "in_dim": int(meta["in_dim"]),
        "hidden_dim": int(meta["hidden_dim"]),
        "out_dim": int(meta["out_dim"]),
    }
    if centroids is not None:
        payload["centroids"] = np.asarray(centroids, dtype=np.float32)
    torch.save(payload, head_path(data_dir))
    tmp = meta_path(data_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, meta_path(data_dir))


def read_meta(data_dir: str) -> dict | None:
    try:
        with open(meta_path(data_dir), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def current_run_id(data_dir: str) -> int:
    """Training-run counter; ``0`` when no head has been trained yet."""
    meta = read_meta(data_dir)
    try:
        return int(meta.get("run_id", 0)) if meta else 0
    except (TypeError, ValueError):
        return 0


def load_head(data_dir: str) -> dict | None:
    """Load weights + meta, or ``None`` when absent/unreadable."""
    meta = read_meta(data_dir)
    path = head_path(data_dir)
    if meta is None or not os.path.isfile(path):
        return None
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.warning("supervised: failed to load head %s: %s", path, e)
        return None
    return {"payload": payload, "meta": meta}


def head_centroids(data_dir: str) -> np.ndarray | None:
    """Per-folder centroids of the installed head, or ``None`` when absent."""
    head = load_head(data_dir)
    if head is None:
        return None
    raw = head["payload"].get("centroids")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    return arr if arr.ndim == 2 and arr.shape[0] > 0 else None


def project_from_state(
    state_dict: dict,
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    mat: np.ndarray,
) -> np.ndarray:
    """Apply head weights to an input matrix, returning L2-normalised rows."""
    import torch

    model = _build_model(int(in_dim), int(hidden_dim), int(out_dim))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(mat, dtype=np.float32))
        z = torch.nn.functional.normalize(model(x), dim=1)
    return z.cpu().numpy().astype(np.float32)


def project(head: dict, mat: np.ndarray) -> np.ndarray:
    """Apply a loaded head (from :func:`load_head`) to an input matrix."""
    payload = head["payload"]
    return project_from_state(
        payload["state_dict"],
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        out_dim=int(payload["out_dim"]),
        mat=mat,
    )


def new_run_id(data_dir: str) -> int:
    return current_run_id(data_dir) + 1


def build_meta(
    *,
    run_id: int,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    input_sources: list[str],
    input_weights: dict[str, float],
    class_names: list[str],
    source_versions: dict[str, int],
    stats: dict,
    train_folders: list[str] | None = None,
    class_match_thresholds: Sequence[float] | None = None,
    match_percentile: float = DEFAULT_MATCH_PERCENTILE,
) -> dict:
    """Sidecar payload; ``trained_at`` is informational, the rest is auditable.

    *train_folders* records which virtual folders the labels were drawn from
    (empty = the whole library), so a head can report what it was trained on.
    *class_match_thresholds* is the per-class cosine floor (aligned with
    ``class_names``) a fan-out uses to decide whether a track is close enough to
    a learned folder to be placed in it.
    """
    return {
        "run_id": int(run_id),
        "trained_at": time.time(),
        "supervised_version": int(SUPERVISED_VERSION),
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "out_dim": int(out_dim),
        "input_sources": list(input_sources),
        "input_weights": {k: float(v) for k, v in input_weights.items()},
        "n_classes": len(class_names),
        "class_names": list(class_names),
        "train_folders": [str(f) for f in (train_folders or [])],
        "source_versions": {k: int(v) for k, v in source_versions.items()},
        "class_match_thresholds": [
            float(v) for v in (class_match_thresholds or [])
        ],
        "match_percentile": float(match_percentile),
        "stats": stats,
    }


# ──────────────────────────────────────────────────────────────
# Centroids, folder prediction, and the portable head bundle
# ──────────────────────────────────────────────────────────────


def compute_centroids(
    projected: np.ndarray, labels: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class means of *projected* rows, L2-normalised, with their counts.

    Aligned with the ``class_names`` order, so index ``c`` is that folder. A
    class with no labelled row keeps a zero row (and count 0), which
    :func:`predict_folders` can never then select by accident.
    """
    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(max(0, n_classes))
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        return np.zeros((n, 0), dtype=np.float32), np.zeros(n, dtype=np.int64)
    centroids = np.zeros((n, int(x.shape[1])), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    for c in range(n):
        rows = x[y == c]
        if rows.shape[0] == 0:
            continue
        centroids[c] = rows.mean(axis=0)
        counts[c] = rows.shape[0]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (centroids / norms).astype(np.float32), counts


def predict_folders(
    centroids: np.ndarray, mat: np.ndarray, *, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid class index and cosine per row of *mat*.

    ``-1`` marks a row that cannot be assigned — empty centroids or a dimension
    mismatch — so callers never read a bogus class. Chunked so the similarity
    matrix never materialises at library scale.
    """
    c = np.asarray(centroids, dtype=np.float32)
    m = np.asarray(mat, dtype=np.float32)
    if (
        c.ndim != 2
        or m.ndim != 2
        or c.shape[0] == 0
        or c.shape[1] != m.shape[1]
    ):
        return (
            np.full(m.shape[0] if m.ndim == 2 else 0, -1, dtype=np.int64),
            np.zeros(m.shape[0] if m.ndim == 2 else 0, dtype=np.float32),
        )
    mn = _l2_normalize_rows(m)
    idx = np.empty(m.shape[0], dtype=np.int64)
    sim = np.empty(m.shape[0], dtype=np.float32)
    step = int(max(1, chunk))
    for start in range(0, m.shape[0], step):
        stop = min(m.shape[0], start + step)
        s = mn[start:stop] @ c.T
        best = np.argmax(s, axis=1)
        idx[start:stop] = best
        sim[start:stop] = s[np.arange(stop - start), best]
    return idx, sim


def compute_match_thresholds(
    projected: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_classes: int,
    *,
    percentile: float = DEFAULT_MATCH_PERCENTILE,
) -> np.ndarray:
    """Per-class cosine floor from the training rows' similarity to their centroid.

    For each class the floor is the *percentile* of its training rows' cosine to
    that class's own centroid, so it is calibrated to how tightly the head actually
    packs that folder. A diffuse class gets a lower floor than a tight one, and a
    genuinely novel track — similar to no learned folder — falls below every class's
    floor and is left unplaced. Classes with no training row get
    :data:`MATCH_ALWAYS`, which can never gate a track.
    """
    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    c = np.asarray(centroids, dtype=np.float32)
    n = int(max(0, n_classes))
    out = np.full(n, MATCH_ALWAYS, dtype=np.float32)
    if (
        x.ndim != 2
        or c.ndim != 2
        or x.shape[0] != y.shape[0]
        or c.shape[0] != n
    ):
        return out
    xn = _l2_normalize_rows(x)
    for cls in range(n):
        rows = xn[y == cls]
        if rows.shape[0] == 0:
            continue
        sims = rows @ c[cls]
        out[cls] = float(np.percentile(sims, float(percentile)))
    return out


def matches_thresholds(
    idx: np.ndarray, sim: np.ndarray, thresholds: np.ndarray | None
) -> np.ndarray:
    """Per-row match decision: winning similarity clears its class's floor.

    ``thresholds`` is aligned with the class list; a missing or short list (an old
    head trained before the gate existed) matches everything, preserving the previous
    fan-out behaviour. A row with no valid class never matches.
    """
    i = np.asarray(idx, dtype=np.int64).reshape(-1)
    s = np.asarray(sim, dtype=np.float32).reshape(-1)
    out = np.ones(i.shape[0], dtype=bool)
    if thresholds is None:
        return out
    t = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if t.size == 0:
        return out
    valid = (i >= 0) & (i < t.size)
    out[valid] = s[valid] >= t[i[valid]]
    out[~valid] = False
    return out


def folder_subpath(class_name: str) -> str:
    """Folder a class maps to *below a fan-out target folder*.

    Class names are virtual paths that include the training library's root id
    (``3f2a/Techno/Deep``); that root is meaningless in the target folder, so
    only the part below it becomes the subfolder. A class that *is* a root maps to
    the target root itself and returns ``""``.
    """
    norm = str(class_name).replace("\\", "/").strip("/")
    if "/" not in norm:
        return ""
    return norm.split("/", 1)[1]


def head_bundle(data_dir: str) -> dict | None:
    """Self-describing, portable bundle for the installed head, or ``None``."""
    head = load_head(data_dir)
    if head is None:
        return None
    payload = head["payload"]
    centroids = payload.get("centroids")
    return {
        "format": HEAD_BUNDLE_FORMAT,
        "format_version": HEAD_BUNDLE_VERSION,
        "exported_at": time.time(),
        "state_dict": payload["state_dict"],
        "in_dim": int(payload["in_dim"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "out_dim": int(payload["out_dim"]),
        "meta": head["meta"],
        "centroids": (
            None if centroids is None else np.asarray(centroids, dtype=np.float32)
        ),
    }


def write_head_bundle(path: str, bundle: dict) -> None:
    import torch

    tmp = path + ".tmp"
    torch.save(bundle, tmp)
    os.replace(tmp, path)


def read_head_bundle(path: str) -> dict:
    """Load and validate a portable bundle; raises ``ValueError`` when malformed."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        # A bad upload is a client error, not a server fault: normalise every
        # deserialisation failure to ValueError so routes can answer 400.
        raise ValueError(f"could not read head bundle: {e}") from e
    if not isinstance(payload, dict) or payload.get("format") != HEAD_BUNDLE_FORMAT:
        raise ValueError("not a trackspace head bundle")
    try:
        version = int(payload.get("format_version", 0))
    except (TypeError, ValueError):
        version = 0
    if version > HEAD_BUNDLE_VERSION:
        raise ValueError(
            f"head bundle version {version} is newer than this server supports "
            f"({HEAD_BUNDLE_VERSION})"
        )
    for key in ("state_dict", "meta", "in_dim", "hidden_dim", "out_dim"):
        if key not in payload:
            raise ValueError(f"head bundle is missing {key!r}")
    return payload


def bundle_version_mismatches(
    bundle: dict, current_versions: dict[str, int]
) -> dict[str, dict[str, int]]:
    """Sources whose recorded training version differs from *current_versions*.

    A head trained before an encoder checkpoint bump produces rows in the wrong
    space, so import must be able to refuse it. Only sources present in both the
    bundle and the current set are compared; an unknown source is not a mismatch.
    """
    recorded = dict(bundle.get("meta", {}).get("source_versions", {}))
    out: dict[str, dict[str, int]] = {}
    for source, trained in recorded.items():
        current = current_versions.get(source)
        if current is None:
            continue
        try:
            if int(current) != int(trained):
                out[str(source)] = {
                    "trained": int(trained),
                    "current": int(current),
                }
        except (TypeError, ValueError):
            continue
    return out


class HeadVersionMismatch(ValueError):
    """An imported head was trained on different source versions.

    Carries the per-source ``{trained, current}`` map so a caller can offer an
    explicit override (``allow_mismatch``) rather than silently installing a head
    that would produce poor rows.
    """

    def __init__(self, mismatches: dict[str, dict[str, int]]) -> None:
        self.mismatches = mismatches
        detail = ", ".join(
            f"{s} (trained {v['trained']}, now {v['current']})"
            for s, v in sorted(mismatches.items())
        )
        super().__init__(
            f"head was trained on different encoder versions: {detail}"
        )


def install_head_bundle(
    data_dir: str, bundle: dict, *, run_id: int | None = None
) -> dict:
    """Install an imported bundle under a *fresh local* run id.

    The run id is the cache version for every ``supervised`` row, so it must be
    local: reusing the exporting library's id would make the imported head collide
    with rows already cached here and stop them being re-projected.
    """
    if bundle.get("format") != HEAD_BUNDLE_FORMAT:
        raise ValueError("not a trackspace head bundle")
    meta = dict(bundle["meta"])
    new_id = int(run_id) if run_id is not None else new_run_id(data_dir)
    meta["run_id"] = new_id
    meta["imported_at"] = time.time()
    save_head(
        data_dir,
        bundle["state_dict"],
        meta,
        centroids=bundle.get("centroids"),
    )
    return meta


def compare_heads(a: dict, b: dict) -> dict:
    """Compare two head bundles: shared folders and per-class centroid cosine.

    Per-class similarity is only reported when the two heads share an input space
    (same sources and width). A head trained on a different encoder set or a
    different checkpoint lives in an unrelated space, so a centroid cosine between
    them would be a meaningless number; ``compatible`` says so instead.
    """
    ma, mb = a.get("meta", {}), b.get("meta", {})
    ca, cb = a.get("centroids"), b.get("centroids")
    compatible = (
        int(a.get("in_dim", 0)) == int(b.get("in_dim", 0))
        and list(ma.get("input_sources", [])) == list(mb.get("input_sources", []))
        and dict(ma.get("source_versions", {}))
        == dict(mb.get("source_versions", {}))
    )
    names_a = list(ma.get("class_names", []))
    names_b = list(mb.get("class_names", []))
    index_a = {n: i for i, n in enumerate(names_a)}
    index_b = {n: i for i, n in enumerate(names_b)}
    shared = sorted(set(names_a) & set(names_b))

    per_class: list[dict] = []
    if compatible and ca is not None and cb is not None:
        ca = np.asarray(ca, dtype=np.float32)
        cb = np.asarray(cb, dtype=np.float32)
        for name in shared:
            ia, ib = index_a[name], index_b[name]
            if ia < ca.shape[0] and ib < cb.shape[0]:
                per_class.append(
                    {"folder": name, "cosine": float(np.dot(ca[ia], cb[ib]))}
                )
    return {
        "compatible": bool(compatible),
        "a": _head_summary(a),
        "b": _head_summary(b),
        "shared_classes": shared,
        "only_a": sorted(set(names_a) - set(names_b)),
        "only_b": sorted(set(names_b) - set(names_a)),
        "per_class": per_class,
        "mean_cosine": (
            float(np.mean([r["cosine"] for r in per_class])) if per_class else None
        ),
    }


def _head_summary(bundle: dict) -> dict:
    meta = bundle.get("meta", {})
    return {
        "run_id": meta.get("run_id"),
        "trained_at": meta.get("trained_at"),
        "n_classes": len(meta.get("class_names", [])),
        "in_dim": bundle.get("in_dim"),
        "input_sources": list(meta.get("input_sources", [])),
        "source_versions": dict(meta.get("source_versions", {})),
        "has_centroids": bundle.get("centroids") is not None,
    }
