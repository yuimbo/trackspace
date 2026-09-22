"""Weak supervision: folder labels, hard-negative mining, head training, persistence.

The training tests use small synthetic class-separated blobs so they exercise the
real torch path (loss, early stopping, save/load) without being slow. The
invalidation tests are the important ones: a retrain must change the layout and
cluster revisions, or cached coordinates from the previous head would be served.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from backend.embeddings import supervised as sup
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.linalg_utils import l2_normalize
from backend.embeddings.layout import _source_versions_token
from backend.jobs import KIND_MAEST, Job, JobQueue
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_library_work,
    fanout_supervised,
    import_supervised_head,
    make_maest_handler,
    make_supervised_handler,
    project_supervised_missing,
    supervised_imprint_status,
    train_supervised,
    undo_fanout_supervised,
)


def _blobs(n_classes: int = 6, per_class: int = 30, dim: int = 16, spread: float = 0.35, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        center = rng.normal(size=dim).astype(np.float32)
        xs.append(center + spread * rng.normal(size=(per_class, dim)).astype(np.float32))
        ys.append(np.full(per_class, c, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


def _paths(n_classes: int = 6, per_class: int = 30) -> list[str]:
    return [
        f"Root/Folder{c}/t{i}.mp3"
        for c in range(n_classes)
        for i in range(per_class)
    ]


# ── labels ────────────────────────────────────────────────────


def test_folder_label_for_path_uses_leaf_folder():
    assert sup.folder_label_for_path("Root/Techno/x.mp3") == "Root/Techno"
    assert sup.folder_label_for_path("Root/x.mp3") == "Root"
    assert sup.folder_label_for_path("x.mp3") == ""


def test_derive_folder_labels_min_size_and_order():
    paths = _paths(n_classes=3, per_class=10) + ["Root/Tiny/only.mp3"]
    labels, names = sup.derive_folder_labels(paths, min_size=8)
    assert names == ["Root/Folder0", "Root/Folder1", "Root/Folder2"]
    # Tiny folder is below the threshold → unlabelled.
    assert labels[-1] == -1
    assert set(labels[:-1]) == {0, 1, 2}


def test_derive_folder_labels_threshold_can_exclude_everything():
    labels, names = sup.derive_folder_labels(_paths(per_class=5), min_size=100)
    assert names == []
    assert (labels == -1).all()


# ── input assembly ───────────────────────────────────────────


def test_build_input_matrix_is_canonical_and_stateless():
    a = np.ones((4, 3), dtype=np.float32)
    b = np.full((4, 5), 2.0, dtype=np.float32)
    mat, used = sup.build_input_matrix({"maest": a, "maest_logits": b}, {"maest": 1.0})
    assert used == ["maest"]
    # Row-wise L2 normalisation makes every row unit length.
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-6)

    # Same row → same output regardless of which other rows are present
    # (the stateless-input property the projection path relies on).
    single, _ = sup.build_input_matrix(
        {"maest": a[:1], "maest_logits": b[:1]}, {"maest": 1.0}
    )
    np.testing.assert_allclose(single[0], mat[0], atol=1e-6)


def test_build_input_matrix_skips_zero_weight_and_missing():
    a = np.ones((3, 4), dtype=np.float32)
    mat, used = sup.build_input_matrix({"maest": a}, {"maest": 0.0})
    assert used == []
    assert mat.size == 0

    mat, used = sup.build_input_matrix({"maest": a}, {"maest": 1.0})
    assert used == ["maest"]
    assert mat.shape == (3, 4)


# ── excerpt mining ───────────────────────────────────────────


def test_decode_excerpts_shape_and_empty():
    flat = np.arange(3 * sup.MAEST_EMBED_DIM, dtype=np.float32)
    assert sup.decode_excerpts(flat).shape == (3, sup.MAEST_EMBED_DIM)
    assert sup.decode_excerpts(np.array([], dtype=np.float32)).shape == (0, sup.MAEST_EMBED_DIM)


def test_excerpt_confidence_penalises_disagreement():
    agree = [np.stack([v, v, v]) for v in np.eye(4, dtype=np.float32)]
    disagree = [np.stack([v, -v, v]) for v in np.eye(4, dtype=np.float32)]
    conf = sup.excerpt_confidence(agree + disagree)
    assert conf[:4].min() > 0.99
    assert conf[4:].max() < 0.1
    # A single excerpt carries no dispersion signal → full confidence.
    single = sup.excerpt_confidence([np.ones((1, 4), dtype=np.float32)])
    assert single[0] == pytest.approx(1.0)


def test_mine_hard_negatives_are_cross_label_and_symmetric():
    x, y = _blobs(n_classes=3, per_class=10, dim=8, spread=0.05)
    pairs = sup.mine_hard_negatives(x, y, k=2)
    assert pairs.shape[1] == 2
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert (y[pairs[:, 0]] != y[pairs[:, 1]]).all()


def test_mine_hard_negatives_empty_guard():
    x, y = _blobs(n_classes=1, per_class=4, dim=4)
    assert sup.mine_hard_negatives(x, y).shape == (0, 2)


# ── training ────────────────────────────────────────────────


def test_train_head_separates_classes():
    x, y = _blobs(n_classes=6, per_class=40, dim=16)
    result = sup.train_head(x, y, epochs=25, patience=25, seed=1)
    assert result["in_dim"] == 16
    assert result["out_dim"] == sup.SUPERVISED_DIM
    assert result["stats"]["n_classes"] == 6

    z = sup.project_from_state(
        result["state_dict"],
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        mat=x,
    )
    assert z.shape == (x.shape[0], sup.SUPERVISED_DIM)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)

    sim = z @ z.T
    off = ~np.eye(len(y), dtype=bool)
    same = y[:, None] == y[None, :]
    assert sim[same & off].mean() > sim[~same & off].mean()


def test_train_head_reports_epoch_progress():
    x, y = _blobs(n_classes=3, per_class=20, dim=8)
    seen: list[tuple[int, int]] = []
    sup.train_head(
        x,
        y,
        epochs=5,
        patience=5,
        seed=4,
        on_epoch=lambda epoch, total: seen.append((epoch, total)),
    )
    # One call per completed epoch, 1-based, with the epoch budget.
    assert [e for e, _ in seen] == [1, 2, 3, 4, 5]
    assert {t for _, t in seen} == {5}


def test_train_head_rejects_insufficient_labels():
    x, y = _blobs(n_classes=1, per_class=8, dim=4)
    with pytest.raises(ValueError):
        sup.train_head(x, y)


def test_train_head_ignores_unlabelled_rows():
    x, y = _blobs(n_classes=3, per_class=20, dim=8)
    y = y.copy()
    y[:10] = -1
    result = sup.train_head(x, y, epochs=5, patience=5, seed=2)
    assert result["stats"]["n_labelled"] == len(y) - 10


# ── persistence + projection ────────────────────────────────


def test_head_roundtrip_and_run_id(tmp_path):
    x, y = _blobs(n_classes=4, per_class=20, dim=8)
    result = sup.train_head(x, y, epochs=3, patience=3, seed=3)
    meta = sup.build_meta(
        run_id=sup.new_run_id(str(tmp_path)),
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        input_sources=["maest"],
        input_weights={"maest": 1.0},
        class_names=["a", "b"],
        source_versions={"maest": 1},
        stats=result["stats"],
    )
    assert sup.current_run_id(str(tmp_path)) == 0
    sup.save_head(str(tmp_path), result["state_dict"], meta)
    assert sup.current_run_id(str(tmp_path)) == 1
    assert sup.new_run_id(str(tmp_path)) == 2

    head = sup.load_head(str(tmp_path))
    assert head is not None
    z = sup.project(head, x)
    assert z.shape == (x.shape[0], sup.SUPERVISED_DIM)


def test_load_head_absent_is_none(tmp_path):
    assert sup.load_head(str(tmp_path)) is None
    assert sup.current_run_id(str(tmp_path)) == 0


# ── cache version / invalidation ──────────────────────────────


def test_cache_version_follows_run_id(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    assert sup.cache_version(fc) == 0
    fc.set_meta(sup.RUN_ID_META_KEY, 3)
    assert sup.cache_version(fc) == 3
    fc.set_meta(sup.RUN_ID_META_KEY, "garbage")
    assert sup.cache_version(fc) == 0


def test_source_versions_token_changes_on_retrain(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    before = _source_versions_token(fc)
    fc.set_meta(sup.RUN_ID_META_KEY, 1)
    after = _source_versions_token(fc)
    assert before != after


# ── pipeline integration (handlers + cache) ────────────────────


class _FakeAnalysis:
    def __init__(self, dim: int = 768, logit_dim: int = 519, n_excerpts: int = 3):
        self.embedding = np.ones(dim, dtype=np.float32)
        self.logits = np.ones(logit_dim, dtype=np.float32)
        self.excerpt_embeddings = np.stack(
            [np.full(dim, k + 1, dtype=np.float32) for k in range(n_excerpts)]
        )
        self.num_excerpts = n_excerpts


class _FakeMaest:
    MAEST_VERSION = 1

    def __init__(self, n_excerpts: int = 3, style_labels: list[str] | None = None):
        self._n = n_excerpts
        self._labels = list(style_labels or [])

    def is_model_ready(self) -> bool:
        return True

    def load_error(self) -> None:
        return None

    def discogs_style_labels(self) -> list[str]:
        return list(self._labels)

    def top_styles(self, logits, k: int = 5) -> list[tuple[str, float]]:
        labels = self.discogs_style_labels()
        v = np.asarray(logits, dtype=np.float32).reshape(-1)
        if not labels or v.size == 0:
            return []
        n = min(len(labels), v.size)
        idx = np.argsort(-v[:n])[: max(1, k)]
        return [(labels[int(i)], float(v[int(i)])) for i in idx]

    def analyze_batch(self, paths, *, on_decode_warning=None, max_batch=12):
        return {p: _FakeAnalysis(n_excerpts=self._n) for p in paths}


def _deps(tmp_path, fc: FeatureCache, *, maest=None) -> AnalysisDeps:
    track = tmp_path / "track.mp3"
    track.write_bytes(b"ID3\x00fake")
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda _v: str(track),
        virtual_from_abs=lambda _a: "track.mp3",
        list_mp3s=lambda _f, _r: [str(track)],
        maest_mod=maest or _FakeMaest(),
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        data_dir=str(tmp_path),
    )


def _maest_job(job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind=KIND_MAEST,
        dedupe_key="track.mp3",
        payload={"path": "track.mp3"},
        attempts=0,
        priority=100,
    )


def test_maest_handler_persists_excerpts(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    handler = make_maest_handler(_deps(tmp_path, fc))

    out = handler([_maest_job()])
    assert out[1].ok is True
    assert fc.has("maest", "fp", version=1)
    assert fc.has("maest_excerpts", "fp", version=1)
    blob = fc.get("maest_excerpts", "fp", version=1)
    assert blob is not None
    assert sup.decode_excerpts(blob).shape == (3, sup.MAEST_EMBED_DIM)

    # Second run is served from the cache — both pooled outputs and excerpts.
    again = handler([_maest_job(2)])
    assert again[2].ok is True
    assert again[2].detail.get("cached") is True


def test_maest_handler_reanalyses_when_excerpts_missing(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    # A row written before excerpt vectors shipped: pooled pair only.
    fc.put_maest("fp", np.ones(768, np.float32), np.ones(519, np.float32), version=1)
    handler = make_maest_handler(_deps(tmp_path, fc))

    out = handler([_maest_job()])
    assert out[1].ok is True
    assert out[1].detail.get("cached") is not True
    assert fc.has("maest_excerpts", "fp", version=1)


def test_enqueue_backfills_tracks_missing_excerpts(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put_maest("fp1", np.ones(768, np.float32), np.ones(519, np.float32), version=1)
    fc.put_maest(
        "fp2",
        np.ones(768, np.float32),
        np.ones(519, np.float32),
        excerpts=np.ones((3, 768), np.float32),
        version=1,
    )
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    queued = enqueue_library_work(queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_MAEST,))
    assert queued[KIND_MAEST] == 1


def test_enqueue_skips_tracks_with_excerpts(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    for fp in ("fp1", "fp2"):
        fc.put_maest(
            fp,
            np.ones(768, np.float32),
            np.ones(519, np.float32),
            excerpts=np.ones((3, 768), np.float32),
            version=1,
        )
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    queued = enqueue_library_work(queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_MAEST,))
    assert queued[KIND_MAEST] == 0


def _library_infos(tmp_path, fc: FeatureCache, *, n_classes: int = 3, per_class: int = 10):
    infos = []
    for c in range(n_classes):
        for i in range(per_class):
            fp = f"fp-{c}-{i}"
            fc.put_maest(
                fp,
                np.full(768, c + 1, np.float32),
                np.full(519, c + 1, np.float32),
                excerpts=np.stack([np.full(768, c + 1, np.float32)] * 2),
                version=1,
            )
            infos.append({"path": f"Root/Folder{c}/t{i}.mp3", "fingerprint": fp})
    return infos


def test_train_supervised_writes_versioned_rows(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    infos = _library_infos(tmp_path, fc)
    deps = _deps(tmp_path, fc)

    stats = train_supervised(deps, build_track_infos=lambda _f, _r: infos)
    assert stats.get("skipped") is not True
    assert stats["run_id"] == 1
    assert stats["folders"] == 3
    assert fc.get_meta(sup.RUN_ID_META_KEY) == "1"

    have = fc.get_all("supervised", [t["fingerprint"] for t in infos], version=1)
    assert len(have) == len(infos)

    # New tracks analysed after the run are projected through the saved head.
    new_fp = "fp-new"
    fc.put_maest(new_fp, np.ones(768, np.float32), np.ones(519, np.float32), version=1)
    infos.append({"path": "Root/Folder0/new.mp3", "fingerprint": new_fp})
    assert project_supervised_missing(deps, infos) == 1
    assert fc.has("supervised", new_fp, version=1)


def test_supervised_handler_skips_without_folders(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put_maest("fp", np.ones(768, np.float32), np.ones(519, np.float32), version=1)
    infos = [{"path": "Root/Only/t.mp3", "fingerprint": "fp"}]
    deps = _deps(tmp_path, fc)

    handler = make_supervised_handler(deps, build_track_infos=lambda _f, _r: infos)
    job = Job(
        id=1,
        kind="supervise",
        dedupe_key="library",
        payload={"reason": "manual"},
        attempts=0,
        priority=150,
    )
    out = handler([job])
    assert out[1].ok is True
    assert out[1].detail.get("skipped") is True


def _two_root_infos(fc: FeatureCache):
    """Two roots of three class-separated folders each, and their track infos."""
    rng = np.random.default_rng(0)
    infos = []
    for root in ("A", "B"):
        for c in range(3):
            center = rng.normal(size=768).astype(np.float32)
            for i in range(10):
                fp = f"{root}-{c}-{i}"
                fc.put_maest(
                    fp, center, np.ones(519, np.float32), version=1
                )
                infos.append(
                    {"path": f"{root}/Folder{c}/t{i}.mp3", "fingerprint": fp}
                )
    return infos


def test_train_supervised_restricts_labels_to_selected_folders(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    infos = _two_root_infos(fc)

    def build(folder: str, _recursive: bool) -> list[dict]:
        prefix = f"{folder}/"
        return [t for t in infos if t["path"].startswith(prefix)]

    stats = train_supervised(
        _deps(tmp_path, fc), build_track_infos=build, folders=["A"]
    )
    assert stats.get("skipped") is not True
    assert stats["folders"] == 3

    meta = sup.read_meta(str(tmp_path))
    assert meta["train_folders"] == ["A"]
    # Only the selected root's folders became classes, so the head cannot place
    # anything into B's folders.
    assert meta["class_names"] and all(
        name.startswith("A/") for name in meta["class_names"]
    )


def test_supervised_handler_reads_folders_from_payload(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    infos = _two_root_infos(fc)

    def build(folder: str, _recursive: bool) -> list[dict]:
        prefix = f"{folder}/"
        return [t for t in infos if t["path"].startswith(prefix)]

    handler = make_supervised_handler(_deps(tmp_path, fc), build_track_infos=build)
    job = Job(
        id=1,
        kind="supervise",
        dedupe_key="library",
        payload={"reason": "manual", "folders": ["B"]},
        attempts=0,
        priority=150,
    )
    assert handler([job])[1].ok is True
    assert sup.read_meta(str(tmp_path))["train_folders"] == ["B"]


def test_supervised_handler_uses_default_folders_when_payload_empty(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    infos = _two_root_infos(fc)

    def build(folder: str, _recursive: bool) -> list[dict]:
        prefix = f"{folder}/"
        return [t for t in infos if t["path"].startswith(prefix)]

    handler = make_supervised_handler(
        _deps(tmp_path, fc),
        build_track_infos=build,
        default_folders=lambda: ["B"],
    )
    job = Job(
        id=1,
        kind="supervise",
        dedupe_key="library",
        payload={"reason": "manual"},
        attempts=0,
        priority=150,
    )
    assert handler([job])[1].ok is True
    # No selection in the payload: the app's library-roots default applies, so
    # external roots are not trained on unless checked explicitly.
    assert sup.read_meta(str(tmp_path))["train_folders"] == ["B"]


def test_supervised_handler_payload_folders_win_over_default(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    infos = _two_root_infos(fc)

    def build(folder: str, _recursive: bool) -> list[dict]:
        prefix = f"{folder}/"
        return [t for t in infos if t["path"].startswith(prefix)]

    handler = make_supervised_handler(
        _deps(tmp_path, fc),
        build_track_infos=build,
        default_folders=lambda: ["A"],
    )
    job = Job(
        id=1,
        kind="supervise",
        dedupe_key="library",
        payload={"reason": "manual", "folders": ["B"]},
        attempts=0,
        priority=150,
    )
    assert handler([job])[1].ok is True
    assert sup.read_meta(str(tmp_path))["train_folders"] == ["B"]


def test_supervised_imprint_status_counts_ready_tracks(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    ready_fp, pending_fp = "ready", "pending"
    for fp in (ready_fp, pending_fp):
        fc.put_maest(
            fp, np.ones(768, np.float32), np.ones(519, np.float32), version=1
        )
    fc.put("rhythm", ready_fp, np.ones(14, np.float32), version=1)
    fc.put("clap", ready_fp, np.ones(512, np.float32), version=0)
    infos = [
        {"path": "A/ready.mp3", "fingerprint": ready_fp},
        {"path": "A/pending.mp3", "fingerprint": pending_fp},
    ]

    status = supervised_imprint_status(_deps(tmp_path, fc), infos)
    assert status == {"tracks": 2, "analyzed": 1, "pending": 1, "ready": False}

    # The second track's missing rhythm + CLAP rows complete the imprint set.
    fc.put("rhythm", pending_fp, np.ones(14, np.float32), version=1)
    fc.put("clap", pending_fp, np.ones(512, np.float32), version=0)
    status = supervised_imprint_status(_deps(tmp_path, fc), infos)
    assert status["ready"] is True and status["pending"] == 0


# ── centroids, folder prediction, and the portable bundle ──────

def test_compute_centroids_aligns_with_classes_and_flags_empty():
    z = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    centroids, counts = sup.compute_centroids(z, y, 2)
    assert counts.tolist() == [2, 2]
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-6)
    # A class with no labelled row stays a zero row, not a spurious direction.
    empty, counts3 = sup.compute_centroids(z, y, 3)
    assert counts3.tolist() == [2, 2, 0]
    np.testing.assert_allclose(empty[2], 0.0)


def test_predict_folders_nearest_centroid():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mat = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    idx, sim = sup.predict_folders(centroids, mat)
    assert idx.tolist() == [0, 1]
    assert (sim > 0.9).all()
    # A dimension mismatch cannot be scored → -1, never a bogus class.
    bad, _ = sup.predict_folders(centroids, np.ones((1, 3), np.float32))
    assert bad.tolist() == [-1]


def test_folder_subpath_strips_training_root():
    assert sup.folder_subpath("3f2a/Techno") == "Techno"
    assert sup.folder_subpath("3f2a/Techno/Deep") == "Techno/Deep"
    # A class that is a root itself maps to the target root, not a subfolder.
    assert sup.folder_subpath("3f2a") == ""


def test_compute_match_thresholds_is_per_class_percentile():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Class 0 is tight (similarities near 1); class 1 is diffuse.
    tight = np.tile(np.array([1.0, 0.0], np.float32), (10, 1))
    diffuse = np.stack(
        [
            l2_normalize(np.array([[1.0, s]], np.float32))[0]
            for s in np.linspace(0.0, 1.0, 10)
        ]
    )
    z = np.vstack([tight, diffuse])
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    thresholds = sup.compute_match_thresholds(z, y, centroids, 2, percentile=10.0)
    assert thresholds[0] > thresholds[1]
    # A class with no training rows can never gate a track.
    empty = sup.compute_match_thresholds(z, y, centroids, 3, percentile=10.0)
    assert empty[2] == sup.MATCH_ALWAYS


def test_matches_thresholds_gates_far_rows_only():
    idx = np.array([0, 1, -1], dtype=np.int64)
    sim = np.array([0.9, 0.2, 0.99], dtype=np.float32)
    thresholds = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    assert sup.matches_thresholds(idx, sim, thresholds).tolist() == [True, False, False]
    # No thresholds (an old head) matches everything, preserving old behaviour.
    assert sup.matches_thresholds(idx, sim, None).tolist() == [True, True, True]


def test_bundle_version_mismatches_reports_only_changed_sources():
    bundle = {"meta": {"source_versions": {"maest": 1, "clap": 2}}}
    assert sup.bundle_version_mismatches(bundle, {"maest": 1, "clap": 3}) == {
        "clap": {"trained": 2, "current": 3}
    }
    assert sup.bundle_version_mismatches(bundle, {"maest": 1, "clap": 2}) == {}
    # An unknown source in the current set is not a mismatch.
    assert sup.bundle_version_mismatches(bundle, {"maest": 1}) == {}


def _trained_head(tmp_path: Path, *, dim: int = 8, n_classes: int = 3):
    """Train a small real head on synthetic blobs and return its bundle."""
    x, y = _blobs(n_classes=n_classes, per_class=20, dim=dim)
    result = sup.train_head(x, y, epochs=3, patience=3, seed=6)
    z = sup.project_from_state(
        result["state_dict"],
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        mat=x,
    )
    centroids, _ = sup.compute_centroids(z, y, n_classes)
    meta = sup.build_meta(
        run_id=1,
        in_dim=result["in_dim"],
        hidden_dim=result["hidden_dim"],
        out_dim=result["out_dim"],
        input_sources=["maest"],
        input_weights={"maest": 1.0},
        class_names=[f"Root/F{c}" for c in range(n_classes)],
        source_versions={"maest": 1},
        stats=result["stats"],
    )
    sup.save_head(
        str(tmp_path), result["state_dict"], meta, centroids=centroids
    )
    return sup.head_bundle(str(tmp_path))


def test_head_bundle_roundtrip_and_install(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    bundle = _trained_head(src)
    assert bundle is not None
    assert bundle["format"] == sup.HEAD_BUNDLE_FORMAT
    assert bundle["centroids"] is not None

    path = str(tmp_path / "head.pt")
    sup.write_head_bundle(path, bundle)
    read = sup.read_head_bundle(path)

    # The imported head gets a fresh local run id, never the source's id — the
    # id is the cache version, so reusing it would serve stale rows.
    installed = sup.install_head_bundle(str(dst), read)
    assert installed["run_id"] == 1
    assert sup.current_run_id(str(dst)) == 1
    loaded = sup.load_head(str(dst))
    assert loaded is not None
    np.testing.assert_allclose(
        loaded["payload"]["centroids"], bundle["centroids"], atol=1e-6
    )
    # A second import bumps the local run id so the cache invalidates again.
    assert sup.install_head_bundle(str(dst), read)["run_id"] == 2


def test_read_head_bundle_rejects_foreign_file(tmp_path):
    import torch

    path = str(tmp_path / "not-a-bundle.pt")
    torch.save({"hello": "world"}, path)
    with pytest.raises(ValueError):
        sup.read_head_bundle(path)


def _bundle(names, centroids, *, dim: int = 2, source_versions=None):
    return {
        "format": sup.HEAD_BUNDLE_FORMAT,
        "in_dim": dim,
        "state_dict": {},
        "meta": {
            "run_id": 1,
            "class_names": list(names),
            "input_sources": ["maest"],
            "source_versions": source_versions or {"maest": 1},
        },
        "centroids": np.asarray(centroids, dtype=np.float32),
    }


def test_compare_heads_scores_shared_folders_only():
    a = _bundle(["Root/A", "Root/B"], [[1, 0], [0, 1]])
    b = _bundle(["Root/A", "Root/C"], [[1, 0], [1, 0]])
    out = sup.compare_heads(a, b)
    assert out["compatible"] is True
    assert out["shared_classes"] == ["Root/A"]
    assert out["only_a"] == ["Root/B"]
    assert out["only_b"] == ["Root/C"]
    assert out["per_class"] == [{"folder": "Root/A", "cosine": pytest.approx(1.0)}]
    assert out["mean_cosine"] == pytest.approx(1.0)


def test_compare_heads_flags_incompatible_input_space():
    a = _bundle(["Root/A"], [[1, 0]], dim=2)
    b = _bundle(["Root/A"], [[1, 0]], dim=16)
    out = sup.compare_heads(a, b)
    assert out["compatible"] is False
    assert out["per_class"] == []
    assert out["mean_cosine"] is None


def _fanout_deps(
    tmp_path: Path, fc: FeatureCache, root: Path, *, maest=None
) -> AnalysisDeps:
    """Deps that map a real temp tree so fan-out can actually move files."""
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda v: str(root / v) if v else str(root),
        virtual_from_abs=lambda a: os.path.relpath(a, root),
        list_mp3s=lambda f, _r: sorted(
            str(p) for p in (root / f if f else root).glob("*.mp3")
        ),
        maest_mod=maest or _FakeMaest(),
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        data_dir=str(tmp_path),
    )


def test_fanout_plans_then_moves_into_predicted_folders(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    # Class-distinguishable *directions*: constant per-class rows would normalise
    # to the same unit vector and leave the head nothing to separate.
    rng = np.random.default_rng(0)
    centers = [rng.normal(size=768).astype(np.float32) for _ in range(2)]
    infos = []
    for c, center in enumerate(centers):
        for i in range(10):
            fp = f"train-{c}-{i}"
            fc.put_maest(
                fp,
                center + 0.05 * rng.normal(size=768).astype(np.float32),
                np.ones(519, np.float32),
                version=1,
            )
            infos.append({"path": f"Root/Folder{c}/t{i}.mp3", "fingerprint": fp})
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    root = tmp_path / "lib"
    loose = root / "loose"
    loose.mkdir(parents=True)
    fanout_infos = []
    for c, center in enumerate(centers):
        fp = f"loose-{c}"
        fc.put_maest(fp, center, np.ones(519, np.float32), version=1)
        (loose / f"t{c}.mp3").write_bytes(b"ID3\x00fake")
        fanout_infos.append({"path": f"loose/t{c}.mp3", "fingerprint": fp})

    fdeps = _fanout_deps(tmp_path, fc, root)
    plan = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=False,
    )
    assert plan["ok"] is True
    assert plan["tracks"] == 2 and plan["moved"] == 0
    # A dry run leaves every file exactly where it was.
    assert (loose / "t0.mp3").exists()
    assert {e["target"] for e in plan["plan"]} == {
        "loose/Folder0/t0.mp3",
        "loose/Folder1/t1.mp3",
    }

    applied = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
    )
    assert applied["moved"] == 2
    assert (loose / "Folder0" / "t0.mp3").exists()
    assert (loose / "Folder1" / "t1.mp3").exists()
    assert not (loose / "t0.mp3").exists()


def _fanout_scenario(
    tmp_path: Path,
    *,
    threshold_override: float | None = None,
    maest=None,
    subdir: str = "",
):
    """Train a real head on two class blobs and lay out two loose tracks."""
    fc = FeatureCache(str(tmp_path / "features.db"))
    rng = np.random.default_rng(0)
    centers = [rng.normal(size=768).astype(np.float32) for _ in range(2)]
    infos = []
    for c, center in enumerate(centers):
        for i in range(10):
            fp = f"train-{c}-{i}"
            fc.put_maest(
                fp,
                center + 0.05 * rng.normal(size=768).astype(np.float32),
                np.ones(519, np.float32),
                version=1,
            )
            infos.append({"path": f"Root/Folder{c}/t{i}.mp3", "fingerprint": fp})
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)
    if threshold_override is not None:
        head = sup.load_head(str(tmp_path))
        assert head is not None
        head["meta"]["class_match_thresholds"] = [threshold_override] * 2
        sup.save_head(
            str(tmp_path),
            head["payload"]["state_dict"],
            head["meta"],
            centroids=head["payload"].get("centroids"),
        )

    root = tmp_path / "lib"
    loose = root / "loose"
    place = loose / subdir if subdir else loose
    place.mkdir(parents=True)
    fanout_infos = []
    for c, center in enumerate(centers):
        fp = f"loose-{c}"
        fc.put_maest(fp, center, np.ones(519, np.float32), version=1)
        (place / f"t{c}.mp3").write_bytes(b"ID3\x00fake")
        rel = f"loose/{subdir}/t{c}.mp3" if subdir else f"loose/t{c}.mp3"
        fanout_infos.append({"path": rel, "fingerprint": fp})
    return (
        fc,
        _fanout_deps(tmp_path, fc, root, maest=maest),
        loose,
        fanout_infos,
    )


def test_fanout_leaves_a_track_far_from_every_folder(tmp_path):
    # A threshold above any achievable cosine (max 1.0) gates everything: a
    # track that resembles no learned folder must not be forced into one.
    fc, fdeps, loose, fanout_infos = _fanout_scenario(
        tmp_path, threshold_override=2.0
    )
    result = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
    )
    assert result["ok"] is True
    assert result["moved"] == 0
    assert result["unmatched"] == 2
    assert {e["status"] for e in result["plan"]} == {"unmatched"}
    assert (loose / "t0.mp3").exists() and (loose / "t1.mp3").exists()
    # Nothing moved, so there is nothing to undo.
    assert fc.latest_move_batch() is None
    assert undo_fanout_supervised(fdeps)["ok"] is False


def test_fanout_journals_moves_and_undo_restores_them(tmp_path):
    fc, fdeps, loose, fanout_infos = _fanout_scenario(tmp_path)
    applied = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
    )
    assert applied["moved"] == 2

    batch = fc.latest_move_batch()
    assert batch is not None and len(batch["moves"]) == 2
    # The journal records the *actual* targets, so a collision suffix is undone
    # back to the original name rather than left behind.
    assert {m["source"] for m in batch["moves"]} == {
        str(loose / "t0.mp3"),
        str(loose / "t1.mp3"),
    }

    undone = undo_fanout_supervised(fdeps)
    assert undone["ok"] is True and undone["restored"] == 2
    assert (loose / "t0.mp3").exists() and (loose / "t1.mp3").exists()
    assert not (loose / "Folder0" / "t0.mp3").exists()
    assert not (loose / "Folder1" / "t1.mp3").exists()
    # A batch is only undoable once.
    assert fc.latest_move_batch() is None
    assert undo_fanout_supervised(fdeps)["ok"] is False


def test_generic_subpath_mirrors_the_discogs_taxonomy():
    assert sup.generic_subpath("Electronic---Techno") == "Generic/Electronic/Techno"
    assert sup.generic_subpath("Electronic") == "Generic/Electronic"
    assert sup.generic_subpath("Folk, World, & Country---Folk") == (
        "Generic/Folk, World, & Country/Folk"
    )
    # A separator inside a label part must not silently split the target folder.
    assert sup.generic_subpath("A/B---C") == "Generic/A-B/C"
    assert sup.generic_subpath("---") is None
    assert sup.generic_subpath("") is None


def test_fanout_files_unmatched_tracks_under_generic(tmp_path):
    fc, fdeps, loose, fanout_infos = _fanout_scenario(
        tmp_path,
        threshold_override=2.0,
        maest=_FakeMaest(style_labels=["Electronic---Techno", "Rock---Indie Rock"]),
    )
    result = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
        generic_unmatched=True,
    )
    assert result["ok"] is True
    assert result["generic"] == 2 and result["unmatched"] == 0
    assert result["moved"] == 2
    assert (loose / "Generic" / "Electronic" / "Techno" / "t0.mp3").exists()
    assert (loose / "Generic" / "Electronic" / "Techno" / "t1.mp3").exists()
    assert all(e.get("generic") is True for e in result["plan"])
    # Generic moves are journalled like any other, so undo restores them.
    assert undo_fanout_supervised(fdeps)["restored"] == 2
    assert (loose / "t0.mp3").exists() and (loose / "t1.mp3").exists()


def test_fanout_generic_falls_back_when_labels_are_unavailable(tmp_path):
    # No Discogs label table: there is no name for the bucket, so the track is
    # left in place rather than moved into a folder called nothing.
    fc, fdeps, loose, fanout_infos = _fanout_scenario(
        tmp_path, threshold_override=2.0
    )
    result = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
        generic_unmatched=True,
    )
    assert result["generic"] == 0 and result["unmatched"] == 2
    assert (loose / "t0.mp3").exists() and (loose / "t1.mp3").exists()


def test_fanout_recursive_lifts_nested_files_and_sweeps(tmp_path):
    fc, fdeps, loose, fanout_infos = _fanout_scenario(tmp_path, subdir="Old/Deep")
    result = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        recursive=True,
        apply=True,
    )
    assert result["moved"] == 2
    assert (loose / "Folder0" / "t0.mp3").exists()
    assert (loose / "Folder1" / "t1.mp3").exists()
    # Both emptied levels of the old tree are swept, bounded to the fan-out root.
    assert result["swept_dirs"] == 2
    assert not (loose / "Old").exists()
    # The pre-move relative folder is remembered per fingerprint.
    assert fc.get_origins(["loose-0", "loose-1"]) == {
        "loose-0": "Old/Deep",
        "loose-1": "Old/Deep",
    }
    # Undo restores the files (recreating the swept directory it needs).
    assert undo_fanout_supervised(fdeps)["restored"] == 2
    assert (loose / "Old" / "Deep" / "t0.mp3").exists()
    assert (loose / "Old" / "Deep" / "t1.mp3").exists()


def test_fanout_non_recursive_does_not_sweep_or_record_origins(tmp_path):
    fc, fdeps, loose, fanout_infos = _fanout_scenario(tmp_path, subdir="Old/Deep")
    result = fanout_supervised(
        fdeps,
        build_track_infos=lambda _f, _r: fanout_infos,
        folder_vpath="loose",
        apply=True,
    )
    assert result["swept_dirs"] == 0
    # Without `recursive` the old tree is neither swept nor remembered.
    assert (loose / "Old" / "Deep").is_dir()
    assert fc.get_origins(["loose-0"]) == {}


def test_import_refuses_a_stale_source_version(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    bundle = _trained_head(src)
    assert bundle is not None
    bundle["meta"]["source_versions"] = {"maest": 999}
    with pytest.raises(sup.HeadVersionMismatch) as exc:
        import_supervised_head(_deps(dst, FeatureCache(str(dst / "f.db"))), bundle)
    assert exc.value.mismatches == {"maest": {"trained": 999, "current": 1}}
    # The explicit override installs it anyway.
    meta = import_supervised_head(
        _deps(dst, FeatureCache(str(dst / "f.db"))),
        bundle,
        allow_mismatch=True,
    )
    assert meta["run_id"] == 1
