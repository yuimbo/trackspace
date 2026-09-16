"""Weak supervision: folder labels, hard-negative mining, head training, persistence.

The training tests use small synthetic class-separated blobs so they exercise the
real torch path (loss, early stopping, save/load) without being slow. The
invalidation tests are the important ones: a retrain must change the layout and
cluster revisions, or cached coordinates from the previous head would be served.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import supervised as sup
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.layout import _source_versions_token
from backend.jobs import KIND_MAEST, Job, JobQueue
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_library_work,
    make_maest_handler,
    make_supervised_handler,
    project_supervised_missing,
    train_supervised,
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

    def __init__(self, n_excerpts: int = 3):
        self._n = n_excerpts

    def is_model_ready(self) -> bool:
        return True

    def load_error(self) -> None:
        return None

    def discogs_style_labels(self) -> list[str]:
        return []

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
