"""The /api/supervised/* routes: head info, export, import, compare, fan-out.

Built with a real FeatureCache in a temp dir and synthetic per-class MAEST rows, so
the head is really trained and the bundle really round-trips without touching audio
or the user's cache.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

from backend.factory import _register_error_handlers
from backend.embeddings import supervised as sup
from backend.embeddings.feature_cache import FeatureCache
from backend.routes.api_supervised import create_api_supervised_blueprint
from backend.services.analysis_pipeline import AnalysisDeps, train_supervised


class _FakeMaest:
    MAEST_VERSION = 1

    def is_model_ready(self) -> bool:
        return True

    def load_error(self) -> None:
        return None

    def discogs_style_labels(self) -> list[str]:
        return ["Electronic---Techno"]

    def top_styles(self, logits, k: int = 5) -> list[tuple[str, float]]:
        labels = self.discogs_style_labels()
        v = np.asarray(logits, dtype=np.float32).reshape(-1)
        if not labels or v.size == 0:
            return []
        n = min(len(labels), v.size)
        idx = np.argsort(-v[:n])[: max(1, k)]
        return [(labels[int(i)], float(v[int(i)])) for i in idx]


def _library(tmp_path: Path, *, n_classes: int = 3, per_class: int = 10):
    """Class-distinguishable MAEST rows, their virtual paths, and class centres."""
    fc = FeatureCache(str(tmp_path / "features.db"))
    rng = np.random.default_rng(0)
    infos = []
    centers = []
    for c in range(n_classes):
        center = rng.normal(size=768).astype(np.float32)
        centers.append(center)
        for i in range(per_class):
            fp = f"fp-{c}-{i}"
            fc.put_maest(
                fp,
                center + 0.05 * rng.normal(size=768).astype(np.float32),
                np.ones(519, np.float32),
                version=1,
            )
            infos.append({"path": f"Root/Folder{c}/t{i}.mp3", "fingerprint": fp})
    return fc, infos, centers


def _deps(tmp_path, fc, *, root: Path | None = None):
    base = root or tmp_path
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda v: str(base / v) if v else str(base),
        virtual_from_abs=lambda a: os.path.relpath(a, base),
        list_mp3s=lambda f, _r: sorted(
            str(p) for p in (base / f if f else base).glob("*.mp3")
        ),
        maest_mod=_FakeMaest(),
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        data_dir=str(tmp_path),
    )


def _app(tmp_path, fc, infos, *, deps=None):
    analysis_deps = deps or _deps(tmp_path, fc)
    bp = create_api_supervised_blueprint(
        analysis_deps=analysis_deps,
        build_track_infos_fn=lambda _f, _r: infos,
    )
    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client(), analysis_deps


def test_head_info_is_unavailable_before_training(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    body = client.get("/api/supervised/head").get_json()
    assert body["available"] is False


def test_export_then_import_roundtrips(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    info = client.get("/api/supervised/head").get_json()
    assert info["available"] is True and info["has_centroids"] is True

    exported = client.get("/api/supervised/export")
    assert exported.status_code == 200
    assert exported.mimetype == "application/octet-stream"
    blob = exported.data

    # Import into a *different* library: it gets a fresh local run id.
    other = tmp_path / "other"
    other.mkdir()
    fc2, infos2, _ = _library(other)
    client2, _ = _app(other, fc2, infos2)
    imported = client2.post(
        "/api/supervised/import",
        data={"bundle": (io.BytesIO(blob), "head.pt")},
        content_type="multipart/form-data",
    )
    assert imported.status_code == 200
    body = imported.get_json()
    assert body["ok"] is True
    assert body["head"]["available"] is True
    assert body["head"]["run_id"] == 1


def test_export_without_a_head_is_404(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    r = client.get("/api/supervised/export")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


def test_import_rejects_a_foreign_file(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    r = client.post(
        "/api/supervised/import",
        data={"bundle": (io.BytesIO(b"not a bundle"), "head.pt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_compare_without_a_head_is_400(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    r = client.post(
        "/api/supervised/compare",
        data={"bundle": (io.BytesIO(b"x"), "head.pt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "compare" in r.get_json()["error"]


def test_compare_against_itself_is_perfectly_compatible(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)
    blob = client.get("/api/supervised/export").data
    body = client.post(
        "/api/supervised/compare",
        data={"bundle": (io.BytesIO(blob), "head.pt")},
        content_type="multipart/form-data",
    ).get_json()
    assert body["ok"] is True
    assert body["compare"]["compatible"] is True
    assert body["compare"]["mean_cosine"] == pytest.approx(1.0, abs=1e-4)


def test_predictions_require_a_head(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    body = client.get("/api/supervised/predictions").get_json()
    assert body["available"] is False
    assert body["assignments"] == {}


def test_predictions_map_each_track_to_its_nearest_folder(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    body = client.get("/api/supervised/predictions").get_json()
    assert body["available"] is True
    # Every training track has a cached row and its own folder is the nearest.
    assert len(body["assignments"]) == len(infos)
    for info in infos:
        assert body["assignments"][info["path"]] == os.path.dirname(info["path"])
    # The match floor is a 5th percentile, so some rows may fall below it, but a
    # healthy majority clears it and every match is a known track.
    matched = set(body["matched"])
    assert 0 < len(matched) <= len(body["assignments"])
    assert matched <= set(body["assignments"])


def test_fanout_requires_a_folder(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    r = client.post("/api/supervised/fanout", json={})
    assert r.status_code == 400
    assert "folder" in r.get_json()["error"]


def test_fanout_plans_then_applies_folder_moves(tmp_path):
    fc, infos, centers = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    root = tmp_path / "lib"
    loose = root / "loose"
    loose.mkdir(parents=True)
    fanout_infos = []
    for c in (0, 1):
        fp = f"loose-{c}"
        fc.put_maest(fp, centers[c], np.ones(519, np.float32), version=1)
        (loose / f"t{c}.mp3").write_bytes(b"ID3\x00fake")
        fanout_infos.append({"path": f"loose/t{c}.mp3", "fingerprint": fp})

    deps = _deps(tmp_path, fc, root=root)
    client, _ = _app(tmp_path, fc, fanout_infos, deps=deps)

    plan = client.post(
        "/api/supervised/fanout",
        json={"folder": "loose", "apply": False},
    ).get_json()
    assert plan["ok"] is True
    assert plan["moved"] == 0
    assert (loose / "t0.mp3").exists()

    applied = client.post(
        "/api/supervised/fanout",
        json={"folder": "loose", "apply": True},
    ).get_json()
    assert applied["moved"] == 2
    assert not (loose / "t0.mp3").exists()


def test_fanout_generic_flag_files_unmatched_tracks(tmp_path):
    fc, infos, centers = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    # Gate every track out of the learned folders: they must go generic.
    head = sup.load_head(str(tmp_path))
    assert head is not None
    head["meta"]["class_match_thresholds"] = [2.0] * head["meta"]["n_classes"]
    sup.save_head(
        str(tmp_path),
        head["payload"]["state_dict"],
        head["meta"],
        centroids=head["payload"].get("centroids"),
    )

    root = tmp_path / "lib"
    loose = root / "loose"
    loose.mkdir(parents=True)
    fanout_infos = []
    for c in (0, 1):
        fp = f"loose-{c}"
        fc.put_maest(fp, centers[c], np.ones(519, np.float32), version=1)
        (loose / f"t{c}.mp3").write_bytes(b"ID3\x00fake")
        fanout_infos.append({"path": f"loose/t{c}.mp3", "fingerprint": fp})

    deps = _deps(tmp_path, fc, root=root)
    client, _ = _app(tmp_path, fc, fanout_infos, deps=deps)

    planned = client.post(
        "/api/supervised/fanout",
        json={"folder": "loose", "generic": True},
    ).get_json()
    assert planned["ok"] is True
    assert planned["generic"] == 2 and planned["unmatched"] == 0
    assert all(e.get("generic") is True for e in planned["plan"])

    applied = client.post(
        "/api/supervised/fanout",
        json={"folder": "loose", "apply": True, "generic": True},
    ).get_json()
    assert applied["ok"] is True
    assert applied["moved"] == 2
    assert (loose / "Generic" / "Electronic" / "Techno" / "t0.mp3").exists()


def test_fanout_undo_route_reverses_the_last_moves(tmp_path):
    fc, infos, centers = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    root = tmp_path / "lib"
    loose = root / "loose"
    loose.mkdir(parents=True)
    fanout_infos = []
    for c in (0, 1):
        fp = f"loose-{c}"
        fc.put_maest(fp, centers[c], np.ones(519, np.float32), version=1)
        (loose / f"t{c}.mp3").write_bytes(b"ID3\x00fake")
        fanout_infos.append({"path": f"loose/t{c}.mp3", "fingerprint": fp})

    deps = _deps(tmp_path, fc, root=root)
    client, _ = _app(tmp_path, fc, fanout_infos, deps=deps)
    client.post("/api/supervised/fanout", json={"folder": "loose", "apply": True})

    status = client.get("/api/supervised/fanout/undo").get_json()
    assert status == {"ok": True, "available": True, "total": 2}

    undone = client.post("/api/supervised/fanout/undo").get_json()
    assert undone["ok"] is True and undone["restored"] == 2
    assert (loose / "t0.mp3").exists() and (loose / "t1.mp3").exists()
    assert client.get("/api/supervised/fanout/undo").get_json()["available"] is False


def test_import_refuses_a_stale_source_version(tmp_path):
    fc, infos, _ = _library(tmp_path)
    client, _ = _app(tmp_path, fc, infos)
    train_supervised(_deps(tmp_path, fc), build_track_infos=lambda _f, _r: infos)

    import torch

    blob = client.get("/api/supervised/export").data
    bundle = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    bundle["meta"]["source_versions"] = {"maest": 999}
    buf = io.BytesIO()
    torch.save(bundle, buf)
    stale = buf.getvalue()

    other = tmp_path / "other"
    other.mkdir()
    fc2, infos2, _ = _library(other)
    client2, _ = _app(other, fc2, infos2)

    refused = client2.post(
        "/api/supervised/import",
        data={"bundle": (io.BytesIO(stale), "head.pt")},
        content_type="multipart/form-data",
    )
    assert refused.status_code == 409
    body = refused.get_json()
    assert body["ok"] is False
    assert body["source_mismatches"]["maest"] == {"trained": 999, "current": 1}

    # An explicit force imports it anyway.
    forced = client2.post(
        "/api/supervised/import",
        data={"bundle": (io.BytesIO(stale), "head.pt"), "force": "1"},
        content_type="multipart/form-data",
    )
    assert forced.status_code == 200
    assert forced.get_json()["ok"] is True
