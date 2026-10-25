"""Genre-space maps gallery: schema-1 serialization, virtual paths, API allowlist.

Tests the safe ids, dataset fingerprint, persisted-roots mapping, and
compatibility with ``layout_evaluation.load_recipe_json`` + the Flask GET routes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backend.experiments.genre_space import dataset as _dataset
from backend.experiments.genre_space import gallery as gal
from backend.experiments.genre_space import scoring as _scoring

DIMS = {"maest": 768, "maest_logits": 519, "maest_layer7": 2304,
        "mert_l19": 1024, "mert_l23": 1024}
MODEL_SOURCES = {"maest": ("maest", "maest_logits", "maest_layer7"),
                 "mert": ("mert_l19", "mert_l23")}


def _make_manifest(root: Path, n: int = 6) -> dict:
    splits = ("dev", "dev", "dev", "dev", "heldout", "heldout")
    tracks = [
        {
            "fp": f"f{i}", "path": str(Path(root) / f"{i}.mp3"), "label": f"genre{i % 2}",
            "artist": "", "release": "", "mtime_ns": i, "size": i,
            "split": splits[i - 1], "windows": [{"start": 0.0, "duration": 30.0}],
        }
        for i in range(1, n + 1)
    ]
    manifest = {
        "schema_version": 1, "dataset_id": "", "root": str(root),
        "track_db": "/tmp/track_cache.db", "seed": 42, "label_depth": 1,
        "limit": n, "tracks": tracks,
    }
    manifest["dataset_id"] = _dataset.dataset_id(manifest)
    return manifest


def _write_manifest_file(tmp_path, root) -> tuple[dict, Path]:
    manifest = _make_manifest(root)
    p = Path(tmp_path) / "manifest.json"
    p.write_text(json.dumps(manifest))
    return manifest, p


def _write_artifacts(adir, model, manifest, rows=None) -> None:
    adir = Path(adir)
    adir.mkdir(parents=True, exist_ok=True)
    n = len(manifest["tracks"]) if rows is None else rows
    fps = [t["fp"] for t in manifest["tracks"]]
    rng = np.random.default_rng(0)
    arrays = {"fingerprints": np.array(fps, dtype=f"<U{max(len(f) for f in fps)}")}
    for name in MODEL_SOURCES[model]:
        arrays[name] = rng.standard_normal((n, DIMS[name])).astype(np.float32)
    npz = adir / f"{model}.npz"
    np.savez(npz, **arrays)
    meta = {
        "dataset_id": manifest["dataset_id"], "model": model,
        "layers": {name: "test" for name in MODEL_SOURCES[model]},
        "coverage": {"total": len(manifest["tracks"]), "ok": n, "failed": 0},
        "checkpoint": {"arch": "x", "url": "https://example.invalid/x.pt"},
        "artifact": {"npz": f"{model}.npz", "size_bytes": npz.stat().st_size,
                     "sha256": hashlib.sha256(npz.read_bytes()).hexdigest()},
    }
    (adir / f"{model}.json").write_text(json.dumps(meta))


def _rewrite_maest(adir, manifest, seed: int) -> None:
    """Simulate a re-extraction: different embeddings -> new npz + matching sha."""
    adir = Path(adir)
    n = len(manifest["tracks"])
    rng = np.random.default_rng(seed)
    arrays = {"fingerprints": np.array([t["fp"] for t in manifest["tracks"]],
                                       dtype="<U6")}
    for name in MODEL_SOURCES["maest"]:
        arrays[name] = rng.standard_normal((n, DIMS[name])).astype(np.float32)
    npz = adir / "maest.npz"
    np.savez(npz, **arrays)
    meta = json.loads((adir / "maest.json").read_text())
    meta["artifact"]["sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    (adir / "maest.json").write_text(json.dumps(meta))


def _run_maps(cli, tmp_path, monkeypatch, mpath, arts, gallery_dir, roots_file, recipes):
    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod

    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: (np.asarray(mat)[:, :2].astype(np.float32), None))
    monkeypatch.setattr(le_mod, "fit_opentsne",
                        lambda reduced, *, recipe, seed, n_jobs:
                        np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32))
    monkeypatch.setattr(le_mod, "orthogonal_align_to",
                        lambda coords, paths, ref_coords, ref_paths: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.5)
    out = Path(tmp_path) / f"maps_{Path(gallery_dir).name}"
    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--gallery-output", str(gallery_dir),
              "--roots-file", str(roots_file), "--recipes", recipes])
    return out


def _make_app(layout_dir):
    from flask import Flask

    from backend.factory import _register_error_handlers
    from backend.routes.api_layout_evaluation import create_api_layout_evaluation_blueprint

    app = Flask(__name__)
    _register_error_handlers(app)
    app.register_blueprint(create_api_layout_evaluation_blueprint(str(layout_dir)))
    return app


@pytest.fixture(scope="module")
def cli():
    import backend.experiments.genre_space.__main__ as m
    return m


# ── safe gallery ids ─────────────────────────────────────────────────────────


def test_safe_gallery_id_deterministic_and_route_safe():
    cases = {
        "maest_logits": "genre_maest_logits",
        "maest_logits+maest@643-357": "genre_maest_logits_maest_643-357",
        "mert_l19": "genre_mert_l19",
    }
    for raw, expected in cases.items():
        assert gal.safe_gallery_id(raw) == expected
        assert gal.safe_gallery_id(raw) == gal.safe_gallery_id(raw)  # deterministic
    # Every safe id satisfies the API recipe allowlist.
    import re
    ok = re.compile(r"^[A-Za-z0-9_-]+$")
    for raw in cases:
        assert ok.match(gal.safe_gallery_id(raw))


def test_unique_safe_ids_is_a_bijection():
    rids = ["maest_logits", "maest_logits+maest@643-357", "maest_logits-maest_643-357"]
    mapping = gal._unique_safe_ids(rids)
    assert len(set(mapping.values())) == len(rids)  # distinct ids
    assert set(mapping) == set(rids)


# ── full gallery write ───────────────────────────────────────────────────────


def _gallery_fixture(tmp_path, monkeypatch, cli, recipes="maest_logits,maest_logits+maest@643-357"):
    music = tmp_path / "music"
    music.mkdir()
    manifest, mpath = _write_manifest_file(tmp_path, music)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)  # maest family only
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)], "external_roots": [], "temp_roots": []}))
    base = tmp_path / "layout-evaluation"
    gallery_dir = base / "genre"
    _run_maps(cli, tmp_path, monkeypatch, mpath, arts, gallery_dir, roots_file, recipes)
    return manifest, gallery_dir, base


def test_maps_gallery_writes_manifest_and_artifacts(tmp_path, monkeypatch, cli):
    manifest, gallery_dir, _base = _gallery_fixture(tmp_path, monkeypatch, cli)
    assert (gallery_dir / "manifest.json").exists()
    assert (gallery_dir / "genre_maest_logits.json").exists()
    assert (gallery_dir / "genre_maest_logits_maest_643-357.json").exists()

    m = json.loads((gallery_dir / "manifest.json").read_text())
    assert m["schema_version"] == 1
    assert m["status"] == "complete"
    assert isinstance(m["dataset_fingerprint"], str)
    assert m["reference_id"] == "genre_maest_logits"
    assert sorted(m["recipes"]) == [
        "genre_maest_logits", "genre_maest_logits_maest_643-357",
    ]
    for safe, entry in m["recipes"].items():
        assert entry["status"] == "complete"
        assert isinstance(entry["label"], str)
        assert entry["file"] == f"{safe}.json"

    # Reference map is unaligned; the second aligns onto the first (safe id).
    assert m["recipes"]["genre_maest_logits"]["aligned_to"] is None
    assert m["recipes"]["genre_maest_logits_maest_643-357"]["aligned_to"] == "genre_maest_logits"

    art = json.loads((gallery_dir / "genre_maest_logits_maest_643-357.json").read_text())
    assert art["id"] == "genre_maest_logits_maest_643-357"
    assert art["representation_id"] == "maest_logits+maest@643-357"
    assert art["dataset_fingerprint"] == m["dataset_fingerprint"]
    assert art["input_fingerprint"] == m["recipes"]["genre_maest_logits_maest_643-357"]["input_fingerprint"]
    assert art["revision"] == m["recipes"]["genre_maest_logits_maest_643-357"]["revision"]
    assert art["aligned_to"] == "genre_maest_logits"
    assert art["tracks"] == len(manifest["tracks"])
    assert set(art["source_weights"]) == {"maest_logits", "maest"}
    assert set(art["fidelity"]) == {"raw_to_2d", "pca50_to_2d"}
    # Positions carry virtual (rid/rel) paths, never absolute paths.
    assert len(art["positions"]) == 6
    for pos in art["positions"]:
        assert isinstance(pos["path"], str)
        assert not Path(pos["path"]).is_absolute()
        assert "/" in pos["path"]  # rid/rel form


def test_gallery_refuses_overwrite_before_fit(tmp_path, monkeypatch, cli, capsys):
    music = tmp_path / "music"
    music.mkdir()
    manifest, mpath = _write_manifest_file(tmp_path, music)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)]}))
    gallery_dir = tmp_path / "genre"
    gallery_dir.mkdir()
    (gallery_dir / "manifest.json").write_text("{}")  # stale previous run
    calls = []

    from backend.embeddings import layout_evaluation as le_mod
    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda *a, **k: calls.append("pca") or (np.zeros((6, 2)), None))
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "maps"), "--gallery-output", str(gallery_dir),
                  "--roots-file", str(roots_file), "--recipes", "maest_logits"])
    assert "refusing to overwrite existing gallery file" in str(ei.value)
    assert calls == []  # preflight refused before any fitting


def test_gallery_unresolved_root_fails_before_writes(tmp_path, monkeypatch, cli):
    # Track paths live outside every persisted root -> a clean error, no gallery file.
    manifest, mpath = _write_manifest_file(tmp_path, tmp_path / "elsewhere")
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(tmp_path / "music")]}))
    (tmp_path / "music").mkdir()
    gallery_dir = tmp_path / "genre"
    from backend.embeddings import layout_evaluation as le_mod
    monkeypatch.setattr(le_mod, "pca_reduce", lambda *a, **k: (np.zeros((6, 2)), None))
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "maps"), "--gallery-output", str(gallery_dir),
                  "--roots-file", str(roots_file), "--recipes", "maest_logits"])
    assert "not under any persisted library root" in str(ei.value)
    assert not gallery_dir.exists()


# ── fingerprint sensitivity to source artifacts ──────────────────────────────


def test_dataset_fingerprint_changes_with_source_checksum(tmp_path, monkeypatch, cli):
    music = tmp_path / "music"
    music.mkdir()
    manifest, mpath = _write_manifest_file(tmp_path, music)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)]}))
    base = tmp_path / "layout-evaluation"

    _run_maps(cli, tmp_path, monkeypatch, mpath, arts, base / "genre_a", roots_file,
              "maest_logits")
    fp_a = json.loads((base / "genre_a" / "manifest.json").read_text())["dataset_fingerprint"]

    # Same cohort, same weights/budget/seed — only the embeddings change.
    _rewrite_maest(arts, manifest, seed=99)
    _run_maps(cli, tmp_path, monkeypatch, mpath, arts, base / "genre_b", roots_file,
              "maest_logits")
    fp_b = json.loads((base / "genre_b" / "manifest.json").read_text())["dataset_fingerprint"]

    assert fp_a != fp_b


# ── roots: stable persisted ids + virtual mapping ────────────────────────────


def test_load_roots_and_virtual_paths_stable_ids(tmp_path):
    music = tmp_path / "music"
    (music / "sub").mkdir(parents=True)
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)], "external_roots": [], "temp_roots": []}))
    roots = gal.load_roots(str(roots_file))
    assert roots
    rid = next(iter(roots))
    # Stable across reloads (the persisted id is deterministic, same as the app).
    roots2 = gal.load_roots(str(roots_file))
    assert next(iter(roots2)) == rid
    vpaths = gal.virtual_paths(
        [{"path": str(music / "sub" / "a.mp3")}, {"path": str(music / "b.mp3")}], roots)
    assert vpaths == [f"{rid}/sub/a.mp3", f"{rid}/b.mp3"]


def test_load_roots_empty_fails_cleanly(tmp_path):
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": []}))
    with pytest.raises(ValueError, match="no library roots"):
        gal.load_roots(str(roots_file))


# ── compatibility: load_recipe_json + Flask GET ──────────────────────────────


def test_compatible_with_load_recipe_json(tmp_path, monkeypatch, cli):
    import backend.embeddings.layout_evaluation as le

    _manifest, gallery_dir, _base = _gallery_fixture(tmp_path, monkeypatch, cli)
    m = json.loads((gallery_dir / "manifest.json").read_text())
    # load_completed lists exactly the safe ids; load_recipe_json validates and
    # returns each completed artifact (fingerprint/revision/input-fingerprint).
    assert sorted(le.load_completed(str(gallery_dir))) == sorted(m["recipes"])
    for safe in m["recipes"]:
        art = le.load_recipe_json(str(gallery_dir), safe)
        assert art is not None
        assert art["id"] == safe


def test_api_genre_allowlist_no_route_permissiveness(tmp_path, monkeypatch, cli):
    _manifest, _gallery_dir, base = _gallery_fixture(tmp_path, monkeypatch, cli)
    client = _make_app(base).test_client()

    # set=genre serves the genre gallery; its safe recipe ids resolve.
    r = client.get("/api/layout-evaluation?set=genre")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == [
        "genre_maest_logits", "genre_maest_logits_maest_643-357",
    ]
    r = client.get("/api/layout-evaluation/genre_maest_logits_maest_643-357?set=genre")
    assert r.status_code == 200
    assert r.get_json()["id"] == "genre_maest_logits_maest_643-357"
    assert len(r.get_json()["positions"]) == 6

    # No route permissiveness: a raw representation id (with '+') is still a 400 —
    # the genre set uses safe ids, it never widens the recipe allowlist.
    assert client.get(
        "/api/layout-evaluation/maest_logits+maest@643-357?set=genre"
    ).status_code == 400
    # An unknown set is still rejected.
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400
    # A genre-only recipe is not present in the root/base set (cross-set isolation).
    assert client.get("/api/layout-evaluation/genre_maest_logits").status_code == 404


def test_genre_directed_set_routes_five_safe_ids(tmp_path, monkeypatch, cli):
    from backend.routes.api_layout_evaluation import SET_SUBDIRS

    # The fixed set is on the allowlist; the old 13-recipe genre set is untouched.
    assert SET_SUBDIRS["genre-directed"] == "genre-directed"
    assert SET_SUBDIRS["genre"] == "genre"

    music = tmp_path / "music"
    music.mkdir()
    manifest, mpath = _write_manifest_file(tmp_path, music)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=6)  # mert_l19 for directed
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)], "external_roots": [], "temp_roots": []}))
    base = tmp_path / "layout-evaluation"
    gallery_dir = base / "genre-directed"

    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod
    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: (np.asarray(mat)[:, :2].astype(np.float32), None))
    monkeypatch.setattr(le_mod, "fit_opentsne",
                        lambda reduced, *, recipe, seed, n_jobs:
                        np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32))
    monkeypatch.setattr(le_mod, "orthogonal_align_to", lambda coords, paths, r, rp: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.5)

    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(tmp_path / "maps"), "--gallery-output", str(gallery_dir),
              "--roots-file", str(roots_file), "--recipe-set", "directed"])

    expected_safe = [gal.safe_gallery_id(rid) for rid in _scoring.DIRECTED_RECIPES]
    assert len(expected_safe) == len(set(expected_safe))  # all distinct
    m = json.loads((gallery_dir / "manifest.json").read_text())
    assert m["schema_version"] == 1 and m["status"] == "complete"
    assert sorted(m["recipes"]) == sorted(expected_safe)
    # Every safe id is route-allowlisted and the label is the readable one.
    import re
    ok = re.compile(r"^[A-Za-z0-9_-]+$")
    for rid, safe in zip(_scoring.DIRECTED_RECIPES, expected_safe):
        assert ok.match(safe)
        assert m["recipes"][safe]["label"] == _scoring.RECIPE_LABELS[rid]
        assert (gallery_dir / f"{safe}.json").exists()

    # The route resolves the five maps under set=genre-directed.
    client = _make_app(base).test_client()
    r = client.get("/api/layout-evaluation?set=genre-directed")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(expected_safe)
    for safe in expected_safe:
        r = client.get(f"/api/layout-evaluation/{safe}?set=genre-directed")
        assert r.status_code == 200
        assert r.get_json()["id"] == safe
        assert len(r.get_json()["positions"]) == len(manifest["tracks"])
    # Cross-set isolation: directed maps are not present under the old genre set.
    assert client.get(f"/api/layout-evaluation/{expected_safe[0]}?set=genre").status_code == 404
    assert client.get(f"/api/layout-evaluation/{expected_safe[0]}").status_code == 404
    # The allowlist still rejects unknown sets (no broad route was added).
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400


def test_genre_logits_ab_set_routes_two_safe_ids(tmp_path, monkeypatch, cli):
    from backend.routes.api_layout_evaluation import SET_SUBDIRS

    # The fixed set is on the allowlist; the older sets are untouched.
    assert SET_SUBDIRS["genre-logits-ab"] == "genre-logits-ab"
    assert SET_SUBDIRS["genre-directed"] == "genre-directed"
    assert SET_SUBDIRS["genre"] == "genre"

    music = tmp_path / "music"
    music.mkdir()
    manifest, mpath = _write_manifest_file(tmp_path, music)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=6)  # mert_l19 for genre_full
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": [str(music)], "external_roots": [], "temp_roots": []}))
    base = tmp_path / "layout-evaluation"
    gallery_dir = base / "genre-logits-ab"

    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod
    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: (np.asarray(mat)[:, :2].astype(np.float32), None))
    monkeypatch.setattr(le_mod, "fit_opentsne",
                        lambda reduced, *, recipe, seed, n_jobs:
                        np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32))
    monkeypatch.setattr(le_mod, "orthogonal_align_to", lambda coords, paths, r, rp: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.5)

    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(tmp_path / "maps"), "--gallery-output", str(gallery_dir),
              "--roots-file", str(roots_file), "--recipe-set", "logits-ab"])

    expected = ["genre_full", "genre_without_logits"]
    expected_safe = [gal.safe_gallery_id(rid) for rid in expected]
    assert len(expected_safe) == len(set(expected_safe))  # distinct
    m = json.loads((gallery_dir / "manifest.json").read_text())
    assert m["schema_version"] == 1 and m["status"] == "complete"
    assert sorted(m["recipes"]) == sorted(expected_safe)
    # genre_full gallery label is the readable source-of-truth label.
    import re
    ok = re.compile(r"^[A-Za-z0-9_-]+$")
    for rid, safe in zip(expected, expected_safe):
        assert ok.match(safe)
        assert m["recipes"][safe]["label"] == _scoring.RECIPE_LABELS[rid]
        assert (gallery_dir / f"{safe}.json").exists()

    # The route resolves the two maps under set=genre-logits-ab.
    client = _make_app(base).test_client()
    r = client.get("/api/layout-evaluation?set=genre-logits-ab")
    assert r.status_code == 200
    assert sorted(r.get_json()["manifest"]["recipes"]) == sorted(expected_safe)
    for safe in expected_safe:
        r = client.get(f"/api/layout-evaluation/{safe}?set=genre-logits-ab")
        assert r.status_code == 200
        assert r.get_json()["id"] == safe
        assert len(r.get_json()["positions"]) == len(manifest["tracks"])
    # Cross-set isolation: logits-ab maps are not under the old genre sets.
    assert client.get(f"/api/layout-evaluation/{expected_safe[0]}?set=genre").status_code == 404
    assert client.get(f"/api/layout-evaluation/{expected_safe[0]}?set=genre-directed").status_code == 404
    # The allowlist still rejects unknown sets (no broad route was added).
    assert client.get("/api/layout-evaluation?set=not-a-set").status_code == 400
    # No route-permissiveness widening: a raw representation id is still a 400.
    assert client.get(
        "/api/layout-evaluation/maest_logits+maest@643-357?set=genre-logits-ab"
    ).status_code == 400
