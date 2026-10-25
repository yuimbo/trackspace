"""CLI tests for backend/experiments/genre_space.

Import-light modules are real; heavy calls (cohort/extract/evaluate, layout
fit, openTSNE) are mocked except one real end-to-end ``evaluate``. Verifies the
CLI import pulls no heavy libraries.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.experiments.genre_space import dataset as _dataset
from backend.experiments.genre_space import extractors as _extractors
from backend.experiments.genre_space import scoring as _scoring

DIMS = {"maest": 768, "maest_logits": 519, "maest_layer7": 2304,
        "mert_l19": 1024, "mert_l23": 1024}
MODEL_SOURCES = {"maest": ("maest", "maest_logits", "maest_layer7"),
                 "mert": ("mert_l19", "mert_l23")}
HEAVY_MODULES = {"librosa", "torch", "sklearn", "openTSNE", "umap", "transformers",
                 "backend.embeddings.layout_evaluation", "backend.embeddings.layout_fidelity"}


@pytest.fixture(scope="session")
def cli():
    import backend.experiments.genre_space.__main__ as m

    return m


# ── helpers ─────────────────────────────────────────────────────────────────


def _manifest_tracks(n: int = 6) -> list[dict]:
    splits = ("dev", "dev", "dev", "dev", "heldout", "heldout")
    return [
        {
            "fp": f"f{i}", "path": f"/music/{i}.mp3", "label": f"genre{i % 2}",
            "artist": "", "release": "", "mtime_ns": i, "size": i,
            "split": splits[i - 1], "windows": [{"start": 0.0, "duration": 30.0}],
        }
        for i in range(1, n + 1)
    ]


def _make_manifest(n: int = 6) -> dict:
    manifest = {
        "schema_version": 1, "dataset_id": "", "root": "/music",
        "track_db": "/tmp/track_cache.db", "seed": 42, "label_depth": 1,
        "limit": n, "tracks": _manifest_tracks(n),
    }
    manifest["dataset_id"] = _dataset.dataset_id(manifest)
    return manifest


def _write_manifest_file(tmp_path, manifest=None) -> tuple[dict, Path]:
    manifest = manifest or _make_manifest()
    p = Path(tmp_path) / "manifest.json"
    p.write_text(json.dumps(manifest))
    return manifest, p


def _write_artifacts(adir, model, manifest, rows=None, fp=None, sha=None,
                     dataset_id=None, zero_rows=None, omit_sha=False,
                     model_name=None) -> None:
    """Write a realistic <model>.npz/.json pair (fingerprints + pooled arrays)."""
    adir = Path(adir)
    adir.mkdir(parents=True, exist_ok=True)
    n = len(manifest["tracks"]) if rows is None else rows
    fps = [t["fp"] for t in manifest["tracks"]] if fp is None else fp
    rng = np.random.default_rng(0)
    arrays = {"fingerprints": np.array(fps, dtype=f"<U{max(len(f) for f in fps)}")}
    for name in MODEL_SOURCES[model]:
        a = rng.standard_normal((n, DIMS[name])).astype(np.float32)
        for i in (zero_rows or {}).get(name, ()):
            a[i] = 0.0
        arrays[name] = a
    npz = adir / f"{model}.npz"
    np.savez(npz, **arrays)
    artifact = {"npz": f"{model}.npz", "size_bytes": npz.stat().st_size}
    if not omit_sha:
        artifact["sha256"] = sha if sha is not None else hashlib.sha256(npz.read_bytes()).hexdigest()
    meta = {
        "dataset_id": dataset_id if dataset_id is not None else manifest["dataset_id"],
        "model": model_name if model_name is not None else model,
        "layers": {name: "test" for name in MODEL_SOURCES[model]},
        "coverage": {"total": len(manifest["tracks"]), "ok": n,
                     "failed": len(manifest["tracks"]) - n},
        "checkpoint": ({"repo": "m-a-p/MERT-v2-30s", "revision": "a" * 40}
                       if model == "mert"
                       else {"arch": "discogs-maest-30s-pw-129e-519l", "url": "https://example.invalid/x.pt"}),
        "artifact": artifact,
    }
    (adir / f"{model}.json").write_text(json.dumps(meta))


def _entry(with_per_query: bool) -> dict:
    entry = {"queries_scored": 1, "effective_k_min": 2, "effective_k_max": 2,
             "micro": 0.5, "macro_per_label": 0.5, "per_label": {"genre0": 0.5}}
    if with_per_query:
        entry["per_query"] = [{
            "path": "/music/5.mp3", "fingerprint": "f5", "label": "genre1",
            "effective_k": 2, "precision": 0.5,
            "neighbours": [
                {"path": "/music/1.mp3", "fingerprint": "f1", "label": "genre0", "cosine": 0.9},
                {"path": "/music/2.mp3", "fingerprint": "f2", "label": "genre0", "cosine": 0.8},
            ],
        }]
    return entry


def _fake_report(with_per_query: bool = True) -> dict:
    geo = {"dev": {"unrestricted": _entry(False), "exclude_related": _entry(False)},
           "heldout": {"unrestricted": _entry(with_per_query),
                       "exclude_related": _entry(with_per_query)}}
    return {
        "selection": "dev", "k": 10, "pca_dim": 50, "seed": 42,
        "dataset": {"tracks": 5, "intersection_rows": 5, "dropped_rows": 0,
                    "splits": {"dev": 3, "heldout": 2}},
        "sources": ["maest", "maest_logits", "maest_layer7", "mert_l19", "mert_l23"],
        "recipes": ["maest_logits"],
        "geometries": ["raw", "pca50"],
        "scores": {
            "maest_logits": {
                "family": "maest_logits", "sources": ["maest_logits"],
                "weights": {"maest_logits": 1.0},
                "geometries": {
                    "raw": dict(geo),
                    "pca50": {"components": 2, "fit_on": "dev rows only", "n_dev_rows": 3,
                              "n_features": 519, "dev": geo["dev"], "heldout": geo["heldout"]},
                },
            }
        },
        "selected": {
            "maest_logits": {
                "recipe": "maest_logits", "geometry": "raw",
                "selection_split": "dev", "selection_scoring": "unrestricted",
                "dev": {"unrestricted": {"macro_per_label": 0.5, "micro": 0.5},
                        "exclude_related": {"macro_per_label": 0.4, "micro": 0.4}},
                "heldout": {"unrestricted": {"macro_per_label": 0.55, "micro": 0.55},
                            "exclude_related": {"macro_per_label": 0.45, "micro": 0.45}},
            }
        },
        "selected_recipe_ids": ["maest_logits"],
        "pca": {"fit_on": "dev rows only", "cap": "min(pca_dim, n_dev_rows - 1, n_features - 1)",
                "n_dev_rows": 3},
        "notes": [],
    }


def _fake_extract_write(recorded=None, bad_sha=False, bad_dataset=False, raises=None):
    def fake(manifest, output, *, model="maest", revision=None, device="auto",
             allow_download=False, trust_remote_code=False, on_progress=None):
        if raises is not None:
            raise raises
        if recorded is not None:
            recorded.update(model=model, revision=revision, device=device,
                            allow_download=allow_download, trust_remote_code=trust_remote_code)
        _write_artifacts(output, model, manifest, rows=len(manifest["tracks"]),
                         sha=("0" * 64) if bad_sha else None,
                         dataset_id=("0" * 64) if bad_dataset else None)
        return {"model": model,
                "coverage": {"total": len(manifest["tracks"]),
                             "ok": len(manifest["tracks"]), "failed": 0}}

    return fake


# ── CLI basics ──────────────────────────────────────────────────────────────


def test_cli_import_loads_no_heavy_libs():
    # Importing the CLI must not pull librosa/torch/sklearn/models — the
    # layout-evaluation machinery is imported only inside cmd_maps.
    saved = {name: sys.modules.pop(name) for name in HEAVY_MODULES if name in sys.modules}
    try:
        import backend.experiments.genre_space.__main__ as m

        pulled = HEAVY_MODULES & set(sys.modules)
        assert not pulled, f"CLI import pulled heavy modules: {sorted(pulled)}"
        assert not hasattr(m, "le") and not hasattr(m, "lf")
    finally:
        sys.modules.update(saved)


def test_help_usage_no_model_work(cli, capsys):
    for argv in (["--help"], ["cohort", "--help"], ["extract", "--help"],
                 ["score", "--help"], ["maps", "--help"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 0
        assert "usage" in capsys.readouterr().out.lower()


def test_clean_cli_errors(cli, tmp_path):
    # argparse rejects bad choices, missing required args, and non-positive bounds.
    for argv in (
        ["extract", "--model", "clap", "--manifest", "m", "--output", "o"],
        ["extract", "--model", "maest", "--device", "gpu", "--manifest", "m", "--output", "o"],
        ["maps", "--manifest", "m", "--artifacts", "a", "--output", "o"],  # --recipes required
        ["maps", "--manifest", "m", "--artifacts", "a", "--output", "o",
         "--recipes", "maest_logits", "--budget", "0"],
        ["score", "--manifest", "m", "--artifacts", "a", "--output", "o", "--k", "0"],
        ["cohort"],  # --root/--output required
    ):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 2
    # CLI validation errors exit with a message, not a traceback.
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(tmp_path / "missing.json"),
                  "--artifacts", "a", "--output", str(tmp_path / "o")])
    assert "cannot read manifest" in str(ei.value)


# ── cohort ──────────────────────────────────────────────────────────────────


def test_cohort_serialization(cli, tmp_path, monkeypatch):
    def fake_create(root, track_db, limit=500, seed=42, label_depth=1):
        return _make_manifest(4)

    monkeypatch.setattr(_dataset, "create_cohort", fake_create)
    out = tmp_path / "manifest.json"
    cli.main(["cohort", "--root", "/music", "--output", str(out)])
    data = json.loads(out.read_text())
    assert len(data["tracks"]) == 4
    assert data["dataset_id"] == _dataset.dataset_id(data)
    assert _dataset.validate_manifest(data) == []
    with pytest.raises(SystemExit) as ei:
        cli.main(["cohort", "--root", "/music", "--output", str(out)])
    assert "refusing to overwrite" in str(ei.value)


# ── extract ─────────────────────────────────────────────────────────────────


def test_extract_flow(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path, _make_manifest(4))
    recorded = {}
    monkeypatch.setattr(_extractors, "extract", _fake_extract_write(recorded))
    out = tmp_path / "artifacts"
    cli.main(["extract", "--manifest", str(mpath), "--output", str(out), "--model", "maest"])
    assert (out / "maest.npz").exists() and (out / "maest.json").exists()
    assert recorded == {"model": "maest", "revision": None, "device": "auto",
                        "allow_download": False, "trust_remote_code": False}
    # the CLI verified provenance (dataset id) and the NPZ checksum from the metadata.
    meta = json.loads((out / "maest.json").read_text())
    assert meta["dataset_id"] == manifest["dataset_id"]
    with pytest.raises(SystemExit) as ei:
        cli.main(["extract", "--manifest", str(mpath), "--output", str(out), "--model", "maest"])
    assert "refusing to overwrite" in str(ei.value)


def test_extract_mert_requires_pin_and_trust(cli, tmp_path, monkeypatch):
    _, mpath = _write_manifest_file(tmp_path, _make_manifest(4))
    monkeypatch.setattr(_extractors, "extract", _fake_extract_write())
    base = ["extract", "--manifest", str(mpath), "--output", str(tmp_path / "o"),
            "--model", "mert"]
    for extra, expect in (
        ([], "40 hex"),
        (["--revision", "abc"], "40 hex"),
        (["--revision", "a" * 40], "trust-remote-code"),
    ):
        with pytest.raises(SystemExit) as ei:
            cli.main(base + extra)
        assert expect in str(ei.value)
    # Pinned revision + trust runs without --allow-download (cached code executes).
    cli.main(base + ["--revision", "a" * 40, "--trust-remote-code"])
    assert (tmp_path / "o" / "mert.npz").exists()


def test_extract_checksum_and_provenance_fail(cli, tmp_path, monkeypatch):
    _, mpath = _write_manifest_file(tmp_path, _make_manifest(4))
    monkeypatch.setattr(_extractors, "extract", _fake_extract_write(bad_sha=True))
    with pytest.raises(SystemExit) as ei:
        cli.main(["extract", "--manifest", str(mpath), "--output", str(tmp_path / "o1"),
                  "--model", "maest"])
    assert "checksum mismatch" in str(ei.value)
    monkeypatch.setattr(_extractors, "extract", _fake_extract_write(bad_dataset=True))
    with pytest.raises(SystemExit) as ei:
        cli.main(["extract", "--manifest", str(mpath), "--output", str(tmp_path / "o2"),
                  "--model", "maest"])
    assert "mismatched provenance" in str(ei.value)


def test_extract_model_unavailable_runtime_error_clean(cli, tmp_path, monkeypatch):
    # Model-unavailable failures raise RuntimeError in the extractor; the CLI
    # must surface them as a clean exit, not a traceback.
    _, mpath = _write_manifest_file(tmp_path, _make_manifest(4))
    monkeypatch.setattr(_extractors, "extract",
                        _fake_extract_write(raises=RuntimeError("MAEST checkpoint not in cache")))
    with pytest.raises(SystemExit) as ei:
        cli.main(["extract", "--manifest", str(mpath), "--output", str(tmp_path / "o"),
                  "--model", "maest"])
    assert "MAEST checkpoint not in cache" in str(ei.value)


# ── score ───────────────────────────────────────────────────────────────────


def test_score_alignment_across_sources(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    # mert rows are a *permutation* of the manifest order — fp-keyed alignment
    # must accept this and reorder back to manifest order.
    _write_artifacts(arts, "mert", manifest, rows=5, fp=["f5", "f1", "f3", "f2", "f4"])
    recorded = {}

    def fake_evaluate(vectors, tracks, *, k, pca_dim, seed, recipe_set=None, on_progress=None):
        recorded.update(vectors=vectors, tracks=tracks, k=k, pca_dim=pca_dim,
                        seed=seed, recipe_set=recipe_set, on_progress=on_progress)
        return _fake_report()

    monkeypatch.setattr(_scoring, "evaluate", fake_evaluate)
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--seed", "7"])
    # fp intersection across ALL loaded sources, manifest order, no zero fill.
    assert sorted(recorded["vectors"]) == ["maest", "maest_layer7", "maest_logits",
                                           "mert_l19", "mert_l23"]
    for name, mat in recorded["vectors"].items():
        assert mat.shape == (5, DIMS[name])
    assert [t["fp"] for t in recorded["tracks"]] == ["f1", "f2", "f3", "f4", "f5"]
    assert all(t["fingerprint"] == t["fp"] for t in recorded["tracks"])
    assert (recorded["k"], recorded["pca_dim"], recorded["seed"]) == (10, 50, 7)
    assert recorded["recipe_set"] == "baseline"  # score defaults to the baseline set
    assert callable(recorded["on_progress"])
    rep = json.loads((out / "report.json").read_text())
    assert rep["dataset_id"] == manifest["dataset_id"]
    assert rep["coverage"] == {"original_n": 6, "intersection": 5,
                               "excluded_fp": ["f6"], "full_cohort": False}
    assert rep["source_metadata"]["models"]["mert"]["present"] is True
    lines = (out / "neighbours.csv").read_text().strip().splitlines()
    assert lines[0] == ("recipe,geometry,mode,query_path,query_label,effective_k,"
                        "precision,rank,neighbour_path,neighbour_label,cosine")
    assert len(lines) == 1 + 2 * 2 * 1 * 2  # 2 geometries x 2 modes x 1 query x 2 neighbours


def test_score_drops_invalid_rows_no_zero_fill(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6, zero_rows={"maest": [1]})  # f2 zeroed
    _write_artifacts(arts, "mert", manifest, rows=6)
    recorded = {}
    monkeypatch.setattr(_scoring, "evaluate",
                        lambda vectors, tracks, *, k, pca_dim, seed, recipe_set=None,
                        on_progress=None: recorded.update(vectors=vectors, tracks=tracks)
                        or _fake_report())
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out)])
    assert [t["fp"] for t in recorded["tracks"]] == ["f1", "f3", "f4", "f5", "f6"]
    rep = json.loads((out / "report.json").read_text())
    assert rep["coverage"]["excluded_fp"] == ["f2"]
    assert rep["coverage"]["full_cohort"] is False


def test_score_integration_real_evaluate(cli, tmp_path, capsys):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)  # mert deliberately absent
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out)])
    rep = json.loads((out / "report.json").read_text())
    assert rep["dataset_id"] == manifest["dataset_id"]
    assert rep["coverage"]["full_cohort"] is True
    assert rep["report"]["selected_recipe_ids"]
    assert rep["source_metadata"]["models"]["mert"]["present"] is False
    text = capsys.readouterr().out
    assert "dev-selected recipe ids" in text
    assert "mert" in text and "NOT compared" in text


def test_score_directed_integration_real_evaluate(cli, tmp_path, capsys):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=6)  # supplies mert_l19
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--recipe-set", "directed"])
    rep = json.loads((out / "report.json").read_text())
    # The full directed preset is scored — never a silently partial set.
    assert rep["report"]["recipes"] == list(_scoring.DIRECTED_RECIPES)
    assert sorted(rep["report"]["sources"]) == ["maest", "maest_layer7", "maest_logits", "mert_l19"]
    # Every directed recipe was scored in both geometries (raw + pca50).
    for rid in _scoring.DIRECTED_RECIPES:
        assert set(rep["report"]["scores"][rid]["geometries"]) == {"raw", "pca50"}
    text = capsys.readouterr().out
    assert "dev-selected recipe ids" in text


def test_score_writes_csv_header_without_per_query(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    monkeypatch.setattr(_scoring, "evaluate",
                        lambda *a, **k: _fake_report(with_per_query=False))
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out)])
    lines = (out / "neighbours.csv").read_text().strip().splitlines()
    assert lines == ["recipe,geometry,mode,query_path,query_label,effective_k,"
                     "precision,rank,neighbour_path,neighbour_label,cosine"]
    rep = json.loads((out / "report.json").read_text())
    assert rep["provenance"]["neighbours_csv"] == "written"


def test_score_refuses_overwrite_before_evaluate(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    calls = []
    monkeypatch.setattr(_scoring, "evaluate",
                        lambda *a, **k: calls.append(1) or _fake_report())
    out = tmp_path / "report"
    cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out)])
    assert len(calls) == 1
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(out)])
    assert "refusing to overwrite" in str(ei.value)
    assert len(calls) == 1  # refused before the expensive evaluate


def test_score_mismatched_provenance_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6, dataset_id="0" * 64)
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "mismatched provenance" in str(ei.value)


def test_score_artifact_checksum_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6, sha="0" * 64)
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "checksum mismatch" in str(ei.value)


def test_score_missing_checksum_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6, omit_sha=True)
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "valid 64-hex artifact.sha256" in str(ei.value)


def test_score_model_name_mismatch_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6, model_name="mert")
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "declares model 'mert', expected 'maest'" in str(ei.value)


def test_score_changed_manifest_fails(cli, tmp_path, monkeypatch):
    manifest, mpath = _write_manifest_file(tmp_path)
    edited = json.loads(mpath.read_text())
    edited["tracks"][0]["label"] = "edited-label"
    mpath.write_text(json.dumps(edited))
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    monkeypatch.setattr(_scoring, "evaluate", lambda *a, **k: _fake_report())
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "edited or is stale" in str(ei.value)


def test_score_no_heldout_split_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=4, fp=["f1", "f2", "f3", "f4"])  # all dev
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "no heldout" in str(ei.value)


def test_score_minimum_three_fails(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=2, fp=["f1", "f2"])
    with pytest.raises(SystemExit) as ei:
        cli.main(["score", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "r")])
    assert "fewer than 3" in str(ei.value)


# ── maps ────────────────────────────────────────────────────────────────────


def test_maps_coords_explicit_recipes(cli, tmp_path, monkeypatch):
    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod

    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)  # maest family only
    calls = {}

    def fake_tsne(reduced, *, recipe, seed, n_jobs):
        calls["tsne"] = (recipe.id, recipe.pca_dim, recipe.method, recipe.tsne_n_iter,
                         dict(recipe.source_weights), seed, n_jobs)
        return (np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32))

    def fake_pca(mat, dim, seed):
        calls["pca"] = (dim, seed)
        return np.asarray(mat)[:, :2].astype(np.float32), None

    monkeypatch.setattr(le_mod, "pca_reduce", fake_pca)
    monkeypatch.setattr(le_mod, "fit_opentsne", fake_tsne)
    monkeypatch.setattr(le_mod, "orthogonal_align_to",
                        lambda coords, paths, ref_coords, ref_paths: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.42)
    out = tmp_path / "maps"
    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--recipes", "maest_logits,maest_logits+maest@643-357"])
    c1 = json.loads((out / "coords_maest_logits.json").read_text())
    c2 = json.loads((out / "coords_maest_logits+maest@643-357.json").read_text())
    assert c1["n"] == 6 and c1["dataset_id"] == manifest["dataset_id"]
    assert c1["fidelity"] == {"raw_to_2d": 0.42, "pca50_to_2d": 0.42}
    assert c1["weights"] == {"maest_logits": 1.0}
    assert c1["reference"] == "maest_logits" and c1["aligned_to"] is None
    assert c2["aligned_to"] == "maest_logits"
    assert set(c1["timings"]) == {"pca", "tsne", "total"}
    assert "all cohort rows" in c1["pca_fit_on"]
    sha = c1["source_metadata"]["models"]["maest"]["artifact"]["sha256"]
    assert isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    assert len(c1["positions"]) == 6 and {"path", "label", "x", "y"} <= set(c1["positions"][0])
    # real scoring.recipes: the anchor+maest mix renormalises 0.45/0.25 -> 643-357.
    assert c2["weights"] == pytest.approx({"maest_logits": 0.45 / 0.7, "maest": 0.25 / 0.7})
    assert calls["tsne"][0] == "maest_logits+maest@643-357"
    assert calls["tsne"][1] == 50 and calls["tsne"][2] == "tsne"
    assert calls["tsne"][4] == pytest.approx({"maest_logits": 0.45 / 0.7, "maest": 0.25 / 0.7})
    assert calls["pca"] == (50, 42)  # one PCA pass, on all rows
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(out), "--recipes", "maest_logits"])
    assert "refusing to overwrite" in str(ei.value)


def test_maps_duplicate_recipe_ids_rejected(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "m"), "--recipes", "maest_logits,maest_logits"])
    assert "duplicate" in str(ei.value)


def test_maps_refuses_before_fitting(cli, tmp_path, monkeypatch):
    from backend.embeddings import layout_evaluation as le_mod

    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    out = tmp_path / "maps"
    out.mkdir()
    (out / "coords_maest_logits.json").write_text("{}")  # stale output
    calls = []
    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: calls.append("pca") or (np.asarray(mat)[:, :2], None))
    monkeypatch.setattr(le_mod, "fit_opentsne", lambda *a, **k: calls.append("tsne"))
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(out), "--recipes", "maest_logits"])
    assert "refusing to overwrite" in str(ei.value)
    assert calls == []  # preflight refused every output before any fitting


def test_maps_unknown_recipe_id(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "m"), "--recipes", "bogus"])
    assert "unknown representation id 'bogus'" in str(ei.value)


def test_maps_directed_defaults_to_all_five(cli, tmp_path, monkeypatch):
    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod

    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=6)  # supplies mert_l19
    calls: dict = {}

    def fake_tsne(reduced, *, recipe, seed, n_jobs):
        calls.setdefault("tsne_ids", []).append(recipe.id)
        calls["last_label"] = recipe.label
        return np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: (np.asarray(mat)[:, :2].astype(np.float32), None))
    monkeypatch.setattr(le_mod, "fit_opentsne", fake_tsne)
    monkeypatch.setattr(le_mod, "orthogonal_align_to",
                        lambda coords, paths, ref_coords, ref_paths: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.42)
    out = tmp_path / "maps"
    # No --recipes: with --recipe-set directed it defaults to all five options.
    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--recipe-set", "directed"])
    expected = ["genre_control", "genre_structure", "genre_without_layer7",
                "genre_without_mert", "genre_full"]
    assert calls["tsne_ids"] == expected
    for rid in expected:
        c = json.loads((out / f"coords_{rid}.json").read_text())
        assert c["id"] == rid and c["n"] == 6
    # Directed maps carry the readable source-of-truth labels and exact weights.
    assert calls["last_label"] == _scoring.RECIPE_LABELS["genre_full"]
    full = json.loads((out / "coords_genre_full.json").read_text())
    assert full["label"] == _scoring.RECIPE_LABELS["genre_full"]
    assert full["weights"] == _scoring.DIRECTED_RECIPES["genre_full"]
    control = json.loads((out / "coords_genre_control.json").read_text())
    assert control["label"] == _scoring.RECIPE_LABELS["genre_control"]
    assert control["weights"] == _scoring.DIRECTED_RECIPES["genre_control"]


def test_maps_directed_missing_source_fails_clean(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)  # no mert -> no mert_l19
    out = tmp_path / "m"
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(out), "--recipe-set", "directed"])
    assert "directed recipe set requires every source" in str(ei.value)
    assert not out.exists()  # fails before creating the output dir


def test_maps_logits_ab_defaults_to_two(cli, tmp_path, monkeypatch):
    from backend.embeddings import layout_evaluation as le_mod
    from backend.embeddings import layout_fidelity as lf_mod

    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    _write_artifacts(arts, "mert", manifest, rows=6)  # supplies mert_l19
    calls: dict = {}

    def fake_tsne(reduced, *, recipe, seed, n_jobs):
        calls.setdefault("tsne_ids", []).append(recipe.id)
        calls.setdefault("labels", []).append(recipe.label)
        return np.random.default_rng(0).standard_normal((len(reduced), 2)).astype(np.float32)

    monkeypatch.setattr(le_mod, "pca_reduce",
                        lambda mat, dim, seed: (np.asarray(mat)[:, :2].astype(np.float32), None))
    monkeypatch.setattr(le_mod, "fit_opentsne", fake_tsne)
    monkeypatch.setattr(le_mod, "orthogonal_align_to",
                        lambda coords, paths, ref_coords, ref_paths: coords)
    monkeypatch.setattr(le_mod, "normalise_aspect", lambda coords: coords)
    monkeypatch.setattr(lf_mod, "neighbourhood_recall", lambda hi, lo, *, k, seed: 0.42)
    out = tmp_path / "maps"
    # No --recipes: with --recipe-set logits-ab it defaults to exactly the two options.
    cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
              "--output", str(out), "--recipe-set", "logits-ab"])
    expected = ["genre_full", "genre_without_logits"]
    assert calls["tsne_ids"] == expected
    for rid in expected:
        c = json.loads((out / f"coords_{rid}.json").read_text())
        assert c["id"] == rid and c["n"] == 6
        assert c["label"] == _scoring.RECIPE_LABELS[rid]
    # genre_full reuses the exact directed weights; without_logits is the derived ablation.
    assert json.loads((out / "coords_genre_full.json").read_text())["weights"] == \
        _scoring.DIRECTED_RECIPES["genre_full"]
    assert json.loads((out / "coords_genre_without_logits.json").read_text())["weights"] == \
        pytest.approx({"maest": 1 / 3, "maest_layer7": 5 / 12, "mert_l19": 0.25})
    assert calls["labels"] == [_scoring.RECIPE_LABELS[rid] for rid in expected]


def test_maps_logits_ab_missing_source_fails_clean(cli, tmp_path):
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)  # no mert -> no mert_l19
    out = tmp_path / "m"
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(out), "--recipe-set", "logits-ab"])
    assert "logits-ab recipe set requires every source" in str(ei.value)
    assert not out.exists()  # fails before creating the output dir


def test_maps_baseline_still_requires_explicit_recipes(cli, tmp_path):
    # --recipe-set baseline (the default) with no --recipes is rejected at the
    # argparse level (clean usage error, exit 2), exactly as before directed.
    manifest, mpath = _write_manifest_file(tmp_path)
    arts = tmp_path / "arts"
    _write_artifacts(arts, "maest", manifest, rows=6)
    with pytest.raises(SystemExit) as ei:
        cli.main(["maps", "--manifest", str(mpath), "--artifacts", str(arts),
                  "--output", str(tmp_path / "m")])
    assert ei.value.code == 2
