"""Genre-space experiment CLI: cohort → extract → score → maps (offline, isolated).

Run as ``python -m backend.experiments.genre_space <subcommand>``. A thin driver
over the sibling ``dataset`` / ``extractors`` / ``scoring`` modules plus the
production layout-evaluation machinery (fit/align/normalise, neighbourhood
fidelity). Never writes production data: the track cache is read only through
``dataset.create_cohort`` and every artifact goes to CLI-chosen paths, refusing
to overwrite existing manifest/report/model/coords files.

Artifact contract (enforced, never guessed): ``<model>.json`` must declare
``model`` matching the file, ``layers`` exactly the model's known sources
(maest/maest_logits/maest_layer7, mert_l19/mert_l23), a valid 64-hex
``artifact.sha256`` matching the NPZ, and the manifest's dataset id. The NPZ is
fingerprint-keyed (non-empty 1-D unicode ``fingerprints``) so rows may be
permuted; alignment is by fingerprint intersection across every loaded source,
never zero-filled. neighbours.csv is always written (header, plus the scorer's
heldout per-query rows when it emits them). ``score``/``maps`` take a
``--recipe-set`` (``baseline`` default | ``directed``): baseline ids come from
``scoring.recipes(recipe_set='baseline')`` and ``maps --recipes`` is explicit;
``--recipe-set directed`` maps all five contrastive options when ``--recipes``
is omitted (baseline still requires an explicit list).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

MIN_USABLE = 3
FIDELITY_K = 10
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
KNOWN_SOURCES = {"maest": ("maest", "maest_logits", "maest_layer7"),
                 "mert": ("mert_l19", "mert_l23")}
EXPECTED_DIMS = {"maest": 768, "maest_logits": 519, "maest_layer7": 2304,
                 "mert_l19": 1024, "mert_l23": 1024}


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return n


def _iso_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _refuse_existing(paths: list[str]) -> None:
    for p in paths:
        if os.path.exists(p):
            raise ValueError(f"refusing to overwrite existing file: {p}")


def _npz_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_dataset_id(manifest: dict, dataset) -> None:
    problems = dataset.validate_manifest(manifest)  # returns list[str], raises nothing
    if problems:
        raise ValueError("manifest invalid: " + "; ".join(str(p) for p in problems))
    if dataset.dataset_id(manifest) != manifest.get("dataset_id"):
        raise ValueError(
            "manifest dataset_id does not match the canonical manifest ID hash "
            "(the manifest was edited or is stale); re-run 'cohort' to regenerate"
        )


def _load_manifest(path: str, dataset) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read manifest {path}: {e}") from None
    if not isinstance(manifest, dict) or "tracks" not in manifest:
        raise ValueError(f"{path} is not a genre-space manifest (no 'tracks')")
    _check_dataset_id(manifest, dataset)
    return manifest


def _read_model_json(model_dir: str, model: str, dataset_id: str) -> dict:
    """Load ``<model>.json``; verify name, dataset id and NPZ checksum (all mandatory)."""
    json_path = os.path.join(model_dir, f"{model}.json")
    npz_path = os.path.join(model_dir, f"{model}.npz")
    if not (os.path.exists(json_path) and os.path.exists(npz_path)):
        raise ValueError(f"incomplete artifact pair for {model!r} in {model_dir} (need {model}.npz/.json)")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {json_path}: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"{json_path} is not a JSON object")
    if meta.get("dataset_id") != dataset_id:
        raise ValueError(
            f"{model}.json dataset id {meta.get('dataset_id')!r} does not match "
            f"the manifest's {dataset_id!r} (mismatched provenance)"
        )
    if meta.get("model") != model:
        raise ValueError(f"{model}.json declares model {meta.get('model')!r}, expected {model!r}")
    artifact = meta.get("artifact")
    sha = artifact.get("sha256") if isinstance(artifact, dict) else None
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        raise ValueError(f"{model}.json must declare a valid 64-hex artifact.sha256")
    if _npz_sha256(npz_path) != sha:
        raise ValueError(f"{model}.npz checksum mismatch (metadata artifact.sha256)")
    return meta


def _validate_source(name: str, arr: np.ndarray, fp: list[str]) -> tuple[np.ndarray, list[str]]:
    """Strict dims + finite/nonzero; failed (invalid) rows dropped, never zero-filled."""
    expected = EXPECTED_DIMS.get(name)
    if expected is None:
        raise ValueError(f"unexpected source {name!r} (known: {', '.join(sorted(EXPECTED_DIMS))})")
    if arr.ndim != 2 or arr.shape[1] != expected:
        raise ValueError(f"source {name!r}: shape {arr.shape}, expected 2-D with {expected} columns")
    if len(arr) != len(fp):
        raise ValueError(f"source {name!r}: {len(arr)} rows != {len(fp)} fingerprints")
    ok = np.isfinite(arr).all(axis=1) & np.any(arr != 0, axis=1)
    if not ok.all():
        arr, fp = arr[ok], [f for f, keep in zip(fp, ok) if keep]
    if len(arr) == 0:
        raise ValueError(f"source {name!r}: no valid (finite, non-zero) rows")
    return arr, fp


def _load_sources(artifacts_dir: str, manifest: dict) -> tuple[dict[str, dict], dict]:
    """Per-source vectors ``{name: {"fp", "matrix"}}`` + metadata from complete pairs."""
    dataset_id = manifest["dataset_id"]
    manifest_fp = [str(t["fp"]) for t in manifest["tracks"]]
    vectors: dict[str, dict] = {}
    meta: dict[str, Any] = {"models": {}, "sources": {}}
    for model in ("maest", "mert"):
        npz_path, json_path = (os.path.join(artifacts_dir, f"{model}.npz"),
                               os.path.join(artifacts_dir, f"{model}.json"))
        has_npz, has_json = os.path.exists(npz_path), os.path.exists(json_path)
        if not has_npz and not has_json:
            meta["models"][model] = {"present": False}
            continue
        if has_npz != has_json:
            raise ValueError(f"orphan artifact for {model!r}: {model}.npz without {model}.json")
        meta_json = _read_model_json(artifacts_dir, model, dataset_id)
        layers = meta_json.get("layers")
        expected = set(KNOWN_SOURCES[model])
        if not isinstance(layers, dict) or set(layers) != expected:
            raise ValueError(
                f"{model}.json layers must be exactly {', '.join(sorted(expected))}, "
                f"got {sorted(layers) if isinstance(layers, dict) else layers!r}"
            )
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                arrays = {k: np.asarray(z[k]) for k in z.files}
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot load {npz_path}: {e}") from None
        fp_arr = arrays.get("fingerprints")
        if fp_arr is None:
            raise ValueError(f"{model}.npz is missing the 'fingerprints' array")
        fp_arr = np.asarray(fp_arr)
        if fp_arr.ndim != 1 or fp_arr.size == 0 or fp_arr.dtype.kind != "U":
            raise ValueError(f"{model}.npz 'fingerprints' must be a non-empty 1-D unicode array")
        fp = [str(f) for f in fp_arr.tolist()]
        if len(set(fp)) != len(fp):
            raise ValueError(f"{model}.npz has duplicate fingerprints")
        if not set(fp) <= set(manifest_fp):
            raise ValueError(f"{model}.npz fingerprints reference tracks absent from the manifest")
        names = list(KNOWN_SOURCES[model])
        missing = [n for n in names if n not in arrays]
        if missing:
            raise ValueError(f"{model}.npz is missing source array(s): {', '.join(missing)}")
        meta["models"][model] = {"present": True, "coverage": meta_json.get("coverage"),
                                 "artifact": meta_json.get("artifact"),
                                 "checkpoint": meta_json.get("checkpoint"), "layers": names}
        for name in names:
            arr, source_fp = _validate_source(name, np.asarray(arrays[name], dtype=np.float32), fp)
            vectors[name] = {"fp": source_fp, "matrix": arr}
            meta["sources"][name] = {"model": model, "file": f"{model}.npz",
                                     "rows": int(arr.shape[0]), "dim": int(arr.shape[1])}
    if not vectors:
        raise ValueError(f"no model artifacts (maest/mert .npz+.json) found in {artifacts_dir}")
    return vectors, meta


def _align_sources(vectors: dict, manifest: dict) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    """Fingerprint intersection across ALL loaded sources, manifest order, no zero fill."""
    common = set.intersection(*(set(v["fp"]) for v in vectors.values()))
    ordered = [f for f in (str(t["fp"]) for t in manifest["tracks"]) if f in common]
    if len(ordered) < MIN_USABLE:
        raise ValueError(
            f"fewer than {MIN_USABLE} tracks have vectors in every loaded source "
            f"(got {len(ordered)}); nothing to score/map fairly"
        )
    aligned = {name: v["matrix"][[v["fp"].index(f) for f in ordered]] for name, v in vectors.items()}
    tracks = [t for t in manifest["tracks"] if str(t["fp"]) in common]
    return aligned, tracks, {
        "original_n": len(manifest["tracks"]),
        "intersection": len(ordered),
        "excluded_fp": [str(t["fp"]) for t in manifest["tracks"] if str(t["fp"]) not in common],
        "full_cohort": len(ordered) == len(manifest["tracks"]),
    }


def _flatten_neighbours(report: dict) -> list[tuple]:
    """Heldout per-query neighbours → CSV rows (recipe, geometry, mode, query, neighbours)."""
    rows: list[tuple] = []
    for rid, s in report.get("scores", {}).items():
        for geom, g in (s.get("geometries") or {}).items():
            if not isinstance(g, dict):
                continue
            for mode in ("unrestricted", "exclude_related"):
                entry = g.get("heldout", {}).get(mode)
                per_query = entry.get("per_query") if isinstance(entry, dict) else None
                if not isinstance(per_query, list):
                    continue
                for q in per_query:
                    for i, nb in enumerate(q.get("neighbours") or [], 1):
                        rows.append((rid, geom, mode, q.get("path"), q.get("label"),
                                     q.get("effective_k"), q.get("precision"),
                                     i, nb.get("path"), nb.get("label"), nb.get("cosine")))
    return rows


def cmd_cohort(args) -> None:
    from . import dataset

    out = os.path.abspath(args.output)
    _refuse_existing([out])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    manifest = dataset.create_cohort(Path(args.root), Path(args.track_db),
                                     limit=args.limit, seed=args.seed, label_depth=args.label_depth)
    if not isinstance(manifest, dict) or "tracks" not in manifest:
        raise ValueError("dataset.create_cohort did not return a manifest dict")
    _check_dataset_id(manifest, dataset)
    _write_json(out, manifest)
    print(f"[cohort] {len(manifest['tracks'])} tracks -> {out}", flush=True)


def cmd_extract(args) -> None:
    from . import dataset, extractors

    manifest = _load_manifest(args.manifest, dataset)
    if args.model == "mert":
        if not (args.revision and HEX40.fullmatch(args.revision)):
            raise ValueError("--model mert requires --revision pinned to exactly 40 hex characters")
        if not args.trust_remote_code:
            raise ValueError("--model mert requires --trust-remote-code (runs pinned repo-specific "
                             "modeling code; --allow-download separately gates fetching weights)")
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    npz_path, json_path = os.path.join(out, f"{args.model}.npz"), os.path.join(out, f"{args.model}.json")
    _refuse_existing([npz_path, json_path])

    def _progress(msg: str) -> None:
        print(f"[extract {args.model}] {msg}", flush=True)

    meta = extractors.extract(manifest, out, model=args.model, revision=args.revision,
                              device=args.device, allow_download=args.allow_download,
                              trust_remote_code=args.trust_remote_code, on_progress=_progress)
    _read_model_json(out, args.model, manifest["dataset_id"])  # name, provenance, NPZ checksum
    try:
        with np.load(npz_path, allow_pickle=False):
            pass
    except (OSError, ValueError) as e:
        raise ValueError(f"extracted {npz_path} does not load: {e}") from None
    coverage = meta.get("coverage") if isinstance(meta, dict) else None
    if isinstance(coverage, dict):
        print(f"[extract {args.model}] {coverage.get('ok')}/{coverage.get('total')} tracks ok; "
              f"{coverage.get('failed', 0)} failed", flush=True)
    print(f"[extract {args.model}] wrote {npz_path} (dataset {manifest['dataset_id'][:8]}...)", flush=True)


def cmd_score(args) -> None:
    from . import dataset, scoring

    manifest = _load_manifest(args.manifest, dataset)
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    report_path, csv_path = os.path.join(out, "report.json"), os.path.join(out, "neighbours.csv")
    _refuse_existing([report_path, csv_path])  # before the expensive evaluate
    print("[score] loading artifacts...", flush=True)
    vectors, meta = _load_sources(args.artifacts, manifest)
    absent = [m for m, d in meta["models"].items() if not d.get("present")]
    if absent:
        print(f"[score] note: model artifact(s) absent - {', '.join(absent)} sources are NOT compared", flush=True)
    aligned, tracks, coverage = _align_sources(vectors, manifest)
    print(f"[score] aligned {coverage['intersection']} tracks across {len(aligned)} sources", flush=True)
    missing = [s for s in ("dev", "heldout") if s not in {t.get("split") for t in tracks}]
    if missing:
        raise ValueError(f"aligned cohort has no {', '.join(missing)} tracks; cannot score a dev/heldout split")
    score_tracks = [dict(t, fingerprint=t["fp"]) for t in tracks]  # scorer reads fingerprint
    print("[score] evaluating...", flush=True)
    # Restrict to the sources the chosen recipe set actually uses so an unused
    # source (e.g. mert_l23 in the directed preset) is not rejected — evaluate is
    # deliberately strict about unused sources for the full baseline sweep.
    used_sources = set().union(*(scoring.recipes(set(aligned), recipe_set=args.recipe_set).values()))
    eval_vectors = {s: aligned[s] for s in used_sources}
    report = scoring.evaluate(
        eval_vectors, score_tracks, k=args.k, pca_dim=args.pca_dim, seed=args.seed,
        recipe_set=args.recipe_set,
        on_progress=lambda msg: print(f"[score] {msg}", flush=True),
    )
    if not isinstance(report, dict):
        raise ValueError("scoring.evaluate did not return a report dict")
    rows = _flatten_neighbours(report)
    _write_json(report_path, {
        "dataset_id": manifest["dataset_id"],
        "coverage": coverage,
        "source_metadata": meta,
        "provenance": {
            "command": "score",
            "manifest": os.path.abspath(args.manifest),
            "artifacts": os.path.abspath(args.artifacts),
            "k": args.k, "pca_dim": args.pca_dim, "seed": args.seed,
            "created_at": _iso_now(),
            "neighbours_csv": "written",
        },
        "report": report,
    })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recipe", "geometry", "mode", "query_path", "query_label", "effective_k",
                    "precision", "rank", "neighbour_path", "neighbour_label", "cosine"])
        w.writerows(rows)
    rids = report.get("selected_recipe_ids")
    if isinstance(rids, list) and rids:
        print(f"[score] dev-selected recipe ids: {', '.join(rids)} - pass to 'maps --recipes'", flush=True)
        for fam in sorted(report.get("selected") or {}):
            s = report["selected"][fam]
            macro = (s.get("heldout") or {}).get("unrestricted", {}).get("macro_per_label")
            print(f"[score]   {s.get('recipe')} ({s.get('geometry')}): heldout macro_per_label={macro}", flush=True)
    if not coverage["full_cohort"]:
        print(f"[score] note: report covers {coverage['intersection']}/{coverage['original_n']} tracks - not the full cohort", flush=True)
    print(f"[score] wrote {report_path} ({coverage['intersection']} tracks, {len(rows)} neighbour rows)", flush=True)


def _weights_for(rid: str, available: set, recipe_set: str = "baseline") -> dict:
    from . import scoring

    raw = scoring.recipes(available, recipe_set=recipe_set)
    if rid not in raw:
        raise ValueError(f"unknown representation id {rid!r} (available: {', '.join(sorted(raw)) or 'none'})")
    return dict(raw[rid])


def cmd_maps(args) -> None:
    from . import dataset, scoring
    from backend.embeddings import layout_evaluation as le
    from backend.embeddings import layout_fidelity as lf

    manifest = _load_manifest(args.manifest, dataset)
    print("[maps] loading artifacts...", flush=True)
    vectors, meta = _load_sources(args.artifacts, manifest)
    aligned, tracks, coverage = _align_sources(vectors, manifest)
    paths = [str(t["path"]) for t in tracks]
    labels = [str(t.get("label", "")) for t in tracks]
    if args.recipes:
        rids = [r.strip() for r in args.recipes.split(",") if r.strip()]
    else:
        # A named --recipe-set (directed, logits-ab, …) defaults to that set's
        # recipes; the baseline set still requires an explicit --recipes list
        # (enforced in main() before any fitting).
        rids = list(scoring.recipes(set(aligned), recipe_set=args.recipe_set))
    if not rids:
        raise ValueError("--recipes must list at least one representation id")
    dups = sorted({r for r in rids if rids.count(r) > 1})
    if dups:
        raise ValueError(f"duplicate --recipes id(s): {', '.join(dups)}")
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    # Preflight every id's weights + output before any fit or write.
    plans: list[tuple[str, dict, np.ndarray, str]] = []
    for rid in rids:
        weights = _weights_for(rid, set(aligned), recipe_set=args.recipe_set)
        mat = np.asarray(scoring.compose(aligned, weights), dtype=np.float32)
        if mat.ndim != 2 or len(mat) != len(paths):
            raise ValueError(f"recipe {rid!r}: compose returned shape {mat.shape} for {len(paths)} tracks")
        plans.append((rid, weights, mat, os.path.join(out, f"coords_{rid}.json")))
    _refuse_existing([plan[3] for plan in plans])
    # Optional gallery: preflight roots/virtual paths + every gallery file before
    # any fit or write (a partial previous gallery is never extended).
    gallery = None
    if args.gallery_output:
        from . import gallery as gal
        gallery = gal.Gallery(
            tracks, meta, rids, seed=args.seed, budget=args.budget, n_jobs=args.n_jobs,
            gallery_dir=args.gallery_output, roots_file=args.roots_file,
        )
        gallery.preflight()
    ref_id = rids[0]
    ref_coords: np.ndarray | None = None
    for rid, weights, mat, out_path in plans:
        print(f"[maps] {rid}: pca...", flush=True)
        t0 = time.perf_counter()
        reduced, _ = le.pca_reduce(mat, 50, args.seed)  # PCA on all rows (transductive map)
        pca_s = time.perf_counter() - t0
        label = scoring.RECIPE_LABELS.get(rid, f"PCA50 -> openTSNE ({rid})")
        recipe = le.Recipe(id=rid, label=label, pca_dim=50, method="tsne",
                           tsne_n_iter=args.budget, tsne_early_exaggeration_iter=250,
                           source_weights=tuple(sorted(weights.items())))
        print(f"[maps] {rid}: tsne...", flush=True)
        t0 = time.perf_counter()
        coords = le.fit_opentsne(reduced, recipe=recipe, seed=args.seed, n_jobs=args.n_jobs)
        tsne_s = time.perf_counter() - t0
        if not np.isfinite(coords).all():
            raise ValueError(f"recipe {rid!r} produced non-finite coordinates")
        aligned_to = None if rid == ref_id else ref_id
        if ref_coords is None:
            ref_coords = coords
        else:
            coords = le.orthogonal_align_to(coords, paths, ref_coords, paths)
        normed = le.normalise_aspect(coords)
        fidelity = {
            "raw_to_2d": lf.neighbourhood_recall(mat, coords, k=FIDELITY_K, seed=args.seed),
            "pca50_to_2d": lf.neighbourhood_recall(reduced, coords, k=FIDELITY_K, seed=args.seed),
        }
        timings = {"pca": round(pca_s, 4), "tsne": round(tsne_s, 4),
                   "total": round(pca_s + tsne_s, 4)}
        if gallery is not None:
            gallery.add(rid, weights=weights, composed=mat, coords=normed,
                        aligned_to_raw=aligned_to, timings=timings, fidelity=fidelity,
                        label=recipe.label)
        _write_json(out_path, {
            "id": rid, "label": label, "dataset_id": manifest["dataset_id"],
            "n": len(paths), "seed": args.seed, "budget": args.budget, "n_jobs": args.n_jobs,
            "reference": ref_id, "aligned_to": aligned_to,
            "weights": weights,
            "recipe": {"pca_dim": 50, "method": "tsne", "tsne_n_iter": args.budget,
                       "tsne_early_exaggeration_iter": 250},
            "timings": timings,
            "pca_fit_on": "all cohort rows (transductive map; score's PCA fits dev rows only)",
            "fidelity": fidelity,
            "coverage": coverage,
            "source_metadata": meta,
            "positions": [
                {"path": paths[i], "label": labels[i],
                 "x": float(normed[i, 0]), "y": float(normed[i, 1])}
                for i in range(len(paths))
            ],
        })
        print(f"[maps] {rid}: {len(paths)} tracks, raw->2D {fidelity['raw_to_2d']:.3f}, "
              f"PCA50->2D {fidelity['pca50_to_2d']:.3f} -> {out_path}", flush=True)

    if gallery is not None:
        gallery.write()
        print(f"[maps] gallery: wrote manifest + {len(gallery.safe_ids)} recipe(s) to "
              f"{gallery.gallery_dir}", flush=True)
        set_name = os.path.basename(gallery.gallery_dir)
        print("[maps] open the blind-eval gallery: "
              f"http://127.0.0.1:5111/?layout_eval=1&set={set_name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    from . import scoring

    parser = argparse.ArgumentParser(
        prog="python -m backend.experiments.genre_space",
        description="Offline genre-space pipeline: cohort -> extract -> score -> maps (isolated).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help, func, *opts):
        p = sub.add_parser(name, help=help)
        for flags, kw in opts:
            p.add_argument(*flags, **kw)
        p.set_defaults(func=func)

    add("cohort", "build a manifest cohort from the track cache (read-only)", cmd_cohort,
        (("--root",), dict(required=True, help="library root to scan")),
        (("--track-db",), dict(default="data/track_cache.db", help="track cache db (default data/track_cache.db)")),
        (("--output",), dict(required=True, help="manifest JSON file (refused if it exists)")),
        (("--limit",), dict(type=int, default=500)),
        (("--seed",), dict(type=int, default=42)),
        (("--label-depth",), dict(type=int, default=1)))
    add("extract", "extract one model over the cohort (writes <model>.npz/.json)", cmd_extract,
        (("--manifest",), dict(required=True)),
        (("--output",), dict(required=True, help="artifact dir (refuses to overwrite <model>.npz/.json)")),
        (("--model",), dict(choices=("maest", "mert"), required=True)),
        (("--revision",), dict(default=None, help="model revision; --model mert requires a 40-hex pin")),
        (("--device",), dict(choices=("auto", "cpu", "mps", "cuda"), default="auto")),
        (("--allow-download",), dict(action="store_true", help="allow model-weight download (off by default)")),
        (("--trust-remote-code",), dict(action="store_true", help="run pinned repo-specific modeling code (required for mert)")))
    add("score", "align the cohort across all artifacts and score it", cmd_score,
        (("--manifest",), dict(required=True)),
        (("--artifacts",), dict(required=True, help="dir with maest/mert .npz+.json pairs")),
        (("--output",), dict(required=True, help="report dir (refuses to overwrite report.json/neighbours.csv)")),
        (("--k",), dict(type=_positive_int, default=10)),
        (("--pca-dim",), dict(type=_positive_int, default=50)),
        (("--recipe-set",), dict(choices=scoring.RECIPE_SETS, default="baseline",
                                 help="recipe set to score (baseline 13-sweep, directed 5-option preset, or logits-ab 2-option ablation)")),
        (("--seed",), dict(type=int, default=42)))
    add("maps", "fit one PCA50->openTSNE map per --recipes representation id", cmd_maps,
        (("--manifest",), dict(required=True)),
        (("--artifacts",), dict(required=True)),
        (("--output",), dict(required=True, help="coords dir (refuses to overwrite coords_<rid>.json)")),
        (("--recipes",), dict(default=None, help="comma-separated representation ids from scoring.recipes. Required for --recipe-set baseline; with a named --recipe-set it may be omitted to fit that set's recipes.")),
        (("--recipe-set",), dict(choices=scoring.RECIPE_SETS, default="baseline",
                                 help="recipe set to map (baseline 13-sweep, directed 5-option preset, or logits-ab 2-option ablation)")),
        (("--seed",), dict(type=int, default=42)),
        (("--budget",), dict(type=_positive_int, default=500, help="openTSNE n_iter (early phase fixed at 250)")),
        (("--n-jobs",), dict(type=int, default=-1)),
        (("--gallery-output",), dict(default=None, help="optional: also write a schema-1 gallery (manifest + <safeid>.json) for the browser blind-eval UI")),
        (("--roots-file",), dict(default="data/roots.json", help="roots.json for absolute->virtual path mapping (default data/roots.json)")))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # baseline maps must still name --recipes explicitly; only the directed
    # preset may default to all five (resolved in cmd_maps). Enforce at the
    # argparse level so the omission is a clean usage error.
    if args.command == "maps" and not args.recipes and args.recipe_set == "baseline":
        parser.error("--recipes is required when --recipe-set is 'baseline'")
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError) as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
