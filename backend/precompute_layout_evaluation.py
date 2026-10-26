#!/usr/bin/env python3
"""Generate the offline layout-evaluation artifacts from cached embeddings.

Read-only over the production caches: it gathers the composite matrix exactly as
the interactive projection does, but fits its own UMAP / openTSNE models directly
and writes JSON artifacts under ``--output``. It never writes to the production
layout cache, the t-SNE fit cache, or MP3 tags.

Usage::

    python -m backend.precompute_layout_evaluation [--output DIR] [--limit N]
        [--recipes id,id,... | --set SET] [--reference id] [--sources s1,s2]
        [--seed 42] [--threads -1] [--root /path] [--data-dir DIR]

The server serves these from ``<data-dir>/layout-evaluation``; point ``--output``
there (the default) so the browser eval can read them. ``TRACKSPACE_DATA_DIR``
selects the cache directory (default ``data/``). ``--set`` selects a built-in
gallery set's recipes, reference, and default output directory from
:data:`backend.embeddings.layout_evaluation.GALLERY_SETS`; externally generated
genre/folder sets are rejected with a pointer to their own CLI.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from collections import OrderedDict

from backend.cache import TrackCache
from backend.embeddings.feature_cache import (
    LIBRARY_MEMORY_BYTES,
    LIBRARY_MEMORY_ENTRIES,
    FeatureCache,
)
from backend.embeddings.layout_evaluation import (
    BASELINE_ID,
    GALLERY_SETS,
    run_generation,
)
from backend.embeddings.projection_config import (
    FROZEN_FEATURE_MASK,
    FROZEN_SOURCES,
    FROZEN_SOURCE_WEIGHTS,
    required_sources_for,
    single_lens_weights,
)
from backend.services.roots_registry import (
    add_root,
    library_root_ids,
    load_roots_state,
)
from backend.services.virtual_paths import list_mp3s, virtual_from_abs
from backend.tuning import FROZEN_FEATURES_BLEND

log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir() -> str:
    return (
        os.environ.get("TRACKSPACE_DATA_DIR")
        or os.path.join(PROJECT_DIR, "data")
    )


def _require_opentsne() -> None:
    if importlib.util.find_spec("openTSNE") is None:
        raise SystemExit(
            "openTSNE is required to generate the layout-evaluation maps "
            "(the baseline is an openTSNE fit). Install it: pip install opentsne"
        )


def _parse_sources(raw: str) -> tuple[str, ...]:
    names = [p.strip() for p in raw.split(",") if p.strip()]
    if not names:
        raise SystemExit("--sources must list at least one source")
    unknown = [n for n in names if n not in FROZEN_SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown --sources: {', '.join(unknown)} "
            f"(allowed: {', '.join(FROZEN_SOURCES)})"
        )
    named = set(names)
    return tuple(s for s in FROZEN_SOURCES if s in named)


def load_infos(
    roots: OrderedDict,
    track_cache: TrackCache,
    root_roles=None,
    *,
    scope: str = "library",
) -> tuple[list[dict], int]:
    """Path + fingerprint for every cached track under *roots*.

    Fingerprints come from the track cache only — nothing is recomputed, so the
    library is never re-scanned or re-fingerprinted. Tracks whose cache entry is
    missing are skipped (reported by the caller via the count).
    ``scope="library"`` keeps the historical library-roots-only selection;
    ``scope="all"`` includes external roots too, matching a 3-D / fit_scope=all
    projection. Returns ``(infos, n_mp3s)``.
    """
    if scope == "library":
        root_ids = library_root_ids(roots, root_roles or {})
    elif scope == "all":
        root_ids = list(roots)
    else:
        raise ValueError(f"unknown scope {scope!r} (expected 'library' or 'all')")
    abs_paths = list_mp3s("", roots, recursive=True, root_ids=root_ids)
    pairs: list[tuple[str, float]] = []
    for p in abs_paths:
        try:
            pairs.append((p, os.path.getmtime(p)))
        except OSError:
            continue
    cached = track_cache.get_many(pairs)
    infos: list[dict] = []
    for p, mtime in pairs:
        data = cached.get((p, mtime))
        fp = data.get("fingerprint") if data else None
        if fp:
            infos.append({"path": virtual_from_abs(p, roots), "fingerprint": fp})
    infos.sort(key=lambda t: t["path"])
    return infos, len(abs_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline layout-evaluation maps from cached embeddings."
    )
    parser.add_argument("--output", default=None, help="Artifact dir (default: <data-dir>/layout-evaluation)")
    parser.add_argument("--data-dir", default=None, help="Cache dir holding roots.json + audio_features.db (default: data/)")
    parser.add_argument("--root", default=None, help="Single library root (overrides roots.json)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tracks used (smoke tests)")
    recipe_source = parser.add_mutually_exclusive_group()
    recipe_source.add_argument("--recipes", default=None, help="Comma-separated recipe ids (default: all)")
    recipe_source.add_argument(
        "--set",
        dest="set_name",
        default=None,
        metavar="SET",
        help=(
            "A built-in gallery set from GALLERY_SETS (recipes, reference and "
            "default output dir); mutually exclusive with --recipes"
        ),
    )
    parser.add_argument("--reference", default=None, help="Recipe id the other maps align onto (default: the set reference, else baseline)")
    parser.add_argument("--sources", default=None, help="Comma-separated source blocks (default: production default lens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=-1, help="n_jobs for UMAP/openTSNE (default -1 = all cores)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Root is always INFO: DEBUG at the root lets umap/Numba emit megabytes of
    # JIT tracing. --verbose raises DEBUG only on this tool's own loggers, which
    # is bounded and useful; every other library stays at INFO.
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
    )
    if args.verbose:
        logging.getLogger("backend.embeddings.layout_evaluation").setLevel(logging.DEBUG)
        logging.getLogger("backend.precompute_layout_evaluation").setLevel(logging.DEBUG)
    _require_opentsne()

    data_dir = os.path.abspath(args.data_dir or _data_dir())

    gallery = None
    if args.set_name is not None:
        gallery = GALLERY_SETS.get(args.set_name)
        if gallery is None:
            builtin = ", ".join(
                name for name, entry in GALLERY_SETS.items() if entry.recipes is not None
            )
            raise SystemExit(
                f"unknown --set {args.set_name!r} (built-in sets: {builtin})"
            )
        if gallery.recipes is None:
            raise SystemExit(
                f"--set {args.set_name!r} is externally generated and cannot be "
                "built by this CLI (genre sets: `python -m "
                "backend.experiments.genre_space maps`; folder sets: `python -m "
                "backend.benchmark_production_projection`; see "
                "layout-evaluation.md); pass --recipes for this tool instead"
            )

    default_output = os.path.join(data_dir, "layout-evaluation")
    if gallery is not None and gallery.directory:
        default_output = os.path.join(default_output, gallery.directory)
    output_dir = os.path.abspath(args.output or default_output)

    sources = _parse_sources(args.sources) if args.sources else FROZEN_SOURCES
    weights = dict(FROZEN_SOURCE_WEIGHTS)
    # A per-source eval keeps the single source at weight 1 and the rest at 0,
    # matching the per-source production lenses.
    if args.sources:
        weights = single_lens_weights(sources)
    try:
        required_sources = required_sources_for(sources)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    recipe_ids = (
        [r.id for r in gallery.recipes]
        if gallery is not None
        else (
            [r.strip() for r in args.recipes.split(",") if r.strip()]
            if args.recipes
            else None
        )
    )
    builtin_reference = (
        gallery.recipes[0].id if gallery is not None and gallery.recipes else None
    )
    reference = args.reference or builtin_reference or BASELINE_ID

    # Resolve roots (persisted, or a single --root).
    roots: OrderedDict[str, str] = OrderedDict()
    root_roles: OrderedDict[str, str] = OrderedDict()
    state_path = os.path.join(data_dir, "roots.json")
    if args.root:
        add_root(os.path.abspath(args.root), roots)
    elif os.path.exists(state_path):
        load_roots_state(roots, state_path, default_root=None, root_roles=root_roles)
    else:
        raise SystemExit(
            f"no roots.json at {state_path} and no --root given; "
            "start the Trackspace server once to register a library root, "
            "or pass --root /path/to/music"
        )
    if not roots:
        raise SystemExit("no library roots configured (roots.json is empty)")

    feature_cache = FeatureCache(
        os.path.join(data_dir, "audio_features.db"),
        max_memory=LIBRARY_MEMORY_ENTRIES,
        max_memory_bytes=LIBRARY_MEMORY_BYTES,
    )
    track_cache = TrackCache(os.path.join(data_dir, "track_cache.db"))
    try:
        infos, n_mp3s = load_infos(roots, track_cache, root_roles)
        log.info(
            "roots: %d | cached fingerprints: %d of %d mp3s",
            len(roots), len(infos), n_mp3s,
        )
        if not infos:
            raise SystemExit(
                "no cached track fingerprints found under the library roots; "
                "run the analysis pipeline first"
            )

        def _progress(stage: str, done: int = 0, total: int = 0) -> None:
            print(f"[layout-eval] {stage} {done}/{total}")

        manifest = run_generation(
            feature_cache=feature_cache,
            infos=infos,
            output_dir=output_dir,
            sources=sources,
            source_weights=weights,
            required_sources=required_sources,
            feature_mask=FROZEN_FEATURE_MASK.copy(),
            features_blend=FROZEN_FEATURES_BLEND,
            seed=args.seed,
            n_jobs=args.threads,
            recipe_ids=recipe_ids,
            reference_id=reference,
            limit=args.limit,
            on_progress=_progress,
        )
    except ValueError as e:
        # Recipe/reference ids, source-mix vs. --sources conflicts: run_generation
        # validates all of these before any artifact write. Surface them as a
        # clean exit (no traceback), like the other CLI validation errors.
        raise SystemExit(str(e)) from None

    print()
    print(f"Artifacts: {output_dir}")
    print(f"Dataset fingerprint: {manifest.get('dataset_fingerprint')}")
    print(f"Sources: {', '.join(manifest.get('sources', []))}")
    print(f"Source weights: {manifest.get('source_weights')}")
    print(f"Tracks: {manifest.get('dataset', {}).get('count')}")
    print("Recipes:")
    for rid, r in (manifest.get("recipes") or {}).items():
        status = r.get("status")
        line = f"  {rid:<14} {status}"
        if status == "complete":
            t = r.get("timings", {})
            line += (
                f"  gather={t.get('gather', 0):6.2f}s "
                f"pca={t.get('pca', 0):6.2f}s "
                f"tsne={t.get('tsne', 0):6.2f}s "
                f"umap={t.get('umap', 0):6.2f}s "
                f"total={t.get('total', 0):6.2f}s"
            )
        elif r.get("reason"):
            line += f"  ({r['reason']})"
        print(line)


if __name__ == "__main__":
    main()
