#!/usr/bin/env python3
"""Bounded benchmark of the *production* projection fit.

The existing ``benchmark_layout_scaling`` intentionally refuses libraries at or
above 10k tracks and fits the evaluation recipes. Production is different: it
uses the ``genre_full`` recipe, folder-semantic weighting, and openTSNE's
multilevel sample-init chain, so at the live ~15.6k-track scope its 2-D final
level resolves to FFT while 3-D stays Barnes-Hut. This tool measures that exact
path — same recipe registry, same composite gather / semantic weighting
(``prepare_projection_input``), same production PCA width and multilevel
schedule — without touching MP3 tags, the production layout cache, the
in-flight t-SNE fit cache, or the gallery artifacts. The JSON checkpoint under
``--output`` is the intended application write. The optional
``--gallery-output`` writes the paired 2-D/3-D folder on/off comparison gallery
described below. The cache constructors may
create schema/metadata/WAL and the track cache may warm its in-process LRU, so
do not describe the run as filesystem-read-only; no cache rows, tags, layouts,
or in-flight fits are written.

What is measured (all wall clock, seconds):

* ``prepare`` — the timed ``prepare_projection_input`` for that case's basis
  mode. The first prepared mode is cold (composite gather + its basis); later
  modes are warm gathers that still build their own basis.
* ``weighted_prepare`` — the timed full-basis
  ``prepare_projection_input(scale_folders=True, folder_basis_mode="full")``
  when that legacy mode is requested. When it runs first it is the actual cold
  shared preparation.
* ``gather_unweighted`` — the warm unweighted control prepare.
* ``basis_estimate`` — ``max(0, weighted_prepare − gather_unweighted)`` for the
  legacy full basis when both were measured. This is an **estimate**, not an
  isolated basis timer: the weighted wrapper includes gather and the unweighted
  call is warm.
* ``pca`` — production ``_pca_reduce`` at ``TRACKSPACE_OPENTSNE_PCA_DIM``
  (default 50).
* ``fit`` — one production ``_fit_tsne`` multilevel fit, stage-split by the
  shared ``fit_multilevel`` timings (``tsne_level*`` / ``tsne_expand`` /
  ``tsne_final``).
* ``pipeline_estimate`` — this case's prepare + PCA + fit. This is *not* an HTTP
  request and is not a cold end-to-end claim: it excludes queueing, Nyström
  placement, the weight Jacobian (only requested by ``jacobian=1``), and the 3-D
  request's extra 2-D reference fit. A folder-off input prepared after the
  weighted one is flagged ``prepare_warm``.

Protocol: for each diagnostic size, one warmup fit per case/dimension (excluded)
then ``--repeats`` measured rounds, each visiting the case/dimension tasks in a
deterministic seeded shuffle. Capped-thread cases run under
``threadpoolctl.threadpool_limits`` with ``n_jobs=<threads>``; the native case
uses ``n_jobs=-1`` and no thread cap. Fidelity (neighbourhood recall @10/@100 and
a 2048-pair Spearman proxy) is scored once per case/dimension on the first
measured map, outside the timed region. The default reference is that case's own
PCA input; ``--metric-reference unweighted`` scores every case against the same
unweighted PCA matrix.

Usage::

    python -m backend.benchmark_production_projection \
        --cases native,threads4,threads8,threads16 --dims 2,3 \
        --scope all --recipe genre_full --repeats 2 \
        --output data/layout-evaluation/production-benchmark.json

Folder-on cases are the production default. The CLAP text tower is lazy: it is
loaded only when the recipe gives CLAP a positive weight or supplies semantic
tags. ``genre_full`` has CLAP at 0, so its data-driven folder directions are
applied without the text tower (and without the slow GPU side effect). A recipe
that genuinely needs text fails loudly if the model cannot load. Use
``--cases folders-off`` to skip folder weighting entirely.

Folder-weighting comparison gallery (opt-in)::

    python -m backend.benchmark_production_projection \
        --scope all --cases native,folders-off --dims 2,3 \
        --gallery-output data/layout-evaluation/folder-separation

This reuses the first measured fit of each case/dimension (no extra fit),
aligns each non-reference map onto the reference of its own dimension, and
writes one group (a 2-D artifact plus its ``_3d`` sibling) per case plus a
schema-1 manifest. Every artifact covers the same cohort and shares the
unweighted control's ``dataset_fingerprint``. The legacy on/off ids stay
``folder_on`` / ``folder_off``.

The cheap folder-basis comparison uses the same writer over six cases::

    python -m backend.benchmark_production_projection \
        --scope all --recipe genre_full \
        --cases folders-off,centroids,approx,off-exag15,off-exag2,centroids-exag15 \
        --dims 2,3 --repeats 3 --metric-reference unweighted \
        --gallery-output data/layout-evaluation/folder-cheap \
        --output data/layout-evaluation/folder-cheap-timings.json

The reference is the ``folders-off`` baseline (native is absent), so every
other map aligns onto the unweighted map of its own dimension. With
``--metric-reference unweighted`` all six cases are scored against the same
unweighted PCA matrix; the default ``own`` keeps the old per-case reference.
The expensive exact-SVD ``full`` basis is never prepared unless a case asks for
it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from backend.benchmark_layout_scaling import (
    _resolve_sizes,
    _summary,
    broad_distance_spearman,
)
from backend.cache import TrackCache
from backend.embeddings import clap
from backend.embeddings.feature_cache import (
    LIBRARY_MEMORY_BYTES,
    LIBRARY_MEMORY_ENTRIES,
    FeatureCache,
)
from backend.embeddings.layout import prepare_projection_input
from backend.embeddings.layout_evaluation import (
    MIN_TRACKS,
    _atomic_write_json,
    _backend_info,
    _source_versions,
    build_manifest,
    dataset_fingerprint,
    normalise_aspect,
    orthogonal_align_to,
)
from backend.embeddings.layout_fidelity import neighbourhood_recall
from backend.embeddings.multilevel_tsne import resolve_fraction_schedule
from backend.embeddings.projection_backends import (
    _fit_tsne,
    _opentsne_max_points,
    _opentsne_pca_dim,
    _pca_reduce,
)
from backend.embeddings.projection_config import (
    FROZEN_SOURCE_WEIGHTS,
    OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER,
    OPENTSNE_COARSE_ITERATION_BUDGET,
    OPENTSNE_FINAL_EXAGGERATION,
    OPENTSNE_MULTILEVEL_FRACTIONS,
    OPENTSNE_PERPLEXITY_GAMMA,
    OPENTSNE_PERPLEXITY_START_FRACTION,
    PROJECTION_RECIPE_IDS,
    PROJECTION_RECIPE_VERSION,
)
from backend.precompute_layout_evaluation import _data_dir, load_infos
from backend.services.projection_request import projection_params_for_recipe
from backend.services.roots_registry import load_roots_state, root_role

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCOPES = ("library", "all")
DIMS = (2, 3)
DEFAULT_CASES = ("native", "threads4", "threads8", "threads16")

#: Folder-basis solver per case. ``none`` skips folder weighting entirely.
BASIS_MODES = ("none", "full", "centroids", "approx")
#: Preparation order: the legacy full basis first when requested (cold), then the
#: cheap alternatives, then the unweighted control last (warm gather).
BASIS_PREPARE_ORDER = ("full", "centroids", "approx", "none")

#: Named cheap-comparison set: one unweighted baseline plus the two cheap bases
#: and two late-exaggeration variants. The expensive full basis is deliberately
#: absent.
FOLDER_CHEAP_CASES = (
    "folders-off",
    "centroids",
    "approx",
    "off-exag15",
    "off-exag2",
    "centroids-exag15",
)

GALLERY_DIMS = (2, 3)
#: The optimizer seed pinned inside production ``_fit_tsne`` (openTSNE
#: ``random_state``). ``--seed`` seeds only this benchmark's sampling, round
#: order and metric pairs; changing it must never imply a different fit.
FIT_SEED = 42
#: Benchmark case → stable gallery artifact id. The legacy on/off ids stay
#: byte-identical (``folder_on`` / ``folder_off``) so old galleries keep loading.
GALLERY_IDS = {
    "native": "folder_on",
    "folders-off": "folder_off",
    "centroids": "centroids",
    "approx": "approx",
    "off-exag15": "off_exag15",
    "off-exag2": "off_exag2",
    "centroids-exag15": "centroids_exag15",
}
GALLERY_LABELS = {
    "folder_on": "Folder weighting on",
    "folder_off": "Folder weighting off",
    "centroids": "Folder basis: centroids",
    "approx": "Folder basis: approx",
    "off_exag15": "Folder off, final exaggeration 1.5",
    "off_exag2": "Folder off, final exaggeration 2",
    "centroids_exag15": "Folder basis: centroids, final exaggeration 1.5",
}

#: openTSNE's own auto method boundary (``tsne.py``): below 10k rows it picks
#: Barnes-Hut, at or above it picks FFT. The 3-D path pins Barnes-Hut because FFT
#: is 2-D only.
_AUTO_FFT_MIN_ROWS = 10_000

#: Short labels for the two quality measures; the actual helpers are imported.
METRICS = {
    "neighbourhood_recall@10": "PCA input vs map, k=10, sample 256",
    "neighbourhood_recall@100": "PCA input vs map, k=100, sample 256",
    "broad_distance_spearman": "2048 seeded pair distances, PCA vs map",
}


@dataclass(frozen=True)
class CaseSpec:
    """One benchmark case: only the knobs that differ from production."""

    name: str
    threads: int | None = None
    engine: str | None = None
    theta: float | None = None
    final_exaggeration: float | None = None
    #: Folder-basis solver: ``none`` / ``full`` / ``centroids`` / ``approx``.
    basis_mode: str = "full"
    #: Recipe final-fit budget, filled from ``ProjectionParams`` before a run.
    tsne_n_iter: int = 50
    tsne_early_exaggeration_iter: int = 0

    def __post_init__(self) -> None:
        if self.basis_mode not in BASIS_MODES:
            raise ValueError(
                f"unknown basis mode {self.basis_mode!r} "
                f"(allowed: {', '.join(BASIS_MODES)})"
            )

    @property
    def folders(self) -> bool:
        """Backwards-compatible boolean view (``none`` = off)."""
        return self.basis_mode != "none"


@dataclass
class CaseInput:
    """Prepared fit input for one folder-basis mode at one size."""

    paths: list[str]
    mat_proj: np.ndarray
    reduced: np.ndarray
    comp_config: str
    tsne_config: str
    prepare_s: float
    weighted_prepare_s: float | None
    gather_unweighted_s: float
    basis_estimate_s: float | None
    prepare_warm: bool
    pca_s: float
    folder_weighting: bool
    fingerprint: str
    basis_mode: str = "full"


@dataclass
class FitOutcome:
    """One fit's result plus the timing / thread evidence the harness records."""

    coords: np.ndarray
    stages: dict[str, float]
    threadpool_info: list[dict]
    wall: float


@dataclass
class GalleryCapture:
    """The first measured fit of one (case, dims) task, kept for the gallery.

    Raw (unanligned, unnormalised) coordinates and the input identity are held in
    memory until every task has run, then serialised once by
    :func:`write_comparison_gallery`. No extra fit is performed.
    """

    coords: np.ndarray
    paths: list[str]
    input_fingerprint: str
    timings: dict[str, float]
    basis_mode: str = "full"
    final_exaggeration: float | None = None


#: Explicit names keep the digit strings unambiguous: ``off-exag15`` is 1.5,
#: while the legacy dynamic ``exagN`` suffix still reads as a bare number
#: (``exag15`` → 15, not 1.5). Named cases win because they are matched first.
NAMED_CASES: dict[str, CaseSpec] = {
    "native": CaseSpec("native", basis_mode="full"),
    "folders-off": CaseSpec("folders-off", basis_mode="none"),
    "centroids": CaseSpec("centroids", basis_mode="centroids"),
    "approx": CaseSpec("approx", basis_mode="approx"),
    "off-exag15": CaseSpec(
        "off-exag15", basis_mode="none", final_exaggeration=1.5
    ),
    "off-exag2": CaseSpec(
        "off-exag2", basis_mode="none", final_exaggeration=2.0
    ),
    "centroids-exag15": CaseSpec(
        "centroids-exag15", basis_mode="centroids", final_exaggeration=1.5
    ),
}


def _parse_cases(raw: str | None, threads: str | None) -> list[CaseSpec]:
    """Parse ``--cases`` plus the convenience ``--threads`` list.

    Recognised named cases: ``native`` (full folder basis), ``folders-off``
    (no folder weighting), ``centroids``, ``approx``, ``off-exag15``,
    ``off-exag2`` and ``centroids-exag15``. Legacy dynamic forms remain:
    ``threadsN``, ``bh``, ``fft``, ``thetaNN`` (``theta07`` → 0.7) and
    ``exagN`` (``exag2`` → 2.0, ``exag15`` → 15.0 — use the named case for 1.5).
    """
    names = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    if threads:
        names.extend(f"threads{p.strip()}" for p in threads.split(",") if p.strip())
    if not names:
        names = list(DEFAULT_CASES)

    specs: list[CaseSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name in NAMED_CASES:
            spec = NAMED_CASES[name]
        elif name == "bh":
            spec = CaseSpec("bh", engine="bh")
        elif name == "fft":
            spec = CaseSpec("fft", engine="fft")
        elif name.startswith("threads"):
            try:
                n = int(name[len("threads"):])
            except ValueError:
                raise ValueError(f"bad thread case {name!r}") from None
            if n < 1:
                raise ValueError(f"thread count must be >= 1 (got {n})")
            spec = CaseSpec(f"threads{n}", threads=n)
        elif name.startswith("theta"):
            suffix = name[len("theta"):]
            try:
                # ``theta07`` / ``theta09`` → 0.7 / 0.9, ``theta5`` → 0.5.
                value = float(suffix) if "." in suffix else int(suffix) / 10.0
            except ValueError:
                raise ValueError(f"bad theta case {name!r}") from None
            if not 0.0 < value <= 1.0:
                raise ValueError(f"theta must be in (0, 1] (got {value})")
            # theta only affects Barnes-Hut, so pin BH for the case to be meaningful.
            spec = CaseSpec(name, engine="bh", theta=value)
        elif name.startswith("exag"):
            suffix = name[len("exag"):]
            try:
                value = float(suffix)
            except ValueError:
                raise ValueError(f"bad exag case {name!r}") from None
            if value <= 0:
                raise ValueError(f"exaggeration must be > 0 (got {value})")
            spec = CaseSpec(name, final_exaggeration=value)
        else:
            raise ValueError(
                f"unknown case {name!r} (known: native, folders-off, centroids, "
                "approx, off-exag15, off-exag2, centroids-exag15, threadsN, "
                "bh, fft, thetaNN, exagN)"
            )
        specs.append(spec)
    return specs


def _parse_dims(raw: str) -> list[int]:
    dims = sorted({int(p.strip()) for p in raw.split(",") if p.strip()})
    if not dims:
        raise ValueError("--dims must list 2 and/or 3")
    bad = [d for d in dims if d not in DIMS]
    if bad:
        raise ValueError(f"unknown dimension(s): {bad} (allowed: 2, 3)")
    return dims


def _needs_clap(params: Any) -> bool:
    """Whether folder weighting can actually consume CLAP text directions.

    Data-driven folder directions need no model. Text enters only through tags or
    a folder text-fallback into a CLAP column, so a recipe with CLAP at weight 0
    (``genre_full``) must not pay for the text tower.
    """
    if params.tag_names:
        return True
    if "clap" not in tuple(params.sources):
        return False
    weights = params.source_weights or FROZEN_SOURCE_WEIGHTS
    return float(weights.get("clap", 0.0)) > 0.0


def _resolved_engine(spec: CaseSpec, dims: int, rows: int) -> str:
    """The engine openTSNE will actually use for a *rows*-row level."""
    if dims == 3:
        return "bh"
    if spec.engine is not None:
        return spec.engine
    return "fft" if rows >= _AUTO_FFT_MIN_ROWS else "bh"


def _schedule_for(n: int) -> list[tuple[int, int, int, float | None]]:
    """The exact production coarse schedule for *n* rows (same constants)."""
    return resolve_fraction_schedule(
        OPENTSNE_MULTILEVEL_FRACTIONS,
        n,
        normal_budget=OPENTSNE_COARSE_ITERATION_BUDGET,
        gamma=OPENTSNE_PERPLEXITY_GAMMA,
        start_fraction=OPENTSNE_PERPLEXITY_START_FRACTION,
        early_iter=OPENTSNE_COARSE_EARLY_EXAGGERATION_ITER,
    )


def _engines(spec: CaseSpec, dims: int, n: int) -> list[dict[str, Any]]:
    """Per-level resolved engine evidence (the final full-data level included)."""
    out: list[dict[str, Any]] = []
    for i, (count, *_rest) in enumerate(_schedule_for(n)):
        out.append(
            {"level": f"tsne_level{i}", "count": int(count),
             "engine": _resolved_engine(spec, dims, count)}
        )
    out.append(
        {"level": "tsne_final", "count": int(n),
         "engine": _resolved_engine(spec, dims, n)}
    )
    return out


def _perform_fit(
    reduced: np.ndarray, dims: int, spec: CaseSpec
) -> FitOutcome:
    """Run exactly one production multilevel fit with the case's knobs.

    Threads are bounded with ``threadpoolctl`` *around* the fit and the wall
    clock starts inside that context, so setup is not attributed to the fit.
    """
    import threadpoolctl

    stages: dict[str, float] = {}

    def _on_timings(payload: dict[str, float]) -> None:
        stages.update(payload)

    if spec.threads is None:
        context = contextlib.nullcontext()
        n_jobs = -1
    else:
        context = threadpoolctl.threadpool_limits(limits=int(spec.threads))
        n_jobs = int(spec.threads)

    with context:
        threadpool_info = threadpoolctl.threadpool_info()
        t0 = time.perf_counter()
        _embedding, coords = _fit_tsne(
            reduced,
            dims=dims,
            tsne_n_iter=spec.tsne_n_iter,
            tsne_early_exaggeration_iter=spec.tsne_early_exaggeration_iter,
            n_jobs=n_jobs,
            negative_gradient_method=spec.engine,
            theta=spec.theta,
            final_exaggeration=spec.final_exaggeration,
            on_timings=_on_timings,
        )
        wall = time.perf_counter() - t0
    return FitOutcome(
        coords=np.asarray(coords, dtype=np.float32),
        stages=stages,
        threadpool_info=threadpool_info,
        wall=wall,
    )


def _skip_reason(spec: CaseSpec, dims: int) -> str | None:
    if dims == 3 and spec.engine not in (None, "bh"):
        return "3-D is Barnes-Hut only; FFT is 2-D only"
    return None


def _input_for(inputs: dict[str, CaseInput], spec: CaseSpec) -> CaseInput:
    """Look up a case's prepared input by basis mode."""
    inp = inputs.get(spec.basis_mode)
    if inp is not None:
        return inp
    raise ValueError(
        f"missing input for basis_mode={spec.basis_mode!r} (case {spec.name!r})"
    )


def run_benchmark(
    cases: list[CaseSpec],
    dims: list[int],
    inputs: dict[str, CaseInput],
    *,
    seed: int = 42,
    repeats: int = 3,
    fit: Callable[[np.ndarray, int, CaseSpec], FitOutcome] | None = None,
    score: bool = True,
    metric_reference: str = "own",
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    on_case_done: Callable[[str, int, dict[str, Any]], None] | None = None,
    on_first_fit: (
        Callable[
            [str, int, np.ndarray, CaseInput, FitOutcome, CaseSpec], None
        ] | None
    ) = None,
) -> dict[str, Any]:
    """Measure every (case, dimension) task over the prepared inputs.

    Pure orchestration over numpy + an injectable *fit* (the default is the
    production ``_fit_tsne`` wrapper). ``inputs`` is keyed by basis mode
    (``none`` / ``full`` / ``centroids`` / ``approx``). Only the modes used by
    *cases* are required, plus ``none`` when ``metric_reference="unweighted"``.

    ``metric_reference="own"`` (the default, and the old behaviour) scores each
    map against its own PCA input. ``"unweighted"`` scores every case against
    the same unweighted PCA matrix, which is the only fair cross-basis
    comparison: the map's own weighted space would flatter each basis.

    ``on_case_done(name, dims, result)`` fires once per completed case/dimension
    (also for skipped 3-D/FFT combinations), which is the CLI's checkpoint
    point. ``on_first_fit(name, dims, coords, inp, outcome, spec)`` fires once,
    on the first measured fit of each task, and is the gallery's capture point
    (no extra fit).
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not cases:
        raise ValueError("at least one case is required")
    if metric_reference not in ("own", "unweighted"):
        raise ValueError(
            f"unknown metric reference {metric_reference!r} "
            "(allowed: own, unweighted)"
        )
    if fit is None:
        fit = _perform_fit

    tasks: list[tuple[CaseSpec, int]] = [
        (spec, d) for spec in cases for d in dims if _skip_reason(spec, d) is None
    ]
    if not tasks:
        raise ValueError("no runnable case/dimension combinations")
    for spec, _d in tasks:
        _input_for(inputs, spec)

    shared: CaseInput | None = None
    if metric_reference == "unweighted":
        shared = inputs.get("none")
        if shared is None:
            raise ValueError(
                "metric_reference='unweighted' requires the unweighted input"
            )
        for spec, d in tasks:
            inp = _input_for(inputs, spec)
            if list(inp.paths) != list(shared.paths):
                raise ValueError(
                    f"case {spec.name!r} {d}-D covers a different track cohort "
                    "than the shared unweighted metric reference"
                )

    def _progress(stage: str, **info: Any) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    rng = np.random.default_rng(seed)
    warmup_order = [tasks[int(i)] for i in rng.permutation(len(tasks))]
    for spec, d in warmup_order:
        inp = _input_for(inputs, spec)
        _progress("warmup-start", case=spec.name, dims=d, tracks=len(inp.paths))
        outcome = fit(inp.reduced, d, spec)
        _progress("warmup-end", case=spec.name, dims=d, seconds=outcome.wall)

    round_orders = [
        [tasks[int(i)] for i in rng.permutation(len(tasks))]
        for _ in range(repeats)
    ]
    samples: dict[tuple[str, int], list[dict[str, Any]]] = {
        (spec.name, d): [] for spec in cases for d in dims
    }
    first: dict[tuple[str, int], np.ndarray] = {}
    for r, order in enumerate(round_orders):
        for spec, d in order:
            inp = _input_for(inputs, spec)
            _progress(
                "fit-start", case=spec.name, dims=d, round=r,
                tracks=len(inp.paths),
            )
            outcome = fit(inp.reduced, d, spec)
            key = (spec.name, d)
            samples[key].append(
                {
                    "round": r,
                    "total": outcome.wall,
                    "stages": dict(outcome.stages),
                    "threadpool_info": outcome.threadpool_info,
                }
            )
            if key not in first:
                first[key] = outcome.coords
                if on_first_fit is not None:
                    on_first_fit(
                        spec.name, d, outcome.coords, inp, outcome, spec
                    )
            _progress(
                "fit-end", case=spec.name, dims=d, round=r,
                seconds=outcome.wall,
            )

    results: dict[str, Any] = {"cases": {}}
    for spec in cases:
        case_out: dict[str, Any] = {}
        for d in dims:
            reason = _skip_reason(spec, d)
            if reason is not None:
                skipped = {
                    "case": spec.name,
                    "dims": d,
                    "folders": spec.folders,
                    "basis_mode": spec.basis_mode,
                    "skipped": reason,
                }
                case_out[str(d)] = skipped
                if on_case_done is not None:
                    on_case_done(spec.name, d, skipped)
                continue
            inp = _input_for(inputs, spec)
            sm = samples[(spec.name, d)]
            stage_keys = sorted({k for x in sm for k in x["stages"]})
            timings = {
                "prepare": inp.prepare_s,
                "weighted_prepare": inp.weighted_prepare_s,
                "gather_unweighted": inp.gather_unweighted_s,
                "basis_estimate": inp.basis_estimate_s,
                "prepare_warm": inp.prepare_warm,
                "pca": inp.pca_s,
                "total": _summary([x["total"] for x in sm]),
                "stages": {
                    k: _summary([x["stages"].get(k, 0.0) for x in sm])
                    for k in stage_keys
                },
            }
            fixed = inp.prepare_s + inp.pca_s
            timings["pipeline_estimate"] = {
                k: fixed + timings["total"][k] for k in ("median", "min", "max")
            }
            metrics: dict[str, Any] = {}
            if score:
                coords = first[(spec.name, d)]
                reference = shared.reduced if shared is not None else inp.reduced
                metrics = {
                    "neighbourhood_recall@10": neighbourhood_recall(
                        reference, coords, k=10, sample=256, seed=seed
                    ),
                    "neighbourhood_recall@100": neighbourhood_recall(
                        reference, coords, k=100, sample=256, seed=seed
                    ),
                    "broad_distance_spearman": broad_distance_spearman(
                        reference, coords, n_pairs=2048, seed=seed
                    ),
                }
            engines = _engines(spec, d, len(inp.paths))
            result = {
                "case": spec.name,
                "dims": d,
                "folders": spec.folders,
                "basis_mode": spec.basis_mode,
                "metric_reference": metric_reference,
                "requested": {
                    "threads": spec.threads,
                    "engine": spec.engine or "auto",
                    "theta": spec.theta,
                    "final_exaggeration": spec.final_exaggeration,
                    "basis_mode": spec.basis_mode,
                },
                "input": {
                    "tracks": len(inp.paths),
                    "shape": list(inp.mat_proj.shape),
                    "pca_dim": int(inp.reduced.shape[1]),
                    "comp_config": inp.comp_config,
                    "tsne_config": inp.tsne_config,
                    "fingerprint": inp.fingerprint,
                    "folder_weighting": inp.folder_weighting,
                    "basis_mode": inp.basis_mode,
                },
                "schedule": [list(level) for level in _schedule_for(len(inp.paths))],
                "engines": engines,
                "theta_effective": spec.theta is not None
                and any(e["engine"] == "bh" for e in engines),
                "threads": {
                    "configured": spec.threads,
                    "n_jobs": -1 if spec.threads is None else spec.threads,
                    "threadpool_info": sm[0].get("threadpool_info", []),
                },
                "samples": sm,
                "timings": timings,
                "metrics": metrics,
                "notes": _case_notes(
                    spec, d, engines, inp.prepare_warm, metric_reference
                ),
            }
            case_out[str(d)] = result
            if on_case_done is not None:
                on_case_done(spec.name, d, result)
        results["cases"][spec.name] = case_out
    return results


def _case_notes(
    spec: CaseSpec,
    dims: int,
    engines: list[dict[str, Any]],
    prepare_warm: bool,
    metric_reference: str = "own",
) -> list[str]:
    notes: list[str] = []
    if dims == 3:
        notes.append("3-D pins Barnes-Hut; the production request also fits a 2-D reference for the gauge (excluded here).")
    if spec.engine is None and dims == 2:
        final_engine = engines[-1]["engine"] if engines else "bh"
        if final_engine == "fft":
            notes.append(
                "auto engine: coarse levels <10k rows use BH, the full-data "
                "level uses FFT."
            )
        else:
            notes.append(
                "auto engine: every level resolves to Barnes-Hut "
                "(all levels are below the FFT row threshold)."
            )
    if prepare_warm:
        notes.append(
            "this input was prepared after the first basis mode, so its prepare "
            "timing is warm; basis_estimate is None for this case."
        )
    if spec.theta is not None and not any(e["engine"] == "bh" for e in engines):
        notes.append("theta is ignored because the resolved engine is FFT.")
    if spec.final_exaggeration is not None:
        notes.append(
            "final_exaggeration sets openTSNE's late exaggeration on the final fit only; "
            "early exaggeration / iteration counts stay at the recipe values."
        )
    if metric_reference == "unweighted":
        notes.append(
            "every case is scored against the same unweighted PCA matrix; "
            "the map's own weighted PCA space is not the reference."
        )
    return notes


# ── input preparation (no cache rows, tags, or layouts written) ─────────────


def _normalise_basis_modes(modes: set[str]) -> list[str]:
    """Order the requested basis modes into preparation order."""
    out: set[str] = set()
    for mode in modes:
        if mode in BASIS_MODES:
            out.add(mode)
        else:
            raise ValueError(
                f"unknown basis mode {mode!r} (allowed: {', '.join(BASIS_MODES)})"
            )
    if not out:
        raise ValueError("at least one basis mode is required")
    return [m for m in BASIS_PREPARE_ORDER if m in out]


def _clear_basis_cache() -> None:
    """Drop the process-local semantic basis cache between modes.

    Each mode is namespaced in the production cache key, so this is not needed
    for correctness; clearing it makes every mode's measured ``prepare`` an
    actual basis build instead of a hit on an earlier mode/size in this private
    benchmark process.
    """
    from backend.embeddings import layout

    cache = getattr(layout, "_semantic_basis_cache", None)
    if cache is not None:
        cache.clear()


def prepare_inputs(
    infos: list[dict],
    feature_cache: Any,
    params: Any,
    basis_modes: set[str],
    *,
    need_unweighted_control: bool = False,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, CaseInput], dict[str, Any]]:
    """Prepare one projection input per folder-basis mode.

    The first mode prepared is the cold one: it pays the composite gather. Every
    later mode is a warm gather that still builds its own basis and PCA. Order is
    ``full`` (legacy production), ``centroids``, ``approx``, ``none``: a cheap
    comparison therefore measures ``centroids`` cold, ``approx`` and the
    unweighted control warm. ``full`` is only prepared when a case asks for it,
    so the cheap set never pays the legacy exact SVD.

    Each mode is prepared and PCA-reduced exactly once per size. The unweighted
    input is the shared cohort control; it is prepared even when no case uses it
    if *need_unweighted_control* is set (the shared metric reference/gallery).
    For the legacy full basis, ``basis_estimate`` remains the old estimate
    ``max(0, full_prepare - unweighted_prepare)``.
    """
    modes = _normalise_basis_modes(basis_modes)
    weights = dict(params.source_weights or FROZEN_SOURCE_WEIGHTS)

    def _emit(stage: str, **info: Any) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    def _prepare(scale_folders: bool, stage: str, basis_mode: str):
        _emit(f"{stage}-start", tracks=len(infos), basis_mode=basis_mode)
        t0 = time.perf_counter()
        out = prepare_projection_input(
            infos,
            feature_cache,
            sources=params.sources,
            feature_mask=params.feature_mask,
            features_blend=params.features_blend,
            source_weights=weights,
            required_sources=params.required_sources,
            scale_folders=scale_folders,
            context_tags=params.tag_names,
            context_folders=params.explicit_folders or None,
            folder_boost=params.folder_boost,
            folder_depth_boost=params.folder_depth_boost,
            folder_basis_mode=(basis_mode if scale_folders else "full"),
        )
        elapsed = time.perf_counter() - t0
        _emit(
            f"{stage}-end", seconds=elapsed, tracks=len(out.paths),
            shape=list(out.mat.shape), basis_mode=basis_mode,
        )
        return out, elapsed

    prepared: dict[str, tuple[Any, float, bool]] = {}
    for i, mode in enumerate(modes):
        _clear_basis_cache()
        scale = mode != "none"
        inp, elapsed = _prepare(scale, f"{mode}-prepare", mode)
        prepared[mode] = (inp, elapsed, i != 0)

    control_inp = prepared["none"][0] if "none" in prepared else None
    control_s = prepared["none"][1] if "none" in prepared else None
    if control_inp is None and ("full" in prepared or need_unweighted_control):
        # Warm unweighted diagnostic: the legacy basis estimate and/or the
        # shared metric/gallery control. Store it so the expose loop below can
        # return it as ``inputs["none"]`` when it was not requested as a case.
        _clear_basis_cache()
        control_inp, control_s = _prepare(False, "gather-unweighted", "none")
        prepared["none"] = (control_inp, control_s, True)

    base = control_inp or next(inp for inp, _s, _w in prepared.values())
    if len(base.paths) < 2 or base.mat.shape[0] < 2:
        raise ValueError(
            f"only {len(base.paths)} tracks have the required sources; "
            "need at least 2"
        )
    composite_fp = dataset_fingerprint(base.paths, base.mat, base.comp_config)

    full_s = prepared["full"][1] if "full" in prepared else None
    weighted_prepare_s = full_s
    gather_unweighted_s = control_s if control_s is not None else 0.0
    basis_estimate = (
        max(0.0, full_s - control_s)
        if full_s is not None and control_s is not None
        else None
    )

    expose = set(modes)
    if need_unweighted_control:
        expose.add("none")

    def _pca(inp, basis_mode: str):
        _emit("pca-start", tracks=len(inp.paths), basis_mode=basis_mode)
        t0 = time.perf_counter()
        reduced, _pca = _pca_reduce(
            inp.mat_proj, _opentsne_pca_dim(inp.mat_proj.shape[1])
        )
        elapsed = time.perf_counter() - t0
        _emit("pca-end", seconds=elapsed, basis_mode=basis_mode)
        return reduced, elapsed

    inputs: dict[str, CaseInput] = {}
    prepare_meta: dict[str, dict[str, Any]] = {}
    for mode in BASIS_PREPARE_ORDER:
        if mode not in prepared or mode not in expose:
            continue
        inp, prepare_s, warm = prepared[mode]
        reduced, pca_s = _pca(inp, mode)
        mat_proj = np.asarray(inp.mat_proj)
        inputs[mode] = CaseInput(
            paths=list(inp.paths),
            mat_proj=mat_proj,
            reduced=np.asarray(reduced),
            comp_config=inp.comp_config,
            tsne_config=inp.tsne_config,
            prepare_s=prepare_s,
            weighted_prepare_s=(full_s if mode == "full" else None),
            gather_unweighted_s=gather_unweighted_s,
            basis_estimate_s=(basis_estimate if mode == "full" else None),
            prepare_warm=warm,
            pca_s=pca_s,
            folder_weighting=inp.tsne_config != inp.comp_config,
            fingerprint=dataset_fingerprint(
                inp.paths, mat_proj, inp.tsne_config
            ),
            basis_mode=mode,
        )
        prepare_meta[mode] = {
            "prepare_s": prepare_s,
            "pca_s": pca_s,
            "prepare_warm": warm,
        }

    return inputs, {
        "tracks": len(base.paths),
        "composite_shape": list(base.mat.shape),
        "composite_fingerprint": composite_fp,
        "comp_config": base.comp_config,
        "basis_modes": modes,
        "prepare_modes": prepare_meta,
        "weighted_prepare_s": weighted_prepare_s,
        "gather_unweighted_s": gather_unweighted_s,
        "basis_estimate_s": basis_estimate,
    }


# ── folder-weighting comparison gallery (opt-in, no extra fits) ─────────────


def _gallery_artifact_id(case: str, dims: int) -> str:
    base = GALLERY_IDS[case]
    return f"{base}_3d" if dims == 3 else base


def _gallery_case_order(cases: list[CaseSpec] | list[str] | None) -> list[str]:
    if cases is None:
        return []
    names = [c if isinstance(c, str) else c.name for c in cases]
    return list(dict.fromkeys(names))


def _validate_gallery_request(
    cases: list[CaseSpec],
    dims: list[int],
    sizes: str | None,
) -> None:
    """Preflight the ``--gallery-output`` combination before any expensive work.

    Any set of at least two *known* cases works: the writer emits one group and
    one paired 2-D/3-D artifact per case. Unknown cases, a dimension other than
    2+3, and ``--sizes`` are rejected before a gather or fit begins.
    """
    names = _gallery_case_order(cases)
    if len(names) < 2:
        raise ValueError(
            "--gallery-output requires at least two known --cases "
            f"(got {','.join(names) or 'none'})"
        )
    unknown = [name for name in names if name not in GALLERY_IDS]
    if unknown:
        raise ValueError(
            "--gallery-output does not know case(s) "
            f"{', '.join(unknown)} (known: {', '.join(GALLERY_IDS)})"
        )
    if list(dims) != list(GALLERY_DIMS):
        raise ValueError(
            "--gallery-output requires --dims 2,3 "
            f"(got {','.join(str(d) for d in dims)})"
        )
    if sizes:
        raise ValueError(
            "--gallery-output writes one full-size gallery; --sizes is not supported"
        )


def _resolve_gallery_reference(
    cases: list[str], reference: str | None
) -> str:
    """Explicit reference, else native (legacy), else the folders-off baseline."""
    if reference is not None:
        if reference not in cases:
            raise ValueError(
                f"--gallery-reference {reference!r} is not one of the gallery cases"
            )
        return reference
    if "native" in cases:
        return "native"
    if "folders-off" in cases:
        return "folders-off"
    return cases[0]


def _effective_final_exaggeration(value: float | None) -> float:
    """The late exaggeration a fit actually uses: ``None`` means production."""
    return OPENTSNE_FINAL_EXAGGERATION if value is None else float(value)


def _gallery_revision(
    dataset_fp: str,
    input_fp: str,
    *,
    basis_mode: str,
    final_exaggeration: float | None,
    dims: int,
    seed: int,
    source_weights: dict[str, float],
    versions: dict[str, Any],
) -> str:
    """Stable identity of one gallery artifact.

    Canonical JSON (sorted keys, fixed separators) over the control dataset
    fingerprint, the artifact's own input fingerprint, the *actual* basis
    parameters (never the CLI alias), the effective late exaggeration, dimension,
    seed, weights, the production recipe version and the effective environment.
    """
    payload = {
        "dataset_fingerprint": dataset_fp,
        "input_fingerprint": input_fp,
        "basis_mode": basis_mode,
        "final_exaggeration": _effective_final_exaggeration(final_exaggeration),
        "dims": dims,
        "seed": seed,
        "source_weights": source_weights,
        "recipe_version": PROJECTION_RECIPE_VERSION,
        "versions": versions,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_comparison_gallery(
    prepared: dict[tuple[str, int], GalleryCapture],
    params: Any,
    *,
    output_dir: str,
    feature_cache: Any,
    seed: int = 42,
    recipe_id: str = "genre_full",
    scope: str = "all",
    limit: int | None = None,
    cases: list[CaseSpec] | list[str] | None = None,
    reference: str | None = None,
    source_versions: dict[str, Any] | None = None,
    backend: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Serialise a paired 2-D/3-D comparison gallery from already-fitted maps.

    The maps were already fitted by :func:`run_benchmark`; this function only
    aligns, normalises and writes. ``cases`` is the ordered case list (names or
    :class:`CaseSpec`); when omitted it is inferred from ``prepared`` in the
    canonical :data:`GALLERY_IDS` order. Every case contributes one group and a
    2-D plus a 3-D artifact (``_3d`` suffix).

    ``reference`` selects the alignment reference; ``None`` keeps the legacy
    behaviour (``native`` when present) and otherwise falls back to
    ``folders-off``, the cheap set's unweighted baseline. Non-reference maps are
    aligned onto the reference **of their own dimension** (never 2-D onto 3-D),
    then normalised.

    Every artifact must cover exactly the same track cohort. The unweighted
    ``folders-off`` input, when present, is the shared control fingerprint. An
    ``in_progress`` manifest with no recipes is written *before* the artifact
    loop, and the ``complete`` manifest last: a reader can never see a recipe
    listed as complete while its file belongs to a previous run. ``scope`` /
    ``limit`` are the CLI cohort selectors recorded in the manifest metadata.
    """
    case_names = _gallery_case_order(cases)
    if not case_names:
        present = {key[0] for key in prepared}
        case_names = [name for name in GALLERY_IDS if name in present]
    unknown = [name for name in case_names if name not in GALLERY_IDS]
    if unknown:
        raise ValueError(f"unknown gallery case(s): {', '.join(unknown)}")
    if len(case_names) < 2:
        raise ValueError("a comparison gallery needs at least two cases")
    required = {(case, d) for case in case_names for d in GALLERY_DIMS}
    missing = sorted(required - set(prepared))
    unexpected = sorted(set(prepared) - required)
    if missing or unexpected:
        raise ValueError(
            "comparison gallery requires exactly the supplied cases × 2-D/3-D "
            f"fits (missing: {missing or 'none'}; unexpected: {unexpected or 'none'})"
        )
    if reference is not None and reference not in case_names:
        raise ValueError(
            f"--gallery-reference {reference!r} is not one of the gallery cases"
        )
    ref_case = _resolve_gallery_reference(case_names, reference)

    control_case = "folders-off" if "folders-off" in case_names else ref_case
    paths = list(prepared[(control_case, 2)].paths)
    for key in sorted(required):
        capture = prepared[key]
        if list(capture.paths) != paths:
            raise ValueError(
                "gallery inputs diverged from the control cohort: "
                f"{key[0]} {key[1]}-D covers a different track set/order"
            )
    if len(paths) < MIN_TRACKS:
        raise ValueError(
            f"gallery needs at least {MIN_TRACKS} tracks (got {len(paths)})"
        )

    versions = dict(
        source_versions if source_versions is not None else _source_versions(feature_cache)
    )
    backend_info = dict(backend if backend is not None else _backend_info())
    env_versions = {**versions, **backend_info}
    control_fp = prepared[(control_case, 2)].input_fingerprint
    weights = {
        k: float(v) for k, v in dict(params.source_weights or FROZEN_SOURCE_WEIGHTS).items()
    }
    n = len(paths)
    os.makedirs(output_dir, exist_ok=True)

    raw: dict[tuple[str, int], np.ndarray] = {}
    for key in sorted(required):
        coords = np.asarray(prepared[key].coords, dtype=np.float32)
        if coords.shape != (n, key[1]):
            raise ValueError(
                f"{key[0]} {key[1]}-D coordinates have shape {coords.shape}, "
                f"expected {(n, key[1])}"
            )
        if not np.isfinite(coords).all():
            raise ValueError(f"{key[0]} {key[1]}-D produced non-finite coordinates")
        raw[key] = coords

    results: dict[str, dict] = {}
    reference_ids = {
        d: _gallery_artifact_id(ref_case, d) for d in GALLERY_DIMS
    }
    prepare_ref = float(prepared[(ref_case, 2)].timings["prepare"])
    prepare_control = float(prepared[(control_case, 2)].timings["prepare"])
    # One preparation and one PCA per basis mode, not per artifact: off-exag
    # variants share the unweighted input, and native/folders-off must not be
    # counted twice when the reference *is* the control. Fits and alignments are
    # counted once per artifact below.
    gather_total = 0.0
    pca_total = 0.0
    gathered_modes: set[str] = set()
    for case in case_names:
        capture = prepared[(case, 2)]
        if capture.basis_mode in gathered_modes:
            continue
        gathered_modes.add(capture.basis_mode)
        gather_total += float(capture.timings["prepare"])
        pca_total += float(capture.timings["pca"])
    manifest_path = os.path.join(output_dir, "manifest.json")

    def _manifest(
        status: str, recipes: dict[str, dict], timings: dict[str, float]
    ) -> dict:
        built = build_manifest(
            status=status,
            dataset_fingerprint=control_fp,
            count=n,
            sources=tuple(params.sources),
            source_weights=weights,
            required_sources=tuple(params.required_sources),
            feature_mask=np.asarray(params.feature_mask),
            features_blend=float(params.features_blend),
            seed=seed,
            n_jobs=None,
            reference_id=reference_ids[2],
            reference_ids=reference_ids,
            backend=backend_info,
            versions=versions,
            recipes=recipes,
            timings=timings,
        )
        # Per-artifact input params live beside the shared manifest entry.
        for rid in list(built["recipes"]):
            extra = results.get(rid)
            if extra and built["recipes"][rid].get("status") == "complete":
                built["recipes"][rid].update(
                    {
                        "folder_basis_mode": extra["folder_basis_mode"],
                        "final_exaggeration": extra["final_exaggeration"],
                        "final_exaggeration_effective": _effective_final_exaggeration(
                            extra["final_exaggeration"]
                        ),
                        "prepare_warm": extra["prepare_warm"],
                    }
                )
        built["dataset"].update({"scope": scope, "limit": limit})
        built["fit_seed"] = FIT_SEED
        return built

    # Empty in-progress manifest first: every complete artifact entry below
    # belongs to a file written after this point. The shared prepare/PCA have
    # already run; no fit or alignment has, so those stay at zero.
    _atomic_write_json(
        manifest_path,
        _manifest(
            "in_progress",
            {},
            {
                "gather": gather_total,
                "pca": pca_total,
                "tsne": 0.0,
                "alignment": 0.0,
                "total": gather_total + pca_total,
            },
        ),
    )

    # 2-D group first, then 3-D: the manifest order is the gallery order.
    for d in GALLERY_DIMS:
        ref_id = reference_ids[d]
        for case in case_names:
            case_id = GALLERY_IDS[case]
            rid = _gallery_artifact_id(case, d)
            label = GALLERY_LABELS[case_id] + (" (3D)" if d == 3 else "")
            capture = prepared[(case, d)]
            coords = raw[(case, d)]
            aligned_to: str | None = None
            t_align = 0.0
            if case != ref_case:
                t0 = time.perf_counter()
                coords = orthogonal_align_to(
                    coords, paths, raw[(ref_case, d)], paths
                )
                t_align = time.perf_counter() - t0
                aligned_to = ref_id
            normed = normalise_aspect(coords)
            if d == 3:
                positions = [
                    {
                        "path": paths[i],
                        "x": float(normed[i, 0]),
                        "y": float(normed[i, 1]),
                        "z": float(normed[i, 2]),
                    }
                    for i in range(n)
                ]
            else:
                positions = [
                    {
                        "path": paths[i],
                        "x": float(normed[i, 0]),
                        "y": float(normed[i, 1]),
                    }
                    for i in range(n)
                ]
            timings = {
                "prepare": float(capture.timings["prepare"]),
                "pca": float(capture.timings["pca"]),
                "tsne": float(capture.timings["fit"]),
                "alignment": float(t_align),
            }
            timings["total"] = sum(timings.values())
            revision = _gallery_revision(
                control_fp,
                capture.input_fingerprint,
                basis_mode=capture.basis_mode,
                final_exaggeration=capture.final_exaggeration,
                dims=d,
                seed=seed,
                source_weights=weights,
                versions=env_versions,
            )
            artifact = {
                "id": rid,
                "label": label,
                "revision": revision,
                "dataset_fingerprint": control_fp,
                "input_fingerprint": capture.input_fingerprint,
                "source_weights": weights,
                "recipe": {
                    "case": case_id,
                    "label": label,
                    "scale_folders": capture.basis_mode != "none",
                    "folder_basis_mode": capture.basis_mode,
                    "final_exaggeration": capture.final_exaggeration,
                    "final_exaggeration_effective": _effective_final_exaggeration(
                        capture.final_exaggeration
                    ),
                    "folder_boost": float(params.folder_boost),
                    "folder_depth_boost": float(params.folder_depth_boost),
                    "sources": list(params.sources),
                    "source_weights": weights,
                    "required_sources": list(params.required_sources),
                    "tsne_n_iter": int(params.tsne_n_iter),
                    "tsne_early_exaggeration_iter": int(
                        params.tsne_early_exaggeration_iter
                    ),
                    "recipe_id": recipe_id,
                    "recipe_version": PROJECTION_RECIPE_VERSION,
                },
                "tracks": n,
                "aligned_to": aligned_to,
                "dims": d,
                "group": case_id,
                "timings": timings,
                "positions": positions,
            }
            _atomic_write_json(os.path.join(output_dir, f"{rid}.json"), artifact)
            results[rid] = {
                "status": "complete",
                "label": label,
                "revision": revision,
                "tracks": n,
                "timings": timings,
                "aligned_to": aligned_to,
                "input_fingerprint": capture.input_fingerprint,
                "source_weights": weights,
                "dims": d,
                "group": case_id,
                "folder_basis_mode": capture.basis_mode,
                "final_exaggeration": capture.final_exaggeration,
                "prepare_warm": bool(capture.timings.get("prepare_warm", False)),
            }

    gallery_timings = {
        "gather": gather_total,
        "pca": pca_total,
        "tsne": sum(float(r["timings"]["tsne"]) for r in results.values()),
        "alignment": sum(float(r["timings"]["alignment"]) for r in results.values()),
    }
    gallery_timings["total"] = sum(gallery_timings.values())
    manifest = _manifest("complete", results, gallery_timings)
    if "native" in case_names and "folders-off" in case_names:
        manifest["folder_weighting"] = {
            "scale_folders": True,
            "folder_boost": float(params.folder_boost),
            "folder_depth_boost": float(params.folder_depth_boost),
            "reference": reference_ids[2],
            "control": GALLERY_IDS[control_case],
        }
    manifest["gallery"] = {
        "reference": reference_ids[2],
        "reference_case": ref_case,
        "control_case": control_case,
        "groups": [GALLERY_IDS[case] for case in case_names],
        "cases": [
            {
                "case": case,
                "artifact_id": GALLERY_IDS[case],
                "folder_basis_mode": prepared[(case, 2)].basis_mode,
                "final_exaggeration": prepared[(case, 2)].final_exaggeration,
                "final_exaggeration_effective": _effective_final_exaggeration(
                    prepared[(case, 2)].final_exaggeration
                ),
                "prepare_s": float(prepared[(case, 2)].timings["prepare"]),
                "pca_s": float(prepared[(case, 2)].timings["pca"]),
                "prepare_warm": bool(
                    prepared[(case, 2)].timings.get("prepare_warm", False)
                ),
            }
            for case in case_names
        ],
    }
    manifest["production"] = {
        "recipe": recipe_id,
        "recipe_version": PROJECTION_RECIPE_VERSION,
        "tsne_n_iter": int(params.tsne_n_iter),
        "tsne_early_exaggeration_iter": int(params.tsne_early_exaggeration_iter),
    }
    native_prepare = (
        float(prepared[("native", 2)].timings["prepare"])
        if "native" in case_names
        else None
    )
    manifest["prepare_estimate"] = {
        "reference_case": ref_case,
        "reference_prepare_s": prepare_ref,
        "control_case": control_case,
        "control_prepare_s": prepare_control,
        # Only the legacy full-basis comparison has an isolated basis estimate.
        "basis_estimate_s": (
            max(0.0, native_prepare - prepare_control)
            if native_prepare is not None
            else None
        ),
        "modes": {
            case: {
                "prepare_s": float(prepared[(case, 2)].timings["prepare"]),
                "pca_s": float(prepared[(case, 2)].timings["pca"]),
                "prepare_warm": bool(
                    prepared[(case, 2)].timings.get("prepare_warm", False)
                ),
            }
            for case in case_names
        },
        "label": (
            "per-case input preparation at gallery time; only the first prepared "
            "basis mode is cold, every later mode pays a warm gather plus its own "
            "basis build. Compare prepare times together with prepare_warm."
        ),
    }
    if native_prepare is not None:
        manifest["prepare_estimate"]["weighted_prepare_s"] = native_prepare
        manifest["prepare_estimate"]["unweighted_prepare_warm_s"] = prepare_control
        manifest["prepare_estimate"]["label"] = (
            "weighted production prepare vs the warm unweighted gather diagnostic; "
            "an estimate, not a cold folder-on/off comparison"
        )
    manifest["notes"] = {
        **manifest.get("notes", {}),
        "timing_source": (
            "each artifact's prepare/pca/tsne timing is from the first measured "
            "fit in its first measured round (warmups excluded); the manifest "
            "total counts each shared prepare/PCA once and every fit/alignment "
            "once per artifact"
        ),
        "fit_seed": (
            f"the fit optimizer is pinned to seed {FIT_SEED} by production; the "
            "manifest's seed is this run's sampling/order/metric seed"
        ),
        "geometry": (
            f"non-reference maps are orthogonally aligned onto the {ref_case!r} "
            "reference of their own dimension, then normalised; x/y are not "
            "comparable across dimensions"
        ),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


# ── CLI ─────────────────────────────────────────────────────────────────────


def _env() -> dict[str, Any]:
    import scipy

    try:
        import threadpoolctl

        native_threadpool = threadpoolctl.threadpool_info()
    except Exception:  # pragma: no cover - environment only
        native_threadpool = []
    return {
        **_backend_info(),
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threadpool_native": native_threadpool,
    }


def _roots_summary(roots: OrderedDict, root_roles: OrderedDict) -> list[dict[str, str]]:
    return [
        {"id": rid, "path": roots[rid], "role": root_role(rid, root_roles)}
        for rid in roots
    ]


def _select_infos(infos: list[dict], sizes: list[int], seed: int) -> dict[int, list[dict]]:
    """Nested seeded subsets that preserve the original (production) row order.

    The full set is returned byte-for-byte in its loaded order — no permutation —
    so a full-scope run gathers exactly the native ordering. Subsets are seeded
    prefixes of one permutation whose selected indices are sorted back into
    original order, keeping them reproducible and nested.
    """
    n = len(infos)
    if sizes == [n]:
        return {n: list(infos)}
    perm = np.random.default_rng(seed).permutation(n)
    out: dict[int, list[dict]] = {}
    for s in sizes:
        if s == n:
            out[s] = list(infos)
            continue
        idx = np.sort(perm[:s])
        out[s] = [infos[int(i)] for i in idx]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only benchmark of the production projection fit "
            "(weighted prepare, PCA, multilevel t-SNE) at any library size."
        )
    )
    parser.add_argument("--data-dir", default=None, help="Cache dir holding roots.json + audio_features.db (default: data/)")
    parser.add_argument("--output", default=None, help="JSON checkpoint path (default: <data-dir>/layout-evaluation/production-benchmark.json)")
    parser.add_argument("--scope", default="all", choices=SCOPES, help="Roots to gather: all (library+external) or library (default all)")
    parser.add_argument("--recipe", default="genre_full", choices=PROJECTION_RECIPE_IDS, help="Canonical projection recipe (default genre_full)")
    parser.add_argument("--dims", default="2,3", help="Comma-separated output dimensions (default 2,3)")
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "Comma-separated cases (default native,threads4,threads8,threads16). "
            "Named: native (full basis), folders-off, centroids, approx, "
            "off-exag15, off-exag2, centroids-exag15; plus threadsN, bh, fft, "
            "thetaNN, exagN."
        ),
    )
    parser.add_argument("--threads", default=None, help="Convenience comma-separated capped thread counts, e.g. 4,8,16")
    parser.add_argument("--repeats", type=int, default=3, help="Measured rounds per case/dimension after one warmup (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Diagnostic cap on eligible tracks (seeded subset)")
    parser.add_argument("--sizes", default=None, help="Optional comma-separated nested track counts (default: full eligible set only)")
    parser.add_argument(
        "--metric-reference",
        choices=("own", "unweighted"),
        default="own",
        dest="metric_reference",
        help=(
            "Fidelity reference: each case's own PCA input (default, old "
            "behaviour) or the shared unweighted PCA matrix for a fair "
            "cross-basis comparison."
        ),
    )
    parser.add_argument(
        "--gallery-output",
        default=None,
        help=(
            "Opt-in: also write a paired 2-D/3-D comparison gallery here (needs "
            "at least two known --cases, --dims 2,3 and no --sizes). Default: no "
            "gallery IO."
        ),
    )
    parser.add_argument(
        "--gallery-reference",
        default=None,
        help=(
            "Gallery alignment reference case (default: native when present, "
            "otherwise folders-off, otherwise the first case)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    try:
        cases = _parse_cases(args.cases, args.threads)
        dims = _parse_dims(args.dims)
        if args.gallery_output:
            _validate_gallery_request(cases, dims, args.sizes)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    try:
        requested_sizes = (
            [int(s) for s in args.sizes.split(",") if s.strip()]
            if args.sizes
            else []
        )
    except ValueError:
        raise SystemExit(f"--sizes must be comma-separated integers (got {args.sizes!r})") from None
    if any(s < 2 for s in requested_sizes):
        raise SystemExit("--sizes must list integers >= 2")

    data_dir = os.path.abspath(args.data_dir or _data_dir())
    output = os.path.abspath(
        args.output
        or os.path.join(data_dir, "layout-evaluation", "production-benchmark.json")
    )
    gallery_output = os.path.abspath(args.gallery_output) if args.gallery_output else None

    roots: OrderedDict[str, str] = OrderedDict()
    root_roles: OrderedDict[str, str] = OrderedDict()
    state_path = os.path.join(data_dir, "roots.json")
    if not os.path.exists(state_path):
        raise SystemExit(
            f"no roots.json at {state_path}; start the Trackspace server once "
            "to register a library root"
        )
    load_roots_state(roots, state_path, default_root=None, root_roles=root_roles)
    if not roots:
        raise SystemExit("no library roots configured (roots.json is empty)")

    params = projection_params_for_recipe(args.recipe, genre_full_enabled=True)
    cases = [
        replace(
            c,
            tsne_n_iter=int(params.tsne_n_iter),
            tsne_early_exaggeration_iter=int(
                params.tsne_early_exaggeration_iter
            ),
        )
        for c in cases
    ]
    print(
        f"scope={args.scope} recipe={args.recipe} dims={dims} "
        f"cases={[c.name for c in cases]} repeats={args.repeats}",
        flush=True,
    )

    track_cache = TrackCache(os.path.join(data_dir, "track_cache.db"))
    feature_cache = FeatureCache(
        os.path.join(data_dir, "audio_features.db"),
        max_memory=LIBRARY_MEMORY_ENTRIES,
        max_memory_bytes=LIBRARY_MEMORY_BYTES,
    )
    try:
        infos, n_mp3s = load_infos(roots, track_cache, root_roles, scope=args.scope)
        if not infos:
            raise SystemExit(
                "no cached track fingerprints found under the selected roots; "
                "run the analysis pipeline first"
            )
        if args.limit is not None:
            if args.limit < 2:
                raise SystemExit("--limit must be >= 2")
            if args.limit < len(infos):
                idx = np.sort(
                    np.random.default_rng(args.seed).permutation(len(infos))[
                        : args.limit
                    ]
                )
                infos = [infos[int(i)] for i in idx]
                print(
                    f"limit: {len(infos)} of {n_mp3s} mp3s (seeded subset)",
                    flush=True,
                )
        sizes = _resolve_sizes(requested_sizes, len(infos))

        basis_modes = {spec.basis_mode for spec in cases}
        clap_required = any(mode != "none" for mode in basis_modes) and _needs_clap(
            params
        )
        if clap_required and not clap.is_model_ready():
            clap.load_model()
        if clap_required and not clap.is_model_ready():
            raise SystemExit(
                "folder-weighted cases need the CLAP text tower for the "
                f"semantic basis, but it failed to load ({clap.load_error()}); "
                "use --cases folders-off to benchmark without it"
            )

        cap = _opentsne_max_points()
        if cap > 0 and max(sizes) > cap:
            raise SystemExit(
                f"{max(sizes)} eligible tracks exceed TRACKSPACE_OPENTSNE_MAX_POINTS="
                f"{cap}; production would fall back to sklearn, so this tool "
                "refuses to measure a path production would not take"
            )

        subsets = _select_infos(infos, sizes, args.seed)
        envelope: dict[str, Any] = {
            "tool": "production-projection-benchmark",
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": args.scope,
            "recipe": args.recipe,
            "recipe_params": {
                "sources": list(params.sources),
                "weights": dict(params.source_weights or FROZEN_SOURCE_WEIGHTS),
                "required_sources": list(params.required_sources),
                "scale_folders": bool(params.scale_folders),
                "folder_boost": float(params.folder_boost),
                "folder_depth_boost": float(params.folder_depth_boost),
                "tsne_n_iter": int(params.tsne_n_iter),
                "tsne_early_exaggeration_iter": int(
                    params.tsne_early_exaggeration_iter
                ),
            },
            "settings": {
                "seed": args.seed,
                "fit_seed": FIT_SEED,
                "repeats": args.repeats,
                "dims": dims,
                "metric_reference": args.metric_reference,
                "cases": [
                    {
                        "name": c.name,
                        "threads": c.threads,
                        "engine": c.engine or "auto",
                        "theta": c.theta,
                        "final_exaggeration": c.final_exaggeration,
                        "folders": c.folders,
                        "basis_mode": c.basis_mode,
                    }
                    for c in cases
                ],
                "requested_sizes": sizes,
                # Actual counts after the required-source filter, appended per size.
                "sizes": [],
            },
            "dataset": {
                # Before the required-source filter; the actual eligible count is
                # filled in once the full-size gather reports its selected rows.
                "fingerprinted_tracks": len(infos),
                "eligible_tracks": None,
                "n_mp3s": n_mp3s,
                "roots": _roots_summary(roots, root_roles),
            },
            "environment": {
                **_env(),
                "clap_required": clap_required,
                "clap_model_ready": clap.is_model_ready(),
                "clap_error": clap.load_error(),
            },
            "metrics": METRICS,
            "results": {},
            "checkpoint": {"sizes_completed": [], "complete": False},
            "notes": [
                "Each case's basis mode is prepared and PCA-reduced once per "
                "size. The first prepared mode is cold (it pays the composite "
                "gather); every later mode is a warm gather that still builds "
                "its own basis. Preparation order: full, centroids, approx, "
                "none.",
                "weighted_prepare = the timed full-basis prepare (gather + "
                "folder weighting) when that legacy mode is requested; it is the "
                "actual shared preparation when measured first.",
                "gather_unweighted is the warm unweighted control prepare; for "
                "the legacy full basis, basis_estimate = max(0, weighted_prepare "
                "- gather_unweighted) is an estimate, not an isolated timer.",
                "pipeline_estimate = this case's prepare + PCA + fit; not an HTTP "
                "request and not a cold end-to-end claim.",
                "Excluded: queue wait, Nyström placement, the weight Jacobian "
                "(only jacobian=1), and the 3-D request's extra 2-D reference fit.",
                "Default GET does not request the Jacobian; a weight drag can add "
                "one fit per source.",
                "Any input prepared after the first basis mode is marked "
                "prepare_warm and is not a cold preparation measurement.",
                "Fidelity scores are scored once per case/dimension on the first "
                "measured map. --metric-reference own (default) uses the case's "
                "own PCA input; unweighted uses the same unweighted PCA matrix "
                "for every case, the fair cross-basis comparison. Indicative, not "
                "proof of global layout quality.",
                "Prepare/PCA are measured once per size without a case thread "
                "cap; only the fit is capped for threadsN cases.",
                "Fidelity is computed on the raw fit coordinates, before the "
                "production gauge and per-axis normalisation.",
                "Cache constructors may create schema/metadata/WAL; no cache "
                "rows, MP3 tags, production layouts, or in-flight fit entries are "
                "written.",
                "The CLAP text tower is not loaded when the recipe's CLAP weight "
                "is 0; genre_full folder directions are data-driven.",
                "Exaggeration changes the t-SNE objective; cluster compactness is "
                "not automatically better fidelity.",
            ],
        }
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        _atomic_write_json(output, envelope)
        print(f"checkpoint: {output}", flush=True)

        for requested_size in sizes:
            subset = subsets[requested_size]
            _emit = lambda stage, info: print(  # noqa: E731
                f"[bench] requested={requested_size} {stage} "
                + " ".join(f"{k}={v}" for k, v in info.items()),
                flush=True,
            )
            try:
                inputs, input_meta = prepare_inputs(
                    subset,
                    feature_cache,
                    params,
                    basis_modes,
                    need_unweighted_control=(
                        args.metric_reference == "unweighted"
                    ),
                    on_progress=_emit,
                )
            except ValueError as e:
                raise SystemExit(str(e)) from None
            actual_size = int(input_meta["tracks"])
            size_out = {
                "requested_tracks": requested_size,
                "selected_tracks": actual_size,
                "input": input_meta,
                "cases": {},
            }
            envelope["results"][str(actual_size)] = size_out
            if requested_size == sizes[-1]:
                envelope["dataset"]["eligible_tracks"] = actual_size
            envelope["settings"]["sizes"].append(actual_size)
            if actual_size != requested_size:
                print(
                    f"[bench] requested={requested_size} selected={actual_size} "
                    "(required-source filter)",
                    flush=True,
                )

            def _on_case_done(
                name: str,
                dim: int,
                result: dict[str, Any],
                size=actual_size,
                size_out=size_out,
            ) -> None:
                size_out["cases"].setdefault(name, {})[str(dim)] = result
                _atomic_write_json(output, envelope)
                state = "skipped" if result.get("skipped") else "done"
                print(
                    f"[bench] size={size} case={name} dims={dim} {state} "
                    f"(checkpoint)",
                    flush=True,
                )

            prepared: dict[tuple[str, int], GalleryCapture] = {}

            def _capture_first(
                name: str,
                dim: int,
                coords: np.ndarray,
                inp: CaseInput,
                outcome: FitOutcome,
                spec: CaseSpec,
            ) -> None:
                prepared[(name, dim)] = GalleryCapture(
                    coords=np.array(coords, dtype=np.float32, copy=True),
                    paths=list(inp.paths),
                    input_fingerprint=inp.fingerprint,
                    timings={
                        "prepare": inp.prepare_s,
                        "pca": inp.pca_s,
                        "fit": outcome.wall,
                        "prepare_warm": inp.prepare_warm,
                    },
                    basis_mode=spec.basis_mode,
                    final_exaggeration=spec.final_exaggeration,
                )

            run_benchmark(
                cases,
                dims,
                inputs,
                seed=args.seed,
                repeats=args.repeats,
                metric_reference=args.metric_reference,
                on_progress=lambda stage, info: _emit(stage, info),
                on_case_done=_on_case_done,
                on_first_fit=_capture_first if gallery_output else None,
            )
            if gallery_output:
                try:
                    gallery = write_comparison_gallery(
                        prepared,
                        params,
                        output_dir=gallery_output,
                        feature_cache=feature_cache,
                        seed=args.seed,
                        recipe_id=args.recipe,
                        scope=args.scope,
                        limit=args.limit,
                        cases=cases,
                        reference=args.gallery_reference,
                    )
                except ValueError as e:
                    raise SystemExit(str(e)) from None
                print(
                    f"[bench] gallery: {gallery_output} "
                    f"({', '.join(gallery['recipes'])})",
                    flush=True,
                )
            done = envelope["checkpoint"]["sizes_completed"]
            if actual_size not in done:
                done.append(actual_size)
            envelope["checkpoint"]["complete"] = requested_size == sizes[-1]
            _atomic_write_json(output, envelope)
            print(
                f"[bench] requested={requested_size} selected={actual_size} "
                f"complete ({len(done)}/{len(sizes)} sizes)",
                flush=True,
            )

        envelope["checkpoint"]["complete"] = True
        _atomic_write_json(output, envelope)
        print(f"\nWrote {output}", flush=True)
        print(
            f"tracks: {envelope['dataset']['eligible_tracks']} | "
            f"sizes: {envelope['settings']['sizes']}",
            flush=True,
        )
    finally:
        del track_cache
        del feature_cache


if __name__ == "__main__":
    main()
