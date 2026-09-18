# Trackspace — engineering TODO

Remaining, actionable work only. Completed work and the reasoning behind it live in
the docs: `architecture.md`, `ui_conventions.md`, `audio_pipeline.md`,
`design-patterns.md`.

## User feedback
- [ ] Don't re-home when pressing "solo" in a card, feels a bit jumpy
  make the "solo thing" more like a sort of "proximity slider"; so you can drag the size of the neighborhood you want to see (instea of top-N)
  (If the slider starts from 100%, then we could actually re-home as we narrow it down; as the jump will be less sudden)

- [ ] Whacko idea? Add a 3d view for the embedding space?

- [ ] When projecting my library onto external tracks; those who don't fit into my paylists, can still end up in an Generic/${name}/ subdirectories; where "name" equals what that genre name that generic discogs model would give to them
- [ ] If i want to project onto an external root which contains a bunch of subdirectories; allow me to just gather all of the nested files to reproject [[maybe stoer some metadata on the old relative folder structure?dd]]
- [ ] How to remove a root folder from the analysis library? 
      Though i might want tracks analyzed, but not be "my" tracks; i.e. a separate section for tracks which are in the system but not "My library" (folders which probably also contain things where i will project my learned folder structure on)
- [ ] **Undo leaves empty fan-out folders.** The move journal restores every
  file but does not remove the subfolders a fan-out created (they are often empty
  afterwards). Sweep empty directories on undo, bounded to the fan-out root so it can
  never touch an unrelated tree. Files:
  `backend/services/analysis_pipeline.py` (`undo_fanout_supervised`).
- [ ] **The move journal is never pruned.** `fanout_journal` keeps every applied
  batch forever (undone rows included). Prune batches older than the last N (or older
  than a few weeks) on startup; the journal is only needed for the most recent undo.
  Files: `backend/embeddings/feature_cache.py`.
- [ ] **The match floor is a fixed 5th percentile.** `DEFAULT_MATCH_PERCENTILE` is
  not user-tunable and there is no preview of what would be left unmatched. Consider
  a slider in the imprint panel that re-plans against a given percentile (the plan
  already reports `unmatched`), and re-measure the default on the real library.
  Files: `backend/embeddings/supervised.py`, `frontend/src/coordinators/head-coordinator.ts`.

### Source-weight comparability (normalising the source blocks)

The rationale is in `audio_pipeline.md` → *Composite vector construction*.

- [ ] **Frozen reference normalisation.** Fit the per-source transform once on a
  **frozen reference set** and cache it in a chart artifact, then apply it
  statelessly. That gives distribution matching without the drift. This is the same
  `chart_revision` split the landmark extension needs — do both together, and bump
  `PROJECTION_RECIPE_VERSION`.
- [ ] **Pick the technique and measure it.** Cheapest first: per-source z-score
  (`clustering._standardize`); **whitening** (ZCA/PCA) to equalise covariance;
  **CORAL** to align second-order statistics; **CCA / Deep CCA** for a shared
  maximally-correlated subspace; **optimal transport** (Wasserstein /
  Gromov-Wasserstein) which needs no shared space at all. Note the codebase already
  uses Procrustes, but only for PCA *orientation* stability
  (`projection_backends._pca_ref_lookup`), not for cross-source alignment.
- [ ] **Do not whiten blindly.** Whitening decorrelates dimensions and can destroy
  the cosine geometry the clustering relies on (`build_weighted_matrix` L2-normalises
  rows because cosine kNN assumes it). Validate with the landmark fidelity harness
  below — neighbourhood recall / trustworthiness, not cluster count.
- [ ] **Reading list for the write-up.** The relevant names to search:
  *distribution matching*, *standardisation / z-scoring*, *whitening (ZCA)*, *CORAL*,
  *canonical correlation analysis (CCA / Deep CCA)*, *Procrustes alignment* (MUSE,
  VecMap), *optimal transport / Wasserstein / Gromov-Wasserstein*, *the modality gap*
  (Liang et al. 2022 — directly relevant, CLAP is CLIP-style), *alignment &
  uniformity* (Wang & Isola 2020), *multi-view / multi-modal representation
  learning*, *domain adaptation*. Document whatever is chosen in
  `audio_pipeline.md` → *Embedding sources*.
- [ ] **Landmark 2-D fidelity is unmeasured.** The landmark extension's derivative
  is exact (checked against central differences of the same `g`), but whether its 2-D
  layout tracks the *real* t-SNE chart across the weight range is empirical. The naive
  comparison is confounded — a refit is not a smooth function, so Procrustes cannot
  align its cluster rearrangements — so measure against a *fixed* chart instead. Tune
  `n_landmarks` (default 128) and `tau` (median of `D²`) on the real library.
- [ ] **One fidelity harness for both.** Build the researcher's §8 evaluation once and
  reuse it for the normalisation choice and the landmark fidelity: neighbourhood
  recall / trustworthiness at k=10/30/100; weight hold-out generalization (hold out
  *weight combinations*, not just tracks — a stationary map is perfectly smooth and
  fails this); derivative correctness (`J·Δw` against the *same frozen mapping* at
  decreasing step sizes, expecting a quadratic residual); and operational checks
  (10k/20k, insertion batches, row permutations, old-coordinate byte-stability).

## Lenses & scopes

- [ ] **Local scope refit.** A projection scope lens reuses its *global* layout,
  min-max normalised into the circle, so the subset's internal geometry is the one
  it had in the full map. A t-SNE fit over just the in-scope tracks would show the
  local structure better; it is a per-scope fit, so it needs a progress indicator and
  a way to invalidate it when the scope moves. Files:
  `frontend/src/coordinators/embedding-coordinator.ts` (`_scopeTargets`),
  `backend/embeddings/layout.py`.
- [ ] **Multiple scopes.** Only one scope is open at a time; a second X + drag
  replaces it. Multiple non-overlapping scopes (and a per-scope close) would let a
  user compare two clusters in two lenses at once. `Model.scope` / `scopePositions`
  would become a list. Files: `frontend/src/model.ts`,
  `frontend/src/components/canvas-view.ts`, `canvas-interaction.ts`.
- [ ] **Energy/dance semantic 2-space (experiment).** The re-encoded `features`
  block leaves energy and danceability linear. Test a more semantic encoding (e.g.
  polar around an energy×dance axis) against the §8 fidelity harness — only keep it
  if neighbourhood recall improves. Files:
  `backend/embeddings/feature_encoding.py`, `tests/`.
- [ ] **Preset export/import.** Lens presets live in `localStorage` only; a small
  JSON export/import would let a tuned view move between machines. Files:
  `frontend/src/model.ts` (`addLensPreset`), `frontend/src/coordinators/options-panel.ts`.
- [ ] **Per-source lens readiness hint.** A lens whose source is disabled on the
  server (e.g. `--no-clap`) shows an empty map rather than an explanatory note.
  Files: `frontend/src/coordinators/embedding-coordinator.ts` (`ensureAndProject`).

## Deferred / optional

### Frontend

- [ ] **Optional sibling hue wiggle.** Family tints currently differ only in
      lightness/saturation (`frontend/src/lib/colors.ts` / `backend/services/dir_colors.py`).
      If same-depth siblings still read too alike on the canvas, add a small hue deviation
      that also decays with depth; bump the shared golden vectors in
      `frontend/src/lib/dir-color-vectors.json` (asserted by both suites).
- [ ] **Inspect card follow-ups.** The pinned card (Alt over a dot) has anti-track,
      nearest, solo-neighbourhood and result focus, and its folder/cluster/key/BPM elements
      are navigation targets (glow + S/H/F/Shift+S) — see `design-patterns.md`. Still open:
      "same cluster" / "same folder" jumps from the result rows, a `k` control instead of
      the fixed `NEAR_K`/`SOLO_K`, and showing that the similarity score is a cosine, not a
      probability. `_fetchSimilar` builds the whole-library matrix on first call — a warm-up
      on library load would hide the first-click latency (cached for `CACHE_TTL_SECONDS`
      after that).
- [ ] **Preview audio may need a user gesture.** The preview volume now tapers
      (`previewGain`) and defaults to a moderate 80 %, but a browser that blocks autoplay
      will silently drop the *first* hover's `play()` (`_startPreview` swallows the
      rejection). Unreproduced in headless Chromium, so unconfirmed: if a user reports a
      silent first hover after a reload, catch `NotAllowedError` and show a one-time
      "click to enable preview audio" hint, then retry on the next gesture.
- [ ] **Projection refit progress is phase-level only.** The layout streams its
      phase as `layout_progress` (`compute_projection`'s `on_progress`), so the pill
      and status bar show "laying out" → "landmarks" → "Jacobian" instead of a static
      "Computing layout…". The base `layout` phase itself is still one opaque step:
      neither the sklearn fallback nor mlx-vis exposes iteration callbacks, so a percent
      needs a custom optimize loop (openTSNE's `optimize` is a generator).
- [ ] **Optional: split `coordinators/canvas-interaction.ts`** (~1040 lines, the
      largest frontend file). The ~200 lines of fit-view math (`_vpFromWorldBounds`,
      `_computeFitVP`, `_p80Radius`, `_fitViewToFolderPercentile`, `_fitViewToTracks`,
      `fitView`, `effectiveHoveredFolderPrefix`) are the natural seam into a
      `canvas-fit-view.ts`.
- [ ] **Optional: `lib/event-bus.ts`, `model/persistence.ts`, `api/types.ts`.**
      Extract the EventBus base, localStorage persistence, and shared API response
      types out of `model.ts` / `lib/api.ts`.
- [ ] **Nit:** `clamp` is duplicated in `coordinators/options-panel.ts`,
      `coordinators/canvas-interaction.ts`, and `coordinators/analysis-coordinator.ts`.
- [ ] **Structure preview follow-up.** `_fetchStructure` (`controller.ts`) fetches one
      path per hover — a batched request for the visible tracks would cut the
      round-trips.

### Backend

- [ ] **The map's "analysis ready" gate spans external roots.** `ensureAndProject`
      (`frontend/src/coordinators/embedding-coordinator.ts`) keys on
      `tracks_pending_maest/rhythm` across *all* roots, but the default layout is
      library-only. A large external root being analysed still holds the map on the PCA
      preview even though none of its tracks are in the default fit. Scope the gate to
      library tracks, or fetch the layout once the library slice is complete.
- [ ] **Clustering still includes external roots.** `/api/clusters` and `ClusterStore`
      build over every root, while the default 2-D layout is library-only. Decide
      whether cluster assignments and cluster colouring should follow the projection
      scope or stay whole-library (the supervised head already learns from library
      folders only).
- [ ] **mlx-vis point cap.** Above `TRACKSPACE_MLX_TSNE_MAX_POINTS` (default 10000)
      the map silently falls back to the PCA-reduced sklearn path; the library is now
      ~15.4k. Re-benchmark mlx-vis vs sklearn at that size and either raise the cap
      or accept the fallback.
- [ ] **`services/analysis_pipeline.py` — split if it grows again.** Handlers are
      already one factory function per stage; the cluster store and the enqueue helpers
      are the natural next seams.
- [ ] **Drop the dead generation helpers.** `coverage.build_generation_work` and
      `clap.batch_ensure_embeddings` have no callers since the durable queue took
      over (the queue's `enqueue_library_work` filter supersedes them); remove them
      and their re-exports in `embeddings/__init__.py` / `layout.py`.
- [ ] **Cheaper loudness envelope for `structure`.** `analyze_structure` decodes the
      whole file a second time, but the `scan` stage's `fpcalc` already decodes it
      and its raw fingerprint is discarded (only the hash is kept). If chromaprint can
      expose its 16-band frame image, the envelope could ride the scan pass and the
      structure stage would need no decode at all.

## Product roadmap (analysis)

Ordered roughly by expected value. Each is additive to the cached feature store —
none require re-running an encoder over the library.

- [ ] **Playlist labels.** Playlists do not exist in the product yet; when they do,
  they are another noisy label source for the same head (the training recipe already takes
  a label vector, so this is a data-plumbing change, not a new model).
- [ ] **Search follow-ups.** Folder-scoped search (`folder` param), search-as-you-type
  debounce, and a minimum-score threshold / "select all" affordance. Fit-on-focus is done
  (`CanvasInteractionController.fitViewToPaths`). Perf: `/api/search` calls the *uncached*
  `build_track_infos_fn` (`backend/routes/api_search.py`), so every query re-reads the
  whole library — use `track_infos_cached` (with `library_mtime_signature`) or resolve only
  the candidate paths, as `/api/tracks/structure` now does.
- [ ] **MERT-v2.** Deliberately deferred until MAEST shows a measured blind spot — on the
  6-genre validation it reached 96.6% purity.
- [ ] **Persist cluster results.** `ClusterStore` is in-memory; a restart now
  self-heals one re-cluster at startup (`app.main`), which recomputes in seconds at
  current library size. Persist only if that startup cost stops being negligible.

## Analysis performance

Making the pipeline faster. Ordered roughly by expected value; measure before and after.
Baselines and measured conclusions live in `audio_pipeline.md` → *Per-stage throughput*.

### Parallelism

- [ ] **Features process pool: cut child startup.** Each spawned child re-imports
      `backend.app` (hence torch, ~3.5 s) before `warm()` builds its madmom
      processors. A `forkserver` pool with a lightweight preload, or a dedicated
      worker entrypoint, would remove that; only worth it if the server is restarted
      often relative to library size.
- [ ] **Tune `scan`.** `batch_size=8` with `TRACKSPACE_FPCALC_CONCURRENCY`
      (default `min(8, cpus)`). Measure whether fpcalc subprocesses or the SQLite
      writes are the limit and raise only the one that is.
- [ ] **Structure pool sizing.** The `structure` stage runs 4 threads by default
      (`TRACKSPACE_STRUCTURE_WORKERS`; measured ~2.1× at 4 on a 16-track sample).
      Sweep the worker count on the real library and confirm threads stay ahead of a
      process pool as the library (and decode mix) grows.
- [ ] **Parallelise within a stage.** `features` and `maest` decode per track
      sequentially; a small per-stage thread pool (decode+compute) could overlap the
      locked open with DSP/GPU work without adding workers.

### Acceleration

- [ ] **Stop using libsndfile for MP3 (correctness, not speed).** It is the SIGBUS
      source (`audio_pipeline.md` → *Audio decoding*) and forces the open lock.
      **The original `torchaudio` plan is stale:** on torchaudio 2.11 the sox/ffmpeg
      file backends are gone — `torchaudio.load` raises `ImportError: TorchCodec is
      required` (torchcodec is not installed). So this now means adding torchcodec +
      FFmpeg shared libs, or shelling out to the `ffmpeg` already on PATH.
      Measured: an `ffmpeg` subprocess is *not* faster (equal on whole tracks,
      ~2.7× slower per 15 s segment from process spawn), so the only win is
      dropping `_DECODE_LOCK` for the SIGBUS invariant.
- [x] **MAEST `inference_mode`, and `torch.compile` rejected.** All inference
      forwards (MAEST, CLAP audio/text, the supervised head) now use
      `torch.inference_mode()`; measured neutral (0.96×/0.97× in an in-process,
      interleaved A/B), kept for correctness of intent. `torch.compile` on the MAEST
      transformer was measured twice, reproduced, and removed: ~2.3× slower per forward on
      MPS and 1.78× end-to-end on real tracks, with `dynamic=True` no better (numbers
      and method in `audio_pipeline.md` → *MAEST model management*). Only retry on CUDA,
      with that same paired measurement.
- [ ] **MAEST mel front end.** Already batched: `analyze_batch` stacks every
      excerpt of the claimed batch (up to `batch_size * MAEST_NUM_SEGMENTS`) into one
      `_mel(wave)` call, not three per track. The remaining CPU mel cost is the
      front end itself, which would need a different transform or device split.
- [ ] **madmom tempo.** `TRACKSPACE_MADMOM_FAST=1` (one BLSTM) is ~4× faster but
      opt-in. Benchmark the tempo error on a labelled sample and consider making it the
      default (bump `FEATURES_VERSION` if the vector changes).
- [ ] **Projection backends.** `TRACKSPACE_CUML` / `TRACKSPACE_MLX_VIS` are chosen by
      import success; re-benchmark mlx-vis vs openTSNE vs sklearn after any
      torch/librosa bump, and revisit `TRACKSPACE_OPENTSNE` being off by default.

### Dependencies

- [ ] **CLAP per-segment backfill.** `clap_segments` / `clap_segment_pos` are blank
      for pooled rows written before they existed (10 197 pooled vs. 112 with segments on
      the real library), so the CLAP stage must re-run once (decode-dominated — hours) to
      fill them. Until it drains, `/api/search` falls back to the pooled mean and hits carry
      no `section`. This is a schema backlog, not a version purge, so the startup backfill
      deliberately leaves it alone — press **CLAP** in the Analysis coverage panel to start it.
- [ ] **Retrain the supervised head after a CLAP backfill.** The checkpoint switch
      bumped `EMBEDDING_VERSION` 2 → 3, so every cached `clap` row is purged on next
      startup and the queue's startup backfill regenerates it (decode-dominated — hours
      on the real library). The supervised head consumes the `clap` block
      (`SUPERVISED_INPUT_WEIGHTS`), so its weights are stale against the new vectors —
      retrain via *Train folder view* once the CLAP backfill drains, or its
      `supervised` block is misleading. (A version purge is now self-healing at startup:
      see *Startup backfill* in `architecture.md`.)
- [ ] **`transformers` is installed at 5.3** against a `>=4.40` floor — a major bump.
      Pin a tested floor and smoke-test the CLAP text/audio paths. (The
      `larger_clap_music` checkpoint bug reproduces on 4.57.1, so it is a broken
      conversion, not a 5.x regression.)
- [ ] **`madmom` is unmaintained** (last release 2020) and already emits
      `pkg_resources` (removal slated) and NumPy 2.4 `align=0` deprecations. It will
      break on a future setuptools/NumPy. Evaluate replacing the tempo/key estimators
      or vendoring the two processors we use.
- [ ] **`libsndfile`/mpg123 thread safety.** Track the upstream fix for the
      `mpeg_decoder_init` FIXME and drop `_DECODE_LOCK` when it lands.
- [ ] **Pin the GPU/ML stack.** `torch` 2.11 / `mlx-vis` 0.7 are
      unpinned floors; a breaking change would land silently. Add upper bounds for the
      ones that affect embeddings.

### Profiling & benchmarks

- [ ] **CLAP `max_batch` sweep.** Measured ~0.14 s/track (~446 tracks/min) at the
      default `batch_size=4` → 12 segments, so CLAP is *not* decode-dominated and
      is one of the fastest stages. Low priority; sweep `max_batch` vs MPS memory
      only if a profile ever puts CLAP near the top.
- [ ] **Purge leftover EffNet cache rows.** `data/audio_features.db` still holds
      `effnet_embedding` / `effnet_version` columns and rows from before the source
      was deleted (~48 MB at 9.5k tracks), and `data/models/discogs-effnet-*.onnx`
      is still on disk. `data/` is gitignored and nothing reads them, so this is
      only worth doing if the disk footprint matters.
- [ ] **Cold vs warm end-to-end run.** The harness gives the warm per-track sum
      (~1.18 s/track across scan+features+structure+maest+clap ≈ 51 tracks/min, so
      ~3.5 h for 10.8k tracks), but a full `force` re-analysis wall-clock and a
      no-op re-run are still unrecorded; capture both.
- [ ] **MAEST batch-size sweep — the top per-track lever.** MAEST is the slowest
      stage (~0.4 s/track), so its batch size is the remaining lever now that
      `torch.compile` is ruled out. A sweep on a **quiet** machine was inconclusive
      under load (batch 1/2/4/8/16 spread ~70–150 tracks/min with no clean winner;
      the loaded machine's run-to-run spread exceeded the effect). Re-run
      `tests/profile_pipeline.py --stages maest --maest-batch N --repeat 3` when idle.
      If the machine never goes idle, an interleaved batch-N pair under identical load
      is still more informative than one under-load run, because the ratio cancels the
      load.
- [ ] **Peak RSS / GPU memory.** No memory baseline exists; capture peak RSS and MPS
      allocation per stage before adding workers or batches.
- [ ] **Cluster/projection timing at scale.** `DEFAULT_RESOLUTIONS` and k=20 were
      calibrated at ~9.5k tracks; re-measure once the library doubles.
