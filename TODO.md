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
  layout tracks the *real* t-SNE chart across the weight range is empirical. The harness
  now exists (`backend/embeddings/layout_fidelity.py`); run it with the landmark model's
  placed rows against the base chart. The naive comparison is confounded — a refit is not
  a smooth function, so Procrustes cannot align its cluster rearrangements — so measure
  against a *fixed* chart instead. Tune `n_landmarks` (default 128) and `tau` (median of
  `D²`) on the real library.
- [ ] **Fidelity harness: the researcher's §8 checks.** The neighbourhood-recall
  half is built (`backend/embeddings/layout_fidelity.py`, bench entrypoint in
  `tests/test_profiling_embeddings.py`, unit tests in `tests/test_layout_fidelity.py`)
  and measured the incremental placement (0.313 vs the base layout's 0.360 at k=10) and
  near-duplicate collapse (0.5% at 0.999). Still open: trustworthiness, weight
  hold-out generalization (hold out *weight combinations*, not just tracks — a stationary
  map is perfectly smooth and fails this), derivative correctness (`J·Δw` against the
  *same frozen mapping* at decreasing step sizes, expecting a quadratic residual), and
  operational checks (10k/20k, insertion batches, row permutations, old-coordinate
  byte-stability).

## Lenses & scopes

- [ ] **Local scope refit.** A projection scope lens reuses its *global* layout,
  min-max normalised into the circle, so the subset's internal geometry is the one
  it had in the full map. A t-SNE fit over just the in-scope tracks would show the
  local structure better; it is a per-scope fit, so it needs a progress indicator and
  a way to invalidate it when the scope moves. Files:
  `frontend/src/coordinators/embedding-coordinator.ts` (`_computeScopeRaw` /
  `_mapIntoScope`), `backend/embeddings/layout.py`.
- [ ] **Multiple scopes.** Only one scope is open at a time; a second X + drag
  replaces it. Multiple non-overlapping scopes (and a per-scope close) would let a
  user compare two clusters in two lenses at once. `Model.scope` / `scopePositions`
  would become a list. Files: `frontend/src/model.ts`,
  `frontend/src/components/canvas-view.ts`, `canvas-interaction.ts`.
- [ ] **Box scope resize is uniform.** X + scroll scales a box scope's half-width and
  half-height by the same factor, so a wide box can only be grown proportionally.
  Edge/corner drag handles on the box outline would resize each axis independently
  (the way the tag-selection transform handles do). Files:
  `frontend/src/coordinators/embedding-coordinator.ts` (`_scaleScope`),
  `frontend/src/coordinators/canvas-interaction.ts`.
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
- [ ] **Prefetch scope.** The `layout` job warms only the library-only,
  default-weights lens set. The external-inclusive variant (`fit_scope=all`) and user
  weight presets are still computed on demand, and every new analysis drain
  re-warms all five lenses (cheap when the revision is unchanged, a full refit when
  it moved). Consider warming `fit_scope=all` too, or gating the sweep behind a size
  threshold. Files: `backend/app.py` (`_prefetch_lens`),
  `backend/embeddings/projection_config.py` (`PROJECTION_LENSES`).
- [ ] **Placed-external jacobian fallback.** With `fit_scope=library` and externals
  shown, the projection extends them by Nyström; but if the chart cannot be fit
  (fewer than two library tracks) the weight-jacobian falls back to a fit-scope refit
  and the external dots drop out until the drag is released. Extend the fallback too,
  or accept the transient. Files: `backend/routes/api_embeddings.py`
  (`_place_into_chart`), `frontend/src/lib/weight-jacobian.ts`.
- [ ] **External placement fidelity is unmeasured.** The Nyström extension places
  external roots onto the library chart, but whether they land where an
  external-inclusive fit would put them is empirical. The harness now exists
  (`backend/embeddings/layout_fidelity.py`); run it on the external tracks' placed
  positions against the library chart. Files:
  `backend/embeddings/landmark_projection.py` (`build_landmark_model`).
- [ ] **Search field resolution.** The field splats the backend's fixed 128-point
  landmark sample, so zooming into a sparse region shows a few large blobs rather than
  detail. Consider a larger sample (or a screen-space grid) when zoomed in; the
  selection is cached by CLAP content, so a bigger sample is a one-time k-means cost.
  Files: `backend/routes/api_search.py` (`_landmark_paths`),
  `frontend/src/components/canvas-view.ts` (`_drawSearchField`).
- [ ] **Search field is layout-independent.** The landmark sample is selected over
  the whole CLAP matrix, so hidden tracks and external roots leave gaps in the field
  even though the displayed layout may differ. Tie the sample to the displayed lens
  revision if the gaps read as noise. Files: `backend/routes/api_search.py`
  (`_landmark_paths`), `frontend/src/components/canvas-view.ts`.

## Deferred / optional

### Frontend

- [ ] **Lens presets are unreachable.** The lens `<select>` and the Save/Delete
      preset buttons were removed from the view panel, so `Model.addLensPreset` /
      `applyLensPreset` / `deleteLensPreset` and `localStorage` persistence are
      dead UI. Re-surface presets in the top lens tabber (a preset row) or delete
      the feature. Files: `frontend/src/model.ts`, `frontend/src/lib/lenses.ts`,
      `frontend/src/components/canvas-view.ts` (`_drawTopTabs`).
- [ ] **The bloom cursor is silent when its variable cannot place a track.** Moving the
      selection-bloom cursor onto *Cluster colour* with no clusters, or *Predicted folder*
      with no head, blooms nothing without saying why — unlike the colour mode's warn toast.
      Either warn on the cursor move or grey the cursor tab. Files:
      `frontend/src/coordinators/analysis-coordinator.ts` (`applySelectionBloom`),
      `frontend/src/components/canvas-view.ts` (`_drawTabCursor`).
- [ ] **Modifier chips: inline editing and keyboard nav.** The `⌘F` popup has no
      arrow-key navigation of the ranked rows (Up/Down + Enter), and a `filters` chip
      is read-only — editing its range still needs the tag panel's slider. Adding both
      makes the chips true view-modifier surfaces. Files:
      `frontend/src/components/search-view.ts`,
      `frontend/src/components/view-modifier-view.ts`,
      `frontend/src/controller.ts` (`_syncFilterModifiers`).
- [ ] **Proximity ranking is pinned to the last inspected track.** The `proximity` chip
      survives the inspect card closing, but re-ranks around that track; re-homing to the
      current selection would decouple it from the card. Files:
      `frontend/src/coordinators/inspect-coordinator.ts`.
- [ ] **Folder action popout is mouse-only.** `#folder-actions` opens on row hover and
      closes on leave; it has no keyboard path (focus the row, open with a key, arrow
      between buttons) and no focus-visible styling. Also the 6 px gap between row and
      popout briefly clears `canvas.hoverTarget` when the pointer crosses it (restored on
      popout `mouseenter`). Files: `frontend/src/coordinators/folder-tree-coordinator.ts`,
      `frontend/src/trackspace.css`.
- [ ] **Alt+click folder exclusions push no chip.** The one-chip-per-action stack is
      pushed from `_runCanvasAction` (the S / Shift+S hotkeys and the row popout), but
      `FolderTreeCoordinator._onFolderLabelActivate`'s Alt+click calls `hideFolderTree`
      directly, so that exclusion has no chip to dismiss. Route it through the same push.
      Files: `frontend/src/coordinators/folder-tree-coordinator.ts`,
      `frontend/src/controller.ts`.
- [ ] **The head modal is mouse-only.** `#head-modal` opens by right-clicking a folder
      inside an External root (or the row popout's *Imprint structure…*); there is no
      keyboard path to open it, and the folder context menu has no arrow-key navigation
      or focus-visible styling. Files:
      `frontend/src/coordinators/folder-tree-coordinator.ts`,
      `frontend/src/coordinators/head-coordinator.ts`.
- [ ] **Flag a head with no centroids or match floors in the info block.**
      `GET /api/supervised/head` reports `has_centroids` and `has_match_thresholds`,
      but the Analysis panel's *Folder head* block ignores both. A head without
      centroids still renders as healthy yet `supervised_predictions` returns
      `available: false`, so *Predicted folder* colouring and fan-out silently do
      nothing. Render a warning row (and point at *Train*) when either is false.
      Files: `frontend/src/coordinators/head-coordinator.ts`,
      `backend/services/analysis_pipeline.py` (`supervised_head_info`).

- [ ] **Residual lens disagreement is real, not gauge noise.** After Procrustes
      alignment the mean pairwise RMSE is still 0.305 (worst pair 0.40): lenses
      genuinely disagree about where tracks belong, which is the point of having
      them. If switching still feels jumpy, the remaining lever is *animating* the
      transition (the shared `animPositions` overlay already does this) rather than
      aligning harder. Do not chase RMSE → 0; that would mean the lenses stopped
      differing. Files: `backend/embeddings/lens_gauge.py`,
      `frontend/src/coordinators/embedding-coordinator.ts`.
- [ ] **Pin the SSE progress contract with a test.** The backend's
      `/api/library/stream` progress event carries a `tracks` batch; a stale
      `frontend/dist` that still read `track` threw on every event (~15k console
      errors, library never loaded) until rebuilt. Add a backend test asserting the
      event shape and a vitest for the controller's handler so a wire change cannot
      land without both sides updated. Files: `backend/services/library_scan.py`,
      `frontend/src/controller.ts`.
- [ ] **Metaball density tuning.** The scatter is one WebGL metaball pass
      (`frontend/src/lib/point-layer.ts`); at far zoom a whole region coalesces into
      one blob, which reads as density but can saturate. The kernel radius and iso
      threshold are module constants (the edge ramp is screen-space derivative-based, so it
      is already ~1 px at any density) — expose the constants (Options → View) if the
      far-zoom look needs live fiddling, or add a density clamp so a very large stack
      does not balloon without bound. A second note: colour is the kernel-weighted
      *mean*, so where many different hues overlap the blob desaturates toward pale;
      if that reads as washed out, switch the composite to a dominant-hue pick (or weight
      the mean toward the densest kernel). Files: `frontend/src/lib/point-layer.ts`.
- [ ] **Top tabbers overflow on a narrow canvas.** `CanvasView._drawTopTabs`
      centres the lens and colour rows without clamping, so on a narrow window the
      8 lens tabs can run off both edges. Wrap to two lines, scroll, or drop to
      the labels' short forms when the row exceeds the canvas width. Files:
      `frontend/src/components/canvas-view.ts` (`_drawTabs`).
- [ ] **No legend for the tempo/key colour modes.** `tempoColor`/`keyColor`
      (`frontend/src/lib/colors.ts`) reuse the lens polar geometry, but the canvas
      gives no key for what a hue or a bright/dark shade means. Add a small legend
      when the colour mode is tempo or key (hue = angle, brightness = radius), or
      fade the mode back to folder when the matching guide is off screen.
- [ ] **Optional sibling hue wiggle.** Family tints currently differ only in
      lightness/saturation (`frontend/src/lib/colors.ts` / `backend/services/dir_colors.py`).
      If same-depth siblings still read too alike on the canvas, add a small hue deviation
      that also decays with depth; bump the shared golden vectors in
      `frontend/src/lib/dir-color-vectors.json` (asserted by both suites).
- [ ] **The map stays interactive while a lens fits.** A lens switch keeps the
      previous lens on screen (with the throbber pill stacked under any analysis
      progress), and pointer gestures still act on the *old* lens's positions. Gate
      selection/drag while `model.projectionPending`, or dim the canvas, so a click
      during the wait cannot act on a stale layout. Files:
      `frontend/src/coordinators/canvas-interaction.ts`,
      `frontend/src/components/canvas-view.ts`.
- [ ] **Hovered search row can go stale on live search.** `SearchView.renderResults`
      rebuilds every row on each debounced query, so a hovered row's `mouseleave`
      may not fire and its glow/card can linger. Clear `canvas.hoveredPath` when the
      list rebuilds. Files: `frontend/src/components/search-view.ts`,
      `frontend/src/controller.ts`.
- [ ] **Inspect card follow-ups.** The pinned card (Alt over a dot) has Navigate
      (a nearest/anti-track toggle + distance-filter slider + result focus) and Describe
      (flat click-to-search CLAP terms) tabs, plus the semantic coordinates next
      to the audio features; its folder/cluster/key/BPM elements
      are navigation targets (glow + S/H/F/Shift+S) — see `design-patterns.md`. Still open:
      "same cluster" / "same folder" jumps from the result rows, a `k` control instead of
      the fixed `NEAR_K`/`DESCRIBE_K`, and showing that the similarity score is a cosine, not a
      probability. The distance filter pre-loads the full ranking on pin, so
      `_fetchSimilar` now builds the whole-library matrix on the first pin of a
      session — a warm-up on library load would hide that latency (cached for
      `CACHE_TTL_SECONDS` after that).
- [ ] **M + scroll is canvas-scoped.** `CanvasInteractionController._onWheel` owns the
      preview-volume gesture, so it only fires over the viewport; holding M and
      scrolling over the sidebar instead falls through to the M tap toggle. Move the
      listener to the document (capture) if the gesture should work anywhere.
      Files: `frontend/src/coordinators/canvas-interaction.ts`.
- [ ] **CLAP description bank is fixed and English-only.** `FACET_BANK`
      (`backend/embeddings/clap_describe.py`) is a ~160-phrase curated
      DJ/electronic list, so the Describe tab only surfaces phrases it contains
      and never this library's own vocabulary. A user-extensible bank, or ranking
      the library's tags/cluster names as candidates, would make it reflect the
      collection.
- [ ] **The semantic axes are hand-authored and reference-calibrated.** `AXES`
      carries a mean/std/reliability measured on the development library (15 408
      cached CLAP vectors), so a materially different collection drifts. Re-fit the
      reference distribution from the live library (cached by CLAP write epoch) when
      that drift shows. A validation harness — new axes must separate known folder
      pairs (Techno vs Ambient, …) at |Cohen's d| ≥ 0.5 — would catch a junk axis
      before it ships, since pole separation alone misleads (`dark <-> bright` has
      near-identical poles yet d = 2.58).
- [ ] **Coordinates use the pooled CLAP mean only.** A section-specific concept
      (a talking-voice interlude, a breakbeat drop) is averaged away by the
      3-segment mean. Scoring each axis against the per-segment vectors and keeping
      the best, tagged with its section, would surface it — the search route
      already does this (`/api/search` best-segment).
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
- [ ] **ApiClient dependency injection.** `Analysis`, `Embedding`, `Head` and
      `FolderTree` coordinators import `lib/api` directly, while `Search` and
      `Inspect` take injectable fetchers — so only the latter are unit-testable
      without `vi.stubGlobal("fetch")`. Introduce one `ApiClient` interface in
      `lib/api.ts`, add it to each coordinator's deps (defaulting to the module
      singleton), and build the four test harnesses from a shared fake. Files:
      `frontend/src/lib/api.ts`, `coordinators/{analysis,embedding,head,folder-tree}-coordinator.ts`
      and their tests.
- [ ] **Structure preview follow-up.** `_fetchStructure` (`controller.ts`) fetches one
      path per hover — a batched request for the visible tracks would cut the
      round-trips.
- [ ] **Shared lens registry.** `frontend/src/lib/lenses.ts` (`BUILTIN_LENSES`) and
      `backend/embeddings/projection_config.py` (`PROJECTION_LENSES`) mirror the
      projection lens ids and source sets, and `tests/test_layout_prefetch.py` pins
      the ids by hand with a "keep in sync" comment. The cluster weights and the
      feature geometry are now shared JSON (`frontend/src/lib/cluster-weights.json`,
      `feature-geometry.json`); do the same for the lens registry, or have the
      backend test parse `lenses.ts`. Files: `backend/embeddings/projection_config.py`,
      `frontend/src/lib/lenses.ts`, `tests/test_layout_prefetch.py`.
- [ ] **CSS base classes.** `trackspace.css` repeats a glass popover shell (~6
      surfaces), a modal backdrop/card (~5), a progress bar (~7), a slider knob (2),
      a scrollbar rule (~12), a truncation triplet (~20) and a list-row hover.
      Extract `.glass`, `.overlay`/`.overlay-card`, `.progress`, `.scroll-thin`,
      `.truncate`, `.list-row`, plus `.tab`/`.chip` base + modifier classes.
      Needs a visual pass (a human verifies taste), so it was deferred.
- [ ] **Tree/row builders.** The head training-folders tree
      (`head-coordinator.ts` `_folderNodeEl`) and the HTMX folder tree share no
      code, and the selected-tracks overlay row and the cluster rows re-build the
      same swatch+label+count row. A shared `buildRow`/`buildTree` would unify
      them; deferred as the smallest payoff of the DOM pass.
- [ ] **Preferences registry.** `model.ts` `saveLS`/`loadLS` list the same ~21
      fields twice with a hand-cast per field, and the four `_migrate*` helpers are
      bespoke. A `PREFERENCES` registry of typed `{key, default, parse, serialize}`
      entries would make both loops; deferred as roughly line-neutral and it risks a
      god-registry if the fields are not truly uniform.
- [ ] **Variable registry.** `COLOR_MODES`/`COLOR_MODE_LABELS` (`lib/lenses.ts`),
      the bloom range keys and max constants (`lib/bloom.ts`) and
      `CLUSTER_WEIGHT_LABELS`/`_DESCRIPTIONS` (`model.ts`) are three parallel
      maps over related ids. One `VARIABLES` table keyed by colour mode would
      remove the risk that a new mode silently misses a bloom slot or label;
      deferred as the three domains (colour by, bloom range, clustering weight) may
      not share enough to justify one registry.

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
- [ ] **`services/analysis_pipeline.py` — split if it grows again.** Handlers are
      already one factory function per stage; the cluster store and the enqueue helpers
      are the natural next seams.
- [ ] **Drop the dead generation helpers.** `coverage.build_generation_work` and
      `clap.batch_ensure_embeddings` have no callers since the durable queue took
      over (the queue's `enqueue_library_work` filter supersedes them); remove them
      and their re-exports in `embeddings/__init__.py` / `layout.py`.
- [ ] **Predicted folders are recomputed per fetch.** `GET /api/supervised/predictions`
      walks the whole library and re-reads every cached `supervised` row on the first
      selection of the *Predicted folder* colour mode (and again after each head
      change). It is a pure function of the cached rows + the head's centroids, so
      cache the payload server-side keyed on the head run id and serve it until the
      head changes. Files: `backend/services/analysis_pipeline.py`
      (`supervised_predictions`).
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
- [ ] **Search follow-ups.** Folder-scoped search (`folder` param) and a minimum-score
  threshold / "select all" affordance. Fit-on-focus is done
  (`CanvasInteractionController.fitViewToPaths`); live search is done
  (`SearchView` debounce, `SEARCH_DEBOUNCE_MS`). Perf: `/api/search` calls the *uncached*
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
- [ ] **MAEST mel front end.** Already batched: `analyze_batch` stacks every
      excerpt of the claimed batch (up to `batch_size * MAEST_NUM_SEGMENTS`) into one
      `_mel(wave)` call, not three per track. The remaining CPU mel cost is the
      front end itself, which would need a different transform or device split.
- [ ] **Re-validate the single-net tempo default.** `TRACKSPACE_MADMOM_FAST`
      defaults to the single BLSTM (~4× faster; 26/29 tracks within 1 BPM), with
      `reconcile_tempo_octave()` guarding the exact half/double lock against librosa;
      `TRACKSPACE_MADMOM_FAST=0` restores the ensemble. `FEATURES_VERSION` was
      bumped to 8 for the recipe change; validate against a labelled tempo set before
      relying on it further.
- [ ] **openTSNE fit cache is cold on the first request after a restart.**
      The fitted embedding + PCA basis persist in `tsne_fit_cache` and the cache is on by
      default, so a restart transforms rather than refits — but the blob is loaded lazily
      on the first request. Preload it at startup, or accept the one cold request.
      Files: `backend/embeddings/projection_backends.py`,
      `backend/embeddings/feature_cache.py`.
- [ ] **Projection backends: re-benchmark after a torch/librosa bump.** mlx-vis only
      wins below ~7k tracks (n=7k: 10.4 s vs sklearn 12.8 s; n=10k: 24.6 vs 19.3;
      n=14k: 45.6 vs 42.0), so the 10k `TRACKSPACE_MLX_TSNE_MAX_POINTS` cap is
      conservative but correct — raising it would regress. Note mlx-vis raises
      `UnboundLocalError` on a pre-reduced float32 matrix (its PCA branch expects the
      raw input); `_project_tsne_fallback` passes the raw composite, so this is latent —
      do not "optimise" by pre-reducing there. Files:
      `backend/embeddings/projection_backends.py`.

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
      retrain via *Train* in the Analysis panel's *Folder head* section once the CLAP
      backfill drains, or its
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
