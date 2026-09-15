# Trackspace — engineering TODO

Concrete, actionable work and deferred items. Design rules live in the docs:
`architecture.md`, `ui_conventions.md`, `audio_pipeline.md`.

## Deferred / optional

### Frontend

- [ ] **Optional: split `coordinators/canvas-interaction.ts`** (~1040 lines, the
      largest frontend file). The ~200 lines of fit-view math (`_vpFromWorldBounds`,
      `_computeFitVP`, `_p80Radius`, `_fitViewToFolderPercentile`, `_fitViewToTracks`,
      `fitView`, `effectiveHoveredFolderPrefix`) are the natural seam into a
      `canvas-fit-view.ts`.
- [ ] **Optional: `lib/event-bus.ts`, `model/persistence.ts`, `api/types.ts`.**
      Extract the EventBus base, localStorage persistence, and shared API response
      types out of `model.ts` / `lib/api.ts`.
- [ ] **Pre-existing: unused locals under strict tsc.** `--noUnusedLocals` flags an
      unused `m` in `coordinators/embedding-coordinator.ts:197` and an unused `model`
      in `embedding-coordinator.test.ts:288`. Remove.
- [ ] **Nit:** `clamp` is duplicated in `coordinators/options-panel.ts` and
      `coordinators/canvas-interaction.ts`.

### Backend

- [ ] **`services/analysis_pipeline.py` — split if it grows again.** Handlers are
      already one factory function per stage; the cluster store and the enqueue helpers
      are the natural next seams.

## Product roadmap (analysis)

Ordered roughly by expected value. Each is additive to the cached feature store —
none require re-running an encoder over the library.

- **Weak supervision.** Folders and playlists are noisy labels; train a small 128-D
  projection head on the cached vectors. Cached `MaestAnalysis.excerpt_embeddings`
  also make hard-negative mining cheap.
- **Per-excerpt analysis.** Excerpt vectors are cached but unused — they can flag
  genre-spanning tracks (intro vs drop) with no new inference.
- **Text search.** CLAP's text encoder already produces the semantic directions used
  for weighting; a query box could reuse it for retrieval.
- **Interactive weights.** `cluster_tracks` already accepts a weights dict, so exposing
  sliders re-runs only the kNN/Leiden layer.
- **MERT-v2.** Deliberately deferred until MAEST shows a measured blind spot — on the
  6-genre validation it reached 96.6% purity.
- **Persist cluster results.** `ClusterStore` is in-memory; a restart recomputes in
  seconds at current library size. Persist only if that stops being true.
