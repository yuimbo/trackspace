# Trackspace — architecture & backlog

Current structure and the work that remains. **Design rules and behavioural
invariants live in `AGENTS.md`** — this file is about shape and scale, not
semantics. Concrete actionable tasks live in `TODO.md`.

## Where things live

```
backend/
  app.py                  CLI entry + wiring (helpers, ANALYSIS_DEPS, worker registration)
  factory.py              create_app() + blueprint registration
  trackspace_state.py     TrackspaceState: caches, roots, queue, workers, clusters
  routes/                 one blueprint per surface (embeddings, jobs, library, folders, tags, static_audio)
  jobs/                   durable queue: kinds, queue (leases/retry), worker (threads)
  services/
    analysis_pipeline.py  job handlers + ClusterStore (queue ↔ models seam)
    ...                   roots registry, path helpers, status caches, payload builders
  embeddings/
    layout.py             composite vectors, semantic weighting, 2-D projection
    maest.py              primary genre model: 519 logits + 768-D embedding
    clustering.py         weighted blocks → PCA → cosine kNN → multi-resolution Leiden
    rhythm_features.py    14-D rhythm/DSP descriptors
    clap.py               CLAP audio + text encoder
    feature_cache.py      content-addressed SQLite cache (6 vector sources)
    ...                   effnet, audio_decode, librosa features, coverage, revision
frontend/src/
  controller.ts           orchestration: events, API calls, queue/SSE, clusters
  model.ts                single source of truth (EventBus)
  components/             one view per file (canvas, tags, properties, batch, status)
```

## Scale snapshot (indicative)

| Area | File | Lines | Notes |
|------|------|-------|-------|
| Backend entry + wiring | `backend/app.py` | ~640 | Helpers, deps assembly, `main()` |
| Process state | `backend/trackspace_state.py` | ~75 | `TrackspaceState.create()` |
| App factory | `backend/factory.py` | ~170 | `create_app()` + blueprints |
| Analysis seam | `backend/services/analysis_pipeline.py` | ~540 | Job handlers + `ClusterStore` |
| Queue | `backend/jobs/` | ~900 | `queue.py` + `worker.py` + `kinds.py` |
| ML / layout | `backend/embeddings/layout.py` | ~940 | **Largest backend module** |
| Feature cache | `backend/embeddings/feature_cache.py` | ~790 | 6 sources × ~6 near-identical methods |
| Clustering | `backend/embeddings/clustering.py` | ~560 | |
| Frontend orchestration | `frontend/src/controller.ts` | ~3140 | **Largest remaining refactor target** |
| Canvas view | `frontend/src/components/canvas-view.ts` | ~1010 | |
| State | `frontend/src/model.ts` | ~730 | Optional persistence / EventBus split |

## Backend — remaining modularisation

1. **`feature_cache.py` — de-duplicate the per-source methods (highest value).**
   Six sources now repeat the same `get / put / has / get_all / count / purge`
   block (~790 lines total, largely mechanical). A small generic descriptor
   (blob column, version column, LRU prefix, `extra_columns`) would collapse it.
   The MAEST pair is the exception: embedding + logits share one version column
   and are written by a single `put_maest()` statement, so the abstraction must
   allow a source to own more than one column under one version.

2. **`layout.py` — extract the two remaining seams.**
   Semantic weighting (folder centroid decomposition, text fallbacks, low-rank
   application) and the projection backends (PCA / cuML / mlx-vis / sklearn t-SNE)
   are each several hundred lines and change for different reasons.

3. **`analysis_pipeline.py` — split if it grows again.**
   Handlers are already one factory function per stage; the cluster store and the
   enqueue helpers are the natural next seams.

4. **Replace unbounded process-lifetime globals in `layout.py`.**
   `_pca_ref_paths` / `_pca_ref_coords` grow without limit. The tier caches
   (`_composite_cache`, `_semantic_basis_cache`, `_revision_layout_cache`,
   `_source_row_cache`) are already bounded; fold the PCA reference alignment
   state into the same bounded-cache pattern with hit/miss/eviction counters.

5. **`TrackspaceState` lifecycle (optional).**
   `TrackspaceState.create()` still runs at import time alongside `create_app()`.
   If needed: build state after CLI parse in `main()`, or accept
   `create_app(state=...)` so tests can inject a scratch state.

## Frontend — remaining modularisation

`controller.ts` is the largest file and the main target. The seams, in the order
they are worth extracting:

- **`AnalysisCoordinator`** — job-queue SSE stream, status polling, cluster fetch
  and the cluster panel. This is the newest and most self-contained block.
- **`EmbeddingCoordinator`** — layout cache, embedding SSE, projection polling,
  embedding-mode animations.
- **`FolderTreeCoordinator`** — HTMX refresh/bind and tree-only UX.
- **`CanvasInteractionController`** — pointer handling + transforms.
- **`OptionsPanelBinder`** — right-column controls that change with server flags.
- **`lib/api.ts`** — centralise `fetch` / `postJSON` and URL constants; the
  controller stays the caller.
- Optional: `lib/event-bus.ts`, `model/persistence.ts`, `api/types.ts`.

## Product roadmap (analysis)

Ordered roughly by expected value. Each is additive to the cached feature store —
none require re-running an encoder over the library.

- **Weak supervision** (attack plan §12–14). Folders and playlists are noisy
  labels; train a small 128-D projection head on the cached vectors. Cached
  `MaestAnalysis.excerpt_embeddings` also make hard-negative mining cheap.
- **Per-excerpt analysis.** Excerpt vectors are cached but unused — they can flag
  genre-spanning tracks (intro vs drop) with no new inference.
- **Text search** (plan §11). CLAP's text encoder already produces the semantic
  directions used for weighting; a query box could reuse it for retrieval.
- **Interactive weights** (plan §16). `cluster_tracks` already accepts a weights
  dict, so exposing sliders re-runs only the kNN/Leiden layer.
- **MERT-v2** (plan §10). Deliberately deferred until MAEST shows a measured
  blind spot — on the 6-genre validation it reached 96.6% purity.
- **Persist cluster results.** `ClusterStore` is in-memory; a restart recomputes in
  seconds at current library size. Persist only if that stops being true.

## Non-goals

- Framework swap, microservices, heavy DI frameworks.
- Storing the job queue anywhere but SQLite — the durability is the point.
