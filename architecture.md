# Trackspace — Architecture

System shape and structure: what lives where, how the layers talk, and the
conventions that keep them coherent. Behavioural/UI rules live in
`ui_conventions.md`; the audio/ML pipeline lives in `audio_pipeline.md`.

---

## Where things live

```
backend/
  app.py                  CLI entry + wiring (helpers, ANALYSIS_DEPS, worker registration)
  factory.py              create_app() + blueprint registration
  trackspace_state.py     TrackspaceState: caches, roots, queue, workers, clusters
  routes/                 one blueprint per surface (embeddings, jobs, library, folders, tags, static_audio)
  tags.py                 ID3 tag read/write via mutagen
  cache.py                TrackCache (path+mtime → metadata, LRU + SQLite)
  fingerprint.py          Chromaprint audio fingerprinting (fpcalc)
  audio_features.py       shim → ``backend.embeddings.audio_features`` (barrel module)
  jobs/                   durable analysis queue: kinds, queue (leases/retry), worker (threads)
  services/
    analysis_pipeline.py  job handlers + ClusterStore (queue ↔ models seam)
    request_validation.py ApiValidationError + payload helpers
    ...                   roots registry, path helpers, status caches, payload builders
  embeddings/
    layout.py             composite vectors + compute_projection orchestration
    semantic_weighting.py folder centroid decomposition + CLAP text directions
    projection_backends.py PCA preview + cuML / mlx-vis / sklearn t-SNE
    maest.py              primary genre model: 519 logits + 768-D embedding
    clustering.py         weighted blocks → PCA → cosine kNN → multi-resolution Leiden
    supervised.py         folder-supervised 128-D projection head (weak supervision)
    rhythm_features.py    14-D rhythm/DSP descriptors
    clap.py               CLAP audio + text encoder
    feature_cache.py      content-addressed SQLite cache (7 vector sources)
    ...                   effnet, audio_decode, librosa features, coverage, revision
  templates/partials/     Jinja templates served by Flask (HTMX)
frontend/src/
  controller.ts           wiring: events, view callbacks, API calls, keyboard dispatch
  model.ts                single source of truth (EventBus)
  hotkeys.ts              HotkeyManager + key binding definitions
  command.ts              Command pattern (optimistic UI + undo)
  coordinators/           controller fragments, one concern per file:
                            analysis-coordinator   queue SSE + status polling + cluster panel
                            embedding-coordinator  layout cache + SSE + position animations
                            folder-tree-coordinator HTMX tree, roots, highlights, overlay
                            canvas-interaction     pointer gestures + transforms + fit view
                            options-panel          right-column controls
  components/             one view per file (canvas, tags, properties, batch, status)
  lib/toast.ts            shared DOM helpers
  lib/api.ts              shared fetch / postJSON
tests/                    pytest suite (backend); vitest lives beside frontend sources
data/                     SQLite caches (gitignored, created at runtime)
```

---

## Scale snapshot (indicative)

| Area | File | Lines | Notes |
|------|------|-------|-------|
| Backend entry + wiring | `backend/app.py` | ~640 | Helpers, deps assembly, `main()` |
| Process state | `backend/trackspace_state.py` | ~75 | `TrackspaceState.create()` |
| App factory | `backend/factory.py` | ~170 | `create_app()` + blueprints |
| Analysis seam | `backend/services/analysis_pipeline.py` | ~870 | Job handlers + `ClusterStore` + supervised training/projection |
| Queue | `backend/jobs/` | ~900 | `queue.py` + `worker.py` + `kinds.py` |
| ML / layout | `backend/embeddings/layout.py` | ~565 | Composite gather + orchestration |
| Semantic weighting | `backend/embeddings/semantic_weighting.py` | ~265 | Folder/text directions |
| Weak supervision | `backend/embeddings/supervised.py` | ~655 | Folder-labelled projection head |
| Projection backends | `backend/embeddings/projection_backends.py` | ~230 | PCA + cuML/mlx-vis/sklearn t-SNE |
| Feature cache | `backend/embeddings/feature_cache.py` | ~365 | One generic `_SOURCES` registry |
| Clustering | `backend/embeddings/clustering.py` | ~560 | |
| Frontend orchestration | `frontend/src/controller.ts` | ~620 | Wiring, data loading, tag-write queue, preview, keyboard |
| Analysis coordinator | `frontend/src/coordinators/analysis-coordinator.ts` | ~310 | Queue SSE + clusters |
| Embedding coordinator | `frontend/src/coordinators/embedding-coordinator.ts` | ~680 | Layout cache + SSE + animations |
| Folder-tree coordinator | `frontend/src/coordinators/folder-tree-coordinator.ts` | ~675 | HTMX tree + roots + highlights |
| Canvas interaction | `frontend/src/coordinators/canvas-interaction.ts` | ~1040 | Pointer gestures + transforms |
| Options panel | `frontend/src/coordinators/options-panel.ts` | ~90 | Right-column controls |
| Canvas view | `frontend/src/components/canvas-view.ts` | ~1010 | |
| State | `frontend/src/model.ts` | ~730 | Optional persistence / EventBus split |

`canvas-interaction.ts` is the largest frontend file; the fit-view math is the
natural next split if it needs to shrink (see `TODO.md`).

---

## Frontend architecture: three-layer MVC

The frontend uses **Vite + TypeScript**, **Alpine.js**, and **HTMX**.

- **UI modules:** `frontend/src/components/` — one file per view (`canvas-view.ts`,
  `tag-panel-view.ts`, …). Shared DOM helpers (toast, loading overlay, `dirColor`) live in
  `frontend/src/lib/toast.ts`.
- **Alpine.js:** Declarative islands in `frontend/index.html` (e.g. shortcuts modal
  `x-data`, `x-cloak`). Heavier logic stays in TypeScript (controller, coordinators, canvas).
- **HTMX:** The folder sidebar list is HTML from Flask (`GET /partials/folder-tree`) loaded
  via `htmx.ajax()` in `FolderTreeCoordinator.refresh()`. Jinja template:
  `backend/templates/partials/folder_tree.html`. In dev, Vite proxies `/partials` to Flask
  (port 5111).
- **Production:** `npm run build` → `frontend/dist/`; Flask serves static assets and keeps
  serving `/partials/...` for any future HTMX fragments.

```
model.ts                    – single source of truth, extends EventBus
components/*-view.ts        – imperative views (canvas, tags, properties, batch, status)
coordinators/*.ts           – controller fragments (analysis, embedding, folder tree,
                              canvas interaction, options panel)
controller.ts               – wires events, view callbacks, API calls, keyboard dispatch
```

**Views never mutate the model directly.** They expose named callbacks
(`onPickFolder`, `onHoverTrack`, …) that the controller implements.

**The controller and its coordinators are the only places that read DOM events and
call API endpoints** (except Alpine handling for purely local UI like the F1 shortcuts
modal). Each coordinator owns one concern and takes a structural `deps` object, so it
is unit-testable without the DOM-heavy views — see `coordinators/*.test.ts`.

### Event bus: two granularities of change

The model is its own event bus (`on` / `emit`).  Two event levels are used:

| Event | When | Who listens |
|-------|------|-------------|
| `"change"` | Selection changed, folder switched, tracks loaded — anything that needs a full UI refresh | All views |
| `"tags-dirty"` | Tag values mutated in memory during an in-flight drag — no structural change | BatchView only (`renderDirty`) |

**Rule:** if you mutate tag values without committing to disk (e.g. live drag), emit
`"tags-dirty"` — not `"change"`. Emitting `"change"` during every animation frame is
expensive and resets state (e.g. BatchView snapshots, props scroll position).

Never emit `"change"` inside a tight animation loop.

---

## Backend: packages, imports, and refactors

Principles that emerged from consolidating the embedding / audio-feature stack into
`backend/embeddings/` and similar cleanups:

- **Co-locate what changes together.**  CLAP, EffNet, librosa features, FeatureCache,
  coverage, layout revision, and 2-D layout share caches, versions, and import edges — a
  dedicated subpackage keeps the mental model and grep scope small.

- **No import-time side effects.**  `backend/app.py` builds `TrackspaceState` and the
  Flask app inside `build_app(state)` / `main()`, not at module level. Importing a module
  must not create caches, start threads, or purge data — that both surprises operators and
  blocks test injection. When a lifecycle helper must close over `state`, expose it (e.g.
  `app.extensions["trackspace_lifecycle"]`) rather than reintroducing a module global.
  Regression test: `tests/test_app_lifecycle.py`.

- **Resolve the “package vs module file” clash.**  A directory `embeddings/` cannot
  coexist with `embeddings.py`; the heavy layout/projection implementation lives in a
  clearly named module (`layout.py`) inside the package, and `embeddings/__init__.py`
  defines the public façade.

- **Preserve stable import paths with thin shims.**  When a widely used entry point moves
  (e.g. `from backend import audio_features`), a tiny module at the old path that
  delegates to the new one avoids churn across `app`, tests, and docs.

- **Lazy package exports when import cost differs wildly.**  If loading
  `package/__init__.py` would pull Torch, projection backends, or other heavy stacks,
  but some callers only need a small submodule, use module-level `__getattr__` (or
  import submodules only) so cheap imports stay cheap.

- **Tests that touch internals should name the implementation module.**  Facades and lazy
  `__init__` may not expose `_cache` globals; import `backend.embeddings.layout` (or the
  relevant submodule) when asserting on tier caches and similar details.

- **Short names inside a package; drop repeated prefixes.**  e.g. `coverage.py` under
  `embeddings/`, not `embedding_coverage.py` — the package path already supplies context.

- **Delete dead code deliberately.**  Unused helpers and redundant `__all__` lists (only
  relevant for `from module import *` or some re-export tooling) add noise; remove them
  when nothing references them.

---

## Process state & lifecycle

`TrackspaceState` (`backend/trackspace_state.py`) owns the caches, roots registry, job
queue, worker pool, and cluster store. `main()` builds it after CLI parsing and
`build_app(state)` wires helpers, workers, and blueprints around it.

- Importing `backend.app` creates **no** SQLite caches, thread pool, or startup purges.
- Tests inject a scratch state instead of mutating a module global
  (`tests/test_app_lifecycle.py`).
- Root-watch / root-loading hooks are returned via
  `app.extensions["trackspace_lifecycle"]` so `main()` needs no module global.
- Flask runs with `use_reloader=False`: the reloader would fork a second set of worker
  threads competing for the same queue.

---

## HTTP API: validation and error shape

- **Every request is validated in one place.**  `backend/services/request_validation.py`
  holds `ApiValidationError` plus helpers (`parse_json_body`, `require_str`,
  `opt_int`, `one_of`, …). Routes parse, validate, and hand plain values to the domain
  layer — never index a payload (`data["path"]`) or coerce with bare `int()`.
- **Every failure is JSON.**  `factory._register_error_handlers` maps
  `ApiValidationError` → 400, `HTTPException` → its status, `OSError` → 500, and any
  other exception → 500, always as `{"ok": false, "error": "…"}`.  Do not add
  per-route HTML error paths; raise (or `abort`) and let the handler shape it.
- **Batch IO keeps partial success.**  A per-file failure (unwritable ID3 frame, missing
  file) is collected in an `errors` array and the rest of the batch still commits —
  see `/api/tracks/tags` and `/api/tracks/move`.  The frontend's `api()` /
  `postJSON` read `error` from the JSON body into the toast, so messages must stay
  client-safe (never include a traceback or absolute paths).

---

## Modularisation log

Structural refactors already completed. Kept as a record of where the seams landed.

**Backend**

- ✅ `TrackspaceState` lifecycle: no import-time side effects; `main()` builds the state
  and `build_app(state)` wires around it (see *Process state & lifecycle*).
- `analysis_pipeline.py` — split again only if it grows; handlers are one factory function
  per stage, so the cluster store and enqueue helpers are the next seams. The
  `supervise` stage (folder-supervised head) was added as one more handler plus the
  supervised projection helpers, without changing the existing seams.
- `embeddings/supervised.py` — the weak-supervision module. Pure functions plus a
  small persistence layer; it depends only on `embeddings/*` (never on
  `analysis_pipeline`), so the job handler owns the cache gathering and the module
  stays unit-testable.

**Frontend** — `controller.ts` went from ~2250 to ~620 lines. Extracted, in order, each
with vitest coverage:

- ✅ `lib/api.ts` — centralised `fetch` / `postJSON`; the controller stays the caller.
  Pulled forward so coordinators could import it without a circular dependency
  (`command.ts` shares it, passing `silent` where its manager toasts).
- ✅ `AnalysisCoordinator` — job-queue SSE stream, status polling, cluster fetch/panel.
- ✅ `EmbeddingCoordinator` — layout cache, embedding SSE, projection polling, and all
  canvas position animations. Owns the shared `canvas.animPositions` overlay and the single
  animation frame, so tag-axis and embedding animations cannot fight over the overlay.
- ✅ `FolderTreeCoordinator` — HTMX refresh/bind, root import/removal, tree interaction,
  highlights, selected-folders overlay. Undoable renames and root-driven library reloads stay
  in the controller via deps callbacks (`renameFolder`, `onRootsChanged`).
- ✅ `CanvasInteractionController` — pointer gestures (click/box/lasso select, tag drag,
  transform handles, pan/zoom, folder-drop), explode-hold key, fit view. Owns the mouse state;
  exposes `isPointerBusy()` (consumed by the folder coordinator), `cancelActiveGesture()`,
  `fitView()`, `effectiveHoveredFolderPrefix()`. The debounced tag-write queue stays in the
  controller (shared with `BatchView.onApply`) behind `queueTagWrite` / `flushTagWrites` deps.
- ✅ `OptionsPanelBinder` — right-column controls (view mode, hover preview + volume,
  explode strength) with `bind()` / `sync()`.

---

## Non-goals

- Framework swap, microservices, heavy DI frameworks.
- Storing the job queue anywhere but SQLite — the durability is the point.
- Running ML inference inside Flask request handlers: all expensive analysis goes through the
  durable queue.
