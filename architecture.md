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
  routes/                 one blueprint per surface (embeddings, jobs, library, folders, tags, search, describe, supervised, imprint, static_audio)
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
    clap_describe.py      zero-shot phrase bank + contrastive semantic coordinates
    feature_cache.py      content-addressed SQLite cache (versioned sources)
    ...                   audio_decode, librosa features, coverage, revision
  templates/partials/     Jinja templates served by Flask (HTMX)
frontend/src/
  controller.ts           wiring: events, view callbacks, API calls, keyboard dispatch
  model.ts                single source of truth (EventBus)
  hotkeys.ts              HotkeyManager + key binding definitions
  command.ts              Command pattern (optimistic UI + undo)
  coordinators/           controller fragments, one concern per file:
                            analysis-coordinator   queue SSE + status polling + cluster panel
                            head-coordinator       head info/train/export/import/compare + imprint + undo
                            embedding-coordinator  layout cache + SSE + position animations
                            folder-tree-coordinator HTMX tree, roots, highlights, overlay
                            canvas-interaction     pointer gestures + transforms + fit view
                            options-panel          status-bar hover-preview control
                            search-coordinator     semantic search: fetch, selection, results panel
                            view-modifier-coordinator  active-view-modifier stack + chips
  components/             one view per file (canvas, tags, properties, batch, status, search)
  lib/toast.ts            shared DOM helpers
  lib/api.ts              shared fetch / postJSON
  lib/util.ts             shared clamp / escape / confirm / cycle helpers
  lib/async.ts            shared debounce / throttle / seq / in-flight / SSE helpers
  lib/lenses.ts           lens registry + deterministic feature geometry
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
| Analysis seam | `backend/services/analysis_pipeline.py` | ~1630 | Job handlers + `ClusterStore` + supervised training/projection/undo |
| Queue | `backend/jobs/` | ~900 | `queue.py` + `worker.py` + `kinds.py` |
| ML / layout | `backend/embeddings/layout.py` | ~565 | Composite gather + orchestration |
| Semantic weighting | `backend/embeddings/semantic_weighting.py` | ~265 | Folder/text directions |
| Weak supervision | `backend/embeddings/supervised.py` | ~1075 | Folder-labelled projection head, match floors, excerpt dispersion |
| Track structure | `backend/embeddings/structure.py` | ~275 | Intro/drop/break from a loudness envelope |
| Projection backends | `backend/embeddings/projection_backends.py` | ~230 | PCA + cuML/mlx-vis/sklearn t-SNE |
| Feature cache | `backend/embeddings/feature_cache.py` | ~560 | One generic `_SOURCES` registry + fan-out journal + origin memory |
| Clustering | `backend/embeddings/clustering.py` | ~560 | |
| Layout fidelity (bench) | `backend/embeddings/layout_fidelity.py` | ~60 | Neighbourhood recall; imported only by tests, not production |
| Frontend orchestration | `frontend/src/controller.ts` | ~840 | Wiring, data loading, tag-write queue, preview, keyboard |
| Analysis coordinator | `frontend/src/coordinators/analysis-coordinator.ts` | ~1040 | Queue SSE + cluster panel + source weights |
| Head coordinator | `frontend/src/coordinators/head-coordinator.ts` | ~750 | Head info + train/export/import/compare, imprint, fan-out undo |
| Embedding coordinator | `frontend/src/coordinators/embedding-coordinator.ts` | ~1080 | Layout cache + SSE + animations + lenses + scopes |
| Folder-tree coordinator | `frontend/src/coordinators/folder-tree-coordinator.ts` | ~675 | HTMX tree + roots + highlights |
| Canvas interaction | `frontend/src/coordinators/canvas-interaction.ts` | ~1200 | Pointer gestures + transforms + scope gesture |
| Options panel | `frontend/src/coordinators/options-panel.ts` | ~130 | Lens select + presets + hover preview |
| Canvas view | `frontend/src/components/canvas-view.ts` | ~1350 | |
| State | `frontend/src/model.ts` | ~1190 | Optional persistence / EventBus split |

`canvas-interaction.ts` is the largest frontend file; the fit-view math is the
natural next split if it needs to shrink (see `TODO.md`).

---

## Frontend architecture: three-layer MVC

The frontend uses **Vite + TypeScript**, **Alpine.js**, and **HTMX**.

- **UI modules:** `frontend/src/components/` — one file per view (`canvas-view.ts`,
  `tag-panel-view.ts`, …). Shared DOM helpers (toast, loading overlay) live in
  `frontend/src/lib/toast.ts`; folder/cluster/tempo/key colours live in `frontend/src/lib/colors.ts`;
  the max-blended glow overlay is `frontend/src/lib/glow-layer.ts`; the metaball
  scatter pass is `frontend/src/lib/point-layer.ts`.
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

- **Co-locate what changes together.**  CLAP, librosa features, FeatureCache,
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
- **Startup backfill.** A source-version bump blanks the stale cache rows on load, but
  the queue still holds their jobs in a terminal state, so an ordinary enqueue would skip
  them: `enqueue_library_work` therefore revives terminal rows for tracks the *cache*
  reports as missing (the cache, not the queue, is the source of truth; `force` only widens
  the filter to already-cached tracks). `main()` then re-queues exactly what the purge
  blanked (`backfill_purged_analysis`, a background thread), so a version bump self-heals.
  A standing schema backlog that is *not* a version purge — e.g. the CLAP segment vectors
  added later — is deliberately not swept into an unrelated restart; it is retriggered from the
  coverage panel, whose per-stage re-run icons now revive terminal jobs too.
- Flask runs with `use_reloader=False`: the reloader would fork a second set of worker
  threads competing for the same queue.
- **Every cache connection goes through `sqlite_conn.ClosingConnection`.** The stdlib
  `with sqlite3.connect(...) as conn:` only ends the transaction and leaves the handle
  open, so the per-access `with self._conn() as conn:` idiom leaked one connection (db +
  wal + shm FDs) per cache call — 112k open `track_cache.db` handles exhausted
  `RLIMIT_NOFILE` and SQLite began failing with "unable to open database file". Each
  cache's `_conn()` wraps the raw connection; do not construct a bare `sqlite3.connect`
  in a `with`.
- **Batch reads before fanning out, and read presence not blobs.** `TrackCache.get_many`
  and `FeatureCache.get_all`/`presence` replace one SQLite connect per track (measured
  ~730 us each, ~10 s per library scan). Layouts persist by `layout_revision` in the
  `layout_cache` table, and coverage/revision tests use `FeatureCache.presence()` (one
  boolean-per-source query) instead of decoding every blob. See `audio_pipeline.md` →
  *Projection caching*.

---

## Library roots and roles

Roots are registered by absolute path in `data/roots.json`; ids are short hashes
recomputed on load (`services/roots_registry.py`). Each root carries a role:

- **`library`** (default) — the user's own tracks; the canvas, tags, search and the
  supervised head's *labels* all use these.
- **`external`** — downloads, reference material, or an imprint target. External roots
  are scanned, analysed and embedded exactly like library roots (the fingerprint-keyed
  `FeatureCache` never cares where a file lives), but they stay out of the default
  projection: `/api/embeddings/projection` walks the library roots only unless the
  request carries `include_external=1`, which the frontend sends when externals are
  shown. Showing externals does **not** refit the map: the fit scope is
  controlled separately by `fit_scope=library|all` (default `library`), so with the
  default the server fits the library chart and places the external tracks onto it by
  Nyström extension (`landmark_projection.build_landmark_model(..., place_infos=…)`).
  The external view-modifier chip's **Refit** toggle sends `fit_scope=all` to opt into
  the old external-inclusive refit. Visibility also gates their drawing
  (`Model.isVisible`), and the head does not learn from their folder labels by
  default.
  Because analysis is content-addressed, marking a root external — or removing it —
  never discards its cached embeddings; re-adding restores it instantly.

Persistence stores only the *paths* whose role is external (`external_roots` in
`roots.json`); library is the default so old files stay valid. The folder-tree partial
renders a muted "External" section, root rows carry `data-root-role` and `data-root-abs`,
and `POST /api/roots/role` flips a root between the two. `POST /api/roots/remove` drops a
root (from the row's hover action popout, or dragging the row out of the sidebar); the files and
their content-addressed embeddings stay, so re-adding restores it. Ephemeral imprint roots
(`temp_roots`) are a separate, shorter-lived concept — they are marked external while alive
and removed when the imprint finishes.

A fan-out can also remember where a moved track used to live: the `origin_folders` table in
the feature cache is keyed by fingerprint (content-addressed like every source), so a recursive
fan-out's pre-move relative folder survives the move journal being pruned.

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
- **Analysis reads are read-only over `FeatureCache`.** `/api/search` (CLAP text
  ranking), `/api/tracks/describe` (zero-shot phrase ranking + semantic
  coordinates against one track's cached CLAP audio vector),
  `/api/tracks/structure` (cached intro/drop/break + genre-spanning flag, plus
  the header-probed track duration the seekbar needs) and
  `/api/tracks/similar` (nearest/anti-track over the composite vectors) never
  re-encode a track's *audio* on a request thread; expensive analysis goes
  through the queue. `/api/search` and `/api/tracks/describe` run only the CLAP
  text tower, under `clap.inference_lock`; describe's phrase bank and axis
  directions are encoded once per process and cached
  (`embeddings/clap_describe.PromptBank` / `AxisDirections`), it drops phrases
  below `MIN_SIMILARITY` so the list is not padded with noise, and its
  coordinates are z-scored against a reference distribution baked into `AXES`, so no
  library scan is needed on a request. `/api/tracks/similar` is the one exception
  to "reads are cheap": it rebuilds the weighted matrix that clustering/layout use,
  cached by `(clustering revision, weights, library mtime)` with a TTL — see
  `services/similarity.py`.
- **`/api/analysis/problems` is the actionable counterpart to the coverage counts.**
  It lists tracks fpcalc cannot fingerprint at all plus per-path terminal job failures
  (aggregated across kinds), so the UI can surface the damaged files behind a short
  coverage bar. `/api/embeddings/generate` accepts a `paths` list to restrict an
  enqueue to specific tracks (the per-row retry).

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
- `embeddings/structure.py` — intro/drop/break detection. A pure envelope detector
  plus a fixed-length cache blob; it decodes through
  `audio_decode.load_resilient_full_mono` (which itself only needs `decode_stderr`), so
  the `structure` stage handler owns the cache write and the thread pool.

**Frontend** — `controller.ts` went from ~2250 to ~700 lines. Extracted, in order, each
with vitest coverage:

- ✅ `lib/api.ts` — centralised `fetch` / `postJSON`; the controller stays the caller.
  Pulled forward so coordinators could import it without a circular dependency
  (`command.ts` shares it, passing `silent` where its manager toasts).
- ✅ `AnalysisCoordinator` — job-queue SSE stream, status polling, cluster fetch/panel,
  and the source-weight sliders.
- ✅ `HeadCoordinator` — the folder-supervised head: the trained-head info block, train
  (and its persisted folder selection), export/import/compare, the imprint flow, and the
  fan-out move undo. The queue reaches it only through `syncTrainEnabled` (and the info
  refresh through `refreshHeadInfo`), so the head panel is independent of the queue +
  cluster panel.
- ✅ `EmbeddingCoordinator` — layout cache, embedding SSE, projection polling, lens
  switching, scope re-layout, and all canvas position animations. Owns the shared
  `canvas.animPositions` overlay and the single animation frame, so tag-axis and
  embedding animations cannot fight over the overlay.
- ✅ `FolderTreeCoordinator` — HTMX refresh/bind, root import/removal, tree interaction,
  highlights, the selected-tracks overlay (tabs: folders / BPMs / keys / clusters).
  Undoable renames and root-driven library reloads stay
  in the controller via deps callbacks (`renameFolder`, `onRootsChanged`).
- ✅ `CanvasInteractionController` — pointer gestures (click/box/lasso select, tag drag,
  transform handles, pan/zoom, folder-drop), scope gesture, fit view. Owns the mouse state;
  exposes `isPointerBusy()` (consumed by the folder coordinator), `cancelActiveGesture()`,
  `fitView()`, `effectiveHoveredFolderPrefix()`. The debounced tag-write queue stays in the
  controller (shared with `BatchView.onApply`) behind `queueTagWrite` / `flushTagWrites` deps.
- ✅ `OptionsPanelBinder` — the status-bar hover-preview control (toggle +
  volume) with `bind()` / `sync()` / `syncPreviewVolume()`. The lens and colour
  controls were removed; the top tabbers are their single surface.
- ✅ `ViewModifierCoordinator` — the active-view-modifier stack (search, proximity, scope,
  exclusions, filters, external) with `push`/`update`/`dismiss`/`remove`/`popNewest`, plus
  the corner chip view. See `design-patterns.md` → *Active view modifiers*.

---

## Non-goals

- Framework swap, microservices, heavy DI frameworks.
- Storing the job queue anywhere but SQLite — the durability is the point.
- Running ML inference inside Flask request handlers: all expensive analysis goes through the
  durable queue.
