# Trackspace — modularization & maintainability plan

This document proposes how to shrink files, reduce repetition, and clarify boundaries without rewriting the product. It builds on the existing layout described in `AGENTS.md` (MVC on the frontend, cache + embedding pipeline on the backend).

## Current scale (indicative)

| Area | File | Lines (approx.) | Role today |
|------|------|-----------------|------------|
| Backend entry + HTTP | `backend/app.py` | ~1.4k | Flask app, globals, routes, FS watcher, SSE, projection params, track JSON |
| ML / layout | `backend/embeddings.py` | ~1.2k | CLAP, semantic weighting, composite vecs, PCA/UMAP/t-SNE, caches |
| Audio features | `backend/audio_features.py` | ~840 | EffNet ONNX, librosa pipeline |
| Frontend orchestration | `frontend/src/controller.ts` | ~2.8k | Everything: API, embeddings UX, canvas input, folder HTMX, sidebar, hotkeys wiring |
| State | `frontend/src/model.ts` | ~680 | Tracks, selection, view mode, localStorage, event bus |

The codebase is already *logically* layered (separate `cache`, `feature_cache`, `embedding_coverage`, `layout_revision`), but **composition roots** (`app.py`, `controller.ts`) have absorbed too many concerns.

---

## Goals

1. **Small files** — each module one primary reason to change.
2. **Thin HTTP layer** — routes delegate to services; no heavy logic in `@app.route` handlers.
3. **No duplicated “paperwork”** — one implementation for layout revision keys, SSE framing, parallel cached reads, etc.
4. **Stable abstractions** — explicit “application context” (backend) and focused coordinator objects (frontend) instead of sprawling globals / god classes.
5. **Incremental migration** — extract pieces without a big-bang rewrite.

---

## Backend: split `app.py`

### 1. Introduce an application context (single place for globals)

Today `ROOTS`, `_CACHE`, `_FEATURES`, `_THREAD_POOL`, `_embed_*`, LRU caches, and path helpers live as module-level state in `app.py`. That makes testing and reuse awkward.

**Suggestion:** a small `AppContext` or `TrackspaceState` dataclass (or namespace) constructed in `main()` and passed into factories:

- `roots`, `track_cache`, `feature_cache`, `executor`
- `embed_status`, `embed_lock`, `embed_subscribers`
- `track_infos_lru`, `status_coverage_lru` (+ their locks)

Routes receive `g.ctx` via `@app.before_request` or a blueprint that closes over context at register time.

This is the enabler for **splitting routes across files** without import cycles.

### 2. Flask Blueprints by domain

| Blueprint | Prefix / scope | Moves out of `app.py` |
|-----------|----------------|------------------------|
| `api_library` | `/api/tracks`, `/api/library/stream`, `/api/tags` | Parallel reads, track list JSON, SSE scan |
| `api_folders` | `/api/folders`, roots, create/rename/reveal, HTMX partial | `_folder_tree`, virtual path resolution used only here |
| `api_tags_mut` | `/api/tracks/tags`, tag rename/delete | Thin wrappers around `backend/tags` |
| `api_embeddings` | `/api/embeddings/*` | Status, generate, stream, projection |
| `static_pages` | `/`, `/assets`, production `DIST_DIR` | Keep tiny |

`app.py` (or `backend/factory.py`) becomes: build context → create Flask app → `register_blueprint` → start watcher / GC / model load — **under ~200 lines**.

### 3. Service modules (no Flask imports)

Extract pure-Python orchestration callable from routes:

| Module | Responsibility |
|--------|----------------|
| `backend/services/library_scan.py` | List MPJs under virtual folder, `as_completed` + `_cached_read_all`, yield progress events |
| `backend/services/track_payload.py` | `_track_dict_from_read`, `_build_track_list`, fingerprint → feature batch fetch |
| `backend/services/virtual_paths.py` | `_virtual_from_abs`, `_resolve`, `_root_and_rel_from_virtual`, `_list_mp3s` family |
| `backend/services/projection_request.py` | `_ProjectionParams`, `_parse_projection_params`, `_feature_mask_as_list`, `_embedding_projection_query_fingerprint` |
| `backend/services/layout_revision_for_status.py` | **Single function** that, given `infos`, `maps`, `proj` (or `None`), returns `layout_revision \| None` |

The last item removes the **duplicated** `compute_layout_revision(...)` block in `api_embeddings_status` (running vs idle branches today repeat the same call with the same arguments).

### 4. Filesystem watcher + SSE helpers

| Module | Responsibility |
|--------|----------------|
| `backend/fs_watch.py` | `_FilesystemHandler`, ingest-new-file thread, remap cache on move |
| `backend/sse.py` | `sse_response(generate_iter)`, shared headers; optional tiny helper for JSON event lines |

Embedding broadcast: `_broadcast_embed_event` moves next to subscriber list or a minimal `EmbeddingBroadcaster` class (subscribe / unsubscribe / publish).

### 5. Split `embeddings.py` (optional second phase)

`embeddings.py` mixes inference, linear algebra, folder semantics, and projection backends. Natural seams:

| New module | Contents |
|------------|----------|
| `embeddings_model.py` | `load_model`, `is_model_ready`, `_inference_lock`, `_embed_single`, `generate_text_embeddings`, `generate_embedding` |
| `embeddings_semantic.py` | Folder directions, tag directions, `_apply_semantic_weighting`, related caches |
| `embeddings_composite.py` | `_gather_composite_vecs`, source normalization, mask/blend parsing helpers (or keep parsers here, HTTP stays in projection_request) |
| `embeddings_projection.py` | `compute_projection`, `_project_*`, revision LRU, cuml/mlx/sklearn branching |

Public API: re-export from `embeddings/__init__.py` (or keep `embeddings.py` as a thin facade importing submodules) so callers (`app` / services) do not churn.

### 6. Repetition to delete early (high ROI, low risk)

- **Layout revision in status:** one helper; delete duplicate `if proj is not None:` blocks.
- **SSE responses:** identical `Response(..., mimetype, headers)` — one factory.
- **Parallel cache reads:** `read_all_tracks_parallel(paths) -> dict[path, dict]` used by `api_tracks`, `api_tags`, and anything else that copies the `ThreadPoolExecutor` + `as_completed` loop.

---

## Frontend: decompose `controller.ts`

The Controller correctly owns “only place that talks to the API” — the problem is **surface area**, not the rule. Prefer **collaborators** over scattering fetch calls into views.

### 1. Extract `lib/api.ts` (or `services/api.ts`)

Move `api`, `postJSON`, base URL/error handling, and typed wrappers:

- `fetchTracks`, `fetchTags`, `postTagUpdates`, `fetchEmbeddingStatus`, `fetchProjection`, `postGenerateEmbeddings`, folder CRUD, roots, etc.

Controller stays the **caller** (wiring + when to call), not the home for URL strings and `fetch` boilerplate.

### 2. `EmbeddingCoordinator` (~400–700 lines)

Owns:

- Layout cache (`_layoutCache`, invalidation, query string builder)
- `_ensureEmbeddingsAndProject`, `_pollModelReady`, stream lifecycle (`EventSource`)
- `_fetchProjection` / incremental PCA / `_applyPositions`
- Mode switch animation hooks that are *embedding-specific* (`_animateToEmbeddings`, etc.)

Controller holds `private embedding = new EmbeddingCoordinator(this)` and forwards `model`/`canvas` references or small interface.

### 3. `FolderTreeCoordinator` (HTMX + DOM)

Owns:

- `_refreshFolderTreeHtmx`, `_bindFolderTree`, `_afterFolderTreeSwap`
- Rename row UX, shift-range selection in tree, `_syncFolderTreeActiveLabels`
- Root import / drag wiring that is sidebar-specific

### 4. `CanvasInteractionController` (pointer + transforms)

Owns:

- `_bindCanvas`, down/move/up/wheel, marquee / transform math, drag-to-folder, `_flushPending` debounce for tags **if** you want all pointer code together; alternatively split “tag drag” vs “navigation”

Keeps **AGENTS.md** rule: views still expose callbacks; this class implements them and calls into model + API.

### 5. `OptionsPanelBinder` (right column)

`_initSourceCheckboxes`, projection toggle, folder-tune sliders, scaling toggles — often edited together when adding a server flag.

### 6. `model.ts` trim

- Move `EventBus` to `lib/event-bus.ts` (optional).
- If model grows further, split **persistence** (`loadLS` / `saveLS` / keys) into `model/persistence.ts` — keeps `Model` focused on fields + derived helpers.

### 7. Shared API types

Today `Track` lives in `model.ts`; API responses are mostly implicit. A small `api/types.ts` with interfaces mirroring `/api/*` JSON reduces coupling and documents drift when the backend changes.

---

## Cross-cutting improvements

### Request / response contracts

- **Minimal:** document query param matrices for projection + status in one markdown table (or OpenAPI YAML) maintained next to `projection_request.py`.
- **Stronger:** generate TypeScript types from the same schema (e.g. openapi-typescript) — only worth it if you expect frequent API churn.

### Testing hooks

After `AppContext` exists, tests can build context with temp dirs and fake/small caches without starting the full watcher. Today, routing and globals are entangled.

### Concurrency

`_embed_lock` and `_inference_lock` are critical. If services multiply, **document which operations must hold which lock** in one place (module docstring on `embeddings_model.py` or a short `CONCURRENCY.md` section inside this plan file’s appendix — only if the team wants it).

---

## Suggested migration order

1. **Backend:** `layout_revision` helper for status + SSE factory + parallel read helper (no behavior change).
2. **Backend:** `virtual_paths` + `track_payload` modules; `app.py` imports them (still one file).
3. **Backend:** Blueprints + `AppContext` registration.
4. **Frontend:** `lib/api.ts` + swap Controller internals to use it.
5. **Frontend:** `EmbeddingCoordinator` (biggest readability win).
6. **Backend:** Split `embeddings.py` submodules when you next touch that area heavily.

Each step should be one PR / logical commit with tests still green (`pytest`, manual smoke on folder tree + embedding mode).

---

## What not to do (yet)

- **Full framework swap** (FastAPI, etc.) — little payoff vs cost.
- **Microservices** — single-process fits the workload (GPU locks, local caches).
- **Over-abstract DI containers** — a `dataclass` context + plain functions is enough.

---

## Summary

Shrink the two god modules (`app.py`, `controller.ts`) by **moving code along domain lines** (library, folders, embeddings, canvas) and **centralizing duplicated utilities** (layout revision for status, SSE boilerplate, parallel reads). Preserve the existing MVC rule: views stay dumb, controller (or its coordinators) remains the integration point — just split that integration across a few small, named types.

---

## Session-derived architecture additions (multi-root + decode reliability)

These are concrete follow-ups from regressions observed during this session.

### 1. Make “focused folder” explicit and non-scoping

The `model.folder` name still reads as “active library scope” and has repeatedly
caused UX confusion (`N tracks in /GENRES`, implicit first-root focus).

**Add a focused-folder abstraction:**

- Rename `model.folder` → `focusedFolderPath` (or `activeFolderPath`).
- Keep invariant: this field is **UI focus only** (tree highlight, default parent,
  selection anchors), never a data-scope filter.
- Ban auto-assignment during HTMX swaps; focus changes only on explicit user action.
- Add a tiny helper module (e.g. `frontend/src/folder-focus.ts`) exposing:
  `setFocusedFolder`, `clearFocusedFolder`, `syncFocusedLabelClass`.

This clarifies intent and prevents accidental reintroduction of folder-scoped view logic.

### 2. Separate “scope query params” from “tree UI params”

We currently mix concepts named like `active`, `active_folders`, and `folder`,
where some are rendering hints and others used to be load scope.

**Introduce explicit param contracts:**

- `ui_active_folder` / `ui_active_folders`: used only for tree partial rendering.
- `scope_folder` (currently always empty for full-library): used only by APIs that
  truly support scoping.
- Frontend request builders should enforce this split so a tree refresh can never
  accidentally alter library/embedding scope.

This is low effort and reduces future regressions when adding endpoints.

### 3. Centralize decode-warning policy by decode mode

The false positives came from comparing intentionally bounded excerpts (~20s) with
full-file metadata duration.

**Extract one policy function** (backend, e.g. `decode_reliability.py`) that takes:

- `mode`: `full_decode` | `partial_window` | `segment_seek`
- `meta_seconds`, `decoded_seconds`, `requested_seconds?`

and returns `warning | None`.

Rules become mode-aware:

- For `partial_window`, compare decoded length against requested window.
- For `full_decode`, compare decoded length against metadata duration.
- Keep one shared message formatter.

Then route all CLAP/EffNet/audio-feature loaders through this helper.

### 4. Add a “no hidden scope” invariant test set

Create lightweight integration tests that lock in global multi-root behavior:

- With multiple roots, `/api/library/stream?folder=&recursive=1` returns tracks from all roots.
- Embedding status/projection with `folder=&recursive=1` use all roots.
- UI status text never displays implied folder scope in default view.
- Tree swap does not auto-focus first root.

These tests should be cheap but protect against accidental re-scoping.

### 5. Introduce a dedicated roots domain service

Roots behavior now includes add/remove, parent-child merge, persisted state,
and virtual path mapping. It has enough complexity to deserve a boundary.

**Proposed module:** `backend/services/roots_registry.py`

- Canonical root list + merge rules (parent absorbs children).
- Load/save `data/roots.json`.
- Virtual path helpers (`root_id/rel`) integration points.
- Validation for add/remove operations.

This pairs naturally with the earlier `virtual_paths.py` extraction and keeps root
lifecycle logic out of route handlers.
