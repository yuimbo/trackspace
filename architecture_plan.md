# Trackspace — architecture backlog

This file lists **remaining** modularisation work. It replaces the old “plan + migration checklist” now that several backend extractions and the embeddings blueprint are landed. Product architecture is still described in `AGENTS.md`.

## Scale snapshot (indicative)

| Area | File | Lines (approx.) | Notes |
|------|------|-----------------|--------|
| Backend entry + HTTP | `backend/app.py` | ~960 | Still owns most routes, roots, FS watcher, caches |
| ML / layout | `backend/embeddings.py` | ~1245 | Still a god module; split is optional/phase 2 |
| Frontend orchestration | `frontend/src/controller.ts` | ~2920 | Largest remaining refactor target |
| State | `frontend/src/model.ts` | ~685 | Optional persistence / EventBus split |

## Backend — still to do

### 1. `AppContext` / `TrackspaceState` (replace sprawling `app.py` globals)

Collapse module-level singletons (`ROOTS`, caches, executor, embed locks, observer, …) into one object built in `main()` and passed into blueprint factories (or hung off `g.ctx`).

**Why:** tests and future blueprints avoid import-order hazards and implicit globals.

### 2. Blueprints for the rest of the API (besides embeddings)

Embeddings live in `backend/routes/api_embeddings.py` (`/api/embeddings/*`). Still in `app.py`:

- **Library:** `/api/tracks`, `/api/library/stream`, `/api/tags`
- **Folders + HTMX:** `/partials/folder-tree`, `/api/folders*`, roots add/remove, reveal, track moves
- **Tag mutation:** `/api/tracks/tags`, `/api/tags/*`
- **Static:** `/`, `/assets/*`, `/api/audio/*` (or keep static/audio in a tiny blueprint)

Target: `app.py` / `backend/factory.py` only wires context + `register_blueprint` + lifecycle (watcher, GC, model warmups).

### 3. Extract filesystem watcher

Move `_FilesystemHandler` + ingest thread + root watch scheduling from `app.py` to e.g. `backend/fs_watch.py` (no Flask imports).

### 4. Optional: split `backend/embeddings.py`

Suggested seam (only when that file is actively being edited):

- Model / inference
- Semantic weighting
- Composite vector assembly
- Projection backends + caches

Keep `backend/embeddings.py` as a thin facade re-exporting public symbols if you want minimal churn for imports.

### 5. Roots registry service

Promote roots merge rules + `data/roots.json` load/save + validation from `app.py` into `backend/services/roots_registry.py`, pairing with `backend/services/virtual_paths.py`.

### 6. Decode-warning policy consolidation

Single mode-aware helper (e.g. `decode_reliability.py`) for `full_decode` vs `partial_window` vs `segment_seek`, used by CLAP / EffNet / audio-feature paths.

### 7. “No hidden scope” tests

Light integration tests for multi-root: full-library listing, embedding status/projection with empty folder, and UI-focused assertions where feasible.

## Frontend — still to do

- `lib/api.ts` (or `services/api.ts`): centralise `fetch` / `postJSON` and URL constants; controller stays the caller.
- `EmbeddingCoordinator`: layout cache, SSE, projection polling, embedding-mode animations.
- `FolderTreeCoordinator`: HTMX refresh/bind, tree-only UX.
- `CanvasInteractionController`: pointer + transforms (+ optional tag-drag grouping).
- `OptionsPanelBinder`: right-column controls that change with server flags.
- Optional: `lib/event-bus.ts`, `model/persistence.ts`, `api/types.ts`.

## Non-goals

- Framework swap, microservices, heavy DI frameworks — same as before.
