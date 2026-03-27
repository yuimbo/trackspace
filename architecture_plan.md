# Trackspace — architecture backlog

This file lists **remaining** modularisation work. HTTP routes live under `backend/routes/`; `backend/factory.py` provides `create_app()`; `backend/trackspace_state.py` holds `TrackspaceState` (caches, roots, embed/SSE, pool, watcher handles); `backend/services/roots_registry.py` owns root ids / merge rules / `roots.json` I/O; `backend/app.py` wires helpers and CLI; the watchdog handler lives in `backend/fs_watch.py`. Product architecture is still described in `AGENTS.md`.

## Scale snapshot (indicative)

| Area | File | Lines (approx.) | Notes |
|------|------|-----------------|--------|
| Backend entry + wiring | `backend/app.py` | ~490 | Helpers, `TrackspaceBlueprintDeps`, `main()`; uses `state` |
| Process state | `backend/trackspace_state.py` | ~70 | `TrackspaceState.create()` |
| App factory | `backend/factory.py` | ~150 | `create_app()` + blueprint registration |
| ML / layout | `backend/embeddings/` (see `layout.py`) | ~920+ | Package: composite layout + projection; CLAP in `clap.py`; audio in `audio_decode` / `effnet` / `librosa_audio_features` |
| Frontend orchestration | `frontend/src/controller.ts` | ~2920 | Largest remaining refactor target |
| State | `frontend/src/model.ts` | ~685 | Optional persistence / EventBus split |

## Backend — still to do

### 1. TrackspaceState lifecycle (optional)

`TrackspaceState.create()` still runs at import time alongside `create_app(...)`; `state` is also on `app.extensions["trackspace_state"]`. Further work if needed: build state after CLI parse in `main()`, or expose `create_app(state=...)` for tests that inject a fresh state.

### 2. Optional: split `backend/embeddings/layout.py` further

CLAP already lives in `embeddings/clap.py`. Remaining seams if `layout.py` grows again: semantic weighting vs projection backends (each ~300+ LOC).

### 3. Decode-warning policy consolidation

Single mode-aware helper (e.g. `decode_reliability.py`) for `full_decode` vs `partial_window` vs `segment_seek`, used by CLAP / EffNet / audio-feature paths.

### 4. “No hidden scope” tests

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
