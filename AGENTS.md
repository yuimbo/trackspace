# Trackspace — Agent & Contributor Guidelines

Trackspace organises music in a 2D tag space. Tags are stored as private ID3
frames (`TXXX:trackspace:*`) directly in the mp3 files; audio similarity comes
from cached embeddings.

**This file is the entry point.** Read it first, then follow the link for the area
you are touching.

## Documentation map

| Doc | Covers |
|-----|--------|
| `architecture.md` | Where things live, frontend MVC + event bus, backend packages, lifecycle, HTTP API, non-goals |
| `ui_conventions.md` | Frontend interaction rules: canvas, tags, batch, selection, folder tree, commands, hotkeys |
| `audio_pipeline.md` | Fingerprinting → embeddings → projection → clustering, and the analysis job queue |
| `TODO.md` | Actionable tasks, deferred work, product roadmap |
| `README.md` | Setup and running |

---

## Repository map

```
backend/    Python package: Flask server + ML pipeline (routes/, services/, jobs/, embeddings/)
frontend/   Vite + TypeScript + Alpine.js + HTMX (src/coordinators/, components/, lib/)
tests/      pytest suite; frontend vitest tests live beside their sources
data/       SQLite caches (gitignored, created at runtime)
```

The detailed file tree and line-count snapshot are in `architecture.md`.

---

## Running

**Backend:**  `python -m backend.app /path/to/music`
(The `dev:all` npm script does this automatically from `frontend/`.)

**Backend tests:**  `.venv/bin/python -m pytest tests/ -v`
  or standalone:  `.venv/bin/python -m tests.test_embeddings`

**Frontend tests:**  from `frontend/`: `npm test` (vitest + happy-dom).
  Vitest picks up `src/**/*.test.ts`; `EventSource` and `fetch` are stubbed with
  `vi.stubGlobal` because the DOM environment implements neither.

**Frontend build:**  from `frontend/`: `npm run build` (`tsc --noEmit` + vite).

**Dev mode:**  from `frontend/`: `ROOT=/path/to/music npm run dev:all`
  (`concurrently` runs Flask + Vite; `/api` and `/partials` are proxied to Flask
  on port 5111).

---

## Core invariants

The rules most likely to be violated. Each links to its full explanation.

1. **No import-time side effects.** `backend/app.py` builds `TrackspaceState` and the
   Flask app inside `build_app(state)` / `main()`. Importing a module must not create
   caches, start threads, or purge data. → `architecture.md`
2. **Views never mutate the model directly.** They expose named callbacks
   (`onPickFolder`, …) that the controller implements. The controller and its
   coordinators are the only places that read DOM events and call API endpoints.
   → `architecture.md`
3. **Two event granularities.** Structural changes emit `"change"`; in-flight tag
   mutations emit `"tags-dirty"`. Never emit `"change"` inside an animation loop.
   → `architecture.md`
4. **Never write tags to disk synchronously** in an event handler: mutate in memory →
   emit `"tags-dirty"` → debounced POST. → `ui_conventions.md`
5. **One animation frame for all position animations.** New position animations route
   through `EmbeddingCoordinator`, never their own `requestAnimationFrame` loop.
   → `ui_conventions.md`
6. **Every HTTP failure is JSON** (`{"ok": false, "error": "…"}`) and every request is
   validated in `backend/services/request_validation.py`. Batch IO keeps partial success
   and error messages stay client-safe. → `architecture.md`
7. **Guard GPU work from request threads** with `clap.inference_lock`, or prove the
   backend is thread-safe under mixed PyTorch/MLX load. → `audio_pipeline.md`
8. **Version every embedding source independently.** Changing how a source is produced
   means bumping that source's version constant; all cache reads use
   empty-blob-as-absent. → `audio_pipeline.md`
9. **Expensive analysis goes through the durable job queue.** Never raise for one bad
   track — return `JobResult.failure(...)` and let the queue own the retry.
   → `audio_pipeline.md`

---

## Continuous improvement

After any major session, ask yourself:

- Any architectural insights about things to abstract or refactor → `architecture.md`
  (modularisation log) or a task in `TODO.md`.
- Any common pitfalls worth recording → the relevant doc above, or `AGENTS.md` if it is
  a top-level invariant.
