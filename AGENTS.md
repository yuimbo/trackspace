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
| `design-patterns.md` | Shared interaction patterns: navigation targets, hover state, coordinators, input controls |
| `audio_pipeline.md` | Fingerprinting → embeddings → projection → clustering, and the analysis job queue |
| `trackspace-projection-methods.md` | Research notes on parametric/kernel projection candidates; background, not normative |
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

**Layout-evaluation CLI + gallery:** `python -m backend.precompute_layout_evaluation`
  generates the offline eval maps; open the app with `?layout_eval=1` for the browser
  gallery. Usage, the named sets and the gallery: README.md → *Layout evaluation*.

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
10. **Serialise the native MP3 open, not the decode.** libsndfile calls the global,
    non-thread-safe `mpg123_init()` on every `sf_open`; concurrent opens from the
    `features` and `maest` workers kill the process with SIGBUS. All decode goes
    through `decode_stderr.librosa_load` / `librosa_get_duration`, which hold
    `decode_stderr._DECODE_LOCK` across the open. → `audio_pipeline.md`
11. **Never `with sqlite3.connect(...) as conn:`.** The stdlib connection context
    manager only commits/rolls back — it does **not** close the handle, so a connection
    (and its db/wal/shm FDs) leaks on every cache access. Return
    `backend.sqlite_conn.ClosingConnection` from each cache's `_conn()`; the
    `with self._conn() as conn:` idiom is then correct at every call site.
    → `architecture.md`

---

## Continuous improvement

After any major session, ask yourself:

- Any architectural insights about things to abstract or refactor → `architecture.md`
  (modularisation log) or a task in `TODO.md`.
- Any common pitfalls worth recording → the relevant doc above, or `AGENTS.md` if it is
  a top-level invariant.

---

## Finishing a task

Before you call a task done, always do all four of these:

1. **Remove finished work from `TODO.md`.** Tick or delete anything this task completed;
   a stale TODO is worse than no TODO. If it is now a `✅` line in the product roadmap,
   keep it only while it is still useful context, otherwise delete it.
2. **Add viable follow-ups to `TODO.md`.** Anything you noticed but did not do —
   refactors, correctness gaps, perf work, dependency risks — goes in as a concrete,
   actionable item with file/line references and a "why". Follow the existing sections
   (`Deferred / optional`, `Product roadmap`, `Analysis performance`); add a new section
   if none fits. Do not pad it with speculative work you would not act on.
3. **Sanity-review your local changes.** Re-read `git diff` for the files you touched:
   no debug leftovers, no dead code, no missing direct dependencies for new imports, no
   docs that now contradict the code, and tests for new behaviour. Run the relevant suites
   (`.venv/bin/python -m pytest tests/`, and `npm test` + `npm run build` from
   `frontend/` for frontend changes) and fix what you broke.
4. **Commit with a sensible message.** Stage the intended files (never secrets or
   unrelated scratch) and write a message that matches the repo style: a short imperative
   subject, then a body explaining the what and why. One topic per commit where the change
   can be separated cleanly; do not bundle unrelated work.

