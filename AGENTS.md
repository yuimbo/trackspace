# Trackspace — Agent & Contributor Guidelines

This file records the architectural principles behind Trackspace.
Read it before making changes so that new work stays coherent with existing decisions.

---

## Project layout

```
trackspace/
  backend/                  ← Python package (Flask server + ML pipeline)
    __init__.py
    app.py                  – Flask routes, CLI entry point
    tags.py                 – ID3 tag read/write via mutagen
    cache.py                – TrackCache (path+mtime → metadata, LRU + SQLite)
    fingerprint.py          – Chromaprint audio fingerprinting (fpcalc)
    audio_features.py       – shim → ``backend.embeddings.audio_features`` (barrel module)
    embeddings/             ← CLAP, caches, projection, classical audio features
      __init__.py           – lazy re-exports of ``layout`` public API
      layout.py             – composite vectors, semantic weighting, UMAP/t-SNE/PCA
      clap.py               – Laion CLAP model (audio + text embeddings)
      feature_cache.py      – FeatureCache (fingerprint → per-source vectors)
      coverage.py           – coverage stats + generation work-list helpers
      layout_revision.py    – deterministic layout revision hash
      audio_decode.py       – resilient librosa decode (segments, seeks, metadata mismatch)
      effnet.py             – Discogs-EffNet ONNX + Essentia-style mel / patches
      librosa_audio_features.py – 6-D classical descriptors (tempo, key, energy, danceability)
      audio_features.py     – re-exports decode / EffNet / librosa (imported by shim above)
    templates/partials/     – Jinja templates served by Flask (HTMX)
  frontend/                 ← Vite + TypeScript + Alpine.js + HTMX
    src/
      main.ts               – entry point
      model.ts              – Model (single source of truth, EventBus)
      controller.ts         – Controller (events, API calls, mouse/keyboard)
      hotkeys.ts            – HotkeyManager + key binding definitions
      command.ts            – Command pattern (optimistic UI + undo)
      components/           – one View per file (canvas, tags, properties, batch, status)
      lib/toast.ts          – shared DOM helpers
    index.html
    vite.config.ts
    package.json
  tests/                    ← test suite
    test_embeddings.py      – smoke tests for fingerprint → CLAP pipeline
    test_audio_features.py  – tests for EffNet + librosa audio features
    test_embedding_coverage.py – coverage, eligibility, and revision stability
    fixtures/test_track.mp3 – 5 s trimmed mp3 for tests
  data/                     – SQLite caches (gitignored, created at runtime)
  .env                      – HF_TOKEN (gitignored)
  requirements.txt
  AGENTS.md
```

**Running the backend:**  `python -m backend.app /path/to/music`
(The `dev:all` npm script does this automatically from `frontend/`.)

**Running tests:**  `.venv/bin/python -m pytest tests/ -v`
  or standalone:  `.venv/bin/python -m tests.test_embeddings`

---

## Frontend stack

The frontend uses **Vite + TypeScript**, **Alpine.js**, and **HTMX**.

- **UI modules:** `frontend/src/components/` — one file per view (`canvas-view.ts`, `tag-panel-view.ts`, …).
  Shared DOM helpers (toast, loading overlay, `dirColor`) live in `frontend/src/lib/toast.ts`.
- **Alpine.js:** Declarative islands in `frontend/index.html` (e.g. shortcuts modal `x-data`, `x-cloak`).
  Heavier logic stays in TypeScript (`Controller`, canvas).
- **HTMX:** The folder sidebar list is HTML from Flask (`GET /partials/folder-tree`) loaded via
  `htmx.ajax()` in `Controller._refreshFolderTreeHtmx()`. Jinja template:
  `backend/templates/partials/folder_tree.html`. In dev, Vite proxies `/partials` to Flask (port 5111).
- **Dev mode:** From `frontend/`, run `ROOT=/path/to/music npm run dev:all`.
  `concurrently` runs Flask + Vite with `[backend]` / `[frontend]` log prefixes.
- **Production:** `npm run build` → `frontend/dist/`; Flask serves static assets and keeps serving
  `/partials/...` for any future HTMX fragments.

---

## Architecture: three-layer MVC

```
model.ts                    – single source of truth, extends EventBus
components/*-view.ts        – imperative views (canvas, tags, properties, batch, status)
controller.ts               – wires events, HTMX folder refresh, API calls
```

**Views never mutate the model directly.** They expose named callbacks
(`onPickFolder`, `onHoverTrack`, …) that the controller implements.

**The controller is the only place that reads DOM events and calls API endpoints**
(except Alpine handling for purely local UI like the F1 shortcuts modal).

---

## Event bus: two granularities of change

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

## Canvas viewport

The canvas works in a **normalised world space (0 – 1 × 0 – 1)**.
All tag values live in this space; `w2s` / `s2w` convert between world and screen pixels.

The viewport state (`vp.ox`, `vp.oy`, `vp.zoom`) lives on the model so it can be persisted,
but viewport pan/zoom **does not emit `"change"`** — it just calls `canvas.scheduleDraw()`
directly to avoid triggering sidebar re-renders on every scroll tick.

---

## Tag values and persistence

Tag values are floats clamped to `[0, 1]`.  They live in MP3 ID3 frames on disk.

Mutations flow: **drag → update `track.tags[name]` in memory → emit `"tags-dirty"` →
enqueue in `pending` map → debounced POST to `/api/tracks/tags`**.

The debounce collapses rapid drag updates into one network round-trip.
Never write to disk synchronously inside an event handler.

---

## BatchView snapshots

`BatchView` keeps a `_snapshots` map (tag → Map<path, originalValue>) that records the
state of selected tracks at the start of a slider interaction.
This is what makes "scale/translate the whole selection" possible.

- Snapshots are **cleared when the selection changes** (different `_selHash`).
- Snapshots are **cleared on `renderDirty()`** so that after a viewport drag the sliders
  reset their baseline to the newly dragged-to values.
- The `_dragging` flag prevents any render while the user is actively dragging a batch slider.

---

## Selection model

`model.selected` is a `Set<string>` (track paths, not objects).
Always use `model.trackByPath(path)` to resolve paths back to track objects.
The path is `folder/filename.mp3` relative to `MUSIC_ROOT`; an empty-string folder
means the track sits directly at the root.

---

## Adding new interaction surfaces

1. Add a callback slot to the relevant View (`onFoo: ((…) => void) | null = null`).
2. Implement it in `Controller._wireViewCallbacks`.
3. If it changes model state structurally → call `m.emit("change")`.
4. If it only mutates in-memory tag values → call `m.emit("tags-dirty")`.
5. If it's purely visual (e.g. hover highlight) → update a field on `CanvasView` and call
   `canvas.scheduleDraw()` — no model event needed.

---

## Folder tree (HTMX + Flask)

- HTML must keep CSS class names expected by `trackspace.css` and by `Controller` delegation
  (`.folder-row`, `.folder-label`, `.folder-children`, `data-folder-toggle`, etc.).
- Query params: `active` and `active_folders` = current `model.folder` (see `Controller._refreshFolderTreeHtmx`);
  `pending_rename` = optional relative path for auto-opening rename after create.

---

## Command pattern (optimistic UI + undo)

`frontend/src/command.ts` implements a command pattern for state-mutating operations.

**Flow:**  `execute()` applies to model synchronously (optimistic) → UI re-renders →
`commit()` fires the API call in the background → on failure, `undo()` auto-reverts.

**Interface:**
- `execute(m)` / `undo(m)` — sync model mutations (inverse of each other).
- `commit()` — async API call; throws on failure to trigger auto-rollback.
- `undoCommit()` — optional reverse API call fired when the user hits Ctrl+Z.
- `effect` — declares which post-apply side-effects to run (`"change"`, `"tags-dirty"`, or `"change+htmx"`).

**CommandManager** lives on the Controller. The `afterEffect` callback (set up in the
constructor) maps effects to event emission, HTMX refresh, and `saveLS()`. Undo stack is
linear, capped at 50 entries. Undo is blocked while a `commit()` is in flight.

**Concrete commands:** `RenameTagCommand`, `RenameFolderCommand`, `MoveTracksCommand`.

**Not covered:** Tag value drags — the existing debounced `pending` queue + `_flushPending`
already provides optimistic behavior for that hot path.

**Adding a new command:**
1. Implement the `Command` interface in `command.ts`.
2. Pick the right `effect` (`"change"` for most, `"change+htmx"` if folder tree must refresh).
3. In the controller, replace the old `postJSON` + model mutation with `this.cmdMgr.run(new YourCommand(...), m)`.
4. Ctrl+Z undo works automatically via the stack.

---

## Hotkey system

`frontend/src/hotkeys.ts` owns all keyboard shortcut definitions and dispatch.

- **`HOTKEY_MAP`** — single array of `{ action, key, mod?, shift?, alt?, label, description }`.
  Each entry defines a named action (e.g. `"undo"`, `"fit-view"`) with its key binding.
- **`MOUSE_HINTS`** — non-keyboard hints (shift+drag, alt+drag) shown in the F1 modal
  but not dispatched.
- **`HotkeyManager`** — listens to `keydown`, matches against `HOTKEY_MAP`, and fires
  registered handlers for the matching action. The controller calls `hk.on("action", fn)`.
- **`renderShortcutList(container)`** — populates the F1 shortcuts modal from the maps.
  Called once at startup in `main.ts`.

Canvas visibility filtering is described to users as **isolate** (only selection or folder remains;
action `"isolate-selection"`, **s**) and **exclude** (remove from view, often repeated; action
`"exclude-selection"`, **Shift+s**), with **Alt+s** as `"show-all-tracks"`. Implementation still
uses `hiddenPaths` and `hiddenFolderPrefixes` on the model.

**Adding a new hotkey:**
1. Add an entry to `HOTKEY_MAP` in `hotkeys.ts`.
2. Register a handler in `Controller._bindKeyboard` via `hk.on("your-action", () => { ... })`.
3. The F1 modal updates automatically.

---

## Audio fingerprinting and embeddings

Trackspace supports content-based audio similarity via **chromaprint** fingerprinting and
**CLAP** (Contrastive Language-Audio Pre-training) embeddings.

### Two-cache architecture

| Cache | File | Key | Stores |
|-------|------|-----|--------|
| `TrackCache` | `data/track_cache.db` | `(path, mtime)` | ID3 tags, metadata, chromaprint fingerprint hash |
| `FeatureCache` | `data/audio_features.db` | `fingerprint` (SHA-256 of chromaprint) | CLAP embedding vector (float32 blob) + version |

The split means identical audio content (same recording, different filenames) shares one
embedding entry. `TrackCache` still invalidates on mtime change; `FeatureCache` is
content-addressed and versioned.

### Embedding versioning

`EMBEDDING_VERSION` (int) in `backend/embeddings/clap.py` encodes the current CLAP embedding
strategy. All `FeatureCache` reads filter by version, so bumping the constant
auto-invalidates stale entries. Old-version rows are purged on server startup.

| Version | Strategy |
|---------|----------|
| 0 | Legacy (pre-versioning) |
| 1 | Single 30 s middle-of-track chunk |
| 2 | Multi-segment: 3 × 15 s at 20 %/50 %/80 %, averaged |

When changing how embeddings are produced (model, chunk strategy, post-processing),
bump `EMBEDDING_VERSION` and add a row to the table above.

### Fingerprinting (`backend/fingerprint.py`)

- Calls `fpcalc` (chromaprint CLI) and SHA-256-hashes the raw fingerprint.
- System dependency: `brew install chromaprint` (provides `fpcalc`).
- `fpcalc` is resolved via `shutil.which` at import time, with a fallback to common
  homebrew/system paths.  npm/concurrently child processes sometimes have a stripped
  `PATH` that omits `/opt/homebrew/bin`, so bare `fpcalc` may not be found.
- Integrated into `_cached_read_all()` — fingerprints are computed once per file and
  stored alongside ID3 data in the track cache.

**Cache poisoning guard:** `_cached_read_all` checks `cached.get("fingerprint")` (truthy),
not just `"fingerprint" in cached`.  If a previous run stored `null` (e.g. fpcalc was
missing), the next run with fpcalc available will retry rather than serving the stale null.
Never cache a `None` fingerprint and treat it as a permanent result.

### CLAP embeddings (`backend/embeddings/clap.py`)

- Model: `laion/larger_clap_music` loaded in a background thread at server startup
  (does not block Flask from accepting requests).
- **Inference lock:** `clap.inference_lock` (imported by `embeddings` as `_inference_lock`)
  serialises all CLAP forward passes (audio + text).  MPS and CUDA are not safe
  for concurrent dispatches from multiple threads; without the lock, a Flask
  request running semantic-weighting text inference while the generation thread
  is producing audio embeddings would crash or hang.
- **Projection lock sharing:** The same lock also wraps mlx-vis (`UMAP` / `TSNE`)
  runs on Apple Metal in `embeddings/layout.py`.  Running PyTorch MPS (CLAP) and MLX
  command buffers concurrently from separate Flask threads can hit IOGPU
  assertion failures (e.g. `commit command buffer with uncommitted encoder`).
- **Rule for new GPU code paths:** If a path dispatches work to MPS/Metal/CUDA
  from request threads, either (1) guard it with `clap.inference_lock`, or
  (2) prove the backend is thread-safe under mixed-framework load before
  allowing concurrent execution.
- **Rule for env-toggle tests:** Backend enable flags are cached lazily; after
  changing `TRACKSPACE_CUML` / `TRACKSPACE_MLX_VIS` in tests, call
  `reset_projection_backend_cache()` before running a projection.
- Uses MPS (Apple Silicon), CUDA, or CPU — whichever is available.
- Audio is loaded via librosa at 48 kHz.  Three 15 s segments (at 20 %/50 %/80 %
  through the file) are embedded independently and averaged for robustness.
  Short tracks (≤ 22.5 s) use the whole file as a single segment.
- Generation runs in a background thread.  Progress is streamed to the frontend
  via SSE (`/api/embeddings/stream`) so tracks appear incrementally.
- The frontend sends `priority_paths` when kicking off generation so that tracks
  currently visible on screen are processed first.

### Projection methods

`compute_projection()` projects cached embeddings to 2D. The method is chosen by
the user via a toggle in the right-panel Options section and stored as
`model.projectionMethod`.

| Method | Backend function | Notes |
|--------|-----------------|-------|
| `"tsne"` (default) | `_project_tsne` (scikit-learn) | Best local cluster preservation; slower |
| `"umap"` | `_project_umap` (umap-learn) | Good global structure; moderate speed |
| `"pca"` | `_project_pca` (numpy SVD) | Instant; used for live incremental updates during generation |

PCA is never shown in the UI toggle — it is only used internally for fast intermediate
projections while embeddings are being generated.

### Projection caching

Projection results are cached at two levels:

**Server-side (revision-keyed LRU):**
`_revision_layout_cache` in `embeddings/layout.py` is an `OrderedDict[str, list[dict]]`
(capped at 16 entries). The key is `layout_revision` — a deterministic SHA-256
computed by `compute_layout_revision()` from sorted eligible paths, method,
sources, feature mask/blend, folder/tag context, and cache version constants.
Because the key is content-addressed, row order in the input data does not
affect cache identity.

**Client-side (query-string → {revision, positions} map):**
`_layoutCache` in `controller.ts` maps `_projectionQueryString()` →
`{ revision, positions }`, capped at 8 entries with LRU eviction.
Before fetching `/api/embeddings/projection`, the controller checks whether
a cached entry's `revision` matches the server's current `layout_revision`
(obtained from `/api/embeddings/status`). If it matches, cached positions
are applied directly — no projection round-trip. This handles A→B→A cycles
(e.g. toggling TSNE↔UMAP↔TSNE) without redundant server work.

Both caches are invalidated when new embeddings are generated (server: implicitly
via new revision; client: `_invalidateLayoutCache()` on SSE `done`).

**Deterministic ordering:** `_build_track_infos` sorts `infos` by path after the
`as_completed` gather loop. This is critical — `as_completed` returns results in
nondeterministic order, which would produce different composite cache keys, different
t-SNE/UMAP row ordering (and thus different layouts), and revision cache misses.
All new code that builds track info lists must maintain this sort invariant.

**UMAP reproducibility:** The `umap-learn` fallback passes `random_state=42` to
ensure identical inputs produce identical layouts across runs.

### Semantic weighting (context-aware projection)

Before projection, the embedding space can be **re-weighted** so directions
reflecting the user's folder organisation and tag names receive higher contrast.

**Two strategies — data-driven folders, text-based tags:**

| Source | Method | Rationale |
|--------|--------|-----------|
| Folders | Hierarchical centroid decomposition from actual track embeddings | Folders are manual curation — their tracks ARE the ground truth |
| Tags | CLAP text embedding of tag names | Tags are labels, not track collections |

**Hierarchical centroid decomposition** (`_build_folder_directions`):

For each folder node the direction is
`centroid(all tracks under node) − centroid(all tracks under parent)`.
The parent centroid *is* the weighted mean of its children's centroids, so
subtracting it naturally strips the shared component among siblings — each
child retains only its unique contrast.  This decomposes the embedding space
hierarchically: broad genre distinctions live at shallow levels, subtle
microgenre differences at deeper levels.

**Depth-increasing boost:**  Deeper levels receive *more* weight
(`boost × depth_boost^(depth−1)`, default `depth_boost = 1.5`) because broad
genre axes are already well-separated in CLAP space — it is the fine sibling
distinctions that dimensionality reduction tends to collapse and that benefit
most from amplification.

**Fallback:** Folders with fewer than 2 embedded tracks cannot produce a
reliable centroid; these fall back to CLAP text embedding of the folder's
leaf name.

**Flow:**
1. The frontend collects `model.contextTags` (tag names) and `model.contextFolders`
   (unique full folder paths from all tracks).
2. These are sent as `context_tags` and `context_folders` query params on
   `GET /api/embeddings/projection` (legacy alias: `/api/embeddings/umap`).
3. `_build_folder_directions()` infers the folder hierarchy from the leaf paths,
   computes centroids per node (including all descendants), and derives one
   normalised contrast direction per node.  Intermediate parent nodes that were
   not in the original folder set are inferred automatically.
4. `_apply_semantic_weighting()` combines data-driven folder directions, text
   fallbacks, and tag text directions into a single matrix `T_n` with weights
   `w`, then applies `X_out = X @ (I + T_n^T diag(w) T_n)`.

Weighting is transparent — when no context is provided, raw embeddings are
projected unmodified.

### Combinable embedding sources

Three sources can be independently enabled via checkboxes in the Options panel.
The user can activate any combination (at least one must stay on):

| Source | Module | Dimensions | Model |
|--------|--------|-----------|-------|
| CLAP | `backend/embeddings/clap.py` | 512 | `laion/larger_clap_music` (HuggingFace) |
| EffNet | `backend/embeddings/effnet.py` | ~1280 | `discogs-effnet-bsdynamic-1.onnx` via `onnxruntime` |
| Audio features | `backend/embeddings/librosa_audio_features.py` | 6 | `librosa` — no ML model needed |

**Audio feature vector (6D):**
`[tempo_norm, key_cos, key_sin, mode, energy_norm, danceability]`

Key is encoded on the **circle of fifths as a unit circle** in 2D:
`key_cos = cos(2π · fifths_pos / 12)`, `key_sin = sin(2π · fifths_pos / 12)`.
This preserves harmonic topology — keys a fifth apart are geometrically close,
and the circular wrap is seamless.  Mode (major=1, minor=0) is a 3rd dimension.

**Composite vector construction** (`_gather_composite_vecs` in `embeddings/layout.py`):
Each enabled source block is **L2-normalized across the batch** before
concatenation so that 1280D EffNet does not dominate 6D audio features.
A track is included only if it has data for every enabled source.

**Semantic weighting with composite vectors:** Folder centroid
decomposition uses the **full** composite matrix.  CLAP text directions
(tags, folder-name fallbacks) are placed in the **CLAP column span**
(``clap_col_lo : clap_col_lo + clap_dim``), which depends on *sources* order —
``mat[:, :clap_dim]`` was wrong when CLAP was not the first block.

### Independent versioning per source

| Source | Version constant | Cache columns |
|--------|-----------------|---------------|
| CLAP | `EMBEDDING_VERSION` (`embeddings/clap.py`) | `clap_embedding`, `version` |
| EffNet | `EFFNET_VERSION` (`embeddings/effnet.py`) | `effnet_embedding`, `effnet_version` |
| Audio features | `FEATURES_VERSION` (`embeddings/librosa_audio_features.py`) | `features`, `features_version` |

Bumping one source's version invalidates only that source's cached data.

### EffNet model management

The `discogs-effnet-bsdynamic-1.onnx` file (~18 MB) is auto-downloaded from
`essentia.upf.edu` to `data/models/` on first use.  The `data/` directory
is gitignored.  Loading is gated behind `--no-effnet` CLI flag (parallel
to `--no-clap`).  If `onnxruntime` is not installed, EffNet features are
silently unavailable.

**Mel-spectrogram preprocessing** (matches Essentia's `TensorflowInputMusiCNN`):
`librosa.feature.melspectrogram(sr=16000, n_fft=512, hop_length=256, n_mels=96)`
followed by `np.log10(10000 * mel + 1)`.  The mel is then patched into
128-frame windows (hop 64) and fed to the ONNX model.  Recipe confirmed
equivalent by Essentia maintainers: https://github.com/MTG/essentia/issues/1471

**Audio feature extraction** uses librosa only (no ML model):
- Tempo: `librosa.feature.tempo()`
- Key: Krumhansl-Schmuckler algorithm on `librosa.feature.chroma_cqt()`
- Energy: `librosa.feature.rms()` (log-scaled)
- Danceability: onset strength autocorrelation regularity

### Embedding Space viewport mode

- `model.viewMode` toggles between `"tags"` (default) and `"embeddings"`.
- Hotkey **E** switches modes.
- In embedding mode: canvas positions come from the selected projection method,
  grid shows "Embedding Space (TSNE)" or "(UMAP)", transform handles and tag-drag
  are disabled.  Tracks without embedding data are **hidden**
  (not shown at tag-space positions).
- Mode transitions animate: positions lerp over 500 ms via `canvas.animPositions`.
  During animation, tracks have interpolated positions; after animation completes,
  `animPositions` is cleared and the canvas falls through to embedding positions.
- The controller auto-triggers embedding generation when switching to embedding mode
  for the first time (`_ensureEmbeddingsAndProject`).  If required models are still
  loading in the background, the frontend polls every 3 s until ready.
- During generation the controller opens an `EventSource` on `/api/embeddings/stream`.
  As embeddings complete, PCA is re-fetched at intervals (roughly every 10 % of the
  batch) so dots appear progressively on canvas rather than all at once.
  Incremental PCA intentionally skips `context_tags` / `context_folders`
  (semantic weighting) to avoid contending with the generation thread for
  the CLAP model; the final TSNE/UMAP projection after generation applies
  full weighting.
- Switching projection method or source checkboxes re-fetches positions from the
  backend and animates the transition.


# Continuous improvement
After any major session, ask yourself:

Based on how our session went above, any architectural insights about things to abstract or refactor we can add to architecture_plan.md?

Or any common pitfalls we can not for future reference in AGENTS.md?