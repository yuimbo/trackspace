# Trackspace — UI conventions

Frontend interaction rules and invariants: the canvas, tag persistence, selection,
batch editing, folder tree, commands, and hotkeys. System shape lives in
`architecture.md`; the audio/ML pipeline in `audio_pipeline.md`.

---

## Canvas viewport

The canvas works in a **normalised world space (0 – 1 × 0 – 1)**.
All tag values live in this space; `w2s` / `s2w` convert between world and screen pixels.

The viewport state (`vp.ox`, `vp.oy`, `vp.zoom`) lives on the model so it can be persisted,
but viewport pan/zoom **does not emit `"change"`** — it just calls `canvas.scheduleDraw()`
directly to avoid triggering sidebar re-renders on every scroll tick.

### Position animations (`animPositions`)

`CanvasView.animPositions` is a single overlay that *all* position animations write
(tag-axis change, tags↔embedding transitions, incremental layout updates). Because they
share one map, they must also share **one** `requestAnimationFrame` handle so that starting
a new animation cancels the previous one — otherwise two loops interleave and the dots
flicker. `EmbeddingCoordinator` owns that frame and the overlay; the tag-axis animation
(`animateAxisChange`) lives there too, precisely because it writes the same overlay. New
position animations must be routed through the coordinator, not start their own rAF loop.

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

- HTML must keep CSS class names expected by `trackspace.css` and by `FolderTreeCoordinator` delegation
  (`.folder-row`, `.folder-label`, `.folder-children`, `data-folder-toggle`, etc.).
- Query params: `active` and `active_folders` = current `model.folder` (see `FolderTreeCoordinator.refresh()`);
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

**S**, **H**, **F** and **Shift+S** all act on the *hovered navigation element* — a folder row, a
cluster row, or a card chip (key/BPM). That is one `HoverTarget` descriptor resolved by
`CanvasInteractionController.effectiveHoverTarget()` (hovered target first, else the hovered dot's
folder); `Model.trackMatchesHoverTarget` is the only place a target is interpreted. See
`design-patterns.md` → *Navigation targets* before adding another element of this kind.

**Cycle preview section** (action `"cycle-preview-section"`, **d**) advances the hover preview through
the sections the auditioned track has (drop → break → intro) and re-seeks the audio; the active chip is
highlighted in the tooltip (`model.previewSection`).

**Inspect card** is pinned by **Alt** (not a `HOTKEY_MAP` action — it is a bare modifier over a hovered
dot, so it lives in `CanvasInteractionController._bindInspectKey`). See *Inspect card* below.

**Adding a new hotkey:**
1. Add an entry to `HOTKEY_MAP` in `hotkeys.ts`.
2. Register a handler in `Controller._bindKeyboard` via `hk.on("your-action", () => { ... })`.
3. The F1 modal updates automatically.

---

## Embedding Space viewport mode

The UI half of the embedding-space view; the pipeline it drives is in `audio_pipeline.md`.

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
- **Source weights drag.** Dragging a *Source weights* slider applies a first-order
  approximation of the layout from the server's cached Jacobian
  (`EmbeddingCoordinator.applyApproximateWeights`) so the map responds without a
  refit; the full re-cluster + re-layout runs only on **Apply**. The Jacobian is
  fetched once per weight vector (cached by the query string) and only after the base
  layout has landed (`_applyWeights` awaits `refreshProjection` first), so its S+1
  fits never contend with the base pass; a missing Jacobian falls back to the full
  recompute. The base layout streams its phase as `layout_progress`, rendered by the
  canvas pill and status bar (`model.layoutProgress`).

---

## Hover preview and tooltip

- **Hover preview** (**M** toggles, volume slider in Options) streams the track
  through `/api/audio/…`. It seeks to the **detected drop** when the track has a
  cached structure row, else to 40 % of the track — `previewStartSeconds` in
  `lib/structure.ts` owns that choice. **D** cycles the preview between the
  sections the track actually has (drop → break → intro) and re-seeks; the intro
  seeks to `intro_start` (the detected audio start after lead-in silence), falling
  back to `0`. `model.previewTrackPath` / `model.previewSection` are in-memory
  preview state, not persisted.
- Structure is **fetched lazily on first hover** (`/api/tracks/structure?paths=…`,
  silent, deduped per path) and cached on `model.trackStructure`. A missing row is
  the normal "not analysed yet" case; the preview keeps its fallback position and
  re-seeks to the drop if the fetch lands while the track is still auditioning
  (unless the user has already cycled to another section).
- The hover **tooltip** is a card (`lib/tooltip.ts`: pure `buildTooltip` +
  `renderTooltip`), not a block of text. It shows the folder kicker and title, a
  BPM/key meta row, **structure chips** (`intro 0:12–0:32`, `drop 1:04`,
  `break 3:12`; the auditioned one is highlighted), the **genre group** (cluster
  swatch + name + a `genre-spanning` badge whose hover text explains that the
  intro and drop excerpts disagree) and two stat-bar areas — user **Tags** and audio
  **Features**. Feature axis ids never render as tags. `buildTooltip` returns a
  `key`; the canvas re-renders the DOM only when that key changes, so pan/zoom
  frames do not rebuild the card.

## Inspect card (pinned, interactive)

The tooltip is `pointer-events: none` and follows the hover. **Alt** over a hovered
dot pins it: `CanvasView.inspect` is set, the card anchors at the dot's position
captured at pin time, gets `pointer-events: auto` and a `Navigate` section. It stays
open when the pointer leaves the canvas (that is how you reach it), and closes on
**Esc**, a plain click on empty canvas, the card's `×`, or another Alt press.
`CanvasInteractionController._bindInspectKey` owns the Alt toggle; `CanvasView`
owns the pin/anchor state and delegates clicks (`data-inspect-action`,
`data-inspect-focus`, `data-inspect-close`) to `onInspectAction`.

In the pinned card the folder kicker, cluster name and the key/BPM feature rows are
**navigation targets** (`data-hover-target`, `.tt-target`): hovering one glows the matching
dots and makes **S**/**H**/**F**/**Shift+S** act on that set, exactly like a sidebar row.
`CanvasView` delegates the card's `mouseover`/`mouseout` onto `canvas.hoverTarget`; see
`design-patterns.md` → *Navigation targets*.

`InspectCoordinator` (`coordinators/inspect-coordinator.ts`) owns the actions —
same seam as `SearchCoordinator`:

- **Anti-track** — `GET /api/tracks/similar?path=…&mode=far&k=1`, the single
  furthest track in the composite embedding space (what the map/clusters describe).
- **Nearest** — `mode=near`, the `NEAR_K` closest tracks.
- **Solo** — isolate to the neighbourhood (the inspected track plus `SOLO_K`
  neighbours) and fit the view to it, reusing `Model.focusPaths` and
  `fitViewToPaths`.
- **Result row** — click focuses it: select alone, fit the view, audition it (only
  when hover preview is on).

Results are shown as title/artist/folder + similarity score, resolved against the
loaded library (the endpoint returns only `path` + `score`). The card shows a
"Searching…" note while a ranking is in flight and an inline error note when the
endpoint fails (e.g. no analysed tracks yet).
