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

### Scatter points (metaballs)

The scatter itself is one WebGL metaball pass (`lib/point-layer.ts`), not N 2-D discs
plus a per-cell count badge. Two passes over a transparent canvas:

1. **Splat** — every point is a screen-space quad whose fragment shader deposits a radial
   kernel (flat core, smooth falloff) additively into a half-float render target.
2. **Composite** — a full-screen quad thresholds the summed field with a `smoothstep`, so an
   isolated point reads as the old 4 px disc and a dense knot bulges into one merged blob.

`CanvasView` builds one `PointSpec` per visible track from the same screen positions the
scatter would have drawn, then blits the layer's canvas between the grid and the overlays. When
the layer is unsupported (no WebGL2 or no renderable half-float target) the scatter falls
back to the plain 2-D `_dot` loop. The blob edge is anti-aliased with a screen-space
derivative ramp (~1 px at any density), and a weak dark ring (~1.5 px, ~0.35 alpha) sits just
outside it so blobs stay legible against the grid and the glow. Colour is the kernel-weighted
average, so a blob of one hue stays pure and a mixed blob blends proportionally; the summed weight
is divided out, so a blob's interior stays at full saturation however many points stack. Point alpha
dims a blob's colour (search fade, external tracks) without shrinking its shape, so faded points still
merge.

**Hovered and selected points are excluded from the field** and drawn crisp on top
(`_dot`), so a single track never dissolves into a blob. Hit testing stays CPU-side on
`CanvasView.positions`; a click inside a merged blob falls back to `nearestInBlob`, which
takes the nearest member of the cursor's 3×3 screen-grid neighbourhood (only when it holds
at least two points, so a far lone dot is never grabbed).

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
- Roots are grouped by role: My library first, then a muted **External** section
  (`.folder-section-label`, `.folder-row-external`). Root rows carry `data-root-id`,
  `data-root-role` and `data-root-abs`.
- **Rows carry no inline action icons.** Hovering a row opens a single action popout
  (`#folder-actions`, `.folder-actions`) to its right; `FolderTreeCoordinator` builds the
  buttons from the row's data attributes and hides the popout on leave, scroll, or resize.
  Each button shows a glyph, a label, and — where the action has a global hotkey — a `<kbd>`
  badge (`focus-folder` **F**, `isolate-selection` **S**, `fit-view` **H**,
  `exclude-selection` **Shift+S**). The buttons dispatch through `runCanvasAction` (the same
  `Controller._runCanvasAction` the hotkeys call), `POST /api/folders/reveal`,
  `POST /api/roots/role`, `POST /api/roots/remove` (after a `confirm`), `POST /api/folders/create`,
  or the head panel's imprint prefill via `onImprintRoot`.
- Root rows: *role* flips My library ↔ External via `POST /api/roots/role`; *remove* drops the
  root via `POST /api/roots/remove` — files stay on disk and cached analysis is kept, so
  re-adding restores it; dragging a root row out of the sidebar does the same without a
  button. *imprint* (external roots only) prefills the head panel's imprint target with the
  root's absolute path through the controller's `onImprintRoot` dep.
- **Adding a root is always an explicit role choice.** *+ Root* opens the modal (path from the
  clipboard); a **Finder drop** on the sidebar opens the same modal with the dropped path
  prefilled rather than silently adding a library root — a dropped folder may be a download the
  user wants **External**. The modal's *External (not My library)* checkbox sets the `role` sent
  to `POST /api/roots/add`.
- `#show-external-toggle` (in `#folder-controls`, hidden while no external roots exist)
  is bound to `model.showExternal`; toggling it is a **pure redraw**, never a refetch.
  When on, external dots are drawn at half alpha (`CanvasView._pointSpec`) so they read as
  secondary to My library. External positions ride along with **every** projection request
  (`EmbeddingCoordinator._projectionQueryString` sends `include_external=1` whenever
  `model.externalRootIds` is non-empty), so the toggle only has to reveal dots the chart
  already carries. A chart fetched before the tree revealed the roots is healed once in the
  background by `FolderTreeCoordinator._healExternalPositions`, so the toggle is never the
  thing that waits. `FolderTreeCoordinator._syncExternalRoots` reads roles from the swapped
  tree, so a role change only needs a tree `refresh()`, never the expensive
  `onRootsChanged()` reload.
- `#external-projection-toggle` (hidden with the same condition) is bound to
  `model.externalInProjection`. It is **off by default**: showing externals keeps the
  library t-SNE layout and the server places the external dots onto it by Nyström
  extension (`fit_scope=library`, the default). Turning it on sends `fit_scope=all`
  and refits the map with externals included — the old behaviour. Only this switch calls
  `refreshProjection()`; it is the only one that changes the fitted track set, so only it
  moves the library dots.

---

## Folder colours (family tints)

`dirColor` (`lib/colors.ts`) is a pure function of a virtual folder path, mirrored
server-side by `backend/services/dir_colors.py` for the Jinja folder-tree swatches.

- A root and its top-level folders keep the original path-hash hue (one vivid colour per
  playlist), so existing colours do not move.
- Folders deeper than the top level are tints/shades of their top-level ancestor's hue:
  lightness/saturation deviate by a per-path hash, and the deviation halves per nesting
  level (`0.5^(depth-2)`). A subtree therefore reads as one colour family that tightens
  as it deepens. Hue is exactly the ancestor's — only lightness/saturation vary.
- The two implementations must agree byte-for-byte; both suites assert the shared golden
  vectors in `frontend/src/lib/dir-color-vectors.json`.

---

## Right panel (tabs)

The right column pins the two "what am I looking at?" sections — **Search** and
**Properties** — above a tab strip (`#rp-tabs`, `RightPanelBinder`). Everything else
lives in one of four tabs, so the panel no longer scrolls to the brim:

| Tab | Sections | Concern |
|-----|----------|---------|
| **Analysis** | Analysis (coverage + per-stage retrigger), Microgenres | what the analysis says |
| **Organise** | Folder head (train/export/import/compare, training folders, imprint) | the learned folder structure |
| **Tracks** | Batch Edit, Features, Personal tags | the selection and its metadata |
| **View** | Options (hover preview, view mode, explode) | how the map is drawn |

- Sections are tagged **in place** with `data-rp-panel="<tab>"`; their cross-tab DOM order
  is irrelevant because only one tab is shown at a time. `RightPanelBinder.sync` toggles each
  tagged section's `hidden` and the tab's `active`/`aria-selected` from `model.rightPanelTab`.
- The active tab is persisted in `localStorage` (like `showExternal`), so a reload returns to
  the same tab; an unknown stored value falls back to **Analysis**.
- `HeadCoordinator.prefillImprint` reveals the **Organise** tab through the controller's
  `showRightPanel` dep before filling `#head-imprint-path`, so an imprint started from an
  external root row is visible even when another tab was active.

---

## Folder head panel

The right-column *Folder head* controls (`HeadCoordinator.bindPanel`):

- **Train folder view** sends the checked *Training folders* as `folders`; an empty
  selection means every **My library** root (the default) — external roots are only
  learned from when checked explicitly. The checkbox tree is the folder tree from
  `GET /api/folders`, fetched lazily when the section is opened — never on startup —
  and rendered as `.head-folder-row` rows with collapsible children. *All library
  folders* clears the selection. External roots are labelled `(external)`. The checked
  folders are mirrored onto `model.trainFolders` and
  persisted in `localStorage`, so a selection survives a reload.
- **Imprint on a folder…** takes an absolute path (or is prefilled from an external
  root row's hover action popout), calls
  `POST /api/supervised/imprint/analyze`, then polls
  `GET /api/supervised/imprint/status` until `pending == 0` before *Review plan* and
  *Apply moves* run the ordinary `POST /api/supervised/fanout`. The fan-out is always
  `recursive`, so nested files under the target are gathered and lifted. The ephemeral root is
  dropped via `POST /api/supervised/imprint/finish` after the moves land. *Apply moves*
  is the only step that touches disk and always confirms first. The plan note reports how
  many tracks went `Generic/` and how many were left `unmatched` (too far from every learned
  folder to be placed). The *Sort unmatched into Generic/&lt;genre&gt;/&lt;style&gt;* checkbox
  (default on) sends `generic`, filing unmatched tracks under their top Discogs label.
- **Undo moves** reverses the last applied fan-out batch through
  `POST /api/supervised/fanout/undo`; it is enabled only while a batch is available
  (`GET` on the same path) and disabled again after a successful undo.
- **Discard** appears only when `GET /api/supervised/imprint/temp_roots` reports an
  imprint a closed browser left behind; it drops those roots via
  `POST /api/supervised/imprint/discard` without touching the files.
- **Import** retries once with `force=1` when the server refuses a head trained on
  different encoder versions, after a `confirm` explains the mismatch.
- Head actions live in the collapsed *Folder head* `<details>`; the folder tree and the
  imprint controls are nested `<details>` so the panel stays short.

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

**Colour mode** (action `"toggle-clusters"`, **C**; backward with **Shift+C**) cycles `model.colorMode` through
folder → predicted folder → cluster → tempo → key. The active mode is always visible in the **top colour tabber** (below the lens
tabber) and can also be clicked there; the **View** tab select mirrors it. Selecting cluster while none are
available is refused with a warn toast. The tempo and key modes reuse the polar geometry of their lenses
(`bpmSpiralPolar` / `camelotPolar`) — angle → hue, radius → brightness — so tracks close on that map read as
close in colour. See `design-patterns.md` → *Persistent canvas tabbers*.

**Predicted folder** colours each track by the folder head's nearest learned folder (`dirColor` of the predicted
virtual path), not the track's own folder, so the map shows where the head *thinks* a track belongs. It is fetched
lazily on first selection from `GET /api/supervised/predictions` (`model.folderPredictions` +
`model.folderPredictionMatches`) and invalidated when the head changes (a drained supervise job or an import).
The **Predicted** select in the View tab picks between *Confident only* (`model.headColorConfident`, the default: only
tracks clearing their class's match floor, the same rule the fan-out uses) and *All nearest* (every track's nearest
folder regardless of similarity). A track with no prediction, or a weak one under *Confident only*, falls back to its
actual folder colour.

**Inspect card** is pinned by **Alt** (not a `HOTKEY_MAP` action — it is a bare modifier over a hovered
dot, so it lives in `CanvasInteractionController._bindInspectKey`). See *Inspect card* below.

**Adding a new hotkey:**
1. Add an entry to `HOTKEY_MAP` in `hotkeys.ts`.
2. Register a handler in `Controller._bindKeyboard` via `hk.on("your-action", () => { ... })`.
3. The F1 modal updates automatically.

---

## Lenses and scopes

The UI half of the embedding-space view; the pipeline it drives is in `audio_pipeline.md`.

### Lenses

- `model.lensId` selects the lens: the tag axes (`tags`), the combined map
  (`combined`), one lens per source block (`maest`, `clap`, `rhythm`, `supervised`),
  and the deterministic feature lenses (`camelot`, `tempo`). `model.viewMode` is a
  derived getter — `"tags"` for the tag lens, `"embeddings"` for every other — so
  the embedding-mode rules below need no per-lens branch.
- Hotkey **E** cycles every built-in lens forward; **Shift+E** cycles backward. The
  **top lens tabber** always shows the built-ins with the active lens highlighted, so the
  change is legible without checking the View tab and a tab can be clicked directly. The
  **View** tab has a lens select plus *Save preset…* / *Delete preset*; a preset captures
  the lens and the display options it was saved with (source weights, colour mode,
  granularity) and is listed in the same select. A preset's tabber highlight follows its
  base lens. Source weights only affect the combined lens, so their panel is hidden for
  the others.
- A projection lens requests a source subset (`sources=`) from the backend; the
  combined lens projects every frozen source. The source set is part of
  `layout_revision`, so each lens caches its own coordinates.
- Layouts are **pre-warmed in the background**: after the cluster pass drains, the
  durable queue runs one low-priority `layout` job per built-in projection lens, so the
  first switch to a lens is a cache hit instead of a ~12–20 s fit. A **user request
  preempts the prefetch** (`services/projection_priority.py`), so a switch never queues
  behind several prefetched fits. The status bar shows it as **Preparing views**; a lens
  whose revision is unchanged is a no-op.
- While a lens is loading, a **throbber pill** names the **target** lens and an elapsed
  readout (`Loading CLAP timbre · laying out · 12s`) — the map keeps showing the previous
  lens until the new layout lands, so the spinner is what makes the wait legible. It is
  drawn **stacked below** the analysis pipeline pill rather than replacing it, so track
  analysis progress can never hide a lens switch. A `layout_progress` phase carries its
  request's query, so an older overlapping request's phase is ignored. The background
  `layout` prefetch is **not** shown on the canvas (only in the status bar), so
  background work never looks like the shown view is computing.
- The deterministic lenses are a pure function of the cached audio features and are
  computed client-side with no server round-trip. They draw their guide under the
  dots: the Camelot rings + key labels, or the tempo spiral.
- In embedding mode: canvas positions come from the selected lens, the grid caption
  names it, transform handles and tag-drag are disabled.  Tracks without embedding
  data are **hidden** (not shown at tag-space positions).
- Mode transitions animate: positions lerp over ~340 ms via `canvas.animPositions`.
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
- **Viewport moves tween.** Fitting to a result set (`fitViewToPaths`) and the
  folder/cluster fit animate `m.vp` through
  `EmbeddingCoordinator.tweenViewport`, which reuses the shared animation frame, so
  the camera glides and the user can see *where* the view went. A single-track (or
  degenerate) result set **pans only** at the current zoom — it has no extent to fit,
  so homing one hit centres it instead of jumping to the ~16× a 0.05-wide window
  would give.
- **The pinned card follows its track.** While pinned, the card's anchor is the
  inspected track's *current* screen position, so a tweened move carries the card with
  it rather than leaving it at a stale screen position.

### Scopes

A scope is a circle or box on the canvas whose tracks are re-laid-out by another lens,
so a cluster can be read in a different space without leaving the map.

- Hold **X** and drag to draw a **box** from the press point; **X + scroll** opens a
  **circle** scope at the cursor or scales the open one (it starts small in *screen* space
  and grows outwards, so it reads the same at any zoom; growth is exponential in the wheel
  delta so a trackpad's burst of events accumulates). The scope starts on the last-used
  scope lens immediately. A bare **X** tap, **Esc**, or a click outside the region closes
  it and animates its tracks back out.
- A tab bar of the available lenses is drawn just outside the region; clicking a tab
  switches the scope's lens (morphing between the two layouts) and remembers it as the
  default for the next scope.
- Membership is frozen from the tracks' *outer* positions when the scope opens, so the
  re-layout cannot move a track across the boundary. A track the scope lens cannot place
  but whose outer position is inside the region is **hidden**, so the region only shows the
  re-layout. Membership is a world-space test (`scopeContains`); the canvas draws the
  region's screen image — an **ellipse** for a circle, a rectangle for a box, either
  stretched by the non-square viewport — so the outline always encloses exactly the scoped
  tracks and never an out-of-scope one.
- The in-scope positions are the scope lens's layout min-max normalised and mapped into the
  region: a circle's inscribed square, or a box's full extent (so a wide or tall box is
  used). A deterministic scope lens is instant, a projection scope lens reuses its cached
  global layout. The tab bar lists the built-ins
  except the tag lens.
- While a scope is open, in-scope dots behave like embedding-mode dots: tag
  drag/transform is disabled. Switching the outer lens drops the scope.

### Search field

While the search box is **focused** and the **CLAP lens** is the one on screen (the
scope's lens when a scope is open, else `model.lensId`), the canvas draws a
**search field**: a glowing "where does my query match" layer.

- All canvas glow — this field and the hover halo — is one WebGL pass in
  `lib/glow-layer.ts` on a transparent canvas layered **under** the scatter
  (which is transparent, so the page background shows through). Each point
  is an SDF quad, so N glowing points cost one draw call instead of N radial
  gradients or a full-screen blur. Overlaps take the brightest coverage (a `MAX`
  blend, not additive) with a gradual Gaussian falloff, so a dense region reads as
  one soft zone rather than a pile of bright circles.

- `/` or **⌘/Ctrl+F** focuses the box; **Esc** clears it and leaves the field,
  which is the only thing that hides the glow.
- The box searches **as you type** (debounced ~300 ms; Enter searches at once, an
  emptied box clears at once). The input stays editable during a fetch, and a
  response superseded by a newer query is dropped. While a fetch is in flight the
  box dims and a **Searching…** row shows above the previous results
  (`SearchView.setBusy`), so a programmatic search has visible feedback.
- **Matched tracks stand out.** `CanvasView.searchResults` (a `Set`, `null` = no
  search) fades every track the query did *not* match to 15 % alpha, so the result
  set reads against the rest of the map. The selected set *is* the result set, so the
  fade follows `replaceSelection` and clears with the query.
- The field is built from the backend's `field=1` landmark sample — the same
  representative tracks the layout's weight chart selects — splatted at their
  *current* map positions (scope/animation included) and weighted by their query
  cosine. `CanvasView.searchActive` / `searchField` own it; the search response
  carries the sample and the controller publishes it.
- It shows *where matching tracks live*, not a calibrated similarity contour:
  t-SNE preserves only local neighbourhoods, so two tracks with equal CLAP cosine
  can sit far apart on screen.

---

## Hover preview and tooltip

- **Hover preview** (**M** toggles, volume slider in Options) streams the track
  through `/api/audio/…`. **M** is owned by `CanvasInteractionController`, not
  `HotkeyManager` (`dispatch: false`): a bare tap toggles the preview, but **M +
  scroll** scales `model.previewVolume` instead and shows a cursor-adjacent
  percentage readout (`CanvasView.showVolumeHud`) — a scroll during the hold marks
  the gesture used, so the keyup does not also toggle the preview. It seeks to the **detected drop** when the track has a
  cached structure row, else to 40 % of the track — `previewStartSeconds` in
  `lib/structure.ts` owns that choice. A **search hit** overrides that: the hit's
  best-matching CLAP segment centre (`model.searchSections`, a duration fraction)
  wins, so hovering a result auditions the part that actually matched the query.
  **D** cycles the preview between the sections the track actually has (drop →
  break → intro) and re-seeks; the intro seeks to `intro_start` (the detected audio
  start after lead-in silence), falling back to `0`.
  `model.previewTrackPath` / `model.previewSection` are in-memory preview state,
  not persisted.
- **A search hit behaves like a track.** Hovering a result row glows its dot
  (`canvas.hoverTarget` = `{kind: "track", path}`), auditions it, and shows the
  same info **card** a canvas hover would, anchored at its dot — so a result reads
  as "this track, here", not a detached list row. The controller implements it in
  `onHoverTrack` / `onHoverEnd`; `CanvasView.hoveredPath` owns the card anchor.
  The Properties panel's rows share the same seam.
- Structure is **fetched lazily on first hover** (`/api/tracks/structure?paths=…`,
  silent, deduped per path) and cached on `model.trackStructure`. A missing row is
  the normal "not analysed yet" case; the preview keeps its fallback position and
  re-seeks to the drop if the fetch lands while the track is still auditioning
  (unless the user has already cycled to another section).
- The hover **tooltip** is a card (`lib/tooltip.ts`: pure `buildTooltip` +
  `renderTooltip`), not a block of text. It shows a **title/artist header** (the
  title is bold; the artist line is dropped when there is no artist metadata, and the
  whole header collapses to just the **filename** when there is neither), a
  **location stack** with the folder and cluster tags on separate lines,
  **structure chips** (`intro 0:12–0:32`, `drop 1:04`, `break 3:12`; the
  auditioned one is highlighted) and two stat areas — user **Personal tags** and
  audio **Features**. The location tags are told apart by their icon: the folder uses a
  **folder** glyph and the cluster a **tag** glyph, and each tag is **filled** with
  its colour (`tt-loc-filled`; a nested folder gets a parent→own **gradient**,
  `dirTagColors`) with the icon/name in the dark canvas colour for contrast. The
  cluster tag also carries a `genre-spanning` badge whose hover text explains that the
  intro and drop excerpts disagree. The cluster tag truncates a long genre name, so
  hovering it opens a **pop-over** (`renderGenrePop`, a body-level `.tt-pop` fixed
  element — never a child of the card, so expanding a name cannot reflow it) with the
  full name. When the folder head predicts a destination that differs from the track's
  current folder, the pop-over also shows it as a **Projected folder** line (the
  confident match floor is marked, a weak one is labelled *low confidence*); the
  predictions are fetched the first time a card is pinned. Feature axis ids never render as tags, and BPM/key
  are **not** duplicated near the title: both render as **chips** in Features
  (`TooltipBar.chip`) because a key or a BPM has no magnitude to bar. The BPM and key
  chips are **filled** with the same colours as the tempo/key colour modes
  (`tempoColor` / `keyColor`), so the card reads with the map's palette. **Mode** is
  folded into the Key chip — `musical_key` already carries the major/minor suffix
  ("Am" = minor, "A" = major) — so it stays a canvas axis/filter but has no tooltip
  row of its own.
  Personal tags are **sorted by label** so the card does not jump as you move between
  dots. The card has a **fixed width** (300 px) so rows wrap the same way for every
  track, and the structure chips are compact enough that the usual trio stays on one
  row. `buildTooltip` returns a `key`; the canvas re-renders the DOM only when that
  key changes, so pan/zoom frames do not rebuild the card.

## Analysis panel (coverage + retrigger)

The right panel's **Analysis** section is the overview of what the pipeline has
produced per track, and the way to re-run one stage.

- `AnalysisCoordinator.fetchCoverage()` reads `/api/embeddings/status` (whole
  library) and renders one row per source (Audio features, Rhythm, Structure, MAEST,
  CLAP, Supervised) as a determinate bar `done / total`; a pending count renders
  red. The summary line is `fingerprinted / total`. This is how a missing
  **structure** row (no section timestamps) becomes visible.
- Each stage has a **retrigger** button that POSTs `/api/embeddings/generate` with
  `stages: ["<kind>"]`, queueing only tracks missing that stage; **Re-run all**
  sends every stage with `force: true`. The durable queue then reports progress in
  the status bar and the on-canvas pill as usual.
- **Broken files.** `AnalysisCoordinator.fetchProblems()` reads
  `/api/analysis/problems` (unfingerprinted tracks + per-path terminal job failures) and
  renders a collapsible **Broken files (N)** list under the coverage bars. A row click
  selects the track and fits the view to it; `↗` reveals its folder in Finder; `↻` retries
  just that track's failed stages (`POST /api/embeddings/generate` with `paths: [path]`,
  which restricts the enqueue to that track). This is where a damaged file that can never be
  analysed is found and dealt with.
- **Every long-running process must show progress.** The status bar and the canvas
  pill render the queue's per-stage counts and the library-wide `stage` phases; the
  overlay (`#loading-bar`) and the centre canvas pill (`_drawCenterStatusPill`) render
  a determinate bar when a count is known and an **indeterminate** one otherwise,
  so a phase with no iteration callback (the layout base phase) never looks stalled.

## Inspect card (pinned, interactive)

The tooltip is `pointer-events: none` and follows the hover. **Alt** over a hovered
dot pins it: `CanvasView.inspect` is set, the card anchors to the dot's
*current* screen position (so a viewport tween carries it along), gets
`pointer-events: auto` and a **Navigate / Describe** tab strip. It stays
open when the pointer leaves the canvas (that is how you reach it), and closes on
**Esc**, a plain click on empty canvas, the card's `×`, or another Alt press.
`CanvasInteractionController._bindInspectKey` owns the Alt toggle; `CanvasView`
owns the pin/anchor state and delegates clicks (`data-inspect-action`,
`data-inspect-focus`, `data-inspect-close`) to `onInspectAction`.

While the card is pinned, the canvas hover is **paused**:
`CanvasInteractionController._updateCanvasHoverAt` keeps `hoveredIdx` on the
inspected track and returns before hit-testing, so moving across the map cannot
retarget the glow, audition or tooltip to another track. Closing the card resumes
free hover on the next move.

In the pinned card the folder tag, cluster tag and the key/BPM feature rows are
**navigation targets** (`data-hover-target`, `.tt-target`): hovering one glows the matching
dots and makes **S**/**H**/**F**/**Shift+S** act on that set, exactly like a sidebar row.
`CanvasView` delegates the card's `mouseover`/`mouseout` onto `canvas.hoverTarget`; see
`design-patterns.md` → *Navigation targets*.

`InspectCoordinator` (`coordinators/inspect-coordinator.ts`) owns the actions —
same seam as `SearchCoordinator`:

- **Mode toggle** — one button that flips the **distance filter** between *Nearest*
  (keep the closest tracks) and *Anti-track* (keep the furthest). It re-ranks the
  result list for the new direction; the slider is the main affordance and follows the
  toggle.
- **Distance-filter slider** — replaces the old discrete top-N *Solo* and the separate
  Anti-track / Nearest buttons. It ranges 0–100 % and starts at 100 % (the whole
  library is visible). Narrowing keeps the nearest *proximity*% of the ranked library
  (or the furthest *proximity*% in anti-track mode), and hides the rest. It
  deliberately does **not** re-home the view, so adjusting the filter never yanks the
  camera. The full ranking is fetched once per inspected track (`…&all=1`, paths
  only, no scores), **pre-loaded when the card is pinned** and cached, so a drag never
  hits the network; widening reveals exactly what the previous step hid
  (`Model.setProximityKeep`). The slider is excluded from the card's re-render key,
  so dragging does not rebuild the input under the pointer.
  Drag steps are **throttled** (`PROXIMITY_THROTTLE_MS`): a burst applies the first
  step at once and the last on the trailing edge, so the filter still tracks the drag
  without the re-hide/"change" cascade running on every pointer move. A superseded
  step's async ranking result is dropped (`_proximitySeq`), and any other card action
  cancels the pending step. A failed pre-load is silent; the error is only shown when
  the slider actually needs the ranking.
- **Nearest / Anti-track list** — the ranked rows for the toggle's current direction
  (`mode=near` → `NEAR_K` closest; `mode=far` → `ANTI_K` furthest), fetched when
  the card is pinned and again when the toggle flips.
- **Result row** — click focuses it: select alone, fit the view, audition it (only
  when hover preview is on). **Hovering** a row glows that track in the map
  (`{kind: "track", path}` hover target) and auditions it immediately, like a
  canvas hover. The audition lasts while the pointer is over the **track or the
  card**: leaving the row (or the dot) does not stop it, and it ends when the
  pointer leaves the card (`$tip` `mouseleave`), the card closes, or another row is
  hovered. While the card is pinned the canvas hover does not stop it either
  (`controller._handlePreview`), so crossing the gap from the dot to the card keeps
  playing.
- **Describe** — opening the tab **folds out a wide two-column panel** beside
  the card (`#coords-panel`, `renderDescribePanel`); the card body stays empty
  there, because the content needs more width than its 300 px. **Left column** is
  the flat list of zero-shot phrases CLAP ranks closest to the track
  (`GET /api/tracks/describe?path=…&limit=…`, `DESCRIBE_K`), each row a **search
  term**: clicking it runs it as a semantic query (`data-search-term` →
  `SearchView.setQuery` → `SearchCoordinator`), so the list is a way into the
  collection, not just a label. Phrases below the backend's similarity floor
  (`MIN_SIMILARITY`) are dropped, so the list stays short rather than padded
  with weak matches; the top phrase is always kept. **Right column** is the
  semantic coordinates (below). There is no panel title — the tab already says
  Describe. The `PromptBank` (`embeddings/clap_describe.py`) is a fixed candidate
  bank — CLAP has no text decoder, so these are ranked candidates, not generated
  text. The fetch is lazy (the pinned card's `open` action loads it) and cached
  per path; the state carries the path it belongs to, so a late response for a
  previous track is never shown on the current card. The panel shows "Asking CLAP…"
  while loading and an inline error note on failure.
- **Coordinates** — the track's position on the contrastive semantic axes
  (`dark <-> bright`, …), rendered in the panel's right column as a **bipolar
  track**: both pole names with the leaning one highlighted and a marker at the
  track's position, so the read is "leans this way, by this much" rather than a
  percentile a non-statistician has to decode. Pole labels share a **fixed width**
  with ellipsis, so a long name never shortens the slider relative to its
  neighbours. Each axis is a difference of the two pole text embeddings, z-scored
  against a reference mean/std, and ranked by `|z| × reliability` (`AXES` in
  `embeddings/clap_describe.py`), so a non-discriminative axis does not crowd
  out the informative ones. The marker position is relative to the reference
  library, not an absolute cosine.

Results are shown as title/artist/folder + similarity score, resolved against the
loaded library (the endpoint returns only `path` + `score`). The card shows a
"Searching…" note while a ranking is in flight and an inline error note when the
endpoint fails (e.g. no analysed tracks yet). The plain hover card carries a footer
hint (*Hold ⌥ (Alt) to interact*); the pinned card omits it, being already
interactive.
