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
merge. A grey point — a track the head cannot place, or an unanalysed one — is **down-prioritised**
(`PointSpec.weight`, `GREY_COLOUR_WEIGHT`): it still adds shape, but almost none of its colour, so a
blob it merges into keeps the hue of its information-bearing members instead of washing out to grey. The
grey points are also splatted and composited **first**, the coloured points over them, so a grey dot never
occludes a coloured one.

**Hovered and selected points are excluded from the field** and drawn crisp on top
(`_dot`), so a single track never dissolves into a blob. Hit testing stays CPU-side on
`CanvasView.positions`; a click inside a merged blob falls back to `nearestInBlob`, which
takes the nearest member of the cursor's 3×3 screen-grid neighbourhood (only when it holds
at least two points, so a far lone dot is never grabbed).

### 3-D view

`model.view3d {on, yaw, pitch}` is a client-side view option, persisted in localStorage.
Two gates read it: `model.is3d` (the *mode*: on, not eval gallery, projection lens — query
string, interactions, scope/tag rules) and `model.shows3d` (the *rendering*: `is3d`'s gate
that stays true while the collapse tween runs). Toggle: the `3D` tab beside the lens tabs, or
the **3** key (`EmbeddingCoordinator.setView3D`, which refetches with `dims=3` — a true 3-D
t-SNE, cached separately from the 2-D layout).

- **Camera.** `lib/camera3d.ts` rotates the unit cube about its centre and applies a slight
  perspective (`?persp=0.28` default, `0` = orthographic; read once at load). The rotated x/y
  go through the existing `w2s`, so pan/zoom are unchanged; `depth` (0.05–1, 1 = nearest) and
  the perspective `scale` (point radius) are the only new values. Every world→screen consumer
  of track positions goes through `CanvasView._trackScreen`.
- **Transition.** `model.view3dBlend` (0 = head-on orthographic, 1 = full 3-D, not persisted)
  scales yaw, pitch, perspective and the depth shading; `EmbeddingCoordinator._tweenBlend`
  animates it on its own rAF handle (it moves a camera parameter, not a dot, so it may run
  alongside a position lerp). A toggle is one gesture: the blend tween and the dot lerp start
  together (`TOGGLE_MS`) once the new layout is applied, so a slow 3-D fit delays the whole
  motion rather than splitting it. The grid and the cube frame crossfade with it.
- **Gauge.** A 3-D layout is fitted so its head-on (x,y) matches the same lens's 2-D layout
  (`lens_gauge.gauge_layout_3d`; z sign anchored per track-set family). The 2-D layouts are
  already aligned across lenses, so 2-D↔3-D barely moves x/y and 3-D lens switches stay
  consistent.
- **Controls.** Alt+drag or middle-drag orbits; Alt+Shift+drag and trackpad scroll pan; wheel
  zooms. In 2-D, Alt/middle-drag still pan. Orbit persists on mouse-up, not per move.
- **Depth-aware metaballs.** `MetaballLayer.render(points, true)` MAX-blends each pixel's front
  depth, then splats only points within a thin slab behind it, so depth layers stay separate
  blobs. Far points are also darkened (`depthShade`). Hit testing picks the front-most
  candidate within the hit radius.
- **Depth-aware glow.** `GlowLayer.render(points, occluders)` depth-tests each halo against the
  scatter points and crisp dots, so a halo is hidden by blobs at its depth or nearer and shows
  over deeper ones. In 3-D the glow canvas is hidden and blitted into the scatter canvas right
  after the blobs (below the crisp dots); in 2-D it stays a DOM layer under the scatter.
- **Scopes are spheres** (`Scope.shape === "sphere"`, centre `cx,cy,cz`, radius `r`), the one
  shape whose members do not change under orbit; membership is tested in cube coordinates
  (`scopeContains(..., wz)`) and the outline is a projected circle. X+scroll opens one at the
  cursor (centre = nearest dot within 20 px, else the cube-centre plane); X+drag originates at
  the nearest dot (same snap) with the drag distance as radius. Toggling 2-D ↔ 3-D with a
  scope open converts it (circle/box ↔ sphere) and re-lays it out.
  - **Contents are the flat 2-D scope view**, never a nested 3-D map: the scope lens is always
    fetched with `dims=2` (shared cache with 2-D), and the members are drawn *billboarded* —
    a flat disc facing the camera at the sphere's projected centre, not rotating with the orbit.
  - **Wormhole.** Non-members nearer the camera than the sphere's centre and near its screen
    circle are pushed radially out to the rim band and faded to ghost silhouettes
    (`wormhole()` in `canvas-view.ts`: `WORMHOLE_REACH`, `GHOST_FADE`). A ghost sinks to the
    lowest depth so it cannot occlude the contents, and is neither hoverable nor selectable.
  - **Frames.** A scope member's position carries its own frame (`MapPos.billboard`: the sphere
    centre it is laid out around), and a move between two frames (scope open/close/lens switch,
    2-D↔3-D conversion) is stored as `{…, from, mix}` and blended in *view* space by
    `CanvasView._viewFrame` — never lerped in world coordinates, which would glide toward an
    unrelated cube point before snapping.
  - **Frosted glass.** The scope disc is a pane of glass over the rest of the map: non-members
    that project inside its circle are pushed to the inner rim band (`rimPush()`, not hoverable
    or selectable) and stay visible *behind* the glass, which blurs and tints the grid, cube and
    outer blobs (`_drawGlass`: clip + `ctx.filter` blur self-copy + background tint). The
    scope's members render sharp on top. Blur radius is `?glass=<css px>` (default 7, `0` =
    tint only); without `ctx.filter` support (Safari) the tint alone is used, stronger.
- **Weight sliders** work live in 3-D: the landmark Jacobian is dimension-agnostic and the
  frontend approximates z like x/y; a Jacobian fetched for the other dimensionality is never
  applied (`_jacobianKey` carries `dims`).
- **Off in 3-D:** the 2-D grid (a cube frame is drawn), tag drag/transform.

---

## Tag values and persistence

Tag values are floats clamped to `[0, 1]`.  They live in MP3 ID3 frames on disk.

Mutations flow: **drag → update `track.tags[name]` in memory → emit `"tags-dirty"` →
enqueue in `pending` map → debounced POST to `/api/tracks/tags`**.

The debounce collapses rapid drag updates into one network round-trip.
Never write to disk synchronously inside an event handler.

Gallery mode (`?layout_eval=1`) blocks every write client-side and never persists
view settings; see README.md → *Layout evaluation*. Hover-to-play works unchanged in it.

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
- **The Folders header** is `#folder-head`: the `<h2>` and a compact **+ Root**
  button (`#btn-add-root`, top-right of the same row). There is no drop-hint text.
- Root rows: *role* flips My library ↔ External via `POST /api/roots/role`; *remove* drops the
  root via `POST /api/roots/remove` — files stay on disk and cached analysis is kept, so
  re-adding restores it. **Right-clicking a folder inside an External root reuses the same row
  action popout** (there is no separate context menu): it carries *Project folder structure onto
  here*, which prefills the projection target with that folder's absolute path (the root's
  `data-root-abs` plus the row's relative path) through the controller's `onImprintRoot` dep,
  which opens the head modal. Dragging a root row out of the sidebar shows the **drop zone**
  (below) and removes the root after a `confirm`.
- **Adding a root is always an explicit role choice.** *+ Root* opens the modal (path from the
  clipboard); a **Finder drop** on the sidebar opens the same modal with the dropped path
  prefilled rather than silently adding a library root — a dropped folder may be a download the
  user wants **External**. The modal's *External (not My library)* checkbox sets the `role` sent
  to `POST /api/roots/add`. A drop with no absolute path shows the warning **anchored at the drop
  point** (`toast(…, { x, y })`), not in the global stack; an internal root-row drag that is
  released in place is a no-op and never reaches the drop handler.
- **Drop zone to remove a root.** While a root row is dragged, `#root-drop-zone` (`.root-drop-zone`,
  at the left edge of `#main`, just right of the sidebar) appears with a red gradient outline and a
  trashcan glyph; dropping the row there (or anywhere outside the sidebar) removes the root after a
  `confirm`.
- **External visibility and refit are view-modifier chips**, not graph controls.
  The permanent **Library + External / Library Only** chip is the click target for
  `model.showExternal` (**V**); toggling it is a **pure redraw**, never a refetch.
  When on, external dots are drawn at half alpha (`CanvasView._pointSpec`) so they read as
  secondary to My library. External positions ride along with **every** projection request
  (`EmbeddingCoordinator._projectionQueryString` sends `include_external=1` whenever
  `model.externalRootIds` is non-empty), so the toggle only has to reveal dots the chart
  already carries. A chart fetched before the tree revealed the roots is healed once in the
  background by `FolderTreeCoordinator._healExternalPositions`, so the toggle is never the
  thing that waits. `FolderTreeCoordinator._syncExternalRoots` reads roles from the swapped
  tree, so a role change only needs a tree `refresh()`, never the expensive
  `onRootsChanged()` reload.
- The chip's inline **Refit** checkbox (`model.externalInProjection`, action
  `"toggle-external-refit"`, **R**) is shown only while externals are on the map. It is
  **off by default**: showing externals keeps the library t-SNE layout and the server places
  the external dots onto it by Nyström extension (`fit_scope=library`, the default).
  Turning it on sends `fit_scope=all` and refits the map with externals included — the old
  behaviour. Only this switch calls `refreshProjection()`; it is the only one that changes
  the fitted track set, so only it moves the library dots.

---

## Folder colours (family tints)

`dirColor` (`lib/colors.ts`) paints a folder as a tint of an ancestor's hue — its
*anchor* — mirrored server-side by `backend/services/dir_colors.py` for the Jinja
folder-tree swatches.

- The default anchor is structural: a root and its top-level folders keep the original
  path-hash hue (one vivid colour per playlist), and a deeper folder inherits its
  top-level ancestor's hue. Lightness/saturation deviate by a per-path hash and the
  deviation halves per nesting level below the anchor (`0.5^(below-1)`), so a subtree
  reads as one colour family that tightens as it deepens. Hue is exactly the
  anchor's — only lightness/saturation vary.
- `backend/services/folder_anchors.py` overrides the anchor per folder with a semantic
  one: a folder is a *variation* of its parent when the two subtree centroids are at
  least `VARIATION_MIN_COSINE` similar in the map's weighted space, and a variation
  inherits its parent's anchor. So `GENRES/<genre>/<subgenre>` reads as one family
  while uncorrelated folders each start their own hue. Folders too thin to measure fall
  back to the root role (external roots are never variations).
- The server embeds the anchors in the folder-tree partial (`#dir-anchors`);
  `FolderTreeCoordinator` reads them into `setDirAnchors`, which the canvas, tooltips
  and rule chips share. With no anchors the structural rule applies, so the golden
  vectors still hold.
- The anchor build runs on a background thread: the partial only reads the cache, so
  the library load never waits on a full-library gather (the first read returns the
  structural fallback). The tree is refetched when the scan finishes and when a run
  drains, so the semantic colours appear once the embeddings exist.
- The build streams its phases as `colors_progress` events (`scanning files` →
  `gathering vectors` → `weighting vectors`), shown as a *Folder colours* row in the
  analysis panel and a hoverable status-bar readout, so a long walk is not silent.
- The two implementations must agree byte-for-byte; both suites assert the shared golden
  vectors in `frontend/src/lib/dir-color-vectors.json`.

---

## Right panel (tabs)

The right column pins **Properties** above a tab strip (`#rp-tabs`,
`RightPanelBinder`). Everything else lives in one of three tabs, so the panel no
longer scrolls to the brim:

| Tab | Sections | Concern |
|-----|----------|---------|
| **Analysis** | Analysis (coverage + per-stage retrigger), Microgenres, Folder head | what the analysis says |
| **Tracks** | Batch Edit, Features, Personal tags | the selection and its metadata |

- Sections are tagged **in place** with `data-rp-panel="<tab>"`; their cross-tab DOM order
  is irrelevant because only one tab is shown at a time. `RightPanelBinder.sync` toggles each
  tagged section's `hidden` and the tab's `active`/`aria-selected` from `model.rightPanelTab`.
- The active tab is persisted in `localStorage` (like `showExternal`), so a reload returns to
  the same tab; an unknown stored value falls back to **Analysis**.
- **Search is not here.** It is a `⌘F` popup (see *Search popup*) that commits to an
  **active view modifier** chip.
- **View options are not here either.** Lens and colour live in the persistent top
  tabbers (a click or **E** / **C**); hover preview lives in the status bar (see
  *Hover preview and tooltip*); and the contextual display toggles live in the
  top-right **View options** panel (`#view-options`) on the canvas edge. There is no
  View tab. The panel appears whenever at least one of its `.view-option` rows is
  relevant — currently only the **Confident only** gate (`#head-confident-fold`),
  shown in predicted-folder colour mode.
- **The imprint controls are not here either.** They live in the **head modal** (see below);
  the head's info and training/export/import/compare live in the Analysis tab's **Folder
  head** section.

---

## Head modal

`HeadCoordinator.openModal` opens `#head-modal` — the dialog for **projecting the folder
head onto a folder outside the library**. It is reached by **right-clicking a folder inside an
External root** and choosing *Project folder structure onto here* in the shared row action popout
(`FolderTreeCoordinator._bindFolderContextMenu` just routes the right-click to
`_showFolderActions`). Training and the head tools live in the **Analysis panel's Folder head
section** (see below), not in this dialog, so the modal stays focused on the fan-out.

- **Project folder structure onto here** fills `#head-imprint-path` with the folder's absolute
  path and shows it in `#head-modal-target`, then opens the modal.
- **Project onto** takes an absolute path (prefilled from the right-clicked external folder),
  calls `POST /api/supervised/imprint/analyze`, then polls
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
- Esc, the `×`, or a click on the backdrop closes the modal without reaching the global Esc
  handler.

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

- **`HOTKEY_MAP`** — single array of `{ action, key, code?, mod?, shift?, alt?, label, description, dispatch? }`.
  Each entry defines a named action (e.g. `"undo"`, `"fit-view"`) with its key binding. Entries with
  `dispatch: false` are still declared here for the F1 list and for UI badges, but their keydown/keyup is
  owned elsewhere: the canvas gesture keys **X**/**M**/**D**/**Alt** by `CanvasInteractionController`, and
  **F1** by the Alpine shortcuts modal.
- **`MOUSE_HINTS`** — pointer-only hints (shift+drag, alt+drag) shown in the F1 modal but not dispatched.
- **`HotkeyManager`** — listens to `keydown`, matches against `HOTKEY_MAP`, and fires
  registered handlers for the matching action. The controller calls `hk.on("action", fn)`.
- **`renderShortcutList(container)`** — populates the F1 shortcuts modal from the maps.
  Called once at startup in `main.ts`.
- **`hotkeyLabel` / `hotkeyBinding` / `hotkeyCode` / `hotkeyKey`** — lookups every render site uses
  (folder-row popout, view-modifier chips, tooltip seekbar) and the gesture handlers use to match their
  physical key, so no key or label is hard-coded outside this module.
- **`applyHotkeyLabels()`** — fills static markup from the table: `[data-hotkey-action]` gets the action's
  label as its text, `[data-hotkey-title]` gets the description as its `title`, and `data-hotkey-template`
  substitutes `{key}` into a prose `title`. Called once at startup in `main.ts`.

Canvas visibility filtering is described to users as **isolate** (only the bloom class, hovered
element, or selection remains; action `"isolate-selection"`, **s**) and **exclude** (remove from view,
often repeated; action `"exclude-selection"`, **Shift+s**), with **Alt+s** as `"show-all-tracks"`.
Implementation still uses `hiddenPaths` and `hiddenFolderPrefixes` on the model.

**S** first solos the **hovered dot's bloom class** (`Model.bloomPaths`) — for folder colour that is the
whole subtree under the bloom's ancestor, i.e. the left-browser folder-click behaviour; for every other
variable it is the range-aware bloom class. With no dot hovered it falls back to the *hovered navigation
element* — a folder row, a cluster row, or a card chip (key/BPM). That is one `HoverTarget` descriptor
resolved by `CanvasInteractionController.effectiveHoverTarget()` (hovered target first, else the hovered dot's
folder); `Model.trackMatchesHoverTarget` is the only place a target is interpreted. See
`design-patterns.md` → *Navigation targets* before adding another element of this kind.

**Cycle preview section** (action `"cycle-preview-section"`, **W**; backward with
**Shift+W**, action `"cycle-preview-section-prev"`) advances the hover preview through
the sections the auditioned track has (drop → break → intro) and re-seeks the audio; the active
marker is highlighted and labelled in the tooltip's seekbar (`model.previewSection`).

**Selection bloom cursor** (action `"cycle-selection-bloom"`, **D**; backward with **Shift+D**) is a
second marker drawn around the **top colour tabber**, separate from the active-mode highlight. It picks
the variable used for the **selection bloom**: hovering a bare dot glows every track that falls in the hovered
track's bloom on the cursor variable (`Model.trackBloomsWith`), the way the old rule only bloomed the hovered
dot's folder. So the map can be coloured by one variable and bloom by another.

**The cursor ring borrows the hovered class's colour** (`trackColor(hovered, selectionBloom)`), so it reads as
that class; with nothing hovered it is the warm default. **A chip under the cursor** names the hovered class
(folder path, cluster, key, `128 BPM`) and, where the variable has a range, its meaning; the status pills dock
below it. **For a folder (or predicted folder) the chip adapts to the range**: the path is drawn segment by segment
with the segment the range climbed to highlighted in the class colour and deeper segments faded to grey — so
`GENRES/Ambient/dark` highlights `dark` at exact, `Ambient` (fading `dark`) at +1, `GENRES` (greying `Ambient/dark`)
at +2. **D** is owned by `CanvasInteractionController` (`dispatch: false`): a bare tap moves the cursor, but
**D + scroll** sets the bloom **range** and shows a cursor-adjacent panel (`CanvasView.showBloomHud`),
mirroring **M**. Clicking a colour tab sets both the colour mode and the cursor, so the mouse can pick the bloom
variable too. The cursor and range persist in `localStorage`.

**The range is the bloom's tolerance, and it means something different per variable** (`lib/bloom.ts`, one
module so the matcher and the HUD cannot disagree): folder and predicted folder climb *N parent directories*
(a wider range includes sibling subtrees under a shared ancestor), tempo allows *± N % BPM*, key allows *± N Camelot
steps*; cluster has no range — an exact id match. **Each variable keeps its own range** (`model.bloomRanges`,
keyed by `BloomRangeKey`; folder and predicted folder share one slot, since they are the same directory climb), so
widening the tempo bloom does not also widen the folder bloom. The scroll is **additive**, not the volume's
multiplicative taper, because range 0 (exact class) is a real state a scroll must be able to leave. The HUD names the
meaning (`+2 parent directories`, `±16% BPM`, `exact key`) instead of a bare percentage, and for a variable with no range
(cluster) it renders **no slider** — there is nothing to slide.

**Colour mode** (action `"toggle-clusters"`, **C**; backward with **Shift+C**) cycles `model.colorMode` through
folder → predicted folder → cluster → tempo → key. The active mode is always visible in the **top colour tabber** (below the lens
tabber) and can also be clicked there; there is no separate select. Selecting cluster while none are
available is refused with a warn toast. Moving the cluster **granularity** slider (resolution) applies
live, as the handle moves, and switches to cluster colour, so the re-partition it triggers is visible rather than
hidden behind another mode. The tempo and key modes reuse the polar geometry of their lenses
(`bpmSpiralPolar` / `camelotPolar`) — angle → hue, radius → brightness — so tracks close on that map read as
close in colour. See `design-patterns.md` → *Persistent canvas tabbers*.

**Predicted folder** colours each track by the folder head's nearest learned folder (`dirColor` of the predicted
virtual path), not the track's own folder, so the map shows where the head *thinks* a track belongs. It is fetched
lazily on first selection from `GET /api/supervised/predictions` (`model.folderPredictions` +
`model.folderPredictionMatches`) and invalidated when the head changes (a drained supervise job or an import).
A track the head has **no prediction for** is **grey** (`CONFIDENT_MISS_COLOR`), so the map never hides a track
the head places — a prediction equal to the track's current folder still takes that folder's colour. The
**Confident only** checkbox (`#head-confident-fold`) is a **view option** in the top-right options
panel, shown only while the *Predicted folder* tab is active. Checked (`model.headColorConfident`, the
default) is the same rule the fan-out uses: a projection too weak to clear its class's match floor is also
grey. Unchecked is *All nearest*: every track with a projected folder takes its nearest folder's colour.

When the head has no prediction for a track, the genres pop-up's **Projected folder** row reads one short word
instead of an em dash (`TooltipCard.projectedNote`): *No head*, *Not analysed*, or *Loading…*. The sentence
that explains it (`TooltipCard.projectedHint`) is the chip's **hover title**: an unanalysed track ("Not analysed yet — the
head has no embedding for it"), a missing head ("No trained folder head yet"), or predictions still loading.
A weak-but-real projection still shows the folder, with *low confidence* only in the hover title.

**Show external tracks** (action `"toggle-external"`, **V**) toggles `model.showExternal`; the
permanent **Library + External / Library Only** chip mirrors it and is the click target. Hiding externals
is a visibility filter, so the chip is never dismissed and carries no ×. While hidden, the External section
is dimmed (`#folder-tree.externals-off`) and the row action popout greys out *Project folder structure onto
here*, offering *Show external tracks* at the top instead. The inline **Refit** checkbox (action
`"toggle-external-refit"`, **R**) is shown only while externals are on the map — there is nothing to refit
while they are hidden.

**Predicting folders** walks the library server-side, so while
`model.folderPredictionsLoading` is set the status bar shows *Predicting folders…* — otherwise selecting the
*Predicted folder* colour mode looks like nothing happened until the map recolours.

**Inspect card** is pinned by **Alt** (action `"pin-inspect"`, `dispatch: false`) — a bare modifier over
a hovered dot, so its keydown lives in `CanvasInteractionController._bindInspectKey`. See *Inspect card* below.

**Adding a new hotkey:**
1. Add an entry to `HOTKEY_MAP` in `hotkeys.ts` with its key, display `label`, and `description`.
2. Register a handler in `Controller._bindKeyboard` via `hk.on("your-action", () => { ... })`; a gesture
   key instead matches in `CanvasInteractionController` using `hotkeyCode` / `hotkeyKey`.
3. The F1 modal and every `<kbd>` badge update automatically; static markup opts in with
   `data-hotkey-action` / `data-hotkey-title`.

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
  first switch to a lens is a cache hit instead of a seconds-long fit (measured;
  see `audio_pipeline.md` → *Projection methods*). A **user request
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
  scope lens immediately. A bare **X** / **Shift+X** tap **near the scope** cycles its lens
  (forward / backward) through `scopeLenses()`; **far outside** it (or with none open) it
  closes the scope and animates its tracks back out, as does **Esc** or a click outside the
  region. The open scope is a `scope` view modifier chip (labelled with its lens), so
  **Esc** pops it as the newest modifier.
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

### Search popup and field

**Search is a `⌘F` popup, not a sidebar box.** The query input and ranked result
list live in `#search-modal`, a centered overlay (`SearchView`); **Esc** in the box
clears the search and closes it, **Enter** (or a result click) commits — the popup
closes and the search stays live as a **search view modifier** chip.

While a search is active and the **CLAP lens** is the one on screen (the scope's lens
when a scope is open, else `model.lensId`), the canvas draws a **search field**: a
glowing "where does my query match" layer.

- All canvas glow — this field and the hover halo — is one WebGL pass in
  `lib/glow-layer.ts` on a transparent canvas layered **under** the scatter
  (which is transparent, so the page background shows through). Each point
  is an SDF quad, so N glowing points cost one draw call instead of N radial
  gradients or a full-screen blur. Overlaps take the brightest coverage (a `MAX`
  blend, not additive) with a gradual Gaussian falloff, so a dense region reads as
  one soft zone rather than a pile of bright circles.

- `/` or **⌘/Ctrl+F** opens the popup and focuses the box. The box searches **as
  you type** (debounced ~300 ms; Enter commits at once, an emptied box clears at
  once). The input stays editable during a fetch, and a response superseded by a newer
  query is dropped. While a fetch is in flight the box dims and a **Searching…** row
  shows above the previous results (`SearchView.setBusy`), so a programmatic search has
  visible feedback.
- **The search is an active view modifier.** Once a query is live, `SearchCoordinator`
  pushes/refreshes one `search` chip (label = the query) and `CanvasView.searchActive`
  follows that, not the input's focus. Dismissing the chip (or **Esc**) clears the
  field, the fade and the selection.
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

### View controls (top tabbers + view options + status bar)

Lens and colour live in the persistent top tabbers (a click or **E** / **C**); the
contextual display toggles live in the top-right **View options** panel
(`#view-options`) — currently the *Confident only* checkbox, shown only while the
*Predicted folder* mode is active (the panel itself appears whenever at least one
`.view-option` row is relevant); and hover preview is a **right-aligned**
status-bar speaker toggle + volume (`#status-preview`). The status bar does **not** repeat the
active lens — the top tabber is that feedback. `OptionsPanelBinder` owns only the hover-preview
control; the lens `<select>`, the colour `<select>` and the preset save/delete buttons were
removed, because the top tabbers are now their single surface. (The lens-preset model methods
remain for a future surface; nothing in the UI reaches them yet.)

### Active view modifiers

The corner chip stack (`#view-modifiers`, bottom-left above the status bar, under an
**Active viewport filters:** heading) shows the **active view modifiers** — search,
proximity, scope, exclusions, tag filters, shown externals. **Esc** dismisses the
newest dismissible chip; a chip's `×` dismisses one. A **permanent** chip (the
external visibility filter) has no `×` and is never popped. The stack, its coordinator
and the clear-callback contract are in `design-patterns.md` → *Active view modifiers*;
each owner pushes/updates its own chip:

- **Search** — `SearchCoordinator` (one chip, refreshed in place per query).
- **Proximity** — `InspectCoordinator`: the distance slider pushes a chip whose inline
  slider still works after the card closes, and whose clear reveals every track it hid.
  Proximity hides are tracked in `proximityHiddenPaths`, **apart** from the explicit
  `hiddenPaths`, so the two chips never mask each other and each is dismissed
  independently.
- **Scope** — `EmbeddingCoordinator`: an open scope pushes a chip naming its lens;
  closing it (a lens switch, a click away, **X**, or the chip) removes it.
- **Exclusions** — **one chip per solo/exclude action** (pushed where the action
  runs), so the stack reads as the history of what filtered the view: a *Solo* chip whose
  × clears the exclusions, and an *Excluded* chip per excluded target whose × reveals
  just that target. Each chip names the **rule** it filtered with as a coloured chip pill
  in place of the plain label — a folder in its colour family, a cluster/key/BPM in its
  lens colour — behind the action glyph (◉ solo / ⊘ excluded); the readout is the
  keep/exclude set size (`Model.hoverTargetRule` / `bloomRule` / `selectionRule`), and a
  long path or a multi-folder selection (capped at two names, `+N more`) lives in the
  chip's hover title. A second **S** stacks another entry rather than updating the first.
  When no exclusions remain (**Alt+s**, or a chip's ×) the stale chips are dropped. It
  deliberately does **not** count the distance filter's hides.
- **Filters** — one chip per active `filterRanges` tag; its `×` drops that range.
- **External** — one **permanent** chip, never dismissed: it reads *Library +
  External* while `showExternal` is on and *Library Only* while off, and clicking it
  (or **V**) toggles. It carries an inline *Refit* toggle (**R**) on a second line,
  shown only while externals are on the map.

---

## Hover preview and tooltip

- **Hover preview** (**M** toggles; the status bar's right-aligned **speaker icon**
  + volume slider, `#status-preview`) streams the
  track through `/api/audio/…`. **M** is owned by `CanvasInteractionController`, not
  `HotkeyManager` (`dispatch: false`): a bare tap toggles the preview, but **M +
  scroll** scales `model.previewVolume` instead and shows a cursor-adjacent **panel**
  with a level progress bar and a glass blur (`CanvasView.showVolumeHud`) — a scroll
  during the hold marks the gesture used, so the keyup does not also toggle the preview.
  Toggling it **on** auditions the track already under the pointer (so **M** works
  without moving the mouse); toggling it **off** stops playback.
  It seeks to the **detected drop** when the track has a
  cached structure row, else **from the start of the track** — `previewStartSeconds` in
  `lib/structure.ts` owns that choice. A **search hit** overrides that: the hit's
  best-matching CLAP segment centre (`model.searchSections`, a duration fraction)
  wins, so hovering a result auditions the part that actually matched the query.
  **W** cycles the preview between the sections the track actually has (drop →
  break → intro; **Shift+W** backward) and re-seeks; the intro seeks to `intro_start`
  (the detected audio start after lead-in silence), falling back to `0`.
  `model.previewTrackPath` / `model.previewSection` / `model.previewTime` are
  in-memory preview state, not persisted.
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
  **location stack** with the folder and cluster tags on separate lines, a
  **preview seekbar** and two stat areas — user **Personal tags** and audio
  **Features**. The **seekbar** (`tt-seek`) is always rendered — the structure
  sections never render as chips: a timeline filled to the current position/duration with
  a **playhead** at the leading edge and a `pos / total` readout (`0:32 / 4:16`), and
  the cached structure sections (`intro 0:12–0:32`, `drop 1:04`, `break 3:12`) as
  markers; only the auditioned section's marker expands to its label, so the timeline
  stays legible. When the track is not auditioning the seekbar is **inactive**
  (`tt-seek-inactive`: dimmed, empty fill, playhead at the start, no label) rather than
  swapping to another layout. Its length is the
  `duration` the `/api/tracks/structure` response carries (probed from the file
  header, so no decode), falling back to the audio element's own duration until the
  fetch lands — but only for the track being auditioned. The fill, playhead, and readout
  are moved by `updateSeekProgress` on every audio `timeupdate`
  (the card's re-render `key` carries the duration and play state but deliberately
  not the live position). The seekbar is **interactive on the pinned card**
  (`CanvasView` delegates its mousedown to `onSeek`): clicking or dragging the
  track seeks the audition to that fraction and resumes playback, and hovering the
  track **highlights** it (`:hover`), so a paused/inactive timeline still reads as
  seekable. The card is re-queried on each drag move, since a seek rebuilds it.
  A **hotkey hint** (`tt-seek-hint`, `W seek · M mute ·
  M+scroll volume`) floats absolutely in the readout row's empty left space, so it sits
  beside the seekbar without reflowing the card; the full descriptions are its hover
  **title**. Every pill is the one `chip` component, parameterised by an
  optional **icon** and **label** around its **value**, with any elaboration in a
  hover **title** so the chip itself stays short: the location tags are told apart by
  their glyph (the folder uses a **folder** glyph and the cluster a **tag** glyph), and
  the stat chips carry a **Tempo** / **Key** caption. A chip with a colour is
  **filled** (`chip-filled`; a nested folder gets a parent→own **gradient**,
  `dirTagColors`) with its text in the dark canvas colour for contrast. The
  **pinned card expands the genres inline** (`genreList`), replacing the compact
  location stack: a caption **above** each chip (so the chip gets the full row width),
  for the track's **Folder**, the head's **Projected folder**, and its **Cluster**,
  all styled the same. The projected row reads one short word when the head has no
  prediction for the track (*No head*, *Not analysed*, *Loading…*); the full destination path,
  the *low confidence* note, and the head's reasoning are **hover titles**. The folder
  and cluster chips are navigation targets, so **S** / **H** / **F** act on them like
  a sidebar row, and a `genre-spanning` note is a **hover title** on the cluster chip,
  not a footer or badge. The plain
  hover card keeps the compact folder/cluster tags. The predictions are fetched the first
  time a card is pinned. Feature axis ids never render as tags, and BPM/key
  are **not** duplicated near the title: both render as **chips** in Features
  (`TooltipBar.chip`) because a key or a BPM has no magnitude to bar. The BPM and key
  chips are **filled** with the same colours as the tempo/key colour modes
  (`tempoColor` / `keyColor`), so the card reads with the map's palette. **Mode** is
  folded into the Key chip — `musical_key` already carries the major/minor suffix
  ("Am" = minor, "A" = major) — so it stays a canvas axis/filter but has no tooltip
  row of its own.
  Personal tags are **sorted by label** so the card does not jump as you move between
  dots. The card has a **fixed width** (300 px) so rows wrap the same way for every
  track. `buildTooltip` returns a `key`; the canvas re-renders the DOM only when that
  key changes, so pan/zoom frames do not rebuild the card.

## Analysis panel (coverage + retrigger)

The right panel's **Analysis** section is the overview of what the pipeline has
produced per track, and the way to re-run one stage.

- `AnalysisCoordinator.fetchCoverage()` reads `/api/embeddings/status` (whole
  library) and renders one row per source (Audio features, Rhythm, Structure, MAEST,
  CLAP, Supervised) as a determinate bar `done / total`; a pending count renders
  red. The payload is already one `sources[]` row per source in display order
  (`id`/`label`/`done`/`pending`/`failed`); the frontend maps the source id to its
  job kind, so adding a source is a one-place backend change. The summary line is
  `fingerprinted / total`. This is how a missing **structure** row (no section
  timestamps) becomes visible.
- Each retriggerable stage's coverage row reveals a **reload icon on hover**
  (`↻`, at the row's trailing edge) that POSTs `/api/embeddings/generate` with
  `stages: ["<kind>"]`, queueing only tracks missing that stage; a row that only
  shares a stage (Rhythm → features) or has no per-track stage (Supervised) has no
  icon. **Re-run all** stays a button below the bars and sends every stage with
  `force: true`. Every retrigger confirms first (`window.confirm`). The durable queue
  then reports progress in the status bar and the on-canvas pill as usual.
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

## Folder head (analysis panel)

The Analysis panel's **Folder head** section (`#head-section`) is the trained head's
at-a-glance status plus its training and bundle actions.

- `HeadCoordinator.refreshHeadInfo()` reads `GET /api/supervised/head` and renders one
  `.head-info-row` per fact: **Trained/Imported** timestamp (the latter wins when a head
  was imported), **Folders** (`n_classes`), **Sources** (`input_sources`), **Input**
  (`in_dim`), **Match floor** (`match_percentile`), and **Run** (`run_id`). With no head
  installed it renders a single *No folder head yet — Train to learn your folder structure*
  hint. It is refetched on startup, after an import, and when the queue sees a `supervise`
  run drain (`AnalysisCoordinator.refreshQueueStatus` → `head.refreshHeadInfo`).
- **Train** sends the checked *Training folders* as `folders`; an empty selection means
  every **My library** root (the default) — external roots are only learned from when checked
  explicitly. The checkbox tree is the folder tree from `GET /api/folders`, fetched lazily when
  the `<details>` is opened — never on startup — and rendered as `.head-folder-row` rows with
  collapsible children. *All library folders* clears the selection. External roots are labelled
  `(external)`. The checked folders are mirrored onto `model.trainFolders` and persisted in
  `localStorage`, so a selection survives a reload. The queue disables Train while a `supervise`
  run is outstanding (`AnalysisCoordinator._syncApplyEnabled` → `head.syncTrainEnabled`).
- **Export / Import / Compare** are the head bundle tools. Export downloads the installed head;
  Import installs an uploaded bundle and re-projects the library (retrying once with `force=1` when
  the server refuses a head trained on different encoder versions, after a `confirm` explains the
  mismatch); Compare reports the shared folders and mean centroid cosine against an uploaded bundle.
  The `#head-note` line carries their results.

## Inspect card (pinned, interactive)

The tooltip is `pointer-events: none` and follows the hover. **Alt** over a hovered
dot pins it: `CanvasView.inspect` is set, the card anchors to the dot's
*current* screen position (so a viewport tween carries it along), gets
`pointer-events: auto` and a **Navigate / Describe** tab strip. It stays
open when the pointer leaves the canvas (that is how you reach it), and closes on
**Esc**, a plain click on empty canvas, the card's `×`, or another Alt press. A
second **Alt** only releases the pin — it does **not** end the audition, which keeps
playing while the pointer is still over the dot.
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
  The filter is a `proximity` **view modifier** chip whose inline slider drives the
  same path, so it survives the card closing; its `×` reveals every track it hid.
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
