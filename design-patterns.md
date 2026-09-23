# Trackspace — design patterns & interaction affordances

How interaction is modelled, so a new element behaves like the ones already there.
`ui_conventions.md` states the rules for specific surfaces (canvas, tags, folders,
hotkeys); this file names the *patterns* those surfaces share. When you add a UI
element, find the pattern it belongs to and implement it through the same seam.

---

## Pattern: navigation targets

**The problem.** Folders, clusters, musical keys and BPM all mean "a set of tracks".
Before this pattern each one had its own hover field and its own solo/fit branch, so the
affordances drifted: a folder row glowed and answered **S**/**H**, a cluster row glowed
but had its own branch, and the card's key/BPM text did nothing.

**The pattern.** One descriptor names the hovered set, and one matcher resolves it:

```ts
type HoverTarget =
  | { kind: "folder"; prefix: string }   // "" = library root
  | { kind: "cluster"; id: number }
  | { kind: "key"; value: string }       // musical key, e.g. "Am"
  | { kind: "bpm"; value: number }       // ± BPM_MATCH_TOLERANCE
  | { kind: "paths"; paths: string[]; color?: string }; // a bloom cluster
```

- `Model.trackMatchesHoverTarget(track, target)` is the **only** place a target is
  interpreted. Everything else composes it.
- `CanvasView.hoverTarget` holds the *currently* hovered target. It is visual-only
  state on the view (no model event), per `ui_conventions.md` → *Adding new
  interaction surfaces*.
- Whichever list renders an element owns setting/clearing that field on
  `mouseenter`/`mouseleave`: the folder tree for folders, the cluster panel for
  clusters, the inspect card for key/BPM chips, and the selected-tracks overlay —
  whose tabs break the current selection down by folder, BPM, key, or cluster — for
  all four. A BPM/key overlay row is a *bloom cluster*, so it names its members as a
  `paths` set (and carries the cluster's colour for the halo) rather than a single
  representative value.

**What every target gets, for free:**

| Affordance | Implementation | Effect |
|---|---|---|
| Glow | `CanvasView._inHoverGlow` / `_hoverGlowColor` | matching dots halo (folder colour, cluster swatch, a bloom cluster's own colour, or the neutral accent) |
| **S** solo | `CanvasInteractionController.solo()` → `Model.focusPaths` / `Model.bloomPaths` | over a bloomed dot: isolate its bloom class (folder subtree for folder colour); else the target's tracks; press again to show all |
| **H** fit | `CanvasInteractionController.fitView()` | 80th-percentile disc around the target's visible tracks |
| **F** select | `Model.selectOnlyHoverTarget` | replace the selection with the target's visible tracks |
| **Shift+S** exclude | folder: `hideFolderTree`; others: `hidePaths` | hide the target; press again on a fully hidden target to reveal it |

`solo()` resolves a hovered **bloomed dot** first (`Model.bloomPaths`), then the target
through `CanvasInteractionController.effectiveHoverTarget()`: a hovered target wins, otherwise
the hovered canvas dot's folder. That fallback is why **S**/**H** over a bare dot still
mean "this dot's folder".

**Adding a new navigation target kind.** Four steps, no new branches:

1. Add the kind to `HoverTarget` and one `case` to `Model.trackMatchesHoverTarget`.
2. Render the element with a `data-hover-target='{"kind":…}'` attribute (the card's
   `markTarget()` helper does this and adds the `.tt-target` class + hint `title`).
3. If the element lives in the inspect card, it is picked up automatically: the card
   delegates `mouseover`/`mouseout` on `#tooltip` to `CanvasView.hoverTarget`.
4. Test the matcher in `model`/`tooltip` tests and the glow in
   `canvas-interaction.test.ts`.

**Presentation.** A navigation target must look and behave like a list row, not like
body text:

- `.tt-target` gives it the pointer cursor and a hover background.
- The `title` hint says what hover does (`Highlight matches · S solo · H fit`), so the
  affordance is discoverable without the F1 modal.
- It must not be a click target with a different meaning: clicking a card result row
  *focuses* one track, while hovering a target *glows a set*.

**Constraint.** The inspect card is `pointer-events: none` until **Alt** pins it
(`.tooltip.pinned`). Card targets are therefore hoverable only in the pinned card —
the unpinned tooltip follows the cursor and must not swallow canvas hovers.

---

## Pattern: hover state is view state

Hover never mutates the model. `hoverTarget`, `hoveredSiblingFolder` and
`hoveredIdx` live on `CanvasView` and repaint via `scheduleDraw()`; they do not emit
`"change"`. The model owns only the *pure matcher* and the mutations an explicit
action (S/H/F/Shift+S) performs. This keeps a mousemove from re-rendering every
sidebar panel.

---

## Pattern: explicit actions go through the controller

A view never calls the API or mutates the model directly. Views expose callbacks
(`onPickFolder`, `onInspectAction`, …) that `Controller` implements, and self-contained
flows get a coordinator with structural deps (`SearchCoordinator`, `InspectCoordinator`,
`AnalysisCoordinator`, …). This is what makes the coordinators unit-testable without the
DOM-heavy component classes — see `architecture.md` → *Frontend orchestration*.

---

## Pattern: perceptual input controls

A raw `0–1` value bound linearly to an output reads as broken at both ends: most of
the travel does nothing audible, then the last few pixels are deafening. For any control
whose output is perceived non-linearly (gain, speed, zoom):

- store the **slider position** on the model (`previewVolume`), and apply a documented
  **taper** at the boundary (`previewGain` = `v²`),
- default to a **moderate** value (`DEFAULT_PREVIEW_VOLUME`), not full scale,
- show a **readout** (`.option-value`) so the position is legible,
- treat an **unset-or-zero** persisted value as unset — `Number(null) === 0` silently
  starts a control muted, so a 0 default is indistinguishable from "never configured".

---

## Pattern: hotkeys describe their target

`HOTKEY_MAP` entries are the single source of truth: the F1 modal is rendered from
them, and their `description` must name every target the action resolves
(e.g. "Isolate to hovered cluster/folder or selection"). When an action gains a new
target kind, update its description — the modal follows automatically.

---

## Pattern: persistent canvas tabbers

A hotkey whose effect is a **mode** (which lens is active, how dots are coloured)
needs feedback. The feedback is the **tabber itself**, always visible at the top of
the canvas: a lens row and a colour-mode row, with the active tab highlighted, so
pressing **E** / **C** moves the highlight and a click switches directly. No
transient flash, no toast — a toast stacks a history of "now X" / "now Y", and a
flash disappears exactly when the user wants to confirm the state.

- `CanvasView._drawTopTabs(w)` draws both rows with `_drawTabs` (the same
  routine the scope tab bar uses) and stores their screen rects.
- The tabbers are **chrome**, not data: clicking one is routed before any gesture
  (`topLensTabAt` / `topColorTabAt` in `canvas-interaction.ts`) and only calls the
  coordinator (`switchLens`, `setColorMode`).
- It is view state: the rows read `model.activeBaseLensId()` and `model.colorMode`
  each frame, so no model event or persistence is needed — just `scheduleDraw()`.
- A mode that cannot apply is still explained with a warn toast; a one-off outcome
  ("Analysis complete") is still a toast.
- **A tabber can carry a second, independent cursor.** The colour row draws both the
  active-mode highlight and the **selection-bloom cursor** (`model.selectionBloom`,
  moved with **D** / **Shift+D**), because the bloom variable is a separate choice from
  the dot colour. The cursor ring borrows the hovered class's colour, and a chip under
  it names the hovered class and (where the variable has a range) the range's semantic
  meaning (`+2 parent directories`, `±16% BPM`). For a folder the chip draws the path
  segment by segment, highlighting the segment the range climbed to and fading deeper
  ones to grey, so the chip shows *which ancestor* the bloom reached. A click sets both;
  the cursor alone moves with the key. The extra marker and chip are drawn by
  `_drawTabCursor` / `_drawBloomChip` after `_drawTabs`, so they never change the hit
  rects a click resolves against.

---

## Pattern: active view modifiers

**The problem.** Several affordances change *what the canvas shows* without being a
mode: a semantic search, a neighbourhood-proximity filter, a scope, the exclusions, a
tag filter, the shown externals. Each was surfaced differently — search lived in the
right panel, "Show all" was a lone button, the proximity slider only existed inside
the pinned inspect card, a scope only had its on-canvas region — so there was no one
place that answered "what is currently constraining my view?", and no one key that
undid the most recent one.

**The pattern.** One ordered stack of descriptors on the model, one owner for the
clear callbacks, one chip stack in a canvas corner:

```ts
type ViewModifierKind =
  | "search" | "proximity" | "scope" | "exclusions" | "filters" | "external";
interface ViewModifier { id: string; kind: ViewModifierKind; label: string; detail?: string; value?: number; checked?: boolean; permanent?: boolean; glyph?: string; rule?: RuleChip; }
interface ViewRule extends ViewModifier {
  matches?(track: Track): boolean;      // true = keep; absent = no visibility effect
  scope?: "tags" | "all";              // a tag filter only applies in tags mode
}
```

An exclusions chip carries a `rule` (`RuleChip`) naming the class it filtered with,
rendered as a coloured chip pill in place of the plain label and prefixed by its action
`glyph` (◉ solo / ⊘ excluded); the chip is built by the shared `lib/chip` pill, so the
stack and the hover card share one geometry. `Model.hoverTargetRule` / `bloomRule` /
`selectionRule` build the readouts, and `CanvasInteractionController.solo` returns the
`{ rule, keep }` it isolated so `Controller` can push a chip whose detail is the keep-set
size.

- `Model.viewRules` is the stack, **oldest → newest**; `Model.viewModifiers` is a
  getter over the same array (the corner view renders the descriptors). A rule that
  carries `matches` also decides visibility, so the chip stack and the visible set
  share one source: `Model.isVisible(track)` ANDs the view union, the explicit
  exclusions and the distance filter, then every active rule's `matches`. The
  exclusions, proximity, tag-filter and external chips each carry their own
  `matches`; a search fade or a scope re-layout has none. Session-only, like the
  exclusions and the scope it may describe.
- `ViewModifierCoordinator` owns the stack operations and a `Map<id, clear>`:
  `push(mod, clear)`, `update(id, patch)`, `dismiss(id)`, `remove(id)`,
  `popNewest()`, `find(kind)`. A modifier owns only its *descriptor*; the
  coordinator owns the callback that undoes it, so the clear runs in one place.
- `ViewModifierView` renders the stack as chips in `#view-modifiers` (bottom-left,
  above the status bar, under an **Active viewport filters:** heading). Each chip has
  a label, an optional detail readout, and a `×` — unless it is a **permanent** view
  filter (the external visibility chip), which has no `×` and is never popped. A
  `proximity` chip exposes its slider inline; an `external` chip exposes its refit
  toggle inline and its label is itself the show/hide control — the chip is the
  control, not just a status light.
- **Esc dismisses the newest dismissible modifier.** `Controller`'s `deselect` handler closes a
  transient surface first (the search popup, the pinned inspect card, an in-flight
  gesture), then calls `popNewest()`, and only clears the selection when the stack is
  empty.

**Clear callbacks must be idempotent, and direct closes must remove their entry.**
A modifier can be cleared from outside the chip (a lens switch drops a scope, **Alt+s**
clears the exclusions, a new inspect pin replaces the proximity filter), so each owner
calls `viewMods.remove(id)` on its own direct-close path *as well as* registering the
callback. `dismiss` removes the entry first and then runs the callback; `remove` removes
it without running the callback, for a path that already reverted the state.

**Adding a view modifier kind:**

1. Add the kind to `VIEW_MODIFIER_KINDS`.
2. Push/update it from the owner as its state changes (`ViewModifierSink`), and
   register a clear callback that reverts the underlying model state. If the kind
   *filters* tracks, give the descriptor a `matches(track)` predicate (and a
   `scope: "tags"` when it only applies in tags mode), so `Model.isVisible`
   consults it without a parallel branch.
3. On the owner's direct-close path, call `viewMods.remove(id)` so the chip never
   goes stale.
4. Refresh in place (`update`) rather than re-pushing on every step, so a slider drag
   or a keystroke burst does not stack chips.

**Not every mode is a modifier.** A *base mode* (the lens, the colour mode) is
persisted and always exactly one active; it stays in the top tabbers. An inspect card
pin is a focused overlay, not a filter of the visible set; it stays separate and **Esc**
closes it before the stack is popped.

---

## Checklist for a new interaction surface

1. Is it a *set of tracks* under the pointer? Make it a `HoverTarget` (above) — do
   not add a parallel hover field.
2. Does it change the model structurally? `m.emit("change")`. Only in-memory tag
   values? `m.emit("tags-dirty")`. Purely visual? `canvas.scheduleDraw()` only.
3. Is it a key binding? Add it to `HOTKEY_MAP`, register a handler in
   `Controller._bindKeyboard`, and it appears in F1 automatically.
4. Is it a click on a list element? It needs the pointer cursor and a hover
   highlight; a tooltip/`title` says what it does.
5. Does it need an API call? Put the fetch in a coordinator with an injectable
   `fetchX` dep so the test never touches `fetch`.
