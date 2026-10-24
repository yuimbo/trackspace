import { createKbd, el } from "./lib/dom";

/** Modifier keys, in the fixed column order the F1 list lays them out. */
export type Modifier = "mod" | "ctrl" | "alt" | "shift";
/** A pointer gesture, alone (`Shift+drag`) or combined with a held key (`M`+scroll). */
export type Gesture = "click" | "dblclick" | "drag" | "middle-drag" | "scroll";
/** What the user presses: modifiers plus a key and/or a pointer gesture. */
export interface Chord {
  mods: Modifier[];
  key?: string;
  gesture?: Gesture;
}

export interface HotkeyBinding {
  action: string;
  /** `KeyboardEvent.key` the binding matches; a bare letter matches either case. */
  key: string;
  /**
   * Optional `KeyboardEvent.code` override for handlers that match the physical
   * key rather than the layout-resolved `key` (the canvas gesture keys X/M/D).
   */
  code?: string;
  mod?: boolean;
  shift?: boolean;
  alt?: boolean;
  /** Pointer gesture performed while the key is held (`X`+drag, `M`+scroll); F1 only. */
  gesture?: Gesture;
  /** Display label for `<kbd>` badges and the F1 list. */
  label: string;
  /** Short heading shown beside the key in the F1 list. */
  title: string;
  description: string;
  /**
   * When ``false`` the entry is listed in F1 but not dispatched by
   * `HotkeyManager` — the controller or the canvas gesture handlers own its
   * keydown/keyup (e.g. **M**, which toggles on a bare tap but adjusts the
   * preview volume when held + scrolled).
   */
  dispatch?: boolean;
}

/**
 * True while *target* is a text-entry field, so a shortcut must not fire there.
 * One source of truth for the hotkey manager, the canvas gestures and the
 * controller's own key handlers.
 */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  if (el.tagName === "TEXTAREA") return true;
  if (el.tagName !== "INPUT") return false;
  const type = (el as HTMLInputElement).type.toLowerCase();
  return (
    type === "" ||
    type === "text" ||
    type === "password" ||
    type === "email" ||
    type === "search" ||
    type === "url" ||
    type === "tel" ||
    type === "number" ||
    type === "date" ||
    type === "time" ||
    type === "datetime-local" ||
    type === "month" ||
    type === "week"
  );
}

const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
const MOD_SYMBOL = IS_MAC ? "⌘" : "Ctrl+";

export const HOTKEY_MAP: HotkeyBinding[] = [
  { action: "focus-search",   key: "/",      mod: false, shift: false, alt: false, label: "/",                      title: "Focus search",    description: "Focus semantic search" },
  { action: "focus-search",   key: "f",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}F`,         title: "Focus search",    description: "Focus semantic search" },
  { action: "toggle-preview",  key: "m",      mod: false, shift: false, alt: false, label: "M",                      title: "Hover preview",   description: "Toggle the hover preview", dispatch: false },
  { action: "preview-volume",  key: "m",      mod: false, shift: false, alt: false, gesture: "scroll", label: "M+scroll", title: "Preview volume", description: "Adjust the hover-preview volume", dispatch: false },
  { action: "fit-view",        key: "h",      mod: false, shift: false, alt: false, label: "H",                      title: "Fit view",        description: "Fit the view to the hovered element or track's folder, else the full map" },
  { action: "focus-folder",    key: "f",      mod: false, shift: false, alt: false, label: "F",                      title: "Focus folder",    description: "Select all tracks of the hovered element or the track's folder, including subfolders" },
  { action: "isolate-selection", key: "s",  mod: false, shift: false, alt: false, label: "s",                      title: "Isolate",         description: "Isolate the hovered class, element or selection; again to show all" },
  { action: "exclude-selection", key: "s",  mod: false, shift: true, alt: false, label: `Shift+s`,                title: "Exclude",         description: "Exclude the hovered element or selection; again to restore it" },
  { action: "show-all-tracks",   key: "s",  mod: false, shift: false, alt: true,  label: "Alt+s",                 title: "Show all tracks", description: "Clear all exclusions" },
  { action: "cycle-preview-section", key: "w", mod: false, shift: false, alt: false, label: "W",                  title: "Next section",    description: "Cycle the hover preview between drop, break and intro" },
  { action: "cycle-preview-section-prev", key: "w", mod: false, shift: true, alt: false, label: "Shift+W",          title: "Previous section", description: "Cycle the hover preview backward" },
  { action: "cycle-selection-bloom", key: "d", mod: false, shift: false, alt: false, label: "D",                    title: "Next bloom tab",  description: "Move the selection-bloom cursor across the colour tabs", dispatch: false },
  { action: "cycle-selection-bloom-prev", key: "d", mod: false, shift: true, alt: false, label: "Shift+D",          title: "Previous bloom tab", description: "Move the selection-bloom cursor backward", dispatch: false },
  { action: "bloom-range",     key: "d",      mod: false, shift: false, alt: false, gesture: "scroll", label: "D+scroll", title: "Bloom range", description: "Adjust how far the selection bloom reaches", dispatch: false },
  { action: "scope-gesture",   key: "x", code: "KeyX", mod: false, shift: false, alt: false, gesture: "drag", label: "X", title: "Box scope",   description: "Scope a box at the cursor; in 3-D a sphere around the nearest dot", dispatch: false },
  { action: "scope-circle",    key: "x",      mod: false, shift: false, alt: false, gesture: "scroll", label: "X+scroll", title: "Circle scope", description: "Scope a circle at the cursor", dispatch: false },
  { action: "scope-lens",      key: "x",      mod: false, shift: false, alt: false, label: "X",                      title: "Next scope lens", description: "Cycle the lens of the scope the cursor is near", dispatch: false },
  { action: "scope-lens-prev", key: "x",      mod: false, shift: true,  alt: false, label: "Shift+X",              title: "Previous scope lens", description: "Cycle the scope lens backward", dispatch: false },
  { action: "toggle-external", key: "v", mod: false, shift: false, alt: false, label: "V",                    title: "External tracks", description: "Show/hide external-root tracks" },
  { action: "toggle-external-refit", key: "r", mod: false, shift: false, alt: false, label: "R",               title: "Refit with external", description: "Refit the layout with external tracks included, while they are shown" },
  { action: "toggle-3d",        key: "3",      mod: false, shift: false, alt: false, label: "3",               title: "3-D view",        description: "Toggle the 3-D orbit view of the map" },
  { action: "select-all",      key: "a",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}A`,         title: "Select all",      description: "Select all tracks" },
  { action: "undo",            key: "z",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}Z`,         title: "Undo",            description: "Undo last action" },
  { action: "deselect",        key: "Escape", mod: false, shift: false, alt: false, label: "Esc",                    title: "Close / deselect", description: "Close the search popup or inspect card, else clear the newest filter, else deselect" },
  { action: "toggle-view",     key: "e",      mod: false, shift: false, alt: false, label: "E",                      title: "Next lens",       description: "Cycle lenses: tags → combined → per-source maps → feature wheels" },
  { action: "toggle-view-prev", key: "e",      mod: false, shift: true,  alt: false, label: "Shift+E",              title: "Previous lens",   description: "Cycle lenses backward" },
  { action: "toggle-clusters", key: "c",      mod: false, shift: false, alt: false, label: "C",                      title: "Next colour",     description: "Cycle dot colour: folder → predicted folder → cluster → tempo → key" },
  { action: "toggle-clusters-prev", key: "c", mod: false, shift: true,  alt: false, label: "Shift+C",              title: "Previous colour", description: "Cycle dot colour backward" },
  { action: "pin-inspect",      key: "Alt",    mod: false, shift: false, alt: false, label: "Alt",                    title: "Pin inspect",     description: "Pin the inspect card for the hovered track", dispatch: false },
  { action: "toggle-shortcuts", key: "F1",     mod: false, shift: false, alt: false, label: "F1",                     title: "Shortcuts",       description: "Show this shortcut list", dispatch: false },
];

export type HotkeyAction = (typeof HOTKEY_MAP)[number]["action"];

/** The first binding declared for an action, or `undefined` when unbound. */
export function hotkeyBinding(action: string): HotkeyBinding | undefined {
  return HOTKEY_MAP.find((b) => b.action === action);
}

/** Display label for an action (e.g. `⇧S`), or `""` when unbound. */
export function hotkeyLabel(action: string): string {
  return hotkeyBinding(action)?.label ?? "";
}

/** `KeyboardEvent.code` a physical-key handler should match, if any. */
export function hotkeyCode(action: string): string | undefined {
  const b = hotkeyBinding(action);
  if (!b) return undefined;
  if (b.code) return b.code;
  return /^[a-z]$/i.test(b.key) ? `Key${b.key.toUpperCase()}` : undefined;
}

/** `KeyboardEvent.key` a key handler should match, if any. */
export function hotkeyKey(action: string): string | undefined {
  return hotkeyBinding(action)?.key;
}

export interface MouseHint {
  /** Stable key the F1 zone table references. */
  id: string;
  /** Alternatives (e.g. Alt+drag or middle-drag) rendered side by side. */
  chords: Chord[];
  /** Short heading shown in the F1 list next to the gesture. */
  title: string;
  description: string;
}

const gesture = (g: Gesture, ...mods: Modifier[]): Chord => ({ mods, gesture: g });

export const MOUSE_HINTS: MouseHint[] = [
  { id: "lasso-union",        chords: [gesture("drag", "shift")],          title: "Lasso select",     description: "Add the lassoed tracks to the selection" },
  { id: "lasso-intersect",    chords: [gesture("drag", "ctrl", "shift")],  title: "Lasso intersect",  description: "Keep only tracks inside every lasso" },
  { id: "box-intersect",      chords: [gesture("drag", "ctrl")],           title: "Box intersect",    description: "Keep only tracks inside the box" },
  { id: "pan",                chords: [gesture("drag", "alt")],            title: "Pan",              description: "Move the viewport" },
  { id: "pan-3d",             chords: [gesture("drag", "alt", "shift")],   title: "Pan in 3-D",       description: "Move the viewport in the 3-D view" },
  { id: "orbit",              chords: [gesture("drag", "alt"), gesture("middle-drag")], title: "Orbit in 3-D", description: "Turn the map around its centre" },
  { id: "click-folder",       chords: [gesture("click")],                  title: "Select folder",    description: "Select all tracks in this folder and its subfolders" },
  { id: "mod-click-folder",   chords: [gesture("click", "mod")],           title: "Add folder",       description: "Add this folder’s tracks to the selection" },
  { id: "shift-click-folder", chords: [gesture("click", "shift")],         title: "Select range",     description: "Add every folder between the anchor and this row; a plain click sets the anchor" },
  { id: "alt-click-folder",   chords: [gesture("click", "alt")],           title: "Exclude folder",   description: "Hide this folder from the view" },
  { id: "click-away",         chords: [gesture("click")],                  title: "Close scope",      description: "Click far from the open scope" },
  { id: "drag-to-folder",     chords: [gesture("drag")],                   title: "Move to folder",   description: "Move dragged files into a folder" },
  { id: "dblclick-tag",       chords: [gesture("dblclick")],               title: "Rename tag",       description: "Edit the tag’s name in place" },
];

/**
 * The F1 modal's wireframe: one zone per main-screen region, positioned like
 * the real UI (see the `grid-template-areas` in trackspace.css). Each group
 * lists actions and mouse-hint ids; `shortcutCoverageProblems()` asserts every
 * binding and hint ends up in exactly one zone, so a new shortcut can't
 * silently vanish from the modal. Siblings (same key or gesture, different
 * modifiers) must sit next to each other: adjacent siblings render as one
 * cluster with aligned modifier slots.
 */
export interface ShortcutItem {
  /** A `HOTKEY_MAP` action; all of its bindings render in one row. */
  action?: string;
  /** A `MOUSE_HINTS` id. */
  mouse?: string;
}

export interface ShortcutGroup {
  title?: string;
  /** Render as a floating card spanning the whole zone (the search popup). */
  card?: boolean;
  items: ShortcutItem[];
}

export interface ShortcutZone {
  /** CSS grid-area (the zone classes are `.zone-<area>`). */
  area: string;
  title: string;
  /** `"stack"` (default): groups in one column. `"row"`: groups side by side, wrapping. */
  layout?: "stack" | "row";
  groups: ShortcutGroup[];
}

const act = (action: string): ShortcutItem => ({ action });
const mouse = (id: string): ShortcutItem => ({ mouse: id });

export const SHORTCUT_ZONES: ShortcutZone[] = [
  {
    area: "folders",
    title: "Folders",
    groups: [
      {
        items: [
          act("focus-folder"),
          mouse("click-folder"),
          mouse("mod-click-folder"),
          mouse("shift-click-folder"),
          mouse("alt-click-folder"),
          mouse("drag-to-folder"),
        ],
      },
    ],
  },
  {
    area: "topbar",
    title: "Top bar",
    layout: "row",
    groups: [
      {
        title: "Active filters",
        items: [
          act("isolate-selection"),
          act("exclude-selection"),
          act("show-all-tracks"),
          act("toggle-external"),
          act("toggle-external-refit"),
        ],
      },
      { title: "Lens", items: [act("toggle-view"), act("toggle-view-prev")] },
      {
        title: "Colour",
        items: [
          act("toggle-clusters"),
          act("toggle-clusters-prev"),
          act("cycle-selection-bloom"),
          act("cycle-selection-bloom-prev"),
          act("bloom-range"),
        ],
      },
    ],
  },
  {
    area: "canvas",
    title: "Canvas",
    layout: "row",
    groups: [
      { title: "Search", card: true, items: [act("focus-search")] },
      {
        title: "Scope",
        items: [act("scope-gesture"), act("scope-circle"), act("scope-lens"), act("scope-lens-prev")],
      },
      {
        title: "Select & pan",
        items: [
          mouse("lasso-union"),
          mouse("lasso-intersect"),
          mouse("box-intersect"),
          mouse("pan"),
          mouse("pan-3d"),
        ],
      },
      { title: "Inspect", items: [act("pin-inspect"), act("fit-view"), mouse("click-away")] },
      { title: "3-D", items: [act("toggle-3d"), mouse("orbit")] },
    ],
  },
  {
    area: "properties",
    title: "Properties",
    groups: [{ items: [mouse("dblclick-tag")] }],
  },
  {
    area: "status",
    title: "Statusbar",
    layout: "row",
    groups: [
      {
        title: "Hover preview",
        items: [
          act("toggle-preview"),
          act("preview-volume"),
          act("cycle-preview-section"),
          act("cycle-preview-section-prev"),
        ],
      },
      {
        title: "General",
        items: [act("select-all"), act("undo"), act("deselect"), act("toggle-shortcuts")],
      },
    ],
  },
];

// ─── F1 rendering ───────────────────────────────────────────

/** Fixed left→right slot order for modifier caps, so a cluster's columns line up. */
const MODIFIER_ORDER: Modifier[] = ["mod", "ctrl", "alt", "shift"];

const MODIFIER_CAP: Record<Modifier, { symbol: string; name: string }> = {
  mod: { symbol: IS_MAC ? "⌘" : "Ctrl", name: IS_MAC ? "Command" : "Control" },
  ctrl: { symbol: IS_MAC ? "⌃" : "Ctrl", name: "Control" },
  alt: { symbol: IS_MAC ? "⌥" : "Alt", name: IS_MAC ? "Option" : "Alt" },
  shift: { symbol: "⇧", name: "Shift" },
};

const GESTURE_CAP: Record<Gesture, string> = {
  click: "click",
  dblclick: "dbl-click",
  drag: "drag",
  "middle-drag": "mid-drag",
  scroll: "scroll",
};

function keyCapLabel(key: string): string {
  if (key === "Escape") return "Esc";
  if (key === "Alt") return MODIFIER_CAP.alt.symbol;
  return key.length === 1 ? key.toUpperCase() : key;
}

function bindingChord(b: HotkeyBinding): Chord {
  const mods: Modifier[] = [];
  if (b.mod) mods.push("mod");
  if (b.alt) mods.push("alt");
  if (b.shift) mods.push("shift");
  return { mods, key: b.key, gesture: b.gesture };
}

/** One F1 row's content: the chords that trigger it plus its heading and blurb. */
interface ShortcutEntry {
  chords: Chord[];
  title: string;
  description: string;
}

function entryFor(item: ShortcutItem): ShortcutEntry | null {
  if (item.mouse) {
    const h = MOUSE_HINTS.find((x) => x.id === item.mouse);
    return h ? { chords: h.chords, title: h.title, description: h.description } : null;
  }
  const bindings = HOTKEY_MAP.filter((b) => b.action === item.action);
  if (bindings.length === 0) return null;
  const first = bindings[0];
  return { chords: bindings.map(bindingChord), title: first.title, description: first.description };
}

/** What siblings share: the key when there is one, else the pointer gesture. */
function siblingBase(e: ShortcutEntry): string | null {
  if (e.chords.length !== 1) return null;
  const c = e.chords[0];
  if (c.key) return `key:${c.key.toLowerCase()}`;
  return c.gesture ? `gesture:${c.gesture}` : null;
}

/** The slots a sibling cluster reserves so its rows' caps line up. */
interface ClusterSlots {
  mods: Modifier[];
  key: boolean;
  gesture: boolean;
}

function clusterSlots(entries: ShortcutEntry[]): ClusterSlots {
  const chords = entries.map((e) => e.chords[0]);
  return {
    mods: MODIFIER_ORDER.filter((m) => chords.some((c) => c.mods.includes(m))),
    key: chords.some((c) => c.key),
    gesture: chords.some((c) => c.gesture),
  };
}

function chordEl(chord: Chord, slots?: ClusterSlots): HTMLElement {
  const wrap = el("span", "chord");
  const mods = slots?.mods ?? MODIFIER_ORDER.filter((m) => chord.mods.includes(m));
  for (const m of mods) {
    if (chord.mods.includes(m)) {
      const cap = createKbd(MODIFIER_CAP[m].symbol, "cap mod");
      cap.title = MODIFIER_CAP[m].name;
      wrap.appendChild(cap);
    } else {
      wrap.appendChild(el("span", "cap-slot mod"));
    }
  }
  if (chord.key) wrap.appendChild(createKbd(keyCapLabel(chord.key), "cap key"));
  else if (slots?.key) wrap.appendChild(el("span", "cap-slot key"));
  if (chord.gesture) wrap.appendChild(createKbd(GESTURE_CAP[chord.gesture], "cap gesture"));
  else if (slots?.gesture) wrap.appendChild(el("span", "cap-slot gesture"));
  return wrap;
}

/** `[chords]` beside `Title` over a small description. */
function shortcutRow(e: ShortcutEntry, slots?: ClusterSlots): HTMLElement {
  const row = el("div", "shortcut-row");
  const chords = el("span", "chords");
  e.chords.forEach((c, i) => {
    if (i > 0) chords.appendChild(el("span", "chord-sep", "/"));
    chords.appendChild(chordEl(c, slots));
  });
  row.appendChild(chords);
  const text = el("div", "shortcut-text");
  text.appendChild(el("span", "shortcut-title", e.title));
  text.appendChild(el("p", "shortcut-desc", e.description));
  row.appendChild(text);
  return row;
}

/**
 * Render a group's items into *list*. Adjacent single-chord entries sharing a
 * base (S / ⇧S / ⌥S, or ⇧drag / ⌥drag) become one `.shortcut-cluster` whose
 * modifier, key and gesture slots line up; everything else is a plain row.
 */
function renderRows(list: HTMLElement, items: ShortcutItem[]): void {
  const entries = items.map(entryFor).filter((e): e is ShortcutEntry => e !== null);
  let i = 0;
  while (i < entries.length) {
    const base = siblingBase(entries[i]);
    let j = i + 1;
    while (base !== null && j < entries.length && siblingBase(entries[j]) === base) j++;
    if (j - i < 2) {
      list.appendChild(shortcutRow(entries[i]));
    } else {
      const run = entries.slice(i, j);
      const slots = clusterSlots(run);
      const box = el("div", "shortcut-cluster");
      for (const e of run) box.appendChild(shortcutRow(e, slots));
      list.appendChild(box);
    }
    i = j;
  }
}

export function renderShortcutList(container: HTMLElement): void {
  container.innerHTML = "";
  const grid = el("div", "shortcut-wireframe");
  for (const zone of SHORTCUT_ZONES) {
    const zoneEl = el("section", `shortcut-zone zone-${zone.area}`);
    zoneEl.appendChild(el("h3", "shortcut-zone-title", zone.title));
    const groups = el("div", zone.layout === "row" ? "shortcut-groups row" : "shortcut-groups");
    for (const g of zone.groups) {
      const groupEl = el("div", g.card ? "shortcut-group card" : "shortcut-group");
      if (g.title) groupEl.appendChild(el("h4", "shortcut-group-title", g.title));
      renderRows(groupEl, g.items);
      groups.appendChild(groupEl);
    }
    zoneEl.appendChild(groups);
    grid.appendChild(zoneEl);
  }
  container.appendChild(grid);
}

/**
 * Every keyboard binding and mouse hint must be listed in exactly one F1 zone,
 * so a shortcut can't silently disappear from the modal. Returns the problems
 * found (empty when the table is complete); covered by vitest.
 */
export function shortcutCoverageProblems(): string[] {
  const problems: string[] = [];
  const seenActions = new Set<string>();
  const seenMice = new Set<string>();
  for (const zone of SHORTCUT_ZONES) {
    for (const g of zone.groups) {
      for (const item of g.items) {
        if (item.action) {
          if (seenActions.has(item.action)) problems.push(`action listed twice: ${item.action}`);
          seenActions.add(item.action);
        }
        if (item.mouse) {
          if (seenMice.has(item.mouse)) problems.push(`mouse hint listed twice: ${item.mouse}`);
          seenMice.add(item.mouse);
          if (!MOUSE_HINTS.some((h) => h.id === item.mouse)) problems.push(`unknown mouse hint: ${item.mouse}`);
        }
      }
    }
  }
  for (const b of HOTKEY_MAP) {
    if (!seenActions.has(b.action)) problems.push(`action missing from zones: ${b.action}`);
  }
  for (const h of MOUSE_HINTS) {
    if (!seenMice.has(h.id)) problems.push(`mouse hint missing from zones: ${h.id}`);
  }
  return problems;
}

/**
 * Fill static markup from the binding table so hotkey hints are never hard-coded:
 *
 * - `[data-hotkey-action]` — the element's text becomes the action's label.
 * - `…` plus `data-hotkey-template` — the template is used as the `title` with
 *   `{key}` replaced by the label (for prose hints that embed a key).
 * - `[data-hotkey-title]` — the element's `title` becomes the action description.
 */
export function applyHotkeyLabels(root: ParentNode = document): void {
  for (const el of root.querySelectorAll<HTMLElement>("[data-hotkey-action]")) {
    const label = hotkeyLabel(el.dataset.hotkeyAction!);
    if (!label) continue;
    const template = el.dataset.hotkeyTemplate;
    if (template) el.title = template.replace(/\{key\}/g, label);
    else el.textContent = label;
  }
  for (const el of root.querySelectorAll<HTMLElement>("[data-hotkey-title]")) {
    const b = hotkeyBinding(el.dataset.hotkeyTitle!);
    if (b) el.title = b.description;
  }
}

// ─── HotkeyManager ──────────────────────────────────────────
type ActionHandler = () => void;

export class HotkeyManager {
  readonly bindings = HOTKEY_MAP;
  private _handlers = new Map<string, ActionHandler[]>();
  private _enabled = true;

  constructor() {
    document.addEventListener("keydown", (e) => this._dispatch(e));
  }

  on(action: string, handler: ActionHandler): void {
    const arr = this._handlers.get(action) ?? [];
    arr.push(handler);
    this._handlers.set(action, arr);
  }

  off(action: string, handler: ActionHandler): void {
    const arr = this._handlers.get(action);
    if (arr) this._handlers.set(action, arr.filter((h) => h !== handler));
  }

  setEnabled(enabled: boolean): void {
    this._enabled = enabled;
  }

  private _dispatch(e: KeyboardEvent): void {
    if (!this._enabled) return;
    // Only suppress hotkeys for text-entry fields; range/checkbox/radio/etc. must
    // not block global shortcuts.
    if (isTextEntryTarget(e.target)) return;

    for (const b of this.bindings) {
      if (b.dispatch === false) continue;
      if (!this._matches(e, b)) continue;
      e.preventDefault();
      const handlers = this._handlers.get(b.action);
      if (handlers) for (const h of handlers) h();
      return;
    }
  }

  private _matches(e: KeyboardEvent, b: HotkeyBinding): boolean {
    const wantMod = b.mod ?? false;
    const hasMod = IS_MAC ? e.metaKey : e.ctrlKey;
    if (wantMod !== hasMod) return false;

    const wantShift = b.shift ?? false;
    if (wantShift !== e.shiftKey) return false;

    const wantAlt = b.alt ?? false;
    if (wantAlt !== e.altKey) return false;

    return this._physicalKeyMatches(e, b.key);
  }

  /** ``e.key`` is wrong for Option/Alt+letter on macOS (e.g. ``ß`` instead of ``s``). */
  private _physicalKeyMatches(e: KeyboardEvent, wantKey: string): boolean {
    if (e.key === "Dead") return false;
    if (e.key.toLowerCase() === wantKey.toLowerCase()) return true;
    if (wantKey.length === 1 && /^[a-z]$/i.test(wantKey)) {
      return e.code === `Key${wantKey.toUpperCase()}`;
    }
    return false;
  }
}
