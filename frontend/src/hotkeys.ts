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
  /** Display label for `<kbd>` badges and the F1 list. */
  label: string;
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
  { action: "focus-search",   key: "/",      mod: false, shift: false, alt: false, label: "/",                      description: "Focus semantic search" },
  { action: "focus-search",   key: "f",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}F`,         description: "Focus semantic search" },
  { action: "toggle-preview",  key: "m",      mod: false, shift: false, alt: false, label: "M",                      description: "Toggle hover preview (hold + scroll adjusts its volume)", dispatch: false },
  { action: "fit-view",        key: "h",      mod: false, shift: false, alt: false, label: "H",                      description: "Fit view to the hovered element (cluster/folder/key/BPM) or track's folder (80% radius), else full view" },
  { action: "focus-folder",    key: "f",      mod: false, shift: false, alt: false, label: "F",                      description: "Select all tracks of the hovered element or the track's folder (and subfolders)" },
  { action: "isolate-selection", key: "s",  mod: false, shift: false, alt: false, label: "s",                      description: "Isolate the hovered bloom class (folder subtree for folder colour), else the hovered element or selection; again when isolated: show all" },
  { action: "exclude-selection", key: "s",  mod: false, shift: true, alt: false, label: `Shift+s`,                description: "Exclude the hovered element or selection; repeat on a fully excluded element: restore it" },
  { action: "show-all-tracks",   key: "s",  mod: false, shift: false, alt: true,  label: "Alt+s",                 description: "Show all tracks again (clear exclusions)" },
  { action: "cycle-preview-section", key: "w", mod: false, shift: false, alt: false, label: "W",                  description: "Cycle hover preview between drop, break and intro" },
  { action: "cycle-preview-section-prev", key: "w", mod: false, shift: true, alt: false, label: "Shift+W",          description: "Cycle hover preview backward" },
  { action: "cycle-selection-bloom", key: "d", mod: false, shift: false, alt: false, label: "D",                    description: "Move the selection-bloom cursor across the colour tabs (Shift+D backward; hold + scroll adjusts bloom range)", dispatch: false },
  { action: "scope-gesture",   key: "x", code: "KeyX", mod: false, shift: false, alt: false, label: "X",            description: "Scope: drag a box / scroll a circle at the cursor; near an open scope cycle its lens (Shift+X backward); far outside close it", dispatch: false },
  { action: "toggle-external", key: "v", mod: false, shift: false, alt: false, label: "V",                    description: "Show/hide external-root tracks" },
  { action: "toggle-external-refit", key: "r", mod: false, shift: false, alt: false, label: "R",               description: "Refit the layout with external tracks included (while they are shown)" },
  { action: "select-all",      key: "a",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}A`,         description: "Select all tracks" },
  { action: "undo",            key: "z",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}Z`,         description: "Undo last action" },
  { action: "deselect",        key: "Escape", mod: false, shift: false, alt: false, label: "Esc",                    description: "Close the search popup / inspect card, else clear the newest view modifier, else deselect" },
  { action: "toggle-view",     key: "e",      mod: false, shift: false, alt: false, label: "E",                      description: "Cycle lenses forward (tags → combined → per-source maps → feature wheels)" },
  { action: "toggle-view-prev", key: "e",      mod: false, shift: true,  alt: false, label: "Shift+E",              description: "Cycle lenses backward" },
  { action: "toggle-clusters", key: "c",      mod: false, shift: false, alt: false, label: "C",                      description: "Cycle dot colour forward: folder → predicted folder → cluster → tempo → key" },
  { action: "toggle-clusters-prev", key: "c", mod: false, shift: true,  alt: false, label: "Shift+C",              description: "Cycle dot colour backward" },
  { action: "pin-inspect",      key: "Alt",    mod: false, shift: false, alt: false, label: "Alt",                    description: "Pin the inspect card for the hovered track (anti-track, nearest, solo)", dispatch: false },
  { action: "toggle-shortcuts", key: "F1",     mod: false, shift: false, alt: false, label: "F1",                     description: "Show this shortcut list", dispatch: false },
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

export const MOUSE_HINTS: { label: string; description: string }[] = [
  { label: "Shift+drag",      description: "Lasso select (union)" },
  { label: "Ctrl+Shift+drag", description: "Lasso intersect (AND)" },
  { label: "Ctrl+drag",       description: "Box intersect (AND)" },
  { label: "Alt+drag",        description: "Pan viewport" },
  { label: "Click folder",    description: "Select all tracks in this folder (and subfolders)" },
  { label: "⌘/Ctrl+click",    description: "Add this folder’s tracks to the selection" },
  { label: "Shift+click",     description: "Add tracks from each folder between anchor and this row (set anchor with a normal click)" },
  { label: "Click away", description: "Close the active scope" },
  { label: "Alt+click folder", description: "Exclude folder from the view" },
  { label: `Hover cluster + ${hotkeyLabel("isolate-selection")} / ${hotkeyLabel("fit-view")}`, description: "Isolate to the cluster / fit the view to it" },
  { label: "Drag→folder",     description: "Move files to folder" },
  { label: "Dbl-click tag",   description: "Rename tag" },
];

export function renderShortcutList(container: HTMLElement): void {
  container.innerHTML = "";
  for (const b of HOTKEY_MAP) {
    const row = document.createElement("div");
    row.className = "shortcut-row";
    row.innerHTML = `<kbd>${b.label}</kbd><span>${b.description}</span>`;
    container.appendChild(row);
  }
  for (const h of MOUSE_HINTS) {
    const row = document.createElement("div");
    row.className = "shortcut-row";
    row.innerHTML = `<kbd>${h.label}</kbd><span>${h.description}</span>`;
    container.appendChild(row);
  }
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
