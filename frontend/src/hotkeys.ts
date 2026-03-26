export interface HotkeyBinding {
  action: string;
  key: string;
  mod?: boolean;
  shift?: boolean;
  alt?: boolean;
  label: string;
  description: string;
}

const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
const MOD_SYMBOL = IS_MAC ? "⌘" : "Ctrl+";

export const HOTKEY_MAP: HotkeyBinding[] = [
  { action: "toggle-preview",  key: "m",      mod: false, shift: false, alt: false, label: "M",                      description: "Toggle hover preview" },
  { action: "fit-view",        key: "h",      mod: false, shift: false, alt: false, label: "H",                      description: "Fit view (folder row hover: 80% radius)" },
  { action: "enter-folder",    key: "i",      mod: false, shift: false, alt: false, label: "I",                      description: "Enter hovered track's folder" },
  { action: "parent-folder",   key: "u",      mod: false, shift: false, alt: false, label: "U",                      description: "Go up to parent folder" },
  { action: "isolate-selection", key: "s",  mod: false, shift: false, alt: false, label: "s",                      description: "Isolate to hovered folder if any, else to selection; fully excluded folder: isolate to that subtree; again when isolated: show all" },
  { action: "exclude-selection", key: "s",  mod: false, shift: true, alt: false, label: `Shift+s`,                description: "Exclude hovered folder if any, else selection; repeat to narrow; on excluded folder: restore it" },
  { action: "show-all-tracks",   key: "s",  mod: false, shift: false, alt: true,  label: "Alt+s",                 description: "Show all tracks again (clear exclusions)" },
  { action: "select-all",      key: "a",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}A`,         description: "Select all tracks" },
  { action: "undo",            key: "z",      mod: true,  shift: false, alt: false, label: `${MOD_SYMBOL}Z`,         description: "Undo last action" },
  { action: "deselect",        key: "Escape", mod: false, shift: false, alt: false, label: "Esc",                    description: "Deselect / cancel" },
  { action: "toggle-view",     key: "e",      mod: false, shift: false, alt: false, label: "E",                      description: "Toggle Embedding Space" },
];

export type HotkeyAction = (typeof HOTKEY_MAP)[number]["action"];

export const MOUSE_HINTS: { label: string; description: string }[] = [
  { label: "Shift+drag",      description: "Lasso select (union)" },
  { label: "Ctrl+Shift+drag", description: "Lasso intersect (AND)" },
  { label: "Ctrl+drag",       description: "Box intersect (AND)" },
  { label: "Alt+drag",        description: "Pan viewport" },
  { label: "Click folder",    description: "Select all tracks in this folder (and subfolders)" },
  { label: "⌘/Ctrl+click",    description: "Multi-select folders in the sidebar view" },
  { label: "Shift+click",     description: "Folder range select (sidebar view)" },
  { label: "Alt+click folder", description: "Exclude folder from the view" },
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
    const el = e.target as HTMLElement;
    if (el.tagName === "TEXTAREA") return;
    if (el.tagName === "INPUT") {
      // Only suppress hotkeys for text-entry inputs; range/checkbox/radio/etc. should not block global shortcuts
      const type = (el as HTMLInputElement).type.toLowerCase();
      const isTextEntry = type === "" || type === "text" || type === "password" ||
        type === "email" || type === "search" || type === "url" ||
        type === "tel" || type === "number" || type === "date" ||
        type === "time" || type === "datetime-local" || type === "month" || type === "week";
      if (isTextEntry) return;
    }

    for (const b of this.bindings) {
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
