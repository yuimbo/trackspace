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
  { action: "toggle-preview",  key: "m",      label: "M",                      description: "Toggle hover preview" },
  { action: "fit-view",        key: "h",      label: "H",                      description: "Fit view to tracks" },
  { action: "enter-folder",    key: "i",      label: "I",                      description: "Enter hovered track's folder" },
  { action: "parent-folder",   key: "u",      label: "U",                      description: "Go up to parent folder" },
  { action: "select-all",      key: "a",      mod: true,  label: `${MOD_SYMBOL}A`,  description: "Select all tracks" },
  { action: "undo",            key: "z",      mod: true,  label: `${MOD_SYMBOL}Z`,  description: "Undo last action" },
  { action: "deselect",        key: "Escape",             label: "Esc",             description: "Deselect / cancel" },
  { action: "toggle-view",     key: "e",                  label: "E",              description: "Toggle Embedding Space" },
];

export type HotkeyAction = (typeof HOTKEY_MAP)[number]["action"];

export const MOUSE_HINTS: { label: string; description: string }[] = [
  { label: "Shift+drag",      description: "Lasso select (union)" },
  { label: "Ctrl+Shift+drag", description: "Lasso intersect (AND)" },
  { label: "Ctrl+drag",       description: "Box intersect (AND)" },
  { label: "Alt+drag",        description: "Pan viewport" },
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

    return e.key.toLowerCase() === b.key.toLowerCase();
  }
}
