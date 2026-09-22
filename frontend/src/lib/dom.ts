// ─── Shared DOM seams ──────────────────────────────────────
//
// The recurring DOM shapes: an element builder, a keyboard badge, a centered
// overlay dialog with one dismissal wiring, a fixed-positioned popover, and a tab
// strip. Keeping them here means "Esc closes a modal" and "the popover stays in
// the viewport" have one implementation instead of one per surface.

/** Build an element with an optional class and text. */
export function el(tag: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** A `<kbd>` hotkey badge. */
export function createKbd(label: string, className = "hotkey"): HTMLElement {
  return el("kbd", className, label);
}

/** A viewport-relative rectangle to anchor a popover against. */
export interface AnchorRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Place a fixed-position element against *anchor*, clamped inside the viewport.
 *
 * ``side: "right"`` puts it to the anchor's right (folder-row actions);
 * ``side: "above"`` centres it over the anchor (the status detail panel).
 */
export function placeAt(
  node: HTMLElement,
  anchor: AnchorRect,
  opts: { side?: "right" | "above"; gap?: number; margin?: number } = {},
): void {
  const margin = opts.margin ?? 6;
  const gap = opts.gap ?? 6;
  const w = node.offsetWidth;
  const h = node.offsetHeight;
  let left: number;
  let top: number;
  if (opts.side === "above") {
    left = anchor.left + (anchor.right - anchor.left) / 2 - w / 2;
    top = anchor.top - h - gap;
  } else {
    left = anchor.right + gap;
    top = anchor.top;
  }
  left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
  top = Math.max(margin, Math.min(top, window.innerHeight - h - margin));
  node.style.left = `${left}px`;
  node.style.top = `${top}px`;
}

/**
 * A centered overlay dialog. ``bind`` wires the backdrop click-self and a
 * document-capture Esc that stops propagation, so a modal always closes on Esc
 * and never leaks the key to the global handler (which would clear selection).
 */
export class Modal {
  constructor(private readonly _id: string) {}

  get el(): HTMLElement | null {
    return document.getElementById(this._id);
  }

  isOpen(): boolean {
    const node = this.el;
    return !!node && !node.classList.contains("hidden");
  }

  open(): void {
    this.el?.classList.remove("hidden");
  }

  close(): void {
    this.el?.classList.add("hidden");
  }

  /** Wire backdrop + Esc once; *onClose* runs for either. */
  bind(onClose: () => void): void {
    const node = this.el;
    if (!node) return;
    node.addEventListener("click", (e) => {
      if (e.target === node) onClose();
    });
    document.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Escape" && !node.classList.contains("hidden")) {
          e.stopImmediatePropagation();
          onClose();
        }
      },
      true,
    );
  }
}

export interface TabItem {
  id: string;
  label: string;
  /** Value for the dataset key (defaults to ``id``). */
  data?: string;
}

export interface TabStrip {
  el: HTMLElement;
  /** Highlight *id* and sync ``aria-selected`` across the tabs. */
  sync(id: string): void;
}

export interface TabStripOptions {
  /** Dataset key each button carries its value in (e.g. ``overlayTab``). */
  dataKey?: string;
  /** Extra class on each button, on top of the base ``tab``. */
  buttonClass?: string;
  onSelect?: (id: string) => void;
}

/**
 * Build a ``role=tablist`` strip of buttons. The base ``.tab`` class carries the
 * look; the container class scopes it (``.tt-tabs``, ``.overlay-tabs``).
 */
export function buildTabStrip(
  containerClass: string,
  items: readonly TabItem[],
  activeId: string,
  opts: TabStripOptions = {},
): TabStrip {
  const strip = el("div", containerClass);
  strip.setAttribute("role", "tablist");
  const buttons = new Map<string, HTMLButtonElement>();
  for (const item of items) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = opts.buttonClass ? `tab ${opts.buttonClass}` : "tab";
    if (opts.dataKey) btn.dataset[opts.dataKey] = item.data ?? item.id;
    btn.textContent = item.label;
    btn.setAttribute("role", "tab");
    if (opts.onSelect) btn.addEventListener("click", () => opts.onSelect!(item.id));
    strip.appendChild(btn);
    buttons.set(item.id, btn);
  }
  const sync = (id: string): void => {
    for (const [tabId, btn] of buttons) {
      const active = tabId === id;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    }
  };
  sync(activeId);
  return { el: strip, sync };
}
