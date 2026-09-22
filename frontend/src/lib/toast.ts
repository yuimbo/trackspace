// ─── Toasts ──────────────────────────────────────────────────
import { clamp01 } from "./util";

let _$toasts: HTMLElement | null = null;
export function toast(
  msg: string,
  type: "info" | "ok" | "error" | "warn" = "info",
  /** Optional viewport point to anchor the toast near (e.g. a failed drop). */
  at?: { x: number; y: number },
): void {
  _$toasts ??= document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  if (at) {
    // Anchor it at the point the user acted, rather than the global stack.
    el.classList.add("toast-anchored");
    el.style.left = `${Math.max(8, Math.min(at.x, window.innerWidth - 8))}px`;
    el.style.top = `${Math.max(8, Math.min(at.y, window.innerHeight - 8))}px`;
    document.body.appendChild(el);
  } else {
    _$toasts!.appendChild(el);
  }
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 400);
  }, 3200);
}

// ─── Loading overlay ─────────────────────────────────────────
let _lc = 0;
let _$loading: HTMLElement | null = null;

export function showLoad(): void {
  _$loading ??= document.getElementById("loading-overlay");
  _lc++;
  _$loading!.classList.remove("hidden");
}

export function hideLoad(): void {
  _$loading ??= document.getElementById("loading-overlay");
  if (--_lc <= 0) {
    _lc = 0;
    _$loading!.classList.add("hidden");
  }
}

// ─── Loading progress bar ─────────────────────────────────────
// The overlay is used by long, opaque loads (library scan, layout). The text
// alone looked stalled; a bar makes "still moving" visible even when the count is
// not yet known (indeterminate).
let _$loadBar: HTMLElement | null = null;
let _$loadBarFill: HTMLElement | null = null;

/** Show the overlay bar at *fraction* (0..1); ``null`` = indeterminate. */
export function setLoadBar(fraction: number | null): void {
  _$loadBar ??= document.getElementById("loading-bar");
  _$loadBarFill ??= document.getElementById("loading-bar-fill");
  if (!_$loadBar || !_$loadBarFill) return;
  _$loadBar.classList.remove("hidden");
  if (fraction == null || !Number.isFinite(fraction)) {
    _$loadBar.classList.add("indeterminate");
    _$loadBarFill.style.width = "";
  } else {
    _$loadBar.classList.remove("indeterminate");
    const pct = Math.round(clamp01(fraction) * 100);
    _$loadBarFill.style.width = `${pct}%`;
  }
}

/** Hide the overlay bar (a determinate load finished, or none is in flight). */
export function clearLoadBar(): void {
  _$loadBar ??= document.getElementById("loading-bar");
  if (_$loadBar) _$loadBar.classList.add("hidden");
}
