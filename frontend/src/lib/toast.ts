// ─── Dir coloring ────────────────────────────────────────────
export function dirColor(folderPath: string): string {
  if (!folderPath) return "hsl(350,60%,55%)";
  let h = 0;
  for (let i = 0; i < folderPath.length; i++)
    h = (h * 31 + folderPath.charCodeAt(i)) & 0x3ffff;
  return `hsl(${h % 360},65%,60%)`;
}

// ─── Cluster coloring ────────────────────────────────────────
/** Golden-angle hue for cluster *id*.
 *
 * Successive ids land far apart on the colour wheel, so adjacent clusters
 * stay visually distinct however many there are — unlike hashing, which
 * happily gives two neighbouring clusters near-identical hues. `-1` means
 * "unclustered" and renders grey.
 */
export function clusterColor(id: number): string {
  if (id < 0) return "hsl(0,0%,45%)";
  const hue = (id * 137.508) % 360;
  // Alternate lightness so even a wrap-around hue collision stays separable.
  const light = id % 2 === 0 ? 62 : 52;
  return `hsl(${hue.toFixed(1)},68%,${light}%)`;
}

// ─── Toasts ──────────────────────────────────────────────────
let _$toasts: HTMLElement | null = null;
export function toast(
  msg: string,
  type: "info" | "ok" | "error" | "warn" = "info",
): void {
  _$toasts ??= document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  _$toasts!.appendChild(el);
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
    const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    _$loadBarFill.style.width = `${pct}%`;
  }
}

/** Hide the overlay bar (a determinate load finished, or none is in flight). */
export function clearLoadBar(): void {
  _$loadBar ??= document.getElementById("loading-bar");
  if (_$loadBar) _$loadBar.classList.add("hidden");
}
