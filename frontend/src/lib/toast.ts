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
