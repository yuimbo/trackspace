// ─── Dir coloring ────────────────────────────────────────────
export function dirColor(folderPath: string): string {
  if (!folderPath) return "hsl(350,60%,55%)";
  let h = 0;
  for (let i = 0; i < folderPath.length; i++)
    h = (h * 31 + folderPath.charCodeAt(i)) & 0x3ffff;
  return `hsl(${h % 360},65%,60%)`;
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
