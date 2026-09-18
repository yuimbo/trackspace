// ─── Scope lens picker ────────────────────────────────────────
//
// A scope is a circle drawn on the canvas whose tracks are re-laid-out by another
// lens. The gesture (X + drag) only draws the circle; choosing the lens is a
// deliberate second step, so a small menu appears at the release point. This module
// owns that menu's DOM so `CanvasInteractionController` stays gesture-only.
//
// Styles are inline rather than in `trackspace.css`: the menu is the only element
// of its kind, and inlining keeps this module self-contained.

import { BUILTIN_LENSES, type Lens } from "./lenses";

/** Lenses a scope can use: the tag lens has no positions to lay out. */
export function scopeLenses(): Lens[] {
  return BUILTIN_LENSES.filter((l) => l.kind !== "tags");
}

/**
 * Show the lens menu at *screen*, calling *onPick* with the chosen lens id.
 *
 * Returns a cleanup function; the menu also closes itself on a pick, Escape, or a
 * click outside. Cleanup is idempotent, so a caller may close it again safely.
 */
export function showScopeLensMenu(
  screen: { x: number; y: number },
  onPick: (lensId: string) => void,
): () => void {
  const menu = document.createElement("div");
  menu.className = "scope-lens-menu";
  Object.assign(menu.style, {
    position: "fixed",
    left: `${Math.min(screen.x, window.innerWidth - 160)}px`,
    top: `${Math.min(screen.y, window.innerHeight - 20 - 26 * scopeLenses().length)}px`,
    zIndex: "50",
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    padding: "4px",
    background: "#1c1c1c",
    border: "1px solid #555",
    borderRadius: "4px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
  });
  for (const lens of scopeLenses()) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = lens.label;
    Object.assign(btn.style, {
      background: "transparent",
      color: "#eee",
      border: "0",
      textAlign: "left",
      padding: "4px 10px",
      cursor: "pointer",
      font: "inherit",
    });
    btn.addEventListener("click", () => {
      cleanup();
      onPick(lens.id);
    });
    menu.appendChild(btn);
  }
  document.body.appendChild(menu);

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") cleanup();
  };
  const onDown = (e: MouseEvent) => {
    if (!menu.contains(e.target as Node)) cleanup();
  };
  let closed = false;
  function cleanup(): void {
    if (closed) return;
    closed = true;
    menu.remove();
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("mousedown", onDown, true);
  }
  document.addEventListener("keydown", onKey);
  document.addEventListener("mousedown", onDown, true);
  return cleanup;
}
