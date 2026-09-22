// ─── Shared helpers ─────────────────────────────────────────
//
// Pure, dependency-free utilities used across coordinators, views and libs, so a
// concept like "clamp to [0,1]" has one definition instead of one per file.

/** Clamp ``v`` to ``[lo, hi]`` (default ``[0, 1]``). */
export function clamp(v: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Clamp ``v`` to ``[0, 1]``. */
export function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

/** Escape a string for interpolation into HTML text or an attribute. */
export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** ``window.confirm`` behind a seam, so callers and tests share one entry point. */
export function confirmAction(message: string): boolean {
  return window.confirm(message);
}

/**
 * Step ``current`` through ``list`` by ``dir``, wrapping at either end.
 *
 * ``fallback`` is returned when ``current`` is not in the list (default: the first
 * entry), so an unknown id still yields a valid cycle position.
 */
export function cycle<T>(list: readonly T[], current: T, dir: 1 | -1 = 1, fallback?: T): T {
  if (list.length === 0) throw new Error("cycle: empty list");
  const idx = list.indexOf(current);
  if (idx === -1) return fallback ?? list[0];
  const n = list.length;
  return list[(idx + dir + n) % n];
}
