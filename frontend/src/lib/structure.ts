// ─── Track structure (intro / drop / break) ───────────────────
//
// The server's ``structure`` stage caches five section timestamps per track; the
// hover preview seeks to the drop and the tooltip renders a one-line summary. Both
// are pure derivations of a fetched ``TrackStructure``, so they live here to stay
// unit-testable and keep the controller free of formatting.

import type { TrackStructure } from "../model";

/** Preview position when no structure is cached yet (legacy behaviour). */
export const DEFAULT_PREVIEW_FRACTION = 0.4;

/**
 * Where the hover preview should seek.
 *
 * The detected drop is the section worth auditioning; when it is absent (not yet
 * analysed, or no clear drop) fall back to the fixed fraction of the track.
 */
export function previewStartSeconds(
  structure: TrackStructure | undefined,
  duration: number,
): number {
  const peak = structure?.peak_start;
  if (peak != null && Number.isFinite(peak) && peak >= 0 && duration > 0) {
    // Keep a little audio after the seek so a seek near the end still plays.
    return Math.min(peak, Math.max(0, duration - 1));
  }
  return duration * DEFAULT_PREVIEW_FRACTION;
}

/** ``m:ss`` for a section timestamp; unknown or invalid renders as an em dash. */
export function formatTimestamp(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * One-line tooltip summary of the known sections, or ``null`` when nothing is
 * cached yet. The genre-spanning flag rides the same line.
 */
export function structureLine(
  structure: TrackStructure | undefined,
): string | null {
  if (!structure) return null;
  const parts: string[] = [];
  if (structure.intro_end != null && structure.intro_end > 0)
    parts.push(`intro ${formatTimestamp(structure.intro_end)}`);
  if (structure.peak_start != null)
    parts.push(`drop ${formatTimestamp(structure.peak_start)}`);
  if (structure.break_start != null)
    parts.push(`break ${formatTimestamp(structure.break_start)}`);
  if (structure.genre_spanning) parts.push("genre-spanning");
  return parts.length ? parts.join(" · ") : null;
}
