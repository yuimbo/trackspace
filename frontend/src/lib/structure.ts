// ─── Track structure (intro / drop / break) ───────────────────
//
// The server's ``structure`` stage caches six section timestamps per track; the
// hover preview seeks to the drop (and cycles sections on **D**) while the tooltip
// renders them as chips. Both are pure derivations of a fetched ``TrackStructure``,
// so they live here to stay unit-testable and keep the controller free of formatting.

import type { TrackStructure } from "../model";

/** Preview position when no structure is cached yet (legacy behaviour). */
export const DEFAULT_PREVIEW_FRACTION = 0.4;

/**
 * Where the hover preview should seek.
 *
 * A search hit's best-matching segment centre (*matchSection*, a fraction of the
 * track duration) wins: that is the part that actually matched the query. Otherwise
 * the detected drop is the section worth auditioning; when it is absent (not yet
 * analysed, or no clear drop) fall back to the fixed fraction of the track.
 */
export function previewStartSeconds(
  structure: TrackStructure | undefined,
  duration: number,
  matchSection?: number | null,
): number {
  if (
    matchSection != null &&
    Number.isFinite(matchSection) &&
    duration > 0
  ) {
    return Math.min(
      Math.max(0, matchSection * duration),
      Math.max(0, duration - 1),
    );
  }
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

/** A section the tooltip renders as a chip and the preview can cycle to. */
export type StructureSection = "intro" | "drop" | "break";

export interface StructureChip {
  section: StructureSection;
  /** Chip text, e.g. ``drop 1:04`` or ``intro 0:12–0:32``. */
  text: string;
  /** Seek target in seconds when the preview cycles to this chip. */
  seconds: number;
}

/**
 * Seek target for a section, or ``null`` when the track has no such section.
 * The intro starts where the audio does (after any lead-in silence).
 */
export function sectionStartSeconds(
  structure: TrackStructure | undefined,
  section: StructureSection,
): number | null {
  if (!structure) return null;
  if (section === "intro") {
    return structure.intro_end != null && structure.intro_end > 0
      ? (structure.intro_start ?? 0)
      : null;
  }
  if (section === "drop") return structure.peak_start ?? null;
  return structure.break_start ?? null;
}

/**
 * Tooltip structure chips (intro, drop, break) in track order, omitting unknown
 * sections. Empty when nothing is cached yet.
 */
export function structureChips(
  structure: TrackStructure | undefined,
): StructureChip[] {
  const chips: StructureChip[] = [];
  const intro = sectionStartSeconds(structure, "intro");
  if (structure && intro != null) {
    const start = structure.intro_start ?? 0;
    const end = formatTimestamp(structure.intro_end);
    chips.push({
      section: "intro",
      seconds: intro,
      text: start > 0 ? `intro ${formatTimestamp(start)}–${end}` : `intro ${end}`,
    });
  }
  const drop = sectionStartSeconds(structure, "drop");
  if (drop != null) {
    chips.push({
      section: "drop",
      seconds: drop,
      text: `drop ${formatTimestamp(drop)}`,
    });
  }
  const brk = sectionStartSeconds(structure, "break");
  if (brk != null) {
    chips.push({
      section: "break",
      seconds: brk,
      text: `break ${formatTimestamp(brk)}`,
    });
  }
  return chips;
}
