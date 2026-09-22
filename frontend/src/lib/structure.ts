// ─── Track structure (intro / drop / break) ───────────────────
//
// The server's ``structure`` stage caches six section timestamps per track; the
// hover preview seeks to the drop (and cycles sections on **W**) while the tooltip
// renders them as markers on the preview seekbar. Both are pure derivations of a
// fetched ``TrackStructure``, so they live here to stay unit-testable and keep the
// controller free of formatting.

import type { TrackStructure } from "../model";

/**
 * Where the hover preview should seek.
 *
 * A search hit's best-matching segment centre (*matchSection*, a fraction of the
 * track duration) wins: that is the part that actually matched the query. Otherwise
 * the detected drop is the section worth auditioning; when it is absent (not yet
 * analysed, or no clear drop) the preview starts at the top of the track.
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
  return 0;
}

/** ``m:ss`` for a section timestamp; unknown or invalid renders as an em dash. */
export function formatTimestamp(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** A section the preview can cycle to and the seekbar marks. */
export type StructureSection = "intro" | "drop" | "break";

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

/** A section placed on the preview seekbar as a fraction of the track duration. */
export interface StructureMarker {
  section: StructureSection;
  /** 0–1 position along the track. */
  fraction: number;
  /** Marker label, e.g. ``drop 1:04`` (shown only for the auditioned section). */
  text: string;
}

/**
 * The structure sections as seekbar markers, positioned by *duration* and in track
 * order, omitting unknown sections. Empty when the duration is unknown (metadata not
 * loaded yet) or nothing is cached — the timeline cannot place a section without a
 * track length to divide by.
 */
export function structureMarkers(
  structure: TrackStructure | undefined,
  duration: number,
): StructureMarker[] {
  if (!(duration > 0)) return [];
  const markers: StructureMarker[] = [];
  const place = (section: StructureSection, seconds: number, text: string) => {
    markers.push({
      section,
      text,
      fraction: Math.max(0, Math.min(1, seconds / duration)),
    });
  };

  const intro = sectionStartSeconds(structure, "intro");
  if (structure && intro != null) {
    const start = structure.intro_start ?? 0;
    const end = formatTimestamp(structure.intro_end);
    place(
      "intro",
      intro,
      start > 0 ? `intro ${formatTimestamp(start)}–${end}` : `intro ${end}`,
    );
  }
  const drop = sectionStartSeconds(structure, "drop");
  if (drop != null) place("drop", drop, `drop ${formatTimestamp(drop)}`);
  const brk = sectionStartSeconds(structure, "break");
  if (brk != null) place("break", brk, `break ${formatTimestamp(brk)}`);
  return markers;
}
