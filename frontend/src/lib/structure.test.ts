import { describe, expect, it } from "vitest";
import type { TrackStructure } from "../model";
import {
  formatTimestamp,
  previewStartSeconds,
  sectionStartSeconds,
  structureMarkers,
} from "./structure";

function structure(over: Partial<TrackStructure> = {}): TrackStructure {
  return {
    intro_start: null,
    intro_end: null,
    peak_start: null,
    peak_end: null,
    break_start: null,
    break_end: null,
    ...over,
  };
}

describe("previewStartSeconds", () => {
  it("seeks to the detected drop", () => {
    expect(previewStartSeconds(structure({ peak_start: 96 }), 300)).toBe(96);
  });

  it("starts at the top without structure", () => {
    expect(previewStartSeconds(undefined, 200)).toBe(0);
    expect(previewStartSeconds(structure(), 200)).toBe(0);
  });

  it("never seeks past the end of the track", () => {
    expect(previewStartSeconds(structure({ peak_start: 199.5 }), 200)).toBe(199);
  });

  it("ignores a negative or non-finite drop", () => {
    expect(previewStartSeconds(structure({ peak_start: -3 }), 100)).toBe(0);
    expect(previewStartSeconds(structure({ peak_start: NaN }), 100)).toBe(0);
  });

  it("prefers a search hit's matching section over the drop", () => {
    expect(previewStartSeconds(structure({ peak_start: 96 }), 200, 0.8)).toBe(160);
  });

  it("clamps a matching section near the end", () => {
    expect(previewStartSeconds(undefined, 100, 1)).toBe(99);
    expect(previewStartSeconds(undefined, 100, NaN)).toBe(0);
  });
});

describe("formatTimestamp", () => {
  it("renders m:ss and pads seconds", () => {
    expect(formatTimestamp(65)).toBe("1:05");
    expect(formatTimestamp(9)).toBe("0:09");
    expect(formatTimestamp(600)).toBe("10:00");
  });

  it("renders an em dash for unknown or invalid values", () => {
    expect(formatTimestamp(null)).toBe("—");
    expect(formatTimestamp(undefined)).toBe("—");
    expect(formatTimestamp(-1)).toBe("—");
    expect(formatTimestamp(NaN)).toBe("—");
  });
});

describe("sectionStartSeconds", () => {
  it("returns null for a section the track does not have", () => {
    expect(sectionStartSeconds(undefined, "drop")).toBeNull();
    expect(sectionStartSeconds(structure(), "drop")).toBeNull();
    expect(sectionStartSeconds(structure(), "break")).toBeNull();
    expect(sectionStartSeconds(structure({ peak_start: 64 }), "break")).toBeNull();
  });

  it("seeks the drop to the peak start and the break to its start", () => {
    const s = structure({ peak_start: 64, break_start: 192 });
    expect(sectionStartSeconds(s, "drop")).toBe(64);
    expect(sectionStartSeconds(s, "break")).toBe(192);
  });

  it("seeks the intro to the detected audio start, falling back to 0", () => {
    expect(sectionStartSeconds(structure({ intro_start: 8, intro_end: 32 }), "intro")).toBe(8);
    expect(sectionStartSeconds(structure({ intro_end: 32 }), "intro")).toBe(0);
  });

  it("treats a zero-length intro as absent", () => {
    expect(sectionStartSeconds(structure({ intro_end: 0, peak_start: 10 }), "intro")).toBeNull();
  });
});

describe("structureMarkers", () => {
  it("returns no markers without a duration to divide by", () => {
    const s = structure({ peak_start: 64 });
    expect(structureMarkers(s, 0)).toEqual([]);
    expect(structureMarkers(s, NaN)).toEqual([]);
  });

  it("returns no markers when nothing is cached", () => {
    expect(structureMarkers(undefined, 200)).toEqual([]);
    expect(structureMarkers(structure(), 200)).toEqual([]);
  });

  it("places each section as a fraction of the duration", () => {
    const markers = structureMarkers(
      structure({ intro_start: 12, intro_end: 32, peak_start: 64, break_start: 192 }),
      256,
    );
    expect(markers.map((m) => m.section)).toEqual(["intro", "drop", "break"]);
    expect(markers.map((m) => m.fraction)).toEqual([12 / 256, 64 / 256, 192 / 256]);
    expect(markers.map((m) => m.text)).toEqual([
      "intro 0:12–0:32",
      "drop 1:04",
      "break 3:12",
    ]);
  });

  it("labels an intro with no lead-in by its end", () => {
    const [marker] = structureMarkers(structure({ intro_end: 32, peak_start: 64 }), 256);
    expect(marker.text).toBe("intro 0:32");
    expect(marker.fraction).toBe(0);
  });

  it("omits a zero intro (track starts loud)", () => {
    const markers = structureMarkers(structure({ intro_end: 0, peak_start: 10 }), 256);
    expect(markers.map((m) => m.section)).toEqual(["drop"]);
  });

  it("clamps a section timestamp past the end onto the timeline", () => {
    const [marker] = structureMarkers(structure({ peak_start: 300 }), 200);
    expect(marker.fraction).toBe(1);
  });
});
