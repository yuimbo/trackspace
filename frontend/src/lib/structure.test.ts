import { describe, expect, it } from "vitest";
import type { TrackStructure } from "../model";
import {
  DEFAULT_PREVIEW_FRACTION,
  formatTimestamp,
  previewStartSeconds,
  sectionStartSeconds,
  structureChips,
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

  it("falls back to a fixed fraction without structure", () => {
    expect(previewStartSeconds(undefined, 200)).toBe(200 * DEFAULT_PREVIEW_FRACTION);
    expect(previewStartSeconds(structure(), 200)).toBe(200 * DEFAULT_PREVIEW_FRACTION);
  });

  it("never seeks past the end of the track", () => {
    expect(previewStartSeconds(structure({ peak_start: 199.5 }), 200)).toBe(199);
  });

  it("ignores a negative or non-finite drop", () => {
    expect(previewStartSeconds(structure({ peak_start: -3 }), 100)).toBe(
      100 * DEFAULT_PREVIEW_FRACTION,
    );
    expect(previewStartSeconds(structure({ peak_start: NaN }), 100)).toBe(
      100 * DEFAULT_PREVIEW_FRACTION,
    );
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

describe("structureChips", () => {
  it("returns no chips when nothing is cached", () => {
    expect(structureChips(undefined)).toEqual([]);
    expect(structureChips(structure())).toEqual([]);
  });

  it("lists the known sections in track order", () => {
    const chips = structureChips(
      structure({ intro_end: 32, peak_start: 64, break_start: 192 }),
    );
    expect(chips.map((c) => c.text)).toEqual([
      "intro 0:32",
      "drop 1:04",
      "break 3:12",
    ]);
    expect(chips.map((c) => c.section)).toEqual(["intro", "drop", "break"]);
    expect(chips.map((c) => c.seconds)).toEqual([0, 64, 192]);
  });

  it("renders the intro span and seeks it to the detected audio start", () => {
    const [chip] = structureChips(
      structure({ intro_start: 12, intro_end: 32, peak_start: 64 }),
    );
    expect(chip.text).toBe("intro 0:12–0:32");
    expect(chip.seconds).toBe(12);
  });

  it("omits a zero intro (track starts loud)", () => {
    const chips = structureChips(structure({ intro_end: 0, peak_start: 10 }));
    expect(chips.map((c) => c.text)).toEqual(["drop 0:10"]);
  });
});
