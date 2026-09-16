import { describe, expect, it } from "vitest";
import type { TrackStructure } from "../model";
import {
  DEFAULT_PREVIEW_FRACTION,
  formatTimestamp,
  previewStartSeconds,
  structureLine,
} from "./structure";

function structure(over: Partial<TrackStructure> = {}): TrackStructure {
  return {
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

describe("structureLine", () => {
  it("returns null when nothing is cached", () => {
    expect(structureLine(undefined)).toBeNull();
    expect(structureLine(structure())).toBeNull();
  });

  it("lists the known sections in order", () => {
    const line = structureLine(
      structure({ intro_end: 32, peak_start: 64, break_start: 192 }),
    );
    expect(line).toBe("intro 0:32 · drop 1:04 · break 3:12");
  });

  it("omits a zero intro (track starts loud)", () => {
    expect(structureLine(structure({ intro_end: 0, peak_start: 10 }))).toBe(
      "drop 0:10",
    );
  });

  it("appends the genre-spanning flag", () => {
    expect(structureLine(structure({ peak_start: 10, genre_spanning: true }))).toBe(
      "drop 0:10 · genre-spanning",
    );
  });
});
