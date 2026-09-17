import { describe, expect, it } from "vitest";
import vectors from "./dir-color-vectors.json";
import { dirColor } from "./colors";

/** The pre-family algorithm, kept to prove depth <= 1 is byte-identical. */
function legacyDirColor(folderPath: string): string {
  if (!folderPath) return "hsl(350,60%,55%)";
  let h = 0;
  for (let i = 0; i < folderPath.length; i++)
    h = (h * 31 + folderPath.charCodeAt(i)) & 0x3ffff;
  return `hsl(${h % 360},65%,60%)`;
}

/** Lightness component (integer) of an `hsl(h,s%,l%)` string. */
function lightness(color: string): number {
  return parseInt(color.split(",")[2], 10);
}

describe("dirColor", () => {
  it("matches the shared golden vectors (keep in sync with dir_colors.py)", () => {
    for (const [path, expected] of Object.entries(vectors)) {
      expect(dirColor(path), path).toBe(expected);
    }
  });

  it("keeps roots and top-level folders byte-identical to the old hash", () => {
    for (const path of [
      "",
      "abc12345",
      "abc12345/GENRES",
      "def67890",
      "def67890/to_sort",
    ]) {
      expect(dirColor(path)).toBe(legacyDirColor(path));
    }
  });

  it("gives a top-level family one shared hue", () => {
    const family = [
      "abc12345/GENRES/Rock",
      "abc12345/GENRES/Rock/70s",
      "abc12345/GENRES/Rock/70s/Live",
      "abc12345/GENRES/Jazz",
    ];
    for (const path of family) {
      expect(dirColor(path).startsWith("hsl(286,")).toBe(true);
    }
  });

  it("halves the lightness deviation per nesting level", () => {
    const devs = [
      "abc12345/GENRES/Rock",
      "abc12345/GENRES/Rock/70s",
      "abc12345/GENRES/Rock/70s/Live",
    ].map((p) => Math.abs(lightness(dirColor(p)) - 60));
    expect(devs[0]).toBeLessThanOrEqual(30);
    expect(devs[1]).toBeLessThanOrEqual(15);
    expect(devs[2]).toBeLessThanOrEqual(8);
  });

  it("gives sibling subfolders distinct tints", () => {
    expect(dirColor("abc12345/GENRES/Rock")).not.toBe(
      dirColor("abc12345/GENRES/Jazz"),
    );
  });

  it("is deterministic", () => {
    expect(dirColor("abc12345/GENRES/Rock/70s")).toBe(
      dirColor("abc12345/GENRES/Rock/70s"),
    );
  });
});
