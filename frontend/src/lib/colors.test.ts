import { afterEach, describe, expect, it } from "vitest";
import vectors from "./dir-color-vectors.json";
import {
  dirColor,
  dirTagColors,
  keyColor,
  parseColor,
  setDirAnchors,
  tempoColor,
} from "./colors";

afterEach(() => setDirAnchors(null));

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

/** Hue component (float) of an `hsl(h,s%,l%)` string. */
function hue(color: string): number {
  return parseFloat(color.slice(4));
}

describe("parseColor", () => {
  it("parses 6-digit hex", () => {
    expect(parseColor("#8e9bff")).toEqual([142 / 255, 155 / 255, 1]);
  });

  it("parses 3-digit hex", () => {
    expect(parseColor("#fff")).toEqual([1, 1, 1]);
  });

  it("parses rgb()", () => {
    expect(parseColor("rgb(142,155,255)")).toEqual([142 / 255, 155 / 255, 1]);
  });

  it("parses hsl() into linear rgb", () => {
    const [r, g, b] = parseColor("hsl(0,100%,50%)");
    expect(r).toBeCloseTo(1);
    expect(g).toBeCloseTo(0);
    expect(b).toBeCloseTo(0);
  });
});

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

  it("rebases the hue family on a semantic anchor", () => {
    setDirAnchors({
      "abc12345/GENRES/Rock": "abc12345/GENRES/Rock",
      "abc12345/GENRES/Rock/70s": "abc12345/GENRES/Rock",
      "abc12345/GENRES/Jazz": "abc12345/GENRES/Jazz",
    });
    const rock = dirColor("abc12345/GENRES/Rock");
    const rock70s = dirColor("abc12345/GENRES/Rock/70s");
    const jazz = dirColor("abc12345/GENRES/Jazz");
    // Rock and Jazz start their own families; 70s tints Rock.
    expect(rock).not.toBe(jazz);
    expect(rock70s.startsWith(`${rock.slice(0, rock.indexOf(","))},`)).toBe(true);
  });

  it("falls back to the structural rule when a folder has no anchor", () => {
    setDirAnchors({ "abc12345/GENRES/Jazz": "abc12345/GENRES/Jazz" });
    expect(dirColor("abc12345/GENRES/Rock")).toBe(dirColor("abc12345/GENRES/Rock"));
  });
});

describe("dirTagColors", () => {
  it("is solid for roots and top-level folders", () => {
    expect(dirTagColors("abc12345")).toEqual([dirColor("abc12345")]);
    expect(dirTagColors("abc12345/GENRES")).toEqual([
      dirColor("abc12345/GENRES"),
    ]);
  });

  it("runs parent→own for nested folders", () => {
    expect(dirTagColors("abc12345/GENRES/Rock")).toEqual([
      dirColor("abc12345/GENRES"),
      dirColor("abc12345/GENRES/Rock"),
    ]);
  });
});

describe("feature-lens colors", () => {
  it("gives one octave the same tempo hue, and higher BPM a brighter shade", () => {
    const lo = (60 - 60) / (250 - 60);
    const hi = (120 - 60) / (250 - 60);
    expect(hue(tempoColor(lo))).toBeCloseTo(hue(tempoColor(hi)), 0);
    expect(lightness(tempoColor(hi))).toBeGreaterThan(lightness(tempoColor(lo)));
  });

  it("brightens a key as its Camelot radius grows: major over minor", () => {
    expect(lightness(keyColor(0, 1))).toBeGreaterThan(lightness(keyColor(0, 0)));
  });

  it("spaces adjacent keys on the circle of fifths 30° apart", () => {
    const a = hue(keyColor(0, 1));
    const b = hue(keyColor(1 / 12, 1));
    const diff = ((b - a) % 360 + 360) % 360;
    expect(diff).toBeCloseTo(30, 0);
  });
});
