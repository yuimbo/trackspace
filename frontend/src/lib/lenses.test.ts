import { describe, expect, it } from "vitest";
import type { Track } from "../model";
import {
  BPM_HI,
  BPM_LO,
  availableLenses,
  bpmSpiralXY,
  camelotGuide,
  camelotPosition,
  camelotXY,
  getLens,
  lensWaitSources,
  lensWeights,
  nextLensId,
  prevLensId,
  scopeLenses,
  spiralGuide,
  tempoSpiralPosition,
  BUILTIN_LENSES,
  GENRE_FULL_LENS,
  GENRE_FULL_SOURCES,
  GENRE_FULL_WEIGHTS,
} from "./lenses";

function track(over: Partial<Track> = {}): Track {
  return {
    path: "a.mp3",
    filename: "a.mp3",
    folder: "",
    tags: {},
    artist: "",
    title: "",
    ...over,
  };
}

/** Cached feature row: key is a circle-of-fifths angle mapped to [0,1). */
function feats(key: number, mode: number, tempo: number) {
  return { key, mode, tempo, energy: 0.5, danceability: 0.5 };
}

const polar = (p: { x: number; y: number }) => {
  const dx = p.x - 0.5;
  const dy = p.y - 0.5;
  return { angle: Math.atan2(dy, dx), r: Math.hypot(dx, dy) * 2 };
};

const KEY_TURNS: Record<string, number> = {
  C: 0,
  G: 1,
  D: 2,
  A: 3,
  E: 4,
  B: 5,
  "F#": 6,
  F: 11,
};

const keyTurn = (k: string) => (KEY_TURNS[k] / 12 + 0.5) % 1;

describe("lens registry", () => {
  it("E cycles the built-ins and wraps", () => {
    expect(nextLensId("tags")).toBe("combined");
    expect(nextLensId("combined")).toBe("maest");
    expect(nextLensId("tempo")).toBe("tags");
  });

  it("Shift+E cycles backward and wraps", () => {
    expect(prevLensId("tags")).toBe("tempo");
    expect(prevLensId("combined")).toBe("tags");
    expect(prevLensId("tempo")).toBe("camelot");
  });

  it("cycles from a preset's base lens", () => {
    expect(nextLensId("preset:1", "clap")).toBe("rhythm");
    expect(prevLensId("preset:1", "clap")).toBe("supervised");
  });

  it("scope lenses exclude the tag lens", () => {
    const ids = scopeLenses().map((l) => l.id);
    expect(ids).not.toContain("tags");
    expect(ids).toContain("camelot");
    expect(ids).toContain("clap");
  });

  it("falls back to tags for an unknown id", () => {
    expect(getLens("bogus").id).toBe("tags");
  });

  it("per-source lenses zero the other blocks", () => {
    const lens = getLens("clap");
    const w = lensWeights(lens, {
      maest_logits: 0.45,
      maest: 0.25,
      rhythm: 0.15,
      clap: 0.1,
      supervised: 0.15,
      features: 0,
    });
    expect(w.clap).toBe(1);
    expect(w.maest_logits).toBe(0);
    expect(w.rhythm).toBe(0);
  });

  it("the combined lens keeps the user's weights", () => {
    const w = { maest_logits: 0.45, maest: 0.25 };
    expect(lensWeights(getLens("combined"), w)).toBe(w);
  });

  it("waits on the right analysis stages per lens", () => {
    expect(lensWaitSources(getLens("tags"))).toEqual([]);
    expect(lensWaitSources(getLens("clap"))).toEqual(["clap"]);
    expect(lensWaitSources(getLens("camelot"))).toEqual(["features"]);
    expect(lensWaitSources(getLens("combined"))).toContain("maest_logits");
  });

  it("genre_full lens is absent until the server enables it", () => {
    // With genre disabled, the legacy combined map stays as the fallback default;
    // the single-source MAEST / Folder-head probes are gone from the cycle.
    expect(availableLenses(false).map((l) => l.id)).not.toContain("genre_full");
    expect(availableLenses(false).map((l) => l.id)).toEqual([
      "tags",
      "combined",
      "clap",
      "rhythm",
      "camelot",
      "tempo",
    ]);
  });

  it("genre_full replaces the redundant map tabs when enabled, and the cycle reaches it", () => {
    // Combined / MAEST / Folder-head are gone; the genre map takes the map slot.
    const ids = availableLenses(true).map((l) => l.id);
    expect(ids).toEqual(["tags", "genre_full", "clap", "rhythm", "camelot", "tempo"]);
    // E from the last feature lens wraps into the genre map; back out on Shift+E.
    expect(nextLensId("tempo", undefined, availableLenses(true))).toBe("tags");
    expect(prevLensId("genre_full", undefined, availableLenses(true))).toBe("tags");
    expect(nextLensId("genre_full", undefined, availableLenses(true))).toBe("clap");
    // Cycle unchanged (no genre map) when the lens is unavailable.
    expect(nextLensId("tempo", undefined, availableLenses(false))).toBe("tags");
  });

  it("E from a hidden legacy lens resolves to the genre map, not tags", () => {
    // With genre enabled, a legacy map id that is no longer on the cycle (a saved
    // preset base, or a transient active id) steps from the genre map it maps to.
    const available = availableLenses(true);
    expect(nextLensId("maest", undefined, available)).toBe("clap");
    expect(nextLensId("supervised", undefined, available)).toBe("clap");
    expect(nextLensId("combined", undefined, available)).toBe("clap");
    // A preset's legacy base lens resolves the same way.
    expect(nextLensId("preset:1", "combined", available)).toBe("clap");
  });

  it("genre_full resolves to its own lens with the four genre analysis sources", () => {
    expect(getLens("genre_full")).toBe(GENRE_FULL_LENS);
    expect(GENRE_FULL_LENS.sources).toEqual([
      "maest_logits",
      "maest",
      "maest_layer7",
      "mert_l19",
    ]);
  });

  it("exposes the seven tunable genre sources and the canonical defaults", () => {
    // The weight UI and the recipe ``w=`` override iterate the 7-key list; the
    // analysis ``sources`` above stay the four blocks the map actually requires.
    expect(GENRE_FULL_SOURCES).toEqual([
      "maest_logits",
      "maest",
      "maest_layer7",
      "mert_l19",
      "rhythm",
      "clap",
      "supervised",
    ]);
    expect(GENRE_FULL_WEIGHTS).toMatchObject({
      maest_logits: 0.4,
      maest: 0.2,
      maest_layer7: 0.25,
      mert_l19: 0.15,
      rhythm: 0,
      clap: 0,
      supervised: 0,
    });
  });

  it("scope lenses include the genre map when enabled and the combined fallback otherwise", () => {
    expect(scopeLenses(true).map((l) => l.id)).toContain("genre_full");
    expect(scopeLenses(true).map((l) => l.id)).not.toContain("tags");
    expect(scopeLenses(false).map((l) => l.id)).toContain("combined");
    expect(scopeLenses(true).map((l) => l.id)).toContain("camelot");
  });
});

describe("camelot geometry", () => {
  it("puts C major on the +x axis of the outer ring", () => {
    const p = camelotXY(keyTurn("C"), 1);
    expect(p.x).toBeCloseTo(0.91, 6);
    expect(p.y).toBeCloseTo(0.5, 6);
  });

  it("aligns relative major/minor on the same ray", () => {
    const pairs: [string, string][] = [
      ["C", "A"],
      ["G", "E"],
      ["D", "B"],
      ["F", "D"],
    ];
    for (const [major, minor] of pairs) {
      const a = camelotXY(keyTurn(major), 1);
      const b = camelotXY(keyTurn(minor), 0);
      expect(polar(b).angle).toBeCloseTo(polar(a).angle, 6);
      expect(polar(b).r).toBeLessThan(polar(a).r);
    }
  });

  it("keeps fifth-neighbours closer than the tritone", () => {
    const c = camelotXY(keyTurn("C"), 1);
    const g = camelotXY(keyTurn("G"), 1);
    const fs = camelotXY(keyTurn("F#"), 1);
    const d = (a: { x: number; y: number }, b: { x: number; y: number }) =>
      Math.hypot(a.x - b.x, a.y - b.y);
    expect(d(c, g)).toBeLessThan(d(c, fs));
  });
});

describe("tempo spiral geometry", () => {
  it("puts half/double tempos on the same ray", () => {
    const norm = (bpm: number) => (bpm - BPM_LO) / (BPM_HI - BPM_LO);
    const a = bpmSpiralXY(norm(60));
    const b = bpmSpiralXY(norm(120));
    const c = bpmSpiralXY(norm(240));
    expect(polar(b).angle).toBeCloseTo(polar(a).angle, 6);
    expect(polar(c).angle).toBeCloseTo(polar(a).angle, 6);
    expect(polar(a).r).toBeLessThan(polar(b).r);
    expect(polar(b).r).toBeLessThan(polar(c).r);
  });

  it("clamps outside the tempo window", () => {
    expect(polar(bpmSpiralXY(-0.5)).r).toBeCloseTo(0.18, 6);
    expect(polar(bpmSpiralXY(1.5)).r).toBeCloseTo(0.86, 6);
  });
});

describe("canvas guides", () => {
  it("labels the twelve Camelot key positions on the outer ring", () => {
    const guide = camelotGuide();
    expect(guide.map((g) => g.label)).toContain("C");
    expect(guide).toHaveLength(12);
    for (const g of guide) {
      expect(Math.hypot(g.x - 0.5, g.y - 0.5)).toBeCloseTo(
        0.5 * 0.82,
        6,
      );
    }
  });

  it("samples a spiral arm inside the unit square", () => {
    const pts = spiralGuide();
    expect(pts.length).toBeGreaterThan(1);
    for (const p of pts) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(1);
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(1);
    }
  });
});

describe("deterministic positions", () => {
  it("returns null without cached features", () => {
    expect(camelotPosition(track())).toBeNull();
    expect(tempoSpiralPosition(track())).toBeNull();
  });

  it("is stable across calls (path-hashed jitter)", () => {
    const t = track({ audio_features: feats(0.5, 1, 0.5) });
    expect(camelotPosition(t)).toEqual(camelotPosition(t));
    expect(tempoSpiralPosition(t)).toEqual(tempoSpiralPosition(t));
  });

  it("falls back to the display BPM when features are absent", () => {
    const t = track({ bpm: 120 });
    expect(tempoSpiralPosition(t)).not.toBeNull();
  });
});
