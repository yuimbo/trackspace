import { describe, expect, it } from "vitest";
import type { Track } from "../model";
import {
  BPM_HI,
  BPM_LO,
  bpmSpiralXY,
  camelotGuide,
  camelotPosition,
  camelotXY,
  getLens,
  lensWaitSources,
  lensWeights,
  nextLensId,
  spiralGuide,
  tempoSpiralPosition,
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

  it("cycles from a preset's base lens", () => {
    expect(nextLensId("preset:1", "clap")).toBe("rhythm");
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
