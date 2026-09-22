import { describe, expect, it } from "vitest";
import {
  bloomFolderLevels,
  bloomKeySteps,
  bloomRangeLabel,
  bloomTempoPct,
  camelotSteps,
  clusterTracksByKey,
  clusterTracksByTempo,
  folderAncestor,
  trackCamelotFifths,
} from "./bloom";
import type { Track } from "../model";

function track(key: number, mode: number): Track {
  return {
    path: "a.mp3",
    filename: "a.mp3",
    folder: "",
    tags: {},
    artist: "",
    title: "",
    audio_features: { tempo: 0.5, key, mode, energy: 0.5, danceability: 0.5 },
  };
}

describe("bloom range", () => {
  it("maps the range onto parent levels, ±BPM and ±key steps", () => {
    expect(bloomFolderLevels(0)).toBe(0);
    expect(bloomFolderLevels(0.5)).toBe(2);
    expect(bloomFolderLevels(1)).toBe(4);
    expect(bloomTempoPct(0.62)).toBe(16);
    expect(bloomKeySteps(1)).toBe(6);
  });

  it("names the range per variable, and gives cluster none", () => {
    expect(bloomRangeLabel("folder", 0)).toBe("exact folder");
    expect(bloomRangeLabel("folder", 0.5)).toBe("+2 parent directories");
    expect(bloomRangeLabel("head", 0.25)).toBe("+1 parent directory");
    expect(bloomRangeLabel("tempo", 0.62)).toBe("±16% BPM");
    expect(bloomRangeLabel("key", 0)).toBe("exact key");
    expect(bloomRangeLabel("key", 1)).toBe("±6 steps");
    expect(bloomRangeLabel("cluster", 0.5)).toBeNull();
  });

  it("climbs parent directory ancestors", () => {
    expect(folderAncestor("Root/A/B/C", 0)).toBe("Root/A/B/C");
    expect(folderAncestor("Root/A/B/C", 2)).toBe("Root/A");
    expect(folderAncestor("Root/A/B/C", 9)).toBe("");
    expect(folderAncestor("", 1)).toBe("");
  });

  it("measures Camelot-wheel distance in fifths", () => {
    // C minor is three fifths counter-clockwise of C major.
    expect(camelotSteps(track(0, 0), track(0, 0))).toBe(0);
    // One fifth clockwise of that minor is a single step away.
    expect(camelotSteps(track(0, 0), track(8 / 12, 1))).toBe(1);
    // No cached features → unplaceable.
    const bare: Track = {
      path: "b.mp3",
      filename: "b.mp3",
      folder: "",
      tags: {},
      artist: "",
      title: "",
    };
    expect(camelotSteps(track(0, 0), bare)).toBeNull();
  });
});

describe("bloom clustering", () => {
  const bpmTrack = (path: string, bpm: number): Track => ({
    path,
    filename: path,
    folder: "",
    tags: {},
    artist: "",
    title: "",
    bpm,
  });

  it("merges tempo rows as the tempo bloom widens", () => {
    const tracks = [
      bpmTrack("a", 120),
      bpmTrack("b", 128),
      bpmTrack("c", 200),
    ];
    const exact = clusterTracksByTempo(tracks, 0);
    expect(exact.map((c) => c.map((t) => t.path))).toEqual([
      ["a"],
      ["b"],
      ["c"],
    ]);
    const wide = clusterTracksByTempo(tracks, 0.5);
    expect(wide.map((c) => c.map((t) => t.path))).toEqual([
      ["a", "b"],
      ["c"],
    ]);
  });

  it("merges key rows as the key bloom widens", () => {
    // A minor sits at fifth 0; C minor three steps round the wheel at 9.
    const am = track(0.25, 0);
    const cm = track(0, 0);
    expect(trackCamelotFifths(am)).toBe(0);
    expect(trackCamelotFifths(cm)).toBe(9);
    expect(clusterTracksByKey([am, cm], 0).map((c) => c.length)).toEqual([1, 1]);
    expect(clusterTracksByKey([am, cm], 1).map((c) => c.length)).toEqual([2]);
  });
});
