import { beforeEach, describe, expect, it } from "vitest";
import { Model, type Track } from "./model";

function track(path: string, folder: string): Track {
  return { path, filename: path, folder, tags: {}, artist: "", title: "" };
}

describe("Model external-root view", () => {
  beforeEach(() => localStorage.clear());

  it("hides external-root tracks by default", () => {
    const m = new Model();
    m.setAllTracks([
      track("a.mp3", "lib"),
      track("b.mp3", "ext"),
      track("c.mp3", "ext/sub"),
    ]);
    m.setExternalRoots(["ext"]);
    expect(m.tracks.map((t) => t.path)).toEqual(["a.mp3"]);
    // External tracks stay loaded (they have positions and tags), just not drawn.
    expect(m.allTracks).toHaveLength(3);
  });

  it("shows external-root tracks when toggled on", () => {
    const m = new Model();
    m.setAllTracks([track("a.mp3", "lib"), track("b.mp3", "ext")]);
    m.setExternalRoots(["ext"]);
    m.setShowExternal(true);
    expect(m.tracks.map((t) => t.path)).toEqual(["a.mp3", "b.mp3"]);
  });

  it("identifies external-root tracks for the canvas treatment", () => {
    const m = new Model();
    m.setExternalRoots(["ext"]);
    expect(m.isExternalTrack(track("b.mp3", "ext/sub"))).toBe(true);
    expect(m.isExternalTrack(track("a.mp3", "lib"))).toBe(false);
  });

  it("persists the show-external toggle", () => {
    const m = new Model();
    m.setShowExternal(true);
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.showExternal).toBe(true);
  });

  it("keeps the projection library-only by default", () => {
    const m = new Model();
    expect(m.externalInProjection).toBe(false);
    m.setExternalInProjection(true);
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.externalInProjection).toBe(true);
  });
});

describe("Model proximity keep", () => {
  beforeEach(() => localStorage.clear());

  function fourTracks(m: Model): void {
    m.setAllTracks([
      track("a.mp3", "x"),
      track("b.mp3", "x"),
      track("c.mp3", "x"),
    ]);
  }

  it("hides everything outside the keep-set and returns what it hid", () => {
    const m = new Model();
    fourTracks(m);
    const hidden = m.setProximityKeep(new Set(["a.mp3"]), new Set());
    expect(hidden).toEqual(new Set(["b.mp3", "c.mp3"]));
    expect(m.isTrackHidden(m.trackByPath("b.mp3")!)).toBe(true);
    expect(m.isTrackHidden(m.trackByPath("a.mp3")!)).toBe(false);
  });

  it("reveals the previous step's hides when the neighbourhood widens", () => {
    const m = new Model();
    fourTracks(m);
    const hidden = m.setProximityKeep(new Set(["a.mp3"]), new Set());
    m.setProximityKeep(new Set(["a.mp3", "b.mp3"]), hidden);
    expect(m.isTrackHidden(m.trackByPath("b.mp3")!)).toBe(false);
    expect(m.isTrackHidden(m.trackByPath("c.mp3")!)).toBe(true);
  });

  it("does not count a proximity hide as an explicit exclusion", () => {
    const m = new Model();
    fourTracks(m);
    m.setProximityKeep(new Set(["a.mp3"]), new Set());
    // The Excluded chip counts only explicit exclusions, not the distance filter.
    expect(m.hasExclusions).toBe(false);
    m.hidePaths(["b.mp3"]);
    expect(m.hasExclusions).toBe(true);
    // Clearing exclusions leaves the proximity filter's hides in place.
    m.clearExclusions();
    expect(m.isTrackHidden(m.trackByPath("b.mp3")!)).toBe(true);
    expect(m.isTrackHidden(m.trackByPath("c.mp3")!)).toBe(true);
  });

  it("reveals only the proximity filter's hides when it is dismissed", () => {
    const m = new Model();
    fourTracks(m);
    m.hidePaths(["a.mp3"]); // an explicit exclusion
    m.setProximityKeep(new Set(["a.mp3"]), new Set());
    expect(m.isTrackHidden(m.trackByPath("c.mp3")!)).toBe(true);
    m.clearProximityHidden();
    expect(m.isTrackHidden(m.trackByPath("c.mp3")!)).toBe(false);
    // The explicit exclusion survives the proximity dismiss.
    expect(m.isTrackHidden(m.trackByPath("a.mp3")!)).toBe(true);
  });
});

describe("Model visible tree prefixes", () => {
  beforeEach(() => localStorage.clear());

  it("marks the root and every ancestor folder with a visible track", () => {
    const m = new Model();
    m.setAllTracks([track("a.mp3", "x/y/z"), track("b.mp3", "q")]);
    const prefixes = m.visibleTreePrefixes();
    expect(prefixes.has("")).toBe(true);
    expect(prefixes.has("x")).toBe(true);
    expect(prefixes.has("x/y")).toBe(true);
    expect(prefixes.has("x/y/z")).toBe(true);
    expect(prefixes.has("q")).toBe(true);
    expect(prefixes.has("x/w")).toBe(false);
    expect(m.hasVisibleTracksUnderTreePrefix("x/y")).toBe(true);
    expect(m.hasVisibleTracksUnderTreePrefix("x/w")).toBe(false);
  });

  it("drops prefixes whose only tracks are excluded", () => {
    const m = new Model();
    m.setAllTracks([track("a.mp3", "x/y"), track("b.mp3", "q")]);
    m.focusPaths(new Set(["a.mp3"]));
    const prefixes = m.visibleTreePrefixes();
    expect(prefixes.has("x/y")).toBe(true);
    expect(prefixes.has("q")).toBe(false);
    expect(m.hasVisibleTracksUnderTreePrefix("q")).toBe(false);
  });

  it("invalidates the cache when visibility changes", () => {
    const m = new Model();
    m.setAllTracks([track("a.mp3", "x"), track("b.mp3", "q")]);
    expect(m.hasVisibleTracksUnderTreePrefix("q")).toBe(true);
    m.focusPaths(new Set(["a.mp3"]));
    expect(m.hasVisibleTracksUnderTreePrefix("q")).toBe(false);
    m.clearExclusions();
    expect(m.hasVisibleTracksUnderTreePrefix("q")).toBe(true);
  });
});

describe("Model right-panel tab", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to Analysis", () => {
    expect(new Model().rightPanelTab).toBe("analysis");
  });

  it("persists the active tab", () => {
    const m = new Model();
    m.rightPanelTab = "tracks";
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.rightPanelTab).toBe("tracks");
  });

  it("falls back to Analysis for an unknown persisted tab", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ rightPanelTab: "bogus" }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.rightPanelTab).toBe("analysis");
  });
});

describe("Model lens", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to the tags lens", () => {
    const m = new Model();
    expect(m.lensId).toBe("tags");
    expect(m.viewMode).toBe("tags");
  });

  it("derives viewMode from the lens kind", () => {
    const m = new Model();
    m.lensId = "camelot";
    expect(m.viewMode).toBe("embeddings");
    m.viewMode = "tags";
    expect(m.lensId).toBe("tags");
  });

  it("migrates a legacy viewMode", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ viewMode: "embeddings" }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.lensId).toBe("combined");
  });

  it("persists the active lens and presets", () => {
    const m = new Model();
    m.lensId = "clap";
    const p = m.addLensPreset("My view");
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.lensId).toBe(p.id);
    expect(m2.lensPresets.map((x) => x.name)).toEqual(["My view"]);
    expect(m2.lensPresets[0].baseLens).toBe("clap");
  });

  it("persists the colour mode and migrates the legacy cluster flag", () => {
    localStorage.clear();
    const m = new Model();
    m.colorMode = "key";
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.colorMode).toBe("key");

    localStorage.setItem("trackspace", JSON.stringify({ colorByCluster: true }));
    const m3 = new Model();
    m3.loadLS();
    expect(m3.colorMode).toBe("cluster");
  });

  it("persists the per-variable selection-bloom ranges", () => {
    localStorage.clear();
    const m = new Model();
    m.selectionBloom = "cluster";
    m.setBloomRange(0.9, "cluster");
    m.setBloomRange(0.4, "tempo");
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.selectionBloom).toBe("cluster");
    expect(m2.bloomRange("cluster")).toBeCloseTo(0.9, 6);
    expect(m2.bloomRange("tempo")).toBeCloseTo(0.4, 6);
    // Folder and predicted folder share one persisted slot.
    m2.setBloomRange(0.3, "head");
    expect(m2.bloomRange("folder")).toBeCloseTo(0.3, 6);
  });

  it("keeps a stored zero bloom range as the exact class", () => {
    localStorage.clear();
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ bloomRanges: { tempo: 0 } }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.bloomRange("tempo")).toBe(0);
  });

  it("migrates the legacy bloom range into every variable", () => {
    localStorage.clear();
    localStorage.setItem("trackspace", JSON.stringify({ bloomAmount: 0.7 }));
    const m = new Model();
    m.loadLS();
    expect(m.bloomRange("tempo")).toBeCloseTo(0.7, 6);
    expect(m.bloomRange("key")).toBeCloseTo(0.7, 6);
  });

  it("persists the predicted-folder confidence gate, defaulting to on", () => {
    localStorage.clear();
    const m = new Model();
    expect(m.headColorConfident).toBe(true);
    m.headColorConfident = false;
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.headColorConfident).toBe(false);
  });

  it("applying a preset restores its display options", () => {
    const m = new Model();
    m.clusterWeights = { ...m.clusterWeights, rhythm: 0.5 };
    m.colorMode = "tempo";
    m.clusterResolution = 3;
    const p = m.addLensPreset("loud");
    m.clusterWeights = { ...m.clusterWeights, rhythm: 0.1 };
    m.colorMode = "folder";
    m.applyLensPreset(p);
    expect(m.lensId).toBe(p.id);
    expect(m.clusterWeights.rhythm).toBeCloseTo(0.5, 6);
    expect(m.colorMode).toBe("tempo");
    expect(m.clusterResolution).toBe(3);
  });

  it("deleting the active preset falls back to tags", () => {
    const m = new Model();
    const p = m.addLensPreset("x");
    m.deleteLensPreset(p.id);
    expect(m.lensId).toBe("tags");
    expect(m.lensPresets).toEqual([]);
  });
});

describe("Model selection bloom", () => {
  function makeTrack(over: Partial<Track> = {}): Track {
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

  it("steps the cursor through the colour modes and wraps", () => {
    const m = new Model();
    m.selectionBloom = "folder";
    m.moveSelectionBloom(1);
    expect(m.selectionBloom).toBe("head");
    m.moveSelectionBloom(-1);
    expect(m.selectionBloom).toBe("folder");
    // Backward from the first wraps to the last.
    m.moveSelectionBloom(-1);
    expect(m.selectionBloom).toBe("key");
  });

  it("blooms same-folder tracks by default", () => {
    const m = new Model();
    const a = makeTrack({ path: "House/a.mp3", folder: "House" });
    expect(m.trackBloomsWith(a, makeTrack({ path: "House/b.mp3", folder: "House" }))).toBe(true);
    expect(m.trackBloomsWith(a, makeTrack({ path: "Techno/c.mp3", folder: "Techno" }))).toBe(false);
  });

  it("blooms by cluster, key, tempo and predicted folder", () => {
    const m = new Model();
    const af = (key: number, mode: number) => ({
      tempo: 0.5,
      key,
      mode,
      energy: 0.5,
      danceability: 0.5,
    });
    // a/b share a key; c is five Camelot steps away (minor vs major).
    const a = makeTrack({ path: "a.mp3", bpm: 128, audio_features: af(0, 0) });
    const b = makeTrack({ path: "b.mp3", bpm: 129, audio_features: af(0, 0) });
    const c = makeTrack({ path: "c.mp3", bpm: 150, audio_features: af(4 / 12, 1) });

    m.selectionBloom = "cluster";
    m.clusterAssignments = new Map([
      ["a.mp3", 3],
      ["b.mp3", 3],
      ["c.mp3", 7],
    ]);
    expect(m.trackBloomsWith(a, b)).toBe(true);
    expect(m.trackBloomsWith(a, c)).toBe(false);

    m.selectionBloom = "key";
    expect(m.trackBloomsWith(a, b)).toBe(true);
    expect(m.trackBloomsWith(a, c)).toBe(false);
    // A full range reaches ±6 Camelot steps, so c's five steps now bloom.
    m.setBloomRange(1);
    expect(m.trackBloomsWith(a, c)).toBe(true);

    m.selectionBloom = "tempo";
    m.setBloomRange(0);
    expect(m.trackBloomsWith(a, b)).toBe(false);
    // 0.5 → ±13%: 128 vs 129 blooms, 128 vs 150 does not.
    m.setBloomRange(0.5);
    expect(m.trackBloomsWith(a, b)).toBe(true);
    expect(m.trackBloomsWith(a, c)).toBe(false);

    m.selectionBloom = "head";
    m.folderPredictions = new Map([
      ["a.mp3", "Root/Techno"],
      ["b.mp3", "Root/Techno"],
      ["c.mp3", "Root/House"],
    ]);
    m.setBloomRange(0);
    expect(m.trackBloomsWith(a, b)).toBe(true);
    expect(m.trackBloomsWith(a, c)).toBe(false);
    // One parent level up, both predicted folders share "Root".
    m.setBloomRange(0.25);
    expect(m.trackBloomsWith(a, c)).toBe(true);
  });

  it("keeps a separate range per variable, sharing folder and predicted folder", () => {
    const m = new Model();
    m.setBloomRange(0.9, "tempo");
    m.setBloomRange(0.2, "cluster");
    // Widening the tempo range leaves the others untouched.
    expect(m.bloomRange("tempo")).toBeCloseTo(0.9, 6);
    expect(m.bloomRange("cluster")).toBeCloseTo(0.2, 6);
    expect(m.bloomRange("key")).toBe(0);
    // Folder and predicted folder are the same directory climb, so one slot.
    m.setBloomRange(0.4, "head");
    expect(m.bloomRange("folder")).toBeCloseTo(0.4, 6);
  });

  it("isolates the folder subtree for folder colour, the class otherwise", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "GENRES/Ambient/a.mp3", folder: "GENRES/Ambient", bpm: 128 }),
      makeTrack({ path: "GENRES/Ambient/sub/b.mp3", folder: "GENRES/Ambient/sub", bpm: 128 }),
      makeTrack({ path: "GENRES/dark/c.mp3", folder: "GENRES/dark", bpm: 150 }),
    ]);
    const a = m.trackByPath("GENRES/Ambient/a.mp3")!;

    // Folder colour: the whole subtree, like the left browser.
    m.selectionBloom = "folder";
    expect([...m.bloomPaths(a)].sort()).toEqual([
      "GENRES/Ambient/a.mp3",
      "GENRES/Ambient/sub/b.mp3",
    ]);

    // Any other variable: the range-aware bloom class.
    m.selectionBloom = "tempo";
    expect([...m.bloomPaths(a)].sort()).toEqual([
      "GENRES/Ambient/a.mp3",
      "GENRES/Ambient/sub/b.mp3",
    ]);
  });

  it("never blooms a track the variable cannot place", () => {
    const m = new Model();
    const a = makeTrack({ path: "a.mp3" });
    const b = makeTrack({ path: "b.mp3" });
    m.selectionBloom = "cluster";
    expect(m.trackBloomsWith(a, b)).toBe(false);
    m.selectionBloom = "head";
    expect(m.trackBloomsWith(a, b)).toBe(false);
    m.selectionBloom = "tempo";
    expect(m.trackBloomsWith(a, b)).toBe(false);
    m.selectionBloom = "key";
    expect(m.trackBloomsWith(a, b)).toBe(false);
  });
});
