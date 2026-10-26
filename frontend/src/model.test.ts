import { beforeEach, describe, expect, it } from "vitest";
import { Model, type Track } from "./model";
import { MAX_PITCH } from "./lib/camera3d";

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
    // The legacy combined map resolves to the genre recipe, now the default map.
    expect(m.lensId).toBe("genre_full");
  });

  it("migrates saved combined/maest/supervised lens ids to the genre map", () => {
    for (const legacy of ["combined", "maest", "supervised"]) {
      localStorage.setItem("trackspace", JSON.stringify({ lensId: legacy }));
      const m = new Model();
      m.loadLS();
      expect(m.lensId).toBe("genre_full");
    }
  });

  it("a fresh loadLS lands on the Genre map, not tags", () => {
    localStorage.clear();
    const m = new Model();
    m.loadLS();
    expect(m.lensId).toBe("genre_full");
    expect(m.viewMode).toBe("embeddings");
  });

  it("preserves an explicitly saved tags lens on reload", () => {
    localStorage.setItem("trackspace", JSON.stringify({ lensId: "tags" }));
    const m = new Model();
    m.loadLS();
    expect(m.lensId).toBe("tags");
  });

  it("preserves saved camelot and the distinct clap/rhythm lenses", () => {
    for (const id of ["camelot", "clap", "rhythm"]) {
      localStorage.setItem("trackspace", JSON.stringify({ lensId: id }));
      const m = new Model();
      m.loadLS();
      expect(m.lensId).toBe(id);
    }
  });

  it("migrates stored preset base lenses to the genre map", () => {
    for (const legacy of ["combined", "maest", "supervised"]) {
      localStorage.setItem(
        "trackspace",
        JSON.stringify({
          lensPresets: [{ id: "preset:1", name: "My view", baseLens: legacy }],
        }),
      );
      const m = new Model();
      m.loadLS();
      expect(m.lensPresets[0].baseLens).toBe("genre_full");
    }
  });

  it("migrates a saved scope lens id to the genre map", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ scopeLensId: "combined" }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.scopeLensId).toBe("genre_full");
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

describe("Model selectPaths", () => {
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

  it("replaces the selection with the in-view paths", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    m.selected.add("c.mp3");
    m.selectPaths(["a.mp3", "b.mp3"], false);
    expect(m.selected).toEqual(new Set(["a.mp3", "b.mp3"]));
  });

  it("appends to the current selection instead of replacing", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    m.selected.add("c.mp3");
    m.selectPaths(["a.mp3"], true);
    expect(m.selected).toEqual(new Set(["a.mp3", "c.mp3"]));
  });

  it("skips hidden tracks", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    m.hidePaths(["b.mp3"]);
    m.selectPaths(["a.mp3", "b.mp3"], false);
    expect(m.selected).toEqual(new Set(["a.mp3"]));
  });

  it("clears the selection when a bare call re-selects the same target", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    m.selected.add("a.mp3");
    m.selectPaths(["a.mp3"], false);
    expect(m.selected.size).toBe(0);
    // An append over the same target is a no-op, not a toggle.
    m.selected.add("a.mp3");
    m.selectPaths(["a.mp3"], true);
    expect(m.selected).toEqual(new Set(["a.mp3"]));
  });

  it("no-ops when nothing would change", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const events: unknown[] = [];
    m.on("change", () => events.push("change"));
    // Empty target with an empty selection: nothing to select or clear.
    m.selectPaths([], false);
    expect(events.length).toBe(0);
  });
});

describe("Model view rules", () => {
  beforeEach(() => localStorage.clear());

  it("exposes the rule stack as the view modifiers", () => {
    const m = new Model();
    m.viewRules = [{ id: "vm-1", kind: "search", label: "kick" }];
    expect(m.viewModifiers).toEqual([
      { id: "vm-1", kind: "search", label: "kick" },
    ]);
  });

  it("hides a track whose rule does not match", () => {
    const m = new Model();
    const a = track("a.mp3", "lib");
    const b = track("b.mp3", "lib");
    m.setAllTracks([a, b]);
    m.viewRules = [
      {
        id: "vm-1",
        kind: "filters",
        label: "Tempo",
        scope: "tags",
        matches: (t) => t.path === "a.mp3",
      },
    ];
    expect(m.isVisible(a)).toBe(true);
    expect(m.isVisible(b)).toBe(false);
    // The explicit-hidden predicate is untouched by the rule.
    expect(m.isTrackHidden(b)).toBe(false);
  });

  it("applies a tags-scoped rule only in tags mode", () => {
    const m = new Model();
    const b = track("b.mp3", "lib");
    m.setAllTracks([b]);
    m.viewRules = [
      {
        id: "vm-1",
        kind: "filters",
        label: "Tempo",
        scope: "tags",
        matches: () => false,
      },
    ];
    expect(m.isVisible(b)).toBe(false);
    m.viewMode = "embeddings";
    expect(m.isVisible(b)).toBe(true);
  });

  it("keeps a rule without a matcher from hiding anything", () => {
    const m = new Model();
    const b = track("b.mp3", "lib");
    m.setAllTracks([b]);
    m.viewRules = [{ id: "vm-1", kind: "search", label: "kick" }];
    expect(m.isVisible(b)).toBe(true);
  });
});

describe("Model rule readouts", () => {
  beforeEach(() => localStorage.clear());

  function featureTrack(path: string, folder: string): Track {
    return {
      ...track(path, folder),
      bpm: 128,
      musical_key: "Am",
      audio_features: {
        tempo: 0.5,
        key: 0.25,
        mode: 1,
        energy: 0.5,
        danceability: 0.5,
      },
    };
  }

  it("names a folder target with its colour family", () => {
    const m = new Model();
    const rule = m.hoverTargetRule({ kind: "folder", prefix: "House/sub" })!;
    expect(rule.value).toBe("House/sub");
    expect(rule.icon).toBe("folder");
    expect(rule.colors).toHaveLength(1);
    expect(rule.title).toBe("House/sub");
    const nested = m.hoverTargetRule({ kind: "folder", prefix: "House/sub/deep" })!;
    expect(nested.colors).toHaveLength(2);
  });

  it("names the library root", () => {
    const m = new Model();
    const rule = m.hoverTargetRule({ kind: "folder", prefix: "" })!;
    expect(rule.value).toBe("Library root");
    expect(rule.title).toBe("Library root");
  });

  it("names a cluster with its label and hue", () => {
    const m = new Model();
    m.clusterNames = new Map([[3, "Deep house"]]);
    const rule = m.hoverTargetRule({ kind: "cluster", id: 3 })!;
    expect(rule.value).toBe("Deep house");
    expect(rule.icon).toBe("tag");
    expect(rule.colors).toHaveLength(1);
    expect(rule.title).toBe("Cluster 3");
  });

  it("names a BPM and key target", () => {
    const m = new Model();
    expect(m.hoverTargetRule({ kind: "bpm", value: 128.4 })!.value).toBe(
      "128 BPM ±2",
    );
    expect(m.hoverTargetRule({ kind: "key", value: "Am" })!.value).toBe("Key Am");
  });

  it("names a track target by title, falling back to filename", () => {
    const m = new Model();
    m.setAllTracks([{ ...track("a.mp3", "x"), title: "Untitled" }]);
    expect(m.hoverTargetRule({ kind: "track", path: "a.mp3" })!.value).toBe(
      "Untitled",
    );
    expect(m.hoverTargetRule({ kind: "track", path: "b.mp3" })!.value).toBe(
      "b.mp3",
    );
  });

  it("uses a path set's own label, else its size", () => {
    const m = new Model();
    expect(
      m.hoverTargetRule({
        kind: "paths",
        paths: ["a.mp3", "b.mp3"],
        label: "120–130 BPM",
      })!.value,
    ).toBe("120–130 BPM");
    expect(m.hoverTargetRule({ kind: "paths", paths: ["a.mp3"] })!.value).toBe(
      "1 track",
    );
  });

  it("names the bloom's folder at its current range", () => {
    const m = new Model();
    m.selectionBloom = "folder";
    const t = featureTrack("House/sub/deep/a.mp3", "House/sub/deep");
    m.setBloomRange(0, "folder");
    expect(m.bloomRule(t).value).toBe("House/sub/deep");
    m.setBloomRange(0.25, "folder");
    expect(m.bloomRule(t).value).toBe("House/sub");
    expect(m.bloomRule(t).title).toContain("parent");
  });

  it("names the bloom's tempo with its tolerance", () => {
    const m = new Model();
    m.selectionBloom = "tempo";
    const t = featureTrack("a.mp3", "x");
    m.setBloomRange(0, "tempo");
    expect(m.bloomRule(t).value).toBe("128 BPM");
    m.setBloomRange(0.4, "tempo");
    expect(m.bloomRule(t).value).toBe("128 BPM ±10%");
  });

  it("names the bloom's key with its step count", () => {
    const m = new Model();
    m.selectionBloom = "key";
    const t = featureTrack("a.mp3", "x");
    m.setBloomRange(0, "key");
    expect(m.bloomRule(t).value).toBe("Key Am");
    m.setBloomRange(0.5, "key");
    expect(m.bloomRule(t).value).toBe("Key Am ±3 steps");
  });

  it("names a single-folder selection with its colour family", () => {
    const m = new Model();
    m.setAllTracks([track("House/a.mp3", "House")]);
    m.selected = new Set(["House/a.mp3"]);
    const rule = m.selectionRule()!;
    expect(rule.value).toBe("House");
    expect(rule.colors).toHaveLength(1);
  });

  it("caps a multi-folder selection and lists it on hover", () => {
    const m = new Model();
    m.setAllTracks([
      track("House/a.mp3", "House"),
      track("Techno/b.mp3", "Techno"),
      track("Trance/c.mp3", "Trance"),
      track("Disco/d.mp3", "Disco"),
    ]);
    m.selected = new Set([
      "House/a.mp3",
      "Techno/b.mp3",
      "Trance/c.mp3",
      "Disco/d.mp3",
    ]);
    const rule = m.selectionRule()!;
    expect(rule.value).toBe("Disco, House +2 more");
    expect(rule.title?.split("\n")).toEqual([
      "Disco",
      "House",
      "Techno",
      "Trance",
    ]);
    expect(rule.colors).toHaveLength(0);
  });
});

describe("Model eval-mode persistence", () => {
  beforeEach(() => localStorage.clear());

  it("writes nothing to localStorage while eval mode is on", () => {
    const m = new Model();
    m.hoverPreview = true;
    m.showExternal = true;
    m.setEvalMode(true);
    m.saveLS();
    // The in-memory state still changed for the session…
    expect(m.hoverPreview).toBe(true);
    // …but a fresh model reads the untouched normal-mode storage.
    const m2 = new Model();
    m2.loadLS();
    expect(m2.hoverPreview).toBe(false);
    expect(m2.showExternal).toBe(false);
  });

  it("persists normally outside eval mode", () => {
    const m = new Model();
    m.hoverPreview = true;
    m.showExternal = true;
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.hoverPreview).toBe(true);
    expect(m2.showExternal).toBe(true);
  });
});

describe("Model 3-D orbit view", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to off", () => {
    expect(new Model().view3d).toEqual({ on: false, yaw: 0.6, pitch: 0.35 });
  });

  it("persists the orbit", () => {
    const m = new Model();
    m.view3d = { on: true, yaw: 1.2, pitch: 0.4 };
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    expect(m2.view3d).toEqual({ on: true, yaw: 1.2, pitch: 0.4 });
  });

  it("falls back to defaults for garbage persisted input", () => {
    const defaults = { on: false, yaw: 0.6, pitch: 0.35 };
    for (const stored of [
      { view3d: "bogus" },
      { view3d: { on: "yes", yaw: 1.2, pitch: 0.4 } },
      { view3d: { on: true } },
      { view3d: { on: true, yaw: "1.2", pitch: 0.4 } },
      { view3d: { on: false, yaw: 1.2, pitch: 0.4 } },
    ]) {
      localStorage.setItem("trackspace", JSON.stringify(stored));
      const m = new Model();
      m.loadLS();
      expect(m.view3d).toEqual(defaults);
    }
  });

  it("clamps a persisted pitch to ±MAX_PITCH", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ view3d: { on: true, yaw: 0.4, pitch: 99 } }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.view3d).toEqual({ on: true, yaw: 0.4, pitch: MAX_PITCH });

    localStorage.setItem(
      "trackspace",
      JSON.stringify({ view3d: { on: true, yaw: 0.4, pitch: -99 } }),
    );
    const m2 = new Model();
    m2.loadLS();
    expect(m2.view3d).toEqual({ on: true, yaw: 0.4, pitch: -MAX_PITCH });
  });

  it("is3d only for a projection lens with the view on, outside eval mode", () => {
    const m = new Model();
    expect(m.is3d).toBe(false);
    m.view3d = { on: true, yaw: 0.6, pitch: 0.35 };
    // Tags lens: no z axis.
    m.lensId = "tags";
    expect(m.is3d).toBe(false);
    // Projection lens: 3-D.
    m.lensId = "combined";
    expect(m.is3d).toBe(true);
    // Deterministic lens: stays 2-D.
    m.lensId = "camelot";
    expect(m.is3d).toBe(false);
    // Eval gallery forces 2-D even for a projection lens.
    m.lensId = "combined";
    m.setEvalMode(true);
    expect(m.is3d).toBe(false);
    m.setEvalMode(false);
    expect(m.is3d).toBe(true);
    // Toggle off.
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    expect(m.is3d).toBe(false);
  });

  it("shows3d follows is3d, plus the still-blending collapse state", () => {
    const m = new Model();
    m.lensId = "combined";
    // Not armed, not blending: neither renders 3-D.
    expect(m.is3d).toBe(false);
    expect(m.shows3d).toBe(false);
    // Armed (mid-entry or full): both render 3-D.
    m.view3d = { on: true, yaw: 0.6, pitch: 0.35 };
    expect(m.is3d).toBe(true);
    expect(m.shows3d).toBe(true);
    // Turned off but still blending back: the collapse tween keeps rendering 3-D.
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    m.view3dBlend = 0.5;
    expect(m.is3d).toBe(false);
    expect(m.shows3d).toBe(true);
    // Blend reaches 0: rendering falls back to 2-D.
    m.view3dBlend = 0;
    expect(m.shows3d).toBe(false);
  });

  it("shows3d honours the lens and eval gates, not just the blend", () => {
    const m = new Model();
    m.lensId = "tags";
    m.view3dBlend = 0.5;
    expect(m.shows3d).toBe(false);
    m.lensId = "combined";
    expect(m.shows3d).toBe(true);
    m.setEvalMode(true);
    expect(m.shows3d).toBe(false);
  });

  it("eval dims drive is3d/shows3d and reset when eval ends", () => {
    const m = new Model();
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    m.setEvalMode(true);
    expect(m.evalDims).toBe(2);
    expect(m.is3d).toBe(false);
    expect(m.shows3d).toBe(false);

    m.setEvalDims(3);
    expect(m.is3d).toBe(true);
    expect(m.shows3d).toBe(true);
    // The eval dimensionality does not depend on the production lens kind.
    m.lensId = "tags";
    expect(m.is3d).toBe(true);

    m.view3dBlend = 0.4;
    m.setEvalDims(2);
    expect(m.is3d).toBe(false);
    // The collapse tween keeps rendering 3-D until the blend reaches 0.
    expect(m.shows3d).toBe(true);
    m.view3dBlend = 0;
    expect(m.shows3d).toBe(false);

    m.setEvalDims(3);
    m.setEvalMode(false);
    expect(m.evalDims).toBe(2);
    expect(m.is3d).toBe(false);
    expect(m.view3dBlend).toBe(0);
  });

  it("loadLS snaps the blend to the persisted on/off state", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ view3d: { on: true, yaw: 0.6, pitch: 0.35 } }),
    );
    const m = new Model();
    m.loadLS();
    expect(m.view3dBlend).toBe(1);

    localStorage.setItem(
      "trackspace",
      JSON.stringify({ view3d: { on: false, yaw: 0.6, pitch: 0.35 } }),
    );
    const m2 = new Model();
    m2.loadLS();
    expect(m2.view3dBlend).toBe(0);
  });

  it("does not persist the blend (it is transient)", () => {
    const m = new Model();
    m.view3d = { on: true, yaw: 1.2, pitch: 0.4 };
    m.view3dBlend = 0.3; // mid-tween
    m.saveLS();
    const m2 = new Model();
    m2.loadLS();
    // The reloaded view starts fully on, not frozen mid-tween.
    expect(m2.view3dBlend).toBe(1);
  });
});

describe("Model per-dimensionality t-SNE iteration budget", () => {
  beforeEach(() => localStorage.clear());

  it("shares one budget per dimensionality across every lens", () => {
    const m = new Model();
    m.genreFullEnabled = true;
    m.lensId = "genre_full";
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    m.setTsneIterationForActiveView(250, 150);
    m.view3d.on = true;
    m.setTsneIterationForActiveView(500, 250);
    m.saveLS();

    const m2 = new Model();
    m2.loadLS();
    m2.genreFullEnabled = true;
    m2.lensId = "genre_full";
    // 2-D and 3-D keep their own budgets.
    m2.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    expect(m2.tsneIterationFor()).toEqual({ nIter: 250, early: 150 });
    m2.view3d.on = true;
    expect(m2.tsneIterationFor()).toEqual({ nIter: 500, early: 250 });
    // A different lens in the same 2-D dimension shares the 2-D budget.
    m2.view3d.on = false;
    m2.lensId = "clap";
    expect(m2.tsneIterationFor()).toEqual({ nIter: 250, early: 150 });
  });

  it("migrates a legacy global override to the active saved dimension only", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({
        lensId: "genre_full",
        view3d: { on: false, yaw: 0.6, pitch: 0.35 },
        // Legacy global shape: one budget for every view.
        tsneIteration: { nIter: 500, early: 250 },
      }),
    );
    const m = new Model();
    m.loadLS();
    // The legacy override lands on the active saved dimensionality (2-D)…
    m.genreFullEnabled = true;
    expect(m.tsneIterationFor()).toEqual({ nIter: 500, early: 250 });
    // …and nowhere else: 3-D keeps the recipe default.
    m.view3d.on = true;
    expect(m.tsneIterationFor()).toEqual({ nIter: null, early: null });
    // …but it is shared across every 2-D lens.
    m.view3d.on = false;
    m.lensId = "clap";
    expect(m.tsneIterationFor()).toEqual({ nIter: 500, early: 250 });
  });

  it("migrates a legacy per-view (lens:dims) map to per-dimensionality", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({
        lensId: "clap",
        view3d: { on: false, yaw: 0.6, pitch: 0.35 },
        // Old per-view shape: keyed by lens + dimensionality.
        tsneIteration: {
          "clap:d2": { nIter: 200, early: 100 },
          "genre_full:d2": { nIter: 900, early: 800 },
          "genre_full:d3": { nIter: 300, early: 150 },
          "rhythm:d3": { nIter: 400, early: 200 },
        },
      }),
    );
    const m = new Model();
    m.loadLS();
    m.genreFullEnabled = true;
    m.lensId = "clap";
    // The active dimension prefers the active saved lens (clap), not genre.
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    expect(m.tsneIterationFor()).toEqual({ nIter: 200, early: 100 });
    // The other dimension picks the genre entry deterministically.
    m.view3d.on = true;
    expect(m.tsneIterationFor()).toEqual({ nIter: 300, early: 150 });
  });

  it("migrates a legacy per-view map's other dimension to the first sorted valid entry", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({
        lensId: "clap",
        view3d: { on: false, yaw: 0.6, pitch: 0.35 },
        // No genre entry for 3-D: the non-active dimension falls back to the
        // first sorted valid entry (rhythm < tempo).
        tsneIteration: {
          "tempo:d3": { nIter: 400, early: 200 },
          "rhythm:d3": { nIter: 300, early: 150 },
          "clap:d2": { nIter: 200, early: 100 },
        },
      }),
    );
    const m = new Model();
    m.loadLS();
    m.view3d = { on: false, yaw: 0.6, pitch: 0.35 };
    expect(m.tsneIterationFor()).toEqual({ nIter: 200, early: 100 });
    m.view3d.on = true;
    expect(m.tsneIterationFor()).toEqual({ nIter: 300, early: 150 });
  });

  it("keeps the per-dimensionality map fresh: no stored override for a fresh 50/0", () => {
    const m = new Model();
    m.genreFullEnabled = true;
    m.lensId = "genre_full";
    expect(m.tsneIterationFor()).toEqual({ nIter: null, early: null });
    // Clearing an override removes the entry entirely (recipe default 50/0).
    m.setTsneIterationForActiveView(250, 150);
    expect(m.tsneIterationFor()).toEqual({ nIter: 250, early: 150 });
    m.setTsneIterationForActiveView(null, null);
    expect(m.tsneIterationFor()).toEqual({ nIter: null, early: null });
    expect(Object.keys(m.tsneIteration)).toHaveLength(0);
  });
});
