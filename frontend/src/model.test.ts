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

  it("applying a preset restores its display options", () => {
    const m = new Model();
    m.clusterWeights = { ...m.clusterWeights, rhythm: 0.5 };
    m.colorByCluster = true;
    m.clusterResolution = 3;
    const p = m.addLensPreset("loud");
    m.clusterWeights = { ...m.clusterWeights, rhythm: 0.1 };
    m.colorByCluster = false;
    m.applyLensPreset(p);
    expect(m.lensId).toBe(p.id);
    expect(m.clusterWeights.rhythm).toBeCloseTo(0.5, 6);
    expect(m.colorByCluster).toBe(true);
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
