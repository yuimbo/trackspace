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
