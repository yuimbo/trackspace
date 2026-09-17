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
