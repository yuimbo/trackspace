import { afterEach, describe, expect, it } from "vitest";
import { Model, type Track } from "../model";
import { buildTooltip, renderTooltip } from "./tooltip";

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

function structure(over: Partial<import("../model").TrackStructure> = {}) {
  return {
    intro_start: null,
    intro_end: null,
    peak_start: null,
    peak_end: null,
    break_start: null,
    break_end: null,
    ...over,
  };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("buildTooltip", () => {
  it("renders the folder kicker, title, and BPM/key meta", () => {
    const card = buildTooltip(
      makeTrack({ folder: "House", artist: "A", title: "B", bpm: 128.4, musical_key: "Am" }),
      new Model(),
    );
    expect(card.folder).toBe("House");
    expect(card.title).toBe("A — B");
    expect(card.meta).toEqual(["128 BPM", "Am"]);
  });

  it("falls back to the filename when artist and title are empty", () => {
    const card = buildTooltip(makeTrack({ filename: "x.mp3" }), new Model());
    expect(card.title).toBe("x.mp3");
  });

  it("builds structure chips and highlights the previewed section", () => {
    const m = new Model();
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 64, break_start: 192 })]]);
    const idle = buildTooltip(makeTrack(), m);
    expect(idle.chips.map((c) => c.text)).toEqual(["drop 1:04", "break 3:12"]);
    expect(idle.activeSection).toBeNull();

    m.previewTrackPath = "a.mp3";
    m.previewSection = "break";
    const previewing = buildTooltip(makeTrack(), m);
    expect(previewing.activeSection).toBe("break");

    // A different track's tooltip is never highlighted by the preview.
    m.previewTrackPath = "other.mp3";
    expect(buildTooltip(makeTrack(), m).activeSection).toBeNull();
  });

  it("groups the cluster with the genre-spanning flag", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 10, genre_spanning: true })]]);

    const card = buildTooltip(makeTrack(), m);
    expect(card.genre?.name).toBe("Techno");
    expect(card.genre?.color).toBeTruthy();
    expect(card.genre?.spanning).toBe(true);
  });

  it("has no genre group without a cluster or spanning flag", () => {
    expect(buildTooltip(makeTrack(), new Model()).genre).toBeNull();
  });

  it("still shows a spanning badge on a track with no cluster", () => {
    const m = new Model();
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 10, genre_spanning: true })]]);
    const card = buildTooltip(makeTrack(), m);
    expect(card.genre).toEqual({ name: null, color: null, spanning: true });
  });

  it("renders user tags as bars and excludes feature axis ids", () => {
    const card = buildTooltip(
      makeTrack({ tags: { "4x": 0.25, energy: 0.5, mood: 1 } }),
      new Model(),
    );
    expect(card.tags).toEqual([
      { label: "4x", value: 0.25, readout: "0.25" },
      { label: "mood", value: 1, readout: "1.00" },
    ]);
  });

  it("renders audio features as bars with real-unit readouts", () => {
    const card = buildTooltip(
      makeTrack({
        bpm: 128.4,
        musical_key: "Am",
        audio_features: {
          tempo: 0.5,
          key: 0.3,
          mode: 1,
          energy: 0.75,
          danceability: 0.6,
        },
      }),
      new Model(),
    );
    expect(card.features.map((f) => f.label)).toEqual([
      "Tempo",
      "Key",
      "Mode",
      "Energy",
      "Dance",
    ]);
    expect(card.features[0]).toEqual({ label: "Tempo", value: 0.5, readout: "128 BPM" });
    expect(card.features[1]).toEqual({ label: "Key", value: 0.3, readout: "Am" });
    expect(card.features[3]).toEqual({ label: "Energy", value: 0.75, readout: "0.75" });
  });

  it("changes the cache key when the previewed section changes", () => {
    const m = new Model();
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 64 })]]);
    const before = buildTooltip(makeTrack(), m).key;
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    expect(buildTooltip(makeTrack(), m).key).not.toBe(before);
  });
});

describe("renderTooltip", () => {
  it("renders chips, the active chip, the badge hint, and stat bars", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ intro_start: 12, intro_end: 32, peak_start: 64, genre_spanning: true })],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack({ tags: { "4x": 0.5 } }), m));

    const active = host.querySelector(".tt-chip.active");
    expect(active?.textContent).toBe("drop 1:04");
    expect(host.querySelector(".tt-chip[data-section='intro']")?.textContent).toBe(
      "intro 0:12–0:32",
    );
    expect(host.querySelector(".tt-badge")?.textContent).toBe("genre-spanning");
    expect(host.querySelector(".tt-badge")?.getAttribute("title")).toContain("dispersion");
    expect(host.querySelectorAll(".tt-stat-bar i").length).toBe(1);
    expect(host.querySelector(".tt-stat-value")?.textContent).toBe("0.50");
  });

  it("uses textContent for tag names (no HTML injection)", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack({ tags: { "<img src=x>": 0.5 } }), new Model()),
    );
    expect(host.querySelector("img")).toBeNull();
    expect(host.querySelector(".tt-stat-label")?.textContent).toBe("<img src=x>");
  });

  it("replaces previous content on re-render", () => {
    const host = document.createElement("div");
    const m = new Model();
    renderTooltip(host, buildTooltip(makeTrack({ title: "One" }), m));
    renderTooltip(host, buildTooltip(makeTrack({ title: "Two" }), m));
    expect(host.querySelectorAll(".tt-card").length).toBe(1);
    expect(host.querySelector(".tt-title")?.textContent).toBe("Two");
  });
});
