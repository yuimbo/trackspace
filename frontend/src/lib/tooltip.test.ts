import { afterEach, describe, expect, it } from "vitest";
import { Model, type Track } from "../model";
import { keyColor, tempoColor } from "./colors";
import {
  buildTooltip,
  renderDescribePanel,
  renderTooltip,
  updateSeekProgress,
  type DescribeState,
  type InspectState,
} from "./tooltip";

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

/** One coordinate, leaning dark at the 98th percentile. */
const coord = {
  axis: "dark <-> bright",
  pos: "dark",
  neg: "bright",
  pole: "dark",
  z: 2.1,
  percentile: 0.98,
  reliability: 2.58,
};

afterEach(() => {
  document.body.innerHTML = "";
});

describe("buildTooltip", () => {
  it("splits the title and artist", () => {
    const card = buildTooltip(
      makeTrack({ folder: "House", artist: "A", title: "B" }),
      new Model(),
    );
    expect(card.folder).toBe("House");
    expect(card.title).toBe("B");
    expect(card.artist).toBe("A");
  });

  it("collapses to just the filename when artist and title are missing", () => {
    const card = buildTooltip(makeTrack({ filename: "x.mp3" }), new Model());
    expect(card.title).toBe("x.mp3");
    expect(card.artist).toBeNull();
  });

  it("gives a nested folder a parent→own gradient tag", () => {
    const root = buildTooltip(
      makeTrack({ folder: "abc12345/GENRES/Rock" }),
      new Model(),
    );
    expect(root.folderColors).toHaveLength(2);
    expect(root.folderColors[0]).not.toBe(root.folderColors[1]);

    const top = buildTooltip(makeTrack({ folder: "abc12345/GENRES" }), new Model());
    expect(top.folderColors).toHaveLength(1);
  });

  it("builds seekbar markers and highlights the previewed section", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, break_start: 192, duration: 256 })],
    ]);
    const idle = buildTooltip(makeTrack(), m);
    // Not auditioning: the seekbar is inactive but still shows its markers.
    expect(idle.seek.playing).toBe(false);
    expect(idle.seek.duration).toBe(256);
    expect(idle.seek.markers.map((c) => c.section)).toEqual(["drop", "break"]);
    expect(idle.activeSection).toBeNull();

    m.previewTrackPath = "a.mp3";
    m.previewSection = "break";
    const previewing = buildTooltip(makeTrack(), m);
    expect(previewing.seek.playing).toBe(true);
    expect(previewing.activeSection).toBe("break");

    // A different track's tooltip is never highlighted by the preview.
    m.previewTrackPath = "other.mp3";
    expect(buildTooltip(makeTrack(), m).activeSection).toBeNull();
  });

  it("places the sections as markers of the structure duration", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, break_start: 192, duration: 256 })],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    m.previewTime = 80;

    const card = buildTooltip(makeTrack(), m);
    expect(card.seek.time).toBe(80);
    expect(card.seek.duration).toBe(256);
    expect(card.seek.markers.map((mk) => mk.section)).toEqual(["drop", "break"]);
    expect(card.seek.markers.map((mk) => mk.fraction)).toEqual([
      64 / 256,
      192 / 256,
    ]);
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

  it("still carries the spanning flag on a track with no cluster", () => {
    const m = new Model();
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 10, genre_spanning: true })]]);
    const card = buildTooltip(makeTrack(), m);
    expect(card.genre).toEqual({
      name: null,
      color: null,
      spanning: true,
      target: null,
    });
    // The chip stays one short word; the sentence is the hover title.
    expect(card.projectedNote).toBe("Loading…");
    expect(card.projectedHint).toContain("Loading the head's predicted folders");
  });

  it("explains every reason the head shows no projected folder", () => {
    const m = new Model();
    const card = (track: Track) => buildTooltip(track, m);

    // No trained head yet.
    const inside = makeTrack({ path: "a.mp3" });
    m.folderPredictionsLoaded = true;
    m.folderPredictionsAvailable = false;
    expect(card(inside).projectedNote).toBe("No head");
    expect(card(inside).projectedHint).toContain("No trained folder head");

    // A track the supervised stage has not reached.
    m.folderPredictionsAvailable = true;
    expect(card(inside).projectedNote).toBe("Not analysed");
    expect(card(inside).projectedHint).toContain("Not analysed yet");

    // The head keeps the track where it is: the folder still renders.
    m.folderPredictions = new Map([["a.mp3", ""]]);
    expect(card(inside).projectedFolder).toBe("Library root");
    expect(card(inside).projectedNote).toBeNull();
  });

  it("predicts for external tracks like any other analysed track", () => {
    const m = new Model();
    m.externalRootIds = new Set(["ext"]);
    m.folderPredictionsLoaded = true;
    m.folderPredictionsAvailable = true;
    m.folderPredictions = new Map([["ext/a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set(["ext/a.mp3"]);

    const card = buildTooltip(
      makeTrack({ path: "ext/a.mp3", folder: "ext" }),
      m,
    );
    expect(card.projectedFolder).toBe("Root/House");
    expect(card.projectedNote).toBeNull();
    expect(card.projectedHint).toContain("Root/House");
  });

  it("shows the head's projected folder even when it matches the track's", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set(["a.mp3"]);
    m.folderPredictionsLoaded = true;
    m.folderPredictionsAvailable = true;

    // Differs from the track's folder ("" here) → shown, marked confident.
    const differs = buildTooltip(makeTrack(), m);
    expect(differs.projectedFolder).toBe("Root/House");
    expect(differs.projectedHint).not.toContain("low confidence");

    // A weak match still shows the destination, just not as confident.
    m.folderPredictionMatches = new Set();
    const weak = buildTooltip(makeTrack(), m);
    expect(weak.projectedFolder).toBe("Root/House");
    expect(weak.projectedHint).toContain("low confidence");

    // Equal to the current folder → still shown, so the chip never hides a place.
    m.folderPredictions = new Map([["a.mp3", ""]]);
    expect(buildTooltip(makeTrack(), m).projectedFolder).toBe("Library root");
  });

  it("re-renders when the head's projection changes", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    const before = buildTooltip(makeTrack(), m).key;
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    expect(buildTooltip(makeTrack(), m).key).not.toBe(before);
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

  it("chips key and BPM and bars the remaining features", () => {
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
    // Mode is folded into the key label, so it is not its own row.
    expect(card.features.map((f) => f.label)).toEqual([
      "Tempo",
      "Key",
      "Energy",
      "Dance",
    ]);
    expect(card.features[0]).toEqual({
      label: "Tempo",
      value: 0,
      readout: "128 BPM",
      chip: true,
      color: tempoColor(0.5),
      target: { kind: "bpm", value: 128.4 },
    });
    expect(card.features[1]).toEqual({
      label: "Key",
      value: 0,
      readout: "Am",
      chip: true,
      color: keyColor(0.3, 1),
      target: { kind: "key", value: "Am" },
    });
    expect(card.features[2]).toEqual({ label: "Energy", value: 0.75, readout: "0.75" });
  });

  it("changes the cache key when the previewed section changes", () => {
    const m = new Model();
    m.trackStructure = new Map([["a.mp3", structure({ peak_start: 64 })]]);
    const before = buildTooltip(makeTrack(), m).key;
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    expect(buildTooltip(makeTrack(), m).key).not.toBe(before);
  });

  it("has no inspect state for the plain hover card", () => {
    const card = buildTooltip(makeTrack(), new Model());
    expect(card.inspect).toBeNull();
    expect(card.results).toEqual([]);
  });

  it("resolves inspect results against the loaded library", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({
        path: "b.mp3",
        folder: "House",
        artist: "A",
        title: "B",
      }),
    ]);
    const card = buildTooltip(makeTrack(), m, {
      path: "a.mp3",
      busy: false,
      message: null,
      mode: "far",
      results: [{ path: "b.mp3", score: -0.42 }],
      proximity: 100,
      tab: "navigate",
      describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
    });
    expect(card.inspect?.mode).toBe("far");
    expect(card.results).toEqual([
      { path: "b.mp3", title: "A — B", folder: "House", score: -0.42 },
    ]);
  });

  it("keeps a result for a track added since the library load", () => {
    const card = buildTooltip(makeTrack(), new Model(), {
      path: "a.mp3",
      busy: false,
      message: null,
      mode: "near",
      results: [{ path: "new.mp3", score: 0.5 }],
      proximity: 100,
      tab: "navigate",
      describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
    });
    expect(card.results[0]).toMatchObject({ path: "new.mp3", title: "new.mp3" });
  });

  it("marks the folder, cluster, key and BPM as navigation targets", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    const card = buildTooltip(
      makeTrack({
        folder: "House",
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
      m,
    );
    expect(card.folderTarget).toEqual({ kind: "folder", prefix: "House" });
    expect(card.genre?.target).toEqual({ kind: "cluster", id: 3 });
    const row = (label: string) => card.features.find((f) => f.label === label);
    expect(row("Tempo")?.target).toEqual({ kind: "bpm", value: 128.4 });
    expect(row("Key")?.target).toEqual({ kind: "key", value: "Am" });
    expect(row("Energy")?.target).toBeUndefined();
  });

  it("marks user tag bars without a target", () => {
    const card = buildTooltip(makeTrack({ tags: { "4x4": 0.5 } }), new Model());
    expect(card.tags[0].target).toBeUndefined();
  });
});

describe("renderTooltip", () => {
  it("renders seekbar markers, the active label, and stat bars", () => {
    const m = new Model();
    m.trackStructure = new Map([
      [
        "a.mp3",
        structure({
          intro_start: 12,
          intro_end: 32,
          peak_start: 64,
          duration: 256,
          genre_spanning: true,
        }),
      ],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack({ tags: { "4x": 0.5 } }), m));

    expect(host.querySelector(".tt-seek-marker.active")?.getAttribute("data-section")).toBe(
      "drop",
    );
    expect(host.querySelector(".tt-seek-marker[data-section='intro']")).not.toBeNull();
    // The spanning flag is a pop-up footer, not a chip badge.
    expect(host.querySelector(".tt-badge")).toBeNull();
    expect(host.querySelectorAll(".tt-stat-bar i").length).toBe(1);
    expect(host.querySelector(".tt-stat-value")?.textContent).toBe("0.50");
  });

  it("renders the inactive seekbar with markers when not auditioning", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, break_start: 192, duration: 256 })],
    ]);
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), m));

    const box = host.querySelector(".tt-seek");
    expect(box?.classList.contains("tt-seek-inactive")).toBe(true);
    // The track is the seek surface the card's delegated mousedown reads.
    expect(host.querySelector("[data-seek]")).not.toBeNull();
    expect(host.querySelector<HTMLElement>(".tt-seek-fill")?.style.width).toBe("0%");
    expect(host.querySelector<HTMLElement>(".tt-seek-head")?.style.left).toBe("0%");
    expect(host.querySelector(".tt-seek-time")?.textContent).toBe("0:00 / 4:16");
    // The hotkey hint floats beside the seekbar; its elaboration is the title.
    const hint = host.querySelector<HTMLElement>(".tt-seek-hint");
    expect(hint?.textContent).toBe("W seek · M mute · M+scroll volume");
    expect(hint?.title).toContain("M + scroll");
    // The markers stay visible, but nothing is auditioning so none is labelled.
    expect(host.querySelectorAll(".tt-seek-marker")).toHaveLength(2);
    expect(host.querySelector(".tt-seek-marker.active")).toBeNull();
    expect(host.querySelector(".tt-seek-label")).toBeNull();
  });

  it("renders a seekbar with markers, labelling only the auditioned section", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, break_start: 192, duration: 256 })],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    m.previewTime = 128;
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), m));

    const fill = host.querySelector<HTMLElement>(".tt-seek-fill");
    expect(fill?.style.width).toBe("50%");
    expect(host.querySelector<HTMLElement>(".tt-seek-head")?.style.left).toBe("50%");
    expect(host.querySelector(".tt-seek-time")?.textContent).toBe("2:08 / 4:16");
    const markers = host.querySelectorAll(".tt-seek-marker");
    expect(markers).toHaveLength(2);
    expect((markers[0] as HTMLElement).style.left).toBe("25%");
    expect((markers[1] as HTMLElement).style.left).toBe("75%");
    // Only the auditioned section expands to a label.
    const label = host.querySelector<HTMLElement>(".tt-seek-label");
    expect(label?.textContent).toBe("drop 1:04");
    expect(label?.style.left).toBe("25%");
    expect(markers[0].classList.contains("active")).toBe(true);
    expect(host.querySelectorAll(".tt-seek-label")).toHaveLength(1);
  });

  it("anchors a late section's label on the right so it stays in the card", () => {
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, break_start: 192, duration: 256 })],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "break";
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), m));

    const label = host.querySelector<HTMLElement>(".tt-seek-label");
    expect(label?.textContent).toBe("break 3:12");
    expect(label?.style.right).toBe("25%");
    expect(label?.style.left).toBe("");
  });

  it("moves the seekbar fill, playhead, and readout without rebuilding the card", () => {
    const host = document.createElement("div");
    const m = new Model();
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 64, duration: 256 })],
    ]);
    m.previewTrackPath = "a.mp3";
    m.previewSection = "drop";
    m.previewTime = 64;
    renderTooltip(host, buildTooltip(makeTrack(), m));

    updateSeekProgress(host, 192, 256);
    expect(host.querySelector<HTMLElement>(".tt-seek-fill")?.style.width).toBe(
      "75%",
    );
    expect(host.querySelector<HTMLElement>(".tt-seek-head")?.style.left).toBe("75%");
    expect(host.querySelector(".tt-seek-time")?.textContent).toBe("3:12 / 4:16");
  });

  it("chips BPM and key, with no meta chips and no bars", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(
        makeTrack({ bpm: 128.4, musical_key: "Am", tags: { tempo: 0.5 } }),
        new Model(),
      ),
    );
    expect(host.querySelector(".tt-meta")).toBeNull();
    // BPM and key are labels: chips in Features, not magnitude bars.
    expect(host.querySelectorAll(".tt-stat-chips .tt-chip")).toHaveLength(2);
    const values = [
      ...host.querySelectorAll(".tt-stat-chips .tt-chip-value"),
    ].map((v) => v.textContent);
    expect(values).toEqual(["128 BPM", "Am"]);
    const targets = [
      ...host.querySelectorAll<HTMLElement>(".tt-stat-chips .tt-chip"),
    ].map((c) => c.dataset.hoverTarget);
    expect(targets).toEqual([
      JSON.stringify({ kind: "bpm", value: 128.4 }),
      JSON.stringify({ kind: "key", value: "Am" }),
    ]);
    expect(host.querySelectorAll(".tt-stat-bar i").length).toBe(0);
  });

  it("fills the key and BPM chips with the feature-lens colours", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(
        makeTrack({
          bpm: 128.4,
          musical_key: "Am",
          audio_features: {
            tempo: 0.5,
            key: 0.3,
            mode: 1,
            energy: 0.5,
            danceability: 0.5,
          },
        }),
        new Model(),
      ),
    );
    const chips = [
      ...host.querySelectorAll<HTMLElement>(".tt-stat-chips .tt-chip"),
    ];
    const tempo = chips.find((c) => c.textContent?.startsWith("Tempo"))!;
    const key = chips.find((c) => c.textContent?.startsWith("Key"))!;
    expect(tempo.classList.contains("tt-chip-filled")).toBe(true);
    expect(key.classList.contains("tt-chip-filled")).toBe(true);
    // happy-dom normalises the `hsl()` spacing, so compare space-insensitively.
    const norm = (c: string) => c.replace(/\s/g, "");
    expect(norm(tempo.style.background)).toBe(norm(tempoColor(0.5)));
    expect(norm(key.style.background)).toBe(norm(keyColor(0.3, 1)));
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

  it("shows the hold-Alt hint on the plain card only, and previews result rows", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    const plain = document.createElement("div");
    renderTooltip(plain, buildTooltip(makeTrack({ path: "a.mp3" }), m));
    expect(plain.querySelector(".tt-hint")?.textContent).toContain("⌥");
    expect(plain.querySelectorAll("[data-preview]").length).toBe(0);

    const pinned = document.createElement("div");
    renderTooltip(
      pinned,
      buildTooltip(makeTrack({ path: "a.mp3" }), m, {
        path: "a.mp3",
        busy: false,
        message: null,
        mode: "far",
        results: [{ path: "b.mp3", score: -0.42 }],
        proximity: 100,
        tab: "navigate",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
      }),
    );
    // The hint is for the plain card; a pinned card is already interactive.
    expect(pinned.querySelector(".tt-hint")).toBeNull();
    expect(
      pinned.querySelector<HTMLElement>("[data-preview]")?.dataset.preview,
    ).toBe("b.mp3");
  });

  it("omits the navigate section while the card is not pinned", () => {
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), new Model()));
    expect(host.querySelector(".tt-inspect")).toBeNull();
    expect(host.querySelectorAll("[data-inspect-action]").length).toBe(0);
  });

  it("renders card navigation targets with their hover hints", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(
        makeTrack({
          folder: "House",
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
        m,
      ),
    );

    const [folder, cluster] = [
      ...host.querySelectorAll<HTMLElement>(".tt-loc .tt-chip"),
    ];
    expect(folder.dataset.hoverTarget).toBe(
      JSON.stringify({ kind: "folder", prefix: "House" }),
    );
    expect(folder.classList.contains("tt-target")).toBe(true);
    expect(folder.title).toContain("S solo");

    expect(cluster.dataset.hoverTarget).toBe(
      JSON.stringify({ kind: "cluster", id: 3 }),
    );

    const chips = [
      ...host.querySelectorAll<HTMLElement>(".tt-stat-chips .tt-chip"),
    ];
    const tempo = chips.find((c) => c.textContent?.startsWith("Tempo"));
    expect(tempo?.dataset.hoverTarget).toBe(
      JSON.stringify({ kind: "bpm", value: 128.4 }),
    );
    const rows = [...host.querySelectorAll<HTMLElement>(".tt-stat")];
    const energy = rows.find((r) => r.textContent?.startsWith("Energy"));
    expect(energy?.dataset.hoverTarget).toBeUndefined();
  });

  it("pairs the folder and cluster tags with distinct icons", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack({ folder: "abc12345/GENRES/Rock", title: "B", artist: "A" }), m),
    );

    const [folder, cluster] = [
      ...host.querySelectorAll<HTMLElement>(".tt-loc .tt-chip"),
    ];
    // The glyph shape, not the colour, tells a folder from a cluster.
    expect(folder.querySelector(".tt-chip-icon-folder")).not.toBeNull();
    expect(cluster.querySelector(".tt-chip-icon-tag")).not.toBeNull();
    // A nested folder fills its chip with a parent→own gradient.
    expect(folder.classList.contains("tt-chip-filled")).toBe(true);
    expect(folder.style.background).toContain("linear-gradient");
    // The cluster fills its chip with its solid colour.
    expect(cluster.classList.contains("tt-chip-filled")).toBe(true);
    expect(cluster.style.background).not.toContain("linear-gradient");
    expect(host.querySelector(".tt-artist")?.textContent).toBe("A");
  });

  it("renders the navigate buttons and ranked rows when pinned", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", artist: "A", title: "B" }),
    ]);
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack(), m, {
        path: "a.mp3",
        busy: false,
        message: null,
        mode: "far",
        results: [{ path: "b.mp3", score: -0.42 }],
        proximity: 100,
        tab: "navigate",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
      }),
    );

    // One toggle flips the filter direction; the slider is the main control.
    const actions = [...host.querySelectorAll(".tt-nav-btn")].map(
      (b) => (b as HTMLElement).dataset.inspectAction,
    );
    expect(actions).toEqual(["toggle-rank-mode"]);
    // The pinned state's mode is anti-track, so the toggle reads "Anti-track".
    expect(host.querySelector(".tt-nav-btn")?.textContent).toBe("Anti-track");
    // The distance filter is a slider, not a discrete top-N "solo" button.
    const range = host.querySelector<HTMLInputElement>(
      "[data-inspect-proximity]",
    );
    expect(range).not.toBeNull();
    expect(range?.value).toBe("100");
    expect(
      host.querySelector("[data-inspect-proximity-value]")?.textContent,
    ).toBe("100%");
    // The anti-track ranking is shown directly, with no expand button.
    expect(
      host.querySelector<HTMLElement>("[data-inspect-action='anti-more']"),
    ).toBeNull();
    expect(host.querySelector("[data-inspect-close]")).not.toBeNull();
    // The tab strip replaces the old "Navigate" heading; the ranking keeps its
    // own heading.
    expect(
      [...host.querySelectorAll(".tt-tab")].map((t) => t.textContent),
    ).toEqual(["Navigate", "Describe"]);
    expect(
      [...host.querySelectorAll(".tt-section-title")].some(
        (t) => t.textContent === "Anti-track",
      ),
    ).toBe(true);

    const row = host.querySelector("[data-inspect-focus]") as HTMLElement;
    expect(row.dataset.inspectFocus).toBe("b.mp3");
    expect(row.textContent).toContain("A — B");
    expect(row.querySelector(".tt-result-score")?.textContent).toBe("-0.42");
    // Hovering the row glows the track in the map.
    expect(JSON.parse(row.dataset.hoverTarget!)).toEqual({
      kind: "track",
      path: "b.mp3",
    });
  });

  it("shows a busy note while a ranking is in flight", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack(), new Model(), {
        path: "a.mp3",
        busy: true,
        message: null,
        mode: null,
        results: [],
        proximity: 100,
        tab: "navigate",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
      }),
    );
    expect(host.querySelector(".tt-note")?.textContent).toContain("Searching");
    expect(
      (host.querySelector(".tt-nav-btn") as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("shows the inspect error message", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack(), new Model(), {
        path: "a.mp3",
        busy: false,
        message: "No analysed tracks yet",
        mode: null,
        results: [],
        proximity: 100,
        tab: "navigate",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
      }),
    );
    expect(host.querySelector(".tt-note-error")?.textContent).toBe(
      "No analysed tracks yet",
    );
  });

  it("renders the Navigate / Describe tabs with the active one marked", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack(), new Model(), {
        path: "a.mp3",
        busy: false,
        message: null,
        mode: null,
        results: [],
        proximity: 100,
        tab: "navigate",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [],
          coordinates: [],
        },
      }),
    );
    const tabs = [...host.querySelectorAll<HTMLElement>(".tt-tab")];
    expect(tabs.map((t) => t.textContent)).toEqual(["Navigate", "Describe"]);
    expect(tabs.map((t) => t.dataset.inspectAction)).toEqual([
      "show-navigate",
      "show-describe",
    ]);
    expect(tabs[0].classList.contains("active")).toBe(true);
    expect(tabs[1].classList.contains("active")).toBe(false);
    expect(tabs[0].getAttribute("aria-selected")).toBe("true");
  });

  it("folds the Describe content out of the card", () => {
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack(), new Model(), {
        path: "a.mp3",
        busy: false,
        message: null,
        mode: null,
        results: [],
        proximity: 100,
        tab: "describe",
        describe: {
          path: "a.mp3",
          busy: false,
          message: null,
          results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
          coordinates: [coord],
        },
      }),
    );
    // The tab strip stays; terms and coordinates fold out into the panel.
    expect(host.querySelector(".tt-nav-btn")).toBeNull();
    expect(host.querySelector(".tt-describe-row")).toBeNull();
    expect(host.querySelector(".tt-coord")).toBeNull();
  });
});

/** A pinned inspect state for *path* (Alt-focusing the card). */
function pinned(path = "a.mp3"): InspectState {
  return {
    path,
    busy: false,
    message: null,
    mode: null,
    results: [],
    proximity: 100,
    tab: "navigate",
    describe: { path, busy: false, message: null, results: [], coordinates: [] },
  };
}

describe("genres list (pinned card)", () => {
  it("expands Folder, Projected folder and Cluster inline with the same styling", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Very Long Genre Name That Would Truncate"]]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set(["a.mp3"]);
    const host = document.createElement("div");
    renderTooltip(
      host,
      buildTooltip(makeTrack({ folder: "Techno" }), m, pinned()),
    );

    const rows = [...host.querySelectorAll<HTMLElement>(".tt-genre-row")];
    expect(rows.map((r) => r.querySelector(".tt-genre-label")?.textContent)).toEqual(
      ["Folder", "Projected folder", "Cluster"],
    );
    // The label sits above the chip, so the chip gets the full row width.
    for (const r of rows) {
      expect(r.firstElementChild?.classList.contains("tt-genre-label")).toBe(
        true,
      );
    }
    const chips = [...host.querySelectorAll<HTMLElement>(".tt-genres .tt-chip")];
    expect(chips).toHaveLength(3);
    expect(chips[0].textContent).toContain("Techno");
    expect(chips[1].textContent).toContain("House");
    expect(chips[2].textContent).toContain(
      "Very Long Genre Name That Would Truncate",
    );
    // The compact location stack is replaced by the inline genres list when pinned.
    expect(host.querySelector(".tt-loc")).toBeNull();
  });

  it("keeps the spanning and low-confidence notes on hover only", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 10, genre_spanning: true })],
    ]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set();
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), m, pinned()));
    const chips = [...host.querySelectorAll<HTMLElement>(".tt-genres .tt-chip")];
    // Low confidence is the projected chip's hover title; spanning the cluster's.
    expect(chips[1].title).toContain("low confidence");
    expect(chips[2].title).toContain("dispersion");
    // Neither is visible text, so the chips stay short.
    expect(chips[1].textContent).not.toContain("low confidence");
    expect(host.querySelector(".tt-genre-foot")).toBeNull();
  });

  it("explains the missing projected folder in the hover title", () => {
    const host = document.createElement("div");
    renderTooltip(host, buildTooltip(makeTrack(), new Model(), pinned()));
    const chips = [...host.querySelectorAll<HTMLElement>(".tt-genres .tt-chip")];
    expect(chips).toHaveLength(3);
    // The projected row reads one short word; the sentence is its hover title.
    expect(chips[1].textContent).toContain("Loading…");
    expect(chips[1].title).toContain("Loading the head's predicted folders");
    // …while the cluster row still reads an em dash when there is none.
    expect(chips[2].textContent).toContain("—");
  });
});

describe("renderDescribePanel", () => {
  const describe = (over: Partial<DescribeState> = {}): DescribeState => ({
    path: "a.mp3",
    busy: false,
    message: null,
    results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
    coordinates: [coord],
    ...over,
  });

  it("renders the terms and coordinates as two columns, with no title", () => {
    const host = document.createElement("div");
    renderDescribePanel(host, describe(), true);
    expect(host.classList.contains("hidden")).toBe(false);
    // No title: the tab already says Describe.
    expect(host.querySelector(".tt-section-title")).toBeNull();
    const panel = host.querySelector(".tt-desc-panel");
    const terms = panel?.querySelector(".tt-desc-col");
    const coords = panel?.querySelector(".tt-coord-col");
    // Terms: the phrase is the row, and clicking it runs a CLAP search.
    const row = terms?.querySelector<HTMLElement>(".tt-describe-row");
    expect(row?.textContent).toBe("a techno track");
    expect(row?.dataset.searchTerm).toBe("a techno track");
    // Coordinates: a bipolar row, leaning pole highlighted at its percentile.
    const poles = [
      ...(coords?.querySelectorAll<HTMLElement>(".tt-coord-pole") ?? []),
    ];
    expect(poles.map((p) => p.textContent)).toEqual(["dark", "bright"]);
    expect(poles[0].classList.contains("lean")).toBe(true);
    expect(coords?.querySelector<HTMLElement>(".tt-coord-marker")?.style.left).toBe(
      "98%",
    );
  });

  it("hides the panel while the Describe tab is not active", () => {
    const host = document.createElement("div");
    renderDescribePanel(host, describe(), false);
    expect(host.classList.contains("hidden")).toBe(true);
    expect(host.querySelector(".tt-desc-panel")).toBeNull();
  });

  it("shows the Describe busy and error notes", () => {
    const busy = document.createElement("div");
    renderDescribePanel(
      busy,
      describe({ busy: true, results: [], coordinates: [] }),
      true,
    );
    expect(busy.querySelector(".tt-note")?.textContent).toContain("CLAP");

    const failed = document.createElement("div");
    renderDescribePanel(
      failed,
      describe({
        message: "No cached CLAP embedding",
        results: [],
        coordinates: [],
      }),
      true,
    );
    expect(failed.querySelector(".tt-note-error")?.textContent).toBe(
      "No cached CLAP embedding",
    );
  });
});
