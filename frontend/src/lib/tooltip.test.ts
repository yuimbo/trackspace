import { afterEach, describe, expect, it } from "vitest";
import { Model, type Track } from "../model";
import { keyColor, tempoColor } from "./colors";
import {
  buildTooltip,
  renderDescribePanel,
  renderGenrePop,
  renderTooltip,
  type DescribeState,
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
    expect(card.genre).toEqual({
      name: null,
      color: null,
      spanning: true,
      target: null,
      projectedFolder: null,
      projectedConfident: false,
    });
  });

  it("shows the head's projected folder only when it differs from the track's", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set(["a.mp3"]);

    // Differs from the track's folder ("" here) → shown, marked confident.
    const differs = buildTooltip(makeTrack(), m);
    expect(differs.genre?.projectedFolder).toBe("Root/House");
    expect(differs.genre?.projectedConfident).toBe(true);

    // A weak match still shows the destination, just not as confident.
    m.folderPredictionMatches = new Set();
    const weak = buildTooltip(makeTrack(), m);
    expect(weak.genre?.projectedFolder).toBe("Root/House");
    expect(weak.genre?.projectedConfident).toBe(false);

    // Equal to the current folder → nothing to move, so no projected folder.
    m.folderPredictions = new Map([["a.mp3", ""]]);
    expect(buildTooltip(makeTrack(), m).genre?.projectedFolder).toBeNull();
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
    expect(host.querySelectorAll(".tt-stat-chip")).toHaveLength(2);
    const values = [...host.querySelectorAll(".tt-stat-chip-value")].map(
      (v) => v.textContent,
    );
    expect(values).toEqual(["128 BPM", "Am"]);
    const targets = [...host.querySelectorAll<HTMLElement>(".tt-stat-chip")].map(
      (c) => c.dataset.hoverTarget,
    );
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
    const chips = [...host.querySelectorAll<HTMLElement>(".tt-stat-chip")];
    const tempo = chips.find((c) => c.textContent?.startsWith("Tempo"))!;
    const key = chips.find((c) => c.textContent?.startsWith("Key"))!;
    expect(tempo.classList.contains("tt-stat-chip-filled")).toBe(true);
    expect(key.classList.contains("tt-stat-chip-filled")).toBe(true);
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

    const folder = host.querySelector<HTMLElement>(".tt-folder");
    expect(folder?.dataset.hoverTarget).toBe(
      JSON.stringify({ kind: "folder", prefix: "House" }),
    );
    expect(folder?.classList.contains("tt-target")).toBe(true);
    expect(folder?.title).toContain("S solo");

    const cluster = host.querySelector<HTMLElement>(".tt-cluster");
    expect(cluster?.dataset.hoverTarget).toBe(
      JSON.stringify({ kind: "cluster", id: 3 }),
    );
    // The chip also marks itself as the trigger for the full-name pop-over.
    expect(cluster?.dataset.genrePop).toBe("");

    const chips = [...host.querySelectorAll<HTMLElement>(".tt-stat-chip")];
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

    const loc = host.querySelector(".tt-loc");
    const folder = loc?.querySelector<HTMLElement>(".tt-folder");
    const cluster = loc?.querySelector<HTMLElement>(".tt-cluster");
    expect(folder).not.toBeNull();
    expect(cluster).not.toBeNull();
    // A nested folder fills its tag with a parent→own gradient.
    expect(folder?.classList.contains("tt-loc-filled")).toBe(true);
    expect(folder?.style.background).toContain("linear-gradient");
    // The cluster fills its tag with its solid colour.
    expect(cluster?.classList.contains("tt-loc-filled")).toBe(true);
    expect(cluster?.style.background).not.toContain("linear-gradient");
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

describe("renderGenrePop", () => {
  it("shows the untruncated genre name and the projected folder", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Very Long Genre Name That Would Truncate"]]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set(["a.mp3"]);
    const host = document.createElement("div");
    renderGenrePop(host, buildTooltip(makeTrack(), m).genre!);
    expect(host.classList.contains("hidden")).toBe(false);
    expect(host.querySelector(".tt-pop-name")?.textContent).toBe(
      "Very Long Genre Name That Would Truncate",
    );
    expect(host.querySelector(".tt-pop-proj-path")?.textContent).toContain(
      "House",
    );
    expect(host.querySelector(".tt-pop-proj-low")).toBeNull();
  });

  it("marks a low-confidence projection and a spanning genre", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    m.trackStructure = new Map([
      ["a.mp3", structure({ peak_start: 10, genre_spanning: true })],
    ]);
    m.folderPredictions = new Map([["a.mp3", "Root/House"]]);
    m.folderPredictionMatches = new Set();
    const host = document.createElement("div");
    renderGenrePop(host, buildTooltip(makeTrack(), m).genre!);
    expect(host.querySelector(".tt-pop-proj-low")).not.toBeNull();
    expect(host.querySelector(".tt-pop-note")).not.toBeNull();
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
