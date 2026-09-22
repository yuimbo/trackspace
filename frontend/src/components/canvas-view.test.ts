import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Model,
  type QueueKindStatus,
  type QueueStatus,
  type Track,
} from "../model";
import {
  clusterColor,
  dirColor,
  keyColor,
  parseColor,
  tempoColor,
} from "../lib/colors";
import {
  CanvasView,
  CONFIDENT_MISS_COLOR,
  nearestInBlob,
  tabAt,
  type DotPosition,
} from "./canvas-view";

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

// ─── Cluster hover glow ───────────────────────────────────────

/** Minimal 2-D context stand-in that records every alpha the dot pass sets. */
function makeFakeCtx(): {
  ctx: CanvasRenderingContext2D;
  alphas: number[];
  ellipses: { x: number; y: number; rx: number; ry: number }[];
  rects: { x: number; y: number; w: number; h: number }[];
} {
  const alphas: number[] = [];
  const ellipses: { x: number; y: number; rx: number; ry: number }[] = [];
  const rects: { x: number; y: number; w: number; h: number }[] = [];
  const target: Record<string, unknown> = {
    measureText: () => ({ width: 10 }),
    ellipse: (x: number, y: number, rx: number, ry: number) => {
      ellipses.push({ x, y, rx, ry });
    },
    rect: (x: number, y: number, w: number, h: number) => {
      rects.push({ x, y, w, h });
    },
  };
  const ctx = new Proxy(target, {
    get(t, prop: string) {
      if (prop in t) return t[prop];
      return () => undefined;
    },
    set(t, prop: string, value) {
      if (prop === "globalAlpha") alphas.push(value as number);
      else t[prop] = value;
      return true;
    },
  });
  return {
    ctx: ctx as unknown as CanvasRenderingContext2D,
    alphas,
    ellipses,
    rects,
  };
}

describe("CanvasView cluster hover glow", () => {
  let alphas: number[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeFakeCtx();
    alphas = fake.alphas;
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function makeCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", folder: "House" }),
    ]);
    return new CanvasView(m);
  }

  it("collects a halo for a hovered cluster's members in its swatch colour", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.hoverTarget = { kind: "cluster", id: 2 };
    cv.draw();
    const pts = cv.collectHoverGlowPoints(cv.model.tracks);
    expect(pts).toHaveLength(1);
    expect(pts[0].color).toEqual(parseColor(clusterColor(2)));
  });

  it("collects no halo when nothing is hovered", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.draw();
    expect(cv.collectHoverGlowPoints(cv.model.tracks)).toEqual([]);
  });

  it("blooms the hovered dot's selection-cursor family, not its folder", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([
      ["a.mp3", 2],
      ["b.mp3", 2],
    ]);
    cv.model.selectionBloom = "cluster";
    cv.hoveredBloomTrack = cv.model.tracks[0];
    cv.draw();
    const pts = cv.collectHoverGlowPoints(cv.model.tracks);
    // Both tracks share cluster 2, though their folders differ.
    expect(pts).toHaveLength(2);
    expect(pts[0].color).toEqual(parseColor(clusterColor(2)));
  });

  it("blooms folder ancestors as the range widens", () => {
    const cv = makeCanvas();
    cv.model.selectionBloom = "folder";
    cv.model.setBloomRange(0);
    cv.hoveredBloomTrack = cv.model.tracks[0];
    cv.draw();
    // a.mp3 is at the root; b.mp3 is in House. At range 0 only the exact
    // folder matches, so the root track blooms alone.
    expect(cv.collectHoverGlowPoints(cv.model.tracks)).toHaveLength(1);
    // +2 parent levels: both share the "" root ancestor.
    cv.model.setBloomRange(0.5);
    expect(cv.collectHoverGlowPoints(cv.model.tracks)).toHaveLength(2);
  });

  it("sets and clears the hover target from a card element", () => {
    const cv = makeCanvas();
    vi.spyOn(cv, "scheduleDraw").mockImplementation(() => {});
    const target = document.createElement("div");
    target.dataset.hoverTarget = JSON.stringify({ kind: "bpm", value: 128 });
    cv.$tip.appendChild(target);

    target.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    expect(cv.hoverTarget).toEqual({ kind: "bpm", value: 128 });

    target.dispatchEvent(
      new MouseEvent("mouseout", { bubbles: true, relatedTarget: document.body }),
    );
    expect(cv.hoverTarget).toBeNull();
  });

  it("ignores moves between children of the same card target", () => {
    const cv = makeCanvas();
    vi.spyOn(cv, "scheduleDraw").mockImplementation(() => {});
    const target = document.createElement("div");
    target.dataset.hoverTarget = JSON.stringify({ kind: "key", value: "Am" });
    const child = document.createElement("span");
    target.appendChild(child);
    cv.$tip.appendChild(target);

    target.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    child.dispatchEvent(
      new MouseEvent("mouseout", { bubbles: true, relatedTarget: target }),
    );
    expect(cv.hoverTarget).toEqual({ kind: "key", value: "Am" });
  });

  it("fades tracks outside the active search result set", () => {
    const cv = makeCanvas();
    cv.searchResults = new Set(["a.mp3"]);
    cv.draw();
    // The non-match fades; the match does not.
    expect(alphas).toContain(0.15);
    expect(alphas.filter((a) => a === 0.15)).toHaveLength(1);
  });

  it("does not fade anything without an active search", () => {
    const cv = makeCanvas();
    cv.draw();
    expect(alphas).not.toContain(0.15);
  });
});

describe("CanvasView search field", () => {
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      makeFakeCtx().ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function makeCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "clap";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    return new CanvasView(m);
  }

  it("collects a glow point per landmark while the box is active", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.draw();
    expect(cv.collectSearchGlowPoints()).toHaveLength(1);
  });

  it("collects no field while the search box is not active", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = false;
    cv.draw();
    expect(cv.collectSearchGlowPoints()).toEqual([]);
  });

  it("collects no field off the CLAP lens", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.model.lensId = "rhythm";
    cv.draw();
    expect(cv.collectSearchGlowPoints()).toEqual([]);
  });

  it("collects the field under a CLAP scope", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.model.lensId = "combined";
    cv.model.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "clap" };
    cv.model.scopePositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    cv.draw();
    expect(cv.collectSearchGlowPoints()).toHaveLength(1);
  });
});

describe("CanvasView scope positions", () => {
  let ellipses: { x: number; y: number; rx: number; ry: number }[];
  let rects: { x: number; y: number; w: number; h: number }[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeFakeCtx();
    ellipses = fake.ellipses;
    rects = fake.rects;
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("resolves in-scope tracks, hides unplaced in-region ones, keeps the rest", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", folder: "House" }),
      makeTrack({ path: "c.mp3", folder: "Techno" }),
    ]);
    const cv = new CanvasView(m);
    m.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
    m.scopePositions = new Map([["a.mp3", { x: 0.4, y: 0.6 }]]);
    // c.mp3's outer position is outside the circle.
    m.embeddingPositions = new Map([
      ["a.mp3", { x: 0.5, y: 0.5 }],
      ["b.mp3", { x: 0.52, y: 0.5 }],
      ["c.mp3", { x: 0.95, y: 0.95 }],
    ]);
    m.viewMode = "embeddings";

    // In scope → its scope position.
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.4,
      wy: 0.6,
    });
    // In the circle but unplaced by the scope lens → hidden, not left inside.
    expect(cv.worldPosForTrack(m.trackByPath("b.mp3")!)).toBeNull();
    // Outside the circle → its normal outer position.
    expect(cv.worldPosForTrack(m.trackByPath("c.mp3")!)).toEqual({
      wx: 0.95,
      wy: 0.95,
    });
  });

  it("draws the scope outline as the projected image of its world circle", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    // Non-square viewport: the world→screen transform scales x and y differently.
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    m.viewMode = "embeddings";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    m.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
    m.scopePositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);

    cv.draw();

    // The scope outline must be the world circle's screen image — an ellipse
    // whose axes are the world radius scaled by each axis — so nothing outside
    // the world circle can render inside it.
    const outline = ellipses.find((e) => e.rx > 1);
    expect(outline).toBeDefined();
    const iw = 800 - 2 * 40;
    const ih = 400 - 2 * 40;
    expect(outline!.rx).toBeCloseTo(0.2 * iw, 3);
    expect(outline!.ry).toBeCloseTo(0.2 * ih, 3);
    expect(outline!.rx / outline!.ry).toBeCloseTo(iw / ih, 3);
  });

  it("draws a box scope as a rectangle matching its world extent", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    m.viewMode = "embeddings";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    m.scope = {
      shape: "box",
      cx: 0.5,
      cy: 0.5,
      r: 0.2,
      rx: 0.2,
      ry: 0.1,
      lensId: "camelot",
    };
    m.scopePositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);

    cv.draw();

    // The box region's screen image is a rectangle: half-width rx scaled by the
    // x axis, half-height ry by the y axis — no ellipse.
    const iw = 800 - 2 * 40;
    const ih = 400 - 2 * 40;
    const box = rects.find((r) => r.w > 1);
    expect(box).toBeDefined();
    expect(box!.w).toBeCloseTo(2 * 0.2 * iw, 3);
    expect(box!.h).toBeCloseTo(2 * 0.1 * ih, 3);
    expect(ellipses.some((e) => e.rx > 1)).toBe(false);
  });
});

// ─── Loading indicators (stacked) ─────────────────────────────

describe("CanvasView loading indicators", () => {
  let texts: string[];
  let arcs: { a0: number; a1: number }[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    texts = [];
    arcs = [];
    const target: Record<string, unknown> = {
      measureText: () => ({ width: 10 }),
      fillText: (t: string) => {
        texts.push(t);
      },
      arc: (_x: number, _y: number, _r: number, a0: number, a1: number) => {
        arcs.push({ a0, a1 });
      },
      roundRect: () => undefined,
    };
    const ctx = new Proxy(target, {
      get(t, prop: string) {
        if (prop in t) return t[prop];
        return () => undefined;
      },
      set(t, prop: string, value) {
        t[prop] = value;
        return true;
      },
    });
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function kind(over: Partial<QueueKindStatus> = {}): QueueKindStatus {
    return {
      label: "MAEST embeddings",
      pending: 0,
      running: 1,
      done: 0,
      failed: 0,
      total: 10,
      finished: 1,
      outstanding: 1,
      ...over,
    };
  }

  function makeCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "clap";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    return new CanvasView(m);
  }

  it("stacks the layout wait under the analysis progress", () => {
    const cv = makeCanvas();
    const m = cv.model;
    m.queueStatus = {
      kinds: { maest: kind() },
      order: ["maest"],
      outstanding: 1,
      total: 10,
      finished: 1,
      active: true,
      epoch: 1,
    } satisfies QueueStatus;
    m.projectionPending = true;
    m.layoutProgress = { phase: "layout", done: 0, total: 0 };
    cv.draw();
    expect(texts.some((t) => t.includes("MAEST embeddings"))).toBe(true);
    expect(texts.some((t) => t.includes("Loading CLAP timbre"))).toBe(true);
  });

  it("draws a rotating throbber while the layout is in flight", () => {
    const cv = makeCanvas();
    cv.model.projectionPending = true;
    cv.model.layoutProgress = { phase: "layout", done: 0, total: 0 };
    cv.draw();
    // A full ring plus a partial highlight arc.
    expect(arcs.some((a) => a.a1 - a.a0 < Math.PI * 2 - 1e-6)).toBe(true);
  });

  it("hides the layout wait once positions are ready", () => {
    const cv = makeCanvas();
    cv.model.embeddingsReady = true;
    cv.draw();
    expect(texts.some((t) => t.includes("Loading"))).toBe(false);
  });
});

// ─── Hovered-row info card ───────────────────────────────────

describe("CanvasView hovered-row info card", () => {
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      makeFakeCtx().ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function makeCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3", title: "Song", artist: "A" })]);
    m.lensId = "clap";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    return new CanvasView(m);
  }

  it("shows the info card for a hovered search-result track", () => {
    const cv = makeCanvas();
    cv.hoveredPath = "a.mp3";
    cv.draw();
    expect(cv.$tip.classList.contains("hidden")).toBe(false);
    expect(cv.$tip.textContent).toContain("Song");
  });

  it("hides the card when no row or dot is hovered", () => {
    const cv = makeCanvas();
    cv.draw();
    expect(cv.$tip.classList.contains("hidden")).toBe(true);
  });

  it("routes an inspect-card tab click to onInspectAction", () => {
    const cv = makeCanvas();
    cv.inspect = {
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
    };
    cv.onInspectAction = vi.fn();
    const btn = document.createElement("button");
    btn.dataset.inspectAction = "show-describe";
    cv.$tip.appendChild(btn);

    btn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(cv.onInspectAction).toHaveBeenCalledWith("show-describe", "a.mp3");
  });

  it("routes a Describe-term click to onSearchTerm", () => {
    const cv = makeCanvas();
    cv.onSearchTerm = vi.fn();
    const row = document.createElement("div");
    row.dataset.searchTerm = "a techno track";
    // Term rows live in the folded-out panel, not the card.
    cv.$coords.appendChild(row);

    row.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(cv.onSearchTerm).toHaveBeenCalledWith("a techno track");
  });

  it("routes a seekbar click to onSeek with the clicked fraction", () => {
    const cv = makeCanvas();
    cv.inspect = {
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
    };
    cv.onSeek = vi.fn();
    const track = document.createElement("div");
    track.dataset.seek = "";
    track.getBoundingClientRect = () => ({ left: 100, width: 200 }) as DOMRect;
    cv.$tip.appendChild(track);

    track.dispatchEvent(
      new MouseEvent("mousedown", { clientX: 150, bubbles: true }),
    );
    expect(cv.onSeek).toHaveBeenCalledWith("a.mp3", 0.25);
    // The drag listens on the document; end it so it cannot leak between tests.
    document.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
  });

  it("folds the Describe content into a panel beside the card", () => {
    const cv = makeCanvas();
    cv.inspect = {
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
        coordinates: [
          {
            axis: "dark <-> bright",
            pos: "dark",
            neg: "bright",
            pole: "dark",
            z: 2.1,
            percentile: 0.98,
            reliability: 2.58,
          },
        ],
      },
    };
    cv.draw();
    expect(cv.$coords.classList.contains("hidden")).toBe(false);
    // One wide area, two columns: terms | coordinates.
    const panel = cv.$coords.querySelector(".tt-desc-panel");
    expect(panel?.querySelector(".tt-desc-col")).not.toBeNull();
    expect(panel?.querySelector(".tt-coord-col")).not.toBeNull();
    expect(cv.$coords.querySelector(".tt-describe-row")).not.toBeNull();
    // The card itself carries none of it on this tab.
    expect(cv.$tip.querySelector(".tt-describe-row")).toBeNull();
    expect(cv.$tip.querySelector(".tt-coord")).toBeNull();
  });

  it("hides the Describe panel while the Navigate tab is active", () => {
    const cv = makeCanvas();
    cv.inspect = {
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
        results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
        coordinates: [],
      },
    };
    cv.draw();
    expect(cv.$coords.classList.contains("hidden")).toBe(true);
  });

  it("keeps the Describe panel DOM across an unchanged redraw", () => {
    const cv = makeCanvas();
    cv.inspect = {
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
        coordinates: [],
      },
    };
    cv.draw();
    const first = cv.$coords.querySelector(".tt-desc-panel");
    expect(first).not.toBeNull();
    // A pan/zoom frame redraws, but must not replace the rows under the pointer.
    cv.draw();
    expect(cv.$coords.querySelector(".tt-desc-panel")).toBe(first);
  });
});

describe("nearestInBlob", () => {
  const at = (idx: number, sx: number, sy: number): DotPosition => ({
    idx,
    sx,
    sy,
  });

  function grid(points: DotPosition[]): Map<string, DotPosition[]> {
    const g = new Map<string, DotPosition[]>();
    for (const p of points) {
      const key = `${Math.round(p.sx / 24)},${Math.round(p.sy / 24)}`;
      const cell = g.get(key);
      if (cell) cell.push(p);
      else g.set(key, [p]);
    }
    return g;
  }

  it("picks the nearest member when the cursor sits in a blob", () => {
    const g = grid([at(0, 100, 100), at(1, 110, 100), at(2, 100, 110)]);
    // 30 px from the cluster, still inside the 3×3 cell reach.
    expect(nearestInBlob(g, 130, 100)).toBe(1);
  });

  it("returns nothing for a lone dot's cell", () => {
    const g = grid([at(0, 100, 100)]);
    expect(nearestInBlob(g, 130, 100)).toBe(-1);
  });

  it("returns nothing when no cell is near the cursor", () => {
    const g = grid([at(0, 100, 100), at(1, 110, 100)]);
    expect(nearestInBlob(g, 400, 400)).toBe(-1);
  });
});

describe("CanvasView track colour modes", () => {
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeFakeCtx();
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function view(track: Track): CanvasView {
    const m = new Model();
    m.setAllTracks([track]);
    return new CanvasView(m);
  }

  it("colours by tempo and key from cached features", () => {
    const t = makeTrack({
      path: "a.mp3",
      audio_features: {
        tempo: 0.5,
        key: 0.25,
        mode: 1,
        energy: 0.5,
        danceability: 0.5,
      },
    });
    const cv = view(t);
    cv.model.colorMode = "tempo";
    expect(cv.trackColor(t)).toBe(tempoColor(0.5));
    cv.model.colorMode = "key";
    expect(cv.trackColor(t)).toBe(keyColor(0.25, 1));
  });

  it("falls back to the folder colour when the feature is missing", () => {
    const t = makeTrack({ path: "a.mp3", folder: "House" });
    const cv = view(t);
    cv.model.colorMode = "tempo";
    expect(cv.trackColor(t)).toBe(dirColor("House"));
    cv.model.colorMode = "key";
    expect(cv.trackColor(t)).toBe(dirColor("House"));
  });

  it("colours by cluster when assignments exist", () => {
    const t = makeTrack({ path: "a.mp3", folder: "House" });
    const cv = view(t);
    cv.model.colorMode = "cluster";
    cv.model.clusterAssignments = new Map([["a.mp3", 4]]);
    expect(cv.trackColor(t)).toBe(clusterColor(4));
  });

  it("accepts an explicit colour mode for the selection bloom", () => {
    const t = makeTrack({ path: "a.mp3", folder: "House" });
    const cv = view(t);
    cv.model.clusterAssignments = new Map([["a.mp3", 4]]);
    cv.model.colorMode = "folder";
    expect(cv.trackColor(t, "cluster")).toBe(clusterColor(4));
  });

  it("colours by the head's predicted folder, gated by confidence", () => {
    const t = makeTrack({ path: "a.mp3", folder: "House" });
    const cv = view(t);
    cv.model.colorMode = "head";
    cv.model.folderPredictions = new Map([["a.mp3", "Root/Techno"]]);
    // Confident (default): a weak match is grey, not its own folder colour.
    cv.model.folderPredictionMatches = new Set();
    expect(cv.trackColor(t)).toBe(CONFIDENT_MISS_COLOR);
    // A confident match takes the predicted folder's colour.
    cv.model.folderPredictionMatches = new Set(["a.mp3"]);
    expect(cv.trackColor(t)).toBe(dirColor("Root/Techno"));
    // "All nearest" drops the gate, so the prediction wins regardless.
    cv.model.headColorConfident = false;
    cv.model.folderPredictionMatches = new Set();
    expect(cv.trackColor(t)).toBe(dirColor("Root/Techno"));
  });

  it("greys only a track the head has no prediction for", () => {
    const t = makeTrack({ path: "a.mp3", folder: "House" });
    const cv = view(t);
    cv.model.colorMode = "head";
    cv.model.folderPredictions = new Map();
    cv.model.folderPredictionMatches = new Set();
    expect(cv.trackColor(t)).toBe(CONFIDENT_MISS_COLOR);
    cv.model.headColorConfident = false;
    expect(cv.trackColor(t)).toBe(CONFIDENT_MISS_COLOR);

    // A prediction equal to the current folder still takes that folder's colour.
    cv.model.folderPredictions = new Map([["a.mp3", "House"]]);
    cv.model.folderPredictionMatches = new Set(["a.mp3"]);
    expect(cv.trackColor(t)).toBe(dirColor("House"));
  });
});

describe("CanvasView top tabbers", () => {
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeFakeCtx();
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("draws the lens and colour tabbers and hit-tests both", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });

    cv.draw();

    expect(cv.lensTabs.length).toBeGreaterThan(0);
    expect(cv.colorTabs.length).toBeGreaterThan(0);
    // The lens tabber is the top row, the colour tabber sits below it.
    expect(cv.lensTabs[0].y1).toBeLessThan(cv.colorTabs[0].y0);
    const lens = cv.lensTabs[0];
    const lensCx = (lens.x0 + lens.x1) / 2;
    expect(cv.topLensTabAt(lensCx, (lens.y0 + lens.y1) / 2)).toBe(lens.id);
    const color = cv.colorTabs[2];
    const colorCx = (color.x0 + color.x1) / 2;
    expect(cv.topColorTabAt(colorCx, (color.y0 + color.y1) / 2)).toBe(color.id);
    // The gap between the two rows belongs to neither.
    const gap = (cv.lensTabs[0].y1 + cv.colorTabs[0].y0) / 2;
    expect(cv.topLensTabAt(lensCx, gap)).toBeNull();
    expect(cv.topColorTabAt(lensCx, gap)).toBeNull();
  });
});

describe("CanvasView volume HUD", () => {
  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
  });

  afterEach(() => {
    document.body.innerHTML = "";
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders a labelled progress bar for the preview volume", () => {
    const m = new Model();
    const cv = new CanvasView(m);
    m.previewVolume = 0.4;
    cv.showVolumeHud(100, 100);
    const hud = document.getElementById("volume-hud")!;
    expect(hud.classList.contains("hidden")).toBe(false);
    expect(hud.querySelector(".volume-hud-label")?.textContent).toBe(
      "Volume 40%",
    );
    expect(
      (hud.querySelector(".volume-hud-track > i") as HTMLElement).style.width,
    ).toBe("40%");
  });

  it("renders the bloom range with its per-variable meaning", () => {
    const m = new Model();
    const cv = new CanvasView(m);
    m.selectionBloom = "tempo";
    m.setBloomRange(0.6);
    cv.showBloomHud(100, 100);
    const hud = document.getElementById("bloom-hud")!;
    expect(hud.classList.contains("hidden")).toBe(false);
    // 0.6 of a full range is ±15% BPM.
    expect(hud.querySelector(".bloom-hud-label")?.textContent).toBe(
      "Bloom · ±15% BPM",
    );
    expect(
      (hud.querySelector(".bloom-hud-track > i") as HTMLElement).style.width,
    ).toBe("60%");
  });

  it("omits the slider when the bloom variable has no range", () => {
    const m = new Model();
    const cv = new CanvasView(m);
    m.selectionBloom = "cluster";
    cv.showBloomHud(100, 100);
    const hud = document.getElementById("bloom-hud")!;
    expect(hud.classList.contains("hidden")).toBe(false);
    expect(hud.querySelector(".bloom-hud-label")?.textContent).toBe(
      "Bloom · no range",
    );
    expect(hud.querySelector(".bloom-hud-track")).toBeNull();
  });
});

describe("tabAt", () => {
  it("returns the id inside a rect and null outside", () => {
    const tabs = [{ id: "x", x0: 10, y0: 10, x1: 20, y1: 20 }];
    expect(tabAt(tabs, 15, 15)).toBe("x");
    expect(tabAt(tabs, 25, 15)).toBeNull();
  });
});
