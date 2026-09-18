import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Model,
  type QueueKindStatus,
  type QueueStatus,
  type Track,
} from "../model";
import { clusterColor } from "../lib/colors";
import { CanvasView } from "./canvas-view";

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

/** Minimal 2-D context stand-in that records every fill colour. */
function makeFakeCtx(): { ctx: CanvasRenderingContext2D; fills: string[] } {
  const fills: string[] = [];
  const target: Record<string, unknown> = {
    measureText: () => ({ width: 10 }),
  };
  const ctx = new Proxy(target, {
    get(t, prop: string) {
      if (prop === "fillStyle") return fills.at(-1) ?? "";
      if (prop in t) return t[prop];
      return () => undefined;
    },
    set(t, prop: string, value) {
      if (prop === "fillStyle") fills.push(value as string);
      else t[prop] = value;
      return true;
    },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, fills };
}

describe("CanvasView cluster hover glow", () => {
  let fills: string[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div>';
    const fake = makeFakeCtx();
    fills = fake.fills;
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

  it("glows a hovered cluster's members in its swatch colour", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.hoverTarget = { kind: "cluster", id: 2 };
    cv.draw();
    expect(fills).toContain(clusterColor(2));
  });

  it("does not glow a cluster colour when nothing is hovered", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.draw();
    expect(fills).not.toContain(clusterColor(2));
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
});

describe("CanvasView search field", () => {
  let gradients: number;
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div>';
    gradients = 0;
    const target: Record<string, unknown> = {
      measureText: () => ({ width: 10 }),
      createRadialGradient: () => {
        gradients += 1;
        return { addColorStop: () => undefined };
      },
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

  function makeCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "clap";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    return new CanvasView(m);
  }

  it("splats a landmark per sample point while the box is active", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.draw();
    expect(gradients).toBe(1);
  });

  it("hides the field while the search box is not active", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = false;
    cv.draw();
    expect(gradients).toBe(0);
  });

  it("hides the field off the CLAP lens", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.model.lensId = "rhythm";
    cv.draw();
    expect(gradients).toBe(0);
  });

  it("shows the field under a CLAP scope", () => {
    const cv = makeCanvas();
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.model.lensId = "combined";
    cv.model.scope = { cx: 0.5, cy: 0.5, r: 0.2, lensId: "clap" };
    cv.model.scopePositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    cv.draw();
    expect(gradients).toBe(1);
  });
});

describe("CanvasView scope positions", () => {
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div>';
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      makeFakeCtx().ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
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
    m.scope = { cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
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
});

// ─── Loading indicators (stacked) ─────────────────────────────

describe("CanvasView loading indicators", () => {
  let texts: string[];
  let arcs: { a0: number; a1: number }[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div>';
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
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div>';
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
});
