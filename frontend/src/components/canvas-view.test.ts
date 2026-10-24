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
  depthShade,
  GLASS_BLUR_PX,
  GLASS_TINT,
  GLASS_TINT_FLAT,
  nearestInBlob,
  readGlassBlur,
  rimPush,
  tabAt,
  wormhole,
  type DotPosition,
} from "./canvas-view";
import type { GlowOccluder, GlowPoint } from "../lib/glow-layer";
import { POINT_SPLAT_R, type PointSpec } from "../lib/point-layer";

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
  arcs: { x: number; y: number; r: number; alpha: number }[];
} {
  const alphas: number[] = [];
  const ellipses: { x: number; y: number; rx: number; ry: number }[] = [];
  const rects: { x: number; y: number; w: number; h: number }[] = [];
  const arcs: { x: number; y: number; r: number; alpha: number }[] = [];
  let currentAlpha = 1;
  const target: Record<string, unknown> = {
    measureText: () => ({ width: 10 }),
    ellipse: (x: number, y: number, rx: number, ry: number) => {
      ellipses.push({ x, y, rx, ry });
    },
    rect: (x: number, y: number, w: number, h: number) => {
      rects.push({ x, y, w, h });
    },
    arc: (x: number, y: number, r: number) => {
      arcs.push({ x, y, r, alpha: currentAlpha });
    },
  };
  const ctx = new Proxy(target, {
    get(t, prop: string) {
      if (prop in t) return t[prop];
      return () => undefined;
    },
    set(t, prop: string, value) {
      if (prop === "globalAlpha") {
        currentAlpha = value as number;
        alphas.push(value as number);
      } else t[prop] = value;
      return true;
    },
  });
  return {
    ctx: ctx as unknown as CanvasRenderingContext2D,
    alphas,
    ellipses,
    rects,
    arcs,
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

  it("coalesces scheduleDraw, so concurrent animation loops cannot cancel each other's draw", () => {
    const callbacks: FrameRequestCallback[] = [];
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      callbacks.push(cb);
      return callbacks.length;
    });
    vi.stubGlobal("cancelAnimationFrame", () => {});
    const cv = makeCanvas();
    const draw = vi.spyOn(cv, "draw").mockImplementation(() => {});

    cv.scheduleDraw();
    cv.scheduleDraw();
    cv.scheduleDraw();
    expect(callbacks).toHaveLength(1);
    callbacks[0](0);
    expect(draw).toHaveBeenCalledTimes(1);

    // A request after the draw ran schedules a fresh frame.
    cv.scheduleDraw();
    expect(callbacks).toHaveLength(2);
  });

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

// ─── 3-D glow compositing (occlusion + blit) ─────────────────

describe("CanvasView 3-D glow compositing", () => {
  let drawImages: unknown[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    drawImages = [];
    const target: Record<string, unknown> = {
      measureText: () => ({ width: 10 }),
      drawImage: (img: unknown) => {
        drawImages.push(img);
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

  /**
   * A 3-D canvas whose WebGL layers are fakes that record their render calls.
   * WebGL is unavailable in happy-dom, so the fakes keep the view's DOM-facing
   * surface — the glow canvas the visibility toggle touches.
   */
  function make3dCanvas(): {
    cv: CanvasView;
    glow: { render: ReturnType<typeof vi.fn>; canvas: { style: { visibility: string } } };
    points: {
      supported: true;
      render: ReturnType<typeof vi.fn>;
      canvas: Record<string, never>;
    };
  } {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", folder: "House" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([
      ["a.mp3", { x: 0.3, y: 0.4, z: 0.8 }],
      ["b.mp3", { x: 0.6, y: 0.5, z: 0.2 }],
    ]);
    const cv = new CanvasView(m);
    const glow = { render: vi.fn(), canvas: { style: { visibility: "" } } };
    const points = { supported: true as const, render: vi.fn(), canvas: {} };
    (cv as unknown as { glow: typeof glow; points: typeof points }).glow = glow;
    (cv as unknown as { glow: typeof glow; points: typeof points }).points = points;
    return { cv, glow, points };
  }

  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  it("occludes the 3-D glow with the blobs and the crisp dots", () => {
    const { cv, glow } = make3dCanvas();
    // a.mp3 is selected → crisp dot; b.mp3 stays in the merged field. Both get
    // a hover halo, so the pass actually draws something.
    cv.model.clusterAssignments = new Map([
      ["a.mp3", 2],
      ["b.mp3", 2],
    ]);
    cv.hoverTarget = { kind: "cluster", id: 2 };
    cv.model.selected.add("a.mp3");
    cv.draw();

    expect(glow.render).toHaveBeenCalledTimes(1);
    const [glowPoints, occluders] = glow.render.mock.calls[0] as [
      GlowPoint[],
      GlowOccluder[] | undefined,
    ];
    expect(glowPoints.length).toBeGreaterThan(0);
    expect(occluders).toBeDefined();
    const ts = trackScreen(cv);
    const pa = ts({ wx: 0.3, wy: 0.4, wz: 0.8 });
    const pb = ts({ wx: 0.6, wy: 0.5, wz: 0.2 });
    const crisp = occluders!.find((o) => o.depth === pa.depth)!;
    const splat = occluders!.find((o) => o.depth === pb.depth)!;
    // The crisp dot is a DOT_R_SEL disc at its own depth; the field splat keeps
    // its perspective-scaled radius, so the occlusion footprint matches the blob.
    expect(crisp.radius).toBe(6);
    expect(crisp.depth).toBe(pa.depth);
    expect(splat.radius).toBeCloseTo(POINT_SPLAT_R * pb.scale, 5);
    expect(splat.depth).toBe(pb.depth);
  });

  it("blits blobs, then glow, then draws the crisp dots on top in 3-D", () => {
    const { cv, glow, points } = make3dCanvas();
    cv.model.selected.add("a.mp3");
    cv.draw();
    // The blit order is the composition order: blobs under glow under dots.
    expect(drawImages[0]).toBe(points.canvas);
    expect(drawImages[1]).toBe(glow.canvas);
  });

  it("passes no occluders and keeps the DOM stack in 2-D", () => {
    const { cv, glow, points } = make3dCanvas();
    cv.model.view3d.on = false;
    cv.model.view3dBlend = 0;
    cv.draw();
    expect(glow.render).toHaveBeenCalledTimes(1);
    // One argument — the glow points — with no occluders.
    expect(glow.render.mock.calls[0]).toHaveLength(1);
    // Only the metaball canvas is blitted; the glow stays on its stacked canvas.
    expect(drawImages).toEqual([points.canvas]);
    expect(glow.canvas.style.visibility).toBe("");
  });

  it("hides the glow canvas in 3-D and restores it in 2-D", () => {
    const { cv, glow } = make3dCanvas();
    cv.draw();
    expect(glow.canvas.style.visibility).toBe("hidden");
    cv.model.view3d.on = false;
    cv.model.view3dBlend = 0;
    cv.draw();
    expect(glow.canvas.style.visibility).toBe("");
  });
});

// ─── Glow depth (halo occlusion input) ──────────────────────

describe("CanvasView glow depth", () => {
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

  function make3dCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", folder: "House" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([
      ["a.mp3", { x: 0.3, y: 0.4, z: 0.8 }],
      ["b.mp3", { x: 0.6, y: 0.5, z: 0.2 }],
    ]);
    return new CanvasView(m);
  }

  it("carries the dots' depth in the hover halo", () => {
    const cv = make3dCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.hoverTarget = { kind: "cluster", id: 2 };
    cv.draw();
    const pts = cv.collectHoverGlowPoints(cv.model.tracks);
    expect(pts).toHaveLength(1);
    const pos = cv.positions.find((p) => p.idx === 0)!;
    expect(pts[0].depth).toBe(pos.depth);
    expect(pts[0].depth).toBeGreaterThanOrEqual(0.05);
  });

  it("keeps the hover halo at depth 1 in 2-D", () => {
    const cv = make3dCanvas();
    cv.model.view3d.on = false;
    cv.model.view3dBlend = 0;
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.hoverTarget = { kind: "cluster", id: 2 };
    cv.draw();
    const pts = cv.collectHoverGlowPoints(cv.model.tracks);
    expect(pts[0].depth).toBe(1);
  });

  it("carries the camera depth in the search-field glow", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "clap";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5, z: 0.5 }]]);
    const cv = new CanvasView(m);
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.draw();
    const pts = cv.collectSearchGlowPoints();
    expect(pts).toHaveLength(1);
    const ts = (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
    // The glow carries the same depth the camera projects for that track.
    expect(pts[0].depth).toBe(ts({ wx: 0.5, wy: 0.5, wz: 0.5 }).depth);
  });

  it("keeps the search-field glow at depth 1 in 2-D", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "clap";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    const cv = new CanvasView(m);
    cv.searchField = [{ path: "a.mp3", score: 0.5 }];
    cv.searchActive = true;
    cv.draw();
    const pts = cv.collectSearchGlowPoints();
    expect(pts[0].depth).toBe(1);
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

// ─── 3-D sphere scopes ────────────────────────────────────────

describe("CanvasView sphere scope", () => {
  let arcs: { x: number; y: number; r: number; alpha: number }[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeFakeCtx();
    arcs = fake.arcs;
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  it("returns the scope position's z for an in-scope track", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.viewMode = "embeddings";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.6,
      r: 0.2,
      lensId: "camelot",
    };
    m.scopePositions = new Map([["a.mp3", { x: 0.4, y: 0.6, z: 0.7 }]]);
    const cv = new CanvasView(m);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.4,
      wy: 0.6,
      wz: 0.7,
    });
  });

  it("tests the sphere hide-in-region branch with the outer z", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
    ]);
    m.viewMode = "embeddings";
    // Both sit at the sphere's xy centre; only b's z falls inside the sphere.
    m.embeddingPositions = new Map([
      ["a.mp3", { x: 0.5, y: 0.5, z: 0.1 }],
      ["b.mp3", { x: 0.5, y: 0.5, z: 0.55 }],
    ]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.2,
      lensId: "camelot",
    };
    const cv = new CanvasView(m);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.5,
      wy: 0.5,
      wz: 0.1,
    });
    expect(cv.worldPosForTrack(m.trackByPath("b.mp3")!)).toBeNull();
  });

  it("draws a sphere scope as a screen circle with a faint inner ring", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5, z: 0.5 }]]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.1,
      lensId: "camelot",
    };
    m.scopePositions = new Map([["a.mp3", { x: 0.5, y: 0.5, z: 0.5 }]]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });

    cv.draw();

    // The sphere's screen circle: the outer ring at the projected radius plus
    // the faint inner ring at half radius.
    const ring = arcs.filter((a) => a.r > 10);
    expect(ring).toHaveLength(2);
    const c = trackScreen(cv)({ wx: 0.5, wy: 0.5, wz: 0.5 });
    const [outer, inner] = ring;
    // 0.1 world radius at the centre plane and zoom 1 → 0.1 * 720 px.
    expect(outer.r).toBeCloseTo(0.1 * 720, 3);
    expect(outer.x).toBeCloseTo(c.sx, 3);
    expect(outer.y).toBeCloseTo(c.sy, 3);
    expect(inner.r).toBeCloseTo(outer.r * 0.5, 3);
    expect(inner.alpha).toBeCloseTo(0.35, 5);
  });
});

// ─── Sphere-scope wormhole (pure) ────────────────────────────

describe("wormhole", () => {
  const circle = { cx: 400, cy: 200, r: 100 };
  const centreDepth = 0.525;
  const depthReach = 0.05;
  // Deep enough in front of the centre plane that s saturates at 1.
  const frontDepth = centreDepth + 2 * depthReach;

  it("leaves points at or behind the centre plane untouched", () => {
    const at = wormhole(
      { sx: 420, sy: 200, depth: centreDepth },
      circle,
      centreDepth,
      depthReach,
    );
    expect(at).toEqual({ sx: 420, sy: 200, ghost: 0 });
    const behind = wormhole(
      { sx: 420, sy: 200, depth: centreDepth - 0.2 },
      circle,
      centreDepth,
      depthReach,
    );
    expect(behind).toEqual({ sx: 420, sy: 200, ghost: 0 });
  });

  it("pushes a fully-influenced point at the circle centre out to the rim", () => {
    const res = wormhole(
      { sx: 400, sy: 200, depth: frontDepth },
      circle,
      centreDepth,
      depthReach,
    );
    // d < 1e-3 picks the fixed +x direction, so the result lands exactly r right.
    expect(res.sx).toBeCloseTo(400 + 100, 5);
    expect(res.sy).toBeCloseTo(200, 5);
    expect(res.ghost).toBeCloseTo(1, 5);
  });

  it("leaves points at or beyond the reach radius untouched", () => {
    const Rinf = 100 * 1.7;
    const atRinf = wormhole(
      { sx: 400 + Rinf, sy: 200, depth: frontDepth },
      circle,
      centreDepth,
      depthReach,
    );
    expect(atRinf).toEqual({ sx: 400 + Rinf, sy: 200, ghost: 0 });
    const beyond = wormhole(
      { sx: 600, sy: 200, depth: frontDepth },
      circle,
      centreDepth,
      depthReach,
    );
    expect(beyond).toEqual({ sx: 600, sy: 200, ghost: 0 });
  });

  it("displaces monotonically and continuously out to the reach radius", () => {
    const distAt = (d: number) => {
      const res = wormhole(
        { sx: 400 + d, sy: 200, depth: frontDepth },
        circle,
        centreDepth,
        depthReach,
      );
      return Math.hypot(res.sx - 400, res.sy - 200);
    };
    expect(distAt(0)).toBeCloseTo(100, 5);
    const ds = [0, 20, 60, 100, 120, 169.999, 170];
    for (let i = 1; i < ds.length; i++) {
      expect(distAt(ds[i])).toBeGreaterThan(distAt(ds[i - 1]));
    }
    expect(distAt(170)).toBeCloseTo(170, 5);
    // Continuous at the reach edge: the last inside point lands ~on the rim.
    expect(Math.abs(distAt(169.999) - 170)).toBeLessThan(0.01);
  });

  it("displaces radially outward along the ray from the centre", () => {
    // A point at 45° inside the circle is pushed further out on the same ray.
    const res = wormhole(
      { sx: 420, sy: 220, depth: frontDepth },
      circle,
      centreDepth,
      depthReach,
    );
    const dx = res.sx - 400;
    const dy = res.sy - 200;
    expect(dx).toBeGreaterThan(20);
    expect(dy).toBeGreaterThan(20);
    expect(dx / 20).toBeCloseTo(dy / 20, 5);
    expect(Math.hypot(dx, dy)).toBeGreaterThan(Math.hypot(20, 20));
  });
});

// ─── Sphere-scope glass rim push (pure) ──────────────────────

describe("rimPush", () => {
  const circle = { cx: 400, cy: 200, r: 100 };

  it("maps inside points onto the inner rim band", () => {
    // The centre point goes to the fixed +x direction at (1 − RIM_BAND)·r.
    const centre = rimPush({ sx: 400, sy: 200 }, circle);
    expect(centre).toEqual({ sx: 400 + 82, sy: 200 });
    for (const d of [10, 50, 90, 99.9]) {
      const p = rimPush({ sx: 400 + d, sy: 200 }, circle);
      const dd = Math.hypot(p.sx - 400, p.sy - 200);
      expect(dd).toBeGreaterThanOrEqual(0.82 * 100);
      expect(dd).toBeLessThanOrEqual(0.964 * 100);
    }
  });

  it("is monotone in the raw distance", () => {
    const distAt = (d: number) => {
      const res = rimPush({ sx: 400 + d, sy: 200 }, circle);
      return Math.hypot(res.sx - 400, res.sy - 200);
    };
    expect(distAt(0)).toBeCloseTo(82, 5);
    const ds = [0, 10, 50, 90, 99.9];
    for (let i = 1; i < ds.length; i++) {
      expect(distAt(ds[i])).toBeGreaterThan(distAt(ds[i - 1]));
    }
    // The band endpoints: [0.82r, 0.964r] across the inside of the circle.
    expect(distAt(99.9)).toBeCloseTo(100 * (0.82 + 0.18 * 0.8 * 0.999), 5);
  });

  it("pushes radially outward along the ray from the centre", () => {
    const res = rimPush({ sx: 420, sy: 220 }, circle);
    const dx = res.sx - 400;
    const dy = res.sy - 200;
    expect(dx).toBeGreaterThan(20);
    expect(dy).toBeGreaterThan(20);
    expect(dx / 20).toBeCloseTo(dy / 20, 5);
    expect(Math.hypot(dx, dy)).toBeGreaterThan(Math.hypot(20, 20));
  });

  it("leaves points at or beyond the circle unchanged", () => {
    expect(rimPush({ sx: 500, sy: 200 }, circle)).toEqual({ sx: 500, sy: 200 });
    expect(rimPush({ sx: 600, sy: 200 }, circle)).toEqual({ sx: 600, sy: 200 });
  });
});

// ─── Sphere-scope glass blur reader (pure) ───────────────────

describe("readGlassBlur", () => {
  it("defaults to the blur preset and reads the glass param", () => {
    expect(readGlassBlur("")).toBe(GLASS_BLUR_PX);
    expect(readGlassBlur("?foo=1")).toBe(GLASS_BLUR_PX);
    expect(readGlassBlur("?glassx=12")).toBe(GLASS_BLUR_PX);
    expect(readGlassBlur("?glass=0")).toBe(0);
    expect(readGlassBlur("?glass=12")).toBe(12);
    expect(readGlassBlur("glass=12")).toBe(12);
    expect(readGlassBlur("?foo=1&glass=3")).toBe(3);
  });

  it("falls back on junk and clamps to [0, 24]", () => {
    expect(readGlassBlur("?glass=abc")).toBe(GLASS_BLUR_PX);
    expect(readGlassBlur("?glass=")).toBe(GLASS_BLUR_PX);
    expect(readGlassBlur("?glass=-3")).toBe(0);
    expect(readGlassBlur("?glass=100")).toBe(24);
    expect(readGlassBlur("?glass=24")).toBe(24);
  });
});

// ─── Sphere-scope billboard + wormhole (drawing) ─────────────

describe("CanvasView sphere scope billboard + wormhole", () => {
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

  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
        billboard?: { cx: number; cy: number; cz: number };
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  /** A 3-D canvas whose WebGL layers are fakes that record their render calls. */
  function make3dCanvas(): {
    cv: CanvasView;
    points: {
      supported: true;
      render: ReturnType<typeof vi.fn>;
      canvas: Record<string, never>;
    };
  } {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "front.mp3" }),
      makeTrack({ path: "behind.mp3" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([
      ["front.mp3", { x: 0.5, y: 0.5, z: 0.8 }],
      ["behind.mp3", { x: 0.5, y: 0.5, z: 0.1 }],
    ]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.25,
      lensId: "camelot",
    };
    const cv = new CanvasView(m);
    const glow = { render: vi.fn(), canvas: { style: { visibility: "" } } };
    const points = { supported: true as const, render: vi.fn(), canvas: {} };
    (cv as unknown as { glow: typeof glow; points: typeof points }).glow = glow;
    (cv as unknown as { glow: typeof glow; points: typeof points }).points =
      points;
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    return { cv, points };
  }

  it("billboards scope members: pinned to the sphere at any orbit, unlike a non-member", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "member.mp3" }),
      makeTrack({ path: "other.mp3" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    // The non-member sits outside the sphere (0.412 > r) so it stays visible.
    m.embeddingPositions = new Map([
      ["member.mp3", { x: 0.5, y: 0.5 }],
      ["other.mp3", { x: 0.6, y: 0.5, z: 0.9 }],
    ]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.2,
      lensId: "camelot",
    };
    // The member's scope position is a flat {x, y} with no z: a billboard at the
    // sphere centre, carried by the position itself.
    m.scopePositions = new Map([
      ["member.mp3", { x: 0.6, y: 0.5, billboard: { cx: 0.5, cy: 0.5, cz: 0.5 } }],
    ]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    const member = m.trackByPath("member.mp3")!;
    const other = m.trackByPath("other.mp3")!;

    // Default camera: the member resolves to the sphere centre + flat offset at
    // the centre's perspective scale (the centre is the cube centre → scale 1).
    expect(cv.worldPosForTrack(member)).toEqual({ wx: 0.6, wy: 0.5 });
    const otherBefore = cv.worldPosForTrack(other);
    expect(otherBefore).not.toEqual({ wx: 0.6, wy: 0.5 });

    // Orbiting leaves the member pinned to the sphere; the non-member moves.
    m.view3d.yaw = 1.2;
    m.view3d.pitch = 0.1;
    expect(cv.worldPosForTrack(member)).toEqual({ wx: 0.6, wy: 0.5 });
    const otherAfter = cv.worldPosForTrack(other);
    expect(otherAfter).not.toEqual(otherBefore);

    // On screen the member sits at the projected centre + (x − cx, y − cy)·scale
    // (scale 1 here), i.e. fixed on the billboarded disc, at both orbits.
    const ts = trackScreen(cv);
    const centre = ts({ wx: 0.5, wy: 0.5, wz: 0.5 });
    const memberScreen = ts({
      wx: 0.6,
      wy: 0.5,
      billboard: { cx: 0.5, cy: 0.5, cz: 0.5 },
    });
    expect(memberScreen.sx - centre.sx).toBeCloseTo(0.1 * 720, 5);
    expect(memberScreen.sy - centre.sy).toBeCloseTo(0, 5);
  });

  it("draws a member at the plain 2-D scope position at blend 0", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 0; // head-on orthographic: the disc is exactly the 2-D scope
    m.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.2,
      lensId: "camelot",
    };
    m.scopePositions = new Map([
      ["a.mp3", { x: 0.6, y: 0.7, billboard: { cx: 0.5, cy: 0.5, cz: 0.5 } }],
    ]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });

    const s = trackScreen(cv)({
      wx: 0.6,
      wy: 0.7,
      billboard: { cx: 0.5, cy: 0.5, cz: 0.5 },
    });
    const [px, py] = cv.w2s(0.6, 0.7);
    expect(s.sx).toBeCloseTo(px, 8);
    expect(s.sy).toBeCloseTo(py, 8);
    // The disc sits at the sphere centre's depth (centre = cube centre → 0.525).
    expect(s.depth).toBeCloseTo(0.05 + 0.95 * 0.5, 8);
  });

  it("ghosts a nearer non-member out of the circle and glasses one behind the disc", () => {
    const { cv, points } = make3dCanvas();
    // front.mp3 projects inside the sphere's circle and is nearer the camera
    // than its centre; behind.mp3 is inside the circle but behind the plane.
    cv.draw();
    const [specs] = points.render.mock.calls[0] as [PointSpec[], boolean];
    const ghostSpec = specs.find((s) => s.alpha < 0.5)!;
    // Faded to 1 − GHOST_FADE, sunk to the depth floor, pushed out of the circle.
    expect(ghostSpec.alpha).toBeCloseTo(0.15, 5);
    expect(ghostSpec.depth).toBeCloseTo(0.05, 5);
    expect(Math.hypot(ghostSpec.x - 400, ghostSpec.y - 200)).toBeGreaterThan(
      0.25 * 720,
    );
    // The behind-track projects inside the circle behind the disc: it is glassed
    // onto the inner rim band of the outer pass — visible under the glass at
    // full colour, but not hittable or selectable.
    const all = points.render.mock.calls.flatMap(
      (call) => call[0] as PointSpec[],
    );
    const glassed = all.find((s) => s.alpha === 1)!;
    const rimDist = Math.hypot(glassed.x - 400, glassed.y - 200);
    expect(rimDist).toBeGreaterThan(0.82 * 0.25 * 720);
    expect(rimDist).toBeLessThan(0.964 * 0.25 * 720);
    expect(cv.positions.map((p) => p.idx)).toEqual([]);
  });
});

// ─── Cross-frame scope animation blending ────────────────────

describe("CanvasView cross-frame scope animation blend", () => {
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

  const BB = { cx: 0.5, cy: 0.5, cz: 0.5 };

  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
        billboard?: { cx: number; cy: number; cz: number };
        from?: {
          wx: number;
          wy: number;
          wz?: number;
          billboard?: { cx: number; cy: number; cz: number };
        };
        mix?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  function viewFrame(cv: CanvasView) {
    return (cv as unknown as {
      _viewFrame(pos: {
        wx: number;
        wy: number;
        wz?: number;
        billboard?: { cx: number; cy: number; cz: number };
      }): { rx: number; ry: number; d: number; scale: number };
    })._viewFrame.bind(cv);
  }

  /** A 3-D canvas with a sphere scope at the cube centre and one member. */
  function make3d(orbit: { yaw: number; pitch: number }): {
    cv: CanvasView;
    m: Model;
  } {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.view3d.yaw = orbit.yaw;
    m.view3d.pitch = orbit.pitch;
    m.embeddingPositions = new Map([["a.mp3", { x: 0.7, y: 0.3, z: 0.5 }]]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.2,
      lensId: "camelot",
    };
    m.scopePositions = new Map([["a.mp3", { x: 0.6, y: 0.5, billboard: BB }]]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    return { cv, m };
  }

  it("blends the cube and billboard frames linearly in view space", () => {
    const { cv } = make3d({ yaw: 0.9, pitch: 0.15 });
    const ts = trackScreen(cv);
    const billboardEnd = ts({ wx: 0.6, wy: 0.5, billboard: BB });
    const cubeEnd = ts({ wx: 0.7, wy: 0.3, wz: 0.5 });
    for (const mix of [0.25, 0.5, 0.75]) {
      const blended = ts({
        wx: 0.6,
        wy: 0.5,
        billboard: BB,
        from: { wx: 0.7, wy: 0.3, wz: 0.5 },
        mix,
      });
      expect(blended.sx).toBeCloseTo(
        billboardEnd.sx * (1 - mix) + cubeEnd.sx * mix,
        6,
      );
      expect(blended.sy).toBeCloseTo(
        billboardEnd.sy * (1 - mix) + cubeEnd.sy * mix,
        6,
      );
      expect(blended.scale).toBeCloseTo(
        billboardEnd.scale * (1 - mix) + cubeEnd.scale * mix,
        6,
      );
      expect(blended.depth).toBeCloseTo(
        billboardEnd.depth * (1 - mix) + cubeEnd.depth * mix,
        6,
      );
    }
  });

  it("lands exactly on the settled billboard at mix 0 and the outer projection at mix 1", () => {
    const { cv, m } = make3d({ yaw: 0, pitch: 0 });
    const ts = trackScreen(cv);
    // At mix 0 the overlay entry is pixel-identical to the settled scope member,
    // so the open animation ends with no snap.
    cv.animPositions = new Map([
      [
        "a.mp3",
        { x: 0.6, y: 0.5, billboard: BB, from: { x: 0.7, y: 0.3, z: 0.5 }, mix: 0 },
      ],
    ]);
    const settled = ts({ wx: 0.6, wy: 0.5, billboard: BB });
    const throughOverlay = ts({
      wx: 0.6,
      wy: 0.5,
      billboard: BB,
      from: { wx: 0.7, wy: 0.3, wz: 0.5 },
      mix: 0,
    });
    expect(throughOverlay.sx).toBe(settled.sx);
    expect(throughOverlay.sy).toBe(settled.sy);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.6,
      wy: 0.5,
    });
    // At mix 1 the entry is exactly the outer cube projection (head-on camera,
    // and the outer sits on the centre depth plane → scale 1).
    cv.animPositions = new Map([
      [
        "a.mp3",
        { x: 0.6, y: 0.5, billboard: BB, from: { x: 0.7, y: 0.3, z: 0.5 }, mix: 1 },
      ],
    ]);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.7,
      wy: 0.3,
    });
  });

  it("anchors the billboard from the entry itself once the scope is gone (close)", () => {
    const { cv, m } = make3d({ yaw: 0, pitch: 0 });
    const ts = trackScreen(cv);
    const vf = viewFrame(cv);
    const ANCHOR = { cx: 0.55, cy: 0.45, cz: 0.6 };
    // The close animation runs after the scope is gone; the overlay entry must
    // still project, its anchor coming from the entry, not `model.scope`.
    m.scope = null;
    m.scopePositions = new Map();
    cv.animPositions = new Map([
      [
        "a.mp3",
        { x: 0.6, y: 0.5, billboard: ANCHOR, from: { x: 0.7, y: 0.3, z: 0.5 }, mix: 0.5 },
      ],
    ]);
    const s = ts({
      wx: 0.6,
      wy: 0.5,
      billboard: ANCHOR,
      from: { wx: 0.7, wy: 0.3, wz: 0.5 },
      mix: 0.5,
    });
    const billboardEnd = ts({ wx: 0.6, wy: 0.5, billboard: ANCHOR });
    const cubeEnd = ts({ wx: 0.7, wy: 0.3, wz: 0.5 });
    expect(s.sx).toBeCloseTo(billboardEnd.sx * 0.5 + cubeEnd.sx * 0.5, 6);
    expect(s.sy).toBeCloseTo(billboardEnd.sy * 0.5 + cubeEnd.sy * 0.5, 6);
    const bbF = vf({ wx: 0.6, wy: 0.5, billboard: ANCHOR });
    const cubeF = vf({ wx: 0.7, wy: 0.3, wz: 0.5 });
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: bbF.rx * 0.5 + cubeF.rx * 0.5,
      wy: bbF.ry * 0.5 + cubeF.ry * 0.5,
    });
  });

  it("orbiting between frames moves the cube part while the billboard stays pinned", () => {
    const { cv: cv1 } = make3d({ yaw: 0.9, pitch: 0.15 });
    const { cv: cv2 } = make3d({ yaw: 1.6, pitch: 0.35 });
    const ts1 = trackScreen(cv1);
    const ts2 = trackScreen(cv2);
    const mix = 0.25;
    const frame = (ts: ReturnType<typeof trackScreen>) => {
      const billboardEnd = ts({ wx: 0.6, wy: 0.5, billboard: BB });
      const cubeEnd = ts({ wx: 0.7, wy: 0.3, wz: 0.5 });
      const blended = ts({
        wx: 0.6,
        wy: 0.5,
        billboard: BB,
        from: { wx: 0.7, wy: 0.3, wz: 0.5 },
        mix,
      });
      return { billboardEnd, cubeEnd, blended };
    };
    const a = frame(ts1);
    const b = frame(ts2);
    // The billboard endpoint is identical across orbits; the cube endpoint moves.
    expect(a.billboardEnd.sx).toBe(b.billboardEnd.sx);
    expect(a.billboardEnd.sy).toBe(b.billboardEnd.sy);
    expect(a.cubeEnd.sx).not.toBeCloseTo(b.cubeEnd.sx, 3);
    // The blend moves by exactly the cube part's share of the mix.
    expect(b.blended.sx - a.blended.sx).toBeCloseTo(
      mix * (b.cubeEnd.sx - a.cubeEnd.sx),
      6,
    );
    expect(b.blended.sy - a.blended.sy).toBeCloseTo(
      mix * (b.cubeEnd.sy - a.cubeEnd.sy),
      6,
    );
  });

  it("ignores the frames in 2-D: an entry draws at the plain lerped position", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.8 }]]);
    m.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
    m.scopePositions = new Map([["a.mp3", { x: 0.6, y: 0.4, billboard: BB }]]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    const ts = trackScreen(cv);
    const s = ts({
      wx: 0.6,
      wy: 0.4,
      billboard: BB,
      from: { wx: 0.2, wy: 0.8 },
      mix: 0.25,
    });
    const [px, py] = cv.w2s(0.6 * 0.75 + 0.2 * 0.25, 0.4 * 0.75 + 0.8 * 0.25);
    expect(s.sx).toBeCloseTo(px, 6);
    expect(s.sy).toBeCloseTo(py, 6);
    expect(s.depth).toBe(1);
    expect(s.scale).toBe(1);
  });

  it("keeps cube-frame rendering for circle-scope positions and from-less entries", () => {
    const { cv, m } = make3d({ yaw: 0, pitch: 0 });
    const ts = trackScreen(cv);
    // A circle scope's flat positions carry no billboard → drawn as cube positions.
    m.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
    m.scopePositions = new Map([["a.mp3", { x: 0.6, y: 0.5 }]]);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.6,
      wy: 0.5,
    });
    // A from-less animation entry is just the position, same as before.
    cv.animPositions = new Map([["a.mp3", { x: 0.65, y: 0.55 }]]);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.65,
      wy: 0.55,
    });
    expect(ts({ wx: 0.65, wy: 0.55 }).depth).toBeCloseTo(0.05 + 0.95 * 0.5, 8);
  });
});

// ─── Sphere-scope frosted glass ─────────────────────────────

describe("CanvasView sphere scope frosted glass", () => {
  type Ev =
    | { type: "drawImage"; img: unknown }
    | { type: "fill"; style: string; alpha: number }
    | { type: "arc"; x: number; y: number; r: number }
    | { type: "clip" }
    | { type: "filter"; value: string };
  let log: Ev[];
  let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

  /**
   * A 2-D context stand-in that logs every drawImage / fill / arc / clip /
   * filter call. With no *filter* the fake cannot blur (an unknown property
   * reads back as a function, so the glass falls back to a flat tint); pass
   * `"none"` for a context that does support the canvas filter.
   */
  function makeCtx(filter?: string): {
    ctx: CanvasRenderingContext2D;
    log: Ev[];
  } {
    const log: Ev[] = [];
    let currentAlpha = 1;
    const target: Record<string, unknown> = {
      measureText: () => ({ width: 10 }),
      ...(filter !== undefined ? { filter } : {}),
      drawImage: (img: unknown) => {
        log.push({ type: "drawImage", img });
      },
      fill: () => {
        log.push({
          type: "fill",
          style: String(target.fillStyle),
          alpha: currentAlpha,
        });
      },
      arc: (x: number, y: number, r: number) => {
        log.push({ type: "arc", x, y, r });
      },
      clip: () => {
        log.push({ type: "clip" });
      },
    };
    const ctx = new Proxy(target, {
      get(t, prop: string) {
        if (prop in t) return t[prop];
        return () => undefined;
      },
      set(t, prop: string, value) {
        if (prop === "globalAlpha") currentAlpha = value as number;
        if (prop === "filter") log.push({ type: "filter", value: String(value) });
        t[prop] = value;
        return true;
      },
    });
    return { ctx: ctx as unknown as CanvasRenderingContext2D, log };
  }

  beforeEach(() => {
    vi.stubGlobal("devicePixelRatio", 1);
    document.body.innerHTML =
      '<canvas id="viewport"></canvas><div id="tooltip" class="hidden"></div><div id="coords-panel" class="hidden"></div>';
    const fake = makeCtx();
    log = fake.log;
    origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = origGetContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /**
   * A 3-D canvas with a sphere scope at the cube centre (projected circle
   * 400, 200, r 180 on the 800×400 viewport) and three tracks: a member laid
   * out by the scope lens, a non-member projecting to the circle centre behind
   * the sphere (inside the circle), and a non-member outside the circle.
   */
  function makeCanvas(): {
    cv: CanvasView;
    points: {
      supported: true;
      render: ReturnType<typeof vi.fn>;
      canvas: Record<string, never>;
    };
    glow: {
      render: ReturnType<typeof vi.fn>;
      canvas: { style: { visibility: string } };
    };
  } {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "member.mp3" }),
      makeTrack({ path: "inside.mp3" }),
      makeTrack({ path: "outside.mp3" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    // Head-on orbit so the world positions map to fixed screen positions.
    m.view3d.yaw = 0;
    m.view3d.pitch = 0;
    m.embeddingPositions = new Map([
      ["member.mp3", { x: 0.5, y: 0.5, z: 0.5 }],
      ["inside.mp3", { x: 0.5, y: 0.5, z: 0.1 }],
      ["outside.mp3", { x: 0.8, y: 0.5, z: 0.5 }],
    ]);
    m.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.25,
      lensId: "clap",
    };
    // The scope lens places the member on the billboarded disc; the others are
    // non-members at their outer positions.
    m.scopePositions = new Map([
      ["member.mp3", { x: 0.55, y: 0.45, billboard: { cx: 0.5, cy: 0.5, cz: 0.5 } }],
    ]);
    const cv = new CanvasView(m);
    const glow = { render: vi.fn(), canvas: { style: { visibility: "" } } };
    const points = { supported: true as const, render: vi.fn(), canvas: {} };
    (cv as unknown as { glow: typeof glow; points: typeof points }).glow = glow;
    (cv as unknown as { glow: typeof glow; points: typeof points }).points =
      points;
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    return { cv, points, glow };
  }

  // Screen geometry: member billboards to (436, 216), inside sits at the
  // circle centre (400, 200), outside at (616, 200) — 216 px from the centre.
  const memberAt = (specs: PointSpec[]) =>
    specs.find((s) => Math.hypot(s.x - 436, s.y - 216) < 1);

  it("glasses a non-member inside the sphere's circle onto the rim and keeps it unhittable", () => {
    const { cv, points } = makeCanvas();
    cv.draw();
    // inside.mp3 projects at the circle centre behind the sphere: it is pushed
    // to the inner rim band of the outer pass — visible under the glass — but
    // never hittable or selectable.
    expect(cv.positions.find((p) => p.idx === 1)).toBeUndefined();
    const all = points.render.mock.calls.flatMap(
      (call) => call[0] as PointSpec[],
    );
    const rim = all.find((s) => {
      const d = Math.hypot(s.x - 400, s.y - 200);
      return d >= 0.82 * 180 && d <= 0.964 * 180;
    });
    expect(rim).toBeDefined();
    // At the exact centre the fixed +x direction puts it at (1 − RIM_BAND)·r.
    expect(rim!.x).toBeCloseTo(400 + 0.82 * 180, 3);
    expect(rim!.y).toBeCloseTo(200, 3);
    expect(rim!.alpha).toBe(1);

    // The same track is present at its raw projection with no scope…
    cv.model.scope = null;
    cv.draw();
    const pos = cv.positions.find((p) => p.idx === 1);
    expect(pos).toBeDefined();
    expect(pos!.sx).toBeCloseTo(400, 3);
    expect(pos!.sy).toBeCloseTo(200, 3);

    // …and with the sphere scope kept but the 3-D view off.
    cv.model.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.25,
      lensId: "clap",
    };
    cv.model.view3d.on = false;
    cv.model.view3dBlend = 0;
    cv.draw();
    expect(cv.positions.find((p) => p.idx === 1)).toBeDefined();
  });

  it("keeps an outside non-member in the outer pass and routes members to the second", () => {
    const { cv, points } = makeCanvas();
    cv.draw();
    expect(points.render).toHaveBeenCalledTimes(2);
    const [outer] = points.render.mock.calls[0] as [PointSpec[], boolean];
    const [inner] = points.render.mock.calls[1] as [PointSpec[], boolean];
    // outside.mp3 (616, 200) stays with the outer field; the member's spec goes
    // to the second pass only.
    expect(outer.find((s) => Math.hypot(s.x - 616, s.y - 200) < 1)).toBeDefined();
    expect(memberAt(outer)).toBeUndefined();
    expect(memberAt(inner)).toBeDefined();
    // Both visible tracks stay hittable.
    expect(cv.positions.map((p) => p.idx)).toEqual([0, 2]);
  });

  it("blits outer blobs, draws the frosted glass, then blits the member blobs", () => {
    const { cv, points, glow } = makeCanvas();
    cv.draw();
    // The composition order: outer blobs → glass → member blobs → glow.
    const blits = log.filter(
      (e): e is Extract<Ev, { type: "drawImage" }> => e.type === "drawImage",
    );
    const glassFill = log.find(
      (e) => e.type === "fill" && e.style === "#1a1a2e",
    ) as Extract<Ev, { type: "fill" }>;
    expect(blits).toHaveLength(3);
    expect(log.indexOf(glassFill)).toBeGreaterThan(log.indexOf(blits[0]));
    expect(log.indexOf(glassFill)).toBeLessThan(log.indexOf(blits[1]));
    // The member blobs go through the same metaball canvas as the outer field;
    // the glow blits last, over both.
    expect(blits[0].img).toBe(points.canvas);
    expect(blits[1].img).toBe(points.canvas);
    expect(blits[2].img).toBe(glow.canvas);
    // The glass is a clip of the projected circle, tinted with the map
    // background, between the two metaball blits.
    const clipIdx = log.findIndex((e) => e.type === "clip");
    expect(clipIdx).toBeGreaterThan(log.indexOf(blits[0]));
    expect(clipIdx).toBeLessThan(log.indexOf(blits[1]));
    const glassArc = log.find(
      (e) => e.type === "arc" && Math.abs(e.r - 180) < 1,
    ) as Extract<Ev, { type: "arc" }>;
    expect(glassArc.x).toBeCloseTo(400, 5);
    expect(glassArc.y).toBeCloseTo(200, 5);
    // The fake context cannot blur (no `filter`), so the tint alone hides more —
    // and there is no canvas self-copy to blur with.
    expect(glassFill.alpha).toBe(GLASS_TINT_FLAT);
    expect(
      log.some((e) => e.type === "drawImage" && e.img === cv.$canvas),
    ).toBe(false);
  });

  it("self-copies the canvas under the clip and blurs it when ctx.filter is a string", () => {
    const fake = makeCtx("none");
    const prev = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      fake.ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
    const { cv, points } = makeCanvas();
    cv.draw();
    HTMLCanvasElement.prototype.getContext = prev;
    // The self-copy of the view canvas happens between the outer and member
    // metaball blits, clipped to the circle.
    const selfCopy = fake.log.find(
      (e) => e.type === "drawImage" && e.img === cv.$canvas,
    );
    expect(selfCopy).toBeDefined();
    const outerIdx = fake.log.findIndex(
      (e) => e.type === "drawImage" && e.img === points.canvas,
    );
    let memberIdx = -1;
    for (let i = fake.log.length - 1; i >= 0; i--) {
      const e = fake.log[i];
      if (e.type === "drawImage" && e.img === points.canvas) {
        memberIdx = i;
        break;
      }
    }
    const copyIdx = fake.log.indexOf(selfCopy!);
    expect(copyIdx).toBeGreaterThan(outerIdx);
    expect(copyIdx).toBeLessThan(memberIdx);
    // The blur filter is set before the copy and reset to "none" after.
    const blurAt = fake.log.findIndex(
      (e) => e.type === "filter" && e.value === "blur(7px)",
    );
    const noneAt = fake.log.findIndex(
      (e) => e.type === "filter" && e.value === "none",
    );
    expect(blurAt).toBeGreaterThan(-1);
    expect(noneAt).toBeGreaterThan(blurAt);
    // With the blur softening the disc, the tint can be lighter.
    const glassFill = fake.log.find(
      (e) => e.type === "fill" && e.style === "#1a1a2e",
    ) as Extract<Ev, { type: "fill" }>;
    expect(glassFill.alpha).toBe(GLASS_TINT);
  });

  it("renders one pass without a sphere scope, for no scope and for a circle scope", () => {
    const { cv, points } = makeCanvas();
    cv.model.scope = null;
    cv.draw();
    expect(points.render).toHaveBeenCalledTimes(1);
    // No glass, so no tint fill anywhere in the frame.
    expect(log.some((e) => e.type === "fill" && e.style === "#1a1a2e")).toBe(
      false,
    );

    // A circle scope keeps the single pass too — the glass is a sphere-only
    // behaviour, even while the 3-D view is still armed.
    points.render.mockClear();
    log.length = 0;
    cv.model.scope = {
      shape: "circle",
      cx: 0.5,
      cy: 0.5,
      r: 0.25,
      lensId: "camelot",
    };
    cv.draw();
    expect(points.render).toHaveBeenCalledTimes(1);
    expect(log.some((e) => e.type === "fill" && e.style === "#1a1a2e")).toBe(
      false,
    );
  });

  it("drops a non-member's search-field glow inside the circle and keeps a member's", () => {
    const { cv } = makeCanvas();
    cv.searchField = [
      { path: "member.mp3", score: 0.5 },
      { path: "inside.mp3", score: 0.5 },
      { path: "outside.mp3", score: 0.5 },
    ];
    cv.searchActive = true;
    cv.draw();
    const pts = cv.collectSearchGlowPoints();
    // inside.mp3 projects at the circle centre (behind the glass): its glow is
    // dropped; the member and outside landmarks keep theirs.
    expect(pts).toHaveLength(2);
    expect(pts.find((p) => Math.hypot(p.x - 400, p.y - 200) < 1)).toBeUndefined();
    expect(pts.find((p) => Math.hypot(p.x - 436, p.y - 216) < 1)).toBeDefined();
    expect(pts.find((p) => Math.hypot(p.x - 616, p.y - 200) < 1)).toBeDefined();
  });

  it("reads the map background from --bg, falling back to #1a1a2e", () => {
    const bg = (cv: CanvasView) =>
      (cv as unknown as { mapBackground(): string }).mapBackground.bind(cv);
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    // No stylesheet in happy-dom: the property is unset → the CSS default.
    expect(bg(cv)()).toBe("#1a1a2e");
    // A second view's own cache picks up a set value.
    document.documentElement.style.setProperty("--bg", "#102030");
    const m2 = new Model();
    m2.setAllTracks([makeTrack({ path: "b.mp3" })]);
    const cv2 = new CanvasView(m2);
    expect(bg(cv2)()).toBe("#102030");
    document.documentElement.style.removeProperty("--bg");
  });
});

// ─── 3-D scope picking ────────────────────────────────────────

describe("CanvasView scope picking", () => {
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

  /** A 3-D canvas whose dots are well separated on screen. */
  function make3d(): CanvasView {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    // Head-on camera so the unproject lands on the z = 0.5 centre plane.
    m.view3d.yaw = 0;
    m.view3d.pitch = 0;
    m.embeddingPositions = new Map([
      ["a.mp3", { x: 0.3, y: 0.3, z: 0.8 }],
      ["b.mp3", { x: 0.6, y: 0.4, z: 0.2 }],
      ["c.mp3", { x: 0.5, y: 0.7, z: 0.5 }],
    ]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    cv.draw();
    return cv;
  }

  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  it("snaps to the nearest on-screen dot within 20 px and returns its cube position", () => {
    const cv = make3d();
    const pos = cv.positions.find((p) => p.idx === 0)!;
    const picked = cv.pickCursorWorld3D(pos.sx + 5, pos.sy - 5);
    expect(picked.snapped).toBe(true);
    // The cube position — membership is orbit-invariant — not the view frame.
    expect(picked.wx).toBeCloseTo(0.3, 8);
    expect(picked.wy).toBeCloseTo(0.3, 8);
    expect(picked.wz).toBeCloseTo(0.8, 8);
  });

  it("prefers the front-most among dots within 2 px of each other", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "back.mp3" }),
      makeTrack({ path: "front.mp3" }),
    ]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.embeddingPositions = new Map([
      ["back.mp3", { x: 0.3, y: 0.3, z: 0.2 }],
      ["front.mp3", { x: 0.31, y: 0.3, z: 0.9 }],
    ]);
    const cv = new CanvasView(m);
    cv.positions = [
      { idx: 0, sx: 100, sy: 100, depth: 0.2 },
      { idx: 1, sx: 101, sy: 100, depth: 0.9 },
    ];
    const picked = cv.pickCursorWorld3D(100, 100);
    expect(picked.snapped).toBe(true);
    expect(picked.wx).toBeCloseTo(0.31, 8);
    expect(picked.wz).toBeCloseTo(0.9, 8);
  });

  it("unprojects onto the cube-centre plane when no dot is near", () => {
    const cv = make3d();
    const picked = cv.pickCursorWorld3D(700, 350);
    expect(picked.snapped).toBe(false);
    expect(picked.wz).toBeCloseTo(0.5, 8);
    // Round-trips: projecting the pick back lands on the cursor.
    const back = trackScreen(cv)({
      wx: picked.wx,
      wy: picked.wy,
      wz: picked.wz,
    });
    expect(back.sx).toBeCloseTo(700, 6);
    expect(back.sy).toBeCloseTo(350, 6);
  });

  it("round-trips with _trackScreen under a rotated camera", () => {
    const cv = make3d();
    cv.model.view3d.yaw = 0.5;
    cv.model.view3d.pitch = 0.3;
    cv.draw();
    const picked = cv.pickCursorWorld3D(300, 120);
    expect(picked.snapped).toBe(false);
    const back = trackScreen(cv)({
      wx: picked.wx,
      wy: picked.wy,
      wz: picked.wz,
    });
    expect(back.sx).toBeCloseTo(300, 6);
    expect(back.sy).toBeCloseTo(120, 6);
  });

  it("inverts the projection scale for worldRadiusForPx at the centre plane", () => {
    const cv = make3d();
    // At the cube-centre plane the perspective scale is 1, so the world radius
    // is px per world unit (720 px across the unit square at zoom 1).
    const r = cv.worldRadiusForPx(0.5, 0.5, 0.5, 14);
    expect(r).toBeCloseTo(14 / 720, 8);
    // A sphere of that radius projects back to a 14 px screen circle.
    cv.model.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r,
      lensId: "camelot",
    };
    expect(cv.scopeScreenCircle()!.r).toBeCloseTo(14, 3);
  });

  it("accounts for perspective at an off-centre depth", () => {
    const cv = make3d();
    cv.model.view3d.yaw = 0.5;
    cv.model.view3d.pitch = 0.3;
    const cz = 1; // rotated depth +0.5 → perspective scale > 1
    const ts = trackScreen(cv);
    const c = ts({ wx: 0.5, wy: 0.5, wz: cz });
    expect(c.scale).not.toBe(1);
    const r = cv.worldRadiusForPx(0.5, 0.5, cz, 14);
    cv.model.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz,
      r,
      lensId: "camelot",
    };
    expect(cv.scopeScreenCircle()!.r).toBeCloseTo(14, 3);
  });

  it("projects the sphere centre to the screen circle and returns null for circle/box", () => {
    const cv = make3d();
    cv.model.scope = {
      shape: "sphere",
      cx: 0.4,
      cy: 0.6,
      cz: 0.8,
      r: 0.12,
      lensId: "camelot",
    };
    const circle = cv.scopeScreenCircle();
    expect(circle).not.toBeNull();
    const c = trackScreen(cv)({ wx: 0.4, wy: 0.6, wz: 0.8 });
    expect(circle!.cx).toBeCloseTo(c.sx, 5);
    expect(circle!.cy).toBeCloseTo(c.sy, 5);
    cv.model.scope = {
      shape: "circle",
      cx: 0.5,
      cy: 0.5,
      r: 0.1,
      lensId: "camelot",
    };
    expect(cv.scopeScreenCircle()).toBeNull();
    cv.model.scope = {
      shape: "box",
      cx: 0.5,
      cy: 0.5,
      r: 0.1,
      rx: 0.1,
      ry: 0.05,
      lensId: "camelot",
    };
    expect(cv.scopeScreenCircle()).toBeNull();
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
    depth: 1,
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

  it("draws the 3-D toggle tab right of the lens row and hit-tests it", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });

    cv.draw();

    expect(cv.viewTabs).toHaveLength(1);
    const tab = cv.viewTabs[0];
    expect(tab.id).toBe("3d");
    const lastLens = cv.lensTabs[cv.lensTabs.length - 1];
    // Just right of the last lens tab, on the lens row.
    expect(tab.x0).toBeGreaterThan(lastLens.x1);
    expect(tab.y0).toBe(lastLens.y0);
    const cx = (tab.x0 + tab.x1) / 2;
    const cy = (tab.y0 + tab.y1) / 2;
    expect(cv.topViewTabAt(cx, cy)).toBe("3d");
    expect(cv.topViewTabAt(cx + 50, cy)).toBeNull();
  });

  it("highlights the 3-D tab while the orbit is armed", () => {
    const activeFill = "rgba(120,220,255,0.92)";
    const fills: string[] = [];
    const target: Record<string, unknown> = { measureText: () => ({ width: 10 }) };
    const ctx = new Proxy(target, {
      get(t, prop: string) {
        if (prop === "fill") return () => fills.push(String(t.fillStyle));
        if (prop in t) return t[prop];
        return () => undefined;
      },
      set(t, prop: string, value) {
        t[prop] = value;
        return true;
      },
    });
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() =>
      ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    cv.model.view3d.on = true;
    cv.draw();
    // Active fills: the active lens tab, the active colour tab, and the armed
    // 3-D toggle.
    expect(fills.filter((f) => f === activeFill)).toHaveLength(3);

    // Disarming the orbit leaves only the lens and colour tabs highlighted.
    fills.length = 0;
    cv.model.view3d.on = false;
    cv.draw();
    expect(fills.filter((f) => f === activeFill)).toHaveLength(2);

    HTMLCanvasElement.prototype.getContext = origGetContext;
  });

  it("hides the 3-D toggle tab during the eval gallery", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.evalMode = true;
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });

    cv.draw();

    expect(cv.viewTabs).toEqual([]);
  });
});

// ─── 3-D orbit view ──────────────────────────────────────────

describe("CanvasView 3-D hit testing", () => {
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

  function make3dCanvas(): CanvasView {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    return new CanvasView(m);
  }

  it("picks the front-most candidate in 3-D, not the nearest", () => {
    const cv = make3dCanvas();
    expect(cv.model.is3d).toBe(true);
    // A nearer but deeper dot loses to a farther but front-most one.
    cv.positions = [
      { idx: 0, sx: 100, sy: 100, depth: 0.3 },
      { idx: 1, sx: 104, sy: 100, depth: 0.9 },
    ];
    expect(cv.hitTest(100, 100)).toBe(1);
  });

  it("breaks a 3-D depth tie by distance", () => {
    const cv = make3dCanvas();
    cv.positions = [
      { idx: 0, sx: 100, sy: 100, depth: 0.9 },
      { idx: 1, sx: 104, sy: 100, depth: 0.9 },
    ];
    expect(cv.hitTest(100, 100)).toBe(0);
  });

  it("keeps the nearest rule in 2-D", () => {
    const cv = make3dCanvas();
    cv.model.view3d.on = false;
    cv.positions = [
      { idx: 0, sx: 100, sy: 100, depth: 1 },
      { idx: 1, sx: 104, sy: 100, depth: 1 },
    ];
    expect(cv.hitTest(100, 100)).toBe(0);
  });

  it("keeps the front-most rule during the collapse tween", () => {
    const cv = make3dCanvas();
    // Turned off but still blending back to head-on: rendering stays 3-D.
    cv.model.view3d.on = false;
    cv.model.view3dBlend = 0.5;
    cv.positions = [
      { idx: 0, sx: 100, sy: 100, depth: 0.3 },
      { idx: 1, sx: 104, sy: 100, depth: 0.9 },
    ];
    expect(cv.hitTest(100, 100)).toBe(1);
  });

  it("ignores dots outside the hit radius in 3-D", () => {
    const cv = make3dCanvas();
    cv.positions = [{ idx: 0, sx: 200, sy: 200, depth: 1 }];
    expect(cv.hitTest(100, 100)).toBe(-1);
  });
});

describe("CanvasView 3-D world positions", () => {
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

  it("returns the rotated view-frame position in 3-D", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1; // full orbit + perspective
    m.view3d.yaw = Math.PI / 2;
    m.view3d.pitch = 0;
    m.embeddingPositions = new Map([["a.mp3", { x: 1, y: 0.5, z: 0.5 }]]);
    const cv = new CanvasView(m);
    // A 90° yaw swings the +z front face to +x: the point's x collapses to the
    // cube centre while its y is untouched. z=0.5 sits at the cube centre, so
    // perspective leaves it unchanged.
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.5,
      wy: 0.5,
    });
  });

  it("applies perspective to the rotated frame position", () => {
    vi.stubGlobal("location", { search: "?persp=0.3" });
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    m.view3d.yaw = 0;
    m.view3d.pitch = 0;
    // A near point (z=1 → rotated depth +0.5) scales its centred offset up.
    m.embeddingPositions = new Map([["a.mp3", { x: 0.8, y: 0.5, z: 1 }]]);
    const cv = new CanvasView(m);
    const pos = cv.worldPosForTrack(m.trackByPath("a.mp3")!);
    expect(pos).not.toBeNull();
    const scale = 1 / (1 - 0.3 * 0.5);
    expect(pos!.wx).toBeCloseTo(0.5 + 0.3 * scale, 8);
    expect(pos!.wy).toBeCloseTo(0.5, 8);
  });

  it("returns the raw position unchanged in 2-D", () => {
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.embeddingPositions = new Map([["a.mp3", { x: 1, y: 0.5, z: 0.5 }]]);
    const cv = new CanvasView(m);
    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 1,
      wy: 0.5,
      wz: 0.5,
    });
  });
});

// ─── 3-D perspective projection ──────────────────────────────

describe("CanvasView 3-D perspective projection", () => {
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

  /** Cast through the private `_trackScreen` so the projection is unit-tested. */
  function trackScreen(cv: CanvasView) {
    return (cv as unknown as {
      _trackScreen(pos: {
        wx: number;
        wy: number;
        wz?: number;
      }): { sx: number; sy: number; depth: number; scale: number };
    })._trackScreen.bind(cv);
  }

  it("scales a near point up and a far point down under perspective", () => {
    vi.stubGlobal("location", { search: "?persp=0.28" });
    const m = new Model();
    m.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m.lensId = "combined";
    m.view3d.on = true;
    m.view3dBlend = 1;
    const cv = new CanvasView(m);
    Object.defineProperty(cv.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv.$canvas, "clientHeight", { value: 400 });
    const ts = trackScreen(cv);
    // z=1 is the near face (rotated depth +0.5), z=0 the far face, z=0.5 centre.
    const near = ts({ wx: 0.5, wy: 0.5, wz: 1 });
    const far = ts({ wx: 0.5, wy: 0.5, wz: 0 });
    const centre = ts({ wx: 0.5, wy: 0.5, wz: 0.5 });
    expect(near.scale).toBeGreaterThan(1);
    expect(far.scale).toBeLessThan(1);
    expect(centre.scale).toBe(1);
    // The centre point never moves, whatever the perspective.
    expect(centre.sx).toBeCloseTo(400, 5);
    expect(centre.sy).toBeCloseTo(200, 5);
  });

  it("projects head-on orthographic at blend 0 even with 3-D armed", () => {
    // A strong persp that must NOT apply while the camera is still at blend 0.
    vi.stubGlobal("location", { search: "?persp=0.5" });
    const m3d = new Model();
    m3d.setAllTracks([makeTrack({ path: "a.mp3" })]);
    m3d.lensId = "combined";
    m3d.view3d.on = true;
    m3d.view3dBlend = 0; // the collapse tween's start: still head-on
    const cv3d = new CanvasView(m3d);
    Object.defineProperty(cv3d.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv3d.$canvas, "clientHeight", { value: 400 });

    const m2d = new Model();
    m2d.setAllTracks([makeTrack({ path: "a.mp3" })]);
    const cv2d = new CanvasView(m2d);
    Object.defineProperty(cv2d.$canvas, "clientWidth", { value: 800 });
    Object.defineProperty(cv2d.$canvas, "clientHeight", { value: 400 });

    const p3d = trackScreen(cv3d)({ wx: 0.7, wy: 0.3, wz: 0.9 });
    const p2d = trackScreen(cv2d)({ wx: 0.7, wy: 0.3 });
    expect(p3d.sx).toBeCloseTo(p2d.sx, 10);
    expect(p3d.sy).toBeCloseTo(p2d.sy, 10);
    expect(p3d.scale).toBe(1);
  });
});

describe("depthShade", () => {
  it("is monotonic in depth at full blend", () => {
    const back = depthShade(0);
    const mid = depthShade(0.5);
    const front = depthShade(1);
    expect(front).toBeGreaterThan(mid);
    expect(mid).toBeGreaterThan(back);
  });

  it("anchors the front at full brightness", () => {
    expect(depthShade(1)).toBe(1);
    expect(depthShade(0)).toBe(0.45);
  });

  it("at blend 0 every depth draws at full brightness (the 2-D colours)", () => {
    expect(depthShade(0, 0)).toBe(1);
    expect(depthShade(0.5, 0)).toBe(1);
    expect(depthShade(1, 0)).toBe(1);
  });

  it("interpolates linearly between the 2-D and full-shade colours", () => {
    expect(depthShade(0, 0.5)).toBeCloseTo((1 + 0.45) / 2, 5);
    expect(depthShade(1, 0.5)).toBeCloseTo(1, 5);
    expect(depthShade(0.5, 0.5)).toBeCloseTo(
      1 - (1 - (0.45 + 0.55 * 0.5)) * 0.5,
      5,
    );
  });

  it("clamps the blend to [0, 1]", () => {
    expect(depthShade(0, -1)).toBe(1);
    expect(depthShade(0, 2)).toBe(0.45);
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
