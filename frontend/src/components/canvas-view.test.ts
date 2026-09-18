import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
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

  it("resolves in-scope tracks to their scope position, others outside", () => {
    const m = new Model();
    m.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3", folder: "House" }),
    ]);
    const cv = new CanvasView(m);
    m.scope = { cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" };
    m.scopePositions = new Map([["a.mp3", { x: 0.4, y: 0.6 }]]);

    expect(cv.worldPosForTrack(m.trackByPath("a.mp3")!)).toEqual({
      wx: 0.4,
      wy: 0.6,
    });
    // Not in scopePositions → the tag axes (both null → 0.5, 0.5).
    expect(cv.worldPosForTrack(m.trackByPath("b.mp3")!)).toEqual({
      wx: 0.5,
      wy: 0.5,
    });
  });
});
