import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import { clusterColor } from "../lib/toast";
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
    cv.hoveredClusterId = 2;
    cv.draw();
    expect(fills).toContain(clusterColor(2));
  });

  it("does not glow a cluster colour when nothing is hovered", () => {
    const cv = makeCanvas();
    cv.model.clusterAssignments = new Map([["a.mp3", 2]]);
    cv.draw();
    expect(fills).not.toContain(clusterColor(2));
  });
});
