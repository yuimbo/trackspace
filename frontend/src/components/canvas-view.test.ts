import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import { clusterColor } from "../lib/toast";
import { CanvasView, tooltipLines } from "./canvas-view";

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

describe("tooltipLines", () => {
  it("renders folder, title, audio features, and tags", () => {
    const m = new Model();
    const lines = tooltipLines(
      makeTrack({
        folder: "House",
        artist: "A",
        title: "B",
        bpm: 128.4,
        musical_key: "Am",
        tags: { energy: 0.5 },
      }),
      m,
    );
    expect(lines).toEqual(["House", "A — B", "128 BPM · Am", "energy: 0.50"]);
  });

  it("inserts the cluster name between the audio line and the tags", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 3]]);
    m.clusterNames = new Map([[3, "Techno"]]);
    const lines = tooltipLines(
      makeTrack({ bpm: 120, tags: { energy: 0.25 } }),
      m,
    );
    expect(lines).toEqual(["Library root", "a.mp3", "120 BPM", "Techno", "energy: 0.25"]);
  });

  it("falls back to a numbered label for an unnamed cluster", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["a.mp3", 9]]);
    const lines = tooltipLines(makeTrack(), m);
    expect(lines).toContain("Cluster 9");
  });

  it("omits the cluster line when the track has no assignment", () => {
    const m = new Model();
    m.clusterAssignments = new Map([["other.mp3", 1]]);
    m.clusterNames = new Map([[1, "Techno"]]);
    const lines = tooltipLines(makeTrack(), m);
    expect(lines).not.toContain("Techno");
    expect(lines.some((l) => l.startsWith("Cluster "))).toBe(false);
  });
});

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
