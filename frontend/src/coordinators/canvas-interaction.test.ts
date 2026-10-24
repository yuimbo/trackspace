import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type HoverTarget, type Track } from "../model";
import { ORBIT_RAD_PER_PX } from "../lib/camera3d";
import { setEvalReadOnly } from "../lib/eval-readonly";
import type { CanvasView } from "../components";
import {
  CanvasInteractionController,
  type CanvasInteractionDeps,
} from "./canvas-interaction";

// ─── Fakes ───────────────────────────────────────────────────

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

interface FakeCanvas {
  $canvas: HTMLCanvasElement;
  $tip: HTMLElement;
  hoveredIdx: number;
  hoveredBloomTrack: Track | null;
  hoverTarget: HoverTarget | null;
  inspect: unknown;
  boxSel: { x0: number; y0: number; x1: number; y1: number } | null;
  lassoPoints: [number, number][];
  intersectMode: boolean;
  scopeDrag: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    sphere?: { cx: number; cy: number; cz: number };
  } | null;
  dragGhosts: Map<
    string,
    { wx: number; wy: number; wz?: number; folder: string }
  > | null;
  positions: { sx: number; sy: number; idx: number }[];
  tipTimer: unknown;
  tipReady: boolean;
  _txWorld: { minX: number; maxX: number; minY: number; maxY: number } | null;
  hitTest: ReturnType<typeof vi.fn>;
  hitTestTransform: ReturnType<typeof vi.fn>;
  s2w: ReturnType<typeof vi.fn>;
  minZoom: ReturnType<typeof vi.fn>;
  clearHover: ReturnType<typeof vi.fn>;
  worldPosForTrack: ReturnType<typeof vi.fn>;
  scopeTabAt: ReturnType<typeof vi.fn>;
  topLensTabAt: ReturnType<typeof vi.fn>;
  topColorTabAt: ReturnType<typeof vi.fn>;
  topViewTabAt: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  unpinInspect: ReturnType<typeof vi.fn>;
  toggleInspect: ReturnType<typeof vi.fn>;
  showVolumeHud: ReturnType<typeof vi.fn>;
  showBloomHud: ReturnType<typeof vi.fn>;
  pickCursorWorld3D: ReturnType<typeof vi.fn>;
  worldRadiusForPx: ReturnType<typeof vi.fn>;
  scopeScreenCircle: ReturnType<typeof vi.fn>;
}

function makeCanvas(): FakeCanvas {
  return {
    $canvas: document.createElement("canvas"),
    $tip: document.createElement("div"),
    hoveredIdx: -1,
    hoveredBloomTrack: null,
    hoverTarget: null,
    inspect: null,
    boxSel: null,
    lassoPoints: [],
    intersectMode: false,
    scopeDrag: null,
    dragGhosts: null,
    positions: [],
    tipTimer: 0,
    tipReady: false,
    _txWorld: null,
    hitTest: vi.fn(() => -1),
    hitTestTransform: vi.fn(() => null),
    // 1 px == 0.001 world units; keeps drag deltas small but > DRAG_THRESH in px.
    s2w: vi.fn((sx: number, sy: number) => [sx / 1000, sy / 1000]),
    minZoom: vi.fn(() => 0.1),
    clearHover: vi.fn(),
    worldPosForTrack: vi.fn(() => ({ wx: 0.5, wy: 0.5 })),
    scopeTabAt: vi.fn(() => null),
    topLensTabAt: vi.fn(() => null),
    topColorTabAt: vi.fn(() => null),
    topViewTabAt: vi.fn(() => null),
    scheduleDraw: vi.fn(),
    unpinInspect: vi.fn(() => false),
    toggleInspect: vi.fn(),
    showVolumeHud: vi.fn(),
    showBloomHud: vi.fn(),
    pickCursorWorld3D: vi.fn(() => ({
      wx: 0.5,
      wy: 0.5,
      wz: 0.5,
      snapped: true,
    })),
    worldRadiusForPx: vi.fn(() => 0.01),
    scopeScreenCircle: vi.fn(() => null),
  };
}

interface Harness {
  model: Model;
  canvas: FakeCanvas;
  deps: CanvasInteractionDeps;
  coord: CanvasInteractionController;
}

function makeHarness(): Harness {
  const model = new Model();
  const canvas = makeCanvas();
  document.body.appendChild(canvas.$canvas);
  const deps = {
    model,
    canvas: canvas as unknown as CanvasView,
    status: { update: vi.fn() },
    props: { render: vi.fn() },
    batch: { render: vi.fn() },
    renderTagPanels: vi.fn(),
    queueTagWrite: vi.fn(),
    flushTagWrites: vi.fn(),
    moveSelectionToFolder: vi.fn(),
    onHoverFolderChanged: vi.fn(),
    onHoverPreview: vi.fn(),
    stopPreview: vi.fn(),
    tweenViewport: vi.fn(),
    openScope: vi.fn(),
    setScopeLens: vi.fn(),
    resizeScope: vi.fn(),
    closeScope: vi.fn(),
    switchLens: vi.fn(),
    setColorMode: vi.fn(),
    toggleView3D: vi.fn(),
    moveSelectionBloom: vi.fn(),
    setSelectionBloom: vi.fn(),
    togglePreview: vi.fn(),
    adjustPreviewVolume: vi.fn(),
    commitPreviewVolume: vi.fn(),
    adjustBloomRange: vi.fn(),
    commitBloomRange: vi.fn(),
  } as CanvasInteractionDeps;
  const coord = new CanvasInteractionController(deps);
  coord.bind();
  return { model, canvas, deps, coord };
}

// ─── Event helpers ────────────────────────────────────────────

/** The viewport target the last fit tweened to. */
function tweenTarget(deps: CanvasInteractionDeps): {
  ox: number;
  oy: number;
  zoom: number;
} {
  const calls = (deps.tweenViewport as ReturnType<typeof vi.fn>).mock.calls;
  return calls[calls.length - 1][0] as { ox: number; oy: number; zoom: number };
}

type XY = { x?: number; y?: number };

function mouseEvent(
  type: string,
  opts: MouseEventInit & XY = {},
): MouseEvent {
  const { x, y, ...init } = opts;
  const ev = new MouseEvent(type, { bubbles: true, cancelable: true, ...init });
  if (x !== undefined) Object.defineProperty(ev, "offsetX", { value: x });
  if (y !== undefined) Object.defineProperty(ev, "offsetY", { value: y });
  return ev;
}

function wheelEvent(opts: WheelEventInit & XY = {}): WheelEvent {
  const { x, y, ...init } = opts;
  const ev = new WheelEvent("wheel", {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  if (x !== undefined) Object.defineProperty(ev, "offsetX", { value: x });
  if (y !== undefined) Object.defineProperty(ev, "offsetY", { value: y });
  // happy-dom does not carry modifier state through the WheelEvent init dict.
  if (opts.ctrlKey !== undefined)
    Object.defineProperty(ev, "ctrlKey", { value: opts.ctrlKey });
  return ev;
}

/** Press, move and release on the canvas element. */
function down(canvas: FakeCanvas, x: number, y: number, init: MouseEventInit = {}) {
  canvas.$canvas.dispatchEvent(mouseEvent("mousedown", { x, y, ...init }));
}
function move(canvas: FakeCanvas, x: number, y: number, init: MouseEventInit = {}) {
  canvas.$canvas.dispatchEvent(mouseEvent("mousemove", { x, y, ...init }));
}
function up(canvas: FakeCanvas, x: number, y: number, init: MouseEventInit = {}) {
  canvas.$canvas.dispatchEvent(mouseEvent("mouseup", { x, y, ...init }));
}

beforeEach(() => {
  document.body.innerHTML = `<div id="toast-container"></div>`;
});

afterEach(() => {
  vi.restoreAllMocks();
  setEvalReadOnly(false);
});

// ─── Selection ──────────────────────────────────────────────

describe("CanvasInteractionController selection", () => {
  it("selects a single track on a plain click", () => {
    const { model, canvas } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    canvas.hitTest.mockReturnValue(0);
    down(canvas, 10, 10);
    up(canvas, 10, 10);
    expect([...model.selected]).toEqual(["a.mp3"]);
  });

  it("clears the selection when clicking empty space", () => {
    const { model, canvas } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.selected.add("a.mp3");
    canvas.hitTest.mockReturnValue(-1);
    down(canvas, 10, 10);
    up(canvas, 10, 10);
    expect(model.selected.size).toBe(0);
  });

  it("toggles an already-selected track off with ctrl", () => {
    const { model, canvas } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.selected.add("a.mp3");
    canvas.hitTest.mockReturnValue(0);
    down(canvas, 10, 10, { ctrlKey: true });
    up(canvas, 10, 10, { ctrlKey: true });
    expect(model.selected.size).toBe(0);
  });

  it("selects tracks inside a shift-lasso polygon", () => {
    const { model, canvas } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "in.mp3" }),
      makeTrack({ path: "out.mp3" }),
    ]);
    canvas.hitTest.mockReturnValue(-1);
    canvas.positions = [
      { sx: 30, sy: 30, idx: 0 },
      { sx: 200, sy: 200, idx: 1 },
    ];
    down(canvas, 10, 10, { shiftKey: true });
    move(canvas, 50, 10);
    move(canvas, 50, 50);
    move(canvas, 10, 50);
    up(canvas, 10, 50);
    expect([...model.selected]).toEqual(["in.mp3"]);
  });
});

// ─── Tag drag ─────────────────────────────────────────────

describe("CanvasInteractionController tag drag", () => {
  it("starts a document drag past the threshold and mutates tag values", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3", tags: { tagA: 0.5 } })]);
    model.axisX = "tagA";
    canvas.hitTest.mockReturnValue(0);

    down(canvas, 10, 10);
    move(canvas, 30, 30); // beyond DRAG_THRESH
    expect(coord.isPointerBusy()).toBe(true);
    move(canvas, 40, 40);
    // dx in world units = (40 - 10) / 1000 = 0.03
    expect(model.trackByPath("a.mp3")!.tags.tagA).toBeCloseTo(0.53, 5);
  });

  it("queues and flushes tag writes when a drag commits", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3", tags: { tagA: 0.5 } })]);
    model.axisX = "tagA";
    canvas.hitTest.mockReturnValue(0);

    down(canvas, 10, 10);
    move(canvas, 30, 30);
    move(canvas, 40, 40);
    document.dispatchEvent(mouseEvent("mouseup", { x: 900, y: 900 }));

    expect(deps.queueTagWrite).toHaveBeenCalledWith(
      "a.mp3",
      "tagA",
      expect.closeTo(0.53, 5),
    );
    expect(deps.flushTagWrites).toHaveBeenCalled();
  });
});

// ─── Transforms ───────────────────────────────────────────

describe("CanvasInteractionController transforms", () => {
  it("scales tag values with a resize handle", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3", tags: { tagA: 0.5 } })]);
    model.selected.add("a.mp3");
    model.axisX = "tagA";
    canvas._txWorld = { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    canvas.hitTestTransform.mockReturnValue({ type: "e", cursor: "ew-resize" });

    down(canvas, 10, 10);
    // East handle anchored at minX=0, original edge maxX=1 → scale = wx.
    move(canvas, 800, 10);
    expect(model.trackByPath("a.mp3")!.tags.tagA).toBeCloseTo(0.4, 5);
    up(canvas, 800, 10);
    expect(deps.queueTagWrite).toHaveBeenCalledWith("a.mp3", "tagA", 0.4);
    expect(deps.flushTagWrites).toHaveBeenCalled();
  });
});

// ─── Wheel ────────────────────────────────────────────────

describe("CanvasInteractionController wheel", () => {
  it("zooms in on ctrl+wheel", () => {
    const { model, canvas } = makeHarness();
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    canvas.$canvas.dispatchEvent(
      wheelEvent({ x: 10, y: 10, ctrlKey: true, deltaY: -100 }),
    );
    expect(model.vp.zoom).toBeGreaterThan(1);
  });

  it("clamps zoom to the canvas minimum", () => {
    const { model, canvas } = makeHarness();
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    canvas.$canvas.dispatchEvent(
      wheelEvent({ x: 10, y: 10, ctrlKey: true, deltaY: 100000 }),
    );
    expect(model.vp.zoom).toBeCloseTo(0.1, 5);
  });

});

// ─── Folder drop ─────────────────────────────────────────

describe("CanvasInteractionController folder drop", () => {
  it("moves the selection into a dropped-on folder", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3", tags: { tagA: 0.5 } })]);
    model.axisX = "tagA";
    canvas.hitTest.mockReturnValue(0);

    down(canvas, 10, 10);
    move(canvas, 30, 30);
    move(canvas, 40, 40);

    const folderLabel = document.createElement("span");
    folderLabel.className = "folder-label";
    folderLabel.dataset.path = "House";
    vi.spyOn(document, "elementFromPoint").mockReturnValue(folderLabel);

    document.dispatchEvent(mouseEvent("mouseup", { x: 900, y: 900 }));

    // Baseline reverted before the move command fires.
    expect(model.trackByPath("a.mp3")!.tags.tagA).toBe(0.5);
    expect(deps.moveSelectionToFolder).toHaveBeenCalledWith("House");
    expect(deps.flushTagWrites).not.toHaveBeenCalled();
  });
});

// ─── Gesture cancellation / hover ─────────────────────────

describe("CanvasInteractionController gesture state", () => {
  it("reports pointer busy during a drag", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3", tags: { tagA: 0.5 } })]);
    model.axisX = "tagA";
    canvas.hitTest.mockReturnValue(0);
    expect(coord.isPointerBusy()).toBe(false);
    down(canvas, 10, 10);
    move(canvas, 30, 30);
    expect(coord.isPointerBusy()).toBe(true);
  });

  it("cancels an active box selection and reports it", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.hitTest.mockReturnValue(-1);
    down(canvas, 10, 10);
    expect(canvas.boxSel).not.toBeNull();
    expect(coord.cancelActiveGesture()).toBe(true);
    expect(canvas.boxSel).toBeNull();
  });

  it("returns false from cancel when no gesture is active", () => {
    const { coord } = makeHarness();
    expect(coord.cancelActiveGesture()).toBe(false);
  });

  it("resyncs hover only when idle and over the canvas", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.hitTest.mockReturnValue(0);
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    coord.resyncHoverFromLastPointer();
    expect(canvas.hoveredIdx).toBe(0);
    expect(canvas.scheduleDraw).toHaveBeenCalled();
  });

  it("opens a box scope on an X + drag with the default lens", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.hitTest.mockReturnValue(-1);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    down(canvas, 100, 100);
    move(canvas, 140, 140);
    up(canvas, 140, 140);
    expect(deps.openScope).toHaveBeenCalledTimes(1);
    const [scope, lensId] = (deps.openScope as ReturnType<typeof vi.fn>).mock
      .calls[0] as [
      { shape: string; cx: number; cy: number; rx: number; ry: number },
      string,
    ];
    expect(scope.shape).toBe("box");
    expect(scope.cx).toBeCloseTo(0.12, 5);
    expect(scope.cy).toBeCloseTo(0.12, 5);
    expect(scope.rx).toBeCloseTo(0.02, 5);
    expect(scope.ry).toBeCloseTo(0.02, 5);
    expect(lensId).toBe(model.scopeLensId);
  });

  it("closes an open scope on a bare X tap", () => {
    const { model, deps } = makeHarness();
    model.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "camelot" };
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyX", bubbles: true }),
    );
    expect(deps.closeScope).toHaveBeenCalledTimes(1);
  });

  it("cycles the scope lens on a bare X tap near the scope", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = { shape: "circle", cx: 0.1, cy: 0.1, r: 0.05, lensId: "camelot" };
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    move(canvas, 100, 100); // pointer at the scope centre
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyX", bubbles: true }),
    );
    // Near the scope a tap switches its lens instead of closing it.
    expect(deps.setScopeLens).toHaveBeenCalled();
    expect(deps.closeScope).not.toHaveBeenCalled();
  });

  it("cycles the scope lens backward on Shift+X near the scope", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = { shape: "circle", cx: 0.1, cy: 0.1, r: 0.05, lensId: "camelot" };
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    move(canvas, 100, 100);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyX", shiftKey: true, bubbles: true }),
    );
    expect(deps.setScopeLens).toHaveBeenCalled();
    expect(deps.closeScope).not.toHaveBeenCalled();
  });

  it("closes the scope on a bare X tap far outside it", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = { shape: "circle", cx: 0.1, cy: 0.1, r: 0.05, lensId: "camelot" };
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    move(canvas, 800, 800); // world (0.8, 0.8), far outside the scope
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyX", bubbles: true }),
    );
    expect(deps.closeScope).toHaveBeenCalledTimes(1);
    expect(deps.setScopeLens).not.toHaveBeenCalled();
  });

  it("opens or resizes a scope on X + scroll", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    canvas.$canvas.dispatchEvent(wheelEvent({ x: 20, y: 20, deltaY: -100 }));
    expect(deps.openScope).toHaveBeenCalledTimes(1);
    const [scope, lensId, allowEmpty] = (
      deps.openScope as ReturnType<typeof vi.fn>
    ).mock.calls[0] as [{ shape: string; r: number }, string, boolean];
    // Starts small in *screen* space (14 px / the 0.001 world-per-px fake), round,
    // and may be empty so the scroll can grow it in.
    expect(scope.shape).toBe("circle");
    expect(scope.r).toBeCloseTo(0.014, 5);
    expect(lensId).toBe(model.scopeLensId);
    expect(allowEmpty).toBe(true);

    model.scope = { shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "camelot" };
    canvas.$canvas.dispatchEvent(wheelEvent({ x: 20, y: 20, deltaY: -100 }));
    expect(deps.resizeScope).toHaveBeenCalledTimes(1);
    // A scroll up scales the scope up by a factor > 1.
    expect((deps.resizeScope as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBeGreaterThan(1);
  });

  it("closes the scope when clicking outside it", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = { shape: "circle", cx: 0.1, cy: 0.1, r: 0.05, lensId: "camelot" };
    canvas.hitTest.mockReturnValue(-1);
    canvas.scopeTabAt.mockReturnValue(null);
    down(canvas, 500, 500);
    expect(deps.closeScope).toHaveBeenCalledTimes(1);
  });

  it("switches the scope lens from a tab click", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = { shape: "circle", cx: 0.1, cy: 0.1, r: 0.05, lensId: "camelot" };
    canvas.scopeTabAt.mockReturnValue("clap");
    down(canvas, 500, 500);
    expect(deps.setScopeLens).toHaveBeenCalledWith("clap");
    expect(deps.closeScope).not.toHaveBeenCalled();
  });

  it("switches the main lens from the top lens tabber", () => {
    const { canvas, deps } = makeHarness();
    canvas.topLensTabAt.mockReturnValue("clap");
    down(canvas, 500, 20);
    expect(deps.switchLens).toHaveBeenCalledWith("clap");
    expect(deps.closeScope).not.toHaveBeenCalled();
  });

  it("sets the colour mode from the top colour tabber", () => {
    const { canvas, deps } = makeHarness();
    canvas.topColorTabAt.mockReturnValue("tempo");
    down(canvas, 500, 40);
    expect(deps.setColorMode).toHaveBeenCalledWith("tempo");
  });

  it("toggles the preview on a bare M tap", () => {
    const { deps } = makeHarness();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyM", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyM", bubbles: true }),
    );
    expect(deps.togglePreview).toHaveBeenCalledTimes(1);
  });

  it("adjusts the preview volume on M + scroll and suppresses the toggle", () => {
    const { canvas, deps } = makeHarness();
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyM", bubbles: true }),
    );
    canvas.$canvas.dispatchEvent(wheelEvent({ x: 20, y: 20, deltaY: -100 }));
    expect(deps.adjustPreviewVolume).toHaveBeenCalledTimes(1);
    // A scroll up scales the volume up by a factor > 1.
    expect(
      (deps.adjustPreviewVolume as ReturnType<typeof vi.fn>).mock.calls[0][0],
    ).toBeGreaterThan(1);
    expect(canvas.showVolumeHud).toHaveBeenCalled();
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyM", bubbles: true }),
    );
    // The scroll marked the gesture used, so the tap does not toggle the preview.
    expect(deps.togglePreview).not.toHaveBeenCalled();
    expect(deps.commitPreviewVolume).toHaveBeenCalledTimes(1);
  });

  it("moves the selection-bloom cursor on a bare D tap", () => {
    const { deps } = makeHarness();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyD", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyD", bubbles: true }),
    );
    expect(deps.moveSelectionBloom).toHaveBeenCalledWith(1);
    expect(deps.commitBloomRange).not.toHaveBeenCalled();
  });

  it("moves the bloom cursor backward on Shift+D", () => {
    const { deps } = makeHarness();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyD", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", {
        code: "KeyD",
        shiftKey: true,
        bubbles: true,
      }),
    );
    expect(deps.moveSelectionBloom).toHaveBeenCalledWith(-1);
  });

  it("adjusts the bloom amount on D + scroll and suppresses the move", () => {
    const { canvas, deps } = makeHarness();
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyD", bubbles: true }),
    );
    canvas.$canvas.dispatchEvent(wheelEvent({ x: 20, y: 20, deltaY: -100 }));
    expect(deps.adjustBloomRange).toHaveBeenCalledTimes(1);
    // A scroll up scales the amount up by a factor > 1.
    expect(
      (deps.adjustBloomRange as ReturnType<typeof vi.fn>).mock.calls[0][0],
    ).toBeGreaterThan(1);
    expect(canvas.showBloomHud).toHaveBeenCalled();
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyD", bubbles: true }),
    );
    expect(deps.moveSelectionBloom).not.toHaveBeenCalled();
    expect(deps.commitBloomRange).toHaveBeenCalledTimes(1);
  });

  it("sets the selection-bloom cursor from a colour-tab click", () => {
    const { canvas, deps } = makeHarness();
    canvas.topColorTabAt.mockReturnValue("cluster");
    down(canvas, 500, 40);
    expect(deps.setColorMode).toHaveBeenCalledWith("cluster");
    expect(deps.setSelectionBloom).toHaveBeenCalledWith("cluster");
  });

  it("disarms the scope gesture on X keyup", () => {
    const { canvas } = makeHarness();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    document.dispatchEvent(
      new KeyboardEvent("keyup", { code: "KeyX", bubbles: true }),
    );
    canvas.hitTest.mockReturnValue(-1);
    down(canvas, 10, 10);
    // Disarmed → a plain empty drag is a box selection, not a scope.
    expect(canvas.boxSel).not.toBeNull();
  });

  it("toggles the inspect card on the Alt key without ending the audition", () => {
    const { canvas, deps } = makeHarness();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Alt", bubbles: true }),
    );
    expect(canvas.toggleInspect).toHaveBeenCalled();
    // Unpinning only releases the card's hold; the hover audition keeps playing.
    expect(deps.stopPreview).not.toHaveBeenCalled();
  });

  it("ignores Alt while typing in a text field", () => {
    const { canvas } = makeHarness();
    const input = document.createElement("input");
    input.type = "text";
    document.body.appendChild(input);
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Alt", bubbles: true }),
    );
    expect(canvas.toggleInspect).not.toHaveBeenCalled();
  });

  it("unpins the inspect card on a plain background click", () => {
    const { canvas } = makeHarness();
    canvas.hitTest.mockReturnValue(-1);
    down(canvas, 10, 10);
    expect(canvas.unpinInspect).toHaveBeenCalled();
  });

  it("keeps the inspect card when shift-clicking to lasso", () => {
    const { canvas } = makeHarness();
    canvas.hitTest.mockReturnValue(-1);
    down(canvas, 10, 10, { shiftKey: true });
    expect(canvas.unpinInspect).not.toHaveBeenCalled();
  });

  it("pauses canvas hover while the inspect card is pinned", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "pinned.mp3", folder: "A" }),
      makeTrack({ path: "other.mp3", folder: "B" }),
    ]);
    // The pointer is over another track, but the pinned card keeps the hover.
    canvas.hitTest.mockReturnValue(1);
    canvas.inspect = { path: "pinned.mp3" };
    move(canvas, 10, 10);
    expect(canvas.hoveredIdx).toBe(0);
    expect(canvas.hoveredBloomTrack?.path).toBe("pinned.mp3");
    expect(deps.onHoverPreview).not.toHaveBeenCalled();
    expect(deps.onHoverFolderChanged).not.toHaveBeenCalled();
  });
});

// ─── 3-D orbit view ─────────────────────────────────────

describe("CanvasInteractionController 3-D orbit", () => {
  /** Flip the model into the 3-D orbit view: toggle on + a projection lens. */
  function set3D(model: Model): void {
    model.view3d.on = true;
    model.lensId = "combined";
  }

  it("orbits on Alt+left-drag in 3-D without touching the viewport", () => {
    const { model, canvas } = makeHarness();
    set3D(model);
    model.view3d.yaw = 0.6;
    model.view3d.pitch = 0.35;
    const vpBefore = { ...model.vp };

    down(canvas, 100, 100, { altKey: true });
    move(canvas, 200, 120); // +100 px x, +20 px y
    up(canvas, 200, 120, { altKey: true });

    // yaw += dx * k, pitch += dy * k (clamped).
    expect(model.view3d.yaw).toBeCloseTo(0.6 + 100 * ORBIT_RAD_PER_PX, 5);
    expect(model.view3d.pitch).toBeCloseTo(0.35 + 20 * ORBIT_RAD_PER_PX, 5);
    expect(model.vp).toEqual(vpBefore);
  });

  it("orbits on middle-drag in 3-D without touching the viewport", () => {
    const { model, canvas } = makeHarness();
    set3D(model);
    model.view3d.yaw = 0.6;
    model.view3d.pitch = 0.35;
    const vpBefore = { ...model.vp };

    down(canvas, 100, 100, { button: 1 });
    move(canvas, 150, 130);
    up(canvas, 150, 130, { button: 1 });

    expect(model.view3d.yaw).toBeCloseTo(0.6 + 50 * ORBIT_RAD_PER_PX, 5);
    expect(model.view3d.pitch).toBeCloseTo(0.35 + 30 * ORBIT_RAD_PER_PX, 5);
    expect(model.vp).toEqual(vpBefore);
  });

  it("pans on Alt+Shift+drag in 3-D without changing the orbit", () => {
    const { model, canvas } = makeHarness();
    set3D(model);
    model.view3d.yaw = 0.6;
    model.view3d.pitch = 0.35;
    const vpBefore = { ...model.vp };
    const orbitBefore = { ...model.view3d };

    down(canvas, 100, 100, { altKey: true, shiftKey: true });
    move(canvas, 200, 120, { altKey: true, shiftKey: true });
    up(canvas, 200, 120, { altKey: true, shiftKey: true });

    // Pan: vp.ox -= (wx2 - wx1); the fake maps 1 px → 0.001 world units.
    expect(model.vp.ox).toBeCloseTo(vpBefore.ox - 100 / 1000, 5);
    expect(model.vp.oy).toBeCloseTo(vpBefore.oy - 20 / 1000, 5);
    expect(model.view3d).toEqual(orbitBefore);
  });

  it("pans on Alt+drag in 2-D, leaving the orbit untouched", () => {
    const { model, canvas } = makeHarness(); // view3d.on is false by default
    model.view3d.yaw = 0.6;
    model.view3d.pitch = 0.35;
    const vpBefore = { ...model.vp };

    down(canvas, 100, 100, { altKey: true });
    move(canvas, 200, 120, { altKey: true });
    up(canvas, 200, 120, { altKey: true });

    expect(model.vp.ox).toBeCloseTo(vpBefore.ox - 100 / 1000, 5);
    expect(model.vp.oy).toBeCloseTo(vpBefore.oy - 20 / 1000, 5);
    expect(model.view3d).toEqual({ on: false, yaw: 0.6, pitch: 0.35 });
  });

  it("toggles 3-D from the on-canvas view tab without starting a gesture", () => {
    const { canvas, deps } = makeHarness();
    canvas.topViewTabAt.mockReturnValue("3d");
    down(canvas, 500, 60);
    expect(deps.toggleView3D).toHaveBeenCalledTimes(1);
    // Chrome: no gesture starts and selection logic never runs.
    expect(canvas.boxSel).toBeNull();
    expect(canvas.hitTest).not.toHaveBeenCalled();
    expect(canvas.$canvas.style.cursor).toBe("");
  });

  it("arms the scope gesture in 3-D and opens a sphere on X + wheel", () => {
    const { model, canvas, deps } = makeHarness();
    set3D(model);
    canvas.pickCursorWorld3D.mockReturnValue({
      wx: 0.4,
      wy: 0.3,
      wz: 0.7,
      snapped: true,
    });
    canvas.worldRadiusForPx.mockReturnValue(0.05);
    canvas.$canvas.dispatchEvent(mouseEvent("mouseenter"));
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    canvas.$canvas.dispatchEvent(wheelEvent({ x: 20, y: 20, deltaY: -100 }));
    expect(deps.openScope).toHaveBeenCalledTimes(1);
    const [scope, lensId, allowEmpty] = (
      deps.openScope as ReturnType<typeof vi.fn>
    ).mock.calls[0] as [
      { shape: string; cx: number; cy: number; cz: number; r: number },
      string,
      boolean,
    ];
    expect(scope.shape).toBe("sphere");
    expect(scope.cx).toBe(0.4);
    expect(scope.cy).toBe(0.3);
    expect(scope.cz).toBe(0.7);
    expect(scope.r).toBe(0.05);
    expect(lensId).toBe(model.scopeLensId);
    expect(allowEmpty).toBe(true);
    // The pick was at the cursor; the radius inverts the 14 px scroll start.
    expect(canvas.pickCursorWorld3D).toHaveBeenCalledWith(20, 20);
    expect(canvas.worldRadiusForPx).toHaveBeenCalledWith(0.4, 0.3, 0.7, 14);
  });

  it("opens a sphere scope on an X + drag from the snapped origin", () => {
    const { model, canvas, deps } = makeHarness();
    set3D(model);
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.pickCursorWorld3D.mockReturnValue({
      wx: 0.42,
      wy: 0.31,
      wz: 0.6,
      snapped: true,
    });
    canvas.worldRadiusForPx.mockReturnValue(0.03);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    down(canvas, 100, 100);
    // The drag preview carries the sphere origin snapped once at mouse-down.
    expect(canvas.scopeDrag).toEqual({
      x0: 100,
      y0: 100,
      x1: 100,
      y1: 100,
      sphere: { cx: 0.42, cy: 0.31, cz: 0.6 },
    });
    move(canvas, 140, 130);
    up(canvas, 140, 130);
    expect(deps.openScope).toHaveBeenCalledTimes(1);
    const [scope, lensId] = (deps.openScope as ReturnType<typeof vi.fn>).mock
      .calls[0] as [
      { shape: string; cx: number; cy: number; cz: number; r: number },
      string,
    ];
    expect(scope.shape).toBe("sphere");
    expect(scope.cx).toBe(0.42);
    expect(scope.cy).toBe(0.31);
    expect(scope.cz).toBe(0.6);
    expect(scope.r).toBe(0.03);
    expect(lensId).toBe(model.scopeLensId);
    // Radius = the drag distance in px (hypot(40, 30) = 50), inverted to world.
    expect(canvas.worldRadiusForPx).toHaveBeenCalledWith(0.42, 0.31, 0.6, 50);
  });

  it("ignores a 3-D scope drag shorter than 6 px", () => {
    const { model, canvas, deps } = makeHarness();
    set3D(model);
    canvas.pickCursorWorld3D.mockReturnValue({
      wx: 0.5,
      wy: 0.5,
      wz: 0.5,
      snapped: true,
    });
    document.dispatchEvent(
      new KeyboardEvent("keydown", { code: "KeyX", bubbles: true }),
    );
    down(canvas, 100, 100);
    move(canvas, 102, 101);
    up(canvas, 102, 101);
    expect(deps.openScope).not.toHaveBeenCalled();
  });

  it("closes a sphere scope on a click outside its screen circle, keeps one inside", () => {
    const { model, canvas, deps } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    model.scope = {
      shape: "sphere",
      cx: 0.5,
      cy: 0.5,
      cz: 0.5,
      r: 0.1,
      lensId: "camelot",
    };
    canvas.scopeScreenCircle.mockReturnValue({ cx: 500, cy: 250, r: 100 });
    canvas.scopeTabAt.mockReturnValue(null);
    canvas.hitTest.mockReturnValue(-1);
    // Inside the screen circle the click falls through to selection.
    down(canvas, 520, 250);
    expect(deps.closeScope).not.toHaveBeenCalled();
    // Outside the circle the scope closes.
    down(canvas, 700, 250);
    expect(deps.closeScope).toHaveBeenCalledTimes(1);
  });
});

// ─── Fit view ───────────────────────────────────────────

describe("CanvasInteractionController fit view", () => {
  it("resets the viewport when no folder is hovered", () => {
    const { model, deps, coord } = makeHarness();
    model.vp = { ox: 0.4, oy: 0.4, zoom: 2 };
    coord.fitView();
    expect(tweenTarget(deps)).toEqual({ ox: 0, oy: 0, zoom: 1 });
  });

  it("fits to the hovered folder disc", () => {
    const { model, canvas, deps, coord } = makeHarness();
    canvas.hoverTarget = { kind: "folder", prefix: "House" };
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "House/b.mp3", folder: "House" }),
    ]);
    canvas.worldPosForTrack.mockImplementation((t: Track) => ({
      wx: t.path.endsWith("a.mp3") ? 0.2 : 0.8,
      wy: 0.5,
    }));
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    coord.fitView();
    expect(canvas.scheduleDraw).toHaveBeenCalled();
    expect(tweenTarget(deps).zoom).not.toBe(1);
  });

  it("fits to the hovered cluster's members", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    model.clusterAssignments = new Map([
      ["a.mp3", 1],
      ["b.mp3", 1],
      ["c.mp3", 2],
    ]);
    canvas.hoverTarget = { kind: "cluster", id: 1 };
    canvas.worldPosForTrack.mockImplementation((t: Track) => ({
      wx: t.path === "a.mp3" ? 0.2 : t.path === "b.mp3" ? 0.8 : 0.1,
      wy: 0.5,
    }));
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    coord.fitView();
    expect(tweenTarget(deps).zoom).not.toBe(1);
  });

  it("fits to the tracks sharing a hovered key", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3", musical_key: "Am" }),
      makeTrack({ path: "b.mp3", musical_key: "Am" }),
      makeTrack({ path: "c.mp3", musical_key: "F" }),
    ]);
    canvas.hoverTarget = { kind: "key", value: "Am" };
    canvas.worldPosForTrack.mockImplementation((t: Track) => ({
      wx: t.path === "a.mp3" ? 0.2 : 0.8,
      wy: 0.5,
    }));
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    coord.fitView();
    expect(tweenTarget(deps).zoom).not.toBe(1);
  });

  it("falls back to the hovered dot's folder when no target is set", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "House/b.mp3", folder: "House" }),
    ]);
    canvas.hoveredIdx = 0;
    canvas.worldPosForTrack.mockImplementation((t: Track) => ({
      wx: t.path.endsWith("a.mp3") ? 0.2 : 0.8,
      wy: 0.5,
    }));
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    coord.fitView();
    expect(tweenTarget(deps).zoom).not.toBe(1);
  });
});

// ─── Fit view to result paths ────────────────────────────

describe("CanvasInteractionController fitViewToPaths", () => {
  it("pans a single track at the current zoom instead of zooming in", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.worldPosForTrack.mockReturnValue({ wx: 0.3, wy: 0.7 });
    model.vp = { ox: 0, oy: 0, zoom: 4 };
    coord.fitViewToPaths(["a.mp3"]);

    const target = tweenTarget(deps);
    // Zoom is retained; only the centre moves.
    expect(target.zoom).toBe(4);
    expect(target.ox).toBeCloseTo(0.3 - 1 / 8);
    expect(target.oy).toBeCloseTo(0.7 - 1 / 8);
  });

  it("fits a multi-track result set and tweens the move", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
    ]);
    canvas.worldPosForTrack.mockImplementation((t: Track) => ({
      wx: t.path === "a.mp3" ? 0.2 : 0.8,
      wy: 0.5,
    }));
    model.vp = { ox: 0, oy: 0, zoom: 1 };
    coord.fitViewToPaths(["a.mp3", "b.mp3"]);

    expect(tweenTarget(deps).zoom).toBeGreaterThan(1);
  });
});

// ─── Solo (isolate) ──────────────────────────────────────

describe("CanvasInteractionController solo", () => {
  it("isolates the hovered dot's bloom class", () => {
    const { model, canvas, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3", bpm: 128 }),
      makeTrack({ path: "b.mp3", bpm: 128 }),
      makeTrack({ path: "c.mp3", bpm: 150 }),
    ]);
    model.selectionBloom = "tempo";
    canvas.hoveredBloomTrack = model.trackByPath("a.mp3")!;
    const outcome = coord.solo();
    expect(model.isTrackHidden(model.trackByPath("c.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("b.mp3")!)).toBe(false);
    // The outcome names the rule the press isolated with and its keep-set.
    expect(outcome?.rule?.value).toBe("128 BPM");
    expect(outcome?.keep).toEqual(new Set(["a.mp3", "b.mp3"]));
  });

  it("isolates the hovered folder's whole subtree for folder colour", () => {
    const { model, canvas, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "House/sub/b.mp3", folder: "House/sub" }),
      makeTrack({ path: "Techno/c.mp3", folder: "Techno" }),
    ]);
    model.selectionBloom = "folder";
    canvas.hoveredBloomTrack = model.trackByPath("House/a.mp3")!;
    const outcome = coord.solo();
    expect(model.isTrackHidden(model.trackByPath("Techno/c.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("House/a.mp3")!)).toBe(false);
    expect(model.isTrackHidden(model.trackByPath("House/sub/b.mp3")!)).toBe(false);
    expect(outcome?.rule?.value).toBe("House");
  });

  it("isolates to a hovered cluster", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    model.clusterAssignments = new Map([
      ["a.mp3", 1],
      ["b.mp3", 1],
      ["c.mp3", 2],
    ]);
    canvas.hoverTarget = { kind: "cluster", id: 1 };
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("c.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("a.mp3")!)).toBe(false);
  });

  it("clears the exclusions when the hovered cluster is already soloed", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    model.clusterAssignments = new Map([
      ["a.mp3", 1],
      ["b.mp3", 2],
    ]);
    canvas.hoverTarget = { kind: "cluster", id: 1 };
    coord.solo();
    expect(model.hasExclusions).toBe(true);
    coord.solo();
    expect(model.hasExclusions).toBe(false);
  });

  it("isolates to a hovered folder and reveals it when fully hidden", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    canvas.hoverTarget = { kind: "folder", prefix: "House" };
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("Techno/b.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("House/a.mp3")!)).toBe(false);

    // Hide the folder entirely, then a second press restores and isolates it.
    coord.solo(); // already soloed → show all
    model.hideFolderTree("House");
    expect(model.isTrackHidden(model.trackByPath("House/a.mp3")!)).toBe(true);
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("House/a.mp3")!)).toBe(false);
    expect(model.isTrackHidden(model.trackByPath("Techno/b.mp3")!)).toBe(true);
  });

  it("isolates to the tracks sharing a hovered key", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3", musical_key: "Am" }),
      makeTrack({ path: "b.mp3", musical_key: "Am" }),
      makeTrack({ path: "c.mp3", musical_key: "F" }),
    ]);
    canvas.hoverTarget = { kind: "key", value: "Am" };
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("c.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("a.mp3")!)).toBe(false);
    expect(model.isTrackHidden(model.trackByPath("b.mp3")!)).toBe(false);
  });

  it("isolates tracks within the BPM tolerance of a hovered BPM chip", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "a.mp3", bpm: 128 }),
      makeTrack({ path: "b.mp3", bpm: 129.5 }),
      makeTrack({ path: "c.mp3", bpm: 140 }),
    ]);
    canvas.hoverTarget = { kind: "bpm", value: 128 };
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("c.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("a.mp3")!)).toBe(false);
    expect(model.isTrackHidden(model.trackByPath("b.mp3")!)).toBe(false);
  });

  it("shows everything when nothing is hovered and nothing is selected", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    model.hidePaths(["b.mp3"]);
    expect(model.hasExclusions).toBe(true);
    coord.solo();
    expect(model.hasExclusions).toBe(false);
  });

  it("still isolates the selection when nothing is hovered", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    model.selected.add("a.mp3");
    coord.solo();
    expect(model.isTrackHidden(model.trackByPath("b.mp3")!)).toBe(true);
    expect(model.isTrackHidden(model.trackByPath("a.mp3")!)).toBe(false);
  });

  it("returns null when a press reveals or clears instead of isolating", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    // Nothing hovered and nothing selected: S shows everything.
    expect(coord.solo()).toBeNull();
  });

  it("names a multi-folder selection's rule", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
      makeTrack({ path: "Disco/c.mp3", folder: "Disco" }),
    ]);
    model.selected.add("House/a.mp3");
    model.selected.add("Techno/b.mp3");
    const outcome = coord.solo();
    expect(outcome?.rule?.value).toBe("House, Techno");
    expect(model.isTrackHidden(model.trackByPath("Disco/c.mp3")!)).toBe(true);
  });
});

// ─── Read-only eval mode keeps hover playback ────────────────

describe("read-only eval mode keeps hover playback", () => {
  it("still drives the hover → audio chain under ?layout_eval=1", () => {
    const { model, canvas, deps, coord } = makeHarness();
    model.setEvalMode(true);
    setEvalReadOnly(true);
    model.setAllTracks([makeTrack({ path: "a.mp3" })]);
    canvas.hitTest.mockReturnValue(0);
    // Moving over a rendered dot must keep calling onHoverPreview → _handlePreview
    // → _startPreview → /api/audio, exactly as in normal mode. Read-only gating
    // must not disable hover to enforce itself.
    move(canvas, 20, 20);
    expect(deps.onHoverPreview).toHaveBeenCalled();
  });
});
