import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "./model";
import { Controller } from "./controller";
import { setEvalReadOnly } from "./lib/eval-readonly";
import type {
  BatchView,
  CanvasView,
  PropertiesView,
  SearchView,
  StatusView,
  TagPanelView,
  ViewModifierView,
} from "./components";

// FolderTreeCoordinator refreshes the tree through htmx; keep it inert in tests.
vi.mock("htmx.org", () => ({ default: { ajax: vi.fn() } }));

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

const SHELL_HTML = `
  <div id="toast-container"></div>
  <div id="eval-toolbar"></div>
  <audio id="preview-audio"></audio>
  <button id="btn-add-tag"></button>
  <div id="folder-section"><div id="folder-tree"></div></div>
  <div id="root-drop-zone" class="hidden"></div>
  <div id="selected-folders-overlay" class="hidden"></div>
  <div id="root-add-modal" class="hidden">
    <input id="root-add-input" />
    <input type="checkbox" id="root-add-external" />
    <button id="root-add-ok"></button>
    <button id="root-add-cancel"></button>
  </div>
  <button id="btn-add-root"></button>
`;

/**
 * The canvas surface the coordinators drive. A Proxy vends a fresh `vi.fn` for
 * any member not explicitly set, so the harness stays small while every
 * call-site member works; data members and the members the tests assert on are
 * set explicitly.
 */
interface FakeCanvas {
  $canvas: HTMLCanvasElement;
  $tip: HTMLElement;
  positions: { sx: number; sy: number; idx: number }[];
  animPositions: Map<string, { x: number; y: number }> | null;
  hoveredIdx: number;
  hoveredBloomTrack: Track | null;
  hoverTarget: { kind: string; path?: string; prefix?: string } | null;
  inspect: unknown;
  boxSel: unknown;
  lassoPoints: unknown[];
  intersectMode: boolean;
  dragGhosts: unknown;
  tipTimer: unknown;
  tipReady: boolean;
  _txWorld: unknown;
  hoveredPath: string | null;
  searchField: unknown;
  searchActive: boolean;
  hitTest: ReturnType<typeof vi.fn>;
  hitTestTransform: ReturnType<typeof vi.fn>;
  s2w: ReturnType<typeof vi.fn>;
  minZoom: ReturnType<typeof vi.fn>;
  worldPosForTrack: ReturnType<typeof vi.fn>;
  outerPosForTrack: ReturnType<typeof vi.fn>;
  scopeTabAt: ReturnType<typeof vi.fn>;
  topLensTabAt: ReturnType<typeof vi.fn>;
  topColorTabAt: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  resize: ReturnType<typeof vi.fn>;
  unpinInspect: ReturnType<typeof vi.fn>;
  toggleInspect: ReturnType<typeof vi.fn>;
  showVolumeHud: ReturnType<typeof vi.fn>;
  showBloomHud: ReturnType<typeof vi.fn>;
  setInspect: ReturnType<typeof vi.fn>;
  clearHover: ReturnType<typeof vi.fn>;
}

function makeCanvas(): FakeCanvas {
  const base: Record<PropertyKey, unknown> = {
    $canvas: document.createElement("canvas"),
    $tip: document.createElement("div"),
    positions: [],
    animPositions: null,
    hoveredIdx: -1,
    hoveredBloomTrack: null,
    hoverTarget: null,
    inspect: null,
    boxSel: null,
    lassoPoints: [],
    intersectMode: false,
    dragGhosts: null,
    tipTimer: 0,
    tipReady: false,
    _txWorld: null,
    hoveredPath: null,
    searchField: null,
    searchActive: false,
    hitTest: vi.fn(() => -1),
    hitTestTransform: vi.fn(() => null),
    s2w: vi.fn((sx: number, sy: number) => [sx / 1000, sy / 1000]),
    minZoom: vi.fn(() => 0.1),
    worldPosForTrack: vi.fn(() => ({ wx: 0.5, wy: 0.5 })),
    outerPosForTrack: vi.fn(() => ({ wx: 0.5, wy: 0.5 })),
    scopeTabAt: vi.fn(() => null),
    topLensTabAt: vi.fn(() => null),
    topColorTabAt: vi.fn(() => null),
    scheduleDraw: vi.fn(),
    resize: vi.fn(),
    unpinInspect: vi.fn(() => false),
    toggleInspect: vi.fn(),
    showVolumeHud: vi.fn(),
    showBloomHud: vi.fn(),
    setInspect: vi.fn(),
    clearHover: vi.fn(),
  };
  return new Proxy(base, {
    get(t, prop) {
      if (prop in t) return t[prop];
      return vi.fn();
    },
    set(t, prop, value) {
      t[prop] = value;
      return true;
    },
  }) as unknown as FakeCanvas;
}

/** The batch view callbacks the controller wires, driven directly by the tests. */
interface BatchLike {
  render: ReturnType<typeof vi.fn>;
  renderDirty: ReturnType<typeof vi.fn>;
  onApply: ((tag: string, updates: [string, number][]) => void) | null;
  onDragEnd: (() => void) | null;
  onClearTagFromSelection: ((tag: string) => void) | null;
}

function makeViews() {
  const canvas = makeCanvas();
  const tagPanel = {
    render: vi.fn(),
    scheduleRenameAfterRender: vi.fn(),
    onAxisChange: null,
    onFilterChange: null,
    onTagRename: null,
  };
  const featuresPanel = { ...tagPanel };
  const props = {
    render: vi.fn(),
    onDeselect: null,
    onRestore: null,
    onFocusRange: null,
    onHoverTrack: null,
    onHoverEnd: null,
  };
  const batch: BatchLike = {
    render: vi.fn(),
    renderDirty: vi.fn(),
    onApply: null,
    onDragEnd: null,
    onClearTagFromSelection: null,
  };
  const status = { update: vi.fn() };
  const search = {
    setQuery: vi.fn(),
    focusInput: vi.fn(),
    isOpen: vi.fn(() => false),
    close: vi.fn(),
    onSearch: null,
    onFocusResult: null,
    onCommit: null,
    onActiveChange: null,
    onHoverTrack: null,
    onHoverEnd: null,
  };
  const viewModifiers = {
    render: vi.fn(),
    onDismiss: null,
    onProximityInput: null,
    onExternalRefit: null,
    onExternalToggle: null,
  };
  return { canvas, tagPanel, featuresPanel, props, batch, status, search, viewModifiers };
}

interface Harness {
  model: Model;
  ctrl: Controller;
  canvas: FakeCanvas;
  batch: BatchLike;
}

function makeHarness(): Harness {
  const model = new Model();
  model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
  const views = makeViews();
  const ctrl = new Controller(
    model,
    views.canvas as unknown as CanvasView,
    views.tagPanel as unknown as TagPanelView,
    views.featuresPanel as unknown as TagPanelView,
    views.props as unknown as PropertiesView,
    views.batch as unknown as BatchView,
    views.status as unknown as StatusView,
    views.search as unknown as SearchView,
    views.viewModifiers as unknown as ViewModifierView,
  );
  return { model, ctrl, canvas: views.canvas, batch: views.batch };
}

const flush = () => new Promise<void>((r) => setTimeout(r, 0));

// ─── Event helpers (same recipe as the interaction tests) ────

type XY = { x?: number; y?: number };

function mouseEvent(type: string, opts: MouseEventInit & XY = {}): MouseEvent {
  const { x, y, ...init } = opts;
  const ev = new MouseEvent(type, { bubbles: true, cancelable: true, ...init });
  if (x !== undefined) Object.defineProperty(ev, "offsetX", { value: x });
  if (y !== undefined) Object.defineProperty(ev, "offsetY", { value: y });
  return ev;
}

function down(canvas: FakeCanvas, x: number, y: number) {
  canvas.$canvas.dispatchEvent(mouseEvent("mousedown", { x, y }));
}
function move(canvas: FakeCanvas, x: number, y: number) {
  canvas.$canvas.dispatchEvent(mouseEvent("mousemove", { x, y }));
}

/** Drag a dot and release it over a folder label — the folder-drop gesture. */
function dragToFolder(canvas: FakeCanvas, folder: string) {
  down(canvas, 10, 10);
  move(canvas, 30, 30);
  move(canvas, 40, 40);
  const folderLabel = document.createElement("span");
  folderLabel.className = "folder-label";
  folderLabel.dataset.path = folder;
  vi.spyOn(document, "elementFromPoint").mockReturnValue(folderLabel);
  document.dispatchEvent(mouseEvent("mouseup", { x: 900, y: 900 }));
}

beforeEach(() => {
  document.body.innerHTML = SHELL_HTML;
  localStorage.clear();
  setEvalReadOnly(false);
  vi.stubGlobal("requestAnimationFrame", vi.fn());
});

afterEach(() => {
  setEvalReadOnly(false);
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("Controller eval read-only tag-write guard", () => {
  it("does not mutate, queue or move while eval read-only is active", async () => {
    const h = makeHarness();
    const { model, batch, canvas } = h;
    const a = model.trackByPath("a.mp3")!;
    const b = model.trackByPath("b.mp3")!;
    const fetchSpy = vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    model.setEvalMode(true);
    setEvalReadOnly(true);

    // Batch apply: nothing is mutated.
    batch.onApply!("mood", [
      ["a.mp3", 0.7],
      ["b.mp3", 0.2],
    ]);
    expect(a.tags.mood).toBeUndefined();
    expect(b.tags.mood).toBeUndefined();

    // Clear tag: the stored value survives.
    a.tags.mood = 0.5;
    model.selected.add("a.mp3");
    await batch.onClearTagFromSelection!("mood");
    expect(a.tags.mood).toBe(0.5);

    // Folder drop: no move command runs (paths and folders stay put).
    a.tags = {};
    model.selected.clear();
    model.selected.add("a.mp3");
    canvas.hitTest.mockReturnValue(0);
    dragToFolder(canvas, "House");
    await flush();
    expect(a.folder).toBe("");

    // No write of any kind reached the server.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("still applies batch tags and moves tracks outside eval mode", async () => {
    const h = makeHarness();
    const { model, batch, canvas } = h;
    const a = model.trackByPath("a.mp3")!;
    const fetchSpy = vi.fn((url: string) => {
      const body = url.includes("/api/tracks/move")
        ? { ok: true, moved: 1, errors: [] }
        : { ok: true, count: 1, errors: [] };
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: "OK",
        json: () => Promise.resolve(body),
      });
    });
    vi.stubGlobal("fetch", fetchSpy);

    // Batch apply mutates and the debounced flush POSTs the queued updates.
    vi.useFakeTimers();
    batch.onApply!("mood", [["a.mp3", 0.7]]);
    expect(a.tags.mood).toBe(0.7);
    await vi.advanceTimersByTimeAsync(600);
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/tracks/tags",
      expect.objectContaining({ method: "POST" }),
    );
    vi.useRealTimers();

    // Folder drop runs the move command (a writable tag axis arms the drag).
    model.axisX = "tagA";
    a.tags = { tagA: 0.5 };
    model.selected.clear();
    model.selected.add("a.mp3");
    canvas.hitTest.mockReturnValue(0);
    dragToFolder(canvas, "House");
    await flush();
    expect(a.folder).toBe("House");
  });
});
