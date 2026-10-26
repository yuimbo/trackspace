import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Model,
  DEFAULT_CLUSTER_WEIGHTS,
  type MapPos,
  type Track,
  type ViewModifier,
} from "../model";
import { EmbeddingCoordinator, lerpPos } from "./embedding-coordinator";
import type { ViewModifierSink } from "./view-modifier-coordinator";

// ─── Fakes ───────────────────────────────────────────────────

/** In-memory stand-in for the active-view-modifier stack. */
function makeViewMods() {
  let seq = 0;
  const stack: ViewModifier[] = [];
  const clearers = new Map<string, () => void>();
  const sink: ViewModifierSink = {
    push: (mod, clear) => {
      const id = `vm-${++seq}`;
      stack.push({ ...mod, id });
      clearers.set(id, clear);
      return id;
    },
    update: (id, patch) => {
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack[i] = { ...stack[i], ...patch };
    },
    dismiss: (id) => {
      const clear = clearers.get(id);
      if (!clear) return;
      clearers.delete(id);
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack.splice(i, 1);
      clear();
    },
    remove: (id) => {
      if (!id) return;
      clearers.delete(id);
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack.splice(i, 1);
    },
    find: (kind) => stack.find((m) => m.kind === kind),
  };
  return { sink, stack };
}

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  closed = false;
  onerror: (() => void) | null = null;
  private _listeners: Record<string, ((e: MessageEvent) => void)[]> = {};

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, fn: EventListener): void {
    (this._listeners[type] ??= []).push(fn as (e: MessageEvent) => void);
  }

  emit(type: string, payload?: unknown): void {
    const ev = { data: JSON.stringify(payload) } as MessageEvent;
    for (const fn of this._listeners[type] ?? []) fn(ev);
  }

  close(): void {
    this.closed = true;
  }
}

function jsonResponse(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 500,
    statusText: ok ? "OK" : "Internal Server Error",
    json: () => Promise.resolve(body),
  } as Response;
}

/** Controllable rAF clock so the 350/500 ms lerps can be pumped deterministically. */
const raf = {
  now: 0,
  nextId: 1,
  pending: new Map<number, FrameRequestCallback>(),
  cancelled: [] as number[],
};

function installRaf(): void {
  raf.now = 0;
  raf.nextId = 1;
  raf.pending = new Map();
  raf.cancelled = [];
  vi.stubGlobal("performance", { now: () => raf.now });
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    const id = raf.nextId++;
    raf.pending.set(id, cb);
    return id;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    raf.cancelled.push(id);
    raf.pending.delete(id);
  });
}

/** Advance the clock and run any frame callbacks queued since the last pump. */
function pump(ms = 600): void {
  raf.now += ms;
  const cbs = [...raf.pending.values()];
  raf.pending.clear();
  for (const cb of cbs) cb(raf.now);
}

const HTML = `
  <div id="toast-container"></div>
  <div id="view-options" class="view-options hidden">
    <div id="head-confident-fold" class="view-option hidden"></div>
    <div id="cluster-weights-fold" class="view-option hidden">
      <details id="cluster-weights-details"></details>
    </div>
    <div id="genre-weights-fold" class="view-option hidden"></div>
    <div id="influences-fold" class="view-option hidden"></div>
    <div id="tsne-iteration-fold" class="view-option hidden"></div>
  </div>
`;

interface Harness {
  model: Model;
  update: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  canvas: {
    animPositions: Map<string, MapPos> | null;
    scheduleDraw: ReturnType<typeof vi.fn>;
    worldPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null;
    outerPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null;
  };
  coord: EmbeddingCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
  viewMods: ReturnType<typeof makeViewMods>;
}

const STATUS_OK = {
  model_ready: true,
  maest_model_ready: true,
  generating: false,
  layout_revision: "rev-1",
  sources: [
    { id: "features", label: "Audio features", done: 1, pending: 0, failed: 0 },
    { id: "rhythm", label: "Rhythm", done: 1, pending: 0, failed: 0 },
    { id: "structure", label: "Structure", done: 1, pending: 0, failed: 0 },
    { id: "maest", label: "MAEST", done: 1, pending: 0, failed: 0 },
    { id: "clap", label: "CLAP", done: 1, pending: 0, failed: 0 },
    { id: "supervised", label: "Supervised", done: 1, pending: 0, failed: 0 },
  ],
};

/** A status payload with the MAEST source left with *n* tracks pending. */
function withMaestPending(n: number): typeof STATUS_OK {
  return {
    ...STATUS_OK,
    sources: STATUS_OK.sources.map((s) =>
      s.id === "maest" ? { ...s, pending: n } : s,
    ),
  };
}

/** Route a URL to its response body (default: `{}`). */
function makeHarness(
  routes: (url: string) => unknown = () => ({}),
  renderProjectionSettings?: (
    settings: unknown,
    error?: string,
  ) => void,
): Harness {
  const model = new Model();
  const update = vi.fn();
  const scheduleDraw = vi.fn();
  const canvas = {
    animPositions: null as Map<string, MapPos> | null,
    scheduleDraw,
    // The real canvas resolves these from the embedding/tag lens; the tests only
    // need a stable outer position for scope membership and anchors.
    worldPosForTrack: (t: Track) => outerPos(model, t),
    outerPosForTrack: (t: Track) => outerPos(model, t),
  };
  const viewMods = makeViewMods();
  const coord = new EmbeddingCoordinator({
    model,
    status: { update },
    canvas,
    viewMods: viewMods.sink,
    renderProjectionSettings,
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return { model, update, scheduleDraw, canvas, coord, fetchSpy, viewMods };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Outer position for scope membership/anchors: the embedding map, else 0.5. */
function outerPos(m: Model, t: Track): { wx: number; wy: number; wz?: number } | null {
  const ep = m.embeddingPositions.get(t.path);
  return ep
    ? ep.z === undefined
      ? { wx: ep.x, wy: ep.y }
      : { wx: ep.x, wy: ep.y, wz: ep.z }
    : { wx: 0.5, wy: 0.5 };
}

function fetchCalls(fetchSpy: ReturnType<typeof vi.fn>, prefix: string): number {
  return fetchSpy.mock.calls.filter(([u]) => String(u).startsWith(prefix)).length;
}

const SETTINGS_OK = {
  ok: true,
  method: "tsne",
  dims: 2,
  recipe: "default",
  sources: [],
  weights_editable: true,
  weights_editable_keys: [],
};

function projectionRoute(url: string): unknown {
  if (url.startsWith("/api/embeddings/status")) return STATUS_OK;
  if (url.startsWith("/api/embeddings/projection-settings")) return SETTINGS_OK;
  if (url.startsWith("/api/embeddings/projection")) {
    return { positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }], revision: "rev-1" };
  }
  return {};
}

beforeEach(() => {
  document.body.innerHTML = HTML;
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  installRaf();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ─── Projection fetch + layout cache ─────────────────────────

describe("EmbeddingCoordinator projection fetch", () => {
  it("fetches a layout, applies positions, and reports ready", async () => {
    const { model, coord, canvas } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    expect(model.embeddingPositions.get("a.mp3")).toEqual({ x: 0.2, y: 0.3 });
    expect(model.embeddingsReady).toBe(true);
    expect(canvas.scheduleDraw).toHaveBeenCalled();
  });

  it("reuses a cached layout when the revision matches", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    await coord.ensureAndProject();
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(1);
  });

  it("refetches after invalidateLayoutCache", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    coord.invalidateLayoutCache();
    await coord.ensureAndProject();
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(2);
  });

  it("omits the weight param when weights are at their defaults", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(call![0])).not.toContain("w=");
  });

  it("genre_full lens requests the canonical recipe, not sources/weights", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return { ...STATUS_OK, genre_full_enabled: true };
      }
      return projectionRoute(url);
    });
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(call![0]);
    expect(url).toContain("recipe=genre_full");
    expect(url).not.toContain("sources=");
    expect(url).not.toContain("w=");
  });

  it("genre_full sends the 7-key w override only when weights deviate from the defaults", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return { ...STATUS_OK, genre_full_enabled: true };
      }
      return projectionRoute(url);
    });
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    model.genreWeights = { ...model.genreWeights, rhythm: 0.3 };
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(call![0]);
    expect(url).toContain("recipe=genre_full");
    // Canonical 7-key order, features excluded.
    expect(url).toContain(
      "w=maest_logits:0.4,maest:0.2,maest_layer7:0.25,mert_l19:0.15,rhythm:0.3,clap:0,supervised:0",
    );
  });

  it("reconciles a legacy active lens to the genre map when the server enables genre", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return { ...STATUS_OK, genre_full_enabled: true };
      }
      return projectionRoute(url);
    });
    model.genreFullEnabled = false;
    model.lensId = "combined";
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    expect(model.genreFullEnabled).toBe(true);
    // The symmetric conversion: an active legacy map converts to the genre map.
    expect(model.lensId).toBe("genre_full");
    const proj = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(proj![0])).toContain("recipe=genre_full");
  });

  it("falls back from the genre map to combined when the server disables genre", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    // STATUS_OK has no genre_full_enabled → the server disabled genre.
    expect(model.genreFullEnabled).toBe(false);
    expect(model.lensId).toBe("combined");
    const proj = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(proj![0])).not.toContain("recipe=genre_full");
  });

  it("sends the tsne iteration override only when set", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    await coord.ensureAndProject();
    const first = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(first![0])).not.toContain("tsne_n_iter");

    coord.setTsneIteration(250, 150);
    await flush();
    const calls = fetchSpy.mock.calls.filter(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(calls[calls.length - 1][0]);
    expect(url).toContain("tsne_n_iter=250");
    expect(url).toContain("tsne_early_exaggeration_iter=150");
  });

  it("setTsneIteration mutates the model and refetches the layout", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    await coord.ensureAndProject();
    const before = fetchCalls(fetchSpy, "/api/embeddings/projection");
    coord.setTsneIteration(500, 250);
    expect(model.tsneIterationFor()).toEqual({ nIter: 500, early: 250 });
    await flush();
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBeGreaterThan(
      before,
    );
  });

  it("keeps the tsne budget per dimensionality", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    // Set a budget on the 2-D genre view.
    coord.setTsneIteration(250, 150);
    expect(model.tsneIterationFor()).toEqual({ nIter: 250, early: 150 });
    // Toggling to 3-D reads an empty (recipe-default) budget — 2-D must not spill.
    model.view3d.on = true;
    expect(model.tsneIterationFor()).toEqual({ nIter: null, early: null });
    // Setting a 3-D budget does not touch the 2-D one.
    coord.setTsneIteration(500, 250);
    model.view3d.on = false;
    expect(model.tsneIterationFor()).toEqual({ nIter: 250, early: 150 });
  });

  it("binds the iteration control to the 50/0 default and accepts 50", () => {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<select id="tsne-iteration-preset">
         <option value="default">Default · 50 / 0</option>
         <option value="fast">Fast · 100 / 100</option>
         <option value="custom" disabled>Custom</option>
       </select>
       <input id="tsne-n-iter" type="number" min="50" />
       <input id="tsne-early-iter" type="number" min="0" />`,
    );
    const { model, coord } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    coord.bindTsneIterationControl();

    const $preset = document.getElementById(
      "tsne-iteration-preset",
    ) as HTMLSelectElement;
    const $nIter = document.getElementById("tsne-n-iter") as HTMLInputElement;

    // A fresh model sends no override: the server uses the recipe's 50/0 budget.
    expect($preset.value).toBe("default");
    expect($nIter.value).toBe("");

    // 50 is the production default and is accepted as an explicit override.
    $nIter.value = "50";
    $nIter.dispatchEvent(new Event("change"));
    expect(model.tsneIterationFor()).toEqual({ nIter: 50, early: null });
    // An explicit budget is remembered as Custom, not mislabelled Default.
    expect($preset.value).toBe("custom");

    // Below the floor is ignored rather than clamped or sent.
    $nIter.value = "49";
    $nIter.dispatchEvent(new Event("change"));
    expect(model.tsneIterationFor()).toEqual({ nIter: 50, early: null });
  });

  it("makes Default clear the override after a preset, and rejects invalid fields", () => {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<select id="tsne-iteration-preset">
         <option value="default">Default · 50 / 0</option>
         <option value="fast">Fast · 100 / 100</option>
         <option value="balanced">Balanced · 250 / 150</option>
         <option value="gallery">Gallery · 500 / 250</option>
         <option value="custom" disabled>Custom</option>
       </select>
       <input id="tsne-n-iter" type="number" min="50" max="2000" />
       <input id="tsne-early-iter" type="number" min="0" max="1000" />`,
    );
    const { model, coord } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    coord.bindTsneIterationControl();

    const $preset = document.getElementById(
      "tsne-iteration-preset",
    ) as HTMLSelectElement;
    const $nIter = document.getElementById("tsne-n-iter") as HTMLInputElement;
    const $early = document.getElementById("tsne-early-iter") as HTMLInputElement;

    $preset.value = "fast";
    $preset.dispatchEvent(new Event("change"));
    expect(model.tsneIterationFor()).toEqual({ nIter: 100, early: 100 });
    expect($preset.value).toBe("fast");

    // ``default`` maps to null, which used to be swallowed by a truthiness guard.
    $preset.value = "default";
    $preset.dispatchEvent(new Event("change"));
    expect(model.tsneIterationFor()).toEqual({ nIter: null, early: null });
    expect($preset.value).toBe("default");
    expect($nIter.value).toBe("");

    // Rejected values show the stored budget again instead of the bad text.
    $preset.value = "fast";
    $preset.dispatchEvent(new Event("change"));
    for (const bad of ["49", "2001", "100.5"]) {
      $nIter.value = bad;
      $nIter.dispatchEvent(new Event("change"));
      expect(model.tsneIterationFor()).toEqual({ nIter: 100, early: 100 });
      expect($nIter.value).toBe("100");
    }
    $early.value = "1001";
    $early.dispatchEvent(new Event("change"));
    expect(model.tsneIterationFor()).toEqual({ nIter: 100, early: 100 });
    expect($early.value).toBe("100");
  });

  it("hides the iteration fold on a tags lens and remembers a custom budget across dims", () => {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<select id="tsne-iteration-preset">
         <option value="default">Default · 50 / 0</option>
         <option value="fast">Fast · 100 / 100</option>
         <option value="custom" disabled>Custom</option>
       </select>
       <input id="tsne-n-iter" type="number" min="50" max="2000" />
       <input id="tsne-early-iter" type="number" min="0" max="1000" />`,
    );
    const { model, coord } = makeHarness(projectionRoute);
    model.lensId = "tags";
    coord.syncLensChrome();
    const $tsneFold = document.getElementById("tsne-iteration-fold")!;
    const $influences = document.getElementById("influences-fold")!;
    const $opts = document.getElementById("view-options")!;
    expect($tsneFold.classList.contains("hidden")).toBe(true);
    expect($influences.classList.contains("hidden")).toBe(true);
    expect($opts.classList.contains("hidden")).toBe(true);

    model.lensId = "combined";
    coord.bindTsneIterationControl();
    coord.syncLensChrome();
    expect($tsneFold.classList.contains("hidden")).toBe(false);
    expect($influences.classList.contains("hidden")).toBe(false);

    const $preset = document.getElementById(
      "tsne-iteration-preset",
    ) as HTMLSelectElement;
    const $nIter = document.getElementById("tsne-n-iter") as HTMLInputElement;
    coord.setTsneIteration(300, 20);
    coord.syncLensChrome();
    expect($preset.value).toBe("custom");
    expect($nIter.value).toBe("300");

    // 3-D has its own empty budget; switching back restores the 2-D custom one.
    model.view3d.on = true;
    coord.syncLensChrome();
    expect($preset.value).toBe("default");
    model.view3d.on = false;
    coord.syncLensChrome();
    expect($preset.value).toBe("custom");
    expect($nIter.value).toBe("300");
  });

  it("includes externals whenever external roots exist, even when hidden", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    // Hidden on purpose: their positions must already ride along so the
    // visibility toggle is a redraw rather than a refetch.
    model.setExternalRoots(["ext"]);
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(call![0]);
    expect(url).toContain("include_external=1");
    // The fit stays on the library; the server places externals onto it.
    expect(url).not.toContain("fit_scope=all");
  });

  it("omits include_external when no external roots exist", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(call![0])).not.toContain("include_external=1");
  });

  it("refits with externals only when opted in", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    model.setExternalRoots(["ext"]);
    model.setShowExternal(true);
    model.setExternalInProjection(true);
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(call![0]);
    expect(url).toContain("include_external=1");
    expect(url).toContain("fit_scope=all");
  });

  it("encodes non-default weights into the layout query", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.5 };
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    const url = String(call![0]);
    expect(url).toContain("w=maest_logits:0.45,maest:0.25,rhythm:0.5");
  });

  it("clears the loading pill when a cached lens supersedes an in-flight fit", async () => {
    let releaseClap: (() => void) | null = null;
    const { model, coord } = makeHarness(projectionRoute);
    const route = globalThis.fetch;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes("sources=clap")) {
          return new Promise<Response>((resolve) => {
            releaseClap = () =>
              resolve(
                jsonResponse({
                  positions: [{ path: "a.mp3", x: 0.9, y: 0.9 }],
                  revision: "rev-1",
                }),
              );
          });
        }
        return route(input, init);
      }),
    );

    model.viewMode = "embeddings";
    await coord.ensureAndProject(); // caches the combined lens
    expect(model.projectionPending).toBe(false);

    const slow = coord.switchToLens("clap"); // uncached: fit left in flight
    await flush();
    expect(model.projectionPending).toBe(true);

    await coord.switchToLens("combined"); // cache hit supersedes the fit
    expect(model.projectionPending).toBe(false);

    releaseClap!();
    await slow;
    expect(model.projectionPending).toBe(false);
  });

  it("refreshes the layout in place when weights change", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const before = fetchCalls(fetchSpy, "/api/embeddings/projection");

    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.5 };
    coord.refreshProjection();
    await flush();

    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(before + 1);
  });

  it("fetches the weight Jacobian once per weight vector", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.01, 0.02]] },
            revision: "r",
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const jacCalls = () =>
      fetchSpy.mock.calls.filter(([u]) =>
        String(u).includes("jacobian=1"),
      ).length;

    await coord.fetchWeightJacobian();
    await coord.fetchWeightJacobian();
    expect(jacCalls()).toBe(1);

    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.5 };
    await coord.fetchWeightJacobian();
    expect(jacCalls()).toBe(2);
  });

  it("drops a stale Jacobian response when a newer fetch supersedes it", async () => {
    let releaseOld!: (v: Response) => void;
    const { model, coord } = makeHarness();
    const base = globalThis.fetch;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes("jacobian=1")) {
          // The first (old) request is held; a later one resolves immediately.
          if (!releaseOld) {
            return new Promise<Response>((r) => (releaseOld = r));
          }
          return Promise.resolve(
            jsonResponse({
              positions: [{ path: "a.mp3", x: 0.2, y: 0.2 }],
              weights: { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.5 },
              jacobian: { rhythm: [[0.1, 0.1]] },
            }),
          );
        }
        return base(input, init);
      }),
    );
    model.viewMode = "embeddings";
    model.embeddingsReady = true;
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.2 }]]);

    const old = coord.fetchWeightJacobian();
    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.5 };
    const newer = coord.fetchWeightJacobian();
    await newer;

    // The stale first response lands after the newer one: its base (rhythm-less)
    // must not overwrite the current Jacobian.
    releaseOld(
      jsonResponse({
        positions: [{ path: "a.mp3", x: 0.9, y: 0.9 }],
        weights: { ...DEFAULT_CLUSTER_WEIGHTS },
        jacobian: { maest: [[0.5, 0.5]] },
      }),
    );
    await old;
    const cached = coord["_weightJacobian"]!;
    expect(cached.weights.rhythm).toBe(0.5);
    expect(cached.jacobian.rhythm).toBeDefined();
    expect(cached.jacobian.maest).toBeUndefined();
  });

  it("refetches the Jacobian after an apply clears it", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.01, 0.02]] },
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    model.embeddingsReady = true;
    model.embeddingPositions = new Map([["a.mp3", { x: 0.1, y: 0.2 }]]);
    await coord.fetchWeightJacobian();
    expect(
      fetchSpy.mock.calls.filter(([u]) => String(u).includes("jacobian=1")).length,
    ).toBe(1);

    coord.invalidateWeightJacobian();
    await coord.fetchWeightJacobian();
    expect(
      fetchSpy.mock.calls.filter(([u]) => String(u).includes("jacobian=1")).length,
    ).toBe(2);
  });

  it("invalidateLayoutCache also drops the cached weight Jacobian", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.01, 0.02]] },
            revision: "r",
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    await coord.fetchWeightJacobian();
    const jacCalls = () =>
      fetchSpy.mock.calls.filter(([u]) => String(u).includes("jacobian=1")).length;
    expect(jacCalls()).toBe(1);
    // A layout-revision change (SSE done / library reload) drops the Jacobian too:
    // its base is tied to the old layout.
    coord.invalidateLayoutCache();
    await coord.fetchWeightJacobian();
    expect(jacCalls()).toBe(2);
  });

  it("drops a Jacobian response whose baseline revision is stale", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.01, 0.02]] },
            revision: "rev-OLD", // older than the layout currently in hand
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "rev-NEW" },
    );
    model.viewMode = "embeddings";
    await coord.ensureAndProject(); // caches revision "rev-NEW"
    await coord.fetchWeightJacobian();
    // The response's baseline (rev-OLD) is not the layout in hand: not cached,
    // so a drag cannot approximate around a stale frame.
    expect(coord["_weightJacobian"]).toBeNull();
    expect(
      coord.applyApproximateWeights({ ...DEFAULT_CLUSTER_WEIGHTS, maest: 0.6 }),
    ).toBe(false);
  });

  it("does not apply a Jacobian whose baseline revision no longer matches the layout", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.1, y: 0.2 }]]);
    const qs = coord["_projectionQueryString"]();
    coord["_jacobianKey"] = qs;
    coord["_weightJacobian"] = {
      weights: { ...DEFAULT_CLUSTER_WEIGHTS },
      paths: ["a.mp3"],
      base: [{ x: 0.1, y: 0.2 }],
      jacobian: { maest: [{ dx: 0.01, dy: 0.02 }] },
      revision: "rev-1",
    };
    // The layout moved on (same query string, a new revision): the Jacobian's
    // base is stale, so the drag must not be applied.
    coord["_layoutCache"].set(qs, {
      revision: "rev-2",
      positions: new Map([["a.mp3", { x: 0.1, y: 0.2 }]]),
    });
    expect(
      coord.applyApproximateWeights({ ...DEFAULT_CLUSTER_WEIGHTS, maest: 0.6 }),
    ).toBe(false);
    expect(model.embeddingPositions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2 });
  });

  it("restores the exact applied baseline from the layout cache without a round-trip", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject(); // caches the applied layout
    const exact = new Map(model.embeddingPositions);
    // A drag approximated the positions on top of the applied baseline.
    model.embeddingPositions = new Map([["a.mp3", { x: 0.9, y: 0.9 }]]);
    const projBefore = () =>
      fetchSpy.mock.calls.filter(([u]) =>
        String(u).startsWith("/api/embeddings/projection"),
      ).length;
    const before = projBefore();
    expect(coord.restoreExactWeightBaseline()).toBe(true);
    expect(model.embeddingPositions.get("a.mp3")).toEqual(exact.get("a.mp3"));
    // Restored locally — no HTTP round-trip.
    expect(projBefore()).toBe(before);
  });
});

// ─── 3-D orbit view ─────────────────────────────────────────

describe("EmbeddingCoordinator 3-D view", () => {
  it("appends dims=3 to the projection URL only when 3-D is on", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    let proj = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.startsWith("/api/embeddings/projection"));
    expect(proj.at(-1)!).not.toContain("dims=3");

    model.view3d.on = true;
    await coord.ensureAndProject();
    proj = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.startsWith("/api/embeddings/projection"));
    expect(proj.at(-1)!).toContain("dims=3");
  });

  it("keeps z from the server in embeddingPositions", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [{ path: "a.mp3", x: 0.2, y: 0.3, z: 0.7 }],
            revision: "rev-1",
          }
        : STATUS_OK,
    );
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    expect(model.embeddingPositions.get("a.mp3")).toEqual({
      x: 0.2,
      y: 0.3,
      z: 0.7,
    });
  });

  it("setView3D refetches with dims=3, back without, and revisits hit the cache", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.3 }]]);

    await coord.setView3D(true);
    let proj = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.startsWith("/api/embeddings/projection"));
    expect(proj.at(-1)!).toContain("dims=3");
    expect(model.view3d.on).toBe(true);

    await coord.setView3D(false);
    proj = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.startsWith("/api/embeddings/projection"));
    expect(proj.at(-1)!).not.toContain("dims=3");
    expect(model.view3d.on).toBe(false);

    const before = proj.length;
    await coord.setView3D(true); // the 3-D layout is cached now: no extra fetch
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(before);
  });

  it("lerpPos lerps x/y, and z only when either side has one", () => {
    expect(lerpPos({ x: 0, y: 0 }, { x: 1, y: 1 }, 0.5)).toEqual({
      x: 0.5,
      y: 0.5,
    });
    expect(
      lerpPos({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 }, 0.5),
    ).toEqual({ x: 0.5, y: 0.5, z: 0.5 });
    // A missing side reads as the cube centre (0.5).
    expect(lerpPos({ x: 0, y: 0 }, { x: 1, y: 1, z: 1 }, 0.5)).toEqual({
      x: 0.5,
      y: 0.5,
      z: 0.75,
    });
  });

  it("lerpPos carries a shared billboard anchor through a same-frame lerp", () => {
    const bb = { cx: 0.5, cy: 0.5, cz: 0.5 };
    expect(
      lerpPos(
        { x: 0.4, y: 0.3, billboard: bb },
        { x: 0.6, y: 0.5, billboard: bb },
        0.5,
      ),
    ).toEqual({ x: 0.5, y: 0.4, billboard: bb });
    // A different anchor is a different frame → cross-frame blend.
    const other = { cx: 0.6, cy: 0.4, cz: 0.5 };
    const blended = lerpPos(
      { x: 0.4, y: 0.3, billboard: bb },
      { x: 0.6, y: 0.5, billboard: other },
      0.5,
    );
    expect(blended).toEqual({
      x: 0.6,
      y: 0.5,
      billboard: other,
      from: { x: 0.4, y: 0.3, billboard: bb },
      mix: 0.5,
    });
  });

  it("lerpPos blends different frames in view space via from/mix", () => {
    const outer = { x: 0.2, y: 0.3, z: 0.7 };
    const scope = { x: 0.6, y: 0.5, billboard: { cx: 0.5, cy: 0.5, cz: 0.5 } };
    // e = 0 → entirely the start frame (mix 1); e = 1 → exactly the target (mix 0).
    expect(lerpPos(outer, scope, 0)).toEqual({
      x: 0.6,
      y: 0.5,
      billboard: scope.billboard,
      from: outer,
      mix: 1,
    });
    expect(lerpPos(outer, scope, 1)).toEqual({
      x: 0.6,
      y: 0.5,
      billboard: scope.billboard,
      from: outer,
      mix: 0,
    });
    expect(lerpPos(outer, scope, 0.25)).toEqual({
      x: 0.6,
      y: 0.5,
      billboard: scope.billboard,
      from: outer,
      mix: 0.75,
    });
    // The reverse (close) keeps the billboard on the `from` side and the target
    // has no z (a flat outer map).
    expect(lerpPos(scope, outer, 0.25)).toEqual({
      x: 0.2,
      y: 0.3,
      z: 0.7,
      from: { x: 0.6, y: 0.5, billboard: scope.billboard },
      mix: 0.75,
    });
  });

  it("carries z through the position animation overlay", async () => {
    const { model, coord, canvas } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [{ path: "a.mp3", x: 0.2, y: 0.3, z: 0.7 }],
            revision: "rev-1",
          }
        : STATUS_OK,
    );
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    await coord.setView3D(true);
    pump(275); // half of the 550 ms toggle tween: ease = 0.5
    expect(canvas.animPositions!.get("a.mp3")).toEqual({
      x: expect.closeTo(0.35, 5),
      y: expect.closeTo(0.4, 5),
      z: expect.closeTo(0.6, 5),
    });
  });

  it("tweens the camera blend to 1 on and back to 0 off", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.3 }]]);
    expect(model.view3dBlend).toBe(0);

    await coord.setView3D(true);
    expect(model.view3dBlend).toBe(0); // the tween has only just started
    pump(275); // half of the 550 ms blend tween: ease = 0.5
    expect(model.view3dBlend).toBeCloseTo(0.5, 5);
    pump(600);
    expect(model.view3dBlend).toBe(1);

    await coord.setView3D(false);
    pump(600);
    expect(model.view3dBlend).toBe(0);
  });

  it("re-toggling mid-tween continues from the current blend", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.3 }]]);

    await coord.setView3D(true);
    pump(137.5); // blend ≈ 0.125 (ease at t=0.25)
    const mid = model.view3dBlend;
    expect(mid).toBeGreaterThan(0.1);
    expect(mid).toBeLessThan(0.2);

    await coord.setView3D(false);
    // The new tween restarts from `mid`, not from 1 (nor from 0): half of the
    // way back is half of the current value.
    pump(275);
    expect(model.view3dBlend).toBeCloseTo(mid / 2, 5);
    pump(600);
    expect(model.view3dBlend).toBe(0);
  });

  it("a position animation started mid-tween does not cancel the blend", async () => {
    const { model, coord, canvas } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.3 }]]);

    await coord.setView3D(true); // starts the blend tween AND the position tween
    pump(137.5);
    const blendAtRestart = model.view3dBlend;
    expect(blendAtRestart).toBeGreaterThan(0);

    // A cache-hit refetch restarts the POSITION animation on the shared frame;
    // the blend tween must keep running on its own frame.
    await coord.refreshProjection();
    pump(137.5);
    expect(model.view3dBlend).toBeGreaterThan(blendAtRestart);
    pump(600);
    expect(model.view3dBlend).toBe(1);
    expect(canvas.animPositions).toBeNull();
  });

  it("sets the blend instantly for a non-projection lens", async () => {
    const { model, coord } = makeHarness();
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    expect(model.lens().kind).toBe("tags");

    await coord.setView3D(true);
    expect(model.view3dBlend).toBe(1);
    expect(raf.pending.size).toBe(0); // no tween frame was started

    await coord.setView3D(false);
    expect(model.view3dBlend).toBe(0);
  });

  /** 2-D and 3-D layouts that differ, so a toggle lerp moves x/y/z visibly. */
  const dimsRoute = (url: string): unknown => {
    if (url.startsWith("/api/embeddings/status")) return STATUS_OK;
    if (url.startsWith("/api/embeddings/projection")) {
      return String(url).includes("dims=3")
        ? {
            positions: [
              { path: "a.mp3", x: 0.5, y: 0.6, z: 0.7 },
              { path: "b.mp3", x: 0.9, y: 0.7, z: 0.1 },
            ],
            revision: "rev-1",
          }
        : {
            positions: [
              { path: "a.mp3", x: 0.2, y: 0.3 },
              { path: "b.mp3", x: 0.8, y: 0.9 },
            ],
            revision: "rev-1",
          };
    }
    return {};
  };

  it("holds the camera still until the toggled layout lands, then runs blend and positions together", async () => {
    const { model, coord, canvas } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([
      ["a.mp3", { x: 0.5, y: 0.5 }],
      ["b.mp3", { x: 0.1, y: 0.1 }],
    ]);

    let resolveProjection!: (v: unknown) => void;
    const base = globalThis.fetch;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).startsWith("/api/embeddings/projection")) {
          return new Promise<Response>((resolve) => {
            resolveProjection = (v: unknown) => resolve(jsonResponse(v));
          });
        }
        return base(input, init);
      }),
    );

    const toggling = coord.setView3D(true);
    // The flag flipped, but with the 3-D layout still in flight the camera has
    // not started: the blend stays 0 and no tween frame is queued.
    expect(model.view3d.on).toBe(true);
    expect(model.view3dBlend).toBe(0);
    expect(raf.pending.size).toBe(0);

    pump(600); // a long cold fit passes — still flat, still no tween
    expect(model.view3dBlend).toBe(0);
    expect(raf.pending.size).toBe(0);

    resolveProjection({
      positions: [
        { path: "a.mp3", x: 0.2, y: 0.3, z: 0.7 },
        { path: "b.mp3", x: 0.8, y: 0.9, z: 0.1 },
      ],
      revision: "rev-1",
    });
    await toggling;

    // Both animations started on the landed layout, same tick, same duration:
    // half way, the blend and the dot lerp are both at ease(0.5).
    pump(275); // half of the 550 ms toggle
    expect(model.view3dBlend).toBeCloseTo(0.5, 5);
    const mid = canvas.animPositions!.get("a.mp3")!;
    expect(mid.x).toBeCloseTo(0.35, 5);
    expect(mid.y).toBeCloseTo(0.4, 5);
    expect(mid.z).toBeCloseTo(0.6, 5);

    pump(600);
    expect(model.view3dBlend).toBe(1);
    expect(canvas.animPositions).toBeNull();
  });

  it("starts both tweens immediately on a cache-hit toggle", async () => {
    const { model, coord, canvas } = makeHarness(dimsRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    await coord.ensureAndProject(); // 2-D layout cached
    await coord.setView3D(true); // 3-D layout fetched + cached, tween runs out
    pump(600);
    expect(model.view3dBlend).toBe(1);
    await coord.setView3D(false); // 2-D cache hit, tween runs out
    pump(600);
    expect(model.view3dBlend).toBe(0);
    expect(canvas.animPositions).toBeNull();

    // Toggle back on: the cached 3-D layout needs no round trip, so the camera
    // blend and the position lerp start in the same tick as the toggle itself.
    const toggling = coord.setView3D(true);
    expect(raf.pending.size).toBe(2); // blend frame + position frame
    await toggling;

    pump(275); // half of the 550 ms toggle: ease = 0.5
    expect(model.view3dBlend).toBeCloseTo(0.5, 5);
    const mid = canvas.animPositions!.get("a.mp3")!;
    expect(mid.x).toBeCloseTo(0.35, 5); // 2-D (0.2,0.3) → 3-D (0.5,0.6,0.7)
    expect(mid.y).toBeCloseTo(0.45, 5);
    expect(mid.z).toBeCloseTo(0.6, 5);

    pump(600);
    expect(model.view3dBlend).toBe(1);
    expect(canvas.animPositions).toBeNull();
  });

  it("completes the camera tween when the toggled layout fetch fails", async () => {
    const { model, coord, canvas } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.embeddingPositions = new Map([["a.mp3", { x: 0.2, y: 0.3 }]]);

    const base = globalThis.fetch;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).startsWith("/api/embeddings/projection")) {
          return Promise.resolve(jsonResponse({ error: "boom" }, false));
        }
        return base(input, init);
      }),
    );

    await coord.setView3D(true);
    expect(model.view3d.on).toBe(true);
    // No 3-D layout arrived: positions are untouched and no position animation
    // started, but the camera still fades in (the `finally` path).
    expect(model.embeddingPositions.get("a.mp3")).toEqual({ x: 0.2, y: 0.3 });
    expect(canvas.animPositions).toBeNull();

    pump(275); // half of the 550 ms blend tween: ease = 0.5
    expect(model.view3dBlend).toBeCloseTo(0.5, 5);
    pump(600);
    expect(model.view3dBlend).toBe(1);
  });

  it("toggles back to 2-D with blend and positions moving together", async () => {
    const { model, coord, canvas } = makeHarness(dimsRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    await coord.ensureAndProject(); // 2-D cached
    await coord.setView3D(true); // 3-D fetched, blend + positions tween in
    pump(600);
    expect(model.view3dBlend).toBe(1);
    expect(model.embeddingPositions.get("a.mp3")).toEqual({
      x: 0.5,
      y: 0.6,
      z: 0.7,
    });

    await coord.setView3D(false); // 2-D cache hit: both tweens start together
    pump(275); // half of the toggle: ease = 0.5
    expect(model.view3dBlend).toBeCloseTo(0.5, 5);
    const mid = canvas.animPositions!.get("a.mp3")!;
    expect(mid.x).toBeCloseTo(0.35, 5);
    expect(mid.y).toBeCloseTo(0.45, 5);
    expect(mid.z).toBeCloseTo(0.6, 5); // z lerps back to the missing side (0.5)

    pump(600);
    expect(model.view3dBlend).toBe(0);
    expect(model.embeddingPositions.get("a.mp3")).toEqual({ x: 0.2, y: 0.3 });
    expect(canvas.animPositions).toBeNull();
  });

  it("fetches the weight Jacobian with dims=3 while 3-D is on", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2, z: 0.7 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.01, 0.02, 0.03]] },
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    model.embeddingsReady = true;
    model.embeddingPositions = new Map([["a.mp3", { x: 0.1, y: 0.2 }]]);
    model.view3d.on = true;
    await coord.fetchWeightJacobian();
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).includes("jacobian=1"),
    );
    expect(String(call![0])).toContain("dims=3");
    expect(String(call![0])).toContain("jacobian=1");
  });

  it("applies approximate weight drags in 3-D, writing z", async () => {
    const { model, coord, canvas } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [
              { path: "a.mp3", x: 0.2, y: 0.3, z: 0.5 },
              { path: "b.mp3", x: 0.7, y: 0.7, z: 0.5 },
            ],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: {
              maest: [
                [0.1, 0, 0.2],
                [0, 0.1, -0.2],
              ],
            },
          }
        : { positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    model.embeddingsReady = true;
    model.embeddingPositions = new Map([
      ["a.mp3", { x: 0.2, y: 0.3 }],
      ["b.mp3", { x: 0.7, y: 0.7 }],
    ]);
    model.view3d.on = true;
    await coord.fetchWeightJacobian();
    expect(
      coord.applyApproximateWeights({ ...DEFAULT_CLUSTER_WEIGHTS, maest: 0.6 }),
    ).toBe(true);
    // Each axis maps its stepped span back to the base extent independently, so
    // both points keep their base x/y here; the flat z base still re-bounds z
    // alone to the unit span.
    const a = model.embeddingPositions.get("a.mp3")!;
    const b = model.embeddingPositions.get("b.mp3")!;
    expect(a.x).toBeCloseTo(0.2, 6);
    expect(a.y).toBeCloseTo(0.3, 6);
    expect(b.x).toBeCloseTo(0.7, 6);
    expect(b.y).toBeCloseTo(0.7, 6);
    // The base was flat in z, so the step re-bounds z alone to the unit span.
    expect(a.z).toBeCloseTo(1, 6);
    expect(b.z).toBeCloseTo(0, 6);
    expect(canvas.scheduleDraw).toHaveBeenCalled();
  });

  it("does not apply a 2-D Jacobian once 3-D is on", async () => {
    const { model, coord } = makeHarness((url) =>
      url.includes("jacobian=1")
        ? {
            positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
            weights: { ...DEFAULT_CLUSTER_WEIGHTS },
            jacobian: { maest: [[0.1, 0]] },
          }
        : { positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }], revision: "r" },
    );
    model.viewMode = "embeddings";
    model.embeddingsReady = true;
    model.embeddingPositions = new Map([["a.mp3", { x: 0.1, y: 0.2 }]]);
    await coord.fetchWeightJacobian(); // cached under the 2-D query key
    model.view3d.on = true; // the key now carries dims=3: stale, refuse
    expect(
      coord.applyApproximateWeights({ ...DEFAULT_CLUSTER_WEIGHTS, maest: 0.6 }),
    ).toBe(false);
    expect(model.embeddingPositions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2 });
  });
});

// ─── Generation stream ───────────────────────────────────────

describe("EmbeddingCoordinator generation stream", () => {
  it("queues generation and opens the stream when work is pending", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return withMaestPending(5);
      }
      return {};
    });
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    expect(model.embeddingsGenerating).toBe(true);
    const post = fetchSpy.mock.calls.find(
      ([u, init]) =>
        String(u) === "/api/embeddings/generate" &&
        (init as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeDefined();
    expect(FakeEventSource.instances.at(-1)!.url).toBe("/api/embeddings/stream");
  });

  it("tracks progress and refetches the layout on done", async () => {
    const { model, coord, update, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return withMaestPending(3);
      }
      if (url.startsWith("/api/embeddings/projection")) {
        return { positions: [{ path: "a.mp3", x: 0.1, y: 0.1 }], revision: "rev-2" };
      }
      return {};
    });
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const es = FakeEventSource.instances.at(-1)!;

    es.emit("progress", { path: "a.mp3", ok: true, done: 1, total: 3 });
    expect(update).toHaveBeenCalled();

    const projectionsBefore = fetchCalls(fetchSpy, "/api/embeddings/projection");
    es.emit("done");
    await flush();
    expect(model.embeddingsGenerating).toBe(false);
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(
      projectionsBefore + 1,
    );
  });

  it("closes the stream on stop", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status")
        ? withMaestPending(3)
        : {},
    );
    model.viewMode = "embeddings";
    await coord.ensureAndProject();
    const es = FakeEventSource.instances.at(-1)!;
    coord.stop();
    expect(es.closed).toBe(true);
  });
});

// ─── Lenses ────────────────────────────────────────────────

describe("EmbeddingCoordinator lens switching", () => {
  it("keeps the camera still across a lens switch", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "tags";
    model.embeddingsReady = true;
    model.vp = { ox: 0.1, oy: 0.2, zoom: 2 };

    await coord.switchToLens("combined");
    expect(model.viewMode).toBe("embeddings");
    expect(model.vp).toEqual({ ox: 0.1, oy: 0.2, zoom: 2 });

    await coord.switchToLens("tags");
    expect(model.viewMode).toBe("tags");
    expect(model.vp).toEqual({ ox: 0.1, oy: 0.2, zoom: 2 });
  });

  it("E cycles the available lenses and wraps back to tags", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    // Genre disabled: the cycle is tags → combined (fallback map) → the feature
    // lenses; the single-source MAEST / Folder-head probes are gone.
    const seen = [model.lensId];
    for (let i = 0; i < 6; i++) {
      await coord.cycleLens();
      seen.push(model.lensId);
    }
    expect(seen).toEqual([
      "tags",
      "combined",
      "clap",
      "rhythm",
      "camelot",
      "tempo",
      "tags",
    ]);
  });

  it("cycles tags → genre map → feature lenses when genre is enabled", async () => {
    const { model, coord } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/status")) {
        return { ...STATUS_OK, genre_full_enabled: true };
      }
      return projectionRoute(url);
    });
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.genreFullEnabled = true;
    const seen = [model.lensId];
    for (let i = 0; i < 6; i++) {
      await coord.cycleLens();
      seen.push(model.lensId);
    }
    expect(seen).toEqual(["tags", "genre_full", "clap", "rhythm", "camelot", "tempo", "tags"]);
  });

  it("adds the source param for a per-source lens", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    await coord.switchToLens("clap");
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).startsWith("/api/embeddings/projection"),
    );
    expect(String(call![0])).toContain("sources=clap");
  });

  it("places deterministic lenses with no server fetch", async () => {
    const { model, coord, fetchSpy } = makeHarness();
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.setAllTracks([
      {
        path: "c.mp3",
        filename: "c.mp3",
        folder: "",
        tags: {},
        artist: "",
        title: "",
        audio_features: { tempo: 0.5, key: 0.5, mode: 1, energy: 0.5, danceability: 0.5 },
      },
    ]);
    await coord.switchToLens("camelot");
    expect(model.embeddingPositions.has("c.mp3")).toBe(true);
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(0);
  });

  it("is a no-op when switchToLens targets the current lens", async () => {
    const { model, coord } = makeHarness();
    const swap = vi.spyOn(coord, "switchToLens");
    await coord.switchToMode("tags");
    expect(swap).not.toHaveBeenCalled();
  });
});

// ─── Position animations (shared overlay owner) ───────────────

describe("EmbeddingCoordinator scopes", () => {
  function feats(key: number, mode: number, tempo: number) {
    return { key, mode, tempo, energy: 0.5, danceability: 0.5 };
  }

  function threeTracks(m: Model): void {
    m.setAllTracks([
      { path: "near.mp3", filename: "near.mp3", folder: "", tags: {}, artist: "", title: "", audio_features: feats(0.5, 1, 0.5) },
      { path: "far.mp3", filename: "far.mp3", folder: "", tags: {}, artist: "", title: "", audio_features: feats(0.75, 0, 0.5) },
      { path: "outside.mp3", filename: "outside.mp3", folder: "", tags: {}, artist: "", title: "", audio_features: feats(0.5, 1, 0.9) },
    ]);
    // Outer positions: near at the centre, far nearby, outside at the far corner.
    m.embeddingPositions = new Map([
      ["near.mp3", { x: 0.5, y: 0.5 }],
      ["far.mp3", { x: 0.52, y: 0.5 }],
      ["outside.mp3", { x: 1, y: 1 }],
    ]);
  }

  it("lays out only the in-scope tracks with the scope lens", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    expect(model.scope?.lensId).toBe("camelot");
    expect(model.scopePositions.has("near.mp3")).toBe(true);
    expect(model.scopePositions.has("far.mp3")).toBe(true);
    expect(model.scopePositions.has("outside.mp3")).toBe(false);
    // Mapped onto the circle's disc: reaches the rim, never escapes it.
    let reach = 0;
    for (const p of model.scopePositions.values()) {
      const d = Math.hypot(p.x - 0.5, p.y - 0.5);
      expect(d).toBeLessThanOrEqual(0.1 + 1e-9);
      reach = Math.max(reach, d);
    }
    // The layout fills the region: an extreme of the map lands on the outline.
    expect(reach).toBeCloseTo(0.1, 9);
  });

  it("closes the scope and clears its positions", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    coord.closeScope();
    expect(model.scope).toBeNull();
    expect(model.scopePositions.size).toBe(0);
  });

  it("pushes a scope chip naming the lens on open", async () => {
    const { model, coord, viewMods } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    expect(viewMods.stack).toHaveLength(1);
    expect(viewMods.stack[0].kind).toBe("scope");
    expect(viewMods.stack[0].label).toBe("Camelot wheel");
  });

  it("dismisses the scope chip on close", async () => {
    const { model, coord, viewMods } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    coord.closeScope();
    expect(viewMods.stack).toEqual([]);
    expect(model.scope).toBeNull();
  });

  it("refreshes the scope chip's lens on a lens switch", async () => {
    const { model, coord, viewMods } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    await coord.setScopeLens("tempo");
    expect(viewMods.stack[0].label).toBe("Tempo spiral");
  });

  it("removes the scope chip when the outer lens switches", async () => {
    const { model, coord, viewMods } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    await coord.switchToLens("combined");
    expect(viewMods.stack).toEqual([]);
  });

  it("does not push a scope chip when the circle is empty", async () => {
    const { model, coord, viewMods } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.99, cy: 0.99, r: 0.01, lensId: "" }, "camelot");
    expect(viewMods.stack).toEqual([]);
  });

  it("warns and stays closed when the circle is empty", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.99, cy: 0.99, r: 0.01, lensId: "" }, "camelot");
    expect(model.scope).toBeNull();
  });

  it("fetches a projection lens layout for a projection scope", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.setAllTracks([
      { path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" },
    ]);
    model.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "" }, "clap");
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(1);
    expect(model.scopePositions.has("a.mp3")).toBe(true);
  });

  it("switches the open scope's lens and remembers it", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    const before = new Map(model.scopePositions);
    await coord.setScopeLens("tempo");
    expect(model.scope?.lensId).toBe("tempo");
    expect(model.scopeLensId).toBe("tempo");
    // Same membership (the circle did not move), a different layout.
    expect([...model.scopePositions.keys()].sort()).toEqual(
      [...before.keys()].sort(),
    );
  });

  it("scales the open scope without a tween", async () => {
    const { model, coord, canvas } = makeHarness();
    threeTracks(model);
    await coord.openScope({ shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    pump(600); // let the open tween finish
    const pending = coord.resizeScope(2);
    // The region updates synchronously so a rapid scroll accumulates.
    expect(model.scope?.r).toBeCloseTo(0.2, 6);
    await pending;
    // A scroll resize is immediate, not animated through the overlay.
    expect(canvas.animPositions).toBeNull();
  });

  it("scales both half-extents of a box scope", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope(
      { shape: "box", cx: 0.5, cy: 0.5, r: 0.1, rx: 0.08, ry: 0.04, lensId: "" },
      "camelot",
    );
    await coord.resizeScope(2);
    expect(model.scope?.rx).toBeCloseTo(0.16, 6);
    expect(model.scope?.ry).toBeCloseTo(0.08, 6);
  });

  it("lays out a box scope across its full width and height", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope(
      { shape: "box", cx: 0.5, cy: 0.5, r: 0.1, rx: 0.1, ry: 0.05, lensId: "" },
      "camelot",
    );
    expect(model.scope?.shape).toBe("box");
    // The lens layout fills the box, so dots stay within the region's extent.
    for (const p of model.scopePositions.values()) {
      expect(Math.abs(p.x - 0.5)).toBeLessThanOrEqual(0.1 + 1e-9);
      expect(Math.abs(p.y - 0.5)).toBeLessThanOrEqual(0.05 + 1e-9);
    }
  });

  it("a sphere's membership uses z: equal (x,y) split by depth", async () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([
      { path: "in.mp3", filename: "in.mp3", folder: "", tags: {}, artist: "", title: "", audio_features: feats(0.5, 1, 0.5) },
      { path: "out.mp3", filename: "out.mp3", folder: "", tags: {}, artist: "", title: "", audio_features: feats(0.75, 0, 0.5) },
    ]);
    // Identical (x, y); only z differs. The sphere at the cube centre (r = 0.2)
    // contains the centre plane but not the far wall.
    model.embeddingPositions = new Map([
      ["in.mp3", { x: 0.5, y: 0.5, z: 0.5 }],
      ["out.mp3", { x: 0.5, y: 0.5, z: 1 }],
    ]);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.5, r: 0.2, lensId: "" },
      "camelot",
    );
    expect(model.scopePositions.has("in.mp3")).toBe(true);
    expect(model.scopePositions.has("out.mp3")).toBe(false);
  });

  it("maps a sphere scope's layout like the circle of its radius, with no z", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [
              { path: "a.mp3", x: 0.1, y: 0.2 },
              { path: "b.mp3", x: 0.9, y: 0.8 },
            ],
            revision: "rev-1",
          }
        : {},
    );
    model.setAllTracks([
      { path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" },
      { path: "b.mp3", filename: "b.mp3", folder: "", tags: {}, artist: "", title: "" },
    ]);
    model.embeddingPositions = new Map([
      ["a.mp3", { x: 0.5, y: 0.5, z: 0.5 }],
      ["b.mp3", { x: 0.5, y: 0.5, z: 0.5 }],
    ]);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.5, r: 0.2, lensId: "" },
      "clap",
    );
    const sphere = new Map(model.scopePositions);
    // A sphere's flat layout is tagged with the sphere centre as its billboard
    // anchor (still no z), so the canvas can draw it facing the camera.
    for (const p of sphere.values()) {
      expect(p.z).toBeUndefined();
      expect(p.billboard).toEqual({ cx: 0.5, cy: 0.5, cz: 0.5 });
    }
    // The same raw layout under a circle with the same centre and radius maps
    // identically — the sphere is the 3-D membership volume, but its contents
    // are the flat disc layout, and scopePositions carry no z.
    await coord.openScope(
      { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "" },
      "clap",
    );
    expect([...model.scopePositions.keys()].sort()).toEqual(
      [...sphere.keys()].sort(),
    );
    for (const [path, p] of model.scopePositions) {
      expect(p.z).toBeUndefined();
      expect(p.billboard).toBeUndefined();
      const s = sphere.get(path)!;
      expect(p.x).toBeCloseTo(s.x, 6);
      expect(p.y).toBeCloseTo(s.y, 6);
    }
  });

  it("does not tag circle or box scope positions with a billboard", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope(
      { shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" },
      "camelot",
    );
    for (const p of model.scopePositions.values()) {
      expect(p.billboard).toBeUndefined();
    }
    await coord.openScope(
      { shape: "box", cx: 0.5, cy: 0.5, r: 0.1, rx: 0.08, ry: 0.04, lensId: "" },
      "camelot",
    );
    for (const p of model.scopePositions.values()) {
      expect(p.billboard).toBeUndefined();
    }
  });

  it("animates a scope open through the overlay with from = the outer position", async () => {
    const { model, coord, canvas } = makeHarness();
    threeTracks(model);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.5, r: 0.2, lensId: "" },
      "camelot",
    );
    // Half of the 280 ms open tween: ease = 0.5.
    pump(140);
    const entry = canvas.animPositions!.get("near.mp3")!;
    // The main frame is the billboarded scope position; `from` is the outer
    // cube position it started at, blended with mix = 1 - ease strictly in (0, 1).
    expect(entry.billboard).toEqual({ cx: 0.5, cy: 0.5, cz: 0.5 });
    expect(entry.from).toEqual({ x: 0.5, y: 0.5 });
    expect(entry.mix).toBeGreaterThan(0);
    expect(entry.mix).toBeLessThan(1);
    expect(entry.mix).toBeCloseTo(0.5, 5);
  });

  it("keeps the billboard frame on the close-animation entries after scope is null", async () => {
    const { model, coord, canvas } = makeHarness();
    threeTracks(model);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.5, r: 0.2, lensId: "" },
      "camelot",
    );
    pump(600); // open animation completes
    coord.closeScope();
    expect(model.scope).toBeNull();
    // Mid-close: the entry's main frame is the outer target (no z) and its
    // `from` still carries the billboard anchor, so the canvas can project the
    // start even though the sphere is gone.
    pump(140);
    const entry = canvas.animPositions!.get("near.mp3")!;
    expect(entry.from!.billboard).toEqual({ cx: 0.5, cy: 0.5, cz: 0.5 });
    expect(entry.mix).toBeCloseTo(0.5, 5);
    expect(entry.z).toBeUndefined();
  });

  it("fetches the scope lens in 2-D while the 3-D view is on", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }],
            revision: "rev-1",
          }
        : {},
    );
    model.viewMode = "embeddings"; // active lens: combined (projection) → is3d
    model.view3d.on = true;
    model.setAllTracks([
      { path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" },
    ]);
    model.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5, z: 0.5 }]]);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.5, r: 0.2, lensId: "" },
      "clap",
    );
    const call = fetchSpy.mock.calls.find(([u]) =>
      String(u).includes("sources=clap"),
    );
    expect(String(call![0])).not.toContain("dims=3");
  });

  it("reuses the cached 2-D scope layout when toggling 3-D with a scope open", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [
              { path: "a.mp3", x: 0.2, y: 0.3 },
              { path: "b.mp3", x: 0.8, y: 0.9 },
            ],
            revision: "rev-1",
          }
        : {},
    );
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    model.setAllTracks([
      { path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" },
      { path: "b.mp3", filename: "b.mp3", folder: "", tags: {}, artist: "", title: "" },
    ]);
    model.embeddingPositions = new Map([
      ["a.mp3", { x: 0.5, y: 0.5 }],
      ["b.mp3", { x: 0.52, y: 0.5 }],
    ]);
    // Open a 2-D scope: the scope lens layout is fetched and cached under the
    // 2-D key.
    await coord.openScope(
      { shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" },
      "clap",
    );
    const lensUrl = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .find((u) => u.startsWith("/api/embeddings/projection"))!;
    expect(lensUrl).not.toContain("dims=3");

    // Toggle to 3-D: the main map refetches with dims=3, but the scope re-lays
    // out from the cached 2-D lens layout — no second lens fetch.
    await coord.setView3D(true);
    expect(model.view3d.on).toBe(true);
    const projectionUrls = fetchSpy.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.startsWith("/api/embeddings/projection"));
    expect(projectionUrls).toHaveLength(2);
    expect(projectionUrls[1]).toContain("dims=3");
  });

  it("converts an open circle scope to a sphere on entering 3-D and back on leaving", async () => {
    const { model, coord, viewMods } = makeHarness((url) => {
      if (!url.startsWith("/api/embeddings/projection")) return {};
      const dims = String(url).includes("dims=3");
      const row = (path: string, x: number, y: number) =>
        dims ? { path, x, y, z: 0.5 } : { path, x, y };
      return {
        positions: [
          row("near.mp3", 0.5, 0.5),
          row("far.mp3", 0.52, 0.5),
          row("outside.mp3", 1, 1),
        ],
        revision: "rev-1",
      };
    });
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    threeTracks(model);
    await coord.openScope(
      { shape: "circle", cx: 0.5, cy: 0.5, r: 0.1, lensId: "" },
      "camelot",
    );
    const chipId = viewMods.stack[0].id;
    expect(model.scope?.shape).toBe("circle");

    await coord.setView3D(true);
    expect(model.scope?.shape).toBe("sphere");
    expect(model.scope?.r).toBe(0.1);
    expect(model.scope?.cz).toBe(0.5);
    expect(model.scope?.lensId).toBe("camelot");
    // The chip survives the toggle: same id, still the scope modifier.
    expect(viewMods.stack).toHaveLength(1);
    expect(viewMods.stack[0].id).toBe(chipId);
    expect(viewMods.stack[0].kind).toBe("scope");

    await coord.setView3D(false);
    expect(model.scope?.shape).toBe("circle");
    expect(model.scope?.r).toBe(0.1);
    expect(model.scope?.cz).toBeUndefined();
    expect(model.scope?.lensId).toBe("camelot");
    expect(viewMods.stack).toHaveLength(1);
    expect(viewMods.stack[0].id).toBe(chipId);
  });

  it("converts an open box scope to a sphere spanning its longer half-extent", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/projection")
        ? {
            positions: [
              { path: "near.mp3", x: 0.5, y: 0.5, z: 0.5 },
              { path: "far.mp3", x: 0.52, y: 0.5, z: 0.5 },
              { path: "outside.mp3", x: 1, y: 1, z: 0.5 },
            ],
            revision: "rev-1",
          }
        : {},
    );
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "embeddings";
    threeTracks(model);
    await coord.openScope(
      { shape: "box", cx: 0.5, cy: 0.5, r: 0.1, rx: 0.08, ry: 0.04, lensId: "" },
      "camelot",
    );
    await coord.setView3D(true);
    expect(model.scope?.shape).toBe("sphere");
    expect(model.scope?.r).toBeCloseTo(0.08, 6); // max(rx, ry)
    expect(model.scope?.rx).toBeUndefined();
    expect(model.scope?.ry).toBeUndefined();
  });

  it("scales a sphere's radius with clamping", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    // 3-D outer positions at the sphere's centre depth so membership holds.
    model.embeddingPositions = new Map([
      ["near.mp3", { x: 0.5, y: 0.5, z: 0.3 }],
      ["far.mp3", { x: 0.52, y: 0.5, z: 0.3 }],
      ["outside.mp3", { x: 1, y: 1, z: 0.3 }],
    ]);
    await coord.openScope(
      { shape: "sphere", cx: 0.5, cy: 0.5, cz: 0.3, r: 0.1, lensId: "" },
      "camelot",
    );
    expect(model.scopePositions.size).toBe(2);
    await coord.resizeScope(2);
    expect(model.scope?.r).toBeCloseTo(0.2, 6);
    expect(model.scope?.cz).toBe(0.3); // centre depth kept, not reset
    await coord.resizeScope(0.001);
    expect(model.scope?.r).toBeCloseTo(0.02, 6); // clamped to the minimum
    await coord.resizeScope(40);
    expect(model.scope?.r).toBeCloseTo(0.75, 6); // clamped to the maximum
  });

  it("opens an empty scope when allowEmpty is set", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    // A tiny circle in a corner: no tracks, but X + scroll must still open it so it
    // can grow in.
    await coord.openScope(
      { shape: "circle", cx: 0.99, cy: 0.99, r: 0.005, lensId: "" },
      "camelot",
      true,
    );
    expect(model.scope).not.toBeNull();
    expect(model.scopePositions.size).toBe(0);
  });

  it("cycles backward with cycleLens(-1)", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    await coord.switchToLens("combined");
    await coord.cycleLens(-1);
    expect(model.lensId).toBe("tags");
  });

  it("hides the source-weight editor for a per-source lens", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    const $fold = document.getElementById("cluster-weights-fold")!;
    await coord.switchToLens("clap");
    expect($fold.classList.contains("hidden")).toBe(true);
    await coord.switchToLens("combined");
    expect($fold.classList.contains("hidden")).toBe(false);
  });
});

describe("EmbeddingCoordinator position animations", () => {
  it("animates an axis change and clears the overlay when finished", () => {
    const { model, coord, canvas } = makeHarness();
    coord.animateAxisChange("axisX", "mood");
    expect(model.axisX).toBe("mood");
    expect(canvas.animPositions).not.toBeNull();
    expect(raf.pending.size).toBe(1);
    pump(600);
    expect(canvas.animPositions).toBeNull();
  });

  it("cancels the in-flight frame on cancelAnimation", () => {
    const { coord } = makeHarness();
    coord.animateAxisChange("axisX", "mood");
    const frameId = [...raf.pending.keys()][0];
    coord.cancelAnimation();
    expect(raf.cancelled).toContain(frameId);
    expect(raf.pending.size).toBe(0);
  });
});

// ─── Effective projection settings (View options) ────────────

describe("EmbeddingCoordinator projection settings", () => {
  const settingsCalls = (fetchSpy: ReturnType<typeof vi.fn>) =>
    fetchSpy.mock.calls.filter(([u]) =>
      String(u).startsWith("/api/embeddings/projection-settings"),
    );

  it("fetches the effective settings once per query and refreshes on a budget change", async () => {
    const render = vi.fn();
    const { model, coord, fetchSpy } = makeHarness(
      projectionRoute,
      render,
    );
    vi.spyOn(model, "saveLS").mockImplementation(() => {});

    await coord.switchToLens("combined");
    await flush();

    expect(settingsCalls(fetchSpy)).toHaveLength(1);
    expect(String(settingsCalls(fetchSpy)[0][0])).toContain(
      "/api/embeddings/projection-settings?",
    );
    expect(render).toHaveBeenCalledWith(
      expect.objectContaining({ recipe: "default" }),
    );

    // The same view is cached: reopening the panel is free.
    coord.syncLensChrome();
    await flush();
    expect(settingsCalls(fetchSpy)).toHaveLength(1);

    // A budget override is part of the query, so it refreshes.
    coord.setTsneIteration(100, 100);
    await flush();
    expect(settingsCalls(fetchSpy)).toHaveLength(2);
    expect(String(settingsCalls(fetchSpy)[1][0])).toContain("tsne_n_iter=100");
  });

  it("keeps an unchanged influences render instead of rebuilding its rows and folds", async () => {
    const render = vi.fn();
    const { coord } = makeHarness(projectionRoute, render);
    await coord.switchToLens("combined");
    await flush();
    const renders = render.mock.calls.length;
    expect(renders).toBeGreaterThan(0);

    // Every later resync resolves the same cached payload, so the open
    // "Processing (fixed)" fold and its rows must survive untouched.
    coord.syncLensChrome();
    coord.syncLensChrome();
    await flush();

    expect(render.mock.calls.length).toBe(renders);
  });

  it("keeps a typed iteration draft across an unchanged sync but resyncs on a dimension change", async () => {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<select id="tsne-iteration-preset">
         <option value="default">Default · 50 / 0</option>
         <option value="fast">Fast · 100 / 100</option>
         <option value="custom" disabled>Custom</option>
       </select>
       <input id="tsne-n-iter" type="number" min="50" max="2000" />
       <input id="tsne-early-iter" type="number" min="0" max="1000" />`,
    );
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.lensId = "combined";
    coord.bindTsneIterationControl();

    // A changed stored budget does resync the control...
    coord.setTsneIteration(300, 20);
    coord.syncLensChrome();
    const $nIter = document.getElementById("tsne-n-iter") as HTMLInputElement;
    expect($nIter.value).toBe("300");

    // ...but a sync that changes nothing (a queue tick, a lens reopen) must not
    // rewrite a half-typed budget under the cursor.
    $nIter.value = "3";
    coord.syncLensChrome();
    expect($nIter.value).toBe("3");

    // 3-D keeps its own (empty) budget, so the switch resyncs the control.
    model.view3d.on = true;
    coord.syncLensChrome();
    expect($nIter.value).toBe("");
    model.view3d.on = false;
    coord.syncLensChrome();
    expect($nIter.value).toBe("300");
  });

  it("surfaces a failed settings fetch inline instead of keeping a stale payload", async () => {
    const render = vi.fn();
    const { model, coord } = makeHarness((url) => {
      if (url.startsWith("/api/embeddings/projection-settings"))
        throw new Error("settings down");
      return projectionRoute(url);
    }, render);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});

    await coord.switchToLens("combined");
    await flush();

    expect(render).toHaveBeenCalledWith(
      null,
      expect.stringContaining("unavailable"),
    );
  });

  it("never calls production settings during the read-only eval gallery", async () => {
    const render = vi.fn();
    const { model, coord, fetchSpy } = makeHarness(
      projectionRoute,
      render,
    );
    model.lensId = "combined";
    model.evalMode = true;

    coord.syncLensChrome();
    await flush();

    expect(settingsCalls(fetchSpy)).toHaveLength(0);
    expect(render).toHaveBeenCalledWith(
      null,
      expect.stringContaining("evaluation"),
    );
  });
});

// ─── Read-only eval compute gates ─────────────────────────────

describe("EmbeddingCoordinator is compute-inert during eval", () => {
  it("does not fetch a projection on lens cycle / scope while evalMode is on", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.setAllTracks([{ path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" }]);
    // Eval gallery is active: positions come from precomputed artifacts, never
    // from the projection endpoint.
    model.setEvalMode(true);
    await coord.cycleLens(1);
    await coord.switchToLens("combined");
    await coord.openScope(
      { shape: "circle", cx: 0.5, cy: 0.5, r: 0.2, lensId: "camelot" },
      "camelot",
    );
    await flush();
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(0);
    expect(fetchCalls(fetchSpy, "/api/embeddings/status")).toBe(0);
    // The lens id must not have been changed by the inert cycleLens.
    expect(model.lensId).toBe("tags");
  });

  it("still applies precomputed positions on the shared frame in eval mode", async () => {
    const { model, coord, canvas, fetchSpy } = makeHarness(projectionRoute);
    model.setEvalMode(true);
    const positions = new Map([
      ["a.mp3", { x: 0.1, y: 0.2 }],
      ["b.mp3", { x: 0.8, y: 0.9 }],
    ]);
    coord.applyPrecomputedPositions(positions);
    // No projection fetch is issued by the eval apply path.
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(0);
    expect(model.embeddingPositions.size).toBe(2);
  });

  it("applies a 3-D artifact through the shared blend without touching production view3d", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.setEvalMode(true);
    expect(model.view3d.on).toBe(false);
    const positions = new Map([
      ["a.mp3", { x: 0.1, y: 0.2, z: 0.3 }],
      ["b.mp3", { x: 0.8, y: 0.9, z: 0.7 }],
    ]);

    coord.applyPrecomputedPositions(positions, 3);
    expect(model.evalDims).toBe(3);
    expect(model.is3d).toBe(true);
    expect(model.shows3d).toBe(true);
    expect(model.view3d.on).toBe(false);
    // The camera blend runs on the existing shared-frame tween, not a new loop.
    expect(raf.pending.size).toBe(1);
    pump(600);
    expect(model.view3dBlend).toBe(1);
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(0);
  });
});
