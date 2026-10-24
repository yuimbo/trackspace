import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Model,
  DEFAULT_CLUSTER_WEIGHTS,
  type Track,
  type ViewModifier,
} from "../model";
import { EmbeddingCoordinator } from "./embedding-coordinator";
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
  <details id="cluster-weights-details"></details>
`;

interface Harness {
  model: Model;
  update: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  canvas: {
    animPositions: Map<string, { x: number; y: number }> | null;
    scheduleDraw: ReturnType<typeof vi.fn>;
    worldPosForTrack(t: Track): { wx: number; wy: number } | null;
    outerPosForTrack(t: Track): { wx: number; wy: number } | null;
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
function makeHarness(routes: (url: string) => unknown = () => ({})): Harness {
  const model = new Model();
  const update = vi.fn();
  const scheduleDraw = vi.fn();
  const canvas = {
    animPositions: null as Map<string, { x: number; y: number }> | null,
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
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return { model, update, scheduleDraw, canvas, coord, fetchSpy, viewMods };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Outer position for scope membership/anchors: the embedding map, else 0.5. */
function outerPos(m: Model, t: Track): { wx: number; wy: number } | null {
  const ep = m.embeddingPositions.get(t.path);
  return ep ? { wx: ep.x, wy: ep.y } : { wx: 0.5, wy: 0.5 };
}

function fetchCalls(fetchSpy: ReturnType<typeof vi.fn>, prefix: string): number {
  return fetchSpy.mock.calls.filter(([u]) => String(u).startsWith(prefix)).length;
}

function projectionRoute(url: string): unknown {
  if (url.startsWith("/api/embeddings/status")) return STATUS_OK;
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

  it("E cycles every built-in lens and wraps back to tags", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    const seen = [model.lensId];
    for (let i = 0; i < 8; i++) {
      await coord.cycleLens();
      seen.push(model.lensId);
    }
    expect(seen).toEqual([
      "tags",
      "combined",
      "maest",
      "supervised",
      "clap",
      "rhythm",
      "camelot",
      "tempo",
      "tags",
    ]);
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
    // Mapped into the circle's inscribed square, so inside the radius.
    for (const p of model.scopePositions.values()) {
      expect(Math.hypot(p.x - 0.5, p.y - 0.5)).toBeLessThanOrEqual(0.1);
    }
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

  it("hides the source-weight sliders for a per-source lens", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    await coord.switchToLens("clap");
    expect(
      document.getElementById("cluster-weights-details")!.hidden,
    ).toBe(true);
    await coord.switchToLens("combined");
    expect(
      document.getElementById("cluster-weights-details")!.hidden,
    ).toBe(false);
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
});
