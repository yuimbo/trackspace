import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, DEFAULT_CLUSTER_WEIGHTS, type Track } from "../model";
import { EmbeddingCoordinator } from "./embedding-coordinator";

// ─── Fakes ───────────────────────────────────────────────────

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
  <select id="lens-select"></select>
`;

interface Harness {
  model: Model;
  update: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  canvas: {
    animPositions: Map<string, { x: number; y: number }> | null;
    scheduleDraw: ReturnType<typeof vi.fn>;
  };
  coord: EmbeddingCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
}

const STATUS_OK = {
  total: 1,
  fingerprinted: 1,
  embedded: 1,
  pending: 0,
  model_ready: true,
  maest_model_ready: true,
  tracks_pending_maest: 0,
  tracks_pending_rhythm: 0,
  features_extracted: 1,
  features_pending: 0,
  generating: false,
  layout_revision: "rev-1",
};

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
  const coord = new EmbeddingCoordinator({
    model,
    status: { update },
    canvas,
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return { model, update, scheduleDraw, canvas, coord, fetchSpy };
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
        return { ...STATUS_OK, tracks_pending_maest: 5 };
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
        return { ...STATUS_OK, tracks_pending_maest: 3 };
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
    expect(model.embeddingProgress?.done).toBe(1);
    expect(update).toHaveBeenCalled();

    const projectionsBefore = fetchCalls(fetchSpy, "/api/embeddings/projection");
    es.emit("done");
    await flush();
    expect(model.embeddingsGenerating).toBe(false);
    expect(model.embeddingProgress).toBeNull();
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(
      projectionsBefore + 1,
    );
  });

  it("closes the stream on stop", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status")
        ? { ...STATUS_OK, tracks_pending_maest: 3 }
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
  it("saves and restores the viewport across a lens switch", async () => {
    const { model, coord } = makeHarness(projectionRoute);
    vi.spyOn(model, "saveLS").mockImplementation(() => {});
    model.viewMode = "tags";
    model.embeddingsReady = true;
    model.vp = { ox: 0.1, oy: 0.2, zoom: 2 };

    await coord.switchToLens("combined");
    expect(model.viewMode).toBe("embeddings");
    expect(model.vp).toEqual({ ox: 0, oy: 0, zoom: 1 });

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
      "clap",
      "rhythm",
      "supervised",
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
    await coord.openScope({ cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
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
    await coord.openScope({ cx: 0.5, cy: 0.5, r: 0.1, lensId: "" }, "camelot");
    coord.closeScope();
    expect(model.scope).toBeNull();
    expect(model.scopePositions.size).toBe(0);
  });

  it("warns and stays closed when the circle is empty", async () => {
    const { model, coord } = makeHarness();
    threeTracks(model);
    await coord.openScope({ cx: 0.99, cy: 0.99, r: 0.01, lensId: "" }, "camelot");
    expect(model.scope).toBeNull();
  });

  it("fetches a projection lens layout for a projection scope", async () => {
    const { model, coord, fetchSpy } = makeHarness(projectionRoute);
    model.setAllTracks([
      { path: "a.mp3", filename: "a.mp3", folder: "", tags: {}, artist: "", title: "" },
    ]);
    model.embeddingPositions = new Map([["a.mp3", { x: 0.5, y: 0.5 }]]);
    await coord.openScope({ cx: 0.5, cy: 0.5, r: 0.2, lensId: "" }, "clap");
    expect(fetchCalls(fetchSpy, "/api/embeddings/projection")).toBe(1);
    expect(model.scopePositions.has("a.mp3")).toBe(true);
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
