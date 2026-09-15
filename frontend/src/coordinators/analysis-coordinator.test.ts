import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type QueueKindStatus, type QueueStatus } from "../model";
import { AnalysisCoordinator } from "./analysis-coordinator";

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

function makeQueueStatus(overrides: Partial<QueueStatus> = {}): QueueStatus {
  return {
    kinds: {},
    order: [],
    outstanding: 0,
    total: 0,
    finished: 0,
    active: false,
    epoch: 0,
    ...overrides,
  };
}

function makeKind(overrides: Partial<QueueKindStatus> = {}): QueueKindStatus {
  return {
    label: "Scan",
    pending: 0,
    running: 0,
    done: 0,
    failed: 0,
    total: 0,
    finished: 0,
    outstanding: 0,
    ...overrides,
  };
}

const PANEL_HTML = `
  <div id="toast-container"></div>
  <div id="cluster-meta"></div>
  <ul id="cluster-list"></ul>
  <input id="cluster-color-toggle" type="checkbox" />
  <input id="cluster-resolution-range" type="range" />
  <button id="btn-rebuild-clusters"></button>
`;

interface Harness {
  model: Model;
  update: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  coord: AnalysisCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
}

/** Route a URL to its response body (default: `{}`). */
function makeHarness(routes: (url: string) => unknown = () => ({})): Harness {
  const model = new Model();
  const update = vi.fn();
  const scheduleDraw = vi.fn();
  const coord = new AnalysisCoordinator({
    model,
    status: { update },
    scheduleDraw,
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return { model, update, scheduleDraw, coord, fetchSpy };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Start the coordinator and let its initial fetches settle. */
async function startSettled(coord: AnalysisCoordinator): Promise<FakeEventSource> {
  coord.start();
  await flush();
  return FakeEventSource.instances.at(-1)!;
}

beforeEach(() => {
  document.body.innerHTML = PANEL_HTML;
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ─── Queue stream ────────────────────────────────────────────

describe("AnalysisCoordinator queue stream", () => {
  it("opens the jobs stream", async () => {
    const { coord } = makeHarness();
    const es = await startSettled(coord);
    expect(es.url).toBe("/api/jobs/stream");
  });

  it("applies a full snapshot and refreshes the status bar", async () => {
    const { model, update, coord } = makeHarness();
    const es = await startSettled(coord);
    const snap = makeQueueStatus({ outstanding: 2, active: true });
    es.emit("snapshot", snap);
    // The event payload is JSON-round-tripped, so compare structurally.
    expect(model.queueStatus).toStrictEqual(snap);
    expect(update).toHaveBeenCalledWith(model);
  });

  it("increments counters optimistically on progress", async () => {
    const { model, coord } = makeHarness();
    const es = await startSettled(coord);
    model.queueStatus = makeQueueStatus({
      kinds: { scan: makeKind({ pending: 3, running: 1, done: 5, total: 9 }) },
    });
    es.emit("progress", { kind: "scan", path: "a.mp3", ok: true });
    const k = model.queueStatus.kinds.scan;
    expect(k.done).toBe(6);
    expect(k.pending).toBe(2);
    expect(k.finished).toBe(6);
    expect(k.outstanding).toBe(3);
    expect(k.last?.path).toBe("a.mp3");
    expect(k.last?.ok).toBe(true);
  });

  it("counts failures on progress", async () => {
    const { model, coord } = makeHarness();
    const es = await startSettled(coord);
    model.queueStatus = makeQueueStatus({
      kinds: { scan: makeKind({ pending: 3, running: 1, done: 5 }) },
    });
    es.emit("progress", { kind: "scan", path: "b.mp3", ok: false });
    const k = model.queueStatus.kinds.scan;
    expect(k.failed).toBe(1);
    expect(k.pending).toBe(2);
    expect(k.last?.ok).toBe(false);
  });

  it("refreshes authoritative status on stats and drained", async () => {
    const { coord } = makeHarness();
    const es = await startSettled(coord);
    const spy = vi.spyOn(coord, "refreshQueueStatus").mockResolvedValue();
    es.emit("stats");
    expect(spy).toHaveBeenCalledTimes(1);
    es.emit("drained");
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("closes the stream on stop", async () => {
    const { coord } = makeHarness();
    const es = await startSettled(coord);
    coord.stop();
    expect(es.closed).toBe(true);
  });
});

// ─── Status polling ─────────────────────────────────────────

describe("AnalysisCoordinator status polling", () => {
  it("refetches clusters when the reported revision changes", async () => {
    const { model, coord } = makeHarness((url) => {
      if (url.startsWith("/api/jobs/status")) {
        return makeQueueStatus({ clusters: { available: true, revision: "rev-2" } });
      }
      if (url.startsWith("/api/clusters")) {
        return {
          available: true,
          revision: "rev-2",
          resolution: 1.5,
          resolutions: [1.5],
          assignments: { "a.mp3": 3 },
          names: { "3": "Techno" },
          sizes: { "3": 1 },
        };
      }
      return {};
    });
    model.clusterRevision = "rev-1";
    await coord.refreshQueueStatus();
    expect(model.clusterRevision).toBe("rev-2");
    expect(model.clusterAssignments.get("a.mp3")).toBe(3);
  });

  it("does not refetch clusters when the revision is unchanged", async () => {
    const { model, coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/jobs/status")) {
        return makeQueueStatus({ clusters: { available: true, revision: "rev-1" } });
      }
      return {};
    });
    model.clusterRevision = "rev-1";
    await coord.refreshQueueStatus();
    const clusterCalls = fetchSpy.mock.calls.filter(([u]) =>
      String(u).startsWith("/api/clusters"),
    );
    expect(clusterCalls).toHaveLength(0);
  });
});

// ─── Cluster fetch + panel ──────────────────────────────────

describe("AnalysisCoordinator cluster fetch", () => {
  it("populates cluster state, renders the panel, and repaints", async () => {
    const { model, coord, scheduleDraw } = makeHarness((url) =>
      url.startsWith("/api/clusters")
        ? {
            available: true,
            revision: "r",
            resolution: 0.4,
            resolutions: [0.4, 1.5],
            assignments: { "a.mp3": 1, "b.mp3": 2 },
            names: { "1": "House", "2": "Techno" },
            sizes: { "1": 1, "2": 1 },
          }
        : {},
    );
    await coord.fetchClusters();
    expect(model.clustersAvailable).toBe(true);
    expect(model.clusterRevision).toBe("r");
    expect(model.clusterResolutions).toEqual([0.4, 1.5]);
    expect(model.clusterResolution).toBe(0.4);
    expect(model.clusterNames.get(1)).toBe("House");
    expect(model.clusterAssignments.get("b.mp3")).toBe(2);
    expect(scheduleDraw).toHaveBeenCalled();
    expect(document.querySelectorAll("#cluster-list .cluster-item")).toHaveLength(2);
  });

  it("clears cluster state when unavailable", async () => {
    const { model, coord } = makeHarness(() => ({ available: false }));
    model.clustersAvailable = true;
    model.clusterAssignments = new Map([["a.mp3", 1]]);
    model.clusterNames = new Map([[1, "House"]]);
    model.clusterSizes = new Map([[1, 1]]);
    await coord.fetchClusters();
    expect(model.clustersAvailable).toBe(false);
    expect(model.clusterAssignments.size).toBe(0);
    expect(model.clusterNames.size).toBe(0);
    expect(document.getElementById("cluster-meta")!.textContent).toContain(
      "Waiting for MAEST",
    );
  });

  it("ignores concurrent fetches while one is in flight", async () => {
    let resolveFetch!: (r: Response) => void;
    const fetchSpy = vi.fn(
      () => new Promise<Response>((r) => (resolveFetch = r)),
    );
    vi.stubGlobal("fetch", fetchSpy);
    const model = new Model();
    const coord = new AnalysisCoordinator({
      model,
      status: { update: vi.fn() },
      scheduleDraw: vi.fn(),
    });
    const p1 = coord.fetchClusters();
    const p2 = coord.fetchClusters();
    resolveFetch(jsonResponse({ available: false }));
    await Promise.all([p1, p2]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("sorts the panel by size and selects a cluster's tracks on click", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/clusters")
        ? {
            available: true,
            revision: "r",
            resolution: 1.5,
            resolutions: [1.5],
            assignments: { "a.mp3": 7, "b.mp3": 7, "c.mp3": 9 },
            names: { "7": "Techno" },
            sizes: { "7": 2, "9": 1 },
          }
        : {},
    );
    const change = vi.fn();
    model.on("change", change);
    await coord.fetchClusters();
    const items = document.querySelectorAll<HTMLElement>(
      "#cluster-list .cluster-item",
    );
    expect(items[0].dataset.cluster).toBe("7");
    items[0].click();
    expect([...model.selected].sort()).toEqual(["a.mp3", "b.mp3"]);
    expect(change).toHaveBeenCalled();
  });
});

// ─── Panel bindings ─────────────────────────────────────────

describe("AnalysisCoordinator panel bindings", () => {
  it("toggles cluster colouring, syncs the checkbox, and fetches when needed", async () => {
    const { model, coord, scheduleDraw } = makeHarness(() => ({ available: false }));
    model.clustersAvailable = false;
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();

    coord.toggleColorByCluster();

    expect(model.colorByCluster).toBe(true);
    expect(
      (document.getElementById("cluster-color-toggle") as HTMLInputElement).checked,
    ).toBe(true);
    expect(fetchSpy).toHaveBeenCalled();
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("does not refetch when toggling colouring off", () => {
    const { model, coord } = makeHarness();
    model.clustersAvailable = true;
    model.colorByCluster = true;
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();
    coord.toggleColorByCluster();
    expect(model.colorByCluster).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("maps the resolution slider index to a resolution value", () => {
    const { model, coord } = makeHarness();
    model.clusterResolutions = [0.4, 1.5, 3.0];
    model.clusterResolution = 1.5;
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();

    const $cr = document.getElementById(
      "cluster-resolution-range",
    ) as HTMLInputElement;
    expect($cr.value).toBe("1");
    $cr.value = "2";
    $cr.dispatchEvent(new Event("change"));

    expect(model.clusterResolution).toBe(3.0);
    expect(fetchSpy).toHaveBeenCalled();
  });

  it("queues a cluster rebuild from the panel button", async () => {
    const { coord, fetchSpy } = makeHarness();
    coord.bindPanel();
    document.getElementById("btn-rebuild-clusters")!.click();
    await flush();
    const post = fetchSpy.mock.calls.find(
      ([u, init]) =>
        String(u) === "/api/clusters/rebuild" &&
        (init as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeDefined();
  });
});
