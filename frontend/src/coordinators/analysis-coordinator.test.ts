import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  Model,
  DEFAULT_CLUSTER_WEIGHTS,
  type HoverTarget,
  type QueueKindStatus,
  type QueueStatus,
} from "../model";
import { AnalysisCoordinator } from "./analysis-coordinator";
import { toast } from "../lib/toast";

vi.mock("../lib/toast", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/toast")>();
  return { ...actual, toast: vi.fn() };
});

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
  <div id="cluster-weights"></div>
  <p id="cluster-weights-note"></p>
  <div id="analysis-coverage"></div>
  <div id="analysis-retrigger"></div>
  <div id="analysis-problems"></div>
  <button id="btn-apply-weights"></button>
  <button id="btn-reset-weights"></button>
  <button id="btn-default-weights"></button>
  <button id="btn-rebuild-clusters"></button>
  <button id="btn-train-supervised"></button>
  <div id="head-train-folders"></div>
  <details id="head-train-folders-details"></details>
  <button id="btn-head-folders-all"></button>
  <input id="head-imprint-path" />
  <button id="btn-imprint-analyze"></button>
  <button id="btn-imprint-plan"></button>
  <button id="btn-imprint-apply"></button>
  <p id="imprint-note"></p>
`;

interface Harness {
  model: Model;
  update: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  refreshProjection: ReturnType<typeof vi.fn>;
  applyApproximateWeights: ReturnType<typeof vi.fn>;
  canvas: { hoverTarget: HoverTarget | null };
  coord: AnalysisCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
}

/** Route a URL to its response body (default: `{}`). */
function makeHarness(routes: (url: string) => unknown = () => ({})): Harness {
  const model = new Model();
  const update = vi.fn();
  const scheduleDraw = vi.fn();
  const refreshProjection = vi.fn();
  const applyApproximateWeights = vi.fn(() => true);
  const canvas = { hoverTarget: null as HoverTarget | null };
  const coord = new AnalysisCoordinator({
    model,
    status: { update },
    scheduleDraw,
    canvas,
    refreshProjection,
    applyApproximateWeights,
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return {
    model,
    update,
    scheduleDraw,
    refreshProjection,
    applyApproximateWeights,
    canvas,
    coord,
    fetchSpy,
  };
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

  it("records the running singleton stage phase", async () => {
    const { model, update, scheduleDraw, coord } = makeHarness();
    const es = await startSettled(coord);
    model.queueStatus = makeQueueStatus({
      active: true,
      outstanding: 1,
      total: 1,
      order: ["cluster"],
      kinds: { cluster: makeKind({ label: "Clustering", running: 1, outstanding: 1, total: 1 }) },
    });
    es.emit("stage_progress", { kind: "cluster", phase: "leiden", done: 2, total: 6 });
    expect(model.queueStatus.kinds.cluster.stage).toMatchObject({
      phase: "leiden",
      done: 2,
      total: 6,
    });
    expect(update).toHaveBeenCalledWith(model);
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("ignores a stage phase for an unknown kind", async () => {
    const { model, coord } = makeHarness();
    const es = await startSettled(coord);
    model.queueStatus = makeQueueStatus({ active: true, outstanding: 1 });
    es.emit("stage_progress", { kind: "bogus", phase: "pca" });
    expect(model.queueStatus.kinds.bogus).toBeUndefined();
  });

  it("records a layout phase that is not a queued job", async () => {
    const { model, update, scheduleDraw, coord } = makeHarness();
    const es = await startSettled(coord);
    es.emit("layout_progress", { phase: "jacobian", done: 2, total: 5 });
    expect(model.layoutProgress).toMatchObject({
      phase: "jacobian",
      done: 2,
      total: 5,
    });
    expect(update).toHaveBeenCalledWith(model);
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("toasts analysis complete only once the whole queue drains", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/jobs/status")
        ? makeQueueStatus({ outstanding: 1, active: true })
        : {},
    );
    const es = await startSettled(coord);
    vi.mocked(toast).mockClear();

    // Another library-wide stage is still queued: not the end of the run.
    es.emit("drained");
    await flush();
    expect(toast).not.toHaveBeenCalled();

    // Now the authoritative status reports a fully idle queue.
    vi.spyOn(coord, "refreshQueueStatus").mockImplementation(async () => {
      model.queueStatus = makeQueueStatus({ outstanding: 0 });
    });
    es.emit("drained");
    await flush();
    expect(toast).toHaveBeenCalledWith("Analysis complete", "ok");
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
      "Rebuild clusters",
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
      canvas: { hoverTarget: null },
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

  it("glows a cluster's points on row hover and clears on leave", async () => {
    const { coord, canvas, scheduleDraw } = makeHarness((url) =>
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
    await coord.fetchClusters();
    const items = document.querySelectorAll<HTMLElement>(
      "#cluster-list .cluster-item",
    );
    items[0].dispatchEvent(new Event("mouseenter"));
    expect(canvas.hoverTarget).toEqual({ kind: "cluster", id: 7 });
    expect(scheduleDraw).toHaveBeenCalled();

    scheduleDraw.mockClear();
    items[0].dispatchEvent(new Event("mouseleave"));
    expect(canvas.hoverTarget).toBeNull();
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("drops a stale cluster hover when the panel re-renders", async () => {
    const { coord, canvas, scheduleDraw } = makeHarness((url) =>
      url.startsWith("/api/clusters")
        ? {
            available: true,
            revision: "r",
            resolution: 1.5,
            resolutions: [1.5],
            assignments: { "a.mp3": 7 },
            names: { "7": "Techno" },
            sizes: { "7": 1 },
          }
        : {},
    );
    await coord.fetchClusters();
    canvas.hoverTarget = { kind: "cluster", id: 7 };
    await coord.fetchClusters();
    expect(canvas.hoverTarget).toBeNull();
    expect(scheduleDraw).toHaveBeenCalled();
  });
});

// ─── Panel bindings ─────────────────────────────────────────

describe("AnalysisCoordinator panel bindings", () => {
  it("enables cluster colouring when clusters are available", () => {
    const { model, coord, scheduleDraw } = makeHarness();
    model.clustersAvailable = true;
    coord.bindPanel();

    coord.toggleColorByCluster();

    expect(model.colorByCluster).toBe(true);
    expect(
      (document.getElementById("cluster-color-toggle") as HTMLInputElement).checked,
    ).toBe(true);
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("refuses to colour by cluster when none are available, and says why", () => {
    const { model, coord, scheduleDraw } = makeHarness();
    model.clustersAvailable = false;
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();
    vi.mocked(toast).mockClear();

    coord.toggleColorByCluster();

    expect(model.colorByCluster).toBe(false);
    const $cc = document.getElementById(
      "cluster-color-toggle",
    ) as HTMLInputElement;
    expect($cc.checked).toBe(false);
    expect($cc.disabled).toBe(true);
    expect(toast).toHaveBeenCalledWith(
      expect.stringContaining("Rebuild clusters"),
      "warn",
    );
    // No point fetching: we already know it is unavailable.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(scheduleDraw).not.toHaveBeenCalled();
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

// ─── Source weights ────────────────────────────────────────

/** The POST body of the last `/api/clusters/rebuild` call. */
function rebuildBody(fetchSpy: ReturnType<typeof vi.fn>): unknown {
  const post = fetchSpy.mock.calls.find(
    ([u, init]) =>
      String(u) === "/api/clusters/rebuild" &&
      (init as RequestInit | undefined)?.method === "POST",
  );
  expect(post).toBeDefined();
  return JSON.parse(String((post![1] as RequestInit).body));
}

/** Whether a `/api/clusters/rebuild` POST was made at all. */
function didRebuild(fetchSpy: ReturnType<typeof vi.fn>): boolean {
  return fetchSpy.mock.calls.some(
    ([u, init]) =>
      String(u) === "/api/clusters/rebuild" &&
      (init as RequestInit | undefined)?.method === "POST",
  );
}

describe("AnalysisCoordinator source weights", () => {
  it("renders one slider per source, initialised to the defaults", () => {
    const { model, coord } = makeHarness();
    coord.bindPanel();
    const inputs = document.querySelectorAll<HTMLInputElement>(
      "#cluster-weights input[type='range']",
    );
    expect(inputs).toHaveLength(Object.keys(DEFAULT_CLUSTER_WEIGHTS).length);
    const bySource = Object.fromEntries(
      [...inputs].map((i) => [i.dataset.source, i.valueAsNumber]),
    );
    expect(bySource).toEqual(DEFAULT_CLUSTER_WEIGHTS);
    expect(model.clusterWeights).toEqual(DEFAULT_CLUSTER_WEIGHTS);
  });

  it("posts the live slider values and refreshes the layout on Apply", async () => {
    const { model, coord, fetchSpy, refreshProjection } = makeHarness();
    coord.bindPanel();
    const $rhythm = document.querySelector<HTMLInputElement>(
      "#cluster-weights input[data-source='rhythm']",
    )!;
    $rhythm.value = "0.5";
    $rhythm.dispatchEvent(new Event("input"));
    document.getElementById("btn-apply-weights")!.click();
    await flush();

    expect(model.clusterWeights.rhythm).toBe(0.5);
    expect(rebuildBody(fetchSpy)).toEqual({ weights: model.clusterWeights });
    expect(refreshProjection).toHaveBeenCalled();
  });

  it("undoes slider drags on Reset without a server round-trip", async () => {
    const { model, coord, fetchSpy, applyApproximateWeights } = makeHarness();
    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.9 };
    coord.bindPanel();
    const $rhythm = document.querySelector<HTMLInputElement>(
      "#cluster-weights input[data-source='rhythm']",
    )!;
    $rhythm.value = "0.5";
    $rhythm.dispatchEvent(new Event("input"));
    document.getElementById("btn-reset-weights")!.click();
    await flush();

    // Sliders snap back to the applied vector and the map is restored locally.
    expect($rhythm.valueAsNumber).toBeCloseTo(0.9);
    expect(model.clusterWeights.rhythm).toBe(0.9);
    expect(applyApproximateWeights).toHaveBeenCalledWith(model.clusterWeights);
    expect(didRebuild(fetchSpy)).toBe(false);
  });

  it("loads and applies the defaults on Defaults", async () => {
    const { model, coord, fetchSpy } = makeHarness();
    model.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, rhythm: 0.9 };
    coord.bindPanel();
    document.getElementById("btn-default-weights")!.click();
    await flush();

    expect(model.clusterWeights).toEqual(DEFAULT_CLUSTER_WEIGHTS);
    expect(rebuildBody(fetchSpy)).toEqual({ weights: DEFAULT_CLUSTER_WEIGHTS });
  });

  it("disables Apply while a cluster job is outstanding", async () => {
    const { coord } = makeHarness();
    coord.bindPanel();
    const $apply = document.getElementById(
      "btn-apply-weights",
    ) as HTMLButtonElement;
    expect($apply.disabled).toBe(false);

    const es = await startSettled(coord);
    es.emit(
      "snapshot",
      makeQueueStatus({
        kinds: { cluster: makeKind({ pending: 1, outstanding: 1 }) },
      }),
    );
    expect($apply.disabled).toBe(true);
  });

  it("flags a requested source that has no cached vectors", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/clusters")
        ? {
            available: true,
            revision: "r",
            resolution: 1.5,
            resolutions: [1.5],
            assignments: {},
            names: {},
            sizes: {},
            sources_used: ["maest_logits", "maest"],
          }
        : {},
    );
    await coord.fetchClusters();
    const note = document.getElementById("cluster-weights-note")!.textContent!;
    expect(note).toContain("Rhythm");
    expect(note).not.toContain("Genre");
  });
});

// ─── Coverage panel ─────────────────────────────────────────

describe("AnalysisCoordinator coverage panel", () => {
  const COVERAGE = {
    tracks_total: 10,
    tracks_with_fingerprint: 10,
    tracks_with_audio_features: 8,
    tracks_pending_audio_features: 2,
    tracks_with_rhythm: 7,
    tracks_pending_rhythm: 3,
    tracks_with_structure: 6,
    tracks_pending_structure: 4,
    tracks_with_maest: 9,
    tracks_pending_maest: 1,
    tracks_with_clap: 5,
    tracks_pending_clap: 5,
    tracks_with_supervised: 0,
    tracks_pending_supervised: 10,
  };

  it("renders one row per source with done / total", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status") ? COVERAGE : {},
    );
    await coord.fetchCoverage();

    const rows = [...document.querySelectorAll(".analysis-coverage-row")];
    expect(rows).toHaveLength(6);
    const names = rows.map(
      (r) => r.querySelector(".analysis-coverage-name")?.textContent,
    );
    expect(names).toEqual([
      "Audio features",
      "Rhythm",
      "Structure",
      "MAEST",
      "CLAP",
      "Supervised",
    ]);
    const structure = rows[2];
    expect(structure.querySelector(".analysis-coverage-count")?.textContent).toBe(
      "6 / 10",
    );
    expect(
      structure.querySelector<HTMLElement>(".analysis-coverage-bar i")?.style.width,
    ).toBe("60%");
  });

  it("flags terminal failures so a short bar is not mistaken for pending", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status")
        ? {
            ...COVERAGE,
            tracks_with_structure: 6,
            tracks_pending_structure: 0,
            tracks_failed_structure: 4,
          }
        : {},
    );

    await coord.fetchCoverage();

    const rows = [...document.querySelectorAll(".analysis-coverage-row")];
    const structure = rows[2];
    expect(
      structure
        .querySelector(".analysis-coverage-count")
        ?.classList.contains("has-failed"),
    ).toBe(true);
    expect(structure.getAttribute("title")).toContain("4 failed");
    expect(
      document.querySelector(".analysis-coverage-summary")?.textContent,
    ).toContain("4 failed");
  });

  it("wires one retrigger button per stage plus a forced re-run", async () => {
    const { coord, fetchSpy } = makeHarness();
    coord.bindPanel();

    const buttons = [
      ...document.querySelectorAll<HTMLButtonElement>("#analysis-retrigger button"),
    ];
    expect(buttons.map((b) => b.textContent)).toEqual([
      "Audio features",
      "Structure",
      "MAEST",
      "CLAP",
      "Re-run all",
    ]);

    buttons[1].click();
    await flush();
    const structurePost = fetchSpy.mock.calls.find(
      ([url, init]) =>
        String(url) === "/api/embeddings/generate" &&
        (init as RequestInit)?.method === "POST",
    );
    expect(JSON.parse((structurePost![1] as RequestInit).body as string)).toEqual({
      folder: "",
      recursive: true,
      stages: ["structure"],
      force: false,
    });
  });
});

// ─── Broken files ───────────────────────────────────────────

describe("AnalysisCoordinator broken files", () => {
  it("lists unfingerprinted and failed tracks with reasons", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/analysis/problems")
        ? {
            ok: true,
            unfingerprinted: ["b.mp3"],
            failed: [
              { path: "a.mp3", kinds: ["structure"], errors: ["decode failed"] },
            ],
          }
        : {},
    );
    await coord.fetchProblems();

    const details = document.querySelector("#analysis-problems details");
    expect(details?.querySelector("summary")?.textContent).toContain(
      "Broken files (2)",
    );
    const rows = [...document.querySelectorAll(".analysis-problem")];
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("no fingerprint");
    expect(rows[1].textContent).toContain("structure failed");
  });

  it("hides the section when nothing is broken", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/analysis/problems")
        ? { ok: true, unfingerprinted: [], failed: [] }
        : {},
    );
    await coord.fetchProblems();
    expect(document.getElementById("analysis-problems")?.children).toHaveLength(0);
  });

  it("retries only the failed track's stages", async () => {
    const { coord, fetchSpy } = makeHarness((url) =>
      url.startsWith("/api/analysis/problems")
        ? {
            ok: true,
            unfingerprinted: [],
            failed: [
              { path: "a.mp3", kinds: ["structure", "maest"], errors: ["x"] },
            ],
          }
        : {},
    );
    await coord.fetchProblems();
    const retry = document.querySelector<HTMLButtonElement>(
      ".analysis-problem .analysis-problem-btn:last-child",
    );
    retry?.click();
    await flush();
    const post = fetchSpy.mock.calls.find(
      ([url, init]) =>
        String(url) === "/api/embeddings/generate" &&
        (init as RequestInit)?.method === "POST",
    );
    expect(JSON.parse((post![1] as RequestInit).body as string)).toMatchObject({
      paths: ["a.mp3"],
      stages: ["structure", "maest"],
    });
  });
});
