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
  <div id="view-options" class="view-options hidden">
    <div id="head-confident-fold" class="view-option hidden">
      <input type="checkbox" id="head-color-confident" />
    </div>
    <div id="genre-weights-fold" class="view-option hidden">
      <div id="genre-weights"></div>
      <p id="genre-weights-note"></p>
      <button id="btn-genre-apply"></button>
      <button id="btn-genre-reset"></button>
      <button id="btn-genre-defaults"></button>
    </div>
  </div>
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
  invalidateWeightJacobian: ReturnType<typeof vi.fn>;
  restoreExactWeightBaseline: ReturnType<typeof vi.fn>;
  canvas: {
    hoverTarget: HoverTarget | null;
  };
  coord: AnalysisCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
}

/** Route a URL to its response body (default: `{}`). */
function makeHarness(
  routes: (url: string) => unknown = () => ({}),
  head?: { syncTrainEnabled(disabled: boolean): void; refreshHeadInfo?(): void },
): Harness {
  const model = new Model();
  const update = vi.fn();
  const scheduleDraw = vi.fn();
  const refreshProjection = vi.fn();
  const applyApproximateWeights = vi.fn(() => true);
  const invalidateWeightJacobian = vi.fn();
  const restoreExactWeightBaseline = vi.fn(() => true);
  const canvas = {
    hoverTarget: null as HoverTarget | null,
  };
  const coord = new AnalysisCoordinator({
    model,
    status: { update },
    scheduleDraw,
    canvas,
    refreshProjection,
    applyApproximateWeights,
    invalidateWeightJacobian,
    restoreExactWeightBaseline,
    head,
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
    invalidateWeightJacobian,
    canvas,
    coord,
    fetchSpy,
    restoreExactWeightBaseline,
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

  it("ignores a layout phase from a different in-flight request", async () => {
    const { model, coord } = makeHarness();
    const es = await startSettled(coord);
    model.pendingProjectionQuery = "sources=clap";
    // A stale request's phase must not overwrite the current one's progress.
    es.emit("layout_progress", {
      phase: "jacobian",
      done: 2,
      total: 5,
      query: "sources=rhythm",
    });
    expect(model.layoutProgress).toBeNull();
    es.emit("layout_progress", {
      phase: "jacobian",
      done: 2,
      total: 5,
      query: "sources=clap",
    });
    expect(model.layoutProgress).toMatchObject({ phase: "jacobian" });
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

  it("invalidates predicted folders when a supervise run finishes", async () => {
    let outstanding = 1;
    const refreshHeadInfo = vi.fn();
    const { coord } = makeHarness(
      (url) => {
        if (url.startsWith("/api/jobs/status")) {
          return makeQueueStatus({
            kinds: { supervise: makeKind({ outstanding }) },
          });
        }
        return {};
      },
      { syncTrainEnabled: vi.fn(), refreshHeadInfo },
    );
    const invalidate = vi.spyOn(coord, "refreshFolderPredictions");
    // First poll records the outstanding run…
    await coord.refreshQueueStatus();
    expect(invalidate).not.toHaveBeenCalled();
    expect(refreshHeadInfo).not.toHaveBeenCalled();
    // …the next poll sees it drain, so the cached head answer is stale.
    outstanding = 0;
    await coord.refreshQueueStatus();
    expect(invalidate).toHaveBeenCalled();
    expect(refreshHeadInfo).toHaveBeenCalled();
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

  it("coalesces concurrent fetches into one trailing fetch", async () => {
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
    // The second call is coalesced, not issued.
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    resolveFetch(jsonResponse({ available: false }));
    await Promise.all([p1, p2]);
    // Settling the first issues exactly one trailing fetch for the latest state.
    expect(fetchSpy).toHaveBeenCalledTimes(2);
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
  it("cycles colour modes and folds the gate out only in head mode", () => {
    const { model, coord, scheduleDraw } = makeHarness();
    model.clustersAvailable = true;
    coord.bindPanel();
    vi.spyOn(coord, "fetchFolderPredictions").mockResolvedValue();

    coord.cycleColorMode();
    expect(model.colorMode).toBe("head");
    expect(
      document.getElementById("head-confident-fold")!.classList.contains("hidden"),
    ).toBe(false);
    expect(
      document.getElementById("view-options")!.classList.contains("hidden"),
    ).toBe(false);
    expect(scheduleDraw).toHaveBeenCalled();

    coord.cycleColorMode();
    expect(model.colorMode).toBe("cluster");

    coord.cycleColorMode();
    expect(model.colorMode).toBe("tempo");

    coord.cycleColorMode();
    expect(model.colorMode).toBe("key");

    coord.cycleColorMode();
    expect(model.colorMode).toBe("folder");
    expect(
      document.getElementById("head-confident-fold")!.classList.contains("hidden"),
    ).toBe(true);
    expect(
      document.getElementById("view-options")!.classList.contains("hidden"),
    ).toBe(true);
  });

  it("cycles colour modes backward with shift", () => {
    const { model, coord } = makeHarness();
    model.clustersAvailable = true;
    model.colorMode = "folder";
    coord.bindPanel();
    vi.spyOn(coord, "fetchFolderPredictions").mockResolvedValue();

    coord.cycleColorMode(-1);
    expect(model.colorMode).toBe("key");
    coord.cycleColorMode(-1);
    expect(model.colorMode).toBe("tempo");
    coord.cycleColorMode(-1);
    expect(model.colorMode).toBe("cluster");
  });

  it("refuses to colour by cluster when none are available, and says why", () => {
    const { model, coord, scheduleDraw } = makeHarness();
    model.clustersAvailable = false;
    // Start one step before cluster so the refused cycle is the one under test.
    model.colorMode = "head";
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();
    vi.mocked(toast).mockClear();

    coord.cycleColorMode();

    expect(model.colorMode).toBe("head");
    expect(toast).toHaveBeenCalledWith(
      expect.stringContaining("Rebuild clusters"),
      "warn",
    );
    // No point fetching: we already know it is unavailable.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(scheduleDraw).not.toHaveBeenCalled();
  });

  it("applies a colour mode picked from the top tabber", () => {
    const { model, coord } = makeHarness();
    model.clustersAvailable = true;
    coord.bindPanel();

    coord.applyColorMode("key", false);

    expect(model.colorMode).toBe("key");
  });

  it("moves the selection-bloom cursor independently of the colour mode", () => {
    const { model, coord, scheduleDraw } = makeHarness();
    coord.bindPanel();
    model.colorMode = "folder";
    model.selectionBloom = "folder";

    coord.moveSelectionBloom(1);

    expect(model.selectionBloom).toBe("head");
    // The dot colour is untouched: the cursor is a separate selection.
    expect(model.colorMode).toBe("folder");
    expect(scheduleDraw).toHaveBeenCalled();

    coord.moveSelectionBloom(-1);
    expect(model.selectionBloom).toBe("folder");
  });

  it("fetches predicted folders when the bloom cursor reaches head", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/supervised/predictions")
        ? {
            available: true,
            assignments: { "a.mp3": "Root/Techno" },
            matched: ["a.mp3"],
          }
        : {},
    );
    coord.bindPanel();
    expect(model.folderPredictionsLoaded).toBe(false);

    coord.applySelectionBloom("head");
    await flush();

    expect(model.selectionBloom).toBe("head");
    expect(model.folderPredictionsLoaded).toBe(true);
  });

  it("fetches predicted folders on first selection of that mode", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/supervised/predictions")
        ? {
            available: true,
            assignments: { "a.mp3": "Root/Techno" },
            matched: ["a.mp3"],
          }
        : {},
    );
    coord.bindPanel();
    expect(model.folderPredictionsLoaded).toBe(false);

    coord.cycleColorMode(); // folder → head
    await flush();

    expect(model.colorMode).toBe("head");
    expect(model.folderPredictionsAvailable).toBe(true);
    expect(model.folderPredictions.get("a.mp3")).toBe("Root/Techno");
    expect(model.folderPredictionMatches.has("a.mp3")).toBe(true);
    expect(model.folderPredictionsLoaded).toBe(true);
  });

  it("warns when no head is trained and falls back to folder colours", async () => {
    const { model, coord } = makeHarness(() => ({
      available: false,
      assignments: {},
      matched: [],
    }));
    coord.bindPanel();
    vi.mocked(toast).mockClear();

    coord.cycleColorMode(); // folder → head
    await flush();

    expect(model.colorMode).toBe("head");
    expect(model.folderPredictionsAvailable).toBe(false);
    expect(model.folderPredictions.size).toBe(0);
    expect(toast).toHaveBeenCalledWith(
      expect.stringContaining("Train the folder head"),
      "warn",
    );
  });

  it("persists the confidence gate and folds it out only in head mode", () => {
    const { model, coord } = makeHarness();
    model.colorMode = "head";
    coord.bindPanel();

    const $hc = document.getElementById(
      "head-color-confident",
    ) as HTMLInputElement;
    expect($hc.checked).toBe(true);

    $hc.checked = false;
    $hc.dispatchEvent(new Event("change"));
    expect(model.headColorConfident).toBe(false);

    // Switching away folds the gate back, which only affects predicted folders.
    model.colorMode = "folder";
    coord.applyColorMode("folder", false);
    expect(
      document.getElementById("head-confident-fold")!.classList.contains("hidden"),
    ).toBe(true);
    expect(
      document.getElementById("view-options")!.classList.contains("hidden"),
    ).toBe(true);
  });

  it("invalidates and refetches predictions when the head changes", async () => {
    const { model, coord } = makeHarness(() => ({ available: false }));
    model.folderPredictionsLoaded = true;
    model.colorMode = "head";
    const fetchSpy = vi
      .spyOn(coord, "fetchFolderPredictions")
      .mockResolvedValue();

    coord.refreshFolderPredictions();

    expect(model.folderPredictionsLoaded).toBe(false);
    expect(fetchSpy).toHaveBeenCalled();
  });

  it("does not refetch when cycling colouring", () => {
    const { model, coord } = makeHarness();
    model.clustersAvailable = true;
    model.colorMode = "head";
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();
    coord.cycleColorMode();
    expect(model.colorMode).toBe("cluster");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("maps the resolution slider index to a resolution value", () => {
    const { model, coord } = makeHarness();
    model.clusterResolutions = [0.4, 1.5, 3.0];
    model.clusterResolution = 1.5;
    model.clustersAvailable = true;
    coord.bindPanel();
    const fetchSpy = vi.spyOn(coord, "fetchClusters").mockResolvedValue();

    const $cr = document.getElementById(
      "cluster-resolution-range",
    ) as HTMLInputElement;
    expect($cr.value).toBe("1");
    $cr.value = "2";
    $cr.dispatchEvent(new Event("input"));

    expect(model.clusterResolution).toBe(3.0);
    expect(fetchSpy).toHaveBeenCalled();
    // Adjusting granularity switches to cluster colouring so the change is visible.
    expect(model.colorMode).toBe("cluster");
  });

  it("coalesces a mid-fetch resolution change into one trailing fetch", async () => {
    const { model, coord } = makeHarness();
    const calls: string[] = [];
    const resolvers: Array<() => void> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        calls.push(String(input));
        return new Promise<Response>((resolve) => {
          resolvers.push(() => resolve(jsonResponse({ available: false })));
        });
      }),
    );

    model.clusterResolution = 1.5;
    const first = coord.fetchClusters();
    // The slider moves again before the first fetch settles.
    model.clusterResolution = 0.4;
    const second = coord.fetchClusters();
    expect(calls).toHaveLength(1);

    resolvers[0]();
    await first;
    await second;
    // The trailing fetch picks up the resolution the drag ended on.
    expect(calls).toHaveLength(2);
    expect(calls[1]).toContain("resolution=0.4");
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

  it("is hidden until the server enables genre_full", () => {
    const { model, coord } = makeHarness();
    coord.bindPanel();
    const $fold = document.getElementById("genre-weights-fold")!;
    expect($fold.classList.contains("hidden")).toBe(true);
    model.genreFullEnabled = true;
    model.lensId = "genre_full";
    coord["_syncColorControls"]();
    expect($fold.classList.contains("hidden")).toBe(false);
  });

  it("applies genre weights as a draft, posting the recipe + vector on Apply", async () => {
    const {
      model,
      coord,
      fetchSpy,
      refreshProjection,
      invalidateWeightJacobian,
    } = makeHarness();
    coord.bindPanel();
    const $rhythm = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='rhythm']",
    )!;
    $rhythm.value = "0.3";
    $rhythm.dispatchEvent(new Event("input"));
    // A live drag only drafts + previews — it never writes disk or the server.
    expect(model.genreDraft?.rhythm).toBe(0.3);
    expect(model.genreWeights.rhythm).toBe(0);
    expect(didRebuild(fetchSpy)).toBe(false);

    document.getElementById("btn-genre-apply")!.click();
    await flush();

    const expected = { ...model.genreWeights, rhythm: 0.3 };
    expect(rebuildBody(fetchSpy)).toEqual({
      recipe: "genre_full",
      weights: expected,
    });
    expect(model.genreWeights.rhythm).toBe(0.3);
    expect(model.genreDraft).toBeNull();
    expect(refreshProjection).toHaveBeenCalled();
    // The applied baseline moved, so the cached Jacobian was invalidated.
    expect(invalidateWeightJacobian).toHaveBeenCalled();
  });

  it("leaves the applied vector and draft intact when the cluster rebuild fails", async () => {
    const { model, coord, fetchSpy, refreshProjection, invalidateWeightJacobian } =
      makeHarness((url) => {
        // The harness always marks responses ok, so a network failure is a throw.
        if (url === "/api/clusters/rebuild") throw new Error("boom");
        return {};
      });
    const original = { ...model.genreWeights };
    coord.bindPanel();
    const $rhythm = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='rhythm']",
    )!;
    $rhythm.value = "0.3";
    $rhythm.dispatchEvent(new Event("input"));
    document.getElementById("btn-genre-apply")!.click();
    await flush();

    // Nothing was applied server-side, so nothing is committed locally or
    // persisted: the applied vector and the live draft are both unchanged.
    expect(model.genreWeights).toEqual(original);
    expect(model.genreDraft?.rhythm).toBe(0.3);
    expect(invalidateWeightJacobian).not.toHaveBeenCalled();
    expect(refreshProjection).not.toHaveBeenCalled();
    // The draft note still reads "not applied".
    expect(document.getElementById("genre-weights-note")!.textContent).toContain(
      "not applied",
    );
    expect(toast).toHaveBeenCalledWith(
      expect.stringMatching(/not applied/i),
      "error",
    );
  });

  it("surfaces an exact-refit failure after the vector was applied server-side", async () => {
    const { model, coord, refreshProjection, invalidateWeightJacobian } = makeHarness();
    // The cluster rebuild succeeds; the exact projection refit fails.
    refreshProjection.mockRejectedValue(new Error("projection down"));
    coord.bindPanel();
    const $rhythm = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='rhythm']",
    )!;
    $rhythm.value = "0.3";
    $rhythm.dispatchEvent(new Event("input"));
    document.getElementById("btn-genre-apply")!.click();
    await flush();

    // The server applied the vector, so it is committed and persisted...
    expect(model.genreWeights.rhythm).toBe(0.3);
    expect(model.genreDraft).toBeNull();
    expect(invalidateWeightJacobian).toHaveBeenCalled();
    // ...but the refit failure is surfaced (toast + note), not silently swallowed,
    // and the note never claims the positions are exact.
    expect(toast).toHaveBeenCalledWith(
      expect.stringMatching(/refit/i),
      "error",
    );
    const note = document.getElementById("genre-weights-note")!.textContent;
    expect(note).toContain("refit");
    expect(note).toContain("not exact");
  });

  it("rejects an all-zero genre vector and keeps the draft", async () => {
    const { model, coord, fetchSpy } = makeHarness();
    coord.bindPanel();
    const inputs = document.querySelectorAll<HTMLInputElement>(
      "#genre-weights input[type='range']",
    );
    for (const i of inputs) {
      i.value = "0";
      i.dispatchEvent(new Event("input"));
    }
    document.getElementById("btn-genre-apply")!.click();
    await flush();
    expect(didRebuild(fetchSpy)).toBe(false);
    expect(model.genreWeights).not.toBeNull();
  });

  it("Reset cancels a genre draft back to the applied weights without a request", async () => {
    const { model, coord, fetchSpy, applyApproximateWeights } = makeHarness();
    model.genreWeights = { ...model.genreWeights, maest_logits: 0.5 };
    coord.bindPanel();
    const $maest = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='maest_logits']",
    )!;
    $maest.value = "0.1";
    $maest.dispatchEvent(new Event("input"));
    expect(model.genreDraft).not.toBeNull();

    document.getElementById("btn-genre-reset")!.click();
    await flush();
    expect($maest.valueAsNumber).toBeCloseTo(0.5);
    expect(model.genreDraft).toBeNull();
    expect(applyApproximateWeights).toHaveBeenCalledWith(model.genreWeights);
    expect(didRebuild(fetchSpy)).toBe(false);
  });

  it("Defaults loads the recipe defaults as a draft, not applied", async () => {
    const { model, coord, fetchSpy } = makeHarness();
    model.genreWeights = { ...model.genreWeights, rhythm: 0.5 };
    coord.bindPanel();
    document.getElementById("btn-genre-defaults")!.click();
    expect(model.genreDraft?.rhythm).toBe(0);
    expect(model.genreWeights.rhythm).toBe(0.5); // still the applied vector
    expect(didRebuild(fetchSpy)).toBe(false);
  });

  it("Reset restores the exact baseline from the local cache when the Jacobian is unavailable", async () => {
    const { model, coord, applyApproximateWeights, restoreExactWeightBaseline, fetchSpy } =
      makeHarness();
    model.genreWeights = { ...model.genreWeights, maest_logits: 0.5 };
    coord.bindPanel();
    const $maest = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='maest_logits']",
    )!;
    $maest.value = "0.1";
    $maest.dispatchEvent(new Event("input"));
    // The drag could not be previewed (no Jacobian): the dots never moved, but a
    // Reset must still restore the exact applied baseline — from the cache, no HTTP.
    applyApproximateWeights.mockReturnValue(false);
    restoreExactWeightBaseline.mockReturnValue(true);
    document.getElementById("btn-genre-reset")!.click();
    expect($maest.valueAsNumber).toBeCloseTo(0.5);
    expect(model.genreDraft).toBeNull();
    expect(restoreExactWeightBaseline).toHaveBeenCalled();
    expect(didRebuild(fetchSpy)).toBe(false);
  });

  it("Reset refetches when neither the Jacobian nor the cache can restore the baseline", async () => {
    const { model, coord, applyApproximateWeights, restoreExactWeightBaseline, refreshProjection } =
      makeHarness();
    model.genreWeights = { ...model.genreWeights, maest_logits: 0.5 };
    coord.bindPanel();
    const $maest = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='maest_logits']",
    )!;
    $maest.value = "0.1";
    $maest.dispatchEvent(new Event("input"));
    applyApproximateWeights.mockReturnValue(false);
    restoreExactWeightBaseline.mockReturnValue(false);
    document.getElementById("btn-genre-reset")!.click();
    expect($maest.valueAsNumber).toBeCloseTo(0.5);
    expect(refreshProjection).toHaveBeenCalled();
  });

  it("does not claim the positions are exact when a preview is held and a drag cannot approximate", async () => {
    const { model, coord, applyApproximateWeights } = makeHarness();
    model.genreWeights = { ...model.genreWeights, maest_logits: 0.5 };
    coord.bindPanel();
    const $maest = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='maest_logits']",
    )!;
    // First drag previews via the Jacobian → a preview is now held on screen.
    applyApproximateWeights.mockReturnValue(true);
    $maest.value = "0.1";
    $maest.dispatchEvent(new Event("input"));
    const note = document.getElementById("genre-weights-note")!;
    expect(note.textContent).toContain("not applied yet");
    // A second drag cannot approximate (Jacobian lost): the note must NOT claim
    // the (approximate) positions are exact.
    applyApproximateWeights.mockReturnValue(false);
    $maest.value = "0.2";
    $maest.dispatchEvent(new Event("input"));
    expect(note.textContent).toContain("preview unavailable");
    expect(note.textContent).not.toContain("exact");
  });

  it("disables a genre contributor the server has off, keeping its canonical weight", async () => {
    const { model, coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status")
        ? {
            tracks_total: 10,
            tracks_with_fingerprint: 10,
            sources: [
              { id: "maest", label: "MAEST", done: 10, pending: 0, failed: 0 },
              { id: "rhythm", label: "Rhythm", done: 10, pending: 0, failed: 0 },
              { id: "clap", label: "CLAP", done: 10, pending: 0, failed: 0 },
              { id: "supervised", label: "Supervised", done: 10, pending: 0, failed: 0 },
              // No mert_l19 row → MERT is not enabled on this server.
            ],
          }
        : {},
    );
    model.mertEnabled = false;
    coord.bindPanel();
    await coord.fetchCoverage();
    const mert = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='mert_l19']",
    )!;
    expect(mert.disabled).toBe(true);
    // The canonical weight is kept — never silently rewritten to zero.
    expect(mert.valueAsNumber).toBeCloseTo(0.15);
    expect(document.querySelector<HTMLElement>(".cluster-weight-unavailable")).toBeTruthy();
    // A source with cached vectors stays enabled.
    const maest = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='maest']",
    )!;
    expect(maest.disabled).toBe(false);
  });

  it("disables a contributor with no cached vectors, but not a zero-weight one that is available", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status")
        ? {
            tracks_total: 10,
            tracks_with_fingerprint: 10,
            sources: [
              { id: "maest", label: "MAEST", done: 10, pending: 0, failed: 0 },
              { id: "rhythm", label: "Rhythm", done: 0, pending: 10, failed: 0 },
              { id: "clap", label: "CLAP", done: 10, pending: 0, failed: 0 },
              { id: "supervised", label: "Supervised", done: 10, pending: 0, failed: 0 },
              { id: "mert_l19", label: "MERT L19", done: 10, pending: 0, failed: 0 },
            ],
          }
        : {},
    );
    coord.bindPanel();
    await coord.fetchCoverage();
    // Rhythm has no cached vectors yet → disabled (dragging it would do nothing).
    const rhythm = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='rhythm']",
    )!;
    expect(rhythm.disabled).toBe(true);
    // clap is at its canonical zero weight but IS available → stays enabled, so a
    // Jacobian can drag it up.
    const clap = document.querySelector<HTMLInputElement>(
      "#genre-weights input[data-source='clap']",
    )!;
    expect(clap.valueAsNumber).toBe(0);
    expect(clap.disabled).toBe(false);
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

  it("reflects the actual server weights, not the default sliders", async () => {
    // A genre_full cluster zeroes rhythm on the server, so the note must not
    // claim "Rhythm has no cached vectors" even though the default slider shows 0.15.
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
            sources_used: ["maest_logits", "maest", "maest_layer7"],
            weights: {
              maest_logits: 0.4,
              maest: 0.2,
              maest_layer7: 0.25,
              mert_l19: 0.15,
              rhythm: 0,
              clap: 0,
              supervised: 0,
              features: 0,
            },
          }
        : {},
    );
    await coord.fetchClusters();
    const note = document.getElementById("cluster-weights-note")!.textContent!;
    expect(note).not.toContain("Rhythm");
    // A genre source with no cached rows IS still flagged.
    expect(note).toContain("MERT L19");
  });
});

// ─── Coverage panel ─────────────────────────────────────────

describe("AnalysisCoordinator coverage panel", () => {
  const COVERAGE = {
    tracks_total: 10,
    tracks_with_fingerprint: 10,
    sources: [
      { id: "features", label: "Audio features", done: 8, pending: 2, failed: 0 },
      { id: "rhythm", label: "Rhythm", done: 7, pending: 3, failed: 0 },
      { id: "structure", label: "Structure", done: 6, pending: 4, failed: 0 },
      { id: "maest", label: "MAEST", done: 9, pending: 1, failed: 0 },
      { id: "clap", label: "CLAP", done: 5, pending: 5, failed: 0 },
      { id: "supervised", label: "Supervised", done: 0, pending: 10, failed: 0 },
    ],
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
            sources: COVERAGE.sources.map((s) =>
              s.id === "structure" ? { ...s, pending: 0, failed: 4 } : s,
            ),
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

  it("re-runs a stage from its coverage row's hover icon, after confirming", async () => {
    const confirm = vi.fn(() => true);
    vi.stubGlobal("confirm", confirm);
    const { coord, fetchSpy } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status") ? COVERAGE : {},
    );
    coord.bindPanel();
    await coord.fetchCoverage();

    const buttons = [
      ...document.querySelectorAll<HTMLButtonElement>("#analysis-retrigger button"),
    ];
    expect(buttons.map((b) => b.textContent)).toEqual(["Re-run all"]);

    const rows = [...document.querySelectorAll(".analysis-coverage-row")];
    const rerun = rows[2].querySelector<HTMLButtonElement>(
      ".analysis-coverage-rerun",
    );
    expect(rerun?.title).toContain("Structure");
    rerun!.click();
    await flush();

    expect(confirm).toHaveBeenCalled();
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

  it("leaves rows sharing a stage or without one icon-free", async () => {
    const { coord } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status") ? COVERAGE : {},
    );
    await coord.fetchCoverage();

    const rows = [...document.querySelectorAll(".analysis-coverage-row")];
    const hasIcon = (r: Element) =>
      Boolean(r.querySelector(".analysis-coverage-rerun"));
    // Audio features / Structure / MAEST / CLAP own a stage; Rhythm shares
    // features, Supervised is library-wide.
    expect(rows.map(hasIcon)).toEqual([true, false, true, true, true, false]);
  });

  it("does not re-run when the confirmation is dismissed", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false));
    const { coord, fetchSpy } = makeHarness((url) =>
      url.startsWith("/api/embeddings/status") ? COVERAGE : {},
    );
    coord.bindPanel();
    await coord.fetchCoverage();

    const rows = [...document.querySelectorAll(".analysis-coverage-row")];
    rows[2]
      .querySelector<HTMLButtonElement>(".analysis-coverage-rerun")!
      .click();
    await flush();

    expect(
      fetchSpy.mock.calls.some(
        ([url, init]) =>
          String(url) === "/api/embeddings/generate" &&
          (init as RequestInit)?.method === "POST",
      ),
    ).toBe(false);
  });

  it("forces a re-run of every stage from Re-run all, after confirming", async () => {
    const confirm = vi.fn(() => true);
    vi.stubGlobal("confirm", confirm);
    const { coord, fetchSpy } = makeHarness();
    coord.bindPanel();

    const all = document.querySelector<HTMLButtonElement>(
      "#analysis-retrigger .analysis-retrigger-all",
    );
    all!.click();
    await flush();

    expect(confirm).toHaveBeenCalled();
    const post = fetchSpy.mock.calls.find(
      ([url, init]) =>
        String(url) === "/api/embeddings/generate" &&
        (init as RequestInit)?.method === "POST",
    );
    expect(JSON.parse((post![1] as RequestInit).body as string)).toEqual({
      folder: "",
      recursive: true,
      stages: [],
      force: true,
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
