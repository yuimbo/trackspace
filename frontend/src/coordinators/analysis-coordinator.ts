// ─── AnalysisCoordinator ──────────────────────────────────────
//
// Owns the analysis job-queue stream, authoritative status polling, and the
// microgenre cluster fetch/render/panel. Extracted from `Controller` — the queue
// and clustering are independent of view mode (analysis runs in the background
// regardless of what the user is looking at), which makes this a clean seam.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type Model, type QueueStatus } from "../model";
import { toast, clusterColor } from "../lib/toast";
import { api, postJSON } from "../lib/api";

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export interface AnalysisDeps {
  model: Model;
  /** Reflect queue/cluster changes in the status bar. */
  status: { update(m: Model): void };
  /** Repaint the canvas after cluster assignments change. */
  scheduleDraw(): void;
}

/** Shape of ``GET /api/clusters``. */
interface ClusterResponse {
  available: boolean;
  revision?: string;
  resolution?: number;
  resolutions?: number[];
  n_clusters?: number;
  assignments?: Record<string, number>;
  names?: Record<string, string>;
  sizes?: Record<string, number>;
  modularity?: number | null;
}

export class AnalysisCoordinator {
  private _jobsEventSource: EventSource | null = null;
  private _clusterFetchPending = false;

  constructor(private deps: AnalysisDeps) {}

  /** Subscribe to the queue stream and pull the initial queue/cluster state. */
  start(): void {
    this._listenJobStream();
    void this.refreshQueueStatus();
    void this.fetchClusters();
  }

  stop(): void {
    this._closeJobStream();
  }

  /** Wire the cluster panel controls (colour toggle, resolution slider, rebuild). */
  bindPanel(): void {
    const m = this.deps.model;

    const $cc = document.getElementById(
      "cluster-color-toggle",
    ) as HTMLInputElement | null;
    if ($cc) {
      $cc.checked = m.colorByCluster;
      $cc.addEventListener("change", () => this._applyColorByCluster($cc.checked, false));
    }

    const $cr = document.getElementById(
      "cluster-resolution-range",
    ) as HTMLInputElement | null;
    if ($cr) {
      // The slider indexes the server's resolution list rather than carrying
      // raw values — the set of resolutions is a server-side decision.
      const syncSlider = () => {
        const list = m.clusterResolutions;
        if (!list.length) return;
        $cr.max = String(list.length - 1);
        const idx = list.indexOf(m.clusterResolution);
        $cr.value = String(idx >= 0 ? idx : Math.floor(list.length / 2));
      };
      syncSlider();
      $cr.addEventListener("change", () => {
        const list = m.clusterResolutions;
        if (!list.length) return;
        const idx = clamp(parseInt($cr.value, 10) || 0, 0, list.length - 1);
        m.clusterResolution = list[idx];
        m.saveLS();
        void this.fetchClusters();
      });
    }

    const $rebuild = document.getElementById("btn-rebuild-clusters");
    if ($rebuild) {
      $rebuild.addEventListener("click", async () => {
        try {
          await postJSON("/api/clusters/rebuild", {});
          toast("Clustering queued…", "ok");
        } catch {
          /* api() already toasted */
        }
      });
    }
  }

  /** Flip cluster colouring; shared by the panel checkbox and the hotkey. */
  toggleColorByCluster(): void {
    this._applyColorByCluster(!this.deps.model.colorByCluster, true);
  }

  private _applyColorByCluster(value: boolean, announce: boolean): void {
    const m = this.deps.model;
    m.colorByCluster = value;
    const $cc = document.getElementById(
      "cluster-color-toggle",
    ) as HTMLInputElement | null;
    if ($cc) $cc.checked = value;
    m.saveLS();
    if (value && !m.clustersAvailable) void this.fetchClusters();
    if (announce) {
      toast(value ? "Colouring by cluster" : "Colouring by folder", "ok");
    }
    this.deps.scheduleDraw();
  }

  /** Subscribe to queue progress.
   *
   * The server opens the stream with a full snapshot, so a client that
   * connects late (or reloads mid-run) immediately renders correct totals
   * rather than inferring them from subsequent events.
   */
  private _listenJobStream(): void {
    this._closeJobStream();
    const es = new EventSource("/api/jobs/stream");
    this._jobsEventSource = es;

    const applySnapshot = (data: QueueStatus) => {
      this.deps.model.queueStatus = data;
      this.deps.status.update(this.deps.model);
    };

    es.addEventListener("snapshot", ((e: MessageEvent) => {
      applySnapshot(JSON.parse(e.data) as QueueStatus);
    }) as EventListener);

    es.addEventListener("stats", (() => {
      void this.refreshQueueStatus();
    }) as EventListener);

    es.addEventListener("progress", ((e: MessageEvent) => {
      const d = JSON.parse(e.data) as {
        kind: string;
        path?: string;
        ok?: boolean;
      };
      const m = this.deps.model;
      const q = m.queueStatus;
      if (q && q.kinds[d.kind]) {
        // Optimistic local increment keeps the counter smooth between the
        // periodic authoritative refreshes below.
        const k = q.kinds[d.kind];
        if (d.ok) k.done++;
        else k.failed++;
        if (k.pending > 0) k.pending--;
        k.finished = k.done + k.failed;
        k.outstanding = k.pending + k.running;
        k.last = {
          path: d.path ?? null,
          ok: d.ok ?? null,
          detail: {},
          at: Date.now() / 1000,
        };
        this.deps.status.update(m);
      }
    }) as EventListener);

    es.addEventListener("drained", (() => {
      void this.refreshQueueStatus();
      toast("Analysis stage complete", "ok");
    }) as EventListener);

    es.onerror = () => {
      // Browser auto-reconnects EventSource; nothing to do but stop shouting.
    };
  }

  private _closeJobStream(): void {
    if (this._jobsEventSource) {
      this._jobsEventSource.close();
      this._jobsEventSource = null;
    }
  }

  /** Pull authoritative queue counts (SQL aggregates on the server). */
  async refreshQueueStatus(): Promise<void> {
    try {
      const st = await api<QueueStatus>("/api/jobs/status");
      const m = this.deps.model;
      const prevRevision = m.clusterRevision;
      m.queueStatus = st;
      this.deps.status.update(m);
      // Clustering finished or changed → refresh assignments.
      const rev = st.clusters?.revision ?? null;
      if (rev && rev !== prevRevision) {
        await this.fetchClusters();
      }
    } catch {
      /* transient; next tick retries */
    }
  }

  /** Fetch cluster assignments at the selected resolution. */
  async fetchClusters(): Promise<void> {
    if (this._clusterFetchPending) return;
    this._clusterFetchPending = true;
    const m = this.deps.model;
    try {
      const d = await api<ClusterResponse>(
        `/api/clusters?resolution=${m.clusterResolution}`,
      );

      if (!d.available) {
        m.clustersAvailable = false;
        m.clusterAssignments = new Map();
        m.clusterNames = new Map();
        m.clusterSizes = new Map();
        this._renderClusterPanel();
        return;
      }

      m.clustersAvailable = true;
      m.clusterRevision = d.revision ?? null;
      m.clusterResolutions = d.resolutions ?? [];
      if (typeof d.resolution === "number") m.clusterResolution = d.resolution;
      m.clusterAssignments = new Map(Object.entries(d.assignments ?? {}));
      m.clusterNames = new Map(
        Object.entries(d.names ?? {}).map(([k, v]) => [Number(k), v]),
      );
      m.clusterSizes = new Map(
        Object.entries(d.sizes ?? {}).map(([k, v]) => [Number(k), v]),
      );
      this._renderClusterPanel();
      this.deps.scheduleDraw();
    } catch {
      /* api() already toasted */
    } finally {
      this._clusterFetchPending = false;
    }
  }

  /** Render the microgenre list (largest clusters first). */
  private _renderClusterPanel(): void {
    const m = this.deps.model;
    const $meta = document.getElementById("cluster-meta");
    const $list = document.getElementById("cluster-list");
    if (!$meta || !$list) return;

    if (!m.clustersAvailable) {
      $meta.textContent = m.queueStatus?.models?.maest_ready
        ? "No clusters yet — run analysis."
        : "Waiting for MAEST model…";
      $list.replaceChildren();
      return;
    }

    const n = m.clusterNames.size || m.clusterSizes.size;
    $meta.textContent = `${n} clusters at resolution ${m.clusterResolution}`;

    const entries = [...m.clusterSizes.entries()].sort((a, b) => b[1] - a[1]);
    $list.replaceChildren();
    for (const [cid, size] of entries) {
      const li = document.createElement("li");
      li.className = "cluster-item";
      li.dataset.cluster = String(cid);

      const sw = document.createElement("span");
      sw.className = "cluster-swatch";
      sw.style.background = clusterColor(cid);

      const label = document.createElement("span");
      label.className = "cluster-name";
      label.textContent = m.clusterNames.get(cid) || `Cluster ${cid}`;

      const count = document.createElement("span");
      count.className = "cluster-count";
      count.textContent = String(size);

      li.append(sw, label, count);
      // Click a cluster to select its tracks — the main way to act on a
      // discovered microgenre (move to a folder, tag it, audition it).
      li.addEventListener("click", () => this._selectCluster(cid));
      $list.appendChild(li);
    }
  }

  private _selectCluster(cid: number): void {
    const m = this.deps.model;
    m.selected.clear();
    for (const [path, id] of m.clusterAssignments) {
      if (id === cid) m.selected.add(path);
    }
    m.emit("change");
    toast(
      `Selected ${m.selected.size} tracks in ${
        m.clusterNames.get(cid) || `cluster ${cid}`
      }`,
      "ok",
    );
  }
}
