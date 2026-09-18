// ─── AnalysisCoordinator ──────────────────────────────────────
//
// Owns the analysis job-queue stream, authoritative status polling, and the
// microgenre cluster fetch/render/panel. Extracted from `Controller` — the queue
// and clustering are independent of view mode (analysis runs in the background
// regardless of what the user is looking at), which makes this a clean seam.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import {
  type HoverTarget,
  type Model,
  type QueueStatus,
  DEFAULT_CLUSTER_WEIGHTS,
  CLUSTER_WEIGHT_LABELS,
  CLUSTER_WEIGHT_DESCRIPTIONS,
  formatSourceWeight,
} from "../model";
import { toast } from "../lib/toast";
import { clusterColor } from "../lib/colors";
import { api, postJSON } from "../lib/api";

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export interface AnalysisDeps {
  model: Model;
  /** Reflect queue/cluster changes in the status bar. */
  status: { update(m: Model): void };
  /** Repaint the canvas after cluster assignments change. */
  scheduleDraw(): void;
  /** Canvas hover state — hovering a cluster row glows its member points. */
  canvas: { hoverTarget: HoverTarget | null };
  /** Refetch the embedding layout after a weight change (optional in tests). */
  refreshProjection?(): Promise<void> | void;
  /** Approximate the layout for a live slider drag (optional in tests). */
  applyApproximateWeights?(weights: Record<string, number>): boolean;
  /** Fetch the source-weight Jacobian the approximation needs (optional). */
  fetchWeightJacobian?(): void;
  /** The head panel, so the queue can disable Train while a run is queued. */
  head?: { syncTrainEnabled(disabled: boolean): void };
  /** Select one track and fit the view to it (broken-files row click). */
  focusTrack?(path: string): void;
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
  /** Sources that actually contributed vectors (nonzero weight + cached rows). */
  sources_used?: string[];
}

/** Per-track coverage from ``GET /api/embeddings/status``. */
interface CoverageResponse {
  tracks_total?: number;
  tracks_with_fingerprint?: number;
  tracks_with_clap?: number;
  tracks_pending_clap?: number;
  tracks_failed_clap?: number;
  tracks_with_audio_features?: number;
  tracks_pending_audio_features?: number;
  tracks_failed_audio_features?: number;
  tracks_with_rhythm?: number;
  tracks_pending_rhythm?: number;
  tracks_failed_rhythm?: number;
  tracks_with_maest?: number;
  tracks_pending_maest?: number;
  tracks_failed_maest?: number;
  tracks_with_supervised?: number;
  tracks_pending_supervised?: number;
  tracks_failed_supervised?: number;
  tracks_with_structure?: number;
  tracks_pending_structure?: number;
  tracks_failed_structure?: number;
  generating?: boolean;
}

/** One coverage row: label, job kind to retrigger, done, pending, total. */
interface CoverageRow {
  label: string;
  stage: string;
  done: number;
  pending: number;
  /** Terminal failures for this stage (from the queue), so a stalled bar is explained. */
  failed: number;
}

/** One track with a terminal job failure (``GET /api/analysis/problems``). */
interface ProblemFailure {
  path: string;
  kinds: string[];
  errors: string[];
}

/** Shape of ``GET /api/analysis/problems``. */
interface ProblemsResponse {
  ok: boolean;
  /** Tracks fpcalc cannot read at all, so they never enter analysis. */
  unfingerprinted?: string[];
  /** Tracks with at least one permanently-failed stage job. */
  failed?: ProblemFailure[];
}

/** Retriggerable stages, in pipeline order. */
const RETRIGGER_STAGES: [string, string][] = [
  ["features", "Audio features"],
  ["structure", "Structure"],
  ["maest", "MAEST"],
  ["clap", "CLAP"],
];

export class AnalysisCoordinator {
  private _jobsEventSource: EventSource | null = null;
  private _clusterFetchPending = false;

  constructor(private deps: AnalysisDeps) {}

  /** Subscribe to the queue stream and pull the initial queue/cluster state. */
  start(): void {
    this._listenJobStream();
    void this.refreshQueueStatus();
    void this.fetchClusters();
    void this.fetchCoverage();
    void this.fetchProblems();
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

    this._renderWeightSliders();

    // Fetch the Jacobian when the panel is opened, not on every drag: it costs
    // one projection per source, so it must be computed once and then applied
    // linearly while a slider moves.
    const $details = document.getElementById("cluster-weights-details");
    if ($details) {
      $details.addEventListener("toggle", () => {
        if (($details as HTMLDetailsElement).open) {
          this.deps.fetchWeightJacobian?.();
        }
      });
    }

    const $apply = document.getElementById("btn-apply-weights");
    if ($apply) {
      $apply.addEventListener("click", () => void this._applyWeights());
    }

    const $reset = document.getElementById("btn-reset-weights");
    if ($reset) {
      $reset.addEventListener("click", () => {
        // Undo slider drags only: back to the *applied* vector, and restore the
        // map to the base layout the Jacobian was evaluated at (dw = 0, exact).
        // No server round-trip — the applied layout is already cached.
        this._syncWeightSliders();
        this.deps.applyApproximateWeights?.(m.clusterWeights);
      });
    }

    const $defaults = document.getElementById("btn-default-weights");
    if ($defaults) {
      $defaults.addEventListener("click", () => {
        m.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS };
        this._syncWeightSliders();
        void this._applyWeights();
      });
    }

    this._syncApplyEnabled();
    this._bindRetrigger();
    this._syncClusterControls();
  }

  /* ── Analysis coverage + per-stage retrigger ─────────────── */

  /** Wire the coverage panel's per-stage retrigger buttons. */
  private _bindRetrigger(): void {
    const host = document.getElementById("analysis-retrigger");
    if (!host) return;
    host.replaceChildren();
    for (const [stage, label] of RETRIGGER_STAGES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = label;
      btn.title = `Analyse every track missing ${label.toLowerCase()}`;
      btn.addEventListener("click", () => void this._retrigger(stage, label));
      host.appendChild(btn);
    }
    const all = document.createElement("button");
    all.type = "button";
    all.className = "analysis-retrigger-all";
    all.textContent = "Re-run all";
    all.title = "Re-run every per-track analysis stage, even where cached";
    all.addEventListener("click", () => void this._retrigger(null, "all stages"));
    host.appendChild(all);
  }

  /** Queue missing (or forced) work for one stage, or all stages. */
  private async _retrigger(stage: string | null, label: string): Promise<void> {
    try {
      await postJSON("/api/embeddings/generate", {
        folder: "",
        recursive: true,
        stages: stage ? [stage] : [],
        force: stage == null,
      });
      toast(`Queued ${label}`, "ok");
      await this.refreshQueueStatus();
      await this.fetchCoverage();
      await this.fetchProblems();
    } catch {
      /* api() already toasted */
    }
  }

  /** Fetch per-track analysis coverage for the coverage panel. */
  async fetchCoverage(): Promise<void> {
    try {
      const d = await api<CoverageResponse>(
        "/api/embeddings/status?folder=&recursive=1",
      );
      this._renderCoverage(d);
    } catch {
      /* transient; the next drain refreshes it */
    }
  }

  /** Fetch the actionable "broken files" list behind the coverage counts. */
  async fetchProblems(): Promise<void> {
    try {
      const d = await api<ProblemsResponse>("/api/analysis/problems");
      this._renderProblems(d);
    } catch {
      /* transient; the next drain refreshes it */
    }
  }

  /** Render the collapsible broken-files list; empty when nothing is broken. */
  private _renderProblems(d: ProblemsResponse): void {
    const host = document.getElementById("analysis-problems");
    if (!host) return;
    const unfingerprinted = d.unfingerprinted ?? [];
    const failed = d.failed ?? [];
    host.replaceChildren();
    if (unfingerprinted.length + failed.length === 0) return;

    const details = document.createElement("details");
    details.className = "analysis-problems";
    const summary = document.createElement("summary");
    summary.textContent = `Broken files (${unfingerprinted.length + failed.length})`;
    summary.title =
      "Tracks that cannot be analysed: unreadable audio, or a stage that failed. " +
      "Click a row to find it on the map.";
    details.appendChild(summary);

    const list = document.createElement("div");
    list.className = "analysis-problem-list";
    for (const path of unfingerprinted)
      list.appendChild(this._problemRow(path, "unreadable (no fingerprint)", []));
    for (const f of failed)
      list.appendChild(this._problemRow(f.path, `${f.kinds.join(", ")} failed`, f.kinds));
    details.appendChild(list);
    host.appendChild(details);
  }

  /** One broken-file row: click to locate it, reveal in Finder, or retry its stages. */
  private _problemRow(
    path: string,
    reason: string,
    kinds: string[],
  ): HTMLElement {
    const row = document.createElement("div");
    row.className = "analysis-problem";
    row.title = path;

    const track = this.deps.model.trackByPath(path);
    const text = document.createElement("div");
    text.className = "analysis-problem-text";
    const name = document.createElement("span");
    name.className = "analysis-problem-name";
    name.textContent = track?.filename || path.split("/").pop() || path;
    const where = document.createElement("span");
    where.className = "analysis-problem-folder";
    where.textContent = [track?.folder, reason].filter(Boolean).join(" · ");
    text.append(name, where);
    text.addEventListener("click", () => this.deps.focusTrack?.(path));
    row.appendChild(text);

    const reveal = document.createElement("button");
    reveal.type = "button";
    reveal.className = "analysis-problem-btn";
    reveal.textContent = "↗";
    reveal.title = "Reveal in Finder";
    reveal.addEventListener("click", (e) => {
      e.stopPropagation();
      const folder = track?.folder ?? path.split("/").slice(0, -1).join("/");
      void postJSON("/api/folders/reveal", { path: folder });
    });
    row.appendChild(reveal);

    if (kinds.length) {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "analysis-problem-btn";
      retry.textContent = "↻";
      retry.title = `Retry ${kinds.join(", ")}`;
      retry.addEventListener("click", (e) => {
        e.stopPropagation();
        void this._retryTrack(path, kinds);
      });
      row.appendChild(retry);
    }
    return row;
  }

  /** Re-enqueue the failed stages for one track only. */
  private async _retryTrack(path: string, kinds: string[]): Promise<void> {
    try {
      await postJSON("/api/embeddings/generate", {
        folder: "",
        recursive: true,
        stages: kinds,
        paths: [path],
      });
      toast("Retrying analysis…", "ok");
      await this.refreshQueueStatus();
      await this.fetchCoverage();
      await this.fetchProblems();
    } catch {
      /* api() already toasted */
    }
  }

  /** Render one row per source: done / total with a progress bar. */
  private _renderCoverage(d: CoverageResponse): void {
    const host = document.getElementById("analysis-coverage");
    if (!host) return;
    const total = d.tracks_total ?? 0;
    const fingerprinted = d.tracks_with_fingerprint ?? 0;
    // Failures come from the backend, already deduped by fingerprint against the
    // cache: a stale failure for a duplicate that is actually analysed is not a gap.
    const rows: CoverageRow[] = [
      {
        label: "Audio features",
        stage: "features",
        done: d.tracks_with_audio_features ?? 0,
        pending: d.tracks_pending_audio_features ?? 0,
        failed: d.tracks_failed_audio_features ?? 0,
      },
      {
        label: "Rhythm",
        stage: "features",
        done: d.tracks_with_rhythm ?? 0,
        pending: d.tracks_pending_rhythm ?? 0,
        failed: d.tracks_failed_rhythm ?? 0,
      },
      {
        label: "Structure",
        stage: "structure",
        done: d.tracks_with_structure ?? 0,
        pending: d.tracks_pending_structure ?? 0,
        failed: d.tracks_failed_structure ?? 0,
      },
      {
        label: "MAEST",
        stage: "maest",
        done: d.tracks_with_maest ?? 0,
        pending: d.tracks_pending_maest ?? 0,
        failed: d.tracks_failed_maest ?? 0,
      },
      {
        label: "CLAP",
        stage: "clap",
        done: d.tracks_with_clap ?? 0,
        pending: d.tracks_pending_clap ?? 0,
        failed: d.tracks_failed_clap ?? 0,
      },
      {
        label: "Supervised",
        stage: "supervise",
        done: d.tracks_with_supervised ?? 0,
        pending: d.tracks_pending_supervised ?? 0,
        failed: d.tracks_failed_supervised ?? 0,
      },
    ];

    // Per-stage failure counts (deduped: features and rhythm share one stage, so a
    // track whose features job failed must count once, not twice).
    const failedByStage = new Map<string, number>();
    for (const row of rows)
      failedByStage.set(
        row.stage,
        Math.max(failedByStage.get(row.stage) ?? 0, row.failed),
      );
    const failedTotal = [...failedByStage.values()].reduce((a, b) => a + b, 0);

    const summary = document.createElement("div");
    summary.className = "analysis-coverage-summary";
    summary.textContent =
      total > 0
        ? `${fingerprinted} / ${total} fingerprinted${
            failedTotal > 0 ? ` · ${failedTotal} failed` : ""
          }`
        : "No tracks in the library yet.";
    if (failedTotal > 0)
      summary.title =
        "Some tracks could not be analysed (unreadable audio). " +
        "Press a stage button below to retry its failed jobs.";
    host.replaceChildren(summary);

    if (total === 0) return;

    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "analysis-coverage-row";

      const name = document.createElement("span");
      name.className = "analysis-coverage-name";
      name.textContent = row.label;

      const count = document.createElement("span");
      count.className = "analysis-coverage-count";
      count.textContent = `${row.done} / ${total}`;
      if (row.pending > 0) count.classList.add("has-pending");
      // A short bar with terminal failures is a real gap, not pending work: flag
      // it and point at the retrigger button.
      const failed = row.failed;
      if (failed > 0) count.classList.add("has-failed");

      const bar = document.createElement("span");
      bar.className = "analysis-coverage-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.round((row.done / total) * 100)}%`;
      if (row.pending > 0) fill.classList.add("has-pending");
      bar.appendChild(fill);

      line.append(name, bar, count);
      line.title = `${row.done} of ${total} analysed${
        row.pending > 0 ? `, ${row.pending} pending` : ""
      }${failed > 0 ? `, ${failed} failed (retrigger to retry)` : ""}`;
      host.appendChild(line);
    }
  }

  /** Build one weight row per source, from the shared default/label maps. */
  private _renderWeightSliders(): void {
    const host = document.getElementById("cluster-weights");
    if (!host) return;
    host.replaceChildren();
    for (const source of Object.keys(DEFAULT_CLUSTER_WEIGHTS)) {
      const row = document.createElement("label");
      row.className = "cluster-weight-row";

      const name = document.createElement("span");
      name.className = "cluster-weight-name";
      name.textContent = CLUSTER_WEIGHT_LABELS[source] ?? source;

      const input = document.createElement("input");
      input.type = "range";
      input.min = "0";
      input.max = "1";
      input.step = "0.05";
      input.dataset.source = source;
      input.value = String(this.deps.model.clusterWeights[source] ?? 0);
      input.title =
        CLUSTER_WEIGHT_DESCRIPTIONS[source] ??
        `Relative influence of the ${source} source`;

      const value = document.createElement("span");
      value.className = "cluster-weight-value";
      value.textContent = formatSourceWeight(input.valueAsNumber);

      input.addEventListener("input", () => {
        value.textContent = formatSourceWeight(input.valueAsNumber);
        // Approximate the layout live; the full re-cluster + refit waits for
        // Apply, so dragging never blocks on the expensive path.
        this.deps.applyApproximateWeights?.(this._readWeightSliders());
      });

      row.append(name, input, value);
      host.appendChild(row);
    }
  }

  /** Mirror the model's weight vector onto the sliders (e.g. after Reset). */
  private _syncWeightSliders(): void {
    const w = this.deps.model.clusterWeights;
    for (const source of Object.keys(DEFAULT_CLUSTER_WEIGHTS)) {
      const input = document.querySelector<HTMLInputElement>(
        `#cluster-weights input[data-source="${source}"]`,
      );
      if (input) input.value = String(w[source] ?? 0);
      const readout = input?.nextElementSibling;
      if (readout) readout.textContent = formatSourceWeight(w[source] ?? 0);
    }
  }

  /** Read the live slider values into a complete weight map. */
  private _readWeightSliders(): Record<string, number> {
    const m = this.deps.model;
    const weights: Record<string, number> = {};
    for (const source of Object.keys(DEFAULT_CLUSTER_WEIGHTS)) {
      const input = document.querySelector<HTMLInputElement>(
        `#cluster-weights input[data-source="${source}"]`,
      );
      weights[source] = input
        ? clamp(input.valueAsNumber, 0, 1)
        : m.clusterWeights[source] ?? 0;
    }
    return weights;
  }

  /** Read the live slider values into the model and re-cluster + re-layout. */
  private async _applyWeights(): Promise<void> {
    const m = this.deps.model;
    const weights = this._readWeightSliders();
    m.clusterWeights = weights;
    m.saveLS();
    try {
      await postJSON("/api/clusters/rebuild", { weights });
      toast("Re-clustering with new weights…", "ok");
      // The layout is re-weighted independently of the cluster job, so refresh on
      // success only — a rejected vector would fail the projection the same way.
      // Await it before the Jacobian: the base pass must land first for the map
      // to respond, and the Jacobian's S+1 fits must not contend with it.
      await this.deps.refreshProjection?.();
      // The Jacobian was evaluated at the *old* weights, so the next drag would
      // step from a stale base; refetch it at the applied weights.
      this.deps.fetchWeightJacobian?.();
    } catch {
      /* api() already toasted */
    }
  }

  /** Disable Apply while a cluster job is outstanding: the queue's single
   *  ``"library"`` dedupe key would coalesce the request into the running one. */
  private _syncApplyEnabled(): void {
    const $apply = document.getElementById(
      "btn-apply-weights",
    ) as HTMLButtonElement | null;
    const kinds = this.deps.model.queueStatus?.kinds;
    const clusterBusy = (kinds?.cluster?.outstanding ?? 0) > 0;
    const trainBusy = (kinds?.supervise?.outstanding ?? 0) > 0;
    if ($apply) {
      $apply.disabled = clusterBusy;
      $apply.title = clusterBusy
        ? "A cluster job is already queued — it will use the current weights"
        : "Re-cluster and re-layout with these weights";
    }
    // The head panel owns the Train button; the queue only tells it to lock.
    this.deps.head?.syncTrainEnabled(trainBusy);
  }

  /* ── Cluster panel ──────────────────────────────────────── */

  /** Flip cluster colouring; shared by the panel checkbox and the hotkey. */
  toggleColorByCluster(): void {
    this._applyColorByCluster(!this.deps.model.colorByCluster, true);
  }

  private _applyColorByCluster(value: boolean, announce: boolean): void {
    const m = this.deps.model;
    if (value && !m.clustersAvailable) {
      // Never claim a colouring that cannot apply; say why and leave the toggle off.
      if (announce) toast(this._clusterUnavailableReason(), "warn");
      this._syncClusterControls();
      return;
    }
    m.colorByCluster = value;
    m.saveLS();
    if (announce) {
      toast(value ? "Colouring by cluster" : "Colouring by folder", "ok");
    }
    this.deps.scheduleDraw();
    this._syncClusterControls();
  }

  /** Why cluster colouring cannot apply yet, for the disabled toggle's tooltip. */
  private _clusterUnavailableReason(): string {
    const m = this.deps.model;
    if (m.queueStatus?.models?.maest_ready === false) {
      return "Waiting for the MAEST model to load…";
    }
    const clusterKind = m.queueStatus?.kinds?.cluster;
    if (clusterKind && clusterKind.outstanding > 0) return "Clustering…";
    return "No clusters yet — press Rebuild clusters.";
  }

  /** Reflect cluster availability on the panel controls (disabled + explained). */
  private _syncClusterControls(): void {
    const m = this.deps.model;
    const available = m.clustersAvailable;
    const reason = this._clusterUnavailableReason();
    const $cc = document.getElementById(
      "cluster-color-toggle",
    ) as HTMLInputElement | null;
    if ($cc) {
      $cc.disabled = !available;
      $cc.checked = available && m.colorByCluster;
      $cc.title = available
        ? "Colour tracks by their microgenre cluster"
        : reason;
    }
    const $cr = document.getElementById(
      "cluster-resolution-range",
    ) as HTMLInputElement | null;
    if ($cr) $cr.disabled = !available;
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
      this.deps.scheduleDraw();
      this._syncApplyEnabled();
    };

    es.addEventListener("snapshot", ((e: MessageEvent) => {
      applySnapshot(JSON.parse(e.data) as QueueStatus);
    }) as EventListener);

    es.addEventListener("stats", (() => {
      void this.refreshQueueStatus();
    }) as EventListener);

    es.addEventListener("stage_progress", ((e: MessageEvent) => {
      // In-flight phase of a library-wide singleton stage (supervise/cluster):
      // update the stage in place so both the status bar and the canvas
      // indicator re-render without waiting for the next snapshot.
      const d = JSON.parse(e.data) as {
        kind: string;
        phase: string;
        done?: number;
        total?: number;
      };
      const m = this.deps.model;
      const k = m.queueStatus?.kinds[d.kind];
      if (!k) return;
      k.stage = {
        phase: d.phase,
        done: d.done ?? 0,
        total: d.total ?? 0,
        at: Date.now() / 1000,
      };
      this.deps.status.update(m);
      this.deps.scheduleDraw();
    }) as EventListener);

    es.addEventListener("layout_progress", ((e: MessageEvent) => {
      // A layout request is not a queued job, so its phase rides its own event
      // instead of `stage_progress`. The canvas pill and status bar render it
      // while the projection is pending. The event echoes its request's query, so a
      // phase from an older overlapping request is ignored.
      const d = JSON.parse(e.data) as {
        phase: string;
        done?: number;
        total?: number;
        query?: string;
      };
      const m = this.deps.model;
      if (d.query && d.query !== m.pendingProjectionQuery) return;
      m.layoutProgress = {
        phase: d.phase,
        done: d.done ?? 0,
        total: d.total ?? 0,
      };
      this.deps.status.update(m);
      this.deps.scheduleDraw();
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
        this.deps.scheduleDraw();
        this._syncApplyEnabled();
      }
    }) as EventListener);

    es.addEventListener("drained", (() => {
      // Workers drain per kind, and every per-track drain queues the next
      // library-wide stage, so only a fully idle queue ends the whole run.
      void this.refreshQueueStatus().then(() => {
        if ((this.deps.model.queueStatus?.outstanding ?? 0) === 0) {
          toast("Analysis complete", "ok");
        }
        void this.fetchCoverage();
        void this.fetchProblems();
      });
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
      this.deps.scheduleDraw();
      this._syncApplyEnabled();
      this._syncClusterControls();
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
        this._renderWeightsNote(null);
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
      this._renderWeightsNote(d);
      this._renderClusterPanel();
      this.deps.scheduleDraw();
    } catch {
      /* api() already toasted */
    } finally {
      this._clusterFetchPending = false;
    }
  }

  /** Flag requested-but-unavailable sources (a nonzero weight that did nothing). */
  private _renderWeightsNote(d: ClusterResponse | null): void {
    const note = document.getElementById("cluster-weights-note");
    if (!note) return;
    if (!d) {
      note.textContent = "";
      return;
    }
    const used = new Set(d.sources_used ?? []);
    const requested = this.deps.model.clusterWeights;
    const dropped = Object.keys(DEFAULT_CLUSTER_WEIGHTS).filter(
      (s) => (requested[s] ?? 0) > 0 && !used.has(s),
    );
    note.textContent = dropped.length
      ? `No cached vectors for ${dropped
          .map((s) => CLUSTER_WEIGHT_LABELS[s] ?? s)
          .join(", ")} — weight ignored`
      : "";
  }

  /** Render the microgenre list (largest clusters first). */
  private _renderClusterPanel(): void {
    const m = this.deps.model;
    const $meta = document.getElementById("cluster-meta");
    const $list = document.getElementById("cluster-list");
    if (!$meta || !$list) return;

    // Rows are recreated below, so no mouseleave fires for a row that was
    // hovered across a re-render; drop the stale highlight explicitly.
    if (this.deps.canvas.hoverTarget?.kind === "cluster") {
      this.deps.canvas.hoverTarget = null;
      this.deps.scheduleDraw();
    }

    if (!m.clustersAvailable) {
      $meta.textContent = this._clusterUnavailableReason();
      $list.replaceChildren();
      this._syncClusterControls();
      return;
    }

    const n = m.clusterNames.size || m.clusterSizes.size;
    $meta.textContent = `${n} clusters at resolution ${m.clusterResolution}`;
    this._syncClusterControls();

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
      // Hover glows the cluster's points on the canvas; click selects them. The
      // highlight is purely visual, so it lives on the canvas view (no model event).
      li.addEventListener("mouseenter", () => {
        this.deps.canvas.hoverTarget = { kind: "cluster", id: cid };
        this.deps.scheduleDraw();
      });
      li.addEventListener("mouseleave", () => {
        if (this.deps.canvas.hoverTarget?.kind === "cluster" && this.deps.canvas.hoverTarget.id === cid) {
          this.deps.canvas.hoverTarget = null;
          this.deps.scheduleDraw();
        }
      });
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
