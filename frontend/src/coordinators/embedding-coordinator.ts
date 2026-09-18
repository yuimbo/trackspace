// ─── EmbeddingCoordinator ─────────────────────────────────────
//
// Owns the embedding-space view: layout fetch/cache, the generation SSE stream,
// the model-ready poll, and every canvas *position* animation. Extracted from
// `Controller` — the embedding view is a self-contained mode whose state (layout
// cache, stream handles, animation frame) otherwise does not touch tag-space
// controller logic.
//
// Position animations all write the single `canvas.animPositions` overlay and
// share one requestAnimationFrame handle, so "a new animation cancels the
// previous one" is an invariant that must have exactly one owner. That is why
// `animateAxisChange` (a tag-space animation) lives here too: it writes the same
// overlay and must cancel the same frame.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type Model, DEFAULT_CLUSTER_WEIGHTS, formatSourceWeight } from "../model";
import { toast } from "../lib/toast";
import { api, postJSON } from "../lib/api";
import {
  approximatePositions,
  parseWeightJacobian,
  type WeightJacobian,
} from "../lib/weight-jacobian";

/** Subset of ``/api/embeddings/status`` used to avoid duplicate layout fetches. */
type EmbeddingLayoutStatusHint = {
  layout_revision?: string;
};

const LAYOUT_CACHE_MAX = 8;

/** True when *w* equals the frozen defaults, so the default query key is unchanged. */
function _weightsAreDefault(w: Record<string, number>): boolean {
  for (const [k, def] of Object.entries(DEFAULT_CLUSTER_WEIGHTS)) {
    if (Math.abs((w[k] ?? NaN) - def) > 1e-9) return false;
  }
  return true;
}

/** ``maest_logits:0.45,maest:0.25,…`` in canonical source order. */
function _encodeWeights(w: Record<string, number>): string {
  return Object.keys(DEFAULT_CLUSTER_WEIGHTS)
    .map((k) => `${k}:${formatSourceWeight(w[k] ?? 0)}`)
    .join(",");
}

interface CachedLayout {
  revision: string;
  positions: Map<string, { x: number; y: number }>;
}

/** ``/api/embeddings/projection`` returns ``{ positions, revision }`` (legacy bare array unsupported). */
function parseProjectionResponse(raw: unknown): {
  positions: { path: string; x: number; y: number }[];
  revision: string;
} {
  if (
    raw &&
    typeof raw === "object" &&
    "positions" in (raw as object) &&
    Array.isArray((raw as { positions: unknown }).positions)
  ) {
    const o = raw as { positions: { path: string; x: number; y: number }[]; revision?: string };
    return { positions: o.positions, revision: o.revision ?? "" };
  }
  return { positions: [], revision: "" };
}

export interface EmbeddingDeps {
  model: Model;
  /** Reflect view-mode / projection changes in the status bar. */
  status: { update(m: Model): void };
  /** The canvas overlay the position animations drive (mutable, not just a repaint). */
  canvas: {
    animPositions: Map<string, { x: number; y: number }> | null;
    scheduleDraw(): void;
  };
}

export class EmbeddingCoordinator {
  /** Saved viewport state per view-mode so pan/zoom persists across toggles. */
  private _savedVpByMode = new Map<string, { ox: number; oy: number; zoom: number }>();

  private _embedPollTimer = 0;
  private _embedEventSource: EventSource | null = null;
  private _embedBatchCount = 0;
  private _projectionFetchPending = false;
  /** Query-string → {revision, positions} map so A→B→A revisits are instant. */
  private _layoutCache = new Map<string, CachedLayout>();

  /** Source-weight Jacobian at the last fetched weights, for live drags. */
  private _weightJacobian: WeightJacobian | null = null;
  /** Query string the cached Jacobian was evaluated at (empty = none). */
  private _jacobianKey = "";

  private _animFrame = 0;
  /** Monotonic request id; a slow projection must not overwrite a newer one. */
  private _projectionReqSeq = 0;

  constructor(private deps: EmbeddingDeps) {}

  /** Cancel the in-flight position animation, if any (shared overlay invariant). */
  cancelAnimation(): void {
    cancelAnimationFrame(this._animFrame);
  }

  /**
   * Animate the viewport (camera) to *target* on the shared animation frame.
   *
   * A viewport move is not a *dot* position animation, but it shares the one frame
   * so a fit can never interleave with an in-flight position lerp. Re-fits during
   * the tween cancel it and start from wherever the camera has reached.
   */
  tweenViewport(
    target: { ox: number; oy: number; zoom: number },
    ms = 420,
  ): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    cancelAnimationFrame(this._animFrame);
    const from = { ox: m.vp.ox, oy: m.vp.oy, zoom: m.vp.zoom };
    const t0 = performance.now();
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / Math.max(1, ms));
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      m.vp.ox = from.ox + (target.ox - from.ox) * e;
      m.vp.oy = from.oy + (target.oy - from.oy) * e;
      m.vp.zoom = from.zoom + (target.zoom - from.zoom) * e;
      cv.scheduleDraw();
      if (t < 1) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        m.vp.ox = target.ox;
        m.vp.oy = target.oy;
        m.vp.zoom = target.zoom;
        m.saveLS();
        cv.scheduleDraw();
      }
    };
    this._animFrame = requestAnimationFrame(step);
  }

  /** Stop background work: the model-ready poll and the generation SSE stream. */
  stop(): void {
    clearTimeout(this._embedPollTimer);
    this._embedPollTimer = 0;
    this._closeEmbeddingStream();
    this.cancelAnimation();
  }

  /** Drop all cached layouts (e.g. after a full library reload). */
  invalidateLayoutCache(): void {
    this._layoutCache.clear();
  }

  /** Refetch the layout after a weight change (the query string carries them).
   *
   *  Safe to call from any view mode: when the map has never been drawn there is
   *  nothing to refresh (the next entry fetches with the new weights), and when it
   *  has, the new positions are stored so switching to the map animates to them.
   */
  refreshProjection(): Promise<void> {
    if (!this.deps.model.embeddingsReady) return Promise.resolve();
    return this._fetchProjection();
  }

  /** Fetch the source-weight Jacobian at the current weights, once per weights.
   *
   *  The server evaluates it at the weights it is queried with, so the base
   *  vector must match the one the drag starts from. It costs one projection per
   *  source, so the result is cached by the query string: reopening the panel or
   *  dragging back to the same weights must not recompute it. A failed fetch
   *  leaves the approximation unavailable and the drag falls back to the full
   *  recompute.
   */
  async fetchWeightJacobian(): Promise<void> {
    const m = this.deps.model;
    if (!m.embeddingsReady) return;
    const qs = this._projectionQueryString();
    if (this._weightJacobian && this._jacobianKey === qs) return;
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${qs}&jacobian=1`,
      );
      const jac = parseWeightJacobian(raw);
      if (jac) {
        this._weightJacobian = jac;
        this._jacobianKey = qs;
      }
    } catch {
      this._weightJacobian = null;
      this._jacobianKey = "";
    }
  }

  /** Approximate positions for a live slider drag; false when unavailable.
   *
   *  The Jacobian carries the base positions it was evaluated at, so repeated
   *  drag events each step from that same base rather than compounding.
   */
  applyApproximateWeights(weights: Record<string, number>): boolean {
    const m = this.deps.model;
    if (!this._weightJacobian || m.embeddingPositions.size === 0) return false;
    m.embeddingPositions = approximatePositions(this._weightJacobian, weights);
    this.deps.canvas.scheduleDraw();
    return true;
  }

  /** LRU bump + cap for _layoutCache. */
  private _layoutCachePut(qs: string, entry: CachedLayout): void {
    this._layoutCache.delete(qs);
    this._layoutCache.set(qs, entry);
    if (this._layoutCache.size > LAYOUT_CACHE_MAX) {
      const oldest = this._layoutCache.keys().next().value;
      if (oldest !== undefined) this._layoutCache.delete(oldest);
    }
  }

  /** Entry point: check status, then generate embeddings or fetch a layout. */
  async ensureAndProject(): Promise<void> {
    const m = this.deps.model;
    try {
      const status = await api<{
        total: number;
        fingerprinted: number;
        embedded: number;
        pending: number;
        model_ready: boolean;
        maest_model_ready?: boolean;
        maest_error?: string | null;
        tracks_with_maest?: number;
        tracks_pending_maest?: number;
        tracks_pending_rhythm?: number;
        features_extracted: number;
        features_pending: number;
        generating: boolean;
        embedding_version?: number;
        layout_revision?: string;
      }>(`/api/embeddings/status?${this._projectionQueryString()}`);

      // Only MAEST gates the map: it is the required source (see the backend's
      // REQUIRED_SOURCES). CLAP is an optional low-weight block, so a server
      // started with --no-clap must still render — blocking on it left the
      // canvas stuck on "loading models…" forever.
      const modelsBlocked = status.maest_model_ready === false;
      m.embeddingModelsLoading = modelsBlocked;
      if (modelsBlocked) {
        toast(status.maest_error || "MAEST model loading…", "ok");
        this._pollModelReady();
        this.deps.canvas.scheduleDraw();
        return;
      }

      m.embeddingModelsLoading = false;

      const maestPending = status.tracks_pending_maest ?? 0;
      const rhythmPending = status.tracks_pending_rhythm ?? 0;
      const needsWork = maestPending > 0 || rhythmPending > 0;

      if (needsWork && !status.generating) {
        const total = Math.max(maestPending, rhythmPending);
        const workParts: string[] = [];
        if (maestPending > 0) workParts.push(`${maestPending} MAEST`);
        if (rhythmPending > 0) workParts.push(`${rhythmPending} rhythm`);
        const detail =
          workParts.length > 1 ? ` (${workParts.join(", ")})` : "";
        toast(`Analysing ${total} tracks${detail}…`, "ok");
        m.embeddingsGenerating = true;
        this.deps.canvas.scheduleDraw();

        await postJSON("/api/embeddings/generate", {
          folder: "",
          recursive: true,
          priority_paths: [],
        });
        this._listenEmbeddingStream();
        return;
      }

      if (status.generating) {
        m.embeddingsGenerating = true;
        this.deps.canvas.scheduleDraw();
        this._listenEmbeddingStream();
        // Keep showing the last stable layout while embeddings regenerate; only
        // fetch a layout when we have no positions yet (first-time progressive PCA).
        if (m.viewMode === "embeddings" && m.embeddingPositions.size === 0) {
          void this._fetchProjection();
        }
        return;
      }

      if (m.viewMode === "embeddings") {
        await this._fetchProjection({ status });
      }
    } catch {
      /* api() already toasted */
    } finally {
      if (this.deps.model.viewMode !== "embeddings") {
        this.deps.model.embeddingModelsLoading = false;
      }
    }
  }

  private _pollModelReady(): void {
    clearTimeout(this._embedPollTimer);
    this._embedPollTimer = setTimeout(async () => {
      try {
        const status = await api<{
          model_ready: boolean;
          maest_model_ready?: boolean;
        }>("/api/embeddings/status?folder=&recursive=1&models_only=1");
        if (status.maest_model_ready !== false) {
          toast("Models ready", "ok");
          await this.ensureAndProject();
        } else {
          this._pollModelReady();
        }
      } catch {
        this._pollModelReady();
      }
    }, 3000) as unknown as number;
  }

  /** Open an SSE connection to stream embedding generation progress. */
  private _listenEmbeddingStream(): void {
    this._closeEmbeddingStream();
    this._embedBatchCount = 0;

    const es = new EventSource("/api/embeddings/stream");
    this._embedEventSource = es;

    es.addEventListener("decode_warning", ((e: MessageEvent) => {
      const data = JSON.parse(e.data) as { path: string; message: string };
      toast(`${data.path}: ${data.message}`, "warn");
    }) as EventListener);

    es.addEventListener("progress", ((e: MessageEvent) => {
      const data = JSON.parse(e.data) as {
        path: string;
        ok: boolean;
        done: number;
        total: number;
        failures?: string[];
      };
      const m = this.deps.model;
      m.embeddingProgress = {
        done: data.done,
        total: data.total,
        lastPath: data.path,
        lastOk: data.ok,
        lastFailures:
          data.failures && data.failures.length ? data.failures : undefined,
      };
      this.deps.status.update(m);
      this.deps.canvas.scheduleDraw();

      this._embedBatchCount++;
      const interval = Math.max(5, Math.floor(data.total / 10));
      if (
        data.done >= 2 &&
        (this._embedBatchCount >= interval || data.done === data.total) &&
        m.viewMode === "embeddings" &&
        m.embeddingPositions.size === 0 &&
        !this._projectionFetchPending
      ) {
        this._embedBatchCount = 0;
        void this._fetchProjectionIncremental();
      }
    }) as EventListener);

    es.addEventListener("done", () => {
      this._closeEmbeddingStream();
      const m = this.deps.model;
      m.embeddingsGenerating = false;
      m.embeddingProgress = null;
      this.invalidateLayoutCache();
      toast("Embeddings ready", "ok");
      this.deps.status.update(m);
      this.deps.canvas.scheduleDraw();
      if (m.viewMode === "embeddings") {
        void this._fetchProjection();
      }
    });

    es.addEventListener("error", () => {
      this._closeEmbeddingStream();
      const m = this.deps.model;
      m.embeddingsGenerating = false;
      m.embeddingProgress = null;
      this.deps.status.update(m);
      this.deps.canvas.scheduleDraw();
    });
  }

  private _closeEmbeddingStream(): void {
    if (this._embedEventSource) {
      this._embedEventSource.close();
      this._embedEventSource = null;
    }
  }

  private _projectionQueryString(methodOverride?: string): string {
    const method = methodOverride ?? "tsne";
    const base = `folder=&recursive=1&method=${method}`;
    // External roots are laid out only on request: the default map is the library.
    const ext = this.deps.model.showExternal ? "&include_external=1" : "";
    // Only append weights when they differ from the defaults: the default query
    // key must stay stable so existing cached layouts remain valid.
    const w = this.deps.model.clusterWeights;
    return _weightsAreDefault(w)
      ? `${base}${ext}`
      : `${base}&w=${_encodeWeights(w)}${ext}`;
  }

  /** Fetch PCA positions during ongoing generation, animating the transition.
   *
   *  Context (semantic weighting) is intentionally skipped: PCA is a quick
   *  preview and the CLAP model is busy with audio inference on the generation
   *  thread — concurrent text inference can fail on MPS / CUDA.
   */
  private async _fetchProjectionIncremental(): Promise<void> {
    this._projectionFetchPending = true;
    const m = this.deps.model;
    m.projectionPending = true;
    this.deps.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${this._projectionQueryString("pca")}`,
      );
      const { positions } = parseProjectionResponse(raw);
      if (!positions.length) return;

      const isFirst = m.embeddingPositions.size === 0;
      const newPositions = new Map<string, { x: number; y: number }>();
      for (const p of positions) {
        newPositions.set(p.path, { x: p.x, y: p.y });
      }
      m.embeddingsReady = true;

      if (isFirst && m.viewMode === "embeddings") {
        m.embeddingPositions = newPositions;
        this._animateToEmbeddings();
      } else {
        this._animatePositionUpdate(newPositions);
      }
    } catch (e) {
      console.warn("[trackspace] incremental PCA fetch failed:", e);
    } finally {
      this._projectionFetchPending = false;
      m.projectionPending = false;
      m.layoutProgress = null;
      this.deps.canvas.scheduleDraw();
    }
  }

  /** Smoothly lerp from current embedding positions to *newPositions*. */
  private _animatePositionUpdate(
    newPositions: Map<string, { x: number; y: number }>,
  ): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    const startPos = new Map<string, { x: number; y: number }>();
    for (const [path] of newPositions) {
      const cur =
        cv.animPositions?.get(path) ?? m.embeddingPositions.get(path);
      if (cur) startPos.set(path, { ...cur });
    }

    m.embeddingPositions = newPositions;

    if (startPos.size === 0) {
      cv.animPositions = null;
      cv.scheduleDraw();
      return;
    }

    cv.animPositions = new Map(startPos);
    const DURATION = 350;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const [path, start] of startPos) {
        const target = m.embeddingPositions.get(path);
        if (!target) continue;
        cv.animPositions!.set(path, {
          x: start.x + (target.x - start.x) * ease,
          y: start.y + (target.y - start.y) * ease,
        });
      }
      for (const [path, pos] of m.embeddingPositions) {
        if (!startPos.has(path)) cv.animPositions!.set(path, pos);
      }
      cv.scheduleDraw();
      if (t < 1) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        cv.animPositions = null;
        cv.scheduleDraw();
      }
    };
    this._animFrame = requestAnimationFrame(step);
  }

  /** Full t-SNE layout.  Checks the client-side layout cache
   * (keyed by query string + server revision) for instant revisits.
   * Falls back to ``/api/embeddings/projection`` on miss.  The server also
   * has a revision-keyed LRU so repeated requests are fast even without client
   * cache (e.g. after page reload). */
  private async _fetchProjection(opts?: {
    status?: EmbeddingLayoutStatusHint;
  }): Promise<void> {
    const m = this.deps.model;
    const qs = this._projectionQueryString();
    const reqSeq = ++this._projectionReqSeq;

    // --- Client-side cache check ---
    const cached = this._layoutCache.get(qs);
    if (cached && m.viewMode === "embeddings" && m.embeddingPositions.size > 0) {
      let rev = opts?.status?.layout_revision;
      if (rev === undefined) {
        if (m.embeddingsGenerating) {
          // Eligibility / revision can move while the batch fills in; confirm with server.
          try {
            const s = await api<EmbeddingLayoutStatusHint>(
              `/api/embeddings/status?${qs}`,
            );
            rev = s.layout_revision;
          } catch {
            /* status failed; fall through to full projection */
          }
        } else {
          // No generation running and cache was invalidated on SSE `done` or library reload:
          // `/status` would repeat the full server scan only to compare this hash — skip it.
          rev = cached.revision;
        }
      }
      if (rev != null && rev === cached.revision) {
        if (reqSeq === this._projectionReqSeq) this._applyPositions(cached.positions);
        return;
      }
    }

    // --- Full projection fetch ---
    m.projectionPending = true;
    this.deps.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${qs}`,
      );
      // A newer projection was requested while this one was in flight (a refit
      // takes seconds); its response must win, not this stale one.
      if (reqSeq !== this._projectionReqSeq) return;
      const { positions, revision } = parseProjectionResponse(raw);
      const newPositions = new Map<string, { x: number; y: number }>();
      for (const p of positions) {
        newPositions.set(p.path, { x: p.x, y: p.y });
      }
      if (revision) {
        this._layoutCachePut(qs, { revision, positions: newPositions });
      }
      m.embeddingsReady = true;
      this._applyPositions(newPositions);
    } catch {
      /* api() already toasted */
    } finally {
      if (reqSeq === this._projectionReqSeq) {
        m.projectionPending = false;
        m.layoutProgress = null;
        this.deps.canvas.scheduleDraw();
      }
    }
  }

  /** Apply a positions map to the model, animating the transition. */
  private _applyPositions(
    newPositions: Map<string, { x: number; y: number }>,
  ): void {
    const m = this.deps.model;
    if (m.viewMode === "embeddings") {
      if (m.embeddingPositions.size > 0) {
        this._animatePositionUpdate(newPositions);
      } else {
        m.embeddingPositions = newPositions;
        this._animateToEmbeddings();
      }
    } else {
      m.embeddingPositions = newPositions;
    }
  }

  private _animateToEmbeddings(): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    // Capture current screen positions as starting points
    const startPos = new Map<string, { x: number; y: number }>();
    for (const t of m.tracks) {
      if (m.viewMode === "embeddings" && m.embeddingPositions.has(t.path)) {
        const existing = cv.animPositions?.get(t.path);
        if (existing) {
          startPos.set(t.path, { ...existing });
        } else {
          const wx = m.axisX ? m.axisScalar(t, m.axisX) : 0.5;
          const wy = m.axisY ? m.axisScalar(t, m.axisY) : 0.5;
          startPos.set(t.path, { x: wx, y: wy });
        }
      }
    }

    cv.animPositions = startPos;
    const DURATION = 500;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const [path, start] of startPos) {
        const target = m.embeddingPositions.get(path);
        if (!target) continue;
        cv.animPositions!.set(path, {
          x: start.x + (target.x - start.x) * ease,
          y: start.y + (target.y - start.y) * ease,
        });
      }
      cv.scheduleDraw();
      if (t < 1) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        cv.animPositions = null;
        cv.scheduleDraw();
      }
    };
    this._animFrame = requestAnimationFrame(step);
  }

  private _animateToTags(): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    // Capture current embedding positions as starting points
    const startPos = new Map<string, { x: number; y: number }>();
    for (const t of m.tracks) {
      const ep = m.embeddingPositions.get(t.path);
      const existing = cv.animPositions?.get(t.path);
      if (existing) {
        startPos.set(t.path, { ...existing });
      } else if (ep) {
        startPos.set(t.path, { x: ep.x, y: ep.y });
      }
    }

    cv.animPositions = startPos;
    const DURATION = 500;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const track of m.tracks) {
        const start = startPos.get(track.path);
        if (!start) continue;
        const tx = m.axisX ? m.axisScalar(track, m.axisX) : 0.5;
        const ty = m.axisY ? m.axisScalar(track, m.axisY) : 0.5;
        cv.animPositions!.set(track.path, {
          x: start.x + (tx - start.x) * ease,
          y: start.y + (ty - start.y) * ease,
        });
      }
      cv.scheduleDraw();
      if (t < 1) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        cv.animPositions = null;
        cv.scheduleDraw();
      }
    };
    this._animFrame = requestAnimationFrame(step);
  }

  /**
   * Animate tag-space dots from their current positions to the positions they
   * will occupy after changing an axis.  Must only be called in "tags" mode.
   *
   * Lives here (not in the controller) because it drives the shared
   * `canvas.animPositions` overlay and must cancel any embedding animation
   * already in flight.
   */
  animateAxisChange(which: "axisX" | "axisY", tag: string): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    // Snapshot where every track currently sits (mid-animation positions take
    // priority, otherwise use current tag values with the old axes).
    const startPos = new Map<string, { x: number; y: number }>();
    for (const t of m.tracks) {
      const existing = cv.animPositions?.get(t.path);
      if (existing) {
        startPos.set(t.path, { ...existing });
      } else {
        startPos.set(t.path, {
          x: m.axisX ? m.axisScalar(t, m.axisX) : 0.5,
          y: m.axisY ? m.axisScalar(t, m.axisY) : 0.5,
        });
      }
    }

    // Apply the axis change (emits "change" → scheduleDraw).
    m.toggleAxis(which, tag);

    // Override the canvas with our start snapshot so the change-event draw
    // shows the old positions — the animation will take it from here.
    cv.animPositions = startPos;

    const DURATION = 500;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const [path, start] of startPos) {
        const track = m.trackByPath(path);
        if (!track) continue;
        // Target is the new tag position (axes are already updated).
        const tx = m.axisX ? m.axisScalar(track, m.axisX) : 0.5;
        const ty = m.axisY ? m.axisScalar(track, m.axisY) : 0.5;
        cv.animPositions!.set(path, {
          x: start.x + (tx - start.x) * ease,
          y: start.y + (ty - start.y) * ease,
        });
      }
      cv.scheduleDraw();
      if (t < 1) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        cv.animPositions = null;
        cv.scheduleDraw();
      }
    };
    this._animFrame = requestAnimationFrame(step);
  }

  async toggleViewMode(): Promise<void> {
    const m = this.deps.model;
    // Persist the current viewport for the mode we're leaving.
    this._savedVpByMode.set(m.viewMode, { ...m.vp });

    if (m.viewMode === "tags") {
      m.setViewMode("embeddings");
      m.embeddingModelsLoading = false;
      // Restore a previously saved embedding viewport, or default to full [0,1] view.
      const saved = this._savedVpByMode.get("embeddings");
      if (saved) {
        m.vp.ox = saved.ox;
        m.vp.oy = saved.oy;
        m.vp.zoom = saved.zoom;
      } else {
        m.vp.ox = 0;
        m.vp.oy = 0;
        m.vp.zoom = 1;
      }
      if (m.embeddingsReady) {
        this._animateToEmbeddings();
      } else {
        await this.ensureAndProject();
      }
    } else {
      m.setViewMode("tags");
      m.embeddingModelsLoading = false;
      // Restore a previously saved tag viewport, or leave as-is (first time back).
      const saved = this._savedVpByMode.get("tags");
      if (saved) {
        m.vp.ox = saved.ox;
        m.vp.oy = saved.oy;
        m.vp.zoom = saved.zoom;
      }
      this._animateToTags();
    }
    m.saveLS();
    this.deps.status.update(m);
    const $vm = document.getElementById(
      "view-mode-toggle",
    ) as HTMLInputElement | null;
    if ($vm) $vm.checked = m.viewMode === "embeddings";
  }

  /** Switch to a specific view mode (no-op if already there). */
  async switchToMode(target: "tags" | "embeddings"): Promise<void> {
    if (this.deps.model.viewMode === target) return;
    await this.toggleViewMode();
  }
}
