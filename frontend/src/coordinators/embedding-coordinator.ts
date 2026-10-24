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
// One camera-only exception to the one-frame rule: the 2-D↔3-D blend tween
// (`_blendFrame`) writes `model.view3dBlend` — a camera parameter, never a dot
// position — so it runs on its own frame and can animate alongside a position
// lerp (a toggle starts them together on the fetched layout) without either
// cancelling the other.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type Billboard, type MapPos, type Model, type Scope, type Track, DEFAULT_CLUSTER_WEIGHTS, formatSourceWeight, scopeContains, scopeHalfExtents } from "../model";
import { type Lens, getLens, lensWaitSources, lensWeights } from "../lib/lenses";
import { toast } from "../lib/toast";
import { api, postJSON } from "../lib/api";
import { clamp } from "../lib/util";
import { SseSubscription, latestSeq } from "../lib/async";
import type { ViewModifierSink } from "./view-modifier-coordinator";
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

/**
 * Length of the 2-D↔3-D toggle. The camera blend tween and the dot-position
 * lerp run together at this length, so the switch reads as one gesture.
 */
const TOGGLE_MS = 550;

/** One per-source coverage row from ``GET /api/embeddings/status``. */
interface CoverageSource {
  id: string;
  label: string;
  done: number;
  pending: number;
  failed: number;
}

/**
 * Lens source name → coverage source id. The MAEST pair is one cache source
 * (``maest``), so a lens waiting on both must not count it twice.
 */
const COVERAGE_SOURCE: Record<string, string> = {
  maest_logits: "maest",
  maest: "maest",
  clap: "clap",
  rhythm: "rhythm",
  features: "features",
  supervised: "supervised",
};

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
  positions: Map<string, MapPos>;
}

/** A single row of ``/api/embeddings/projection``'s ``positions`` array. */
type ProjectionPos = { path: string; x: number; y: number; z?: number };

/** ``/api/embeddings/projection`` returns ``{ positions, revision }`` (legacy bare array unsupported). */
function parseProjectionResponse(raw: unknown): {
  positions: ProjectionPos[];
  revision: string;
} {
  if (
    raw &&
    typeof raw === "object" &&
    "positions" in (raw as object) &&
    Array.isArray((raw as { positions: unknown }).positions)
  ) {
    const o = raw as { positions: ProjectionPos[]; revision?: string };
    return { positions: o.positions, revision: o.revision ?? "" };
  }
  return { positions: [], revision: "" };
}

/** Build a path → position map from a response's ``positions``, keeping ``z`` when sent. */
function toPositionMap(positions: ProjectionPos[]): Map<string, MapPos> {
  const out = new Map<string, MapPos>();
  for (const p of positions) {
    out.set(
      p.path,
      p.z === undefined ? { x: p.x, y: p.y } : { x: p.x, y: p.y, z: p.z },
    );
  }
  return out;
}

/** The frame a position lives in: null for the cube, else its billboard anchor. */
function frameOf(p: MapPos): Billboard | null {
  return p.billboard ?? null;
}

/** True when *a* and *b* draw in the same frame (same billboard anchor, or neither). */
function sameFrame(a: MapPos, b: MapPos): boolean {
  const fa = frameOf(a);
  const fb = frameOf(b);
  if (!fa || !fb) return fa === fb;
  return fa.cx === fb.cx && fa.cy === fb.cy && fa.cz === fb.cz;
}

/** A position without its own in-flight blend fields (the pure frame to blend from). */
function stripped(p: MapPos): MapPos {
  const out: MapPos = { x: p.x, y: p.y };
  if (p.z !== undefined) out.z = p.z;
  if (p.billboard) out.billboard = p.billboard;
  return out;
}

/**
 * Lerp two positions on the shared animation frame. ``z`` (the 3-D orbit axis)
 * is interpolated only when either side has one; a missing side reads as the
 * cube centre (0.5), so a 2-D↔3-D toggle tweens depth in while x/y slide.
 *
 * When the two positions live in DIFFERENT frames — one a cube position and one
 * a sphere-scope member's flat ``billboard`` layout, or two billboards with
 * different anchors — no single world position exists: the dot must move in
 * VIEW space. The result then carries the target frame ``b`` plus ``from`` (the
 * start position, stripped of any animation fields) and ``mix = 1 - e``; the
 * canvas blends the two frames' projections linearly, so at e = 1 the dot is
 * exactly ``b``'s frame and at e = 0 exactly ``a``'s. Same-frame positions keep
 * today's x/y (and z) lerp, and share the billboard anchor through.
 */
export function lerpPos(a: MapPos, b: MapPos, e: number): MapPos {
  if (!sameFrame(a, b)) {
    return {
      x: b.x,
      y: b.y,
      z: b.z,
      billboard: b.billboard,
      from: stripped(a),
      mix: 1 - e,
    };
  }
  const x = a.x + (b.x - a.x) * e;
  const y = a.y + (b.y - a.y) * e;
  if (a.billboard) return { x, y, billboard: a.billboard };
  if (a.z === undefined && b.z === undefined) return { x, y };
  return { x, y, z: (a.z ?? 0.5) + ((b.z ?? 0.5) - (a.z ?? 0.5)) * e };
}

export interface EmbeddingDeps {
  model: Model;
  /** Reflect view-mode / projection changes in the status bar. */
  status: { update(m: Model): void };
  /** The active-view-modifier stack: an open scope is one chip. */
  viewMods: ViewModifierSink;
  /** The canvas overlay the position animations drive (mutable, not just a repaint). */
  canvas: {
    animPositions: Map<string, MapPos> | null;
    scheduleDraw(): void;
    /** Current resolved position (scope-aware), or null when hidden. */
    worldPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null;
    /** Position ignoring any scope, for membership and open/close anchors. */
    outerPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null;
  };
}

/** Smallest scope half-extent, in world units. */
const SCOPE_MIN_R = 0.02;
/** Largest scope half-extent, in world units (roughly the whole unit square). */
const SCOPE_MAX_R = 0.75;

export class EmbeddingCoordinator {
  private _embedPollTimer = 0;
  private _embedStream = new SseSubscription();
  private _embedBatchCount = 0;
  private _projectionFetchPending = false;
  /** Query-string → {revision, positions} map so A→B→A revisits are instant. */
  private _layoutCache = new Map<string, CachedLayout>();

  /** Source-weight Jacobian at the last fetched weights, for live drags. */
  private _weightJacobian: WeightJacobian | null = null;
  /** Query string the cached Jacobian was evaluated at (empty = none). Also
   *  gates application: the key carries ``dims=3``, so a Jacobian fetched for
   *  the other dimensionality is never applied to the current view. */
  private _jacobianKey = "";

  private _animFrame = 0;
  /**
   * Own rAF handle for the 3-D camera-blend tween. Deliberately NOT the shared
   * `_animFrame`: it writes only `view3dBlend` (a camera parameter, never a dot
   * position), so it must be able to run alongside a position lerp — a 2-D↔3-D
   * toggle starts the blend tween and the position tween together on the new
   * layout, on their own frames — without either cancelling the other.
   */
  private _blendFrame = 0;
  /**
   * Camera-blend target (0 or 1) of an in-flight 2-D↔3-D toggle whose layout
   * has not applied yet. `_applyPositions` consumes it and starts the blend and
   * position tweens together; a re-toggle simply overwrites it, dropping the
   * old pending target. Null while no toggle is waiting on a layout.
   */
  private _toggleTarget: 0 | 1 | null = null;
  /** Monotonic request id; a slow projection must not overwrite a newer one. */
  private _projectionSeq = latestSeq();
  /** Monotonic request id for scope recomputes (lens switch / resize). */
  private _scopeSeq = latestSeq();
  /** Id of the active `scope` modifier chip, or null when no scope is open. */
  private _scopeModifierId: string | null = null;

  constructor(private deps: EmbeddingDeps) {}

  /** Eval-gallery compute gate: no projection/lens/scope/weight work while active. */
  private get _evalInert(): boolean {
    return this.deps.model.evalMode;
  }

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
    cancelAnimationFrame(this._blendFrame);
    this.cancelAnimation();
  }

  /** Drop all cached layouts (e.g. after a full library reload). */
  invalidateLayoutCache(): void {
    this._layoutCache.clear();
  }

  /**
   * Apply a precomputed eval-artifact position map (the layout-evaluation
   * gallery), animating from the current positions on the one shared animation
   * frame. Intentionally independent of lens/weights: the map is read-only and
   * never triggers a projection fetch. Switching maps animates the transition the
   * same way a lens switch does, so selection/colour/filters stay stable.
   */
  applyPrecomputedPositions(positions: Map<string, { x: number; y: number }>): void {
    const m = this.deps.model;
    m.embeddingsReady = true;
    // `_animatePositionUpdate` snapshots the current positions first, then sets
    // the new map and lerps between them on the shared frame.
    this._animatePositionUpdate(positions);
  }

  /** Refetch the layout after a weight change (the query string carries them).
   *
   *  Safe to call from any view mode: when the map has never been drawn there is
   *  nothing to refresh (the next entry fetches with the new weights), and when it
   *  has, the new positions are stored so switching to the map animates to them.
   */
  refreshProjection(): Promise<void> {
    if (this._evalInert) return Promise.resolve();
    const m = this.deps.model;
    if (!m.embeddingsReady || m.lens().kind !== "projection")
      return Promise.resolve();
    return this._fetchProjection();
  }

  /**
   * Toggle the 3-D orbit view. A projection lens is refetched so the positions
   * carry z; tags/deterministic lenses stay 2-D and the flag just flips without
   * a fetch. Layouts are cached by query string (with vs. without `dims=3`), so
   * a back-and-forth toggle is instant. The camera blend tween and the position
   * lerp start together, at the same length, once the toggled layout is in hand
   * (`_applyPositions` consumes `_toggleTarget`): entering fades from head-on
   * orthographic out to the orbit angle while the dots gain depth, leaving
   * reverses both. If the fetch never applies (failure, superseded, nothing to
   * place), `finally` still runs the camera tween alone so the blend completes.
   *
   * An open scope survives the toggle: its shape converts with the view (circle
   * → sphere entering, sphere → circle leaving) and it re-lays out over the NEW
   * outer layout once the refetch has landed — membership is decided in the
   * dimensionality being entered. The scope chip stays put.
   */
  async setView3D(on: boolean): Promise<void> {
    const m = this.deps.model;
    if (this._evalInert || m.view3d.on === on) return;
    m.view3d.on = on;
    m.saveLS();
    this.deps.status.update(m);
    this.deps.canvas.scheduleDraw();
    if (m.lens().kind === "projection") {
      // Hold the blend target until the toggled layout is in hand: the camera
      // tween and the position lerp then start together (see `_applyPositions`),
      // so the switch reads as one gesture instead of rotate-then-move. A
      // re-toggle just overwrites the target; a fetch that never applies
      // (failure, superseded, nothing to place) still completes the camera in
      // `finally` below.
      this._toggleTarget = on ? 1 : 0;
      try {
        if (m.embeddingPositions.size === 0) await this.ensureAndProject();
        else await this._fetchProjection();
      } finally {
        const target = this._toggleTarget;
        this._toggleTarget = null;
        if (target !== null) this._tweenBlend(target);
      }
    } else {
      // Tags/deterministic lenses never render 3-D: flip the blend instantly.
      this._tweenBlend(on ? 1 : 0);
    }
    // Convert and re-lay out an open scope over the new outer positions. Only a
    // projection lens ever renders 3-D, so `is3d` (not the bare flag) picks the
    // shape: on a tags/deterministic lens the toggle is a no-op and the scope
    // keeps its 2-D shape.
    if (m.scope) {
      const next = this._convertScopeShape(m.scope);
      if (next !== m.scope) {
        const start = new Map(m.scopePositions);
        m.scope = next;
        const targets = await this._relayoutScope(next);
        if (targets) this._animateOverlay(start, targets);
      }
    }
  }

  /**
   * Tween the 3-D camera blend (`view3dBlend`) to *target* (0 = head-on
   * orthographic, 1 = full orbit + perspective) on its OWN frame (`_blendFrame`),
   * so a re-toggle cancels the in-flight tween and continues from wherever the
   * blend has reached. A 2-D↔3-D toggle starts it from `_applyPositions`
   * alongside the position lerp at the same length, so the camera and the dots
   * arrive together. On a non-projection lens nothing ever renders 3-D, so the
   * blend is set instantly instead of animating a no-op.
   */
  private _tweenBlend(target: number, ms = TOGGLE_MS): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    if (m.lens().kind !== "projection") {
      m.view3dBlend = target;
      return;
    }
    const from = m.view3dBlend;
    if (from === target) return;
    cancelAnimationFrame(this._blendFrame);
    const t0 = performance.now();
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / ms);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      m.view3dBlend = from + (target - from) * e;
      cv.scheduleDraw();
      if (t < 1) {
        this._blendFrame = requestAnimationFrame(step);
      } else {
        m.view3dBlend = target;
        cv.scheduleDraw();
      }
    };
    this._blendFrame = requestAnimationFrame(step);
  }

  /** Fetch the source-weight Jacobian at the current weights, once per weights.
   *
   *  The server evaluates it at the weights it is queried with, so the base
   *  vector must match the one the drag starts from. It costs one projection per
   *  source, so the result is cached by the query string — 2-D and 3-D payloads
   *  cache separately, as the key carries ``dims=3``: reopening the panel or
   *  dragging back to the same weights must not recompute it. A failed fetch
   *  leaves the approximation unavailable and the drag falls back to the full
   *  recompute.
   */
  async fetchWeightJacobian(): Promise<void> {
    if (this._evalInert) return;
    const m = this.deps.model;
    if (!m.embeddingsReady || m.lens().kind !== "projection") return;
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
   *  drag events each step from that same base rather than compounding. It is
   *  only applied when its query key matches the current projection query, so a
   *  Jacobian fetched for the other dimensionality (2-D vs 3-D) is never applied
   *  to the wrong view.
   */
  applyApproximateWeights(weights: Record<string, number>): boolean {
    if (this._evalInert) return false;
    const m = this.deps.model;
    if (
      !this._weightJacobian ||
      // A Jacobian fetched for the other dimensionality (2-D vs 3-D) must never
      // be applied: the key carries ``dims=3``, so it only matches the current
      // projection query.
      this._jacobianKey !== this._projectionQueryString() ||
      m.embeddingPositions.size === 0
    )
      return false;
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
    if (this._evalInert) return;
    const lens = m.lens();
    if (lens.kind === "tags") return;
    if (lens.kind === "deterministic") {
      this._applyDeterministic(lens);
      return;
    }
    try {
      const status = await api<{
        model_ready: boolean;
        maest_model_ready?: boolean;
        maest_error?: string | null;
        generating: boolean;
        embedding_version?: number;
        layout_revision?: string;
        sources?: CoverageSource[];
      }>(`/api/embeddings/status?${this._projectionQueryString()}`);

      // Only a lens that needs MAEST waits on the MAEST model: a CLAP/rhythm
      // lens must still render on a server where MAEST is loading or disabled —
      // blocking on it left the canvas stuck on "loading models…" forever.
      const waitSources = lensWaitSources(lens);
      const needsMaest = waitSources.some(
        (s) => s === "maest" || s === "maest_logits",
      );
      const modelsBlocked = needsMaest && status.maest_model_ready === false;
      m.embeddingModelsLoading = modelsBlocked;
      if (modelsBlocked) {
        toast(status.maest_error || "MAEST model loading…", "ok");
        this._pollModelReady();
        this.deps.canvas.scheduleDraw();
        return;
      }

      m.embeddingModelsLoading = false;

      if (
        lens.sources?.includes("supervised") &&
        (this._coverageSource(status, "supervised")?.done ?? 0) === 0
      ) {
        toast("Train the folder head to use this lens", "warn");
      }

      // One pending count per coverage source, deduped across lens sources that
      // share one (maest_logits and maest are the same stage).
      const pending = new Map<string, { pending: number; label: string }>();
      for (const s of waitSources) {
        const row = this._coverageSource(status, s);
        if (!row) continue;
        const cur = pending.get(row.id);
        pending.set(row.id, {
          pending: (cur?.pending ?? 0) + row.pending,
          label: row.label,
        });
      }
      const needsWork = [...pending.values()].some((v) => v.pending > 0);

      if (needsWork && !status.generating) {
        const total = Math.max(...[...pending.values()].map((v) => v.pending));
        const workParts = [...pending.values()]
          .filter((v) => v.pending > 0)
          .map((v) => `${v.pending} ${v.label}`);
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
        if (m.embeddingPositions.size === 0) {
          void this._fetchProjection();
        }
        return;
      }

      await this._fetchProjection({ status });
    } catch {
      /* api() already toasted */
    } finally {
      if (this.deps.model.viewMode !== "embeddings") {
        this.deps.model.embeddingModelsLoading = false;
      }
    }
  }

  /** Deterministic lens: positions are a pure function of cached features. */
  private _applyDeterministic(lens: Lens): void {
    const m = this.deps.model;
    const positions = new Map<string, { x: number; y: number }>();
    if (lens.compute) {
      for (const t of m.allTracks) {
        const p = lens.compute(t);
        if (p) positions.set(t.path, p);
      }
    }
    m.embeddingsReady = true;
    this._applyPositions(positions);
  }

  /** The coverage row for a lens source, or undefined when it has none. */
  private _coverageSource(
    status: { sources?: CoverageSource[] },
    source: string,
  ): CoverageSource | undefined {
    const id = COVERAGE_SOURCE[source];
    return status.sources?.find((s) => s.id === id);
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
    this._embedBatchCount = 0;

    const es = this._embedStream.open("/api/embeddings/stream");

    es.addEventListener("decode_warning", ((e: MessageEvent) => {
      const data = JSON.parse(e.data) as { path: string; message: string };
      toast(`${data.path}: ${data.message}`, "warn");
    }) as EventListener);

    es.addEventListener("progress", ((e: MessageEvent) => {
      const data = JSON.parse(e.data) as { done: number; total: number };
      const m = this.deps.model;
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
      this.deps.status.update(m);
      this.deps.canvas.scheduleDraw();
    });
  }

  private _closeEmbeddingStream(): void {
    this._embedStream.close();
  }

  private _projectionQueryString(
    methodOverride?: string,
    lensOverride?: Lens,
    dimsOverride?: 2 | 3,
  ): string {
    const method = methodOverride ?? "tsne";
    const m = this.deps.model;
    const lens = lensOverride ?? m.lens();
    const base = `folder=&recursive=1&method=${method}`;
    // External roots' positions ride along with every chart so toggling their
    // visibility is a redraw, not a refetch: they are Nyström-placed onto the
    // library chart, which never changes its fit. `fit_scope=all` is the only
    // option that refits with them, so only it may move the library.
    const ext = m.externalRootIds.size > 0 ? "&include_external=1" : "";
    const fit =
      m.showExternal && m.externalInProjection ? "&fit_scope=all" : "";
    // A per-source lens projects only its own blocks; the combined lens omits the
    // param so its default query key stays stable.
    const src = lens.sources ? `&sources=${lens.sources.join(",")}` : "";
    // Only append weights when they differ from the defaults: the default query
    // key must stay stable so existing cached layouts remain valid.
    const w = lensWeights(lens, m.clusterWeights);
    const wq = _weightsAreDefault(w) ? "" : `&w=${_encodeWeights(w)}`;
    // 3-D is a client-side toggle with a server z axis, so it gets its own
    // query key: 2-D and 3-D layouts cache separately. A scope lens layout is
    // always 2-D — a sphere's contents are a flat map, and the 2-D key is the
    // one a plain 2-D request uses — so `_fetchLensPositions` overrides the
    // dimensionality; everything else follows the view.
    const dim = dimsOverride ?? (m.is3d ? 3 : 2);
    const dims = dim === 3 ? "&dims=3" : "";
    return `${base}${src}${wq}${ext}${fit}${dims}`;
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
    const reqSeq = this._projectionSeq.next();
    const qs = this._projectionQueryString("pca");
    m.projectionPending = true;
    m.pendingProjectionQuery = qs;
    m.projectionStartedAt = Date.now();
    m.layoutProgress = null;
    this.deps.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${qs}`,
      );
      // A newer projection (a lens switch, or the full layout after the batch)
      // may have started; its response must win, not this stale PCA preview.
      if (!this._projectionSeq.isCurrent(reqSeq)) return;
      const { positions } = parseProjectionResponse(raw);
      if (!positions.length) return;

      const isFirst = m.embeddingPositions.size === 0;
      const newPositions = toPositionMap(positions);
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
      this._settleProjection(reqSeq);
    }
  }

  /**
   * Smoothly lerp from current embedding positions to *newPositions*. A 2-D↔3-D
   * toggle passes TOGGLE_MS so the dots arrive in step with the camera blend
   * tween (see `_applyPositions`); every other transition keeps 260 ms.
   */
  private _animatePositionUpdate(
    newPositions: Map<string, MapPos>,
    ms = 260,
  ): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    const startPos = new Map<string, MapPos>();
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
    const DURATION = ms;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const [path, start] of startPos) {
        const target = m.embeddingPositions.get(path);
        if (!target) continue;
        cv.animPositions!.set(path, lerpPos(start, target, ease));
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
    const reqSeq = this._projectionSeq.next();

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
        if (this._projectionSeq.isCurrent(reqSeq)) {
          this._applyPositions(cached.positions);
          this._settleProjection(reqSeq);
        }
        return;
      }
    }

    // --- Full projection fetch ---
    m.projectionPending = true;
    m.pendingProjectionQuery = qs;
    m.projectionStartedAt = Date.now();
    m.layoutProgress = null;
    this.deps.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${qs}`,
      );
      // A newer projection was requested while this one was in flight (a refit
      // takes seconds); its response must win, not this stale one.
      if (!this._projectionSeq.isCurrent(reqSeq)) return;
      const { positions, revision } = parseProjectionResponse(raw);
      const newPositions = toPositionMap(positions);
      if (revision) {
        this._layoutCachePut(qs, { revision, positions: newPositions });
      }
      m.embeddingsReady = true;
      this._applyPositions(newPositions);
    } catch {
      /* api() already toasted */
    } finally {
      this._settleProjection(reqSeq);
    }
  }

  /** Clear the "Loading <lens>" pill once the newest projection request settles.
   *
   *  Ownership is by sequence, not by which request happens to finish: a cache hit
   *  can apply the final positions while an older, slower fit is still in flight,
   *  and that stale fit must not leave the pill up (nor take it down twice). */
  private _settleProjection(reqSeq: number): void {
    if (!this._projectionSeq.isCurrent(reqSeq)) return;
    const m = this.deps.model;
    m.projectionPending = false;
    m.pendingProjectionQuery = null;
    m.projectionStartedAt = null;
    m.layoutProgress = null;
    this.deps.canvas.scheduleDraw();
  }

  /**
   * Apply a positions map to the model, animating the transition. Every layout
   * application funnels through here (fetches and client-cache hits alike), so a
   * pending 2-D↔3-D toggle target is consumed here: its camera blend tween and
   * the position lerp start together at TOGGLE_MS — one gesture — instead of the
   * camera rotating ahead of a slow refetch. A re-toggle overwrote the target,
   * so the newest toggle always wins.
   */
  private _applyPositions(newPositions: Map<string, MapPos>): void {
    const m = this.deps.model;
    const target = this._toggleTarget;
    this._toggleTarget = null;
    if (m.viewMode === "embeddings") {
      if (m.embeddingPositions.size > 0) {
        this._animatePositionUpdate(
          newPositions,
          target !== null ? TOGGLE_MS : undefined,
        );
      } else {
        m.embeddingPositions = newPositions;
        this._animateToEmbeddings();
      }
    } else {
      m.embeddingPositions = newPositions;
    }
    if (target !== null) this._tweenBlend(target, TOGGLE_MS);
  }

  private _animateToEmbeddings(): void {
    const cv = this.deps.canvas;
    const m = this.deps.model;

    // Capture current screen positions as starting points
    const startPos = new Map<string, MapPos>();
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
    const DURATION = 340;
    const t0 = performance.now();

    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const elapsed = performance.now() - t0;
      const t = Math.min(1, elapsed / DURATION);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      for (const [path, start] of startPos) {
        const target = m.embeddingPositions.get(path);
        if (!target) continue;
        cv.animPositions!.set(path, lerpPos(start, target, ease));
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
    const DURATION = 340;
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

    const DURATION = 340;
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

  /** Advance the lens cycle; *dir* is +1 for **E**, -1 for **Shift+E**. */
  async cycleLens(dir: 1 | -1 = 1): Promise<void> {
    if (this._evalInert) return;
    const m = this.deps.model;
    await this.switchToLens(dir === 1 ? m.nextLens() : m.prevLens());
  }

  /**
   * Switch to a lens (built-in id or preset id), animating the transition.
   *
   * The camera stays where it is across the switch. Tag-space ↔ map transitions
   * animate through the shared `animPositions` overlay; map → map and feature →
   * feature animate between their stored positions.
   */
  async switchToLens(target: string): Promise<void> {
    const m = this.deps.model;
    if (this._evalInert) return;
    if (m.lensId === target) return;
    // A scope re-lays out its tracks over the *outer* lens; switching that lens
    // would leave the scope's positions describing a space that is no longer shown.
    if (m.scope) {
      m.scope = null;
      m.scopePositions = new Map();
      this.deps.viewMods.remove(this._scopeModifierId);
      this._scopeModifierId = null;
    }
    m.lensId = target;
    // A preset carries the display options it was saved with; applying them here
    // keeps the select and the weights panel consistent with the canvas.
    const preset = m.lensPresets.find((p) => p.id === target);
    if (preset) m.applyLensPreset(preset);
    const lens = m.lens();
    m.embeddingModelsLoading = false;

    // Feedback first: the persistent tabber and status must land before the
    // (async) layout, so a slow fit never makes the switch feel unresponsive.
    m.saveLS();
    this.deps.status.update(m);
    this.syncLensChrome();

    if (lens.kind === "tags") {
      this._animateToTags();
    } else if (lens.kind === "deterministic") {
      this._applyDeterministic(lens);
    } else if (m.embeddingPositions.size === 0) {
      // First entry into map space: fetch the layout (may trigger analysis).
      await this.ensureAndProject();
    } else {
      // Map → map: animate from the current positions to the new lens's layout.
      await this._fetchProjection();
    }
  }

  /** Switch to a specific view mode (tags ↔ the last map lens). */
  async switchToMode(target: "tags" | "embeddings"): Promise<void> {
    if (this._evalInert) return;
    if (this.deps.model.viewMode === target) return;
    await this.switchToLens(target === "tags" ? "tags" : "combined");
  }

  /** Sync the source-weight panel visibility to the active lens. */
  syncLensChrome(): void {
    // Source weights only re-weight the combined lens; a per-source lens fixes
    // them, so hide the sliders where they would do nothing.
    const lens = this.deps.model.lens();
    const details = document.getElementById(
      "cluster-weights-details",
    ) as HTMLDetailsElement | null;
    if (details) details.hidden = lens.kind !== "projection" || !!lens.sources;
  }

  /**
   * Open a scope: re-lay out the tracks inside its region by ``lensId``.
   *
   * Membership is decided from the tracks' outer positions (where the user drew
   * the region), then frozen into ``scopePositions`` so the re-layout cannot move
   * a track across the boundary. Tracks the scope lens cannot place are hidden
   * (the canvas hides in-region dots that have no scope position).
   */
  async openScope(
    scope: Scope,
    lensId: string,
    allowEmpty = false,
  ): Promise<void> {
    if (this._evalInert) return;
    const m = this.deps.model;
    const active = { ...scope, lensId };
    const seq = this._scopeSeq.next();
    // Show the circle immediately so X + scroll can resize it while the layout
    // loads; the no-tracks case reverts below.
    m.scopeLensId = lensId;
    m.scope = active;
    m.scopePositions = new Map();
    this.deps.canvas.scheduleDraw();

    const { raw, inScope } = await this._computeScopeRaw(active);
    if (!this._scopeSeq.isCurrent(seq)) return;
    if (inScope.length === 0 && !allowEmpty) {
      m.scope = null;
      toast("No tracks inside the scope", "warn");
      return;
    }
    if (raw.size === 0 && !allowEmpty) {
      m.scope = null;
      toast("That lens has no positions for these tracks", "warn");
      return;
    }
    const targets = this._mapIntoScope(raw, active);
    const start = this._startFromOuter(targets.keys());
    m.scopePositions = targets;
    // The open scope is an active view modifier: one chip naming its lens.
    if (this._scopeModifierId) {
      this.deps.viewMods.update(this._scopeModifierId, {
        label: getLens(lensId).label,
      });
    } else {
      this._scopeModifierId = this.deps.viewMods.push(
        { kind: "scope", label: getLens(lensId).label },
        () => {
          this._scopeModifierId = null;
          this._closeScopeState();
        },
      );
    }
    if (start.size > 0) this._animateOverlay(start, targets);
    else this.deps.canvas.scheduleDraw();
  }

  /**
   * Re-lay the open scope with a different lens, morphing between them.
   *
   * Membership is unchanged (the circle did not move), so only the lens positions
   * are recomputed; a lens the tracks have no data for leaves them at their outer
   * positions.
   */
  async setScopeLens(lensId: string): Promise<void> {
    if (this._evalInert) return;
    const m = this.deps.model;
    const scope = m.scope;
    if (!scope || scope.lensId === lensId) return;
    m.scopeLensId = lensId;
    const next = { ...scope, lensId };
    const seq = this._scopeSeq.next();
    const { raw } = await this._computeScopeRaw(next);
    if (!this._scopeSeq.isCurrent(seq)) return;
    const targets = this._mapIntoScope(raw, next);
    const start = new Map(m.scopePositions);
    m.scope = next;
    m.scopePositions = targets;
    // The chip names the scope's lens, so switching it refreshes the chip.
    if (this._scopeModifierId) {
      this.deps.viewMods.update(this._scopeModifierId, {
        label: getLens(lensId).label,
      });
    }
    this._animateOverlay(start, targets);
  }

  /**
   * Scale the open scope (X + scroll). Membership can change with the region, so
   * the in-scope set and the mapped positions are recomputed; the region updates
   * synchronously (so a rapid scroll accumulates and the shape moves at once),
   * while the positions follow in the same tick.
   */
  resizeScope(factor: number): Promise<void> {
    if (this._evalInert) return Promise.resolve();
    const m = this.deps.model;
    const scope = m.scope;
    if (!scope) return Promise.resolve();
    const next = this._scaleScope(scope, factor);
    m.scope = next;
    // No animation here: a scroll-resize accumulates and the region updates
    // synchronously, so the dots follow in the same tick.
    return this._relayoutScope(next).then(() => undefined);
  }

  /**
   * Recompute the open scope's positions over the current outer layout and
   * apply them (no animation — the caller animates when a tween is wanted).
   * Guarded by `_scopeSeq`: a newer scope op (resize, lens switch, 3-D toggle)
   * supersedes an in-flight recompute. Returns the new positions, or null when
   * a newer op won.
   */
  private async _relayoutScope(
    next: Scope,
  ): Promise<Map<string, MapPos> | null> {
    const m = this.deps.model;
    const seq = this._scopeSeq.next();
    const { raw } = await this._computeScopeRaw(next);
    if (!this._scopeSeq.isCurrent(seq)) return null;
    const targets = this._mapIntoScope(raw, next);
    m.scopePositions = targets;
    this.deps.canvas.scheduleDraw();
    return targets;
  }

  /** Convert an open scope between its 2-D and 3-D shapes for the current view.
   *
   *  In 3-D a circle becomes the sphere inscribed at its centre, a box the
   *  sphere whose radius spans its longer half-extent; the reverse drops the
   *  3-D-only fields. A shape already matching the view is returned unchanged.
   */
  private _convertScopeShape(scope: Scope): Scope {
    if (this.deps.model.is3d) {
      if (scope.shape === "sphere") return scope;
      const { hw, hh } = scopeHalfExtents(scope);
      return {
        shape: "sphere",
        cx: scope.cx,
        cy: scope.cy,
        cz: 0.5,
        r: scope.shape === "box" ? Math.max(hw, hh) : scope.r,
        lensId: scope.lensId,
      };
    }
    if (scope.shape === "circle") return scope;
    return {
      shape: "circle",
      cx: scope.cx,
      cy: scope.cy,
      r: scope.r,
      lensId: scope.lensId,
    };
  }

  /** Close the active scope, animating its tracks back to their outer positions. */
  closeScope(): void {
    if (this._scopeModifierId) {
      const id = this._scopeModifierId;
      this._scopeModifierId = null;
      this.deps.viewMods.dismiss(id);
    } else {
      this._closeScopeState();
    }
  }

  /** The raw scope close: drop the scope and animate its tracks back out. */
  private _closeScopeState(): void {
    const m = this.deps.model;
    if (!m.scope) return;
    const start = new Map<string, MapPos>();
    for (const path of m.scopePositions.keys()) {
      const p = m.scopePositions.get(path);
      if (p) start.set(path, { ...p });
    }
    m.scope = null;
    m.scopePositions = new Map();

    const targets = new Map<string, MapPos>();
    for (const path of start.keys()) {
      const t = m.trackByPath(path);
      if (!t) continue;
      const p = this.deps.canvas.outerPosForTrack(t);
      if (p) {
        targets.set(
          path,
          p.wz === undefined
            ? { x: p.wx, y: p.wy }
            : { x: p.wx, y: p.wy, z: p.wz },
        );
      }
    }
    this._animateOverlay(start, targets);
  }

  /** In-scope tracks and their raw scope-lens positions, in the lens's own space. */
  private async _computeScopeRaw(
    scope: Scope,
  ): Promise<{
    raw: Map<string, MapPos>;
    inScope: Track[];
  }> {
    const m = this.deps.model;
    const lens = getLens(scope.lensId);
    const inScope = m.allTracks.filter((t) => {
      const p = this.deps.canvas.outerPosForTrack(t);
      return p != null && scopeContains(scope, p.wx, p.wy, p.wz);
    });
    const all =
      lens.kind === "deterministic" && lens.compute
        ? null
        : await this._fetchLensPositions(lens);
    const raw = new Map<string, MapPos>();
    for (const t of inScope) {
      const p = lens.compute ? lens.compute(t) : (all?.get(t.path) ?? null);
      if (p) raw.set(t.path, p);
    }
    return { raw, inScope };
  }

  /** Outer positions for *paths*, as scope-open anchors. */
  private _startFromOuter(
    paths: Iterable<string>,
  ): Map<string, MapPos> {
    const m = this.deps.model;
    const out = new Map<string, MapPos>();
    for (const path of paths) {
      const t = m.trackByPath(path);
      if (!t) continue;
      const p = this.deps.canvas.outerPosForTrack(t);
      if (p) {
        out.set(
          path,
          p.wz === undefined
            ? { x: p.wx, y: p.wy }
            : { x: p.wx, y: p.wy, z: p.wz },
        );
      }
    }
    return out;
  }

  /** Scale a scope's region by *factor*, clamped to the min/max size. */
  private _scaleScope(scope: Scope, factor: number): Scope {
    const clampR = (v: number) => clamp(v, SCOPE_MIN_R, SCOPE_MAX_R);
    const { hw, hh } = scopeHalfExtents(scope);
    if (scope.shape === "box") {
      const rx = clampR(hw * factor);
      const ry = clampR(hh * factor);
      return { ...scope, rx, ry, r: Math.max(rx, ry) };
    }
    if (scope.shape === "sphere") {
      return { ...scope, r: clampR(scope.r * factor) };
    }
    return { ...scope, r: clampR(hw * factor) };
  }

  /**
   * Normalise a scope lens's subset and map it into the region.
   *
   * A circle gets the lens square inscribed in it (no dot escapes the outline); a
   * box fills the box, so a wide or tall region uses its whole extent. A sphere's
   * contents are a flat map too: the sphere is the 3-D membership volume, but its
   * tracks are laid out in the same inscribed square as the circle of its radius,
   * centred on the sphere's (cx, cy). Every scope maps to {x, y} with no z; a
   * sphere's positions additionally carry the sphere centre as their `billboard`
   * anchor, so the canvas can draw them facing the camera without reading the
   * (possibly already-closed) scope.
   */
  private _mapIntoScope(
    raw: Map<string, MapPos>,
    scope: Scope,
  ): Map<string, MapPos> {
    if (raw.size === 0) return new Map();
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of raw.values()) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    const rx = maxX - minX || 1;
    const ry = maxY - minY || 1;
    const { hw, hh } = scopeHalfExtents(scope);
    const sideX = scope.shape === "box" ? hw * 2 : hw * Math.SQRT2;
    const sideY = scope.shape === "box" ? hh * 2 : hh * Math.SQRT2;
    const bb = scope.shape === "sphere" ? { cx: scope.cx, cy: scope.cy, cz: scope.cz ?? 0.5 } : undefined;
    const out = new Map<string, MapPos>();
    for (const [path, p] of raw) {
      const x = scope.cx + ((p.x - minX) / rx - 0.5) * sideX;
      const y = scope.cy + ((p.y - minY) / ry - 0.5) * sideY;
      out.set(path, bb ? { x, y, billboard: bb } : { x, y });
    }
    return out;
  }

  /** Fetch one lens's layout without touching the active lens or its positions.
   *
   *  Always fetched in 2-D: a scope's contents are laid out as a flat map in
   *  every view dimensionality, and the key is the one a plain 2-D request
   *  uses, so toggling 3-D with a scope open hits the cache. */
  private async _fetchLensPositions(
    lens: Lens,
  ): Promise<Map<string, MapPos>> {
    const qs = this._projectionQueryString(undefined, lens, 2);
    const cached = this._layoutCache.get(qs);
    if (cached) return new Map(cached.positions);
    const raw = await api<unknown>(`/api/embeddings/projection?${qs}`);
    const { positions, revision } = parseProjectionResponse(raw);
    const out = toPositionMap(positions);
    if (revision) this._layoutCachePut(qs, { revision, positions: out });
    return out;
  }

  /** Lerp a set of tracks between two position maps on the shared frame. */
  private _animateOverlay(
    start: Map<string, MapPos>,
    targets: Map<string, MapPos>,
  ): void {
    const cv = this.deps.canvas;
    if (start.size === 0 || targets.size === 0) {
      cv.scheduleDraw();
      return;
    }
    cv.animPositions = new Map(start);
    const DURATION = 280;
    const t0 = performance.now();
    cancelAnimationFrame(this._animFrame);
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / DURATION);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      for (const [path, from] of start) {
        const to = targets.get(path);
        if (!to) continue;
        cv.animPositions!.set(path, lerpPos(from, to, e));
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
}
