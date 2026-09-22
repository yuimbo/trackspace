// ─── InspectCoordinator ─────────────────────────────────────
//
// Owns the pinned inspect card's navigation actions: the "anti-track" (furthest
// neighbour in the embedding space), nearest neighbours, and the neighbourhood
// proximity slider. The card itself is rendered by the canvas (`lib/tooltip.ts`); this
// coordinator turns a card interaction into one fetch plus model mutations, so the
// canvas view stays free of API calls — same seam as `SearchCoordinator`.

import { type Model } from "../model";
import {
  type CoordinateResult,
  type DescribeResult,
  type InspectState,
  type InspectTab,
  type TooltipResult,
  emptyDescribe,
} from "../lib/tooltip";
import type { ViewModifierSink } from "./view-modifier-coordinator";

/** Neighbours requested for the "nearest" ranking. */
export const NEAR_K = 5;
/** The anti-track ranking: the most distant tracks, shown directly. */
export const ANTI_K = 10;
/** Phrases requested for the Describe tab (server max is 20). */
export const DESCRIBE_K = 8;
/** ``k`` sentinel for "the whole near ranking" (the server's ``all=1`` mode). */
export const PROXIMITY_ALL = 0;
/**
 * Proximity-slider throttle window. A drag fires an input event per pointer
 * move, and every one of them re-hides the library, re-homes the camera and runs
 * the model "change" cascade (status, panels, tree). Applying at most one step per
 * window — the first immediately, the last on the trailing edge — keeps the
 * neighbourhood live without the drag fighting the main thread.
 */
export const PROXIMITY_THROTTLE_MS = 100;

/** Shape of ``GET /api/tracks/similar``. */
export interface SimilarResponse {
  ok: boolean;
  error?: string;
  results?: TooltipResult[];
  /** Full near ranking (paths only), present in ``all`` mode. */
  ranking?: string[];
}

/** Shape of ``GET /api/tracks/describe``. */
export interface DescribeResponse {
  ok: boolean;
  error?: string;
  results?: DescribeResult[];
  coordinates?: CoordinateResult[];
}

/** The parts of `CanvasView` this coordinator drives (structural for tests). */
export interface InspectViewLike {
  setInspect(patch: Partial<InspectState>): void;
  unpinInspect(): void;
}

export interface InspectCoordinatorDeps {
  model: Model;
  view: InspectViewLike;
  /** The active-view-modifier stack: the proximity filter is one chip. */
  viewMods: ViewModifierSink;
  /** Fetch a ranking (injectable so tests avoid `fetch`). */
  fetchSimilar(
    path: string,
    mode: "near" | "far",
    k: number,
  ): Promise<SimilarResponse>;
  /** Fetch the zero-shot CLAP descriptions (injectable so tests avoid `fetch`). */
  fetchDescribe(path: string, limit: number): Promise<DescribeResponse>;
  /** Fit the canvas viewport to the given paths. */
  fitToPaths(paths: Iterable<string>): void;
  /** Start the hover preview for a path (no-op when hover preview is off). */
  preview(path: string): void;
  /** Stop the hover preview (the pointer left the previewing element). */
  stopPreview(path: string): void;
}

export class InspectCoordinator {
  /** Paths the proximity slider hid on its last step (revealed on the next). */
  private _proximityHidden = new Set<string>();
  /** Cached full near ranking for the inspected path (one fetch per pin). */
  private _rankingPath: string | null = null;
  private _ranking: string[] = [];
  /** In-flight ranking fetch, so a drag burst shares one request. */
  private _rankingPending: Promise<string[] | null> | null = null;
  private _rankingPendingPath: string | null = null;
  /** Proximity throttle: the latest requested step and its callers. */
  private _proximityPending: { path: string; percent: number } | null = null;
  private _proximityWaiters: Array<() => void> = [];
  private _proximityTimer: ReturnType<typeof setTimeout> | null = null;
  private _proximityLastRun = 0;
  /** Monotonic step id: a stale async apply must not overwrite a newer one. */
  private _proximitySeq = 0;
  /** Path whose descriptions loaded successfully (avoids a refetch per tab switch). */
  private _describePath: string | null = null;
  /** Path with a describe fetch in flight, so a tab burst shares one request. */
  private _describeInFlight: string | null = null;
  /** Slider direction: keep the closest (near) or the furthest (anti-track). */
  private _rankMode: "near" | "far" = "near";
  /** Error from the last full-ranking fetch, shown when the slider has no data. */
  private _rankingError: string | null = null;
  /** Id of the active `proximity` modifier chip, or null when not filtering. */
  private _proximityModifierId: string | null = null;
  /** Path the proximity modifier filters around (the inspected track). */
  private _proximityPath: string | null = null;
  /** Last applied proximity percent, so a direction flip can refresh the chip. */
  private _proximityPercent = 100;

  constructor(private deps: InspectCoordinatorDeps) {}

  /** Run one inspect-card action (`data-inspect-action` or a result focus). */
  async action(action: string, path: string, value?: number): Promise<void> {
    // Any other action supersedes a pending slider step, so a trailing apply
    // cannot re-hide the library after e.g. a focus or close.
    if (action !== "proximity") this._cancelProximity();
    if (action === "close") {
      // The card owns the preview once pinned, so closing it ends the audition.
      this.deps.stopPreview(path);
      this._describePath = null;
      this._describeInFlight = null;
      // A proximity filter outlives the card as a view modifier, so its
      // direction must survive; only an unfiltered card resets to nearest.
      if (!this._proximityModifierId) this._rankMode = "near";
      this._rankingError = null;
      this.deps.view.unpinInspect();
      return;
    }
    if (action === "show-navigate" || action === "show-describe") {
      this.setTab(path, action === "show-describe" ? "describe" : "navigate");
      return;
    }
    if (action === "open") {
      // The card was just pinned: default to nearest, load descriptions so the
      // coordinates sit next to the audio features, list the nearest tracks, and
      // pre-load the full ranking so the slider filters locally at once.
      // A proximity filter belongs to the track it was narrowed around, so pinning
      // another track drops the old neighbourhood.
      if (this._proximityPath && this._proximityPath !== path) {
        this._clearProximity();
      }
      this._rankMode = "near";
      this.deps.view.setInspect({ mode: "near" });
      void this.ensureDescribe(path);
      void this.rankedFor(path);
      void this.rank(path, "near");
      return;
    }
    if (action === "toggle-rank-mode") {
      // Invert the slider: nearest keeps the closest, anti-track the furthest.
      this._rankMode = this._rankMode === "near" ? "far" : "near";
      await this.rank(path, this._rankMode);
      if (this._proximityModifierId && this._proximityPath === path) {
        this._syncProximityModifier(path, this._proximityPercent);
      }
      return;
    }
    if (action === "focus") {
      this.focus(path);
      return;
    }
    if (action === "hover") {
      // Hovering a result row auditions it straight away, like a canvas hover.
      // It deliberately does *not* stop on leaving the row: the audition lasts
      // while the pointer is over the track *or* the card.
      this.deps.preview(path);
      return;
    }
    if (action === "leave") {
      // The pointer left the open card, so nothing is hovered any more.
      this.deps.stopPreview(path);
      return;
    }
    if (action === "proximity") {
      await this._scheduleProximity(path, value ?? 100);
    }
  }

  /** Select one result alone, fit the view to it, and audition it. */
  private focus(path: string): void {
    if (!this.deps.model.trackByPath(path)) return;
    this.deps.model.replaceSelection([path]);
    this.deps.fitToPaths([path]);
    this.deps.preview(path);
  }

  /** Switch the pinned card's tab; the Describe tab loads its phrases on demand. */
  private setTab(path: string, tab: InspectTab): void {
    this.deps.view.setInspect({ tab });
    if (tab === "describe") void this.ensureDescribe(path);
  }

  /**
   * Load the zero-shot descriptions + coordinates for *path* once, lazily.
   *
   * Called when the card is pinned, so the coordinates show next to the audio
   * features without waiting for the Describe tab, and again when the tab is opened.
   * The result is cached per path, so a tab switch does not refetch. The state
   * carries the path it belongs to, so a response that lands after the user pinned
   * another track is never shown on the wrong card.
   */
  private async ensureDescribe(path: string): Promise<void> {
    if (this._describePath === path) return;
    if (this._describeInFlight === path) return;
    this._describeInFlight = path;
    this.deps.view.setInspect({ describe: emptyDescribe(path, true) });
    let res: DescribeResponse;
    try {
      res = await this.deps.fetchDescribe(path, DESCRIBE_K);
    } catch (err) {
      res = {
        ok: false,
        error: err instanceof Error ? err.message : "Describe lookup failed",
      };
    }
    if (this._describeInFlight === path) this._describeInFlight = null;
    if (!res.ok) {
      this.deps.view.setInspect({
        describe: {
          ...emptyDescribe(path, false),
          message: res.error ?? "Describe lookup failed",
        },
      });
      return;
    }
    this._describePath = path;
    this.deps.view.setInspect({
      describe: {
        path,
        busy: false,
        message: null,
        results: res.results ?? [],
        coordinates: res.coordinates ?? [],
      },
    });
  }

  private async rank(path: string, mode: "near" | "far"): Promise<void> {
    const limit = mode === "far" ? ANTI_K : NEAR_K;
    this.deps.view.setInspect({ busy: true, message: null, mode, results: [] });
    const res = await this.fetch(path, mode, limit);
    if (!res.ok) {
      this.deps.view.setInspect({
        busy: false,
        message: res.error ?? "Similarity lookup failed",
        results: [],
      });
      return;
    }
    this.deps.view.setInspect({
      busy: false,
      message: null,
      mode,
      results: res.results ?? [],
    });
  }

  /**
   * Coalesce a slider drag into at most one step per `PROXIMITY_THROTTLE_MS`
   * window: the first step runs immediately (so the map tracks the drag), the last
   * runs on the trailing edge (so the final value always lands). The returned
   * promise resolves once the applied step has finished, which keeps the tests
   * deterministic without exposing the timer.
   */
  private _scheduleProximity(path: string, percent: number): Promise<void> {
    this._proximityPending = { path, percent };
    const wait = new Promise<void>((resolve) =>
      this._proximityWaiters.push(resolve),
    );
    if (this._proximityTimer !== null) return wait;
    const elapsed = Date.now() - this._proximityLastRun;
    if (elapsed >= PROXIMITY_THROTTLE_MS) this._runProximity();
    else
      this._proximityTimer = setTimeout(
        () => this._runProximity(),
        PROXIMITY_THROTTLE_MS - elapsed,
      );
    return wait;
  }

  /** Apply the pending slider step and release everyone waiting on it. */
  private _runProximity(): void {
    if (this._proximityTimer !== null) {
      clearTimeout(this._proximityTimer);
      this._proximityTimer = null;
    }
    const pending = this._proximityPending;
    this._proximityPending = null;
    if (!pending) return;
    this._proximityLastRun = Date.now();
    const waiters = this._proximityWaiters;
    this._proximityWaiters = [];
    const seq = ++this._proximitySeq;
    void this.setProximity(pending.path, pending.percent, seq)
      .catch(() => {})
      .finally(() => {
        for (const resolve of waiters) resolve();
      });
  }

  /** Drop a pending slider step and release its waiters (superseded action). */
  private _cancelProximity(): void {
    if (this._proximityTimer !== null) {
      clearTimeout(this._proximityTimer);
      this._proximityTimer = null;
    }
    this._proximityPending = null;
    const waiters = this._proximityWaiters;
    this._proximityWaiters = [];
    for (const resolve of waiters) resolve();
  }

  /**
   * Distance-filter slider: keep the closest *percent* of the ranked library
   * (nearest) or the furthest *percent* (anti-track), and hide the rest. 100 %
   * reveals everything (no isolation). The filter deliberately does *not* re-home the
   * view: the camera stays put while the visible set changes, so adjusting the
   * filter never yanks the map. *seq* guards the async ranking fetch: a superseded
   * step must not apply its stale result.
   */
  private async setProximity(
    path: string,
    percent: number,
    seq: number,
  ): Promise<void> {
    const m = this.deps.model;
    const p = Math.max(0, Math.min(100, percent));
    if (p >= 100) {
      const all = new Set(m.allTracks.map((t) => t.path));
      this._proximityHidden = m.setProximityKeep(all, this._proximityHidden);
      this._syncProximityModifier(path, p);
      return;
    }
    const ranked = await this.rankedFor(path);
    if (ranked === null) {
      if (seq === this._proximitySeq) {
        this.deps.view.setInspect({
          busy: false,
          message: this._rankingError ?? "No ranking available",
        });
      }
      return;
    }
    if (seq !== this._proximitySeq) return;
    const keepCount = Math.ceil((p / 100) * ranked.length);
    const slice =
      this._rankMode === "far"
        ? ranked.slice(Math.max(0, ranked.length - keepCount))
        : ranked.slice(0, keepCount);
    const keep = new Set<string>([path, ...slice]);
    this._proximityHidden = m.setProximityKeep(keep, this._proximityHidden);
    this._syncProximityModifier(path, p);
  }

  /**
   * The proximity chip's slider moved. Applies the filter around the track the
   * modifier belongs to, so it works after the inspect card has closed.
   */
  async setProximityFromChip(percent: number): Promise<void> {
    const path = this._proximityPath;
    if (!path) return;
    await this._scheduleProximity(path, percent);
  }

  /**
   * Mirror the proximity filter as one chip: pushed on the first narrowing,
   * refreshed in place on each step, removed at 100 %. The chip's clear reveals
   * every track the slider hid.
   */
  private _syncProximityModifier(path: string, percent: number): void {
    const p = Math.max(0, Math.min(100, percent));
    this._proximityPercent = p;
    if (p >= 100) {
      this._removeProximityModifier();
      return;
    }
    const label = this.deps.model.trackByPath(path)?.filename ?? path;
    const detail = `${this._rankMode === "far" ? "anti-track" : "nearest"} ${Math.round(p)}%`;
    this._proximityPath = path;
    if (this._proximityModifierId) {
      this.deps.viewMods.update(this._proximityModifierId, {
        label,
        detail,
        value: p,
      });
    } else {
      this._proximityModifierId = this.deps.viewMods.push(
        { kind: "proximity", label, detail, value: p },
        () => this._clearProximity(),
      );
    }
  }

  /** Drop the proximity chip without re-applying the filter. */
  private _removeProximityModifier(): void {
    const id = this._proximityModifierId;
    this._proximityModifierId = null;
    this._proximityPath = null;
    if (id) this.deps.viewMods.remove(id);
  }

  /** Reveal every track the proximity slider hid and drop its chip. */
  private _clearProximity(): void {
    this._removeProximityModifier();
    const all = new Set(this.deps.model.allTracks.map((t) => t.path));
    this._proximityHidden = this.deps.model.setProximityKeep(
      all,
      this._proximityHidden,
    );
  }

  /**
   * The full near ranking for *path*, fetched once (paths only) and cached, so the
   * slider filters locally instead of paying a request per drag step. A drag burst
   * before the first fetch resolves shares that one in-flight request.
   */
  private rankedFor(path: string): Promise<string[] | null> {
    if (this._rankingPath === path && this._ranking.length) {
      return Promise.resolve(this._ranking);
    }
    if (this._rankingPending && this._rankingPendingPath === path) {
      return this._rankingPending;
    }
    this._rankingPendingPath = path;
    this._rankingPending = this.fetchRanking(path);
    return this._rankingPending;
  }

  private async fetchRanking(path: string): Promise<string[] | null> {
    const res = await this.fetch(path, "near", PROXIMITY_ALL);
    // Only this request's own slot is cleared: a stale fetch for another path
    // must not cancel the newer path's in-flight request.
    if (this._rankingPendingPath === path) {
      this._rankingPending = null;
      this._rankingPendingPath = null;
    }
    // The pre-load is silent: the error is only surfaced when the slider
    // actually needs the ranking (see `setProximity`).
    if (!res.ok || !res.ranking?.length) {
      this._rankingError = res.error ?? "No ranking available";
      return null;
    }
    this._rankingError = null;
    this._rankingPath = path;
    this._ranking = res.ranking;
    return res.ranking;
  }

  /** Fetch, turning a thrown transport error into a failed response. */
  private async fetch(
    path: string,
    mode: "near" | "far",
    k: number,
  ): Promise<SimilarResponse> {
    try {
      return await this.deps.fetchSimilar(path, mode, k);
    } catch (err) {
      return {
        ok: false,
        error: err instanceof Error ? err.message : "Similarity lookup failed",
      };
    }
  }
}
