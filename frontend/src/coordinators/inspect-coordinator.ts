// ─── InspectCoordinator ─────────────────────────────────────
//
// Owns the pinned inspect card's navigation actions: the "anti-track" (furthest
// neighbour in the embedding space), nearest neighbours, and the neighbourhood
// proximity slider. The card itself is rendered by the canvas (`lib/tooltip.ts`); this
// coordinator turns a card interaction into one fetch plus model mutations, so the
// canvas view stays free of API calls — same seam as `SearchCoordinator`.

import { type Model } from "../model";
import { type InspectState, type TooltipResult } from "../lib/tooltip";

/** Neighbours requested for the "nearest" ranking. */
export const NEAR_K = 5;
/** The anti-track ranking: the most distant tracks, shown directly. */
export const ANTI_K = 10;
/** ``k`` sentinel for "the whole near ranking" (the server's ``all=1`` mode). */
export const PROXIMITY_ALL = 0;

/** Shape of ``GET /api/tracks/similar``. */
export interface SimilarResponse {
  ok: boolean;
  error?: string;
  results?: TooltipResult[];
  /** Full near ranking (paths only), present in ``all`` mode. */
  ranking?: string[];
}

/** The parts of `CanvasView` this coordinator drives (structural for tests). */
export interface InspectViewLike {
  setInspect(patch: Partial<InspectState>): void;
  unpinInspect(): void;
}

export interface InspectCoordinatorDeps {
  model: Model;
  view: InspectViewLike;
  /** Fetch a ranking (injectable so tests avoid `fetch`). */
  fetchSimilar(
    path: string,
    mode: "near" | "far",
    k: number,
  ): Promise<SimilarResponse>;
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

  constructor(private deps: InspectCoordinatorDeps) {}

  /** Run one inspect-card action (`data-inspect-action` or a result focus). */
  async action(action: string, path: string, value?: number): Promise<void> {
    if (action === "close") {
      // The card owns the preview once pinned, so closing it ends the audition.
      this.deps.stopPreview(path);
      this.deps.view.unpinInspect();
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
      await this.setProximity(path, value ?? 100);
      return;
    }
    if (action === "anti" || action === "near") {
      await this.rank(path, action === "anti" ? "far" : "near");
    }
  }

  /** Select one result alone, fit the view to it, and audition it. */
  private focus(path: string): void {
    if (!this.deps.model.trackByPath(path)) return;
    this.deps.model.replaceSelection([path]);
    this.deps.fitToPaths([path]);
    this.deps.preview(path);
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
   * Neighbourhood slider: keep the closest *percent* of the ranked library and
   * hide the rest, re-homing the view as it narrows. 100 % reveals everything
   * (no isolation); because the visible set shrinks gradually the camera move is
   * far less sudden than the old discrete top-N "solo" it replaces.
   */
  private async setProximity(path: string, percent: number): Promise<void> {
    const m = this.deps.model;
    const p = Math.max(0, Math.min(100, percent));
    if (p >= 100) {
      const all = new Set(m.allTracks.map((t) => t.path));
      this._proximityHidden = m.setProximityKeep(all, this._proximityHidden);
      this.deps.fitToPaths(m.tracks.map((t) => t.path));
      return;
    }
    const ranked = await this.rankedFor(path);
    if (ranked === null) return;
    const keepCount = Math.ceil((p / 100) * ranked.length);
    const keep = new Set<string>([path, ...ranked.slice(0, keepCount)]);
    this._proximityHidden = m.setProximityKeep(keep, this._proximityHidden);
    this.deps.fitToPaths(keep);
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
    this._rankingPending = null;
    this._rankingPendingPath = null;
    if (!res.ok || !res.ranking?.length) {
      this.deps.view.setInspect({
        busy: false,
        message: res.error ?? "No ranking available",
      });
      return null;
    }
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
