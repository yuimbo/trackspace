// ─── EvaluationCoordinator ───────────────────────────────────
//
// Owns the read-only layout-evaluation gallery (`?layout_eval=1`): fetching the
// manifest + recipe artifacts, applying precomputed positions to the shared
// canvas, restricting the view to the artifact's tracks, and driving the toolbar
// view's state. It is deliberately *bounded*: it does not know about lenses,
// weights, scopes or analysis — the controller wires it to the embedding
// coordinator's one `applyPrecomputedPositions` entry point and never lets it
// reach the compute/analysis surface.
//
// Read-only is enforced centrally (see `lib/eval-readonly.ts` + the HTTP gate in
// `lib/api.ts`), so the coordinator itself never issues a write and never has to
// second-guess a panel.

import { type Model } from "../model";
import {
  blindLetter,
  computeEvalUniverse,
  evalExitUrl,
  isUnsupportedEvalSchema,
  parseEvalArtifact,
  parseEvalManifest,
  type EvalArtifact,
  type EvalDims,
  type EvalManifest,
  type EvalManifestRecipe,
} from "../lib/layout-eval";
import {
  buildEvalExport,
  loadAllRatings,
  ratedCount,
  saveRating,
  type EvalRatingRecord,
} from "../lib/eval-ratings";
import { setEvalReadOnly } from "../lib/eval-readonly";
import { clearOnNewTimer, latestSeq, type Seq } from "../lib/async";
import type { EvalToolbarState, EvalToolbarView } from "../components/eval-toolbar-view";

/** Max artifact maps kept in memory; the least-recently-used one is evicted past this. */
const EVAL_ARTIFACT_CACHE_MAX = 8;

export interface EvaluationDeps {
  model: Model;
  view: EvalToolbarView;
  /** Apply a precomputed position map on the shared canvas animation frame. */
  applyPositions(
    positions: Map<string, { x: number; y: number; z?: number }>,
    dims: EvalDims,
  ): void;
  /** Move the camera to the default full-map view. */
  resetCamera(): void;
  fetchManifest(): Promise<unknown>;
  fetchArtifact(id: string): Promise<unknown>;
}

/** One blind navigation step: a 2-D base and/or its 3-D sibling. */
interface EvalGroup {
  id: string;
  letter: string;
  members: Map<EvalDims, EvalManifestRecipe>;
}

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class EvaluationCoordinator {
  private seq: Seq = latestSeq();
  private manifest: EvalManifest | null = null;
  private groups: EvalGroup[] = [];
  // The applied map is one identity — index + artifact + dims — committed only
  // when its artifact is ready to draw, so a failed load leaves the canvas and
  // its ratings on the map the user is still looking at.
  private currentIndex = 0;
  private currentArtifact: EvalArtifact | null = null;
  /** Navigation cursor: the last selection asked for; may run ahead while it loads. */
  private requestedIndex = 0;
  /** User's latest dimensionality intent; may differ from the applied artifact. */
  private requestedDims: EvalDims = 2;
  private artifacts = new Map<string, EvalArtifact>();
  private loaded = false;
  private revealed = false;
  private busy = false;
  private error: string | null = null;
  private fp = "";
  private eligible = 0;
  private total = 0;
  private unavailable = 0;
  /** The one debounced note save; flushed before a map switch, export or exit. */
  private pendingNote: (() => void) | null = null;
  private noteDebounce = clearOnNewTimer();

  constructor(private deps: EvaluationDeps) {
    this.deps.view.onSelect = (i) =>
      void this._selectGroup(i, this.requestedDims);
    this.deps.view.onPrev = () => void this._step(-1);
    this.deps.view.onNext = () => void this._step(1);
    this.deps.view.onResetCamera = () => this.deps.resetCamera();
    this.deps.view.onToggleReveal = (c) => this._toggleReveal(c);
    this.deps.view.onRate = (cat, v) => this._rate(cat, v);
    this.deps.view.onNote = (t) => this._note(t);
    this.deps.view.onExport = () => this._export();
    this.deps.view.onExit = () => this._exit();
  }

  /** True once a usable manifest has loaded (eval is active). */
  get active(): boolean {
    return this.manifest !== null;
  }

  /**
   * Enter eval mode and load the gallery. *initialRecipeId* (from ``&layout=<id>``)
   * selects a starting map; defaults to the first.
   */
  async start(initialRecipeId?: string): Promise<void> {
    const m = this.deps.model;
    m.setEvalMode(true);
    setEvalReadOnly(true);
    document.body.classList.add("eval-mode");
    this.deps.resetCamera();
    this._render();
    try {
      const raw = await this.deps.fetchManifest();
      // The route returns {ok:true, manifest}; tolerate a bare manifest too.
      const manifestPayload =
        raw && typeof raw === "object" && "manifest" in (raw as object)
          ? (raw as { manifest: unknown }).manifest
          : raw;
      const manifest = parseEvalManifest(manifestPayload);
      if (!manifest) {
        this._error(
          isUnsupportedEvalSchema(manifestPayload)
            ? "Unsupported layout-evaluation manifest schema — this build cannot display this gallery."
            : "No completed layout-evaluation generation found on this server.",
        );
        return;
      }
      this.manifest = manifest;
      this.fp = manifest.datasetFingerprint;
      this.groups = this._buildGroups(manifest.recipes);
      const initial = initialRecipeId
        ? manifest.recipes.find((r) => r.id === initialRecipeId)
        : undefined;
      let start = 0;
      if (initial) {
        const idx = this.groups.findIndex((x) => x.id === initial.group);
        if (idx >= 0) start = idx;
        this.requestedDims = initial.dims;
      }
      await this._selectGroup(start, this.requestedDims);
    } catch (e) {
      this._error(`Failed to load the layout-evaluation manifest: ${msg(e)}`);
    }
  }

  /** Call once the library stream has finished loading. */
  onLibraryLoaded(): void {
    this.loaded = true;
    if (this.currentArtifact) this._applyUniverse(this.currentArtifact);
    this._render();
  }

  /* ── Map selection ───────────────────────────────────────── */

  /**
   * Collapse manifest recipes into navigation groups. A group is the paired
   * family (``group`` defaults to the recipe id), so a 2-D base and its 3-D
   * sibling share one blind letter and one navigation step. The first recipe in
   * manifest order wins for a duplicated dimensionality.
   */
  private _buildGroups(recipes: EvalManifestRecipe[]): EvalGroup[] {
    const byId = new Map<string, EvalGroup>();
    const groups: EvalGroup[] = [];
    for (const recipe of recipes) {
      let group = byId.get(recipe.group);
      if (!group) {
        group = {
          id: recipe.group,
          letter: blindLetter(groups.length),
          members: new Map(),
        };
        byId.set(recipe.group, group);
        groups.push(group);
      }
      if (!group.members.has(recipe.dims)) group.members.set(recipe.dims, recipe);
    }
    return groups;
  }

  /** The group's artifact for *desired*, falling back to 2-D then 3-D. */
  private _pickDims(group: EvalGroup, desired: EvalDims): EvalDims | null {
    if (group.members.has(desired)) return desired;
    if (group.members.has(2)) return 2;
    if (group.members.has(3)) return 3;
    return null;
  }

  /** The group's label without the backend's raw 3-D suffix. */
  private _groupLabel(group: EvalGroup): string {
    const recipe = group.members.get(2) ?? group.members.get(3);
    return recipe ? recipe.label.replace(/\s*\(3D\)$/, "") : group.id;
  }

  /**
   * Switch the current group between its 2-D and 3-D artifact (the **3** key in
   * eval mode). An unpaired group is a no-op, and the switch is a cache/fetch of
   * the sibling artifact only — never a production projection request.
   */
  async toggleDims(): Promise<void> {
    const group = this.groups[this.requestedIndex];
    if (!group) return;
    const target: EvalDims = this.requestedDims === 3 ? 2 : 3;
    if (!group.members.has(target)) return;
    this.requestedDims = target;
    await this._selectGroup(this.requestedIndex, target);
  }

  private async _step(dir: 1 | -1): Promise<void> {
    const next = this.requestedIndex + dir;
    if (next < 0 || next >= this.groups.length) return;
    await this._selectGroup(next, this.requestedDims);
  }

  private async _selectGroup(index: number, desired: EvalDims): Promise<void> {
    const group = this.groups[index];
    if (!group) return;
    const dims = this._pickDims(group, desired);
    if (dims === null) return;
    const meta = group.members.get(dims)!;
    // The note typed so far belongs to the map being left, not the one loading.
    this._flushNote();
    this.requestedDims = desired;
    this.requestedIndex = index;
    const token = this.seq.next();
    this.busy = true;
    this._render();

    let artifact = this.artifacts.get(meta.id) ?? null;
    if (artifact) {
      // LRU: re-insert to move the hit to the most-recently-used end.
      this.artifacts.delete(meta.id);
      this.artifacts.set(meta.id, artifact);
    } else {
      try {
        const raw = await this.deps.fetchArtifact(meta.id);
        // A newer selection won the race: drop this stale response.
        if (!this.seq.isCurrent(token)) return;
        artifact = parseEvalArtifact(raw);
        if (
          !artifact ||
          artifact.id !== meta.id ||
          artifact.dims !== meta.dims ||
          artifact.group !== group.id
        ) {
          this._failed(`Map ${group.letter} artifact is missing or invalid.`);
          return;
        }
        this._cacheArtifact(meta.id, artifact);
      } catch (e) {
        if (!this.seq.isCurrent(token)) return;
        this._failed(`Failed to load Map ${group.letter}: ${msg(e)}`);
        return;
      }
    }
    if (!this.seq.isCurrent(token)) return;

    this.currentIndex = index;
    this.currentArtifact = artifact;
    this.requestedDims = artifact.dims;
    this.busy = false;
    this.error = null;
    this.deps.applyPositions(artifact.positions, artifact.dims);
    if (this.loaded) this._applyUniverse(artifact);
    this._render();
  }

  /** Drop the failed request: keep the applied map (and its ratings) untouched. */
  private _failed(message: string): void {
    this.requestedIndex = this.currentIndex;
    this.requestedDims = this.currentArtifact?.dims ?? 2;
    this._error(message);
  }

  /** Insert into the artifact cache, evicting the least-recently-used entry past the bound. */
  private _cacheArtifact(recipeId: string, artifact: EvalArtifact): void {
    this.artifacts.delete(recipeId);
    this.artifacts.set(recipeId, artifact);
    if (this.artifacts.size > EVAL_ARTIFACT_CACHE_MAX) {
      const oldest = this.artifacts.keys().next().value;
      if (oldest !== undefined) this.artifacts.delete(oldest);
    }
  }

  private _applyUniverse(artifact: EvalArtifact): void {
    const loaded = new Set(this.deps.model.allTracks.map((t) => t.path));
    const { eligible, unavailable } = computeEvalUniverse(
      new Set(artifact.positions.keys()),
      loaded,
    );
    this.eligible = eligible.size;
    this.total = artifact.positions.size;
    this.unavailable = unavailable;
    this.deps.model.setEvalUniverse(eligible);
  }

  /* ── Reveal / ratings ────────────────────────────────────── */

  private _toggleReveal(checked: boolean): void {
    this.revealed = checked;
    if (checked && this.fp) {
      const group = this.groups[this.currentIndex];
      const recipeId = this.currentArtifact?.id;
      const revision = this.currentArtifact?.revision;
      const label = group ? this._groupLabel(group) : "";
      if (recipeId && revision) saveRating(this.fp, revision, { label, recipeId });
    }
    this._render();
  }

  /** The stored record for the current map, looked up in a pre-parsed store. */
  private _currentRecord(
    all: Record<string, Record<string, EvalRatingRecord>>,
  ): EvalRatingRecord | null {
    const revision = this.currentArtifact?.revision;
    return this.fp && revision ? (all[this.fp]?.[revision] ?? null) : null;
  }

  private _rate(
    category: "neighbours" | "organisation" | "crowding" | "overall",
    value: number | null,
  ): void {
    const revision = this.currentArtifact?.revision;
    const recipeId = this.currentArtifact?.id ?? "";
    const group = this.groups[this.currentIndex];
    if (!this.fp || !revision) return;
    saveRating(this.fp, revision, {
      [category]: value,
      recipeId,
      label: this.revealed && group ? this._groupLabel(group) : "",
    });
    this._render();
  }

  private _note(text: string): void {
    const artifact = this.currentArtifact;
    const group = this.groups[this.currentIndex];
    if (!this.fp || !artifact) return;
    // Capture the target so a map switch mid-debounce lands the note on the
    // right record instead of the map that happens to be showing when it fires.
    const fp = this.fp;
    const revision = artifact.revision;
    const recipeId = artifact.id;
    const label = this.revealed && group ? this._groupLabel(group) : "";
    this.pendingNote = () => saveRating(fp, revision, { note: text, recipeId, label });
    this.noteDebounce.set(() => {
      this._flushNote();
      this._render();
    }, 400);
  }

  /** Persist the debounced note now, if one is waiting. */
  private _flushNote(): void {
    this.noteDebounce.cancel();
    const save = this.pendingNote;
    this.pendingNote = null;
    save?.();
  }

  private _export(): void {
    if (!this.fp) return;
    this._flushNote();
    const records = Object.values(loadAllRatings()[this.fp] ?? {});
    const json = buildEvalExport(this.fp, records);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `layout-eval-${this.fp.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private _exit(): void {
    this._flushNote();
    window.location.href = evalExitUrl(window.location.href);
  }

  /* ── Render ──────────────────────────────────────────────── */

  private _error(message: string): void {
    this.error = message;
    this.busy = false;
    this._render();
  }

  private _timingLabel(meta: EvalManifestRecipe): string {
    const parts: string[] = [];
    for (const k of ["pca", "tsne", "umap", "alignment"]) {
      const v = meta.timings?.[k];
      if (typeof v === "number") parts.push(`${k} ${v.toFixed(1)}s`);
    }
    return parts.length ? parts.join(" · ") : meta.label;
  }

  private _render(): void {
    if (!this.manifest) {
      this.deps.view.render(this._emptyState());
      return;
    }
    const group = this.groups[this.currentIndex];
    const meta = this.currentArtifact
      ? this.manifest.recipes.find((r) => r.id === this.currentArtifact!.id)
      : undefined;
    // Parse the ratings store once per render; per-map lookups are plain reads.
    const ratings = loadAllRatings();
    const fpRatings = this.fp ? (ratings[this.fp] ?? {}) : {};
    const record = this._currentRecord(ratings);
    const state: EvalToolbarState = {
      busy: this.busy,
      currentIndex: this.currentIndex,
      count: this.groups.length,
      maps: this.groups.map((g, i) => {
        const shown = this._artifactForGroup(g, i);
        const rev = shown.artifact?.revision ?? "";
        const rec = rev ? (fpRatings[rev] ?? null) : null;
        return {
          index: i,
          letter: g.letter,
          label: this.revealed ? this._groupLabel(g) : undefined,
          dims: shown.dims,
          rated: ratedCount(rec),
          revealed: this.revealed,
        };
      }),
      revealed: this.revealed,
      currentTiming: this.revealed && meta ? this._timingLabel(meta) : "",
      eligible: this.eligible,
      total: this.total,
      unavailable: this.unavailable,
      ratings: record,
      note: record?.note ?? "",
      canRate: this.currentArtifact !== null && !this.busy,
      dims: this.currentArtifact?.dims ?? 2,
      paired: group ? group.members.has(2) && group.members.has(3) : false,
      datasetLabel: this.fp ? `dataset ${this.fp.slice(0, 8)}` : "",
      error: this.error,
    };
    this.deps.view.render(state);
  }

  /**
   * The artifact whose rating badge a group shows. A group's badge is always
   * tied to one explicit dimension: the current group's applied artifact, else
   * the requested dimension, else a cached 2-D/3-D sibling.
   */
  private _artifactForGroup(
    group: EvalGroup,
    index: number,
  ): { artifact: EvalArtifact | null; dims: EvalDims } {
    const current = this.currentArtifact;
    if (
      index === this.currentIndex &&
      current &&
      group.members.get(current.dims)?.id === current.id
    ) {
      return { artifact: current, dims: current.dims };
    }
    const dims = this._pickDims(group, this.requestedDims) ?? 2;
    const recipe = group.members.get(dims);
    const cached = recipe ? (this.artifacts.get(recipe.id) ?? null) : null;
    if (cached) return { artifact: cached, dims };
    for (const fallback of [2, 3] as EvalDims[]) {
      const member = group.members.get(fallback);
      const artifact = member ? (this.artifacts.get(member.id) ?? null) : null;
      if (artifact) return { artifact, dims: fallback };
    }
    return { artifact: null, dims };
  }

  private _emptyState(): EvalToolbarState {
    return {
      busy: this.busy,
      currentIndex: 0,
      count: 0,
      maps: [],
      revealed: false,
      currentTiming: "",
      eligible: 0,
      total: 0,
      unavailable: 0,
      ratings: null,
      note: "",
      canRate: false,
      dims: 2,
      paired: false,
      datasetLabel: "",
      error: this.error,
    };
  }
}
