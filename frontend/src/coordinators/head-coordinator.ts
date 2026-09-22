// ─── HeadCoordinator ────────────────────────────────────────
//
// Owns the folder-supervised head's controls: training (and its folder
// selection), export/import/compare, and the imprint fan-out flow. Extracted from
// `AnalysisCoordinator`, where the head panel had grown into a separate concern
// from the job queue + cluster panel. The queue only touches the head through
// `syncTrainEnabled`, so the two stay independent.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type Model } from "../model";
import { toast } from "../lib/toast";
import { api, apiBlob, postJSON, postForm } from "../lib/api";
import { confirmAction } from "../lib/util";
import { Poller } from "../lib/async";

export interface HeadDeps {
  model: Model;
  /** Refetch the embedding layout after the head or its rows change. */
  refreshProjection?(): Promise<void> | void;
  /** Refetch clusters after an import re-projects the library. */
  refreshClusters?(): void;
  /** Refetch predicted folders after the head changes (train/import). */
  refreshPredictions?(): void;
  /** Refresh the folder tree after an imprint root is added or removed. */
  onRootsChanged?(): Promise<void> | void;
}

/** One node of ``GET /api/folders`` — the training-folder selector's tree. */
interface FolderNode {
  name: string;
  path: string;
  children?: FolderNode[];
  /** Present on root nodes only: ``"library"`` or ``"external"``. */
  role?: string;
}

/** Shape of ``GET /api/supervised/imprint/status``. */
interface ImprintStatus {
  ok: boolean;
  tracks?: number;
  analyzed?: number;
  pending?: number;
  ready?: boolean;
  error?: string;
}

/** Shape of ``GET /api/supervised/fanout/undo``. */
interface UndoStatus {
  ok: boolean;
  available?: boolean;
  total?: number;
}

/** Shape of ``GET /api/supervised/head`` — the installed head's metadata. */
interface HeadInfo {
  available: boolean;
  run_id?: number;
  /** Unix seconds the head was trained locally. */
  trained_at?: number;
  /** Unix seconds the head was imported (present instead of a local train). */
  imported_at?: number;
  n_classes?: number;
  in_dim?: number;
  input_sources?: string[];
  train_folders?: string[];
  source_versions?: Record<string, number>;
  /** Percentile (0–100) used to calibrate each class's match floor. */
  match_percentile?: number;
  has_match_thresholds?: boolean;
  has_centroids?: boolean;
}

export class HeadCoordinator {
  /** Checked virtual folders for the next training run; empty = all roots. */
  private _trainFolders: Set<string>;
  /** Last ``GET /api/folders`` roots, kept for re-render after "All roots". */
  private _trainFolderRoots: FolderNode[] = [];
  /** Ephemeral imprint root id, removed when the imprint finishes. */
  private _imprintRootId: string | null = null;
  /** Virtual folder of the current imprint target, for status/plan/apply. */
  private _imprintFolder: string | null = null;
  /** Analysis-status poll for the imprint target. */
  private _imprintPoll = new Poller(2000, () => void this._pollImprint());

  constructor(private deps: HeadDeps) {
    this._trainFolders = new Set(deps.model.trainFolders);
  }

  /** Check for an unfinished imprint left behind by a closed browser. */
  start(): void {
    void this._checkStaleImprints();
    void this._refreshUndo();
    void this.refreshHeadInfo();
  }

  stop(): void {
    this._stopImprintPoll();
  }

  /** Wire the head panel controls (train/export/import/compare, imprint, undo). */
  bindPanel(): void {
    const $train = document.getElementById("btn-train-supervised");
    if ($train) {
      $train.addEventListener("click", () => void this._trainSupervised());
    }

    const $export = document.getElementById("btn-export-head");
    if ($export) {
      $export.addEventListener("click", () => void this._exportHead());
    }
    const $import = document.getElementById("btn-import-head");
    const $importFile = document.getElementById(
      "head-import-file",
    ) as HTMLInputElement | null;
    if ($import && $importFile) {
      $import.addEventListener("click", () => $importFile.click());
      $importFile.addEventListener("change", () =>
        void this._importHead($importFile),
      );
    }
    const $compare = document.getElementById("btn-compare-head");
    const $compareFile = document.getElementById(
      "head-compare-file",
    ) as HTMLInputElement | null;
    if ($compare && $compareFile) {
      $compare.addEventListener("click", () => $compareFile.click());
      $compareFile.addEventListener("change", () =>
        void this._compareHead($compareFile),
      );
    }
    const $foldersAll = document.getElementById("btn-head-folders-all");
    if ($foldersAll) {
      $foldersAll.addEventListener("click", () => {
        this._trainFolders.clear();
        this._syncTrainFolders();
        this._renderTrainingFolders();
      });
    }
    // The folder tree can be large, so fetch it when the selector is opened.
    const $foldersDetails = document.getElementById("head-train-folders-details");
    if ($foldersDetails) {
      $foldersDetails.addEventListener("toggle", () => {
        if (($foldersDetails as HTMLDetailsElement).open) {
          void this._loadTrainingFolders();
        }
      });
    }

    const $imprintAnalyze = document.getElementById("btn-imprint-analyze");
    if ($imprintAnalyze) {
      $imprintAnalyze.addEventListener("click", () =>
        void this._imprintAnalyze(),
      );
    }
    const $imprintPlan = document.getElementById("btn-imprint-plan");
    if ($imprintPlan) {
      $imprintPlan.addEventListener("click", () => void this._imprintFanout(false));
    }
    const $imprintApply = document.getElementById("btn-imprint-apply");
    if ($imprintApply) {
      $imprintApply.addEventListener("click", () => void this._imprintFanout(true));
    }
    const $undo = document.getElementById("btn-imprint-undo");
    if ($undo) {
      $undo.addEventListener("click", () => void this._undoMoves());
    }
    const $discard = document.getElementById("btn-imprint-discard");
    if ($discard) {
      $discard.addEventListener("click", () => void this._discardImprints());
    }

    this._bindHeadModal();
  }

  /** Disable Train while a supervise job is outstanding (queue owns dedupe). */
  syncTrainEnabled(disabled: boolean): void {
    const $train = document.getElementById(
      "btn-train-supervised",
    ) as HTMLButtonElement | null;
    if (!$train) return;
    $train.disabled = disabled;
    $train.title = disabled
      ? "A training run is already queued"
      : "Train the folder-supervised view (retrains on your folder structure)";
  }

  /* ── Training folders ─────────────────────────────────────── */

  /** Queue one folder-supervised head training run. */
  private async _trainSupervised(): Promise<void> {
    const folders = [...this._trainFolders];
    try {
      await postJSON("/api/supervised/train", { folders });
      toast(
        folders.length
          ? `Training on ${folders.length} folder(s)…`
          : "Training folder-supervised view…",
        "ok",
      );
    } catch {
      /* api() already toasted */
    }
  }

  /** Mirror the checked folders onto the model so they survive a reload. */
  private _syncTrainFolders(): void {
    this.deps.model.trainFolders = [...this._trainFolders];
    this.deps.model.saveLS();
  }

  /** Load the folder tree and render the training-folder checkboxes. */
  private async _loadTrainingFolders(): Promise<void> {
    try {
      const res = await api<{ roots: FolderNode[] }>("/api/folders");
      this._trainFolderRoots = res.roots ?? [];
      this._renderTrainingFolders();
    } catch {
      /* api() already toasted */
    }
  }

  /** Re-render the checkbox tree from the cached roots and checked set. */
  private _renderTrainingFolders(): void {
    const host = document.getElementById("head-train-folders");
    if (!host) return;
    host.replaceChildren();
    for (const root of this._trainFolderRoots) {
      host.appendChild(this._folderNodeEl(root, 0));
    }
  }

  /** One checkbox row plus a collapsible children container. */
  private _folderNodeEl(node: FolderNode, depth: number): HTMLElement {
    const wrap = document.createElement("div");
    const row = document.createElement("div");
    row.className = "head-folder-row";
    row.style.paddingLeft = `${depth * 12}px`;

    const kids = node.children ?? [];
    const toggle = document.createElement("span");
    toggle.className = "head-folder-toggle" + (kids.length ? "" : " leaf");
    toggle.textContent = kids.length ? "▸" : "·";

    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.folder = node.path;
    box.checked = this._trainFolders.has(node.path);
    box.addEventListener("change", () => {
      if (box.checked) this._trainFolders.add(node.path);
      else this._trainFolders.delete(node.path);
      this._syncTrainFolders();
    });
    const name = document.createElement("span");
    name.className = "head-folder-name";
    name.textContent =
      node.role === "external" ? `${node.name} (external)` : node.name;
    label.append(box, name);
    row.append(toggle, label);
    wrap.appendChild(row);

    if (kids.length) {
      const children = document.createElement("div");
      children.style.display = "none";
      toggle.addEventListener("click", () => {
        const open = children.style.display !== "none";
        children.style.display = open ? "none" : "block";
        toggle.textContent = open ? "▸" : "▾";
      });
      for (const child of kids) {
        children.appendChild(this._folderNodeEl(child, depth + 1));
      }
      wrap.appendChild(children);
    }
    return wrap;
  }

  /* ── Imprint (fan-out an external folder) ─────────────────── */

  /** Fill the imprint target from a folder row and open the head modal. */
  prefillImprint(absPath: string): void {
    this.openModal(absPath);
  }

  /** Open the head modal (the only home for head tools + Train folder view). */
  openModal(absPath = ""): void {
    const $modal = document.getElementById("head-modal");
    const $path = document.getElementById(
      "head-imprint-path",
    ) as HTMLInputElement | null;
    const $target = document.getElementById("head-modal-target");
    if ($path && absPath) $path.value = absPath;
    if ($target) {
      $target.textContent = absPath
        ? `Projecting onto ${absPath}`
        : "Project the folder head onto a destination folder.";
    }
    $modal?.classList.remove("hidden");
    this._refreshUndo();
    if (absPath) $path?.focus();
  }

  closeModal(): void {
    document.getElementById("head-modal")?.classList.add("hidden");
  }

  private _bindHeadModal(): void {
    const $modal = document.getElementById("head-modal");
    if (!$modal) return;
    document
      .getElementById("head-modal-close")
      ?.addEventListener("click", () => this.closeModal());
    $modal.addEventListener("click", (e) => {
      if (e.target === $modal) this.closeModal();
    });
    // Capture phase + stopImmediatePropagation, so Esc closes the modal without
    // also reaching the global Esc handler (which would clear the selection).
    document.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Escape" && !$modal.classList.contains("hidden")) {
          e.stopImmediatePropagation();
          this.closeModal();
        }
      },
      true,
    );
  }

  private _setImprintNote(text: string): void {
    const el = document.getElementById("imprint-note");
    if (el) el.textContent = text;
  }

  /** Register the target folder and queue analysis for its tracks. */
  private async _imprintAnalyze(): Promise<void> {
    const $path = document.getElementById(
      "head-imprint-path",
    ) as HTMLInputElement | null;
    const path = $path?.value.trim() ?? "";
    if (!path) {
      toast("Enter a folder path to imprint", "error");
      return;
    }
    try {
      const res = await postJSON<{
        ok: boolean;
        root_id?: string;
        folder?: string;
        tracks?: number;
        analyzed?: number;
        pending?: number;
        error?: string;
      }>("/api/supervised/imprint/analyze", { path });
      if (!res.ok) {
        this._setImprintNote(res.error || "Analyze failed");
        return;
      }
      this._imprintRootId = res.root_id ?? null;
      this._imprintFolder = res.folder ?? null;
      const pending = res.pending ?? 0;
      this._setImprintNote(
        pending
          ? `Analysing ${res.tracks ?? 0} tracks — ${res.analyzed ?? 0} done, ${pending} left…`
          : `${res.tracks ?? 0} tracks ready — Review plan.`,
      );
      toast(`Imprint: ${res.tracks ?? 0} tracks`, "ok");
      if (pending > 0) this._startImprintPoll();
      void this.deps.onRootsChanged?.();
      this.deps.refreshClusters?.();
    } catch {
      /* api() already toasted */
    }
  }

  private _startImprintPoll(): void {
    this._imprintPoll.start();
  }

  private _stopImprintPoll(): void {
    this._imprintPoll.stop();
  }

  /** Poll analysis readiness until the imprint target is fully analysed. */
  private async _pollImprint(): Promise<void> {
    if (!this._imprintFolder) return;
    try {
      const res = await api<ImprintStatus>(
        `/api/supervised/imprint/status?folder=${encodeURIComponent(this._imprintFolder)}`,
      );
      if (!res.ok) {
        this._stopImprintPoll();
        this._setImprintNote(res.error || "Status failed");
        return;
      }
      const pending = res.pending ?? 0;
      this._setImprintNote(
        pending
          ? `Analysing ${res.tracks ?? 0} tracks — ${res.analyzed ?? 0} done, ${pending} left…`
          : `${res.tracks ?? 0} tracks ready — Review plan.`,
      );
      if (pending === 0) {
        this._stopImprintPoll();
        this.deps.refreshProjection?.();
        this.deps.refreshClusters?.();
      }
    } catch {
      this._stopImprintPoll();
    }
  }

  /** Plan (``apply=false``) or apply the imprint target's predicted moves. */
  private async _imprintFanout(apply: boolean): Promise<void> {
    if (!this._imprintFolder) {
      toast("Analyze a folder first", "error");
      return;
    }
    if (
      apply &&
      !confirmAction(
        `Move every track under "${this._imprintFolder}" into the head's folders? ` +
          "You can undo this from the head panel afterwards.",
      )
    ) {
      return;
    }
    const generic =
      (document.getElementById("imprint-generic-toggle") as HTMLInputElement | null)
        ?.checked ?? true;
    try {
      const res = await postJSON<{
        ok: boolean;
        tracks?: number;
        moved?: number;
        generic?: number;
        unmatched?: number;
        plan?: { target: string; folder: string }[];
        error?: string;
      }>("/api/supervised/fanout", {
        folder: this._imprintFolder,
        recursive: true,
        apply,
        generic,
      });
      if (!res.ok) {
        this._setImprintNote(res.error || "Fan out failed");
        return;
      }
      const tracks = res.tracks ?? 0;
      const unmatched = res.unmatched ?? 0;
      const genericCount = res.generic ?? 0;
      const parts: string[] = [];
      if (genericCount) parts.push(`${genericCount} into Generic/`);
      if (unmatched) parts.push(`${unmatched} left unmatched`);
      const suffix = parts.length ? ` (${parts.join(", ")})` : "";
      if (apply) {
        this._setImprintNote(`Moved ${res.moved ?? 0} / ${tracks} tracks${suffix}.`);
        toast(`Moved ${res.moved ?? 0} tracks`, "ok");
        this._setUndoAvailable((res.moved ?? 0) > 0);
        this.deps.refreshProjection?.();
        this.deps.refreshClusters?.();
        await this._finishImprint();
      } else {
        const folders = new Set(
          (res.plan ?? [])
            .map((e) => e.target)
            .filter((t) => t.includes("/"))
            .map((t) => t.slice(0, t.lastIndexOf("/"))),
        );
        this._setImprintNote(
          `Planned ${tracks} tracks → ${folders.size} folders${suffix} — Apply moves to run it.`,
        );
        toast(`Planned ${tracks} tracks`, "ok");
      }
    } catch {
      /* api() already toasted */
    }
  }

  /** Drop the ephemeral imprint root once the moves are done. */
  private async _finishImprint(): Promise<void> {
    const rootId = this._imprintRootId;
    this._imprintRootId = null;
    this._imprintFolder = null;
    this._stopImprintPoll();
    if (!rootId) return;
    try {
      await postJSON("/api/supervised/imprint/finish", { root_id: rootId });
    } catch {
      /* api() already toasted */
    }
    void this.deps.onRootsChanged?.();
  }

  /** Reveal the Discard button when a closed browser left an imprint behind. */
  private async _checkStaleImprints(): Promise<void> {
    try {
      const res = await api<{ ok: boolean; roots?: { root_id: string }[] }>(
        "/api/supervised/imprint/temp_roots",
      );
      const roots = res.roots ?? [];
      const $discard = document.getElementById(
        "btn-imprint-discard",
      ) as HTMLButtonElement | null;
      if (!roots.length) return;
      if ($discard) $discard.hidden = false;
      this._setImprintNote(
        `${roots.length} unfinished imprint folder(s) — Discard to clean up.`,
      );
    } catch {
      /* transient; the next reload retries */
    }
  }

  /** Drop every unfinished imprint root without touching the files. */
  private async _discardImprints(): Promise<void> {
    try {
      const res = await postJSON<{ ok: boolean; discarded?: number }>(
        "/api/supervised/imprint/discard",
        {},
      );
      const n = res.discarded ?? 0;
      this._setImprintNote(
        n ? `Discarded ${n} unfinished imprint folder(s).` : "Nothing to discard.",
      );
      toast(n ? `Discarded ${n} imprint folder(s)` : "Nothing to discard", "ok");
      const $discard = document.getElementById(
        "btn-imprint-discard",
      ) as HTMLButtonElement | null;
      if ($discard) $discard.hidden = true;
      void this.deps.onRootsChanged?.();
    } catch {
      /* api() already toasted */
    }
  }

  /* ── Undo moves ─────────────────────────────────────────── */

  /** Enable Undo only when the server has an applied fan-out batch. */
  private async _refreshUndo(): Promise<void> {
    try {
      const res = await api<UndoStatus>("/api/supervised/fanout/undo");
      this._setUndoAvailable(Boolean(res.available));
    } catch {
      /* transient; the next apply/undo updates the button */
    }
  }

  private _setUndoAvailable(available: boolean): void {
    const $undo = document.getElementById(
      "btn-imprint-undo",
    ) as HTMLButtonElement | null;
    if (!$undo) return;
    $undo.disabled = !available;
    $undo.title = available
      ? "Restore the files moved by the last fan-out"
      : "No fan-out moves to undo";
  }

  /** Restore the previous layout from the server's move journal. */
  private async _undoMoves(): Promise<void> {
    try {
      const res = await postJSON<{
        ok: boolean;
        restored?: number;
        total?: number;
        error?: string;
      }>("/api/supervised/fanout/undo", {});
      if (!res.ok) {
        this._setImprintNote(res.error || "Undo failed");
        return;
      }
      this._setImprintNote(
        `Restored ${res.restored ?? 0} / ${res.total ?? 0} tracks.`,
      );
      toast(`Undid ${res.restored ?? 0} moves`, "ok");
      this._setUndoAvailable(false);
      this.deps.refreshProjection?.();
      this.deps.refreshClusters?.();
      void this.deps.onRootsChanged?.();
    } catch {
      /* api() already toasted */
    }
  }

  /* ── Head bundles ───────────────────────────────────────── */

  private _setHeadNote(text: string): void {
    const el = document.getElementById("head-note");
    if (el) el.textContent = text;
  }

  /** Fetch the installed head's metadata and render the info block. */
  async refreshHeadInfo(): Promise<void> {
    try {
      const info = await api<HeadInfo>("/api/supervised/head");
      this._renderHeadInfo(info);
    } catch {
      /* transient; the next train/import refreshes it */
    }
  }

  /** Render one row per head fact, or a "train me" hint when there is none. */
  private _renderHeadInfo(info: HeadInfo): void {
    const host = document.getElementById("head-info");
    if (!host) return;
    host.replaceChildren();
    if (!info.available) {
      const empty = document.createElement("p");
      empty.className = "head-info-empty";
      empty.textContent =
        "No folder head yet — Train to learn your folder structure.";
      host.appendChild(empty);
      return;
    }
    const rows: [string, string][] = [];
    const when = info.imported_at ?? info.trained_at;
    if (when) {
      rows.push([
        info.imported_at ? "Imported" : "Trained",
        new Date(when * 1000).toLocaleString(),
      ]);
    }
    if (info.n_classes != null) rows.push(["Folders", String(info.n_classes)]);
    if (info.input_sources?.length)
      rows.push(["Sources", info.input_sources.join(", ")]);
    if (info.in_dim != null) rows.push(["Input", `${info.in_dim}-D`]);
    if (info.match_percentile != null)
      rows.push(["Match floor", `${Math.round(info.match_percentile)}th pct`]);
    if (info.run_id != null) rows.push(["Run", String(info.run_id)]);
    for (const [label, value] of rows) {
      const line = document.createElement("div");
      line.className = "head-info-row";
      const key = document.createElement("span");
      key.className = "head-info-key";
      key.textContent = label;
      const val = document.createElement("span");
      val.className = "head-info-value";
      val.textContent = value;
      line.append(key, val);
      host.appendChild(line);
    }
  }

  /** Download this library's trained head as a portable bundle. */
  private async _exportHead(): Promise<void> {
    try {
      const blob = await apiBlob("/api/supervised/export", true);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "trackspace_head.pt";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Export failed", "error");
    }
  }

  /** Install an uploaded head, then re-project the library through it. */
  private async _importHead(input: HTMLInputElement): Promise<void> {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    await this._uploadHead(file, false);
  }

  /**
   * Upload a head; on a version mismatch, offer an explicit override.
   *
   * ``silent`` is used so the mismatch message is rendered as a confirm rather
   * than toasted, and any other failure is toasted here.
   */
  private async _uploadHead(file: File, force: boolean): Promise<void> {
    const form = new FormData();
    form.append("bundle", file);
    if (force) form.append("force", "1");
    try {
      const res = await postForm<{
        ok: boolean;
        head?: { run_id?: number; n_classes?: number };
      }>("/api/supervised/import", form, true);
      this._setHeadNote(
        `Imported head: run ${res.head?.run_id ?? "?"}, ` +
          `${res.head?.n_classes ?? "?"} folders.`,
      );
      toast("Imported folder head — re-projecting…", "ok");
      // The imported run id invalidates every cached supervised row, so the map
      // must be refetched (and the next cluster pass re-embeds).
      void this.refreshHeadInfo();
      this.deps.refreshProjection?.();
      this.deps.refreshClusters?.();
      this.deps.refreshPredictions?.();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (!force && message.includes("different encoder versions")) {
        if (confirmAction(`${message}\n\nImport it anyway?`)) {
          await this._uploadHead(file, true);
        }
        return;
      }
      toast(message, "error");
    }
  }

  /** Compare an uploaded head with this one and render the summary. */
  private async _compareHead(input: HTMLInputElement): Promise<void> {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("bundle", file);
    try {
      const res = await postForm<{
        ok: boolean;
        compare?: {
          compatible: boolean;
          shared_classes: string[];
          only_a: string[];
          only_b: string[];
          mean_cosine: number | null;
        };
      }>("/api/supervised/compare", form);
      const c = res.compare;
      if (!c) {
        this._setHeadNote("No comparison returned.");
        return;
      }
      if (!c.compatible) {
        this._setHeadNote("Heads are not comparable (different input space).");
        return;
      }
      const mean = c.mean_cosine == null ? "—" : c.mean_cosine.toFixed(3);
      this._setHeadNote(
        `Shared ${c.shared_classes.length} folders · mean centroid cosine ` +
          `${mean} · ${c.only_a.length} only here, ${c.only_b.length} only there.`,
      );
    } catch {
      /* api() already toasted */
    }
  }
}
