// ─── FolderTreeCoordinator ─────────────────────────────────────
//
// Owns the sidebar folder tree (HTMX partial), root import/removal, folder-row
// interaction, the selected-tracks overlay, and folder highlight/auto-open state.
// Extracted from `Controller` — the tree is self-contained DOM UX whose only
// outward effects are model mutations, the undoable rename command, and a library
// reload after roots change.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type HoverTarget, type Model, type Track } from "../model";
import { toast } from "../lib/toast";
import {
  clusterColor,
  dirColor,
  keyColor,
  setDirAnchors,
  tempoColor,
} from "../lib/colors";
import { trackTempoNorm } from "../lib/lenses";
import {
  clusterTracksByKey,
  clusterTracksByTempo,
  trackCamelotFifths,
} from "../lib/bloom";
import { postJSON } from "../lib/api";
import { confirmAction } from "../lib/util";
import { clearOnNewTimer } from "../lib/async";
import { Modal, buildTabStrip, createKbd, placeAt } from "../lib/dom";
import { displayFolderLeaf, displayFolderPath } from "../lib/path-presenter";
import { hotkeyLabel } from "../hotkeys";
import htmx from "htmx.org";

/** Hotkey-driven canvas actions the folder popout can trigger. */
export type FolderCanvasAction =
  | "focus-folder"
  | "isolate-selection"
  | "fit-view"
  | "exclude-selection";

/** One button in the folder-row action popout. */
interface FolderAction {
  action: string;
  glyph: string;
  label: string;
  hotkey?: string;
  /** Greyed out while the action cannot apply (externals off the map). */
  disabled?: boolean;
  /** Absolute path a `project-here` action projects onto. */
  path?: string;
}

/**
 * Tabs of the selected-tracks overlay, in strip order: the classes the current
 * selection is broken down by. Only tabs with data are shown; folders is always
 * present while anything is selected.
 */
export const OVERLAY_TABS = ["folders", "bpms", "keys", "clusters"] as const;
export type OverlayTab = (typeof OVERLAY_TABS)[number];

/** One row of a selected-tracks overlay tab: a class value and its count. */
interface OverlayItem {
  /** Display text, without the count. */
  text: string;
  /** Hover tooltip. */
  title: string;
  count: number;
  /** Swatch colour, or a gradient across a cluster's class colours. */
  swatch?: string;
  /** The canvas navigation element this row glows while hovered. */
  hover: HoverTarget;
  /** Folder path (folder rows only), for tree-action semantics on hover/click. */
  folderPath?: string;
}

/** One tab of the selected-tracks overlay: a class and its selected members. */
interface OverlayClass {
  id: OverlayTab;
  label: string;
  items: OverlayItem[];
}

export interface FolderTreeDeps {
  model: Model;
  canvas: {
    hoveredIdx: number;
    hoverTarget: HoverTarget | null;
    hoveredBloomTrack: Track | null;
    scheduleDraw(): void;
  };
  /** True while a canvas drag/transform is in progress (hover clearing is suppressed). */
  isPointerBusy(): boolean;
  /** Run the undoable folder-rename command. */
  renameFolder(path: string, newName: string): void;
  /** Re-render the tag panels (deferred after a label-driven selection change). */
  renderTagPanels(): void;
  /** Reload library + tree after a root was added or removed. */
  onRootsChanged(): Promise<void>;
  /** Refetch the map after the external-root layout scope changed. */
  refreshProjection?(): Promise<void> | void;
  /** Prefill the imprint target with a root's absolute path. */
  onImprintRoot?(absPath: string): void;
  /** Run a hotkey-driven canvas action on the currently hovered folder. */
  runCanvasAction?(action: FolderCanvasAction): void;
}

export class FolderTreeCoordinator {
  private $folderTree: HTMLElement;

  /** Passed to `/partials/folder-tree` on next HTMX refresh; cleared after swap. */
  private _pendingRenameForTree: string | null = null;
  /** Anchor path (normalised) for Shift+click folder range select. */
  private _folderShiftAnchor: string | null = null;

  /** Folders the user explicitly toggled open (never auto-closed). */
  private _manuallyOpenedFolders = new Set<string>();
  /** Folders opened automatically because they contain selected tracks. */
  private _autoOpenedFolders = new Set<string>();

  private _rootDragId: string | null = null;
  private _rootDragRow: HTMLElement | null = null;

  /** True while the one-off background heal of missing external positions runs. */
  private _externalHealInFlight = false;

  /** The row whose action popout is open, if any. */
  private _actionsRow: HTMLElement | null = null;
  private _actionsEl!: HTMLElement;
  private _actionsHide = clearOnNewTimer();

  /** Active tab of the selected-tracks overlay; falls back to folders when empty. */
  private _overlayTab: OverlayTab = "folders";

  constructor(private deps: FolderTreeDeps) {
    this.$folderTree = document.getElementById("folder-tree")!;
  }

  /* ── Binding ────────────────────────────────────────────── */

  bind(): void {
    this._bindFolderActions();
    this._bindFolderTree();
    this._bindRootImport();
    this._bindFolderContextMenu();
    this._bindOverlay();
  }

  /* ── Right-click (reuses the row action popout) ────────────── */

  /**
   * Right-clicking a folder inside an External root opens the ordinary row action
   * popout, which carries *Project folder structure onto here* (and, while
   * externals are hidden, *Show external tracks* at the top). There is no separate
   * context-menu system.
   */
  private _bindFolderContextMenu(): void {
    this.$folderTree.addEventListener("contextmenu", (e) => {
      const row = (e.target as HTMLElement).closest<HTMLElement>(".folder-row");
      if (!row || row.classList.contains("folder-row-create")) return;
      if (!this._externalFolderAbsPath(row)) return;
      e.preventDefault();
      this._showFolderActions(row);
    });
  }

  /** Absolute path of a folder row inside an External root, or null. */
  private _externalFolderAbsPath(row: HTMLElement): string | null {
    const section = row.closest(".folder-section-external");
    if (!section) return null;
    const rootRow = section.querySelector<HTMLElement>(
      ".folder-row[data-root-id]",
    );
    const rootAbs = rootRow?.dataset.rootAbs;
    const rootId = rootRow?.dataset.rootId;
    if (!rootAbs || !rootId) return null;
    const vpath = row.dataset.folderPath ?? ".";
    const rest =
      vpath === "." || vpath === rootId
        ? ""
        : vpath.startsWith(rootId + "/")
          ? vpath.slice(rootId.length + 1)
          : "";
    return rest ? `${rootAbs}/${rest}` : rootAbs;
  }

  /* ── Row action popout ──────────────────────────────────── */

  /**
   * A single toolbar that floats to the right of whichever folder row is hovered,
   * replacing the per-row inline icons. Every action the row supports becomes a
   * button; actions that also have a global hotkey carry a `<kbd>` badge.
   */
  private _bindFolderActions(): void {
    let el = document.getElementById("folder-actions");
    if (!el) {
      el = document.createElement("div");
      el.id = "folder-actions";
      el.className = "folder-actions hidden";
      el.setAttribute("role", "toolbar");
      el.setAttribute("aria-label", "Folder actions");
      document.body.appendChild(el);
    }
    this._actionsEl = el;
    el.addEventListener("mouseenter", () => {
      this._cancelHideActions();
      // Moving off the row onto the toolbar must keep its canvas hover alive.
      this._setHoveredRow(this._actionsRow);
    });
    el.addEventListener("mouseleave", () => this._scheduleHideActions());
    el.addEventListener("click", (e) => this._onFolderActionClick(e));
    // The toolbar is pinned to a row's viewport position; scrolling or resizing
    // would strand it, so hide rather than try to track the row.
    window.addEventListener("scroll", () => this._hideActions(), true);
    window.addEventListener("resize", () => this._hideActions());
  }

  private _showFolderActions(row: HTMLElement): void {
    if (row.classList.contains("folder-row-create")) return;
    if (row.dataset.folderPath == null) return;
    this._cancelHideActions();
    this._actionsRow = row;
    this._renderFolderActions(row);
    this._actionsEl.classList.remove("hidden");
    this._positionFolderActions(row);
  }

  private _renderFolderActions(row: HTMLElement): void {
    this._actionsEl.innerHTML = "";
    for (const def of this._folderActionDefs(row)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "folder-action-btn";
      btn.dataset.folderAction = def.action;
      btn.disabled = !!def.disabled;
      if (def.path) btn.dataset.folderActionPath = def.path;
      btn.title = def.label;
      btn.setAttribute("aria-label", def.label);
      const glyph = document.createElement("span");
      glyph.className = "folder-action-glyph";
      glyph.textContent = def.glyph;
      btn.appendChild(glyph);
      const label = document.createElement("span");
      label.className = "folder-action-label";
      label.textContent = def.label;
      btn.appendChild(label);
      if (def.hotkey) {
        btn.appendChild(createKbd(def.hotkey, "folder-action-hotkey"));
      }
      this._actionsEl.appendChild(btn);
    }
  }

  /** Every action a row supports, in display order, with its hotkey badge. */
  private _folderActionDefs(row: HTMLElement): FolderAction[] {
    const path = row.dataset.folderPath ?? ".";
    const isRoot = row.dataset.rootId != null;
    const isExternal = row.dataset.rootRole === "external";
    const isLeaf = row.querySelector("[data-folder-toggle]") == null;
    const projectAbs = this._externalFolderAbsPath(row);
    const hidden = projectAbs != null && !this.deps.model.showExternal;

    const defs: FolderAction[] = [];
    if (hidden) {
      // Externals are off the map: offer to show them first, then grey out
      // everything below (there is nothing to act on while they are hidden).
      defs.push({
        action: "show-external",
        glyph: "◉",
        label: "Show external tracks",
        hotkey: hotkeyLabel("toggle-external"),
      });
    }
    const base: FolderAction[] = [
      { action: "focus-folder", glyph: "◆", label: "Select tracks", hotkey: hotkeyLabel("focus-folder") },
      { action: "isolate-selection", glyph: "◉", label: "Isolate", hotkey: hotkeyLabel("isolate-selection") },
      { action: "fit-view", glyph: "⤢", label: "Fit view", hotkey: hotkeyLabel("fit-view") },
      { action: "exclude-selection", glyph: "⊘", label: "Exclude", hotkey: hotkeyLabel("exclude-selection") },
      { action: "reveal", glyph: "↗", label: "Reveal in Finder" },
    ];
    if (path !== ".") base.push({ action: "rename", glyph: "✎", label: "Rename" });
    if (isLeaf) base.push({ action: "add-subfolder", glyph: "+", label: "New subfolder" });
    if (isRoot) {
      base.push({
        action: "role",
        glyph: isExternal ? "▤" : "◇",
        label: isExternal ? "Move to My library" : "Mark as External",
      });
      if (isExternal)
        base.push({ action: "imprint", glyph: "⇥", label: "Imprint structure…" });
      base.push({ action: "remove", glyph: "✕", label: "Remove root" });
    }
    if (projectAbs) {
      // Any folder inside an External root is a projection destination.
      base.push({
        action: "project-here",
        glyph: "⇥",
        label: "Project folder structure onto here",
        path: projectAbs,
        disabled: hidden,
      });
    }
    if (hidden) for (const d of base) d.disabled = true;
    defs.push(...base);
    return defs;
  }

  private _positionFolderActions(row: HTMLElement): void {
    placeAt(this._actionsEl, row.getBoundingClientRect(), { side: "right" });
  }

  private _onFolderActionClick(e: MouseEvent): void {
    const btn = (e.target as HTMLElement).closest<HTMLElement>("[data-folder-action]");
    if (!btn || !this._actionsRow) return;
    e.stopPropagation();
    const row = this._actionsRow;
    const path = row.dataset.folderPath ?? ".";
    switch (btn.dataset.folderAction) {
      case "focus-folder":
      case "isolate-selection":
      case "fit-view":
      case "exclude-selection":
        this.deps.runCanvasAction?.(btn.dataset.folderAction as FolderCanvasAction);
        break;
      case "reveal":
        void postJSON("/api/folders/reveal", { path: path === "." ? "" : path });
        break;
      case "rename": {
        const label = row.querySelector(".folder-label") as HTMLElement | null;
        if (label) this._beginFolderLabelRename(label);
        break;
      }
      case "add-subfolder":
        void this._createSubfolder(path === "." ? "" : path);
        break;
      case "role":
        void this._setRootRole(
          row.dataset.rootId ?? "",
          row.dataset.rootRole === "external" ? "library" : "external",
        );
        break;
      case "imprint":
        this.deps.onImprintRoot?.(row.dataset.rootAbs ?? "");
        break;
      case "project-here":
        this.deps.onImprintRoot?.(btn.dataset.folderActionPath ?? "");
        break;
      case "show-external":
        this.deps.model.setShowExternal(true);
        this.deps.model.saveLS();
        break;
      case "remove":
        this._confirmRemoveRoot(row);
        break;
    }
    this._hideActions();
  }

  private _scheduleHideActions(): void {
    this._actionsHide.set(() => this._hideActions(), 150);
  }

  private _cancelHideActions(): void {
    this._actionsHide.cancel();
  }

  private _hideActions(): void {
    this._cancelHideActions();
    this._actionsRow = null;
    this._actionsEl?.classList.add("hidden");
  }

  /** Point the canvas hover at a folder row (or clear it when *row* is null). */
  private _setHoveredRow(row: HTMLElement | null): void {
    const path = row?.dataset.folderPath;
    if (row == null || path == null) return;
    this.deps.canvas.hoveredBloomTrack = null;
    this.deps.canvas.hoverTarget = {
      kind: "folder",
      prefix: path === "." ? "" : path,
    };
    this.deps.canvas.scheduleDraw();
  }

  /**
   * Dim the External section while its tracks are off the map. The visibility is
   * owned by the external view-modifier chip now, so this follows every
   * "change" rather than a checkbox this panel owns.
   */
  syncExternalVisibility(): void {
    document
      .getElementById("folder-tree")
      ?.classList.toggle("externals-off", !this.deps.model.showExternal);
  }

  private _bindFolderTree(): void {
    this.$folderTree.addEventListener("dblclick", (e) => {
      const label = (e.target as HTMLElement).closest(".folder-label");
      if (!label) return;
      e.preventDefault();
      e.stopPropagation();
      this._beginFolderLabelRename(label as HTMLElement);
    });
    this.$folderTree.addEventListener("click", (e) =>
      this._onFolderTreeClick(e),
    );
    this.$folderTree.addEventListener("mouseleave", () => {
      if (this.deps.isPointerBusy()) return;
      this.deps.canvas.hoverTarget = null;
      this.deps.canvas.scheduleDraw();
    });
  }

  /** The selected-tracks overlay shares label activation and hover semantics. */
  private _bindOverlay(): void {
    const $overlay = document.getElementById("selected-folders-overlay");
    $overlay?.addEventListener("click", (e) => {
      const tab = (e.target as HTMLElement).closest<HTMLElement>(
        "[data-overlay-tab]",
      );
      if (tab) {
        e.stopPropagation();
        this._overlayTab = tab.dataset.overlayTab as OverlayTab;
        this._renderSelectionOverlay();
        return;
      }
      const label = (e.target as HTMLElement).closest<HTMLElement>(
        ".folder-label[data-path]",
      );
      if (!label) return;
      e.stopPropagation();
      this._onFolderLabelActivate(label, e as MouseEvent);
    });
    $overlay?.addEventListener("mouseleave", () => {
      if (this.deps.isPointerBusy()) return;
      this.deps.canvas.hoverTarget = null;
      this.deps.canvas.scheduleDraw();
    });
  }

  private _afterFolderTreeSwap(): void {
    this._hideActions();
    for (const row of this.$folderTree.querySelectorAll(".folder-row")) {
      row.addEventListener("mouseenter", () => {
        this._setHoveredRow(row as HTMLElement);
        this._showFolderActions(row as HTMLElement);
      });
      row.addEventListener("mouseleave", (e) => {
        if (this.deps.isPointerBusy()) return;
        const to = (e as MouseEvent).relatedTarget as Node | null;
        if (to && this._actionsEl.contains(to)) {
          // Moving onto the popout: keep the row's canvas hover alive.
          this._cancelHideActions();
          return;
        }
        this.deps.canvas.hoverTarget = null;
        this.deps.canvas.scheduleDraw();
        this._scheduleHideActions();
      });
    }

    const start = this.$folderTree.querySelector("[data-start-rename]");
    if (start) {
      start.removeAttribute("data-start-rename");
      this._beginFolderLabelRename(start as HTMLElement);
    }

    // Restore manually-opened folders after the DOM was replaced.
    for (const folderPath of this._manuallyOpenedFolders) {
      this._setFolderOpen(folderPath, true);
    }
    // Auto-opened state must be recomputed from scratch after a swap.
    this._autoOpenedFolders.clear();
    this._syncExternalRoots();
    this._syncDirAnchors();
    this.updateHighlights();
    this.syncActiveLabels();
    this._wireRootRowDrags();
  }

  /** Read root roles from the swapped-in tree and update the model. */
  private _syncExternalRoots(): void {
    const external = new Set<string>();
    for (const row of this.$folderTree.querySelectorAll<HTMLElement>(
      ".folder-row[data-root-role='external']",
    )) {
      const rid = row.dataset.rootId;
      if (rid) external.add(rid);
    }
    this.deps.model.setExternalRoots(external);
    this.syncExternalVisibility();
    this._healExternalPositions();
  }

  /**
   * Read the semantic colour anchors the server embedded in the swapped tree,
   * then repaint: the canvas colours each dot from them, so the map's folders
   * match the sidebar swatches. A missing or malformed block falls back to the
   * structural rule.
   */
  private _syncDirAnchors(): void {
    const el = this.$folderTree.querySelector("#dir-anchors");
    let anchors: Record<string, string> = {};
    if (el?.textContent) {
      try {
        anchors = JSON.parse(el.textContent) as Record<string, string>;
      } catch {
        anchors = {};
      }
    }
    setDirAnchors(anchors);
    this.deps.canvas.scheduleDraw();
  }

  /**
   * Fetch the chart once more if it was built before the tree revealed the
   * external roots: their positions ride along with every projection, so a chart
   * fetched without them has none, and the visibility toggle must not be the thing
   * that pays for them.
   */
  private _healExternalPositions(): void {
    const m = this.deps.model;
    if (this._externalHealInFlight || m.externalRootIds.size === 0) return;
    if (!m.embeddingsReady) return;
    for (const t of m.allTracks) {
      if (!m.isExternalTrack(t) || m.embeddingPositions.has(t.path)) continue;
      this._externalHealInFlight = true;
      Promise.resolve(this.deps.refreshProjection?.()).finally(() => {
        this._externalHealInFlight = false;
      });
      return;
    }
  }

  /* ── Root import / removal ──────────────────────────────── */

  private _bindRootImport(): void {
    const $section = document.getElementById("folder-section");
    const $addRoot = document.getElementById("btn-add-root");
    const $modal = new Modal("root-add-modal");
    const $input = document.getElementById("root-add-input") as HTMLInputElement | null;
    const $external = document.getElementById("root-add-external") as HTMLInputElement | null;
    const $ok = document.getElementById("root-add-ok");
    const $cancel = document.getElementById("root-add-cancel");

    const closeModal = () => {
      $modal.close();
    };
    /** Open the modal; *prefill* (a Finder drop) skips the clipboard guess. */
    const openModal = async (prefill?: string) => {
      $modal.open();
      if ($external) $external.checked = false;
      if ($input) {
        $input.value = prefill ?? "";
        if (prefill) {
          $input.select();
        } else {
          try {
            const clip = (await navigator.clipboard.readText()).trim();
            const firstLine = clip.split(/\r?\n/, 1)[0]?.trim() ?? "";
            const normalized = firstLine.startsWith("file://")
              ? decodeURIComponent(firstLine.replace(/^file:\/\//, ""))
              : firstLine;
            if (normalized.startsWith("/")) {
              $input.value = normalized;
              $input.select();
            }
          } catch {
            // Clipboard access may be blocked; keep the input blank.
          }
        }
        $input.focus();
      }
    };
    const submitModal = () => {
      const raw = $input?.value.trim() ?? "";
      if (!raw) return;
      const role = $external?.checked ? "external" : "library";
      closeModal();
      void this._addRootPath(raw, role);
    };

    if ($addRoot) $addRoot.addEventListener("click", () => void openModal());
    if ($ok) $ok.addEventListener("click", submitModal);
    if ($cancel) $cancel.addEventListener("click", closeModal);
    // Backdrop click-self and a document-capture Esc, so the modal always
    // closes on Esc (not only while its input holds focus).
    $modal.bind(closeModal);
    if ($input) {
      $input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          submitModal();
        }
      });
    }
    if (!$section) return;
    $section.addEventListener("dragover", (e) => {
      const dt = e.dataTransfer;
      if (!dt) return;
      e.preventDefault();
      $section.classList.add("root-drop-active");
    });
    $section.addEventListener("dragleave", (e) => {
      if (e.currentTarget !== e.target) return;
      $section.classList.remove("root-drop-active");
    });
    $section.addEventListener("drop", (e) => {
      e.preventDefault();
      $section.classList.remove("root-drop-active");
      // An internal root-row drag is not a Finder drop: releasing it in place is
      // a cancel, so it must not offer to add a root.
      if (this._rootDragId) {
        this._rootDragId = null;
        this._rootDragRow = null;
        return;
      }
      const dt = e.dataTransfer;
      if (!dt) return;
      const candidates: string[] = [];
      for (const f of Array.from(dt.files || [])) {
        const p = (f as File & { path?: string }).path;
        if (p) candidates.push(p);
      }
      const text = dt.getData("text/plain").trim();
      if (text) candidates.push(text);
      const first = candidates.find((p) => p.startsWith("/"));
      if (!first) {
        toast(
          "Drop a folder from Finder, or paste an absolute path.",
          "warn",
          { x: e.clientX, y: e.clientY },
        );
        return;
      }
      // Confirm the role in the modal rather than silently adding a library root:
      // a dropped folder may be a download/reference root the user wants External.
      void openModal(first);
    });
  }

  private async _addRootPath(path: string, role: string): Promise<void> {
    const res = await postJSON<{ ok: boolean; changed?: boolean; error?: string }>(
      "/api/roots/add",
      { path, role },
    );
    if (!res.ok) {
      toast(res.error || "Failed to add root", "error");
      return;
    }
    await this.deps.onRootsChanged();
    toast(res.changed ? "Root added" : "Root already covered", "ok");
  }

  private _wireRootRowDrags(): void {
    for (const row of this.$folderTree.querySelectorAll<HTMLElement>(".folder-row[data-root-id]")) {
      row.draggable = true;
      row.addEventListener("dragstart", () => {
        const rootId = row.dataset.rootId;
        if (!rootId) return;
        this._rootDragId = rootId;
        this._rootDragRow = row;
        row.classList.add("root-dragging");
        // Reveal the drop zone to the right of the sidebar.
        document.getElementById("root-drop-zone")?.classList.remove("hidden");
      });
      row.addEventListener("dragend", (e) => {
        const rootId = this._rootDragId;
        const rowEl = this._rootDragRow;
        this._rootDragId = null;
        this._rootDragRow = null;
        rowEl?.classList.remove("root-dragging");
        const zone = document.getElementById("root-drop-zone");
        zone?.classList.add("hidden");
        if (!rootId || !rowEl) return;
        const panel = document.getElementById("folder-section");
        const inZone = zone ? this._isInside(e.clientX, e.clientY, zone) : false;
        const inPanel = panel
          ? this._isInside(e.clientX, e.clientY, panel)
          : false;
        // Dropping back in the sidebar is a no-op; the drop zone (or anywhere
        // outside the sidebar) removes the root, after a confirmation.
        if (inPanel && !inZone) return;
        this._confirmRemoveRoot(rowEl);
      });
    }
  }

  private _isInside(x: number, y: number, el: HTMLElement): boolean {
    const r = el.getBoundingClientRect();
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  }

  private async _removeRoot(rootId: string): Promise<void> {
    const res = await postJSON<{ ok: boolean; error?: string }>("/api/roots/remove", { root_id: rootId });
    if (!res.ok) {
      toast(res.error || "Failed to remove root", "error");
      return;
    }
    await this.deps.onRootsChanged();
    toast("Root removed", "ok");
  }

  /** Confirm and drop a root from trackspace (files and cache are kept). */
  private _confirmRemoveRoot(row: HTMLElement): void {
    const rootId = row.dataset.rootId ?? "";
    if (!rootId) return;
    const label = row.dataset.rootAbs || rootId;
    if (
      !confirmAction(
        `Remove "${label}" from trackspace?\n\n` +
          "Its files stay on disk and its cached analysis is kept, so " +
          "re-adding the folder restores it instantly.",
      )
    ) {
      return;
    }
    void this._removeRoot(rootId);
  }

  /** Toggle a root between My library and External (analysed either way). */
  private async _setRootRole(rootId: string, role: string): Promise<void> {
    const res = await postJSON<{ ok: boolean; error?: string }>(
      "/api/roots/role",
      { root_id: rootId, role },
    );
    if (!res.ok) {
      toast(res.error || "Failed to change root role", "error");
      return;
    }
    await this.refresh();
    toast(
      role === "external" ? "Root marked external" : "Root is in My library",
      "ok",
    );
  }

  /* ── Tree interaction ───────────────────────────────────── */

  private _onFolderTreeClick(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    const toggle = t.closest("[data-folder-toggle]");
    if (toggle) {
      e.stopPropagation();
      const row = toggle.closest(".folder-row") as HTMLElement | null;
      const folderPath = row?.dataset.folderPath ?? "";
      const sib = row?.nextElementSibling;
      if (sib?.classList.contains("folder-children")) {
        const open = sib instanceof HTMLElement && sib.style.display !== "none";
        (sib as HTMLElement).style.display = open ? "none" : "block";
        toggle.textContent = open ? "▸" : "▾";
        if (open) {
          this._manuallyOpenedFolders.delete(folderPath);
          this._autoOpenedFolders.delete(folderPath);
        } else {
          this._manuallyOpenedFolders.add(folderPath);
          this._autoOpenedFolders.delete(folderPath);
        }
      }
      return;
    }

    if (t.closest(".folder-add-btn")) {
      e.stopPropagation();
      const btn = t.closest(".folder-add-btn") as HTMLElement;
      const parent = btn.dataset.parent ?? "";
      void this._createSubfolder(parent);
      return;
    }

    const label = t.closest(".folder-label") as HTMLElement | null;
    if (!label) return;
    e.stopPropagation();
    this._onFolderLabelActivate(label as HTMLElement, e);
  }

  private _normFolderPath(dataPath: string): string {
    return dataPath === "." || dataPath === "" ? "" : dataPath;
  }

  private _flatFolderPathsFromDom(): string[] {
    const out: string[] = [];
    for (const row of this.$folderTree.querySelectorAll<HTMLElement>(
      ".folder-row",
    )) {
      out.push(row.dataset.folderPath ?? ".");
    }
    return out;
  }

  private _folderPathsInRangeFromDom(aPath: string, bPath: string): Set<string> {
    const flat = this._flatFolderPathsFromDom();
    let ia = flat.indexOf(aPath);
    let ib = flat.indexOf(bPath);
    if (ia < 0) ia = 0;
    if (ib < 0) ib = 0;
    const lo = Math.min(ia, ib);
    const hi = Math.max(ia, ib);
    return new Set(
      flat.slice(lo, hi + 1).map((p) => this._normFolderPath(p)),
    );
  }

  /** Apply the `active` class to labels matching the current folder. */
  syncActiveLabels(): void {
    const activeNorm = new Set<string>([this.deps.model.folder]);
    for (const label of document.querySelectorAll<HTMLElement>(
      "#folder-tree .folder-label[data-path], #selected-folders-overlay .folder-label[data-path]",
    )) {
      const norm = this._normFolderPath(label.dataset.path ?? ".");
      label.classList.toggle("active", activeNorm.has(norm));
    }
    this._renderSelectionOverlay();
  }

  private _onFolderLabelActivate(label: HTMLElement, e: MouseEvent): void {
    const m = this.deps.model;
    const path = label.dataset.path ?? ".";
    const norm = this._normFolderPath(path);
    const anchorPath = this._folderShiftAnchor;

    if (e.altKey) {
      e.preventDefault();
      m.hideFolderTree(norm);
      m.saveLS();
      return;
    }

    if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      m.folder = norm;
      m.selectInFolder(path);
      m.saveLS();
      this._folderShiftAnchor = norm;
      setTimeout(() => {
        this.deps.renderTagPanels();
      }, 0);
      return;
    }

    if (e.shiftKey && anchorPath !== null) {
      e.preventDefault();
      const anchorData =
        anchorPath === "" ? "." : anchorPath;
      const range = this._folderPathsInRangeFromDom(anchorData, path);
      m.folder = norm;
      m.addTracksUnderFolderPrefixes(range);
      this._folderShiftAnchor = norm;
      m.saveLS();
      setTimeout(() => this.deps.renderTagPanels(), 0);
      return;
    }

    this._folderShiftAnchor = norm;
    m.folder = norm;
    m.selectOnlyInFolder(path);
    m.saveLS();
  }

  private _beginFolderLabelRename(labelEl: HTMLElement): void {
    const path = labelEl.dataset.path ?? ".";
    const oldName = labelEl.textContent ?? "";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "folder-rename-input";
    input.value = oldName;

    let done = false;
    labelEl.replaceWith(input);
    input.select();
    input.focus();

    const commit = () => {
      if (done) return;
      done = true;
      const newName = input.value.trim();
      if (!newName || newName === oldName) {
        input.replaceWith(labelEl);
        return;
      }
      this.deps.renameFolder(path, newName);
    };
    const cancel = () => {
      if (done) return;
      done = true;
      input.replaceWith(labelEl);
    };
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        commit();
      }
      if (ev.key === "Escape") {
        ev.preventDefault();
        cancel();
      }
    });
    input.addEventListener("blur", () => commit());
  }

  /* ── HTMX refresh ──────────────────────────────────────── */

  /** Re-fetch the folder-tree partial and rebind after the swap. */
  async refresh(): Promise<void> {
    const pr = this._pendingRenameForTree ?? "";
    this._pendingRenameForTree = null;
    const m = this.deps.model;
    const params = new URLSearchParams({ pending_rename: pr });
    params.set("active", m.folder === "" ? "." : m.folder);
    params.append("active_folders", m.folder === "" ? "." : m.folder);
    try {
      await htmx.ajax("get", `/partials/folder-tree?${params}`, {
        target: "#folder-tree",
        swap: "innerHTML",
      });
    } catch {
      /* network / server error */
    }
    this._afterFolderTreeSwap();
  }

  private async _createSubfolder(parentPath: string): Promise<void> {
    let name = "New folder",
      n = 1;
    while (true) {
      try {
        const res = await postJSON<{
          ok: boolean;
          path: string;
          error?: string;
        }>("/api/folders/create", { parent: parentPath, name });
        if (res.ok) {
          this._pendingRenameForTree = res.path;
          await this.refresh();
          return;
        }
        if (res.error === "Already exists") {
          n++;
          name = `New folder ${n}`;
        } else {
          toast(res.error || "Failed to create folder", "error");
          return;
        }
      } catch {
        return;
      }
    }
  }

  /* ── Folder open/close helpers ──────────────────────────── */

  private _setFolderOpen(folderPath: string, open: boolean): void {
    const $tree = this.$folderTree;
    const row = $tree.querySelector<HTMLElement>(
      `.folder-row[data-folder-path="${CSS.escape(folderPath)}"]`,
    );
    const sib = row?.nextElementSibling;
    if (!sib?.classList.contains("folder-children")) return;
    (sib as HTMLElement).style.display = open ? "block" : "none";
    const arrow = row?.querySelector<HTMLElement>(
      "[data-folder-toggle]",
    );
    if (arrow) arrow.textContent = open ? "▾" : "▸";
  }

  /**
   * Expand *folderPath* and its ancestors in the tree so the focused folder is
   * visible in the left browser. The rows count as manually opened, so the
   * selection-driven auto-close leaves them alone. Used by **F** when it
   * focuses a folder.
   */
  revealFolder(folderPath: string): void {
    const chain: string[] = [];
    if (folderPath) {
      const parts = folderPath.split("/");
      let acc = "";
      for (const p of parts) {
        acc = acc ? `${acc}/${p}` : p;
        chain.push(acc);
      }
    }
    for (const path of chain) {
      if (this._manuallyOpenedFolders.has(path)) continue;
      this._setFolderOpen(path, true);
      this._manuallyOpenedFolders.add(path);
      this._autoOpenedFolders.delete(path);
    }
  }

  /** Returns the set of data-folder-path values (including ancestors) that
   *  contain at least one selected or hovered track. Root is represented
   *  as ".". */
  private _getFoldersWithSelection(): Set<string> {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const folders = new Set<string>();

    const addFolderAndAncestors = (folder: string) => {
      folders.add(".");
      if (folder) {
        const parts = folder.split("/");
        for (let i = 1; i <= parts.length; i++) {
          folders.add(parts.slice(0, i).join("/"));
        }
      }
    };

    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t) addFolderAndAncestors(t.folder ?? "");
    }

    if (cv.hoveredIdx >= 0) {
      const t = m.tracks[cv.hoveredIdx];
      if (t) addFolderAndAncestors(t.folder ?? "");
    }

    return folders;
  }

  /** Apply glow class and auto-open / auto-close folder rows based on the
   *  current selection.  Called after every "change" event and after HTMX
   *  tree swaps. */
  updateHighlights(): void {
    const m = this.deps.model;
    const $tree = this.$folderTree;
    const foldersWithSel = this._getFoldersWithSelection();
    const hoveredTrack =
      this.deps.canvas.hoveredIdx >= 0
        ? m.tracks[this.deps.canvas.hoveredIdx]
        : null;
    const hoveredLeafPath =
      hoveredTrack != null ? (hoveredTrack.folder || ".") : null;

    // Update glow class on every folder row in tree and selected-folders overlay.
    for (const row of document.querySelectorAll<HTMLElement>(
      "#folder-tree .folder-row, #selected-folders-overlay .folder-row",
    )) {
      if (row.classList.contains("folder-row-create")) continue;
      const path = row.dataset.folderPath;
      if (!path) continue;
      const inOverlay = !!row.closest("#selected-folders-overlay");
      row.classList.toggle("folder-sel", !inOverlay && foldersWithSel.has(path));
      row.classList.toggle(
        "folder-hover-track",
        inOverlay && hoveredLeafPath != null && path === hoveredLeafPath,
      );
      const norm = path === "." ? "" : path;
      row.classList.toggle(
        "folder-fully-excluded",
        m.hasExclusions && !m.hasVisibleTracksUnderTreePrefix(norm),
      );
    }

    // Close auto-opened folders that no longer contain selected tracks.
    for (const folderPath of [...this._autoOpenedFolders]) {
      if (!foldersWithSel.has(folderPath)) {
        this._setFolderOpen(folderPath, false);
        this._autoOpenedFolders.delete(folderPath);
      }
    }

    // Open folders that have selected tracks, haven't been manually opened,
    // and are currently closed.
    for (const folderPath of foldersWithSel) {
      if (folderPath === ".") continue; // root is always visible
      if (this._manuallyOpenedFolders.has(folderPath)) continue;
      if (this._autoOpenedFolders.has(folderPath)) continue;
      const row = $tree.querySelector<HTMLElement>(
        `.folder-row[data-folder-path="${CSS.escape(folderPath)}"]`,
      );
      const sib = row?.nextElementSibling;
      if (!sib?.classList.contains("folder-children")) continue;
      const isOpen = (sib as HTMLElement).style.display !== "none";
      if (!isOpen) {
        this._setFolderOpen(folderPath, true);
        this._autoOpenedFolders.add(folderPath);
      }
    }
  }

  /**
   * Render the selected-tracks overlay: a tab strip plus the active class's rows.
   * Folders group by folder path, BPMs by rounded tempo, keys by musical key,
   * clusters by assignment. Only classes present in the selection get a tab, so an
   * unanalysed selection shows folders alone.
   */
  private _renderSelectionOverlay(): void {
    const host = document.getElementById("selected-folders-overlay");
    if (!host) return;
    const classes = this._overlayClasses();
    if (classes.length === 0) {
      host.classList.add("hidden");
      host.innerHTML = "";
      return;
    }
    // A class can vanish when the selection changes; fall back to folders.
    if (!classes.some((c) => c.id === this._overlayTab))
      this._overlayTab = "folders";
    const active = classes.find((c) => c.id === this._overlayTab) ?? classes[0];
    host.innerHTML = "";
    const title = document.createElement("div");
    title.className = "selected-folders-title";
    title.textContent = "Selected";
    host.appendChild(title);
    if (classes.length > 1) host.appendChild(this._buildOverlayTabs(classes));
    for (const item of active.items)
      host.appendChild(this._buildOverlayRow(item, active.id));
    host.classList.remove("hidden");
  }

  /** Count the selection by each class, then build one tab per non-empty class. */
  private _overlayClasses(): OverlayClass[] {
    const m = this.deps.model;
    const selectedTracks: Track[] = [];
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t) selectedTracks.push(t);
    }
    if (selectedTracks.length === 0) return [];
    const folderCounts = new Map<string, number>();
    const clusterCounts = new Map<number, number>();
    for (const t of selectedTracks) {
      const folder = t.folder ?? "";
      folderCounts.set(folder, (folderCounts.get(folder) ?? 0) + 1);
      const cid = m.clusterAssignments.get(t.path);
      if (cid !== undefined) clusterCounts.set(cid, (clusterCounts.get(cid) ?? 0) + 1);
    }
    const classes: OverlayClass[] = [
      {
        id: "folders",
        label: "Folders",
        items: this._sortedCounts(folderCounts).map(([folder, count]) => ({
          text: displayFolderLeaf(folder),
          title: displayFolderPath(folder),
          count,
          swatch: dirColor(folder),
          hover: { kind: "folder", prefix: folder },
          folderPath: folder === "" ? "." : folder,
        })),
      },
    ];
    // BPMs and keys are clustered at the current bloom range, so their rows merge
    // the way the hover bloom would — and each row's swatch spans its members'
    // class colours.
    const tempoClusters = clusterTracksByTempo(
      selectedTracks,
      m.bloomRange("tempo"),
    );
    if (tempoClusters.length > 0)
      classes.push({
        id: "bpms",
        label: "BPMs",
        items: this._sortedClusters(tempoClusters, (t) =>
          Math.round(t.bpm as number),
        ).map((cluster) => this._tempoItem(cluster)),
      });
    const keyClusters = clusterTracksByKey(selectedTracks, m.bloomRange("key"));
    if (keyClusters.length > 0)
      classes.push({
        id: "keys",
        label: "Keys",
        items: this._sortedClusters(
          keyClusters,
          (t) => trackCamelotFifths(t) ?? 0,
        ).map((cluster) => this._keyItem(cluster)),
      });
    if (clusterCounts.size > 0)
      classes.push({
        id: "clusters",
        label: "Clusters",
        items: this._sortedCounts(clusterCounts).map(([cid, count]) => {
          const name = m.clusterNames.get(cid) || `Cluster ${cid}`;
          return {
            text: name,
            title: name,
            count,
            swatch: clusterColor(cid),
            hover: { kind: "cluster", id: cid },
          };
        }),
      });
    return classes;
  }

  /** One BPM cluster row: a tempo range, a gradient swatch, and a path-set target. */
  private _tempoItem(cluster: Track[]): OverlayItem {
    const byBpm = new Map<number, string>();
    for (const t of cluster) {
      const bpm = Math.round(t.bpm as number);
      if (!byBpm.has(bpm)) byBpm.set(bpm, tempoColor(trackTempoNorm(t) as number));
    }
    const sorted = [...byBpm.entries()].sort((a, b) => a[0] - b[0]);
    const lo = sorted[0][0];
    const hi = sorted[sorted.length - 1][0];
    const colours = sorted.map(([, c]) => c);
    const label = lo === hi ? `${lo} BPM` : `${lo}–${hi} BPM`;
    return {
      text: label,
      title:
        lo === hi ? `${lo} BPM` : `Tracks from ${lo} to ${hi} BPM`,
      count: cluster.length,
      swatch: this._classSwatch(colours),
      hover: {
        kind: "paths",
        paths: cluster.map((t) => t.path),
        color: colours[0],
        label,
      },
    };
  }

  /** One key cluster row: the commonest key, a gradient swatch, and a path set. */
  private _keyItem(cluster: Track[]): OverlayItem {
    const nameCounts = new Map<string, number>();
    const byFifth = new Map<number, string>();
    for (const t of cluster) {
      if (t.musical_key)
        nameCounts.set(t.musical_key, (nameCounts.get(t.musical_key) ?? 0) + 1);
      const f = trackCamelotFifths(t);
      const af = t.audio_features;
      if (f !== null && af && !byFifth.has(f))
        byFifth.set(f, keyColor(af.key, af.mode));
    }
    const names = [...nameCounts.entries()].sort((a, b) =>
      b[1] !== a[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : 1,
    );
    const colours = [...byFifth.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, c]) => c);
    const label = names[0]?.[0] ?? "—";
    return {
      text: label,
      title: names.map(([n]) => n).join(", ") || "No key",
      count: cluster.length,
      swatch: this._classSwatch(colours),
      hover: {
        kind: "paths",
        paths: cluster.map((t) => t.path),
        color: colours[0],
        label,
      },
    };
  }

  /** A solid colour for one class, or a gradient spanning a cluster's classes. */
  private _classSwatch(colours: string[]): string | undefined {
    if (colours.length === 0) return undefined;
    if (colours.length === 1) return colours[0];
    return `linear-gradient(90deg, ${colours.join(", ")})`;
  }

  /** Clusters sorted by size (desc), then by their lowest sort key. */
  private _sortedClusters(
    clusters: Track[][],
    key: (t: Track) => number,
  ): Track[][] {
    return [...clusters].sort((a, b) => {
      if (b.length !== a.length) return b.length - a.length;
      return Math.min(...a.map(key)) - Math.min(...b.map(key));
    });
  }

  /** Counts sorted by frequency (desc), then by key. */
  private _sortedCounts<K extends string | number>(
    counts: Map<K, number>,
  ): [K, number][] {
    return [...counts.entries()].sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
    });
  }

  private _buildOverlayTabs(classes: OverlayClass[]): HTMLElement {
    return buildTabStrip(
      "overlay-tabs",
      classes.map((c) => ({ id: c.id, label: c.label })),
      this._overlayTab,
      { dataKey: "overlayTab", buttonClass: "overlay-tab" },
    ).el;
  }

  /**
   * One overlay row. Folder rows keep the tree's label semantics (folder-actions
   * popout, label activation); every other class row is a plain navigation target:
   * hover glows its class, click selects its tracks.
   */
  private _buildOverlayRow(item: OverlayItem, tab: OverlayTab): HTMLElement {
    const isFolder = tab === "folders";
    const row = document.createElement("div");
    row.className = "folder-row selected-folder-row";
    if (isFolder) row.dataset.folderPath = item.folderPath ?? ".";
    row.addEventListener("mouseenter", () => {
      row.classList.add("folder-hover");
      if (isFolder) {
        this._setHoveredRow(row);
        this._showFolderActions(row);
        return;
      }
      this.deps.canvas.hoveredBloomTrack = null;
      this.deps.canvas.hoverTarget = item.hover;
      this.deps.canvas.scheduleDraw();
    });
    row.addEventListener("mouseleave", (e) => {
      row.classList.remove("folder-hover");
      if (this.deps.isPointerBusy()) return;
      const to = (e as MouseEvent).relatedTarget as Node | null;
      if (isFolder && to && this._actionsEl.contains(to)) {
        this._cancelHideActions();
        return;
      }
      this.deps.canvas.hoverTarget = null;
      this.deps.canvas.scheduleDraw();
      if (isFolder) this._scheduleHideActions();
    });
    if (!isFolder) {
      row.addEventListener("click", (e) => {
        e.stopPropagation();
        this.deps.model.selectOnlyHoverTarget(item.hover);
        this.deps.model.saveLS();
      });
    }
    const arrow = document.createElement("span");
    arrow.className = "folder-arrow";
    arrow.textContent = "\u2003";
    row.appendChild(arrow);
    if (item.swatch) {
      const swatch = document.createElement("span");
      swatch.className = "folder-swatch";
      swatch.style.background = item.swatch;
      row.appendChild(swatch);
    }
    const label = document.createElement("span");
    label.className = "folder-label";
    if (isFolder) {
      const norm = this._normFolderPath(item.folderPath ?? ".");
      if (this.deps.model.folder === norm) label.classList.add("active");
      label.dataset.path = item.folderPath ?? ".";
    }
    label.title = item.title;
    label.textContent = `${item.text} (${item.count})`;
    row.appendChild(label);
    return row;
  }
}
