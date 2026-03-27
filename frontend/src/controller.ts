import type { Model, Track, ProjectionMethod } from "./model";
import type {
  CanvasView,
  TagPanelView,
  PropertiesView,
  BatchView,
  StatusView,
  TxHandle,
} from "./components";
import { toast, showLoad, hideLoad, dirColor } from "./lib/toast";
import { displayFolderLeaf, displayFolderPath } from "./lib/path-presenter";
import {
  CommandManager,
  RenameTagCommand,
  RenameFolderCommand,
  MoveTracksCommand,
} from "./command";
import { HotkeyManager } from "./hotkeys";
import htmx from "htmx.org";

// ─── Constants ───────────────────────────────────────────────
const DRAG_THRESH = 4;
const ZOOM_FACTOR = 1.12;
const FLUSH_MS = 500;

// ─── Utilities ───────────────────────────────────────────────
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

function debounce<Args extends unknown[]>(
  fn: (...args: Args) => void,
  ms: number,
): (...args: Args) => void {
  let t: number;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

function pointInPoly(
  px: number,
  py: number,
  poly: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i],
      [xj, yj] = poly[j];
    if (
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    )
      inside = !inside;
  }
  return inside;
}

// ─── API helpers ─────────────────────────────────────────────
async function api<T = unknown>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const t = `${res.status} ${res.statusText}`;
    toast(t, "error");
    throw new Error(t);
  }
  return res.json();
}

function postJSON<T = unknown>(url: string, body: unknown): Promise<T> {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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

// ─── Internal types ──────────────────────────────────────────
interface DragSnap {
  track: Track;
  x: number;
  y: number;
}

interface TxFormSnap {
  wx: number;
  wy: number;
  track: Track;
}

interface TxBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface PendingUpdate {
  path: string;
  tag: string;
  value: number;
}

/** Subset of ``/api/embeddings/status`` used to avoid duplicate layout fetches. */
type EmbeddingLayoutStatusHint = {
  layout_revision?: string;
};

const LAYOUT_CACHE_MAX = 8;

interface CachedLayout {
  revision: string;
  positions: Map<string, { x: number; y: number }>;
}

type MouseMode =
  | "idle"
  | "pan"
  | "lasso"
  | "boxsel"
  | "pending"
  | "drag"
  | "txform";

interface MouseState {
  mode: MouseMode;
  sx: number;
  sy: number;
  lx: number;
  ly: number;
  hitIdx: number;
  /** Was the hit track already in the selection when mousedown fired? */
  hitWasSelected: boolean;
  /** Did mousedown land inside the selection bounding box (but not on a dot)? */
  hitInSelectionBox: boolean;
  snap: Map<string, DragSnap> | null;
  draggingOutside: boolean;
  highlightedFolder: HTMLElement | null;
  txHandle: string | null;
  txSnap: Map<string, TxFormSnap> | null;
  txBounds: TxBounds | null;
  txStart: { wx: number; wy: number } | null;
}

// ═════════════════════════════════════════════════════════════
//  Controller
// ═════════════════════════════════════════════════════════════
export class Controller {
  private model: Model;
  private canvas: CanvasView;
  private $folderTree: HTMLElement;
  private tagPanel: TagPanelView;
  private props: PropertiesView;
  private batch: BatchView;
  private status: StatusView;

  /** Passed to `/partials/folder-tree` on next HTMX refresh; cleared after swap. */
  pendingRenameForTree: string | null = null;
  /** Anchor path (normalised) for Shift+click folder range select. */
  private _folderShiftAnchor: string | null = null;

  private mouse: MouseState;
  private previewPath: string | null = null;
  private $audio: HTMLAudioElement;
  private pending = new Map<string, PendingUpdate>();
  private cmdMgr: CommandManager;
  readonly hotkeyMgr = new HotkeyManager();

  /** Folders the user explicitly toggled open (never auto-closed). */
  private _manuallyOpenedFolders = new Set<string>();
  /** Folders opened automatically because they contain selected tracks. */
  private _autoOpenedFolders = new Set<string>();
  /** True while the pointer is inside the viewport canvas (for hover resync). */
  private _pointerOverCanvas = false;
  private _rootDragId: string | null = null;
  private _rootDragRow: HTMLElement | null = null;

  constructor(
    model: Model,
    canvasView: CanvasView,
    tagPanelView: TagPanelView,
    propsView: PropertiesView,
    batchView: BatchView,
    statusView: StatusView,
  ) {
    this.model = model;
    this.canvas = canvasView;
    this.$folderTree = document.getElementById("folder-tree")!;
    this.tagPanel = tagPanelView;
    this.props = propsView;
    this.batch = batchView;
    this.status = statusView;

    this.cmdMgr = new CommandManager((effect) => {
      if (effect === "tags-dirty") {
        model.emit("tags-dirty");
      } else {
        model.emit("change");
      }
      if (effect === "change+htmx") {
        void this._refreshFolderTreeHtmx();
      }
      model.saveLS();
    });

    this.mouse = {
      mode: "idle",
      sx: 0,
      sy: 0,
      lx: 0,
      ly: 0,
      hitIdx: -1,
      hitWasSelected: false,
      hitInSelectionBox: false,
      snap: null,
      draggingOutside: false,
      highlightedFolder: null,
      txHandle: null,
      txSnap: null,
      txBounds: null,
      txStart: null,
    };

    this.$audio = document.getElementById("preview-audio") as HTMLAudioElement;

    this._wireModel();
    this._wireViewCallbacks();
    this._bindFolderTree();
    this._bindRootImport();
    this._bindCanvas();
    this._bindSidebar();
    this._bindKeyboard();
    this._bindResize();
    this._bindViewportChrome();
    this._syncModeVisuals();
  }

  /* ── Model events ───────────────────────────────────────── */

  private _wireModel(): void {
    this.model.on("change", () => {
      this.canvas.scheduleDraw();
      this.status.update(this.model);
      this.props.render();
      this.batch.render();
      this.tagPanel.render();
      this._syncModeVisuals();
      this._updateFolderHighlights();
      this._syncFolderTreeActiveLabels();
      this._updateClearHidesButton();
      // ``m.tracks`` is rebuilt when hides/filters change; ``hoveredIdx`` must be
      // re-derived from the pointer after the next paint or hotkeys see a stale index.
      requestAnimationFrame(() => this._resyncCanvasHoverFromLastPointer());
    });
    this.model.on("tags-dirty", () => this.batch.renderDirty());
  }

  /** Sync all sidebar controls to match current model state. */
  private _syncModeVisuals(): void {
    const m = this.model;
    const isEmbed = m.viewMode === "embeddings";

    // Dim embedding-only controls when in tag mode
    const embedEls = [
      document.getElementById("projection-method"),
      document.getElementById("embedding-sources"),
      document.getElementById("folder-tune-controls"),
      document.getElementById("scale-tags-toggle")?.closest("label"),
      document.getElementById("scale-folders-toggle")?.closest("label"),
    ];
    for (const el of embedEls) {
      el?.classList.toggle("mode-dimmed", !isEmbed);
    }

    // View mode checkbox
    const $vm = document.getElementById("view-mode-toggle") as HTMLInputElement | null;
    if ($vm) $vm.checked = isEmbed;

    // Projection toggle active state
    const $toggle = document.getElementById("projection-toggle");
    if ($toggle) {
      for (const btn of $toggle.querySelectorAll<HTMLElement>(".toggle-btn")) {
        btn.classList.toggle("active", btn.dataset.method === m.projectionMethod);
      }
    }

    // Source checkboxes
    const srcMap: [string, boolean][] = [
      ["source-clap", m.useCLAP],
      ["source-effnet", m.useEffNet],
      ["source-feat-tempo", m.useAudioFeatureTempo],
      ["source-feat-key", m.useAudioFeatureKey],
      ["source-feat-mode", m.useAudioFeatureMode],
      ["source-feat-energy", m.useAudioFeatureEnergy],
      ["source-feat-dance", m.useAudioFeatureDance],
    ];
    for (const [id, val] of srcMap) {
      const $cb = document.getElementById(id) as HTMLInputElement | null;
      if ($cb) $cb.checked = val;
    }

    const $fb = document.getElementById("folder-boost-range") as HTMLInputElement | null;
    const $fdb = document.getElementById("folder-depth-boost-range") as HTMLInputElement | null;
    if ($fb) $fb.value = String(m.folderContrastBoost);
    if ($fdb) $fdb.value = String(m.folderDepthBoost);

    // Scaling checkboxes
    const $st = document.getElementById("scale-tags-toggle") as HTMLInputElement | null;
    const $sf = document.getElementById("scale-folders-toggle") as HTMLInputElement | null;
    if ($st) $st.checked = m.scaleByTags;
    if ($sf) $sf.checked = m.scaleByFolders;
  }

  /* ── View callbacks ─────────────────────────────────────── */

  private _wireViewCallbacks(): void {
    const m = this.model;

    this.tagPanel.onAxisChange = (which, tag) => {
      if (m.viewMode === "tags") {
        this._animateAxisChange(which as "axisX" | "axisY", tag);
      } else {
        m.toggleAxis(which as "axisX" | "axisY", tag);
        void this._switchToMode("tags");
      }
      this.tagPanel.render();
      m.saveLS();
    };

    this.tagPanel.onFilterChange = (tag, range) => {
      m.setFilterRange(tag, range);
      m.saveLS();
    };

    this.tagPanel.onTagRename = (oldName, newName) => {
      void this.cmdMgr.run(new RenameTagCommand(oldName, newName), m);
    };

    this.props.onDeselect = (path) => {
      m.selected.delete(path);
      m.emit("change");
    };
    this.props.onRestore = (path) => {
      m.selected.add(path);
      m.emit("change");
    };
    this.props.onFocusRange = (keepPaths) => {
      for (const p of [...m.selected])
        if (!keepPaths.has(p)) m.selected.delete(p);
      m.emit("change");
    };
    this.props.onHoverTrack = (path) => {
      if (!m.hoverPreview) return;
      const t = m.trackByPath(path);
      if (t) this._startPreview(t);
    };
    this.props.onHoverEnd = () => {
      this._stopPreview();
    };

    this.batch.onApply = (tag, updates) => {
      for (const [path, value] of updates) {
        const t = m.trackByPath(path);
        if (t) {
          t.tags[tag] = value;
          this.pending.set(`${path}|${tag}`, { path, tag, value });
        }
      }
      this.canvas.scheduleDraw();
      this._flushPending();
    };

    this.batch.onDragEnd = () => {
      m.emit("change");
    };
  }

  /* ── Folder tree (HTMX + server partial) ────────────────── */

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
      const fromDrag =
        this.mouse.mode === "drag" || this.mouse.mode === "txform";
      if (fromDrag) return;
      this.canvas.hoveredFolderPrefix = null;
      this.canvas.scheduleDraw();
    });
  }

  private _afterFolderTreeSwap(): void {
    for (const row of this.$folderTree.querySelectorAll(".folder-row")) {
      row.addEventListener("mouseenter", () => {
        const path = row.getAttribute("data-folder-path");
        if (path == null) return;
        this.canvas.hoveredSiblingFolder = null;
        this.canvas.hoveredFolderPrefix = path === "." ? "" : path;
        this.canvas.scheduleDraw();
      });
      row.addEventListener("mouseleave", () => {
        const fromDrag =
          this.mouse.mode === "drag" || this.mouse.mode === "txform";
        if (fromDrag) return;
        this.canvas.hoveredFolderPrefix = null;
        this.canvas.scheduleDraw();
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
    this._updateFolderHighlights();
    this._syncFolderTreeActiveLabels();
    this._wireRootRowDrags();
  }

  private _bindRootImport(): void {
    const $section = document.getElementById("folder-section");
    const $addRoot = document.getElementById("btn-add-root");
    const $modal = document.getElementById("root-add-modal");
    const $input = document.getElementById("root-add-input") as HTMLInputElement | null;
    const $ok = document.getElementById("root-add-ok");
    const $cancel = document.getElementById("root-add-cancel");

    const closeModal = () => {
      if (!$modal) return;
      $modal.classList.add("hidden");
    };
    const openModal = async () => {
      if (!$modal) return;
      $modal.classList.remove("hidden");
      if ($input) {
        $input.value = "";
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
        $input.focus();
      }
    };
    const submitModal = () => {
      const raw = $input?.value.trim() ?? "";
      if (!raw) return;
      closeModal();
      void this._addRootPath(raw);
    };

    if ($addRoot) $addRoot.addEventListener("click", openModal);
    if ($ok) $ok.addEventListener("click", submitModal);
    if ($cancel) $cancel.addEventListener("click", closeModal);
    if ($modal) {
      $modal.addEventListener("click", (e) => {
        if (e.target === $modal) closeModal();
      });
    }
    if ($input) {
      $input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          submitModal();
        } else if (e.key === "Escape") {
          e.preventDefault();
          closeModal();
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
        toast("Drop a folder from Finder, or paste an absolute path.", "warn");
        return;
      }
      void this._addRootPath(first);
    });
  }

  private async _addRootPath(path: string): Promise<void> {
    const res = await postJSON<{ ok: boolean; changed?: boolean; error?: string }>(
      "/api/roots/add",
      { path },
    );
    if (!res.ok) {
      toast(res.error || "Failed to add root", "error");
      return;
    }
    await this._reloadLibraryAndTree();
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
      });
      row.addEventListener("dragend", (e) => {
        const rootId = this._rootDragId;
        const rowEl = this._rootDragRow;
        this._rootDragId = null;
        this._rootDragRow = null;
        rowEl?.classList.remove("root-dragging");
        const panel = document.getElementById("folder-section");
        const underPanel = panel
          ? e.clientX >= panel.getBoundingClientRect().left &&
            e.clientX <= panel.getBoundingClientRect().right &&
            e.clientY >= panel.getBoundingClientRect().top &&
            e.clientY <= panel.getBoundingClientRect().bottom
          : false;
        if (!rootId || underPanel) return;
        void this._removeRoot(rootId);
      });
    }
  }

  private async _removeRoot(rootId: string): Promise<void> {
    const res = await postJSON<{ ok: boolean; error?: string }>("/api/roots/remove", { root_id: rootId });
    if (!res.ok) {
      toast(res.error || "Failed to remove root", "error");
      return;
    }
    await this._reloadLibraryAndTree();
    toast("Root removed", "ok");
  }

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

    if (t.closest(".folder-reveal-btn")) {
      e.stopPropagation();
      const btn = t.closest(".folder-reveal-btn") as HTMLElement;
      const path = btn.dataset.path ?? ".";
      void postJSON("/api/folders/reveal", { path: path === "." ? "" : path });
      return;
    }

    if (t.closest(".folder-rename-btn")) {
      e.stopPropagation();
      const row = t.closest(".folder-row") as HTMLElement;
      const label = row.querySelector(".folder-label") as HTMLElement | null;
      if (label) this._beginFolderLabelRename(label);
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

  private _syncFolderTreeActiveLabels(): void {
    const activeNorm = new Set<string>([this.model.folder]);
    for (const label of document.querySelectorAll<HTMLElement>(
      "#folder-tree .folder-label, #selected-folders-overlay .folder-label",
    )) {
      const norm = this._normFolderPath(label.dataset.path ?? ".");
      label.classList.toggle("active", activeNorm.has(norm));
    }
    this._renderSelectedLeafFoldersOverlay();
  }

  private _updateClearHidesButton(): void {
    const btn = document.getElementById("viewport-clear-hides");
    if (!btn) return;
    btn.classList.toggle("hidden", !this.model.hasExclusions);
  }

  private _onFolderLabelActivate(label: HTMLElement, e: MouseEvent): void {
    const path = label.dataset.path ?? ".";
    const norm = this._normFolderPath(path);
    const anchorPath = this._folderShiftAnchor;

    if (e.altKey) {
      e.preventDefault();
      this.model.hideFolderTree(norm);
      this.model.saveLS();
      return;
    }

    if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      this.model.folder = norm;
      this.model.selectInFolder(path);
      this.model.saveLS();
      this._folderShiftAnchor = norm;
      setTimeout(() => this.tagPanel.render(), 0);
      return;
    }

    if (e.shiftKey && anchorPath !== null) {
      e.preventDefault();
      const anchorData =
        anchorPath === "" ? "." : anchorPath;
      const range = this._folderPathsInRangeFromDom(anchorData, path);
      this.model.folder = norm;
      this.model.addTracksUnderFolderPrefixes(range);
      this._folderShiftAnchor = norm;
      this.model.saveLS();
      setTimeout(() => this.tagPanel.render(), 0);
      return;
    }

    this._folderShiftAnchor = norm;
    this.model.folder = norm;
    this.model.selectOnlyInFolder(path);
    this.model.saveLS();
  }

  private _commitFolderRename(path: string, newName: string): void {
    void this.cmdMgr.run(
      new RenameFolderCommand(path, newName),
      this.model,
    );
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
      this._commitFolderRename(path, newName);
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

  private async _refreshFolderTreeHtmx(): Promise<void> {
    const pr = this.pendingRenameForTree ?? "";
    this.pendingRenameForTree = null;
    const m = this.model;
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

  /* ── Data loading ───────────────────────────────────────── */

  // ── Background library loading state ──────────────────────
  private _libFlushTimer = 0;
  private _libPending: Track[] = [];

  private _loadLibrary(): Promise<void> {
    const $progress = document.getElementById("loading-progress");
    const m = this.model;
    m.setAllTracks([]);

    return new Promise((resolve) => {
      let firstFlushDone = false;
      let streamDone = false;

      const flush = (final = false) => {
        if (this._libPending.length) {
          const batch = this._libPending.splice(0);
          m.appendTracks(batch);
          this.canvas.scheduleDraw();
          this.status.update(m);
        }

        if (!firstFlushDone && (m.allTracks.length > 0 || final)) {
          firstFlushDone = true;
          if ($progress) $progress.textContent = "";
          resolve();
        }

        if (final) {
          clearInterval(this._libFlushTimer);
          this._libFlushTimer = 0;
          m.libraryLoadProgress = null;
          // Eligible paths / layout_revision depend on the track set; drop layout cache
          // so we never reuse coordinates after a full library rescan.
          this._invalidateLayoutCache();
          m.emit("change");
        }
      };

      // Flush buffered tracks to the canvas every 150 ms while stream is open.
      this._libFlushTimer = setInterval(() => {
        if (!streamDone) flush();
      }, 150) as unknown as number;

      const es = new EventSource("/api/library/stream?folder=&recursive=1");

      es.addEventListener("progress", ((e: MessageEvent) => {
        const data = JSON.parse(e.data) as {
          done: number;
          total: number;
          track: Track;
        };
        this._libPending.push(data.track);
        m.libraryLoadProgress = { done: data.done, total: data.total };
        if ($progress) {
          $progress.textContent = `Loading library… ${data.done} / ${data.total}`;
        }
      }) as EventListener);

      es.addEventListener("done", (() => {
        es.close();
        streamDone = true;
        flush(true);
      }) as EventListener);

      es.onerror = () => {
        es.close();
        streamDone = true;
        if ($progress) $progress.textContent = "";
        flush(true);
      };
    });
  }

  async init(): Promise<void> {
    showLoad();
    try {
      await Promise.all([this._refreshFolderTreeHtmx(), this._loadLibrary()]);
    } finally {
      hideLoad();
    }
    this.tagPanel.render();
    this.props.render();
    this.batch.render();
    this.canvas.resize();
    this.canvas.scheduleDraw();
    this.status.update(this.model);

    if (this.model.viewMode === "embeddings") {
      void this._ensureEmbeddingsAndProject();
    }
  }

  private async _reloadLibraryAndTree(): Promise<void> {
    showLoad();
    try {
      await Promise.all([this._refreshFolderTreeHtmx(), this._loadLibrary()]);
    } finally {
      hideLoad();
    }
    this.tagPanel.render();
    this.props.render();
    this.batch.render();
    this.canvas.scheduleDraw();
    this.status.update(this.model);
    if (this.model.viewMode === "embeddings") {
      void this._ensureEmbeddingsAndProject();
    }
  }

  /* ── Embedding space ───────────────────────────────────── */

  /** Saved viewport state per view-mode so pan/zoom persists across toggles. */
  private _savedVpByMode = new Map<string, { ox: number; oy: number; zoom: number }>();

  private _embedPollTimer = 0;
  private _embedEventSource: EventSource | null = null;
  private _embedBatchCount = 0;
  private _projectionFetchPending = false;
  /** Query-string → {revision, positions} map so A→B→A revisits are instant. */
  private _layoutCache = new Map<string, CachedLayout>();

  /** Drop all cached layouts (e.g. after embedding batch completes). */
  private _invalidateLayoutCache(): void {
    this._layoutCache.clear();
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

  private async _ensureEmbeddingsAndProject(opts?: {
    /** If true, do not kick librosa backfill from this call (CLAP/EffNet clicks only). */
    omitFeaturesGeneration?: boolean;
  }): Promise<void> {
    const m = this.model;
    try {
      const status = await api<{
        total: number;
        fingerprinted: number;
        embedded: number;
        pending: number;
        model_ready: boolean;
        effnet_embedded: number;
        effnet_pending: number;
        effnet_model_ready: boolean;
        features_extracted: number;
        features_pending: number;
        generating: boolean;
        embedding_version?: number;
        layout_revision?: string;
      }>(`/api/embeddings/status?${this._projectionQueryString()}`);

      // Check if required models are still loading.
      if (m.useCLAP && !status.model_ready) {
        toast("CLAP model loading…", "ok");
        this._pollModelReady();
        return;
      }
      if (m.useEffNet && !status.effnet_model_ready) {
        toast("EffNet model loading…", "ok");
        this._pollModelReady();
        return;
      }

      // Determine which sources need generation.
      const sourcesNeeded: string[] = [];
      if (m.useCLAP && status.pending > 0) sourcesNeeded.push("clap");
      if (m.useEffNet && status.effnet_pending > 0) sourcesNeeded.push("effnet");
      if (
        m.anyAudioFeaturesEnabled &&
        status.features_pending > 0 &&
        !opts?.omitFeaturesGeneration
      )
        sourcesNeeded.push("features");

      if (sourcesNeeded.length > 0 && !status.generating) {
        const total = Math.max(
          m.useCLAP ? status.pending : 0,
          m.useEffNet ? status.effnet_pending : 0,
          m.anyAudioFeaturesEnabled ? status.features_pending : 0,
        );
        const workParts: string[] = [];
        if (m.useCLAP && status.pending > 0)
          workParts.push(`${status.pending} CLAP`);
        if (m.useEffNet && status.effnet_pending > 0)
          workParts.push(`${status.effnet_pending} EffNet`);
        if (m.anyAudioFeaturesEnabled && status.features_pending > 0)
          workParts.push(`${status.features_pending} audio features`);
        const detail =
          workParts.length > 1 ? ` (${workParts.join(", ")})` : "";
        toast(`Generating embeddings for ${total} tracks${detail}…`, "ok");
        m.embeddingsGenerating = true;

        await postJSON("/api/embeddings/generate", {
          folder: "",
          recursive: true,
          priority_paths: [],
          sources: sourcesNeeded,
        });
        this._listenEmbeddingStream();
        return;
      }

      if (status.generating) {
        m.embeddingsGenerating = true;
        this._listenEmbeddingStream();
        // Still refetch layout: option changes must hit /api/embeddings/projection even
        // while a batch job runs — otherwise only /status appears and the map never
        // updates for the new source / scaling / method mix.
        if (m.viewMode === "embeddings") {
          void this._fetchProjection();
        }
        return;
      }

      // In embedding mode always refresh the layout — `hasData` keyed only to
      // individual sources missed the case where the *combination* of enabled
      // sources changes (or counts are briefly inconsistent with the cache).
      const hasData =
        (m.useCLAP && status.embedded > 0) ||
        (m.useEffNet && status.effnet_embedded > 0) ||
        (m.anyAudioFeaturesEnabled && status.features_extracted > 0);
      if (m.viewMode === "embeddings" || hasData) {
        await this._fetchProjection({ status });
      }
    } catch {
      /* api() already toasted */
    }
  }

  private _pollModelReady(): void {
    clearTimeout(this._embedPollTimer);
    this._embedPollTimer = setTimeout(async () => {
      try {
        const m = this.model;
        const status = await api<{
          model_ready: boolean;
          effnet_model_ready: boolean;
        }>("/api/embeddings/status?folder=&recursive=1&models_only=1");
        const clapOk = !m.useCLAP || status.model_ready;
        const effnetOk = !m.useEffNet || status.effnet_model_ready;
        if (clapOk && effnetOk) {
          toast("Models ready", "ok");
          await this._ensureEmbeddingsAndProject();
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
      };
      const m = this.model;
      m.embeddingProgress = { done: data.done, total: data.total };
      this.status.update(m);
      this.canvas.scheduleDraw();

      this._embedBatchCount++;
      const interval = Math.max(5, Math.floor(data.total / 10));
      if (
        data.done >= 2 &&
        (this._embedBatchCount >= interval || data.done === data.total) &&
        m.viewMode === "embeddings" &&
        !this._projectionFetchPending
      ) {
        this._embedBatchCount = 0;
        void this._fetchProjectionIncremental();
      }
    }) as EventListener);

    es.addEventListener("done", () => {
      this._closeEmbeddingStream();
      const m = this.model;
      m.embeddingsGenerating = false;
      m.embeddingProgress = null;
      this._invalidateLayoutCache();
      toast("Embeddings ready", "ok");
      this.status.update(m);
      this.canvas.scheduleDraw();
      if (m.viewMode === "embeddings") {
        void this._fetchProjection();
      }
    });

    es.addEventListener("error", () => {
      this._closeEmbeddingStream();
      const m = this.model;
      m.embeddingsGenerating = false;
      m.embeddingProgress = null;
      this.status.update(m);
      this.canvas.scheduleDraw();
    });
  }

  private _closeEmbeddingStream(): void {
    if (this._embedEventSource) {
      this._embedEventSource.close();
      this._embedEventSource = null;
    }
  }

  private _projectionQueryString(methodOverride?: string, skipContext = false): string {
    const m = this.model;
    const method = methodOverride ?? m.projectionMethod;
    const sources = m.activeSources.join(",");
    let qs = `folder=&recursive=1&method=${method}&sources=${encodeURIComponent(sources)}`;
    qs += `&feature_mask=${m.audioFeatureMask()}`;
    qs += `&features_blend=${encodeURIComponent(String(m.featuresBlend))}`;
    qs += `&folder_boost=${encodeURIComponent(String(m.folderContrastBoost))}`;
    qs += `&folder_depth_boost=${encodeURIComponent(String(m.folderDepthBoost))}`;
    if (!skipContext) {
      if (m.scaleByTags) {
        const tags = m.contextTags.join(",");
        if (tags) qs += `&context_tags=${encodeURIComponent(tags)}`;
      }
      if (m.scaleByFolders) {
        qs += "&scale_folders=1";
      }
    }
    return qs;
  }

  /** Fetch PCA positions during ongoing generation, animating the transition.
   *
   *  Context (semantic weighting) is intentionally skipped: PCA is a quick
   *  preview and the CLAP model is busy with audio inference on the generation
   *  thread — concurrent text inference can fail on MPS / CUDA.
   */
  private async _fetchProjectionIncremental(): Promise<void> {
    this._projectionFetchPending = true;
    const m = this.model;
    m.projectionPending = true;
    this.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${this._projectionQueryString("pca", true)}`,
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
      this.canvas.scheduleDraw();
    }
  }

  /** Smoothly lerp from current embedding positions to *newPositions*. */
  private _animatePositionUpdate(
    newPositions: Map<string, { x: number; y: number }>,
  ): void {
    const cv = this.canvas;
    const m = this.model;

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

  /** Full TSNE/UMAP/… layout.  Checks the client-side layout cache
   * (keyed by query string + server revision) so A→B→A revisits are instant.
   * Falls back to ``/api/embeddings/projection`` on miss.  The server also
   * has a revision-keyed LRU so repeated requests are fast even without client
   * cache (e.g. after page reload). */
  private async _fetchProjection(opts?: {
    status?: EmbeddingLayoutStatusHint;
  }): Promise<void> {
    const m = this.model;
    const qs = this._projectionQueryString();

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
        this._applyPositions(cached.positions);
        return;
      }
    }

    // --- Full projection fetch ---
    m.projectionPending = true;
    this.canvas.scheduleDraw();
    try {
      const raw = await api<unknown>(
        `/api/embeddings/projection?${qs}`,
      );
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
      m.projectionPending = false;
      this.canvas.scheduleDraw();
    }
  }

  /** Apply a positions map to the model, animating the transition. */
  private _applyPositions(
    newPositions: Map<string, { x: number; y: number }>,
  ): void {
    const m = this.model;
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

  private _animFrame = 0;

  private _animateToEmbeddings(): void {
    const cv = this.canvas;
    const m = this.model;

    // Capture current screen positions as starting points
    const startPos = new Map<string, { x: number; y: number }>();
    for (const t of m.tracks) {
      if (m.viewMode === "embeddings" && m.embeddingPositions.has(t.path)) {
        const existing = cv.animPositions?.get(t.path);
        if (existing) {
          startPos.set(t.path, { ...existing });
        } else {
          const wx = m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5;
          const wy = m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5;
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
    const cv = this.canvas;
    const m = this.model;

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
        const tx = m.axisX ? (track.tags[m.axisX] ?? 0.5) : 0.5;
        const ty = m.axisY ? (track.tags[m.axisY] ?? 0.5) : 0.5;
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
   */
  private _animateAxisChange(which: "axisX" | "axisY", tag: string): void {
    const cv = this.canvas;
    const m = this.model;

    // Snapshot where every track currently sits (mid-animation positions take
    // priority, otherwise use current tag values with the old axes).
    const startPos = new Map<string, { x: number; y: number }>();
    for (const t of m.tracks) {
      const existing = cv.animPositions?.get(t.path);
      if (existing) {
        startPos.set(t.path, { ...existing });
      } else {
        startPos.set(t.path, {
          x: m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5,
          y: m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5,
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
        const tx = m.axisX ? (track.tags[m.axisX] ?? 0.5) : 0.5;
        const ty = m.axisY ? (track.tags[m.axisY] ?? 0.5) : 0.5;
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

  private async _toggleViewMode(): Promise<void> {
    const m = this.model;
    // Persist the current viewport for the mode we're leaving.
    this._savedVpByMode.set(m.viewMode, { ...m.vp });

    if (m.viewMode === "tags") {
      m.setViewMode("embeddings");
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
        await this._ensureEmbeddingsAndProject();
      }
    } else {
      m.setViewMode("tags");
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
    this.status.update(m);
    const $vm = document.getElementById(
      "view-mode-toggle",
    ) as HTMLInputElement | null;
    if ($vm) $vm.checked = m.viewMode === "embeddings";
  }

  /** Switch to a specific view mode (no-op if already there). */
  private async _switchToMode(target: "tags" | "embeddings"): Promise<void> {
    if (this.model.viewMode === target) return;
    await this._toggleViewMode();
  }

  /* ── Subfolder creation ─────────────────────────────────── */

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
          this.pendingRenameForTree = res.path;
          await this._refreshFolderTreeHtmx();
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

  /* ── Canvas mouse events ────────────────────────────────── */

  private _bindCanvas(): void {
    const $c = this.canvas.$canvas;
    $c.addEventListener("mouseenter", () => {
      this._pointerOverCanvas = true;
    });
    $c.addEventListener("mousedown", (e) => this._onDown(e));
    $c.addEventListener("mousemove", (e) => this._onMove(e));
    $c.addEventListener("mouseup", (e) => this._onUp(e));
    $c.addEventListener("mouseleave", () => this._onLeave());
    $c.addEventListener("wheel", (e) => this._onWheel(e), {
      passive: false,
    });
    $c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  private _onDown(e: MouseEvent): void {
    const m = this.model;
    const cv = this.canvas;
    const mouse = this.mouse;
    const sx = e.offsetX,
      sy = e.offsetY;
    mouse.sx = sx;
    mouse.sy = sy;
    mouse.lx = sx;
    mouse.ly = sy;
    mouse.hitWasSelected = false;
    mouse.hitInSelectionBox = false;

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      mouse.mode = "pan";
      cv.$canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;

    // Resize handles always take priority (disabled in embedding mode).
    const txh =
      m.selected.size > 0 && m.viewMode !== "embeddings"
        ? cv.hitTestTransform(sx, sy)
        : null;
    if (txh && txh.type !== "move") {
      this._startTxform(txh.type, sx, sy);
      return;
    }

    const tracks = m.tracks;
    const hit = cv.hitTest(sx, sy);
    mouse.hitIdx = hit;

    // Shift on empty space (no dot, not inside selection box) → lasso.
    if (e.shiftKey && hit < 0 && !(txh?.type === "move")) {
      mouse.mode = "lasso";
      cv.lassoPoints = [[sx, sy]];
      cv.intersectMode = e.ctrlKey || e.metaKey;
      return;
    }

    const additive = e.ctrlKey || e.metaKey || e.shiftKey;

    if (hit >= 0) {
      // Clicked on a track dot.
      const hitPath = tracks[hit].path;
      mouse.hitWasSelected = m.selected.has(hitPath);
      if (!mouse.hitWasSelected) {
        if (!additive) m.selected.clear();
        m.selected.add(hitPath);
      }
      // In embedding mode allow pending so a threshold drag can trigger folder-drop,
      // but skip the tag-position snap (embedding positions are used for ghosts instead).
      mouse.mode = "pending";
      if (m.viewMode !== "embeddings") {
        this._fillDragSnapFromSelection();
      }
    } else if (txh?.type === "move" && m.viewMode !== "embeddings") {
      // Clicked inside the selection bounding box — same drag path as track drag.
      mouse.hitInSelectionBox = true;
      mouse.mode = "pending";
      this._fillDragSnapFromSelection();
    } else {
      // Clicked on empty background → box selection.
      mouse.mode = "boxsel";
      cv.boxSel = { x0: sx, y0: sy, x1: sx, y1: sy };
      cv.intersectMode = e.ctrlKey || e.metaKey;
    }

    cv.scheduleDraw();
    this.status.update(m);
    this.props.render();
    this.batch.render();
  }

  /** Tag-drag snapshot for the current selection; used for canvas drag and folder-drop drag. */
  private _fillDragSnapFromSelection(): void {
    const m = this.model;
    const mouse = this.mouse;
    mouse.snap = new Map();
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t)
        mouse.snap.set(path, {
          track: t,
          x: m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5,
          y: m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5,
        });
    }
  }

  /** Shared path: tag drag on canvas + drop on folder tree (from track pending or box-select). */
  private _startDocDragFromSelection(): void {
    const m = this.model;
    const cv = this.canvas;
    const mouse = this.mouse;
    this._fillDragSnapFromSelection();
    mouse.mode = "drag";
    cv.$canvas.style.cursor = "move";
    const ghosts = new Map<string, { wx: number; wy: number; folder: string }>();
    for (const [path, snap] of mouse.snap!) {
      // In embedding mode use the embedding position for the ghost dot so it sits
      // on top of the dot the user clicked rather than at the (invisible) tag position.
      const ep = m.viewMode === "embeddings" ? m.embeddingPositions.get(path) : null;
      ghosts.set(path, {
        wx: ep ? ep.x : snap.x,
        wy: ep ? ep.y : snap.y,
        folder: snap.track.folder ?? "",
      });
    }
    cv.dragGhosts = ghosts;
    document.addEventListener("mousemove", this._onDocDragMove);
    document.addEventListener("mouseup", this._onDocDragUp);
  }

  private _startTxform(
    handleType: string,
    sx: number,
    sy: number,
  ): void {
    const m = this.model;
    const cv = this.canvas;
    const mouse = this.mouse;
    const [wx, wy] = cv.s2w(sx, sy);

    mouse.mode = "txform";
    mouse.txHandle = handleType;
    mouse.txStart = { wx, wy };
    mouse.txBounds = cv._txWorld
      ? { ...cv._txWorld }
      : { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    mouse.txSnap = new Map();

    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t)
        mouse.txSnap.set(path, {
          wx: m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5,
          wy: m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5,
          track: t,
        });
    }

    const cursor =
      handleType === "move"
        ? "move"
        : handleType === "n" || handleType === "s"
          ? "ns-resize"
          : handleType === "e" || handleType === "w"
            ? "ew-resize"
            : handleType === "nw" || handleType === "se"
              ? "nwse-resize"
              : "nesw-resize";
    cv.$canvas.style.cursor = cursor;
  }

  private _applyTxform(wx: number, wy: number): void {
    const m = this.model;
    const mouse = this.mouse;
    const h = mouse.txHandle!;
    const { minX, maxX, minY, maxY } = mouse.txBounds!;

    if (h === "move") {
      const dx = wx - mouse.txStart!.wx;
      const dy = wy - mouse.txStart!.wy;
      for (const [, snap] of mouse.txSnap!) {
        if (m.axisX) snap.track.tags[m.axisX] = clamp(snap.wx + dx);
        if (m.axisY) snap.track.tags[m.axisY] = clamp(snap.wy + dy);
      }
      return;
    }

    const affectsX = m.axisX && /e|w/.test(h);
    const affectsY = m.axisY && /n|s/.test(h);

    let scaleX = 1,
      axAnchor = minX;
    if (affectsX) {
      const isLeft = /w/.test(h);
      axAnchor = isLeft ? maxX : minX;
      const origEdge = isLeft ? minX : maxX;
      const edgeDist = origEdge - axAnchor;
      scaleX = edgeDist !== 0 ? (wx - axAnchor) / edgeDist : 1;
    }

    let scaleY = 1,
      ayAnchor = minY;
    if (affectsY) {
      const isTop = /n/.test(h);
      ayAnchor = isTop ? minY : maxY;
      const origEdge = isTop ? maxY : minY;
      const edgeDist = origEdge - ayAnchor;
      scaleY = edgeDist !== 0 ? (wy - ayAnchor) / edgeDist : 1;
    }

    for (const [, snap] of mouse.txSnap!) {
      const t = snap.track;
      if (affectsX)
        t.tags[m.axisX!] = clamp(axAnchor + (snap.wx - axAnchor) * scaleX);
      if (affectsY)
        t.tags[m.axisY!] = clamp(ayAnchor + (snap.wy - ayAnchor) * scaleY);
    }
  }

  /**
   * Recompute scatter hit/hover after ``tracks`` changed (indices are not stable).
   * Runs on the frame after ``scheduleDraw`` so ``positions`` matches the new list.
   */
  private _resyncCanvasHoverFromLastPointer(): void {
    if (!this._pointerOverCanvas || this.mouse.mode !== "idle") return;
    this._updateCanvasHoverAt(this.mouse.lx, this.mouse.ly);
  }

  /** Update hovered dot + sibling glow from canvas pixel coordinates. */
  private _updateCanvasHoverAt(sx: number, sy: number): void {
    const m = this.model;
    const cv = this.canvas;
    const prevIdx = cv.hoveredIdx;
    const prevPath =
      prevIdx >= 0 ? (m.tracks[prevIdx]?.path ?? null) : null;

    this.mouse.lx = sx;
    this.mouse.ly = sy;

    cv.hoveredIdx = cv.hitTest(sx, sy);
    const newPath =
      cv.hoveredIdx >= 0 ? (m.tracks[cv.hoveredIdx]?.path ?? null) : null;

    if (cv.hoveredIdx >= 0) {
      const ht = m.tracks[cv.hoveredIdx];
      cv.hoveredSiblingFolder = ht ? (ht.folder ?? "") : null;
    } else {
      cv.hoveredSiblingFolder = null;
    }

    if (cv.hoveredIdx === prevIdx && newPath === prevPath) return;

    clearTimeout(cv.tipTimer);
    cv.tipReady = false;
    cv.$tip.classList.add("hidden");
    if (cv.hoveredIdx >= 0)
      cv.tipTimer = setTimeout(() => {
        cv.tipReady = true;
        cv.scheduleDraw();
      }, 150) as unknown as number;
    cv.scheduleDraw();
    this._handlePreview();
    this._updateFolderHighlights();
  }

  private _onMove(e: MouseEvent): void {
    const m = this.model;
    const cv = this.canvas;
    const mouse = this.mouse;
    const sx = e.offsetX,
      sy = e.offsetY;

    if (mouse.mode === "pan") {
      const [wx1, wy1] = cv.s2w(mouse.lx, mouse.ly);
      const [wx2, wy2] = cv.s2w(sx, sy);
      m.vp.ox -= wx2 - wx1;
      m.vp.oy -= wy2 - wy1;
      mouse.lx = sx;
      mouse.ly = sy;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "boxsel") {
      cv.boxSel = { x0: mouse.sx, y0: mouse.sy, x1: sx, y1: sy };
      cv.intersectMode = e.ctrlKey || e.metaKey;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "lasso") {
      cv.lassoPoints.push([sx, sy]);
      cv.intersectMode = e.ctrlKey || e.metaKey;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "txform") {
      const [wx, wy] = cv.s2w(sx, sy);
      this._applyTxform(wx, wy);
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "pending") {
      if (
        Math.hypot(sx - mouse.sx, sy - mouse.sy) > DRAG_THRESH &&
        m.selected.size > 0
      ) {
        this._startDocDragFromSelection();
      }
      return;
    }

    if (mouse.mode === "drag") {
      // In embedding mode the drag is only for folder-drop; skip tag mutations.
      if (m.viewMode !== "embeddings") {
        const [swx, swy] = cv.s2w(mouse.sx, mouse.sy);
        const [cwx, cwy] = cv.s2w(sx, sy);
        const dx = cwx - swx,
          dy = cwy - swy;
        for (const [, snap] of mouse.snap!) {
          if (m.axisX) snap.track.tags[m.axisX] = clamp(snap.x + dx);
          if (m.axisY) snap.track.tags[m.axisY] = clamp(snap.y + dy);
        }
      }
      cv.scheduleDraw();
      return;
    }

    // Idle hover
    this._updateCanvasHoverAt(sx, sy);

    const txh: TxHandle | { type: string; cursor: string } | null =
      m.selected.size > 0 ? cv.hitTestTransform(sx, sy) : null;
    if (txh) {
      cv.$canvas.style.cursor = txh.cursor;
    } else {
      cv.$canvas.style.cursor = "";
    }
  }

  private _onUp(e: MouseEvent): void {
    const m = this.model;
    const cv = this.canvas;
    const mouse = this.mouse;
    const tracks = m.tracks;

    if (mouse.mode === "boxsel") {
      const bs = cv.boxSel;
      cv.boxSel = null;
      cv.intersectMode = false;
      if (bs) {
        const bL = Math.min(bs.x0, bs.x1),
          bR = Math.max(bs.x0, bs.x1);
        const bT = Math.min(bs.y0, bs.y1),
          bB = Math.max(bs.y0, bs.y1);
        const isClick = bR - bL < 4 && bB - bT < 4;
        if (e.ctrlKey || e.metaKey) {
          // AND / intersect: keep only the existing selection that also falls inside the box
          if (!isClick) {
            const inBox = new Set<string>(
              cv.positions
                .filter((p) => p.sx >= bL && p.sx <= bR && p.sy >= bT && p.sy <= bB)
                .map((p) => tracks[p.idx].path),
            );
            for (const path of [...m.selected])
              if (!inBox.has(path)) m.selected.delete(path);
          }
        } else {
          m.selected.clear();
          if (!isClick) {
            for (const p of cv.positions)
              if (p.sx >= bL && p.sx <= bR && p.sy >= bT && p.sy <= bB)
                m.selected.add(tracks[p.idx].path);
          }
        }
      }
      this._finalizeSelection();
      return;
    }

    if (mouse.mode === "txform") {
      for (const path of m.selected) {
        const t = m.trackByPath(path);
        if (!t) continue;
        if (m.axisX && t.tags[m.axisX] !== undefined)
          this.pending.set(`${path}|${m.axisX}`, {
            path,
            tag: m.axisX,
            value: t.tags[m.axisX],
          });
        if (m.axisY && t.tags[m.axisY] !== undefined)
          this.pending.set(`${path}|${m.axisY}`, {
            path,
            tag: m.axisY,
            value: t.tags[m.axisY],
          });
      }
      this._flushPending();
      mouse.mode = "idle";
      mouse.txSnap = null;
      cv.$canvas.style.cursor = "";
      cv.scheduleDraw();
      this.status.update(m);
      this.props.render();
      this.batch.render();
      this.tagPanel.render();
      return;
    }

    if (mouse.mode === "lasso") {
      const lassoPts = cv.lassoPoints;
      cv.lassoPoints = [];
      cv.intersectMode = false;
      const inLasso = new Set<string>(
        cv.positions
          .filter((p) => pointInPoly(p.sx, p.sy, lassoPts))
          .map((p) => tracks[p.idx].path),
      );
      if (e.ctrlKey || e.metaKey) {
        // AND / intersect: keep only the existing selection that also falls inside the lasso
        for (const path of [...m.selected])
          if (!inLasso.has(path)) m.selected.delete(path);
      } else {
        // Shift starts lasso, so treat shift as additive union
        if (!e.shiftKey) m.selected.clear();
        for (const path of inLasso) m.selected.add(path);
      }
    }

    if (mouse.mode === "pending") {
      const hitPath =
        mouse.hitIdx >= 0 ? tracks[mouse.hitIdx]?.path : null;
      const additive = e.ctrlKey || e.metaKey || e.shiftKey;

      if (mouse.hitInSelectionBox) {
        // Released inside the selection box without dragging: keep selection unchanged.
      } else if (additive) {
        if (hitPath && mouse.hitWasSelected) {
          // Toggle off a track that was already selected.
          m.selected.delete(hitPath);
        }
        // If !hitWasSelected the track was already added in _onDown, keep it.
      } else {
        // Plain click: narrow down to just this track (or clear if none).
        m.selected.clear();
        if (hitPath) m.selected.add(hitPath);
      }
    }

    if (mouse.mode !== "drag") {
      this._finalizeSelection();
    }
  }

  /** Shared post-selection cleanup: resets interaction state and emits
   *  "change" so all views (including folder highlights) update together. */
  private _finalizeSelection(): void {
    this.mouse.mode = "idle";
    this.mouse.snap = null;
    this.canvas.$canvas.style.cursor = "";
    this.model.emit("change");
  }

  private _onLeave(): void {
    const mouse = this.mouse;
    const cv = this.canvas;
    if (mouse.mode === "drag" || mouse.mode === "txform") {
      mouse.draggingOutside = true;
      return;
    }
    if (mouse.mode !== "idle")
      this._onUp({
        offsetX: mouse.lx,
        offsetY: mouse.ly,
        ctrlKey: false,
        metaKey: false,
      } as MouseEvent);
    cv.boxSel = null;
    cv.intersectMode = false;
    this._pointerOverCanvas = false;
    cv.clearHover();
    cv.$canvas.style.cursor = "";
    this._stopPreview();
    cv.scheduleDraw();
    this._updateFolderHighlights();
  }

  private _onWheel(e: WheelEvent): void {
    e.preventDefault();
    const m = this.model;
    const cv = this.canvas;
    const vp = m.vp;
    const sx = e.offsetX,
      sy = e.offsetY;

    const norm =
      e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? cv.$canvas.clientHeight : 1;
    const dx = e.deltaX * norm,
      dy = e.deltaY * norm;

    if (e.ctrlKey) {
      const factor = Math.exp(-dy * 0.005);
      const [wx, wy] = cv.s2w(sx, sy);
      const zMin = cv.minZoom();
      vp.zoom = clamp(vp.zoom * factor, zMin, 50);
      const [wx2, wy2] = cv.s2w(sx, sy);
      vp.ox += wx - wx2;
      vp.oy += wy - wy2;
    } else {
      const isTrackpad =
        e.deltaMode === 0 && (Math.abs(dx) > 0 || Math.abs(dy) < 50);
      if (isTrackpad) {
        const [wx0, wy0] = cv.s2w(0, 0);
        const [wx1, wy1] = cv.s2w(dx, dy);
        vp.ox += wx1 - wx0;
        vp.oy += wy1 - wy0;
      } else {
        const f = dy < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const [wx, wy] = cv.s2w(sx, sy);
        vp.zoom = clamp(vp.zoom * f, cv.minZoom(), 50);
        const [wx2, wy2] = cv.s2w(sx, sy);
        vp.ox += wx - wx2;
        vp.oy += wy - wy2;
      }
    }
    cv.scheduleDraw();
  }

  /* ── Document-level drag (folder drop) ──────────────────── */

  private _cleanupDocDrag(): void {
    document.removeEventListener("mousemove", this._onDocDragMove);
    document.removeEventListener("mouseup", this._onDocDragUp);
    if (this.mouse.highlightedFolder) {
      this.mouse.highlightedFolder.classList.remove("drop-target");
      this.mouse.highlightedFolder = null;
    }
    this.mouse.draggingOutside = false;
  }

  private _onDocDragMove = (e: MouseEvent): void => {
    if (this.mouse.mode !== "drag") {
      this._cleanupDocDrag();
      return;
    }
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const folderEl =
      el?.closest<HTMLElement>(".folder-label[data-path]") ?? null;
    if (this.mouse.highlightedFolder !== folderEl) {
      if (this.mouse.highlightedFolder)
        this.mouse.highlightedFolder.classList.remove("drop-target");
      this.mouse.highlightedFolder = folderEl;
      if (folderEl) folderEl.classList.add("drop-target");
    }
  };

  private _onDocDragUp = async (e: MouseEvent): Promise<void> => {
    if (this.mouse.mode !== "drag") {
      this._cleanupDocDrag();
      return;
    }
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const folderEl =
      el?.closest<HTMLElement>(".folder-label[data-path]") ?? null;
    this._cleanupDocDrag();

    if (folderEl) {
      this._revertDragSnapToBaseline();
    } else {
      this._commitDrag();
    }

    // Reset interaction state immediately so that any canvas mousemove events
    // fired during the async network call below are ignored.
    this.mouse.mode = "idle";
    this.mouse.snap = null;
    this.canvas.dragGhosts = null;
    this.canvas.$canvas.style.cursor = "";
    this.canvas.scheduleDraw();
    this.status.update(this.model);
    this.props.render();
    this.batch.render();
    this.tagPanel.render();

    if (folderEl) {
      const dest =
        folderEl.dataset.path === "." ? "" : folderEl.dataset.path!;
      this._moveSelectedToFolder(dest);
    }
  };

  private _moveSelectedToFolder(destPath: string): void {
    const m = this.model;
    const paths = [...m.selected];
    if (!paths.length) return;
    void this.cmdMgr.run(new MoveTracksCommand(paths, destPath), m);
  }

  /* ── Debounced tag writes ───────────────────────────────── */

  /** Undo in-viewport tag offsets when the drag ends with a folder drop instead of a canvas commit. */
  private _revertDragSnapToBaseline(): void {
    const m = this.model;
    // In embedding mode no tag values were mutated during the drag, nothing to revert.
    if (m.viewMode === "embeddings") return;
    for (const [, snap] of this.mouse.snap ?? []) {
      if (m.axisX) snap.track.tags[m.axisX] = snap.x;
      if (m.axisY) snap.track.tags[m.axisY] = snap.y;
    }
    m.emit("tags-dirty");
  }

  private _commitDrag(): void {
    const m = this.model;
    // In embedding mode no tag values were changed during the drag; nothing to persist.
    if (m.viewMode === "embeddings") return;
    for (const [path, snap] of this.mouse.snap ?? []) {
      if (m.axisX)
        this.pending.set(`${path}|${m.axisX}`, {
          path,
          tag: m.axisX,
          value: snap.track.tags[m.axisX],
        });
      if (m.axisY)
        this.pending.set(`${path}|${m.axisY}`, {
          path,
          tag: m.axisY,
          value: snap.track.tags[m.axisY],
        });
    }
    this._flushPending();
  }

  private _flushPending = debounce(async () => {
    const updates = [...this.pending.values()];
    this.pending.clear();
    if (!updates.length) return;
    try {
      await postJSON("/api/tracks/tags", { updates });
      toast(`Saved ${updates.length} tag values`, "ok");
    } catch {
      /* api() already toasted */
    }
  }, FLUSH_MS);

  /* ── Fit view to visible track bounds ───────────────────── */

  private _vpFromWorldBounds(
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
  ): { ox: number; oy: number; zoom: number } {
    const midX = (minX + maxX) / 2,
      midY = (minY + maxY) / 2;
    const margin = 0.1;
    const zx =
      maxX - minX > 0.01 ? (1 - 2 * margin) / (maxX - minX) : 50;
    const zy =
      maxY - minY > 0.01 ? (1 - 2 * margin) / (maxY - minY) : 50;
    const zoom = Math.max(this.canvas.minZoom(), Math.min(50, Math.min(zx, zy)));
    return {
      zoom,
      ox: midX - 1 / (2 * zoom),
      oy: midY - 1 / (2 * zoom),
    };
  }

  private _computeFitVP(): { ox: number; oy: number; zoom: number } {
    const m = this.model;

    // In embedding mode positions are already normalised to [0,1], so the
    // default viewport shows everything.
    if (m.viewMode === "embeddings") return { ox: 0, oy: 0, zoom: 1 };

    const visible = m.tracks.filter((t) => m.passesFilter(t));
    if (visible.length === 0) return { ox: 0, oy: 0, zoom: 1 };

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const t of visible) {
      const wx = m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5;
      const wy = m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5;
      if (wx < minX) minX = wx;
      if (wx > maxX) maxX = wx;
      if (wy < minY) minY = wy;
      if (wy > maxY) maxY = wy;
    }
    if (maxX - minX < 0.01 && maxY - minY < 0.01)
      return { ox: 0, oy: 0, zoom: 1 };

    return this._vpFromWorldBounds(minX, maxX, minY, maxY);
  }

  private _p80Radius(dists: number[]): number {
    if (!dists.length) return 0;
    const s = [...dists].sort((a, b) => a - b);
    const idx = Math.min(
      s.length - 1,
      Math.max(0, Math.ceil(0.8 * s.length) - 1),
    );
    return s[idx];
  }

  /** Fit viewport to an 80th-percentile disc around the centroid (folder/track folder hover + H). */
  private _fitViewToFolderPercentile(prefixNorm: string): void {
    const m = this.model;
    const subset: Track[] = [];
    for (const t of m.allTracks) {
      if (!m.trackInViewUnion(t) || m.isTrackHidden(t)) continue;
      if (!m.passesFilter(t)) continue;
      const tf = t.folder ?? "";
      if (prefixNorm !== "") {
        if (tf !== prefixNorm && !tf.startsWith(prefixNorm + "/")) continue;
      }
      subset.push(t);
    }
    const pos: { wx: number; wy: number }[] = [];
    for (const t of subset) {
      const p = this.canvas.worldPosForTrack(t);
      if (p) pos.push(p);
    }
    if (pos.length === 0) return;
    let cx = 0,
      cy = 0;
    for (const p of pos) {
      cx += p.wx;
      cy += p.wy;
    }
    cx /= pos.length;
    cy /= pos.length;
    const dists = pos.map((p) => Math.hypot(p.wx - cx, p.wy - cy));
    const rmax = this._p80Radius(dists);
    const inside = pos.filter(
      (p) => Math.hypot(p.wx - cx, p.wy - cy) <= rmax + 1e-9,
    );
    if (inside.length === 0) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of inside) {
      if (p.wx < minX) minX = p.wx;
      if (p.wx > maxX) maxX = p.wx;
      if (p.wy < minY) minY = p.wy;
      if (p.wy > maxY) maxY = p.wy;
    }
    if (maxX - minX < 0.01 && maxY - minY < 0.01) {
      Object.assign(m.vp, { ox: 0, oy: 0, zoom: 1 });
      return;
    }
    Object.assign(m.vp, this._vpFromWorldBounds(minX, maxX, minY, maxY));
  }

  private _fitViewToTracks(): void {
    const target = this._computeFitVP();
    Object.assign(this.model.vp, target);
  }

  /* ── Hover preview ──────────────────────────────────────── */

  private _handlePreview(): void {
    const m = this.model;
    const cv = this.canvas;
    if (!m.hoverPreview || cv.hoveredIdx < 0) {
      this._stopPreview();
      return;
    }
    this._startPreview(m.tracks[cv.hoveredIdx]);
  }

  private _startPreview(track: Track): void {
    if (!track || this.previewPath === track.path) return;
    this.previewPath = track.path;
    this.$audio.src = `/api/audio/${encodeURIComponent(track.path)}`;
    this.$audio.onloadedmetadata = () => {
      this.$audio.currentTime = this.$audio.duration * 0.4;
    };
    this.$audio.play().catch(() => {});
  }

  private _stopPreview(): void {
    if (!this.previewPath) return;
    this.previewPath = null;
    this.$audio.pause();
    this.$audio.removeAttribute("src");
    this.$audio.load();
  }

  /* ── Sidebar controls ───────────────────────────────────── */

  private _bindSidebar(): void {
    const m = this.model;
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement;
    const $vm = document.getElementById(
      "view-mode-toggle",
    ) as HTMLInputElement;
    const $addT = document.getElementById("btn-add-tag")!;

    $hov.checked = m.hoverPreview;
    $hov.addEventListener("change", () => {
      m.hoverPreview = $hov.checked;
      if (!m.hoverPreview) this._stopPreview();
      m.saveLS();
    });

    $vm.checked = m.viewMode === "embeddings";
    $vm.addEventListener("change", () => void this._toggleViewMode());

    $addT.addEventListener("click", () => {
      let name = "new_tag";
      let n = 1;
      while (m.tags.includes(name)) {
        n++;
        name = `new_tag_${n}`;
      }
      m.addKnownTag(name);
      m.saveLS();
      this.tagPanel.scheduleRenameAfterRender(name);
      this.tagPanel.render();
    });

    this._initProjectionToggle();
    this._initSourceCheckboxes();
    this._initFolderTuneSliders();
    this._initScalingToggles();
  }

  private _initSourceCheckboxes(): void {
    const m = this.model;
    const main: [string, "clap" | "effnet"][] = [
      ["source-clap", "clap"],
      ["source-effnet", "effnet"],
    ];
    for (const [elId, source] of main) {
      const $cb = document.getElementById(elId) as HTMLInputElement | null;
      if (!$cb) continue;
      $cb.checked = source === "clap" ? m.useCLAP : m.useEffNet;
      $cb.addEventListener("change", () => {
        const before = m.activeSources.slice().sort().join(",");
        m.toggleSource(source);
        const actual = source === "clap" ? m.useCLAP : m.useEffNet;
        $cb.checked = actual;
        m.saveLS();
        if (m.viewMode !== "embeddings") return;
        const after = m.activeSources.slice().sort().join(",");
        if (before === after) {
          void this._fetchProjection();
          return;
        }
        void this._ensureEmbeddingsAndProject({ omitFeaturesGeneration: true });
      });
    }

    const feats: [string, "tempo" | "key" | "mode" | "energy" | "dance"][] = [
      ["source-feat-tempo", "tempo"],
      ["source-feat-key", "key"],
      ["source-feat-mode", "mode"],
      ["source-feat-energy", "energy"],
      ["source-feat-dance", "dance"],
    ];
    for (const [elId, dim] of feats) {
      const $cb = document.getElementById(elId) as HTMLInputElement | null;
      if (!$cb) continue;
      $cb.addEventListener("change", () => {
        const sourcesBefore = m.activeSources.slice().sort().join(",");
        m.toggleAudioFeature(dim);
        m.saveLS();
        this._syncAudioFeatureCheckbox(elId, dim);
        if (m.viewMode !== "embeddings") return;
        const sourcesAfter = m.activeSources.slice().sort().join(",");
        // Turning individual librosa dims on/off only changes the feature *mask* for
        // projection.  The cache always stores the full 6D vector per fingerprint —
        // do not run /generate (and the straggler pending toast) on every mask tweak.
        // When the ``features`` layer is added or removed entirely, run the full path.
        if (sourcesBefore === sourcesAfter) {
          void this._fetchProjection();
        } else {
          void this._ensureEmbeddingsAndProject();
        }
      });
    }
  }

  private _syncAudioFeatureCheckbox(
    elId: string,
    dim: "tempo" | "key" | "mode" | "energy" | "dance",
  ): void {
    const m = this.model;
    const $cb = document.getElementById(elId) as HTMLInputElement | null;
    if (!$cb) return;
    const val =
      dim === "tempo"
        ? m.useAudioFeatureTempo
        : dim === "key"
          ? m.useAudioFeatureKey
          : dim === "mode"
            ? m.useAudioFeatureMode
            : dim === "energy"
              ? m.useAudioFeatureEnergy
              : m.useAudioFeatureDance;
    $cb.checked = val;
  }

  private _initFolderTuneSliders(): void {
    const m = this.model;
    const $fb = document.getElementById("folder-boost-range") as HTMLInputElement | null;
    const $fdb = document.getElementById("folder-depth-boost-range") as HTMLInputElement | null;
    if ($fb) {
      $fb.value = String(m.folderContrastBoost);
      $fb.addEventListener("input", () => {
        m.folderContrastBoost = parseFloat($fb.value);
      });
      $fb.addEventListener("change", () => {
        m.folderContrastBoost = parseFloat($fb.value);
        m.saveLS();
        if (m.viewMode === "embeddings") void this._fetchProjection();
      });
    }
    if ($fdb) {
      $fdb.value = String(m.folderDepthBoost);
      $fdb.addEventListener("input", () => {
        m.folderDepthBoost = parseFloat($fdb.value);
      });
      $fdb.addEventListener("change", () => {
        m.folderDepthBoost = parseFloat($fdb.value);
        m.saveLS();
        if (m.viewMode === "embeddings") void this._fetchProjection();
      });
    }
  }

  private _initScalingToggles(): void {
    const m = this.model;
    const $tags = document.getElementById("scale-tags-toggle") as HTMLInputElement;
    const $folders = document.getElementById("scale-folders-toggle") as HTMLInputElement;

    $tags.checked = m.scaleByTags;
    $folders.checked = m.scaleByFolders;

    const onToggle = () => {
      m.scaleByTags = $tags.checked;
      m.scaleByFolders = $folders.checked;
      m.saveLS();
      if (m.viewMode === "embeddings") {
        void this._fetchProjection();
      } else {
        void this._switchToMode("embeddings");
      }
    };

    $tags.addEventListener("change", onToggle);
    $folders.addEventListener("change", onToggle);
  }

  private _initProjectionToggle(): void {
    const m = this.model;
    const $toggle = document.getElementById("projection-toggle");
    if (!$toggle) return;

    const syncActive = () => {
      for (const btn of $toggle.querySelectorAll<HTMLElement>(".toggle-btn")) {
        btn.classList.toggle("active", btn.dataset.method === m.projectionMethod);
      }
    };
    syncActive();

    $toggle.addEventListener("click", (e) => {
      const btn = (e.target as HTMLElement).closest<HTMLElement>(".toggle-btn");
      if (!btn || !btn.dataset.method) return;
      const method = btn.dataset.method as ProjectionMethod;
      if (method === m.projectionMethod) return;
      m.setProjectionMethod(method);
      syncActive();
      m.saveLS();
      if (m.viewMode === "embeddings") {
        void this._fetchProjection();
      } else {
        void this._switchToMode("embeddings");
      }
    });
  }

  /* ── Keyboard shortcuts ─────────────────────────────────── */

  /** True if isolating would still exclude at least one currently visible track. */
  private _wouldIsolateExcludeMore(
    m: Model,
    kind: { type: "selection" } | { type: "folder"; prefix: string },
  ): boolean {
    for (const t of m.allTracks) {
      if (!m.trackInViewUnion(t) || m.isTrackHidden(t)) continue;
      if (kind.type === "selection") {
        if (!m.selected.has(t.path)) return true;
      } else {
        const tf = t.folder ?? "";
        const p = kind.prefix;
        const under =
          p === "" || tf === p || tf.startsWith(p + "/");
        if (!under) return true;
      }
    }
    return false;
  }

  /**
   * Folder prefix for H / S hotkeys: sidebar folder-row hover wins;
   * otherwise the hovered canvas track’s folder (Library root = ``""``).
   */
  private _effectiveHoveredFolderPrefix(): string | null {
    const cv = this.canvas;
    if (cv.hoveredFolderPrefix !== null) return cv.hoveredFolderPrefix;
    if (cv.hoveredIdx >= 0) {
      const t = this.model.tracks[cv.hoveredIdx];
      if (t) return t.folder ?? "";
    }
    return null;
  }

  private _bindKeyboard(): void {
    const m = this.model;
    const hk = this.hotkeyMgr;
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement;

    hk.on("toggle-preview", () => {
      m.hoverPreview = !m.hoverPreview;
      $hov.checked = m.hoverPreview;
      if (!m.hoverPreview) this._stopPreview();
      m.saveLS();
    });

    hk.on("fit-view", () => {
      const hp = this._effectiveHoveredFolderPrefix();
      if (hp !== null) {
        this._fitViewToFolderPercentile(hp);
        this.canvas.scheduleDraw();
        m.saveLS();
        return;
      }
      const vp = m.vp;
      const fit = this._computeFitVP();
      const atFit =
        Math.abs(vp.ox - fit.ox) < 0.002 &&
        Math.abs(vp.oy - fit.oy) < 0.002 &&
        Math.abs(vp.zoom - fit.zoom) < 0.05;
      if (atFit) {
        vp.ox = 0;
        vp.oy = 0;
        vp.zoom = 1;
      } else {
        this._fitViewToTracks();
      }
      this.canvas.scheduleDraw();
      m.saveLS();
    });

    hk.on("focus-folder", () => {
      const hp = this._effectiveHoveredFolderPrefix();
      if (hp === null) return;
      m.selectOnlyInFolder(hp);
      m.saveLS();
    });

    hk.on("exclude-selection", () => {
      const hp = this._effectiveHoveredFolderPrefix();
      if (hp !== null) {
        if (m.folderSubtreeFullyExcluded(hp)) {
          m.revealFolderSubtree(hp);
        } else {
          m.hideFolderTree(hp);
        }
        m.saveLS();
        return;
      }
      if (m.selected.size) {
        m.hidePaths([...m.selected]);
        m.saveLS();
      }
    });

    hk.on("isolate-selection", () => {
      const hp = this._effectiveHoveredFolderPrefix();
      if (hp !== null) {
        if (m.folderSubtreeFullyExcluded(hp)) {
          m.revealThenFocusFolderSubtree(hp);
        } else if (this._wouldIsolateExcludeMore(m, { type: "folder", prefix: hp })) {
          m.focusFolderSubtree(hp);
        } else {
          m.clearExclusions();
        }
        m.saveLS();
        return;
      }
      if (m.selected.size) {
        if (this._wouldIsolateExcludeMore(m, { type: "selection" })) {
          m.focusSelection();
        } else {
          m.clearExclusions();
        }
        m.saveLS();
      }
    });

    hk.on("show-all-tracks", () => {
      m.clearExclusions();
      m.saveLS();
    });

    hk.on("select-all", () => m.selectAll());

    hk.on("undo", () => void this.cmdMgr.undo(m));

    hk.on("toggle-view", () => void this._toggleViewMode());

    hk.on("deselect", () => {
      const cv = this.canvas;
      const mouse = this.mouse;
      if (mouse.mode === "drag") {
        this._revertDragSnapToBaseline();
        this._cleanupDocDrag();
        mouse.mode = "idle";
        mouse.snap = null;
        cv.dragGhosts = null;
        cv.$canvas.style.cursor = "";
        cv.scheduleDraw();
        this.status.update(m);
        this.props.render();
        this.batch.render();
      } else if (mouse.mode === "boxsel") {
        cv.boxSel = null;
        cv.intersectMode = false;
        mouse.mode = "idle";
        cv.scheduleDraw();
      } else if (mouse.mode === "lasso") {
        cv.lassoPoints = [];
        cv.intersectMode = false;
        mouse.mode = "idle";
        cv.scheduleDraw();
      } else {
        m.clearSelection();
      }
    });
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

  /** Returns the set of data-folder-path values (including ancestors) that
   *  contain at least one selected or hovered track. Root is represented
   *  as ".". */
  private _getFoldersWithSelection(): Set<string> {
    const m = this.model;
    const cv = this.canvas;
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
  private _updateFolderHighlights(): void {
    const m = this.model;
    const $tree = this.$folderTree;
    const foldersWithSel = this._getFoldersWithSelection();
    const hoveredTrack =
      this.canvas.hoveredIdx >= 0 ? m.tracks[this.canvas.hoveredIdx] : null;
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

  /* ── Window resize ──────────────────────────────────────── */

  private _bindResize(): void {
    window.addEventListener("resize", () => {
      this.canvas.resize();
      this.canvas.scheduleDraw();
    });
  }

  private _bindViewportChrome(): void {
    document.getElementById("viewport-clear-hides")?.addEventListener("click", () => {
      this.model.clearExclusions();
      this.model.saveLS();
    });
    document
      .getElementById("selected-folders-overlay")
      ?.addEventListener("click", (e) => {
        const label = (e.target as HTMLElement).closest<HTMLElement>(
          ".folder-label[data-path]",
        );
        if (!label) return;
        e.stopPropagation();
        this._onFolderLabelActivate(label, e as MouseEvent);
      });
    document
      .getElementById("selected-folders-overlay")
      ?.addEventListener("mouseleave", () => {
        const fromDrag =
          this.mouse.mode === "drag" || this.mouse.mode === "txform";
        if (fromDrag) return;
        this.canvas.hoveredFolderPrefix = null;
        this.canvas.scheduleDraw();
      });
  }

  private _renderSelectedLeafFoldersOverlay(): void {
    const host = document.getElementById("selected-folders-overlay");
    if (!host) return;
    const counts = new Map<string, number>();
    for (const path of this.model.selected) {
      const t = this.model.trackByPath(path);
      if (!t) continue;
      const folder = t.folder ?? "";
      counts.set(folder, (counts.get(folder) ?? 0) + 1);
    }
    if (counts.size === 0) {
      host.classList.add("hidden");
      host.innerHTML = "";
      return;
    }
    const items = [...counts.entries()].sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    });
    const current = this.model.folder;
    host.innerHTML = "";
    const title = document.createElement("div");
    title.className = "selected-folders-title";
    title.textContent = "Selected folders";
    host.appendChild(title);
    for (const [folder, count] of items) {
      const row = document.createElement("div");
      row.className = "folder-row selected-folder-row";
      row.dataset.folderPath = folder === "" ? "." : folder;
      row.addEventListener("mouseenter", () => {
        row.classList.add("folder-hover");
        this.canvas.hoveredSiblingFolder = null;
        this.canvas.hoveredFolderPrefix = folder;
        this.canvas.scheduleDraw();
      });
      row.addEventListener("mouseleave", () => {
        row.classList.remove("folder-hover");
        const fromDrag =
          this.mouse.mode === "drag" || this.mouse.mode === "txform";
        if (fromDrag) return;
        this.canvas.hoveredFolderPrefix = null;
        this.canvas.scheduleDraw();
      });
      const arrow = document.createElement("span");
      arrow.className = "folder-arrow";
      arrow.textContent = "\u2003";
      row.appendChild(arrow);
      const swatch = document.createElement("span");
      swatch.className = "folder-swatch";
      swatch.style.background = dirColor(folder);
      row.appendChild(swatch);
      const item = document.createElement("span");
      item.className = "folder-label";
      if (current === folder) item.classList.add("active");
      item.dataset.path = folder === "" ? "." : folder;
      item.title = displayFolderPath(folder);
      const label = displayFolderLeaf(folder);
      item.textContent = `${label} (${count})`;
      row.appendChild(item);
      host.appendChild(row);
    }
    host.classList.remove("hidden");
  }
}
