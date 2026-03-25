import type { Model, Track } from "./model";
import type {
  CanvasView,
  TreeView,
  TagPanelView,
  PropertiesView,
  BatchView,
  StatusView,
  TxHandle,
} from "./views";
import { toast, showLoad, hideLoad } from "./views";

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
  private tree: TreeView;
  private tagPanel: TagPanelView;
  private props: PropertiesView;
  private batch: BatchView;
  private status: StatusView;

  private mouse: MouseState;
  private previewPath: string | null = null;
  private $audio: HTMLAudioElement;
  private pending = new Map<string, PendingUpdate>();

  constructor(
    model: Model,
    canvasView: CanvasView,
    treeView: TreeView,
    tagPanelView: TagPanelView,
    propsView: PropertiesView,
    batchView: BatchView,
    statusView: StatusView,
  ) {
    this.model = model;
    this.canvas = canvasView;
    this.tree = treeView;
    this.tagPanel = tagPanelView;
    this.props = propsView;
    this.batch = batchView;
    this.status = statusView;

    this.mouse = {
      mode: "idle",
      sx: 0,
      sy: 0,
      lx: 0,
      ly: 0,
      hitIdx: -1,
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
    this._bindCanvas();
    this._bindSidebar();
    this._bindKeyboard();
    this._bindResize();
  }

  /* ── Model events ───────────────────────────────────────── */

  private _wireModel(): void {
    this.model.on("change", () => {
      this.canvas.scheduleDraw();
      this.status.update(this.model);
      this.props.render();
      this.batch.render();
      this.tagPanel.render();
    });
    this.model.on("tags-dirty", () => this.batch.renderDirty());
    this.model.on("tree", () => this.tree.render());
  }

  /* ── View callbacks ─────────────────────────────────────── */

  private _wireViewCallbacks(): void {
    const m = this.model;

    this.tree.onPickFolder = (rel) => {
      m.setFolder(rel);
      setTimeout(() => this.tagPanel.render(), 0);
      m.saveLS();
    };

    this.tree.onSelectFolder = (path) => m.selectInFolder(path);
    this.tree.onCreateSubfolder = (parent) => this._createSubfolder(parent);

    this.tree.onHoverFolder = (path) => {
      this.canvas.hoveredFolderPrefix = path === "." ? "" : path;
      this.canvas.scheduleDraw();
    };
    this.tree.onHoverFolderEnd = () => {
      this.canvas.hoveredFolderPrefix = null;
      this.canvas.scheduleDraw();
    };

    this.tree.onRenameFolder = async (node, newName) => {
      try {
        const res = await postJSON<{
          ok: boolean;
          path: string;
          error?: string;
        }>("/api/folders/rename", { path: node.path, name: newName });
        if (!res.ok) {
          toast(res.error || "Rename failed", "error");
          return false;
        }
        const oldRel = node.path === "." ? "" : node.path;
        m.applyFolderRename(oldRel, res.path);
        await this._loadTree();
        m.saveLS();
        m.emit("change");
        return true;
      } catch {
        return false;
      }
    };

    this.tagPanel.onAxisChange = (which, tag) => {
      m.toggleAxis(which as "axisX" | "axisY", tag);
      this.tagPanel.render();
      m.saveLS();
    };

    this.tagPanel.onFilterChange = (tag, range) => {
      m.setFilterRange(tag, range);
      m.saveLS();
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

  /* ── Data loading ───────────────────────────────────────── */

  private async _loadTree(): Promise<void> {
    const tree = await api<import("./model").FolderNode>("/api/folders");
    this.model.setFolderTree(tree);
  }

  private async _loadLibrary(): Promise<void> {
    const tracks = await api<Track[]>("/api/tracks?folder=&recursive=1");
    this.model.setAllTracks(tracks);
  }

  async init(): Promise<void> {
    showLoad();
    try {
      await Promise.all([this._loadTree(), this._loadLibrary()]);
    } finally {
      hideLoad();
    }
    this.tagPanel.render();
    this.props.render();
    this.batch.render();
    this.canvas.resize();
    this.canvas.scheduleDraw();
    this.status.update(this.model);
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
          this.tree.pendingRename = res.path;
          await this._loadTree();
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

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      mouse.mode = "pan";
      cv.$canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;

    if (e.shiftKey) {
      mouse.mode = "lasso";
      cv.lassoPoints = [[sx, sy]];
      return;
    }

    const txh =
      m.selected.size > 0 ? cv.hitTestTransform(sx, sy) : null;
    if (txh && txh.type !== "move") {
      this._startTxform(txh.type, sx, sy);
      return;
    }

    const tracks = m.tracks;
    const hit = cv.hitTest(sx, sy);
    mouse.hitIdx = hit;

    if (hit >= 0) {
      const hitPath = tracks[hit].path;
      if (!m.selected.has(hitPath)) {
        if (!(e.ctrlKey || e.metaKey)) m.selected.clear();
        m.selected.add(hitPath);
      }
      mouse.mode = "pending";
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
    } else if (txh && txh.type === "move") {
      this._startTxform("move", sx, sy);
      return;
    } else {
      mouse.mode = "boxsel";
      cv.boxSel = { x0: sx, y0: sy, x1: sx, y1: sy };
    }

    cv.scheduleDraw();
    this.status.update(m);
    this.props.render();
    this.batch.render();
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
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "lasso") {
      cv.lassoPoints.push([sx, sy]);
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "txform") {
      const [wx, wy] = cv.s2w(sx, sy);
      this._applyTxform(wx, wy);
      m.emit("tags-dirty");
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "pending") {
      if (Math.hypot(sx - mouse.sx, sy - mouse.sy) > DRAG_THRESH) {
        if (m.axisX || m.axisY) {
          mouse.mode = "drag";
          cv.$canvas.style.cursor = "move";
          document.addEventListener("mousemove", this._onDocDragMove);
          document.addEventListener("mouseup", this._onDocDragUp);
        } else {
          mouse.mode = "idle";
        }
      }
      return;
    }

    if (mouse.mode === "drag") {
      const [swx, swy] = cv.s2w(mouse.sx, mouse.sy);
      const [cwx, cwy] = cv.s2w(sx, sy);
      const dx = cwx - swx,
        dy = cwy - swy;
      for (const [, snap] of mouse.snap!) {
        if (m.axisX) snap.track.tags[m.axisX] = clamp(snap.x + dx);
        if (m.axisY) snap.track.tags[m.axisY] = clamp(snap.y + dy);
      }
      m.emit("tags-dirty");
      cv.scheduleDraw();
      return;
    }

    // Idle hover
    const prev = cv.hoveredIdx;
    cv.hoveredIdx = cv.hitTest(sx, sy);

    const txh: TxHandle | { type: string; cursor: string } | null =
      m.selected.size > 0 ? cv.hitTestTransform(sx, sy) : null;
    if (txh) {
      cv.$canvas.style.cursor = txh.cursor;
    } else {
      cv.$canvas.style.cursor = "";
    }

    if (cv.hoveredIdx !== prev) {
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
      if (bs) {
        const bL = Math.min(bs.x0, bs.x1),
          bR = Math.max(bs.x0, bs.x1);
        const bT = Math.min(bs.y0, bs.y1),
          bB = Math.max(bs.y0, bs.y1);
        const isClick = bR - bL < 4 && bB - bT < 4;
        if (!(e.ctrlKey || e.metaKey)) m.selected.clear();
        if (!isClick) {
          for (const p of cv.positions)
            if (
              p.sx >= bL &&
              p.sx <= bR &&
              p.sy >= bT &&
              p.sy <= bB
            )
              m.selected.add(tracks[p.idx].path);
        }
      }
      mouse.mode = "idle";
      m.emit("change");
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
      if (!(e.ctrlKey || e.metaKey)) m.selected.clear();
      for (const p of cv.positions)
        if (pointInPoly(p.sx, p.sy, cv.lassoPoints))
          m.selected.add(tracks[p.idx].path);
      cv.lassoPoints = [];
    }

    if (mouse.mode === "pending") {
      const hitPath =
        mouse.hitIdx >= 0 ? tracks[mouse.hitIdx]?.path : null;
      if (e.ctrlKey || e.metaKey) {
        if (hitPath && m.selected.has(hitPath)) m.selected.delete(hitPath);
        else if (hitPath) m.selected.add(hitPath);
      } else {
        m.selected.clear();
        if (hitPath) m.selected.add(hitPath);
      }
    }

    if (mouse.mode !== "drag") {
      mouse.mode = "idle";
      mouse.snap = null;
      cv.$canvas.style.cursor = "";
      cv.scheduleDraw();
      this.status.update(m);
      this.props.render();
      this.batch.render();
    }
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
    cv.clearHover();
    cv.$canvas.style.cursor = "";
    this._stopPreview();
    cv.scheduleDraw();
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
      vp.zoom = clamp(vp.zoom * factor, 1, 50);
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
        vp.zoom = clamp(vp.zoom * f, 1, 50);
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
      const dest =
        folderEl.dataset.path === "." ? "" : folderEl.dataset.path!;
      await this._moveSelectedToFolder(dest);
    } else {
      this._commitDrag();
    }

    this.mouse.mode = "idle";
    this.mouse.snap = null;
    this.canvas.$canvas.style.cursor = "";
    this.canvas.scheduleDraw();
    this.status.update(this.model);
    this.props.render();
    this.batch.render();
    this.tagPanel.render();
  };

  private async _moveSelectedToFolder(destPath: string): Promise<void> {
    const m = this.model;
    const paths = [...m.selected];
    if (!paths.length) return;
    try {
      const res = await postJSON<{
        ok: boolean;
        moved: number;
        errors: string[];
      }>("/api/tracks/move", { paths, dest: destPath });
      if (res.errors?.length) toast(res.errors.join("; "), "error");
      if (res.moved)
        toast(
          `Moved ${res.moved} track${res.moved > 1 ? "s" : ""} to /${destPath || "(root)"}`,
          "ok",
        );
      m.selected.clear();
      await Promise.all([this._loadLibrary(), this._loadTree()]);
    } catch {
      /* already toasted */
    }
  }

  /* ── Debounced tag writes ───────────────────────────────── */

  private _commitDrag(): void {
    const m = this.model;
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

  private _computeFitVP(): { ox: number; oy: number; zoom: number } {
    const m = this.model;
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

    const midX = (minX + maxX) / 2,
      midY = (minY + maxY) / 2;
    const margin = 0.1;
    const zx =
      maxX - minX > 0.01 ? (1 - 2 * margin) / (maxX - minX) : 50;
    const zy =
      maxY - minY > 0.01 ? (1 - 2 * margin) / (maxY - minY) : 50;
    const zoom = Math.max(1, Math.min(50, Math.min(zx, zy)));
    return {
      zoom,
      ox: midX - 1 / (2 * zoom),
      oy: midY - 1 / (2 * zoom),
    };
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
    const $addT = document.getElementById("btn-add-tag")!;
    const $addF = document.getElementById("btn-add-folder")!;

    $hov.checked = m.hoverPreview;
    $hov.addEventListener("change", () => {
      m.hoverPreview = $hov.checked;
      if (!m.hoverPreview) this._stopPreview();
      m.saveLS();
    });

    $addT.addEventListener("click", () => {
      const raw = prompt("New tag name:");
      if (!raw || !raw.trim()) return;
      const name = raw.trim().toLowerCase().replace(/\s+/g, "_");
      if (m.tags.includes(name)) {
        toast("Tag already exists", "error");
        return;
      }
      m.addKnownTag(name);
      this.tagPanel.render();
      m.saveLS();
      toast(`Tag "${name}" created`, "ok");
    });

    $addF.addEventListener("click", () => this._createSubfolder(m.folder));
  }

  /* ── Keyboard shortcuts ─────────────────────────────────── */

  private _bindKeyboard(): void {
    const m = this.model;
    const $mod = document.getElementById("shortcuts-modal")!;
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement;

    document.addEventListener("keydown", (e) => {
      if (e.key === "F1") {
        e.preventDefault();
        $mod.classList.remove("hidden");
        return;
      }
      if (
        (e.target as HTMLElement).tagName === "INPUT" ||
        (e.target as HTMLElement).tagName === "TEXTAREA"
      )
        return;

      if (e.key === "m" || e.key === "M") {
        m.hoverPreview = !m.hoverPreview;
        $hov.checked = m.hoverPreview;
        if (!m.hoverPreview) this._stopPreview();
        m.saveLS();
      }
      if (e.key === "h" || e.key === "H") {
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
      }
      if (e.key === "i" || e.key === "I") {
        const cv = this.canvas;
        if (cv.hoveredIdx >= 0) {
          const track = m.tracks[cv.hoveredIdx];
          const folder = track?.folder ?? "";
          if (folder !== m.folder) {
            m.setFolder(folder);
            this._treeActivateFolder(folder);
            setTimeout(() => this.tagPanel.render(), 0);
            m.saveLS();
          }
        }
      }
      if (e.key === "u" || e.key === "U") {
        const slash = m.folder.lastIndexOf("/");
        const parent =
          slash > 0
            ? m.folder.slice(0, slash)
            : m.folder !== ""
              ? ""
              : null;
        if (parent !== null) {
          m.setFolder(parent);
          this._treeActivateFolder(parent);
          setTimeout(() => this.tagPanel.render(), 0);
          m.saveLS();
        }
      }
      if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        m.selectAll();
      }
      if (e.key === "Escape") m.clearSelection();
    });

    document.addEventListener("keyup", (e) => {
      if (e.key === "F1") $mod.classList.add("hidden");
    });
  }

  /* ── Tree folder highlight helper ───────────────────────── */

  private _treeActivateFolder(folder: string): void {
    const $tree = this.tree.$el;
    $tree
      .querySelectorAll(".folder-label.active")
      .forEach((el) => el.classList.remove("active"));
    const dataPath = folder === "" ? "." : folder;
    const label = $tree.querySelector(
      `.folder-label[data-path="${CSS.escape(dataPath)}"]`,
    );
    if (label) label.classList.add("active");
  }

  /* ── Window resize ──────────────────────────────────────── */

  private _bindResize(): void {
    window.addEventListener("resize", () => {
      this.canvas.resize();
      this.canvas.scheduleDraw();
    });
  }
}
