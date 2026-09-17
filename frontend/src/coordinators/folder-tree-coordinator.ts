// ─── FolderTreeCoordinator ─────────────────────────────────────
//
// Owns the sidebar folder tree (HTMX partial), root import/removal, folder-row
// interaction, the selected-folders overlay, and folder highlight/auto-open state.
// Extracted from `Controller` — the tree is self-contained DOM UX whose only
// outward effects are model mutations, the undoable rename command, and a library
// reload after roots change.
//
// Deps are minimal structural interfaces (not the concrete views) so the
// coordinator is unit-testable without the DOM-heavy component classes.

import { type HoverTarget, type Model } from "../model";
import { toast } from "../lib/toast";
import { dirColor } from "../lib/colors";
import { postJSON } from "../lib/api";
import { displayFolderLeaf, displayFolderPath } from "../lib/path-presenter";
import htmx from "htmx.org";

export interface FolderTreeDeps {
  model: Model;
  canvas: {
    hoveredIdx: number;
    hoverTarget: HoverTarget | null;
    hoveredSiblingFolder: string | null;
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

  constructor(private deps: FolderTreeDeps) {
    this.$folderTree = document.getElementById("folder-tree")!;
  }

  /* ── Binding ────────────────────────────────────────────── */

  bind(): void {
    this._bindFolderTree();
    this._bindRootImport();
    this._bindShowExternal();
    this._bindOverlay();
  }

  /** Toggle whether external-root tracks are drawn (persisted on the model). */
  private _bindShowExternal(): void {
    const $toggle = document.getElementById(
      "show-external-toggle",
    ) as HTMLInputElement | null;
    if (!$toggle) return;
    $toggle.checked = this.deps.model.showExternal;
    $toggle.addEventListener("change", () => {
      this.deps.model.setShowExternal($toggle.checked);
      this.deps.model.saveLS();
    });
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

  /** The selected-folders overlay shares label activation and hover semantics. */
  private _bindOverlay(): void {
    const $overlay = document.getElementById("selected-folders-overlay");
    $overlay?.addEventListener("click", (e) => {
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
    for (const row of this.$folderTree.querySelectorAll(".folder-row")) {
      row.addEventListener("mouseenter", () => {
        const path = row.getAttribute("data-folder-path");
        if (path == null) return;
        this.deps.canvas.hoveredSiblingFolder = null;
        this.deps.canvas.hoverTarget = {
          kind: "folder",
          prefix: path === "." ? "" : path,
        };
        this.deps.canvas.scheduleDraw();
      });
      row.addEventListener("mouseleave", () => {
        if (this.deps.isPointerBusy()) return;
        this.deps.canvas.hoverTarget = null;
        this.deps.canvas.scheduleDraw();
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
    this.updateHighlights();
    this.syncActiveLabels();
    this._wireRootRowDrags();
  }

  /** Read root roles from the swapped-in tree and update the model + toggle. */
  private _syncExternalRoots(): void {
    const external = new Set<string>();
    for (const row of this.$folderTree.querySelectorAll<HTMLElement>(
      ".folder-row[data-root-role='external']",
    )) {
      const rid = row.dataset.rootId;
      if (rid) external.add(rid);
    }
    this.deps.model.setExternalRoots(external);
    const $toggle = document.getElementById(
      "show-external-toggle",
    ) as HTMLInputElement | null;
    if ($toggle) $toggle.checked = this.deps.model.showExternal;
    const $label = document.getElementById("show-external-label");
    $label?.classList.toggle("hidden", external.size === 0);
  }

  /* ── Root import / removal ──────────────────────────────── */

  private _bindRootImport(): void {
    const $section = document.getElementById("folder-section");
    const $addRoot = document.getElementById("btn-add-root");
    const $modal = document.getElementById("root-add-modal");
    const $input = document.getElementById("root-add-input") as HTMLInputElement | null;
    const $external = document.getElementById("root-add-external") as HTMLInputElement | null;
    const $ok = document.getElementById("root-add-ok");
    const $cancel = document.getElementById("root-add-cancel");

    const closeModal = () => {
      if (!$modal) return;
      $modal.classList.add("hidden");
    };
    const openModal = async () => {
      if (!$modal) return;
      $modal.classList.remove("hidden");
      if ($external) $external.checked = false;
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
      const role = $external?.checked ? "external" : "library";
      closeModal();
      void this._addRootPath(raw, role);
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
      void this._addRootPath(first, "library");
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
    await this.deps.onRootsChanged();
    toast("Root removed", "ok");
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

    if (t.closest(".folder-reveal-btn")) {
      e.stopPropagation();
      const btn = t.closest(".folder-reveal-btn") as HTMLElement;
      const path = btn.dataset.path ?? ".";
      void postJSON("/api/folders/reveal", { path: path === "." ? "" : path });
      return;
    }

    if (t.closest(".folder-role-btn")) {
      e.stopPropagation();
      const btn = t.closest(".folder-role-btn") as HTMLElement;
      const rootId = btn.dataset.rootId ?? "";
      const role = btn.dataset.role === "external" ? "library" : "external";
      void this._setRootRole(rootId, role);
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

  /** Apply the `active` class to labels matching the current folder. */
  syncActiveLabels(): void {
    const activeNorm = new Set<string>([this.deps.model.folder]);
    for (const label of document.querySelectorAll<HTMLElement>(
      "#folder-tree .folder-label, #selected-folders-overlay .folder-label",
    )) {
      const norm = this._normFolderPath(label.dataset.path ?? ".");
      label.classList.toggle("active", activeNorm.has(norm));
    }
    this._renderSelectedLeafFoldersOverlay();
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

  private _renderSelectedLeafFoldersOverlay(): void {
    const host = document.getElementById("selected-folders-overlay");
    if (!host) return;
    const counts = new Map<string, number>();
    for (const path of this.deps.model.selected) {
      const t = this.deps.model.trackByPath(path);
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
    const current = this.deps.model.folder;
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
        this.deps.canvas.hoveredSiblingFolder = null;
        this.deps.canvas.hoverTarget = { kind: "folder", prefix: folder };
        this.deps.canvas.scheduleDraw();
      });
      row.addEventListener("mouseleave", () => {
        row.classList.remove("folder-hover");
        if (this.deps.isPointerBusy()) return;
        this.deps.canvas.hoverTarget = null;
        this.deps.canvas.scheduleDraw();
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
