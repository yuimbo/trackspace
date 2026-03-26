import type { Model } from "./model";
import { toast } from "./lib/toast";

// ─── Types ───────────────────────────────────────────────────
export type PostEffect = "change" | "tags-dirty" | "change+htmx";

export interface Command {
  readonly description: string;
  readonly effect: PostEffect;
  execute(m: Model): void;
  undo(m: Model): void;
  commit(): Promise<void>;
  undoCommit?(): Promise<void>;
}

// ─── API helper (mirrors controller's postJSON) ──────────────
function postJSON<T = unknown>(url: string, body: unknown): Promise<T> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  });
}

// ─── CommandManager ──────────────────────────────────────────
export class CommandManager {
  private _stack: Command[] = [];
  private _maxDepth = 50;
  private _committing = false;
  private _afterEffect: (effect: PostEffect) => void;

  constructor(afterEffect: (effect: PostEffect) => void) {
    this._afterEffect = afterEffect;
  }

  get canUndo(): boolean {
    return this._stack.length > 0 && !this._committing;
  }

  async run(cmd: Command, m: Model): Promise<void> {
    cmd.execute(m);
    this._afterEffect(cmd.effect);

    this._stack.push(cmd);
    if (this._stack.length > this._maxDepth) this._stack.shift();

    this._committing = true;
    try {
      await cmd.commit();
    } catch {
      cmd.undo(m);
      this._stack.pop();
      this._afterEffect(cmd.effect);
      toast("Action failed — reverted", "error");
    } finally {
      this._committing = false;
    }
  }

  async undo(m: Model): Promise<void> {
    if (!this.canUndo) {
      toast(this._committing ? "Undo blocked — action in progress" : "Nothing to undo", "ok");
      return;
    }
    const cmd = this._stack.pop()!;
    cmd.undo(m);
    this._afterEffect(cmd.effect);
    toast(`Undo: ${cmd.description}`, "ok");

    if (cmd.undoCommit) {
      try {
        await cmd.undoCommit();
      } catch {
        toast("Undo sync failed — local state may differ from disk", "error");
      }
    }
  }
}

// ═════════════════════════════════════════════════════════════
//  Concrete Commands
// ═════════════════════════════════════════════════════════════

// ─── RenameTagCommand ────────────────────────────────────────
export class RenameTagCommand implements Command {
  readonly effect: PostEffect = "change";
  readonly description: string;

  constructor(
    private oldName: string,
    private newName: string,
  ) {
    this.description = `Rename tag "${oldName}" → "${newName}"`;
  }

  execute(m: Model): void {
    m.renameTag(this.oldName, this.newName);
  }
  undo(m: Model): void {
    m.renameTag(this.newName, this.oldName);
  }

  async commit(): Promise<void> {
    const res = await postJSON<{ ok: boolean }>("/api/tags/rename", {
      old: this.oldName,
      new: this.newName,
      folder: "",
      recursive: true,
    });
    if (!res.ok) throw new Error("Tag rename rejected by server");
  }
  async undoCommit(): Promise<void> {
    await postJSON("/api/tags/rename", {
      old: this.newName,
      new: this.oldName,
      folder: "",
      recursive: true,
    });
  }
}

// ─── RenameFolderCommand ─────────────────────────────────────
export class RenameFolderCommand implements Command {
  readonly effect: PostEffect = "change+htmx";
  readonly description: string;
  private serverNewRel = "";

  constructor(
    private path: string,
    private newName: string,
  ) {
    this.description = `Rename folder "${path}" → "${newName}"`;
  }

  get oldRel(): string {
    return this.path === "." ? "" : this.path;
  }

  execute(m: Model): void {
    const parent = this.oldRel.includes("/")
      ? this.oldRel.slice(0, this.oldRel.lastIndexOf("/") + 1)
      : "";
    this.serverNewRel = parent + this.newName;
    m.applyFolderRename(this.oldRel, this.serverNewRel);
  }
  undo(m: Model): void {
    m.applyFolderRename(this.serverNewRel, this.oldRel);
  }

  async commit(): Promise<void> {
    const res = await postJSON<{ ok: boolean; path: string; error?: string }>(
      "/api/folders/rename",
      { path: this.path, name: this.newName },
    );
    if (!res.ok) throw new Error(res.error || "Folder rename failed");
    this.serverNewRel = res.path;
  }
  async undoCommit(): Promise<void> {
    const oldName = this.oldRel.includes("/")
      ? this.oldRel.slice(this.oldRel.lastIndexOf("/") + 1)
      : this.oldRel;
    await postJSON("/api/folders/rename", {
      path: this.serverNewRel,
      name: oldName,
    });
  }
}

// ─── MoveTracksCommand ───────────────────────────────────────
export class MoveTracksCommand implements Command {
  readonly effect: PostEffect = "change+htmx";
  readonly description: string;
  private originalFolders = new Map<string, string>();

  constructor(
    private paths: string[],
    private dest: string,
  ) {
    this.description = `Move ${paths.length} track${paths.length > 1 ? "s" : ""} to /${dest || "(root)"}`;
  }

  execute(m: Model): void {
    for (const p of this.paths) {
      const t = m.trackByPath(p);
      if (t) this.originalFolders.set(p, t.folder);
    }
    m.applyTracksMoved(this.paths, this.dest);
  }

  undo(m: Model): void {
    const byFolder = new Map<string, string[]>();
    for (const [oldPath, origFolder] of this.originalFolders) {
      const filename = oldPath.includes("/")
        ? oldPath.slice(oldPath.lastIndexOf("/") + 1)
        : oldPath;
      const currentPath = this.dest
        ? `${this.dest}/${filename}`
        : filename;
      const arr = byFolder.get(origFolder) ?? [];
      arr.push(currentPath);
      byFolder.set(origFolder, arr);
    }
    for (const [folder, paths] of byFolder) {
      m.applyTracksMoved(paths, folder);
    }
  }

  async commit(): Promise<void> {
    const res = await postJSON<{
      ok: boolean;
      moved: number;
      errors: string[];
    }>("/api/tracks/move", { paths: this.paths, dest: this.dest });
    if (res.errors?.length) throw new Error(res.errors.join("; "));
    if (!res.moved) throw new Error("No tracks moved");
  }

  async undoCommit(): Promise<void> {
    const byFolder = new Map<string, string[]>();
    for (const [oldPath, origFolder] of this.originalFolders) {
      const filename = oldPath.includes("/")
        ? oldPath.slice(oldPath.lastIndexOf("/") + 1)
        : oldPath;
      const currentPath = this.dest
        ? `${this.dest}/${filename}`
        : filename;
      const arr = byFolder.get(origFolder) ?? [];
      arr.push(currentPath);
      byFolder.set(origFolder, arr);
    }
    for (const [folder, paths] of byFolder) {
      await postJSON("/api/tracks/move", { paths, dest: folder });
    }
  }
}
