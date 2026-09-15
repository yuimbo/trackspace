import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import { FolderTreeCoordinator } from "./folder-tree-coordinator";

vi.mock("htmx.org", () => ({ default: { ajax: vi.fn() } }));
import htmx from "htmx.org";

// ─── Fakes ───────────────────────────────────────────────────

function makeTrack(over: Partial<Track> = {}): Track {
  return {
    path: "House/a.mp3",
    filename: "a.mp3",
    folder: "House",
    tags: {},
    artist: "",
    title: "",
    ...over,
  };
}

function jsonResponse(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 500,
    statusText: ok ? "OK" : "Internal Server Error",
    json: () => Promise.resolve(body),
  } as Response;
}

const TREE_HTML = `
  <div class="folder-row" data-folder-path=".">
    <span class="folder-label" data-path=".">Library</span>
    <span data-folder-toggle>▾</span>
  </div>
  <div class="folder-children">
    <div class="folder-row" data-folder-path="House">
      <span class="folder-label" data-path="House">House</span>
      <span data-folder-toggle>▸</span>
    </div>
    <div class="folder-children" style="display:none"></div>
    <div class="folder-row" data-folder-path="Techno">
      <span class="folder-label" data-path="Techno">Techno</span>
      <span data-folder-toggle>▸</span>
    </div>
    <div class="folder-children" style="display:none"></div>
  </div>
`;

const SHELL_HTML = `
  <div id="toast-container"></div>
  <div id="folder-section">
    <div id="folder-tree">${TREE_HTML}</div>
  </div>
  <div id="selected-folders-overlay" class="hidden"></div>
  <div id="root-add-modal" class="hidden">
    <input id="root-add-input" />
    <button id="root-add-ok"></button>
    <button id="root-add-cancel"></button>
  </div>
  <button id="btn-add-root"></button>
`;

interface Harness {
  model: Model;
  canvas: ReturnType<typeof makeCanvas>;
  renameFolder: ReturnType<typeof vi.fn>;
  renderTagPanels: ReturnType<typeof vi.fn>;
  onRootsChanged: ReturnType<typeof vi.fn>;
  isPointerBusy: ReturnType<typeof vi.fn>;
  coord: FolderTreeCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
}

function makeCanvas() {
  return {
    hoveredIdx: -1,
    hoveredFolderPrefix: null as string | null,
    hoveredSiblingFolder: null as string | null,
    scheduleDraw: vi.fn(),
  };
}

function makeHarness(routes: (url: string) => unknown = () => ({})): Harness {
  const model = new Model();
  const canvas = makeCanvas();
  const renameFolder = vi.fn();
  const renderTagPanels = vi.fn();
  const onRootsChanged = vi.fn(async () => {});
  const isPointerBusy = vi.fn(() => false);
  const coord = new FolderTreeCoordinator({
    model,
    canvas,
    isPointerBusy,
    renameFolder,
    renderTagPanels,
    onRootsChanged,
  });
  coord.bind();
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return {
    model,
    canvas,
    renameFolder,
    renderTagPanels,
    onRootsChanged,
    isPointerBusy,
    coord,
    fetchSpy,
  };
}

const $tree = () => document.getElementById("folder-tree")!;
const $overlay = () => document.getElementById("selected-folders-overlay")!;
const row = (path: string) =>
  $tree().querySelector<HTMLElement>(
    `.folder-row[data-folder-path="${path}"]`,
  )!;
const label = (path: string) =>
  $tree().querySelector<HTMLElement>(`.folder-label[data-path="${path}"]`)!;
const childrenOf = (path: string) =>
  row(path).nextElementSibling as HTMLElement;

function click(el: Element, init: MouseEventInit = {}): void {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, ...init }));
}

function mouseenter(el: Element): void {
  el.dispatchEvent(new MouseEvent("mouseenter"));
}

function mouseleave(el: Element): void {
  el.dispatchEvent(new MouseEvent("mouseleave"));
}

beforeEach(() => {
  document.body.innerHTML = SHELL_HTML;
  vi.mocked(htmx.ajax).mockReset();
  vi.mocked(htmx.ajax).mockResolvedValue(undefined as never);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ─── Refresh / HTMX ────────────────────────────────────────

describe("FolderTreeCoordinator refresh", () => {
  it("requests the partial with active folder params", async () => {
    const { model, coord } = makeHarness();
    model.folder = "House";
    await coord.refresh();
    expect(htmx.ajax).toHaveBeenCalledTimes(1);
    const [verb, url, opts] = vi.mocked(htmx.ajax).mock.calls[0];
    expect(verb).toBe("get");
    expect(String(url)).toContain("/partials/folder-tree?");
    expect(String(url)).toContain("active=House");
    expect(String(url)).toContain("active_folders=House");
    expect(String(url)).toContain("pending_rename=");
    expect(opts).toMatchObject({ target: "#folder-tree", swap: "innerHTML" });
  });

  it("uses the root marker when no folder is focused", async () => {
    const { coord } = makeHarness();
    await coord.refresh();
    const url = String(vi.mocked(htmx.ajax).mock.calls[0][1]);
    expect(url).toContain("active=.");
  });

  it("rebinds row hover after the swap", async () => {
    const { canvas, coord } = makeHarness();
    await coord.refresh();
    mouseenter(row("House"));
    expect(canvas.hoveredFolderPrefix).toBe("House");
    expect(canvas.scheduleDraw).toHaveBeenCalled();
    mouseleave(row("House"));
    expect(canvas.hoveredFolderPrefix).toBeNull();
  });

  it("restores manually opened folders after a swap", async () => {
    const { coord } = makeHarness();
    // Open House manually, then swap in fresh HTML where its children are closed.
    click(row("House").querySelector("[data-folder-toggle]")!);
    expect(childrenOf("House").style.display).toBe("block");
    vi.mocked(htmx.ajax).mockImplementation(async () => {
      $tree().innerHTML = TREE_HTML;
      return undefined as never;
    });
    await coord.refresh();
    expect(childrenOf("House").style.display).toBe("block");
  });

  it("starts an inline rename when the partial asks for it", async () => {
    const { coord } = makeHarness();
    vi.mocked(htmx.ajax).mockImplementation(async () => {
      $tree().innerHTML = `
        <div class="folder-row" data-folder-path="House">
          <span class="folder-label" data-path="House" data-start-rename>House</span>
        </div>`;
      return undefined as never;
    });
    await coord.refresh();
    expect(
      $tree().querySelector("input.folder-rename-input"),
    ).not.toBeNull();
  });
});

// ─── Toggle ───────────────────────────────────────────────

describe("FolderTreeCoordinator toggle", () => {
  it("closes an open folder row", () => {
    const {  } = makeHarness();
    childrenOf("House").style.display = "block";
    click(row("House").querySelector("[data-folder-toggle]")!);
    expect(childrenOf("House").style.display).toBe("none");
  });

  it("opens a closed folder row", () => {
    const {  } = makeHarness();
    childrenOf("House").style.display = "none";
    click(row("House").querySelector("[data-folder-toggle]")!);
    expect(childrenOf("House").style.display).toBe("block");
  });
});

// ─── Label activation ─────────────────────────────────────

describe("FolderTreeCoordinator label activation", () => {
  it("plain click focuses the folder and selects only its tracks", () => {
    const { model } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    click(label("House"));
    expect(model.folder).toBe("House");
    expect([...model.selected]).toEqual(["House/a.mp3"]);
  });

  it("cmd/ctrl click adds the folder's tracks to the selection", () => {
    const { model } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    model.selected.add("Techno/b.mp3");
    click(label("House"), { ctrlKey: true });
    expect([...model.selected].sort()).toEqual(["House/a.mp3", "Techno/b.mp3"]);
    expect(model.folder).toBe("House");
  });

  it("shift click selects the folder range from the anchor", () => {
    const { model } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    click(label("."));
    click(label("Techno"), { shiftKey: true });
    expect([...model.selected].sort()).toEqual([
      "House/a.mp3",
      "Techno/b.mp3",
    ]);
    expect(model.folder).toBe("Techno");
  });

  it("alt click excludes the folder subtree", () => {
    const { model } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    click(label("House"), { altKey: true });
    expect(model.hiddenFolderPrefixes.has("House")).toBe(true);
  });
});

// ─── Inline rename ───────────────────────────────────────

describe("FolderTreeCoordinator inline rename", () => {
  it("commits a rename on Enter", () => {
    const { renameFolder } = makeHarness();
    label("House").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    const input = $tree().querySelector<HTMLInputElement>(
      "input.folder-rename-input",
    )!;
    input.value = "Deep House";
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(renameFolder).toHaveBeenCalledWith("House", "Deep House");
  });

  it("cancels a rename on Escape", () => {
    const { renameFolder } = makeHarness();
    label("House").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    const input = $tree().querySelector<HTMLInputElement>(
      "input.folder-rename-input",
    )!;
    input.value = "Deep House";
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(renameFolder).not.toHaveBeenCalled();
    expect(label("House")).not.toBeNull();
  });

  it("does not rename when the name is unchanged", () => {
    const { renameFolder } = makeHarness();
    label("House").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    const input = $tree().querySelector<HTMLInputElement>(
      "input.folder-rename-input",
    )!;
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(renameFolder).not.toHaveBeenCalled();
  });
});

// ─── Subfolder creation ──────────────────────────────────

describe("FolderTreeCoordinator subfolder creation", () => {
  it("retries on a name collision and opens a rename", async () => {
    let creates = 0;
    const { fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/folders/create")) {
        creates++;
        return creates === 1
          ? { ok: false, error: "Already exists" }
          : { ok: true, path: "House/New folder 2" };
      }
      return {};
    });
    const btn = document.createElement("span");
    btn.className = "folder-add-btn";
    btn.dataset.parent = "House";
    row("House").appendChild(btn);
    click(btn);
    await vi.waitFor(() => expect(creates).toBe(2));
    const urls = fetchSpy.mock.calls.map(([u]) => String(u));
    expect(urls.filter((u) => u.startsWith("/api/folders/create"))).toHaveLength(2);
    const refreshUrl = String(vi.mocked(htmx.ajax).mock.calls.at(-1)![1]);
    expect(refreshUrl).toContain("pending_rename=House%2FNew+folder+2");
  });
});

// ─── Highlights ─────────────────────────────────────────

describe("FolderTreeCoordinator highlights", () => {
  it("glows rows containing selected tracks and auto-opens ancestors", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    model.selected.add("House/a.mp3");
    coord.updateHighlights();
    expect(row("House").classList.contains("folder-sel")).toBe(true);
    expect(row("Techno").classList.contains("folder-sel")).toBe(false);
    expect(childrenOf("House").style.display).toBe("block");
  });

  it("closes auto-opened folders that lost their selection", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "House/a.mp3", folder: "House" })]);
    model.selected.add("House/a.mp3");
    coord.updateHighlights();
    expect(childrenOf("House").style.display).toBe("block");
    model.selected.clear();
    coord.updateHighlights();
    expect(childrenOf("House").style.display).toBe("none");
  });

  it("keeps manually opened folders open without selection", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "House/a.mp3", folder: "House" })]);
    childrenOf("House").style.display = "none";
    click(row("House").querySelector("[data-folder-toggle]")!);
    coord.updateHighlights();
    expect(childrenOf("House").style.display).toBe("block");
  });

  it("marks a fully excluded folder", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "Techno/b.mp3", folder: "Techno" }),
    ]);
    model.hideFolderTree("House");
    coord.updateHighlights();
    expect(row("House").classList.contains("folder-fully-excluded")).toBe(true);
    expect(row("Techno").classList.contains("folder-fully-excluded")).toBe(false);
  });
});

// ─── Active labels + overlay ─────────────────────────────

describe("FolderTreeCoordinator active labels and overlay", () => {
  it("marks the focused folder label active", () => {
    const { model, coord } = makeHarness();
    model.folder = "House";
    coord.syncActiveLabels();
    expect(label("House").classList.contains("active")).toBe(true);
    expect(label("Techno").classList.contains("active")).toBe(false);
  });

  it("renders the selected-folders overlay sorted by count", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([
      makeTrack({ path: "House/a.mp3", folder: "House" }),
      makeTrack({ path: "House/b.mp3", folder: "House" }),
      makeTrack({ path: "Techno/c.mp3", folder: "Techno" }),
    ]);
    model.selected.add("House/a.mp3");
    model.selected.add("House/b.mp3");
    model.selected.add("Techno/c.mp3");
    coord.syncActiveLabels();
    expect($overlay().classList.contains("hidden")).toBe(false);
    const labels = $overlay().querySelectorAll(".folder-label");
    expect(labels).toHaveLength(2);
    expect(labels[0].textContent).toContain("(2)");
    expect(labels[1].textContent).toContain("(1)");
  });

  it("hides the overlay when nothing is selected", () => {
    const { model, coord } = makeHarness();
    model.setAllTracks([makeTrack({ path: "House/a.mp3", folder: "House" })]);
    coord.syncActiveLabels();
    expect($overlay().classList.contains("hidden")).toBe(true);
    expect($overlay().innerHTML).toBe("");
  });
});

// ─── Hover ──────────────────────────────────────────────

describe("FolderTreeCoordinator hover", () => {
  it("sets and clears the hovered folder prefix", async () => {
    const { canvas, coord } = makeHarness();
    await coord.refresh();
    mouseenter(row("House"));
    expect(canvas.hoveredFolderPrefix).toBe("House");
    mouseleave(row("House"));
    expect(canvas.hoveredFolderPrefix).toBeNull();
  });

  it("keeps the hover while a canvas drag is active", async () => {
    const { canvas, coord, isPointerBusy } = makeHarness();
    await coord.refresh();
    isPointerBusy.mockReturnValue(true);
    mouseenter(row("House"));
    canvas.hoveredFolderPrefix = "House";
    mouseleave(row("House"));
    expect(canvas.hoveredFolderPrefix).toBe("House");
  });
});

// ─── Roots ─────────────────────────────────────────────

describe("FolderTreeCoordinator roots", () => {
  it("adds a root from a Finder drop and reloads", async () => {
    const { onRootsChanged, fetchSpy } = makeHarness(() => ({
      ok: true,
      changed: true,
    }));
    const ev = new Event("drop", { bubbles: true, cancelable: true });
    Object.assign(ev, {
      dataTransfer: {
        files: [{ path: "/music/root" }],
        getData: () => "",
      },
    });
    document.getElementById("folder-section")!.dispatchEvent(ev);
    await vi.waitFor(() => expect(onRootsChanged).toHaveBeenCalled());
    const urls = fetchSpy.mock.calls.map(([u]) => String(u));
    expect(urls).toContain("/api/roots/add");
  });

  it("ignores a drop without an absolute path", () => {
    const { onRootsChanged, fetchSpy } = makeHarness();
    const ev = new Event("drop", { bubbles: true, cancelable: true });
    Object.assign(ev, {
      dataTransfer: { files: [], getData: () => "relative/path" },
    });
    document.getElementById("folder-section")!.dispatchEvent(ev);
    expect(onRootsChanged).not.toHaveBeenCalled();
    const urls = fetchSpy.mock.calls.map(([u]) => String(u));
    expect(urls.filter((u) => u.includes("/api/roots/add"))).toHaveLength(0);
  });

  it("removes a root dragged out of the sidebar", async () => {
    const { coord, onRootsChanged, fetchSpy } = makeHarness(() => ({ ok: true }));
    const rootRow = document.createElement("div");
    rootRow.className = "folder-row";
    rootRow.dataset.rootId = "r1";
    $tree().appendChild(rootRow);
    await coord.refresh();
    rootRow.dispatchEvent(new Event("dragstart", { bubbles: true }));
    rootRow.dispatchEvent(
      new MouseEvent("dragend", { bubbles: true, clientX: 900, clientY: 900 }),
    );
    await vi.waitFor(() => expect(onRootsChanged).toHaveBeenCalled());
    const urls = fetchSpy.mock.calls.map(([u]) => String(u));
    expect(urls).toContain("/api/roots/remove");
  });
});
