import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model } from "../model";
import { HeadCoordinator } from "./head-coordinator";
import { toast } from "../lib/toast";

vi.mock("../lib/toast", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/toast")>();
  return { ...actual, toast: vi.fn() };
});

function jsonResponse(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 400,
    statusText: ok ? "OK" : "Bad Request",
    json: () => Promise.resolve(body),
  } as Response;
}

const PANEL_HTML = `
  <div id="toast-container"></div>
  <button id="btn-train-supervised"></button>
  <div id="head-train-folders"></div>
  <details id="head-train-folders-details"></details>
  <button id="btn-head-folders-all"></button>
  <details id="head-imprint-details">
    <input id="head-imprint-path" />
    <input id="imprint-generic-toggle" type="checkbox" checked />
  </details>
  <input id="head-import-file" type="file" hidden />
  <input id="head-compare-file" type="file" hidden />
  <button id="btn-export-head"></button>
  <button id="btn-import-head"></button>
  <button id="btn-compare-head"></button>
  <button id="btn-imprint-analyze"></button>
  <button id="btn-imprint-plan"></button>
  <button id="btn-imprint-apply"></button>
  <button id="btn-imprint-undo"></button>
  <button id="btn-imprint-discard" hidden></button>
  <p id="imprint-note"></p>
  <p id="head-note"></p>
`;

const flush = () => new Promise((r) => setTimeout(r, 0));

interface Harness {
  model: Model;
  coord: HeadCoordinator;
  fetchSpy: ReturnType<typeof vi.fn>;
  refreshProjection: ReturnType<typeof vi.fn>;
  refreshClusters: ReturnType<typeof vi.fn>;
  onRootsChanged: ReturnType<typeof vi.fn>;
}

function makeHarness(
  routes: (url: string) => unknown = () => ({}),
  model = new Model(),
): Harness {
  const refreshProjection = vi.fn();
  const refreshClusters = vi.fn();
  const onRootsChanged = vi.fn();
  const coord = new HeadCoordinator({
    model,
    refreshProjection,
    refreshClusters,
    onRootsChanged,
  });
  const fetchSpy = vi.fn((input: RequestInfo | URL) =>
    Promise.resolve(jsonResponse(routes(String(input)))),
  );
  vi.stubGlobal("fetch", fetchSpy);
  return { model, coord, fetchSpy, refreshProjection, refreshClusters, onRootsChanged };
}

function postBody(fetchSpy: ReturnType<typeof vi.fn>, url: string): unknown {
  const call = fetchSpy.mock.calls
    .filter(
      ([u, init]) =>
        String(u) === url && (init as RequestInit | undefined)?.method === "POST",
    )
    .at(-1);
  expect(call).toBeDefined();
  return JSON.parse((call![1] as RequestInit).body as string);
}

beforeEach(() => {
  document.body.innerHTML = PANEL_HTML;
  localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("HeadCoordinator training", () => {
  function openTrainFolders(coord: HeadCoordinator): void {
    coord.bindPanel();
    const $details = document.getElementById(
      "head-train-folders-details",
    ) as HTMLDetailsElement;
    $details.open = true;
    $details.dispatchEvent(new Event("toggle"));
  }

  it("queues a supervised training run from the Train button", async () => {
    const { coord, fetchSpy } = makeHarness();
    coord.bindPanel();
    document.getElementById("btn-train-supervised")!.click();
    await flush();

    const post = fetchSpy.mock.calls.find(
      ([u, init]) =>
        String(u) === "/api/supervised/train" &&
        (init as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeDefined();
  });

  it("trains on the checked folders only and persists the selection", async () => {
    const { coord, fetchSpy, model } = makeHarness((url) =>
      url === "/api/folders"
        ? {
            roots: [
              {
                name: "Music",
                path: "abc",
                children: [{ name: "Techno", path: "abc/Techno" }],
              },
            ],
          }
        : {},
    );
    const save = vi.spyOn(model, "saveLS");
    openTrainFolders(coord);
    await flush();

    const techno = document.querySelector<HTMLInputElement>(
      '#head-train-folders input[data-folder="abc/Techno"]',
    )!;
    expect(techno).toBeTruthy();
    techno.checked = true;
    techno.dispatchEvent(new Event("change"));

    document.getElementById("btn-train-supervised")!.click();
    await flush();

    expect(postBody(fetchSpy, "/api/supervised/train")).toEqual({
      folders: ["abc/Techno"],
    });
    // The checked folders survive a reload via the model + localStorage.
    expect(model.trainFolders).toEqual(["abc/Techno"]);
    expect(save).toHaveBeenCalled();
  });

  it("initialises the checked folders from the persisted model", async () => {
    const model = new Model();
    model.trainFolders = ["abc/Techno"];
    const { coord } = makeHarness(
      (url) =>
        url === "/api/folders"
          ? {
              roots: [
                {
                  name: "Music",
                  path: "abc",
                  children: [{ name: "Techno", path: "abc/Techno" }],
                },
              ],
            }
          : {},
      model,
    );
    openTrainFolders(coord);
    await flush();
    const techno = document.querySelector<HTMLInputElement>(
      '#head-train-folders input[data-folder="abc/Techno"]',
    )!;
    expect(techno.checked).toBe(true);
  });

  it("clears the selection on All roots", async () => {
    const { coord, model } = makeHarness((url) =>
      url === "/api/folders"
        ? { roots: [{ name: "Music", path: "abc" }] }
        : {},
    );
    openTrainFolders(coord);
    await flush();
    const box = document.querySelector<HTMLInputElement>(
      '#head-train-folders input[data-folder="abc"]',
    )!;
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    expect(model.trainFolders).toEqual(["abc"]);

    document.getElementById("btn-head-folders-all")!.click();
    expect(model.trainFolders).toEqual([]);
  });
});

describe("HeadCoordinator imprint", () => {
  it("analyzes, plans and applies an imprint, enabling undo", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const { coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/supervised/imprint/analyze")) {
        return {
          ok: true,
          root_id: "tmp",
          folder: "tmp/Downloads",
          tracks: 3,
          analyzed: 3,
          pending: 0,
        };
      }
      if (url.startsWith("/api/supervised/fanout")) {
        return {
          ok: true,
          tracks: 3,
          moved: 3,
          unmatched: 0,
          plan: [{ target: "tmp/Downloads/Techno/a.mp3", folder: "abc/Techno" }],
        };
      }
      return {};
    });
    coord.bindPanel();
    const $path = document.getElementById(
      "head-imprint-path",
    ) as HTMLInputElement;
    $path.value = "/tmp/Downloads";

    document.getElementById("btn-imprint-analyze")!.click();
    await flush();
    expect(document.getElementById("imprint-note")!.textContent).toContain("ready");

    document.getElementById("btn-imprint-plan")!.click();
    await flush();
    expect(document.getElementById("imprint-note")!.textContent).toContain(
      "Planned 3 tracks",
    );

    document.getElementById("btn-imprint-apply")!.click();
    await flush();
    expect(document.getElementById("imprint-note")!.textContent).toContain(
      "Moved 3",
    );
    expect(postBody(fetchSpy, "/api/supervised/fanout")).toEqual({
      folder: "tmp/Downloads",
      recursive: true,
      apply: true,
      generic: true,
    });
    // The ephemeral imprint root is dropped once the moves are done.
    expect(
      fetchSpy.mock.calls.some(
        ([u]) => String(u) === "/api/supervised/imprint/finish",
      ),
    ).toBe(true);
    // Undo is now available for the moves that just landed.
    const $undo = document.getElementById(
      "btn-imprint-undo",
    ) as HTMLButtonElement;
    expect($undo.disabled).toBe(false);
  });

  it("reports unmatched tracks in the plan", async () => {
    const { coord } = makeHarness((url) => {
      if (url.startsWith("/api/supervised/imprint/analyze")) {
        return {
          ok: true,
          root_id: "tmp",
          folder: "tmp/Downloads",
          tracks: 3,
          analyzed: 3,
          pending: 0,
        };
      }
      if (url.startsWith("/api/supervised/fanout")) {
        return {
          ok: true,
          tracks: 3,
          moved: 2,
          unmatched: 1,
          plan: [
            { target: "tmp/Downloads/Techno/a.mp3", folder: "abc/Techno" },
            { target: "tmp/Downloads/c.mp3", folder: "abc/Techno" },
          ],
        };
      }
      return {};
    });
    coord.bindPanel();
    (
      document.getElementById("head-imprint-path") as HTMLInputElement
    ).value = "/tmp/Downloads";
    document.getElementById("btn-imprint-analyze")!.click();
    await flush();
    document.getElementById("btn-imprint-plan")!.click();
    await flush();
    expect(document.getElementById("imprint-note")!.textContent).toContain(
      "1 left unmatched",
    );
  });

  it("sends generic: false when the generic toggle is off", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const { coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/supervised/imprint/analyze")) {
        return {
          ok: true,
          root_id: "tmp",
          folder: "tmp/Downloads",
          tracks: 1,
          analyzed: 1,
          pending: 0,
        };
      }
      if (url.startsWith("/api/supervised/fanout")) {
        return { ok: true, tracks: 1, moved: 1, generic: 0, unmatched: 0, plan: [] };
      }
      return {};
    });
    coord.bindPanel();
    (
      document.getElementById("imprint-generic-toggle") as HTMLInputElement
    ).checked = false;
    (
      document.getElementById("head-imprint-path") as HTMLInputElement
    ).value = "/tmp/Downloads";
    document.getElementById("btn-imprint-analyze")!.click();
    await flush();
    document.getElementById("btn-imprint-apply")!.click();
    await flush();

    expect(postBody(fetchSpy, "/api/supervised/fanout")).toMatchObject({
      recursive: true,
      generic: false,
    });
  });

  it("reports generic buckets in the plan note", async () => {
    const { coord } = makeHarness((url) => {
      if (url.startsWith("/api/supervised/imprint/analyze")) {
        return {
          ok: true,
          root_id: "tmp",
          folder: "tmp/Downloads",
          tracks: 3,
          analyzed: 3,
          pending: 0,
        };
      }
      if (url.startsWith("/api/supervised/fanout")) {
        return {
          ok: true,
          tracks: 3,
          moved: 0,
          generic: 2,
          unmatched: 1,
          plan: [
            { target: "tmp/Downloads/Generic/Techno/a.mp3", folder: "Generic/Techno" },
          ],
        };
      }
      return {};
    });
    coord.bindPanel();
    (
      document.getElementById("head-imprint-path") as HTMLInputElement
    ).value = "/tmp/Downloads";
    document.getElementById("btn-imprint-analyze")!.click();
    await flush();
    document.getElementById("btn-imprint-plan")!.click();
    await flush();
    const note = document.getElementById("imprint-note")!.textContent ?? "";
    expect(note).toContain("2 into Generic/");
    expect(note).toContain("1 left unmatched");
  });

  it("prefills the imprint target from a root row", () => {
    const { coord } = makeHarness();
    coord.bindPanel();
    coord.prefillImprint("/tmp/Downloads");

    const $path = document.getElementById(
      "head-imprint-path",
    ) as HTMLInputElement;
    const $details = document.getElementById(
      "head-imprint-details",
    ) as HTMLDetailsElement;
    expect($path.value).toBe("/tmp/Downloads");
    expect($details.open).toBe(true);
  });

  it("reveals the Organise tab when prefilling an imprint target", () => {
    const showRightPanel = vi.fn();
    const coord = new HeadCoordinator({ model: new Model(), showRightPanel });
    coord.prefillImprint("/tmp/Downloads");
    expect(showRightPanel).toHaveBeenCalledWith("organise");
  });

  it("undoes the last fan-out from the Undo button", async () => {
    const { coord, fetchSpy, refreshProjection } = makeHarness((url) => {
      if (url === "/api/supervised/fanout/undo") {
        return { ok: true, batch_id: 1, total: 2, restored: 2, skipped: 0 };
      }
      return {};
    });
    coord.bindPanel();
    const $undo = document.getElementById(
      "btn-imprint-undo",
    ) as HTMLButtonElement;
    $undo.disabled = false;
    $undo.click();
    await flush();

    expect(postBody(fetchSpy, "/api/supervised/fanout/undo")).toEqual({});
    expect(document.getElementById("imprint-note")!.textContent).toContain(
      "Restored 2",
    );
    expect($undo.disabled).toBe(true);
    expect(refreshProjection).toHaveBeenCalled();
  });

  it("reveals Discard for an unfinished imprint on start", async () => {
    const { coord, fetchSpy } = makeHarness((url) => {
      if (url.startsWith("/api/supervised/imprint/temp_roots")) {
        return { ok: true, roots: [{ root_id: "tmp", path: "/tmp/Downloads" }] };
      }
      return {};
    });
    coord.bindPanel();
    coord.start();
    await flush();

    const $discard = document.getElementById(
      "btn-imprint-discard",
    ) as HTMLButtonElement;
    expect($discard.hidden).toBe(false);
    expect(document.getElementById("imprint-note")!.textContent).toContain(
      "1 unfinished",
    );

    $discard.click();
    await flush();
    expect(postBody(fetchSpy, "/api/supervised/imprint/discard")).toEqual({});
    expect($discard.hidden).toBe(true);
  });
});

describe("HeadCoordinator import", () => {
  it("retries with force after a version-mismatch refusal", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    let sawForce = false;
    const refreshProjection = vi.fn();
    const model = new Model();
    const coord = new HeadCoordinator({ model, refreshProjection });
    const fetchSpy = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      const form = init?.body as FormData;
      if (form.get("force") === "1") {
        sawForce = true;
        return Promise.resolve(
          jsonResponse({ ok: true, head: { run_id: 2, n_classes: 4 } }),
        );
      }
      return Promise.resolve(
        jsonResponse(
          {
            ok: false,
            error:
              "head was trained on different encoder versions: maest (trained 1, now 2)",
          },
          false,
        ),
      );
    });
    vi.stubGlobal("fetch", fetchSpy);
    coord.bindPanel();

    const $file = document.getElementById("head-import-file") as HTMLInputElement;
    const file = new File([new Uint8Array([1])], "head.pt");
    Object.defineProperty($file, "files", { value: [file] });
    $file.dispatchEvent(new Event("change"));
    await flush();

    expect(sawForce).toBe(true);
    expect(document.getElementById("head-note")!.textContent).toContain(
      "run 2",
    );
    expect(refreshProjection).toHaveBeenCalled();
    expect(toast).toHaveBeenCalled();
  });
});
