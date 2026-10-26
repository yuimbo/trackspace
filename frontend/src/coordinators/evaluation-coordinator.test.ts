import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import {
  EvaluationCoordinator,
  type EvaluationDeps,
} from "./evaluation-coordinator";
import { isEvalReadOnly, setEvalReadOnly } from "../lib/eval-readonly";
import { loadRating } from "../lib/eval-ratings";
import type { EvalToolbarView } from "../components/eval-toolbar-view";

// ─── Fakes ───────────────────────────────────────────────────

function makeTrack(path: string): Track {
  return { path, filename: path, folder: "", tags: {}, artist: "", title: "" };
}

const MANIFEST = {
  schema_version: 1,
  status: "complete",
  dataset_fingerprint: "fp-1",
  recipes: {
    baseline: {
      status: "complete",
      label: "PCA50 → openTSNE",
      tracks: 3,
      timings: { pca: 0.2, tsne: 1.1, alignment: 0 },
      aligned_to: null,
    },
    pca2: {
      status: "complete",
      label: "PCA2 → openTSNE",
      tracks: 3,
      timings: { pca: 0.1, tsne: 0.9, alignment: 0.01 },
      aligned_to: "baseline",
    },
  },
};

function artifact(
  id: string,
  rev: string,
  dims: 2 | 3 = 2,
  group = id,
): unknown {
  const positions = [
    { path: "a.mp3", x: 0.1, y: 0.2 },
    { path: "b.mp3", x: 0.8, y: 0.9 },
    { path: "x.mp3", x: 0.5, y: 0.5 },
  ].map((p, i) => (dims === 3 ? { ...p, z: 0.2 + i * 0.1 } : p));
  return {
    ok: true,
    id,
    label: "Recipe",
    revision: rev,
    dataset_fingerprint: "fp-1",
    tracks: 3,
    aligned_to: null,
    timings: { pca: 0.2, tsne: 1.1 },
    dims,
    group,
    positions,
  };
}

const PAIRED_MANIFEST = {
  schema_version: 1,
  status: "complete",
  dataset_fingerprint: "fp-paired",
  recipes: {
    a: { status: "complete", label: "Map A base", dims: 2, group: "a" },
    a_3d: { status: "complete", label: "Map A base (3D)", dims: 3, group: "a" },
    b: { status: "complete", label: "Map B base", dims: 2, group: "b" },
    unpaired: { status: "complete", label: "Unpaired base", dims: 2, group: "unpaired" },
  },
};

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const flush = () => new Promise<void>((r) => setTimeout(r, 0));

function makeFakeView(): EvalToolbarView {
  return {
    render: vi.fn(),
    onSelect: null,
    onPrev: null,
    onNext: null,
    onResetCamera: null,
    onToggleReveal: null,
    onRate: null,
    onNote: null,
    onExport: null,
    onExit: null,
  } as unknown as EvalToolbarView;
}

interface Harness {
  model: Model;
  view: EvalToolbarView;
  applyPositions: ReturnType<typeof vi.fn>;
  resetCamera: ReturnType<typeof vi.fn>;
  fetchManifest: ReturnType<typeof vi.fn>;
  fetchArtifact: ReturnType<typeof vi.fn>;
  coord: EvaluationCoordinator;
  manifestDefer: ReturnType<typeof deferred<unknown>>;
  artifactDefers: Map<string, ReturnType<typeof deferred<unknown>>>;
}

function makeHarness(): Harness {
  const model = new Model();
  model.setAllTracks([
    makeTrack("a.mp3"),
    makeTrack("b.mp3"),
    makeTrack("x.mp3"),
    makeTrack("not-in-artifact.mp3"),
  ]);
  const view = makeFakeView();
  const applyPositions = vi.fn(
    (_positions: Map<string, { x: number; y: number; z?: number }>, dims: 2 | 3) => {
      model.setEvalDims(dims);
    },
  );
  const resetCamera = vi.fn();
  const manifestDefer = deferred<unknown>();
  const artifactDefers = new Map<string, ReturnType<typeof deferred<unknown>>>();
  const fetchManifest = vi.fn(() => manifestDefer.promise);
  const fetchArtifact = vi.fn((id: string) => {
    const d = deferred<unknown>();
    artifactDefers.set(id, d);
    return d.promise;
  });
  const deps: EvaluationDeps = {
    model,
    view,
    applyPositions,
    resetCamera,
    fetchManifest,
    fetchArtifact,
  };
  const coord = new EvaluationCoordinator(deps);
  return {
    model,
    view,
    applyPositions,
    resetCamera,
    fetchManifest,
    fetchArtifact,
    coord,
    manifestDefer,
    artifactDefers,
  };
}

async function booted(h: Harness = makeHarness()): Promise<Harness> {
  const p = h.coord.start();
  h.manifestDefer.resolve({ ok: true, manifest: MANIFEST });
  await flush();
  const base = h.artifactDefers.get("baseline");
  base?.resolve(artifact("baseline", "rev-A"));
  await p;
  await flush();
  h.coord.onLibraryLoaded();
  await flush();
  return h;
}

/** The state object of the last toolbar render. */
function lastState(h: Harness): Parameters<EvalToolbarView["render"]>[0] {
  const render = h.view.render as unknown as { mock: { calls: [Parameters<EvalToolbarView["render"]>[0]][] } };
  const calls = render.mock.calls;
  return calls[calls.length - 1][0];
}

async function bootedPaired(initialRecipeId?: string): Promise<Harness> {
  const h = makeHarness();
  const p = h.coord.start(initialRecipeId);
  h.manifestDefer.resolve({ ok: true, manifest: PAIRED_MANIFEST });
  await flush();
  const first = initialRecipeId === "a_3d" ? "a_3d" : "a";
  h.artifactDefers
    .get(first)!
    .resolve(artifact(first, `rev-${first}`, first === "a_3d" ? 3 : 2, "a"));
  await p;
  await flush();
  h.coord.onLibraryLoaded();
  await flush();
  return h;
}

beforeEach(() => localStorage.clear());
afterEach(() => {
  setEvalReadOnly(false);
  localStorage.clear();
});

describe("EvaluationCoordinator", () => {
  it("enters read-only eval mode and applies the initial map", async () => {
    const h = await booted();
    expect(h.model.evalMode).toBe(true);
    expect(isEvalReadOnly()).toBe(true);
    expect(h.resetCamera).toHaveBeenCalled();
    expect(h.applyPositions).toHaveBeenCalledTimes(1);
    const positions = h.applyPositions.mock.calls[0][0] as Map<string, { x: number; y: number }>;
    expect(positions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2 });
  });

  it("filters non-artifact library tracks out of the view universe", async () => {
    const h = await booted();
    // Library has 4 tracks; the artifact only knows a.mp3, b.mp3, x.mp3.
    expect(h.model.tracks.map((t) => t.path).sort()).toEqual([
      "a.mp3",
      "b.mp3",
      "x.mp3",
    ]);
  });

  it("reports unavailable artifact tracks", async () => {
    const h = await booted();
    const lastRender = lastState(h);
    expect(lastRender.total).toBe(3);
    // Every artifact path is in the library in this harness.
    expect(lastRender.unavailable).toBe(0);
  });

  it("draws artifact tracks from external roots without enabling showExternal", async () => {
    const h = makeHarness();
    h.model.setAllTracks([
      { ...makeTrack("a.mp3"), folder: "lib" },
      { ...makeTrack("b.mp3"), folder: "ext" },
      { ...makeTrack("x.mp3"), folder: "ext/sub" },
      { ...makeTrack("not-in-artifact.mp3"), folder: "lib" },
    ]);
    h.model.setExternalRoots(["ext"]);
    expect(h.model.showExternal).toBe(false);
    expect(h.model.tracks.map((t) => t.path)).toEqual([
      "a.mp3",
      "not-in-artifact.mp3",
    ]);

    await booted(h);
    // The artifact universe is the authority in eval mode: external artifact
    // tracks are drawn even though showExternal is still false.
    expect(h.model.showExternal).toBe(false);
    expect(h.model.tracks.map((t) => t.path).sort()).toEqual([
      "a.mp3",
      "b.mp3",
      "x.mp3",
    ]);
    // The toolbar's eligible count matches what the canvas draws.
    expect(lastState(h).eligible).toBe(h.model.tracks.length);

    // Leaving eval restores the normal external filter, still without mutating
    // the dismissed-in-eval preference.
    h.model.setEvalUniverse(null);
    expect(h.model.showExternal).toBe(false);
    expect(h.model.tracks.map((t) => t.path)).toEqual([
      "a.mp3",
      "not-in-artifact.mp3",
    ]);
  });

  it("recomputes the universe once the full library streams in (empty → full)", async () => {
    const h = makeHarness();
    // The library stream starts empty; `_loadLibrary` resolves on its first
    // flush, which may contain none of the artifact's (library-ordered) paths.
    h.model.setAllTracks([]);
    const p = h.coord.start();
    h.manifestDefer.resolve({ ok: true, manifest: MANIFEST });
    await flush();
    h.artifactDefers.get("baseline")!.resolve(artifact("baseline", "rev-A"));
    await p;
    await flush();
    // First "library loaded" signal arrives while the stream is still partial.
    h.coord.onLibraryLoaded();
    await flush();
    let last = lastState(h);
    expect(last.eligible).toBe(0);
    expect(last.total).toBe(3);
    expect(last.unavailable).toBe(3);

    // The stream finishes and appends the artifact's tracks; the universe must
    // recompute against the full set rather than stay pinned at the first flush.
    h.model.setAllTracks([
      makeTrack("a.mp3"),
      makeTrack("b.mp3"),
      makeTrack("x.mp3"),
      makeTrack("not-in-artifact.mp3"),
    ]);
    h.coord.onLibraryLoaded();
    await flush();
    last = lastState(h);
    expect(last.eligible).toBe(3);
    expect(last.total).toBe(3);
    expect(last.unavailable).toBe(0);
    // The view-universe now admits exactly the artifact's tracks.
    expect(h.model.tracks.map((t) => t.path).sort()).toEqual(["a.mp3", "b.mp3", "x.mp3"]);
  });

  it("drops a stale artifact response when the selection raced ahead", async () => {
    const h = await booted();
    expect(h.applyPositions).toHaveBeenCalledTimes(1);
    const select = h.view.onSelect!;

    // Switch to pca2 (fetch in flight), then immediately back to baseline.
    select(1);
    await flush();
    const pcaDefer = h.artifactDefers.get("pca2")!;
    select(0);
    await flush();

    // Re-selecting baseline is a cache hit → applied.
    expect(h.applyPositions).toHaveBeenCalledTimes(2);

    // The late pca2 response must NOT apply (stale).
    pcaDefer.resolve(artifact("pca2", "rev-B"));
    await flush();
    expect(h.applyPositions).toHaveBeenCalledTimes(2);

    // The selected map stays baseline.
    const last = lastState(h);
    expect(last.currentIndex).toBe(0);
  });

  it("applies the newly selected cached map positions on switch", async () => {
    const h = await booted();
    // Pre-fetch pca2 by selecting it once.
    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("pca2")!.resolve(artifact("pca2", "rev-B"));
    await flush();
    const before = h.applyPositions.mock.calls.length;
    // Switch away and back to baseline — cache hit, instant apply.
    h.view.onSelect!(1);
    await flush();
    expect(h.applyPositions.mock.calls.length).toBe(before + 1);
  });

  it("shows a visible error for a corrupt artifact and keeps the toolbar usable", async () => {
    const h = await booted();
    const before = (h.view.render as ReturnType<typeof vi.fn>).mock.calls.length;
    // Select pca2 but resolve it with a corrupt body.
    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("pca2")!.resolve({ ok: true, id: "pca2", positions: [] });
    await flush();
    const last = lastState(h);
    expect(last.error).toMatch(/invalid/i);
    expect(last.busy).toBe(false);
  });

  it("persists ratings keyed by fingerprint + revision and reveals names", async () => {
    const h = await booted();
    // Rate the baseline map.
    h.view.onRate!("neighbours", 4);
    await flush();
    expect(lastState(h).ratings!.neighbours).toBe(4);

    // Reveal: names + timings surface; the record gains the label.
    h.view.onToggleReveal!(true);
    await flush();
    const last = lastState(h);
    expect(last.revealed).toBe(true);
    expect(last.maps[0].label).toBe("PCA50 → openTSNE");
  });

  it("shows an unsupported-schema error instead of the no-generation error", async () => {
    const h = makeHarness();
    const p = h.coord.start();
    h.manifestDefer.resolve({
      ok: true,
      manifest: { ...MANIFEST, schema_version: 99 },
    });
    await p;
    await flush();
    expect(lastState(h).error).toMatch(/unsupported .*schema/i);
  });

  it("evicts the least-recently-used artifact past the cache bound", async () => {
    const h = makeHarness();
    const recipes: Record<string, unknown> = {};
    for (let i = 0; i < 10; i++)
      recipes[`r${i}`] = { status: "complete", label: `R${i}` };
    const manifest = {
      schema_version: 1,
      status: "complete",
      dataset_fingerprint: "fp-1",
      recipes,
    };
    const p = h.coord.start();
    h.manifestDefer.resolve({ ok: true, manifest });
    await flush();
    h.artifactDefers.get("r0")!.resolve(artifact("r0", "rev-0"));
    await p;
    await flush();

    // Visit r1..r9; the cache holds 8 maps, so the oldest are evicted along the way.
    for (let i = 1; i <= 9; i++) {
      h.view.onSelect!(i);
      await flush();
      h.artifactDefers.get(`r${i}`)!.resolve(artifact(`r${i}`, `rev-${i}`));
      await flush();
    }
    const fetches = (id: string) =>
      h.fetchArtifact.mock.calls.filter((c) => c[0] === id).length;

    // r0 was evicted first → re-selecting it refetches.
    h.view.onSelect!(0);
    await flush();
    h.artifactDefers.get("r0")!.resolve(artifact("r0", "rev-0"));
    await flush();
    expect(fetches("r0")).toBe(2);

    // r1 was evicted when r0 came back → re-selecting it refetches.
    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("r1")!.resolve(artifact("r1", "rev-1"));
    await flush();
    expect(fetches("r1")).toBe(2);

    // A recently-used map is still cached → no refetch.
    h.view.onSelect!(8);
    await flush();
    expect(fetches("r8")).toBe(1);
  });

  it("navigates paired groups as one blind letter per group", async () => {
    const h = await bootedPaired();
    const last = lastState(h);
    expect(last.count).toBe(3);
    expect(last.maps.map((m) => m.letter)).toEqual(["A", "B", "C"]);
    // The 3-D sibling shares group A's letter and is not a navigation step.
    expect(h.fetchArtifact).toHaveBeenCalledTimes(1);
    expect(h.fetchArtifact).toHaveBeenCalledWith("a");

    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();
    expect(lastState(h).currentIndex).toBe(1);
    expect(lastState(h).maps[1].letter).toBe("B");
  });

  it("toggles the paired group's dimensionality with a sibling fetch only", async () => {
    const h = await bootedPaired();
    const open = h.coord.toggleDims();
    await flush();
    expect(h.fetchArtifact.mock.calls.filter((c) => c[0] === "a_3d")).toHaveLength(1);
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 3, "a"));
    await open;
    await flush();
    expect(h.model.evalDims).toBe(3);
    const applied = h.applyPositions.mock.calls.at(-1)!;
    expect(applied[1]).toBe(3);
    expect((applied[0] as Map<string, { z?: number }>).get("a.mp3")!.z).toBeCloseTo(0.2);
    expect(lastState(h).dims).toBe(3);
    expect(lastState(h).paired).toBe(true);
    expect(lastState(h).maps[0].dims).toBe(3);

    await h.coord.toggleDims();
    await flush();
    // Back to the cached 2-D artifact: no second sibling fetch.
    expect(h.fetchArtifact.mock.calls.filter((c) => c[0] === "a_3d")).toHaveLength(1);
    expect(h.model.evalDims).toBe(2);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);
  });

  it("toggles back when 3 is pressed twice before the sibling fetch lands", async () => {
    const h = await bootedPaired();
    const to3 = h.coord.toggleDims();
    await flush();
    expect(h.fetchArtifact.mock.calls.filter((c) => c[0] === "a_3d")).toHaveLength(1);

    // The second press reads the *requested* 3 and turns it back to 2, which is
    // an immediate cache hit; the slow 3-D response must be dropped.
    await h.coord.toggleDims();
    await flush();
    expect(h.model.evalDims).toBe(2);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);
    const applied = h.applyPositions.mock.calls.length;

    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 3, "a"));
    await to3;
    await flush();
    expect(h.applyPositions.mock.calls.length).toBe(applied);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);
    expect(h.model.evalDims).toBe(2);
  });

  it("keeps ratings per artifact revision across a dims toggle", async () => {
    const h = await bootedPaired();
    h.view.onRate!("neighbours", 4);
    await flush();
    expect(lastState(h).ratings!.neighbours).toBe(4);

    const open = h.coord.toggleDims();
    await flush();
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 3, "a"));
    await open;
    await flush();
    expect(lastState(h).ratings).toBeNull();
    h.view.onRate!("overall", 5);
    await flush();
    expect(lastState(h).ratings!.overall).toBe(5);

    await h.coord.toggleDims();
    await flush();
    expect(lastState(h).ratings!.neighbours).toBe(4);
    expect(lastState(h).ratings!.overall).toBeFalsy();
  });

  it("falls back to 2-D when the next group has no 3-D sibling", async () => {
    const h = await bootedPaired();
    const open = h.coord.toggleDims();
    await flush();
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 3, "a"));
    await open;
    await flush();
    expect(h.model.evalDims).toBe(3);

    h.view.onSelect!(2);
    await flush();
    h.artifactDefers.get("unpaired")!.resolve(artifact("unpaired", "rev-u", 2, "unpaired"));
    await flush();
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);
    expect(h.model.evalDims).toBe(2);
    const last = lastState(h);
    expect(last.paired).toBe(false);
    expect(last.dims).toBe(2);
  });

  it("keeps the applied map rateable when a sibling artifact is invalid", async () => {
    const h = await bootedPaired();
    h.view.onRate!("neighbours", 4);
    await flush();

    const open = h.coord.toggleDims();
    await flush();
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 2, "a"));
    await open;
    await flush();

    // The 3-D request failed, so the 2-D map is still on screen and still the
    // rating target: same group, same dims, same record.
    const last = lastState(h);
    expect(last.error).toMatch(/invalid/i);
    expect(last.currentIndex).toBe(0);
    expect(last.dims).toBe(2);
    expect(last.canRate).toBe(true);
    expect(last.ratings!.neighbours).toBe(4);
    expect(h.model.evalDims).toBe(2);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);

    h.view.onRate!("overall", 5);
    await flush();
    expect(loadRating("fp-paired", "rev-a")!.overall).toBe(5);
    expect(loadRating("fp-paired", "rev-a3")).toBeNull();
  });

  it("clears the error once a later map applies", async () => {
    const h = await bootedPaired();
    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("b")!.reject(new Error("boom"));
    await flush();
    expect(lastState(h).error).toMatch(/failed to load map b/i);
    expect(lastState(h).currentIndex).toBe(0);

    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();
    const last = lastState(h);
    expect(last.error).toBeNull();
    expect(last.currentIndex).toBe(1);
    expect(last.canRate).toBe(true);
  });

  it("leaves the first map unrated when its artifact fails to load", async () => {
    const h = makeHarness();
    const p = h.coord.start();
    h.manifestDefer.resolve({ ok: true, manifest: MANIFEST });
    await flush();
    h.artifactDefers.get("baseline")!.reject(new Error("boom"));
    await p;
    await flush();
    const last = lastState(h);
    expect(last.error).toMatch(/failed to load map a/i);
    expect(last.currentIndex).toBe(0);
    expect(last.canRate).toBe(false);
    expect(last.ratings).toBeNull();
    expect(h.applyPositions).not.toHaveBeenCalled();
  });

  it("keeps a valid pending note on its original artifact after a failed sibling fetch", async () => {
    const h = await bootedPaired();
    h.view.onNote!("kept on the 2-D map");
    await flush();

    const open = h.coord.toggleDims();
    await flush();
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 2, "a"));
    await open;
    await flush();
    expect(lastState(h).error).toMatch(/invalid/i);
    expect(lastState(h).canRate).toBe(true);

    await new Promise((r) => setTimeout(r, 450));
    expect(loadRating("fp-paired", "rev-a")?.note).toBe("kept on the 2-D map");
    expect(loadRating("fp-paired", "rev-a3")).toBeNull();
  });

  it("drops a late sibling fetch when another group wins the race", async () => {
    const h = await bootedPaired();
    void h.coord.toggleDims();
    await flush();
    h.view.onSelect!(1);
    await flush();
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();
    const appliedAfterB = h.applyPositions.mock.calls.length;
    h.artifactDefers.get("a_3d")!.resolve(artifact("a_3d", "rev-a3", 3, "a"));
    await flush();
    expect(h.applyPositions.mock.calls.length).toBe(appliedAfterB);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);
    expect(lastState(h).currentIndex).toBe(1);
  });

  it("starts on the requested 3-D sibling's group and dimensionality", async () => {
    const h = await bootedPaired("a_3d");
    expect(h.applyPositions.mock.calls[0][1]).toBe(3);
    expect(h.model.evalDims).toBe(3);
    expect(lastState(h).currentIndex).toBe(0);
    expect(lastState(h).dims).toBe(3);
  });

  it("steps from the applied map and keeps a rapid second press", async () => {
    const h = await bootedPaired();
    h.view.onNext!();
    h.view.onNext!(); // both presses land while nothing has been applied yet
    await flush();

    // The second target applies; the first target's late response is dropped.
    h.artifactDefers.get("unpaired")!.resolve(artifact("unpaired", "rev-u", 2, "unpaired"));
    await flush();
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();
    expect(lastState(h).currentIndex).toBe(2);
    expect(h.applyPositions.mock.calls.at(-1)![1]).toBe(2);

    // Prev shows the applied map until the target lands.
    h.view.onPrev!();
    await flush();
    expect(lastState(h).currentIndex).toBe(2);
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();
    expect(lastState(h).currentIndex).toBe(1);
  });

  it("flushes a pending note when the map is switched before the debounce", async () => {
    const h = await bootedPaired();
    h.view.onNote!("note on a");
    h.view.onSelect!(1); // no debounce wait
    await flush();
    h.artifactDefers.get("b")!.resolve(artifact("b", "rev-b", 2, "b"));
    await flush();

    h.view.onNote!("note on b");
    h.view.onSelect!(0);
    await flush();

    expect(loadRating("fp-paired", "rev-a")!.note).toBe("note on a");
    expect(loadRating("fp-paired", "rev-b")!.note).toBe("note on b");

    // Nothing is left to land on the wrong map when the timer would have fired.
    await new Promise((r) => setTimeout(r, 450));
    expect(loadRating("fp-paired", "rev-a")!.note).toBe("note on a");
    expect(loadRating("fp-paired", "rev-b")!.note).toBe("note on b");
  });

  it("flushes the pending note into the export payload", async () => {
    const h = await bootedPaired();
    const blobs: Blob[] = [];
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = ((b: Blob) => {
      blobs.push(b);
      return "blob:eval";
    }) as typeof URL.createObjectURL;
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL;
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    try {
      h.view.onNote!("pending at export");
      h.view.onExport!();
      expect(blobs).toHaveLength(1);
      const payload = JSON.parse(await blobs[0].text());
      expect(payload.maps[0].note).toBe("pending at export");
    } finally {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
      click.mockRestore();
    }
  });
});
