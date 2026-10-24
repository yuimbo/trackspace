import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import {
  EvaluationCoordinator,
  type EvaluationDeps,
} from "./evaluation-coordinator";
import { isEvalReadOnly, setEvalReadOnly } from "../lib/eval-readonly";
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

function artifact(id: string, rev: string): unknown {
  return {
    ok: true,
    id,
    label: "Recipe",
    revision: rev,
    dataset_fingerprint: "fp-1",
    tracks: 3,
    aligned_to: null,
    timings: { pca: 0.2, tsne: 1.1 },
    positions: [
      { path: "a.mp3", x: 0.1, y: 0.2 },
      { path: "b.mp3", x: 0.8, y: 0.9 },
      { path: "x.mp3", x: 0.5, y: 0.5 },
    ],
  };
}

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
  const applyPositions = vi.fn();
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

async function booted(): Promise<Harness> {
  const h = makeHarness();
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
});
