import { describe, expect, it, vi } from "vitest";
import { Model, type Track, type ViewModifier } from "../model";
import { type InspectState } from "../lib/tooltip";
import {
  ANTI_K,
  DESCRIBE_K,
  InspectCoordinator,
  NEAR_K,
  PROXIMITY_ALL,
  type DescribeResponse,
  type InspectViewLike,
  type SimilarResponse,
} from "./inspect-coordinator";
import type { ViewModifierSink } from "./view-modifier-coordinator";

/** In-memory stand-in for the active-view-modifier stack. */
function makeViewMods() {
  let seq = 0;
  const stack: ViewModifier[] = [];
  const clearers = new Map<string, () => void>();
  const sink: ViewModifierSink = {
    push: (mod, clear) => {
      const id = `vm-${++seq}`;
      stack.push({ ...mod, id });
      clearers.set(id, clear);
      return id;
    },
    update: (id, patch) => {
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack[i] = { ...stack[i], ...patch };
    },
    dismiss: (id) => {
      const clear = clearers.get(id);
      if (!clear) return;
      clearers.delete(id);
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack.splice(i, 1);
      clear();
    },
    remove: (id) => {
      if (!id) return;
      clearers.delete(id);
      const i = stack.findIndex((m) => m.id === id);
      if (i >= 0) stack.splice(i, 1);
    },
    find: (kind) => stack.find((m) => m.kind === kind),
  };
  return { sink, stack };
}

function makeTrack(over: Partial<Track> = {}): Track {
  return {
    path: "a.mp3",
    filename: "a.mp3",
    folder: "",
    tags: {},
    artist: "",
    title: "",
    ...over,
  };
}

interface FakeView {
  setInspect: ReturnType<typeof vi.fn>;
  unpinInspect: ReturnType<typeof vi.fn>;
  /** The merged inspect state, as the real view merges every patch. */
  state: Partial<InspectState>;
}

interface Harness {
  model: Model;
  view: FakeView;
  fitToPaths: ReturnType<typeof vi.fn>;
  preview: ReturnType<typeof vi.fn>;
  stopPreview: ReturnType<typeof vi.fn>;
  fetchSimilar: ReturnType<typeof vi.fn>;
  fetchDescribe: ReturnType<typeof vi.fn>;
  viewMods: ReturnType<typeof makeViewMods>;
  coord: InspectCoordinator;
}

function makeHarness(
  response: SimilarResponse = { ok: true, results: [] },
  describe: DescribeResponse = { ok: true, results: [], coordinates: [] },
): Harness {
  const model = new Model();
  // The real `CanvasView.setInspect` merges each patch into the pinned card's
  // state, so the fake does too; the last-call patch alone would be racy.
  const state: Partial<InspectState> = {};
  const view: FakeView = {
    setInspect: vi.fn((patch: Partial<InspectState>) => {
      Object.assign(state, patch);
    }),
    unpinInspect: vi.fn(),
    state,
  };
  const fitToPaths = vi.fn();
  const preview = vi.fn();
  const stopPreview = vi.fn();
  const fetchSimilar = vi.fn(async () => response);
  const fetchDescribe = vi.fn(async () => describe);
  const viewMods = makeViewMods();
  const coord = new InspectCoordinator({
    model,
    view: view as unknown as InspectViewLike,
    viewMods: viewMods.sink,
    fetchSimilar,
    fetchDescribe,
    fitToPaths,
    preview,
    stopPreview,
  });
  return {
    model,
    view,
    fitToPaths,
    preview,
    stopPreview,
    fetchSimilar,
    fetchDescribe,
    viewMods,
    coord,
  };
}

/** The merged inspect state (busy/results/…). */
function lastInspect(view: Harness["view"]): Partial<InspectState> {
  return view.state;
}

describe("InspectCoordinator", () => {
  it("toggles the ranking between nearest and anti-track", async () => {
    const h = makeHarness({
      ok: true,
      results: [{ path: "far.mp3", score: -0.42 }],
    });
    // Pinned defaults to nearest; the toggle inverts it to anti-track.
    await h.coord.action("toggle-rank-mode", "a.mp3");
    expect(h.fetchSimilar).toHaveBeenCalledWith("a.mp3", "far", ANTI_K);
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      mode: "far",
      results: [{ path: "far.mp3", score: -0.42 }],
    });
    // Toggling again goes back to nearest.
    await h.coord.action("toggle-rank-mode", "a.mp3");
    expect(h.fetchSimilar).toHaveBeenLastCalledWith("a.mp3", "near", NEAR_K);
    expect(lastInspect(h.view)).toMatchObject({ mode: "near" });
  });

  it("shows the backend error and clears the results", async () => {
    const h = makeHarness({ ok: false, error: "No analysed tracks yet" });
    await h.coord.action("toggle-rank-mode", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      message: "No analysed tracks yet",
      results: [],
    });
  });

  it("recovers when the fetch itself rejects", async () => {
    const h = makeHarness();
    h.fetchSimilar.mockRejectedValue(new Error("Failed to fetch"));
    await h.coord.action("toggle-rank-mode", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      message: "Failed to fetch",
      results: [],
    });
  });

  it("narrows the distance filter without re-homing the view", async () => {
    const h = makeHarness({
      ok: true,
      ranking: ["b.mp3", "c.mp3", "d.mp3"],
    });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
      makeTrack({ path: "d.mp3" }),
    ]);

    await h.coord.action("proximity", "a.mp3", 50);
    // One fetch of the full ranking, then local filtering.
    expect(h.fetchSimilar).toHaveBeenCalledWith("a.mp3", "near", PROXIMITY_ALL);
    // 50 % of 3 ranked → the closest 2 (b, c); d is hidden, a stays.
    expect(h.model.isTrackHidden(h.model.trackByPath("d.mp3")!)).toBe(true);
    expect(h.model.isTrackHidden(h.model.trackByPath("a.mp3")!)).toBe(false);
    expect(h.model.isTrackHidden(h.model.trackByPath("b.mp3")!)).toBe(false);
    // Adjusting the filter never re-homes the camera.
    expect(h.fitToPaths).not.toHaveBeenCalled();
  });

  it("keeps the furthest tracks in anti-track mode", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3", "d.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
      makeTrack({ path: "d.mp3" }),
    ]);
    await h.coord.action("toggle-rank-mode", "a.mp3"); // nearest → anti-track
    await h.coord.action("proximity", "a.mp3", 40);
    // 40 % of 3 → keep the furthest 2 (c, d); b is hidden.
    expect(h.model.isTrackHidden(h.model.trackByPath("b.mp3")!)).toBe(true);
    expect(h.model.isTrackHidden(h.model.trackByPath("c.mp3")!)).toBe(false);
    expect(h.model.isTrackHidden(h.model.trackByPath("d.mp3")!)).toBe(false);
  });

  it("reveals everything again at 100 %", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3", "d.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
      makeTrack({ path: "d.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 10);
    expect(h.model.isTrackHidden(h.model.trackByPath("d.mp3")!)).toBe(true);

    await h.coord.action("proximity", "a.mp3", 100);
    expect(h.model.isTrackHidden(h.model.trackByPath("d.mp3")!)).toBe(false);
    expect(h.fitToPaths).not.toHaveBeenCalled();
  });

  it("pre-loads the full ranking when the card is pinned", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("open", "a.mp3");
    // The pin pre-loads the full ranking so the slider filters at once.
    const allFetches = h.fetchSimilar.mock.calls.filter(
      (c) => c[2] === PROXIMITY_ALL,
    );
    expect(allFetches).toHaveLength(1);
    await h.coord.action("proximity", "a.mp3", 50);
    // The slider reuses the pre-loaded ranking; no second full fetch.
    expect(
      h.fetchSimilar.mock.calls.filter((c) => c[2] === PROXIMITY_ALL),
    ).toHaveLength(1);
  });

  it("fetches the full ranking once across slider steps", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 60);
    await h.coord.action("proximity", "a.mp3", 30);
    expect(h.fetchSimilar).toHaveBeenCalledTimes(1);
  });

  it("shows a message when no ranking is available", async () => {
    const h = makeHarness({ ok: false, error: "No analysed tracks yet" });
    await h.coord.action("proximity", "a.mp3", 50);
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      message: "No analysed tracks yet",
    });
  });

  it("pushes a proximity chip on the first narrowing", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 50);
    expect(h.viewMods.stack).toHaveLength(1);
    expect(h.viewMods.stack[0]).toMatchObject({
      kind: "proximity",
      value: 50,
    });
  });

  it("refreshes the same proximity chip on each step", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 60);
    await h.coord.action("proximity", "a.mp3", 30);
    expect(h.viewMods.stack).toHaveLength(1);
    expect(h.viewMods.stack[0].value).toBe(30);
  });

  it("drops the proximity chip at 100 %", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 50);
    await h.coord.action("proximity", "a.mp3", 100);
    expect(h.viewMods.stack).toEqual([]);
  });

  it("clears the canvas filter when the proximity chip is dismissed", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 50);
    const keep = vi.spyOn(h.model, "setProximityKeep");
    h.viewMods.sink.dismiss(h.viewMods.stack[0].id);
    expect(h.viewMods.stack).toEqual([]);
    expect(keep).toHaveBeenLastCalledWith(
      new Set(["a.mp3", "b.mp3", "c.mp3"]),
      expect.anything(),
    );
  });

  it("keeps the proximity chip when the inspect card closes", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 50);
    await h.coord.action("close", "a.mp3");
    expect(h.viewMods.stack).toHaveLength(1);
  });

  it("drops the old proximity chip when another track is pinned", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    await h.coord.action("proximity", "a.mp3", 50);
    await h.coord.action("open", "b.mp3");
    expect(h.viewMods.stack).toEqual([]);
  });

  it("coalesces a slider drag into a leading and a trailing step", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3", "d.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
      makeTrack({ path: "d.mp3" }),
    ]);
    const keep = vi.spyOn(h.model, "setProximityKeep");

    // A burst of input events: the first step applies at once, the rest
    // coalesce into a single trailing step with the final value.
    void h.coord.action("proximity", "a.mp3", 100);
    void h.coord.action("proximity", "a.mp3", 50);
    await h.coord.action("proximity", "a.mp3", 10);

    expect(keep).toHaveBeenCalledTimes(2);
    expect(keep).toHaveBeenLastCalledWith(new Set(["a.mp3", "b.mp3"]), expect.anything());
  });

  it("drops a pending slider step when another action supersedes it", async () => {
    const h = makeHarness({ ok: true, ranking: ["b.mp3", "c.mp3"] });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
    ]);
    const keep = vi.spyOn(h.model, "setProximityKeep");

    void h.coord.action("proximity", "a.mp3", 100); // leading, applies now
    void h.coord.action("proximity", "a.mp3", 30); // pending trailing step
    await h.coord.action("close", "a.mp3"); // supersedes the pending step

    expect(h.view.unpinInspect).toHaveBeenCalled();
    // Only the leading step applied; the trailing one was cancelled.
    expect(keep).toHaveBeenCalledTimes(1);
  });

  it("focuses a result: select it alone, fit the view, audition it", async () => {
    const h = makeHarness();
    h.model.setAllTracks([makeTrack({ path: "a.mp3" }), makeTrack({ path: "b.mp3" })]);
    await h.coord.action("focus", "b.mp3");
    expect([...h.model.selected]).toEqual(["b.mp3"]);
    expect(h.fitToPaths).toHaveBeenCalledWith(["b.mp3"]);
    expect(h.preview).toHaveBeenCalledWith("b.mp3");
  });

  it("ignores a focus for a path the model does not know", async () => {
    const h = makeHarness();
    await h.coord.action("focus", "ghost.mp3");
    expect(h.model.selected.size).toBe(0);
    expect(h.fitToPaths).not.toHaveBeenCalled();
  });

  it("closes the card and ends the audition it owned", async () => {
    const h = makeHarness();
    await h.coord.action("close", "a.mp3");
    expect(h.stopPreview).toHaveBeenCalled();
    expect(h.view.unpinInspect).toHaveBeenCalled();
  });

  it("auditions a result row on hover and keeps playing off the row", async () => {
    const h = makeHarness();
    await h.coord.action("hover", "b.mp3");
    expect(h.preview).toHaveBeenCalledWith("b.mp3");
    // Leaving the row does not stop it; leaving the *card* does.
    await h.coord.action("hover", "c.mp3");
    expect(h.stopPreview).not.toHaveBeenCalled();
    await h.coord.action("leave", "c.mp3");
    expect(h.stopPreview).toHaveBeenCalled();
  });

  it("switches to the Describe tab and fetches once across tab switches", async () => {
    const h = makeHarness(
      { ok: true, results: [] },
      {
        ok: true,
        results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
      },
    );
    await h.coord.action("show-describe", "a.mp3");
    expect(h.view.setInspect).toHaveBeenCalledWith({ tab: "describe" });
    expect(h.fetchDescribe).toHaveBeenCalledWith("a.mp3", DESCRIBE_K);
    expect(lastInspect(h.view)).toMatchObject({
      describe: {
        path: "a.mp3",
        busy: false,
        results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
      },
    });

    // Going back to Navigate and returning reuses the loaded descriptions.
    await h.coord.action("show-navigate", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({ tab: "navigate" });
    await h.coord.action("show-describe", "a.mp3");
    expect(h.fetchDescribe).toHaveBeenCalledTimes(1);
  });

  it("surfaces the Describe error and clears the results", async () => {
    const h = makeHarness({ ok: true, results: [] }, { ok: false, error: "No cached CLAP embedding" });
    await h.coord.action("show-describe", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      describe: {
        path: "a.mp3",
        busy: false,
        message: "No cached CLAP embedding",
        results: [],
      },
    });
  });

  it("recovers when the Describe fetch itself rejects", async () => {
    const h = makeHarness();
    h.fetchDescribe.mockRejectedValue(new Error("Failed to fetch"));
    await h.coord.action("show-describe", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      describe: { path: "a.mp3", busy: false, message: "Failed to fetch", results: [] },
    });
  });

  it("reloads descriptions after the card is closed and re-pinned", async () => {
    const h = makeHarness();
    await h.coord.action("show-describe", "a.mp3");
    expect(h.fetchDescribe).toHaveBeenCalledTimes(1);
    await h.coord.action("close", "a.mp3");
    await h.coord.action("show-describe", "a.mp3");
    expect(h.fetchDescribe).toHaveBeenCalledTimes(2);
  });

  it("loads descriptions and coordinates when the card is pinned", async () => {
    const h = makeHarness(
      { ok: true, results: [] },
      {
        ok: true,
        results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
        coordinates: [
          {
            axis: "dark <-> bright",
            pos: "dark",
            neg: "bright",
            pole: "dark",
            z: 2.1,
            percentile: 0.98,
            reliability: 2.58,
          },
        ],
      },
    );
    await h.coord.action("open", "a.mp3");
    expect(h.fetchDescribe).toHaveBeenCalledWith("a.mp3", DESCRIBE_K);
    expect(lastInspect(h.view)).toMatchObject({
      describe: {
        path: "a.mp3",
        busy: false,
        results: [{ text: "a techno track", facet: "Genre", score: 0.31 }],
        coordinates: [{ pole: "dark", percentile: 0.98 }],
      },
    });
  });
});
