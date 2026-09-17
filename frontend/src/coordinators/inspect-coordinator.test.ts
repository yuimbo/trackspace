import { describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import { type InspectState } from "../lib/tooltip";
import {
  ANTI_K,
  InspectCoordinator,
  NEAR_K,
  SOLO_K,
  type SimilarResponse,
} from "./inspect-coordinator";

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

interface Harness {
  model: Model;
  view: { setInspect: ReturnType<typeof vi.fn>; unpinInspect: ReturnType<typeof vi.fn> };
  fitToPaths: ReturnType<typeof vi.fn>;
  preview: ReturnType<typeof vi.fn>;
  stopPreview: ReturnType<typeof vi.fn>;
  fetchSimilar: ReturnType<typeof vi.fn>;
  coord: InspectCoordinator;
}

function makeHarness(response: SimilarResponse = { ok: true, results: [] }): Harness {
  const model = new Model();
  const view = { setInspect: vi.fn(), unpinInspect: vi.fn() };
  const fitToPaths = vi.fn();
  const preview = vi.fn();
  const stopPreview = vi.fn();
  const fetchSimilar = vi.fn(async () => response);
  const coord = new InspectCoordinator({
    model,
    view,
    fetchSimilar,
    fitToPaths,
    preview,
    stopPreview,
  });
  return { model, view, fitToPaths, preview, stopPreview, fetchSimilar, coord };
}

/** The last patch passed to setInspect (busy/results/…). */
function lastInspect(view: Harness["view"]): Partial<InspectState> {
  const calls = view.setInspect.mock.calls;
  return calls[calls.length - 1][0] as Partial<InspectState>;
}

describe("InspectCoordinator", () => {
  it("ranks the anti-track as the most distant neighbours", async () => {
    const h = makeHarness({
      ok: true,
      results: [{ path: "far.mp3", score: -0.42 }],
    });
    await h.coord.action("anti", "a.mp3");
    expect(h.fetchSimilar).toHaveBeenCalledWith("a.mp3", "far", ANTI_K);
    expect(h.view.setInspect.mock.calls[0][0]).toMatchObject({ busy: true });
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      mode: "far",
      results: [{ path: "far.mp3", score: -0.42 }],
    });
  });

  it("ranks the nearest neighbours", async () => {
    const h = makeHarness({ ok: true, results: [{ path: "near.mp3", score: 0.9 }] });
    await h.coord.action("near", "a.mp3");
    expect(h.fetchSimilar).toHaveBeenCalledWith("a.mp3", "near", NEAR_K);
    expect(lastInspect(h.view)).toMatchObject({ mode: "near" });
  });

  it("shows the backend error and clears the results", async () => {
    const h = makeHarness({ ok: false, error: "No analysed tracks yet" });
    await h.coord.action("anti", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      message: "No analysed tracks yet",
      results: [],
    });
  });

  it("recovers when the fetch itself rejects", async () => {
    const h = makeHarness();
    h.fetchSimilar.mockRejectedValue(new Error("Failed to fetch"));
    await h.coord.action("near", "a.mp3");
    expect(lastInspect(h.view)).toMatchObject({
      busy: false,
      message: "Failed to fetch",
      results: [],
    });
  });

  it("soloes the neighbourhood, keeping the inspected track", async () => {
    const h = makeHarness({
      ok: true,
      results: [
        { path: "b.mp3", score: 0.9 },
        { path: "c.mp3", score: 0.8 },
      ],
    });
    h.model.setAllTracks([
      makeTrack({ path: "a.mp3" }),
      makeTrack({ path: "b.mp3" }),
      makeTrack({ path: "c.mp3" }),
      makeTrack({ path: "d.mp3" }),
    ]);

    await h.coord.action("solo", "a.mp3");
    expect(h.fetchSimilar).toHaveBeenCalledWith("a.mp3", "near", SOLO_K);
    expect(h.model.isTrackHidden(h.model.trackByPath("d.mp3")!)).toBe(true);
    expect(h.model.isTrackHidden(h.model.trackByPath("a.mp3")!)).toBe(false);
    expect(h.fitToPaths).toHaveBeenCalledWith(["a.mp3", "b.mp3", "c.mp3"]);
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
});
