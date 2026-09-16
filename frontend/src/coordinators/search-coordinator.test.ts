import { describe, expect, it, vi } from "vitest";
import { Model, type Track } from "../model";
import {
  SearchCoordinator,
  SEARCH_LIMIT,
  type SearchResponse,
} from "./search-coordinator";

// ─── Fakes ───────────────────────────────────────────────────

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

function makeHarness(fetchImpl: (q: string, limit: number) => Promise<SearchResponse>) {
  const model = new Model();
  model.setAllTracks([
    makeTrack({ path: "a.mp3" }),
    makeTrack({ path: "b.mp3" }),
  ]);
  const view = {
    setBusy: vi.fn(),
    clear: vi.fn(),
    showError: vi.fn(),
    renderResults: vi.fn(),
  };
  const fitToPaths = vi.fn();
  const fetchResults = vi.fn(fetchImpl);
  const coord = new SearchCoordinator({ model, view, fetchResults, fitToPaths });
  return { model, view, fetchResults, fitToPaths, coord };
}

const HITS: SearchResponse = {
  ok: true,
  results: [
    { path: "b.mp3", score: 0.9 },
    { path: "a.mp3", score: 0.5 },
  ],
  searched: 2,
  total_tracks: 2,
};

// ─── Tests ───────────────────────────────────────────────────

describe("SearchCoordinator", () => {
  it("clears the results and skips the fetch for a blank query", async () => {
    const { coord, view, fetchResults } = makeHarness(async () => HITS);
    await coord.search("   ");
    expect(fetchResults).not.toHaveBeenCalled();
    expect(view.clear).toHaveBeenCalledTimes(1);
  });

  it("trims the query and passes it with the result limit", async () => {
    const { coord, fetchResults } = makeHarness(async () => HITS);
    await coord.search("  dark techno  ");
    expect(fetchResults).toHaveBeenCalledWith("dark techno", SEARCH_LIMIT);
  });

  it("makes the ranked hits the selection and renders them", async () => {
    const { coord, model, view } = makeHarness(async () => HITS);
    await coord.search("x");
    expect([...model.selected]).toEqual(["b.mp3", "a.mp3"]);
    expect(view.renderResults).toHaveBeenCalledWith(HITS.results, {
      searched: 2,
      totalTracks: 2,
    });
  });

  it("toggles the busy state around the fetch", async () => {
    const { coord, view } = makeHarness(async () => HITS);
    await coord.search("x");
    expect(view.setBusy.mock.calls).toEqual([[true], [false]]);
  });

  it("drops hits the model does not know", async () => {
    const response: SearchResponse = {
      ok: true,
      results: [
        { path: "ghost.mp3", score: 0.8 },
        { path: "a.mp3", score: 0.4 },
      ],
      searched: 3,
      total_tracks: 3,
    };
    const { coord, model, view } = makeHarness(async () => response);
    await coord.search("x");
    expect([...model.selected]).toEqual(["a.mp3"]);
    expect(view.renderResults).toHaveBeenCalledWith(
      [{ path: "a.mp3", score: 0.4 }],
      { searched: 3, totalTracks: 3 },
    );
  });

  it("ignores a response superseded by a newer query", async () => {
    let resolveFirst!: (body: SearchResponse) => void;
    const first = new Promise<SearchResponse>((res) => {
      resolveFirst = res;
    });
    const second: SearchResponse = {
      ok: true,
      results: [{ path: "a.mp3", score: 0.1 }],
      searched: 2,
      total_tracks: 2,
    };
    const { coord, model, view, fetchResults } = makeHarness(async () => HITS);
    fetchResults
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(Promise.resolve(second));

    const stale = coord.search("old");
    const fresh = coord.search("new");
    resolveFirst(HITS);
    await Promise.all([stale, fresh]);

    expect([...model.selected]).toEqual(["a.mp3"]);
    expect(view.renderResults).toHaveBeenCalledTimes(1);
  });

  it("surfaces a fetch failure without touching the selection", async () => {
    const { coord, model, view } = makeHarness(async () => {
      throw new Error("CLAP model is still loading");
    });
    model.selected.add("a.mp3");
    await coord.search("x");
    expect(view.setBusy.mock.calls).toEqual([[true], [false]]);
    expect(view.showError).toHaveBeenCalledWith("CLAP model is still loading");
    expect(view.renderResults).not.toHaveBeenCalled();
    expect([...model.selected]).toEqual(["a.mp3"]);
  });

  it("focuses one hit by selecting it alone and fitting the view", () => {
    const { coord, model, fitToPaths } = makeHarness(async () => HITS);
    model.selected.add("a.mp3");
    coord.focusResult("b.mp3");
    expect([...model.selected]).toEqual(["b.mp3"]);
    expect(fitToPaths).toHaveBeenCalledWith(["b.mp3"]);
  });

  it("clears the panel", () => {
    const { coord, view } = makeHarness(async () => HITS);
    coord.clear();
    expect(view.clear).toHaveBeenCalledTimes(1);
  });
});
