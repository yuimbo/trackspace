import { describe, expect, it, vi } from "vitest";
import { Model, type Track, type ViewModifier } from "../model";
import {
  SearchCoordinator,
  SEARCH_LIMIT,
  type SearchResponse,
} from "./search-coordinator";
import type { ViewModifierSink } from "./view-modifier-coordinator";

// ─── Fakes ───────────────────────────────────────────────────

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
  const viewMods = makeViewMods();
  const fitToPaths = vi.fn();
  const setField = vi.fn();
  const setResults = vi.fn();
  const fetchResults = vi.fn(fetchImpl);
  const coord = new SearchCoordinator({
    model,
    view,
    viewMods: viewMods.sink,
    fetchResults,
    fitToPaths,
    setField,
    setResults,
  });
  return { model, view, viewMods, fetchResults, fitToPaths, setField, setResults, coord };
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

  it("clears the canvas field for a blank query", async () => {
    const { coord, setField } = makeHarness(async () => HITS);
    await coord.search("   ");
    expect(setField).toHaveBeenCalledWith(null);
  });

  it("publishes the landmark field from the response", async () => {
    const field = [{ path: "a.mp3", score: 0.7 }];
    const { coord, setField } = makeHarness(async () => ({ ...HITS, field }));
    await coord.search("x");
    expect(setField).toHaveBeenCalledWith(field);
  });

  it("clears the canvas field when the fetch fails", async () => {
    const { coord, setField } = makeHarness(async () => {
      throw new Error("boom");
    });
    await coord.search("x");
    expect(setField).toHaveBeenCalledWith(null);
  });

  it("publishes the matched paths so the canvas can fade the rest", async () => {
    const { coord, setResults } = makeHarness(async () => HITS);
    await coord.search("x");
    expect(setResults).toHaveBeenCalledWith(["b.mp3", "a.mp3"]);
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

  it("clears the panel, the canvas field, and the fade", () => {
    const { coord, view, setField, setResults } = makeHarness(async () => HITS);
    coord.clear();
    expect(view.clear).toHaveBeenCalledTimes(1);
    expect(setField).toHaveBeenCalledWith(null);
    expect(setResults).toHaveBeenCalledWith(null);
  });

  it("pushes one search chip once the search is active", async () => {
    const { coord, viewMods } = makeHarness(async () => HITS);
    await coord.search("dark techno");
    expect(viewMods.stack).toEqual([
      { id: "vm-1", kind: "search", label: "dark techno" },
    ]);
  });

  it("refreshes the same chip on a later query instead of stacking", async () => {
    const { coord, viewMods } = makeHarness(async () => HITS);
    await coord.search("dark techno");
    await coord.search("dark ambient");
    expect(viewMods.stack).toHaveLength(1);
    expect(viewMods.stack[0].label).toBe("dark ambient");
  });

  it("drops the chip when the search is cleared", async () => {
    const { coord, viewMods } = makeHarness(async () => HITS);
    await coord.search("x");
    coord.clear();
    expect(viewMods.stack).toEqual([]);
  });

  it("drops the chip for a blank query", async () => {
    const { coord, viewMods } = makeHarness(async () => HITS);
    await coord.search("x");
    await coord.search("   ");
    expect(viewMods.stack).toEqual([]);
  });

  it("drops the chip when the fetch fails", async () => {
    const { coord, viewMods } = makeHarness(async () => {
      throw new Error("boom");
    });
    await coord.search("x");
    expect(viewMods.stack).toEqual([]);
  });

  it("clears the canvas state when the chip is dismissed", async () => {
    const { coord, viewMods, setField, setResults, view } = makeHarness(
      async () => HITS,
    );
    await coord.search("x");
    viewMods.sink.dismiss("vm-1");
    expect(setField).toHaveBeenCalledWith(null);
    expect(setResults).toHaveBeenCalledWith(null);
    expect(view.clear).toHaveBeenCalled();
  });
});
