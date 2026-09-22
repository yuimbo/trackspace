import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model } from "../model";
import { SearchView, SEARCH_DEBOUNCE_MS } from "./search-view";

function makeView(): { view: SearchView; input: HTMLInputElement } {
  const section = document.createElement("section");
  section.className = "search-modal hidden";
  section.innerHTML =
    '<input id="search-input" type="search" /><div id="search-results"></div>';
  document.body.appendChild(section);
  const view = new SearchView(new Model(), section);
  return {
    view,
    input: section.querySelector("#search-input") as HTMLInputElement,
  };
}

describe("SearchView live search", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = "";
  });

  it("searches while typing, not only on Enter", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    view.onSearch = onSearch;
    input.value = "tec";
    input.dispatchEvent(new Event("input"));
    expect(onSearch).not.toHaveBeenCalled();
    vi.advanceTimersByTime(SEARCH_DEBOUNCE_MS);
    expect(onSearch).toHaveBeenCalledWith("tec");
  });

  it("collapses a keystroke burst into one query", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    view.onSearch = onSearch;
    for (const value of ["t", "te", "tec", "tech"]) {
      input.value = value;
      input.dispatchEvent(new Event("input"));
      vi.advanceTimersByTime(50);
    }
    vi.advanceTimersByTime(SEARCH_DEBOUNCE_MS);
    expect(onSearch).toHaveBeenCalledTimes(1);
    expect(onSearch).toHaveBeenCalledWith("tech");
  });

  it("clears at once when the box is emptied", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    view.onSearch = onSearch;
    input.value = "x";
    input.dispatchEvent(new Event("input"));
    input.value = "";
    input.dispatchEvent(new Event("input"));
    expect(onSearch).toHaveBeenCalledWith("");
  });

  it("searches at once on Enter", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    view.onSearch = onSearch;
    input.value = "techno";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(onSearch).toHaveBeenCalledWith("techno");
  });

  it("reports active when the popup opens and clears it on Esc", () => {
    const { view, input } = makeView();
    const onActiveChange = vi.fn();
    view.onActiveChange = onActiveChange;
    view.open();
    expect(onActiveChange).toHaveBeenLastCalledWith(true);
    input.value = "techno";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(onActiveChange).toHaveBeenLastCalledWith(false);
  });

  it("opens the popup and closes it on Esc, clearing the query", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    view.onSearch = onSearch;
    view.open();
    expect(view.isOpen()).toBe(true);
    input.value = "techno";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(input.value).toBe("");
    expect(onSearch).toHaveBeenLastCalledWith("");
    expect(view.isOpen()).toBe(false);
  });

  it("commits on Enter: searches and asks the popup to close", () => {
    const { view, input } = makeView();
    const onSearch = vi.fn();
    const onCommit = vi.fn();
    view.onSearch = onSearch;
    view.onCommit = onCommit;
    input.value = "techno";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(onSearch).toHaveBeenCalledWith("techno");
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  it("commits when a result row is clicked", () => {
    const { view } = makeView();
    const onFocusResult = vi.fn();
    const onCommit = vi.fn();
    view.onFocusResult = onFocusResult;
    view.onCommit = onCommit;
    view.renderResults([{ path: "a.mp3", score: 0.5 }], {
      searched: 1,
      totalTracks: 1,
    });
    const row = document.querySelector<HTMLElement>(".search-result-row")!;
    row.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(onFocusResult).toHaveBeenCalledWith("a.mp3");
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  it("closes without clearing when the backdrop is clicked", () => {
    const { view } = makeView();
    const onCommit = vi.fn();
    view.onCommit = onCommit;
    view.open();
    view.$el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    expect(onCommit).toHaveBeenCalledTimes(1);
  });
});
