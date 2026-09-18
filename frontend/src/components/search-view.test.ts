import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model } from "../model";
import { SearchView, SEARCH_DEBOUNCE_MS } from "./search-view";

function makeView(): { view: SearchView; input: HTMLInputElement } {
  const section = document.createElement("section");
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

  it("reports active on focus and inactive on blur", () => {
    const { view, input } = makeView();
    const onActiveChange = vi.fn();
    view.onActiveChange = onActiveChange;
    input.focus();
    input.dispatchEvent(new FocusEvent("focus"));
    expect(onActiveChange).toHaveBeenLastCalledWith(true);
    input.blur();
    input.dispatchEvent(new FocusEvent("blur"));
    expect(onActiveChange).toHaveBeenLastCalledWith(false);
  });

  it("leaves the box on Esc and hides the field", () => {
    const { view, input } = makeView();
    const onActiveChange = vi.fn();
    view.onActiveChange = onActiveChange;
    input.focus();
    input.value = "techno";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(input.value).toBe("");
    expect(onActiveChange).toHaveBeenLastCalledWith(false);
  });
});
