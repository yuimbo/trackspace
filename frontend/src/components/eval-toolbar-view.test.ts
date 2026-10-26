import { afterEach, describe, expect, it, vi } from "vitest";
import { EvalToolbarView, type EvalToolbarState } from "./eval-toolbar-view";

function makeView(): { view: EvalToolbarView; el: HTMLElement } {
  const el = document.createElement("div");
  el.className = "eval-toolbar hidden";
  document.body.appendChild(el);
  return { view: new EvalToolbarView(el), el };
}

function state(over: Partial<EvalToolbarState> = {}): EvalToolbarState {
  return {
    busy: false,
    currentIndex: 0,
    count: 3,
    maps: [0, 1, 2].map((index) => ({
      index,
      letter: String.fromCharCode(65 + index),
      dims: 2 as const,
      rated: 0,
      revealed: false,
    })),
    revealed: false,
    currentTiming: "",
    eligible: 3,
    total: 3,
    unavailable: 0,
    ratings: null,
    note: "",
    canRate: true,
    dims: 2,
    paired: false,
    datasetLabel: "dataset fp-1",
    error: null,
    ...over,
  };
}

function selectOf(el: HTMLElement): HTMLSelectElement {
  return el.querySelector<HTMLSelectElement>('[data-role="select"]')!;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("EvalToolbarView", () => {
  it("shows the applied map in the dropdown instead of a pending selection", () => {
    const { view, el } = makeView();
    view.render(state({ currentIndex: 2 }));
    expect(selectOf(el).value).toBe("2");

    // A switch in flight keeps the previous map selected in the state.
    view.render(state({ currentIndex: 1, busy: true }));
    expect(selectOf(el).value).toBe("1");

    // A failed switch leaves the applied map selected.
    view.render(state({ currentIndex: 1, error: "boom" }));
    expect(selectOf(el).value).toBe("1");
  });

  it("mirrors a deeplink-selected index over the previous selection", () => {
    const { view, el } = makeView();
    view.render(state({ currentIndex: 2 }));
    expect(selectOf(el).value).toBe("2");

    // A fresh state (deeplink / reload) must not keep the old option.
    view.render(state({ currentIndex: 0 }));
    expect(selectOf(el).value).toBe("0");
  });

  it("reports the dropdown choice as an index", () => {
    const { view, el } = makeView();
    const onSelect = vi.fn();
    view.onSelect = onSelect;
    view.render(state());
    const select = selectOf(el);
    select.value = "2";
    select.dispatchEvent(new Event("change"));
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it("reports prev and next without moving the rendered selection", () => {
    const { view, el } = makeView();
    const onPrev = vi.fn();
    const onNext = vi.fn();
    view.onPrev = onPrev;
    view.onNext = onNext;
    view.render(state({ currentIndex: 1 }));
    el.querySelector<HTMLElement>('[data-role="prev"]')!.click();
    el.querySelector<HTMLElement>('[data-role="next"]')!.click();
    expect(onPrev).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
    expect(selectOf(el).value).toBe("1");
  });

  it("disables rating controls while no map can be rated", () => {
    const { view, el } = makeView();
    view.render(state({ canRate: false, busy: true }));
    expect(el.querySelector<HTMLTextAreaElement>('[data-role="note"]')!.disabled).toBe(true);
    const stars = el.querySelectorAll<HTMLButtonElement>(".ev-stars button");
    expect(stars).toHaveLength(20);
    expect([...stars].every((b) => b.disabled)).toBe(true);

    view.render(state());
    expect(el.querySelector<HTMLTextAreaElement>('[data-role="note"]')!.disabled).toBe(false);
    expect(
      [...el.querySelectorAll<HTMLButtonElement>(".ev-stars button")].some((b) => b.disabled),
    ).toBe(false);
  });
});
