import { afterEach, describe, expect, it, vi } from "vitest";
import type { ViewModifier } from "../model";
import { ViewModifierView } from "./view-modifier-view";

function makeView(): { view: ViewModifierView; el: HTMLElement } {
  const el = document.createElement("div");
  el.className = "view-modifiers hidden";
  document.body.appendChild(el);
  return { view: new ViewModifierView(el), el };
}

function mod(over: Partial<ViewModifier> = {}): ViewModifier {
  return { id: "vm-1", kind: "search", label: "techno", ...over };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("ViewModifierView", () => {
  it("renders one chip per modifier with its label", () => {
    const { view, el } = makeView();
    view.render([mod(), mod({ id: "vm-2", kind: "scope", label: "Camelot" })]);
    const chips = el.querySelectorAll<HTMLElement>(".view-modifier");
    expect(chips).toHaveLength(2);
    expect(chips[0].querySelector(".vm-label")?.textContent).toBe("techno");
    expect(chips[1].querySelector(".vm-label")?.textContent).toBe("Camelot");
    expect(el.classList.contains("hidden")).toBe(false);
  });

  it("hides the stack when there are no modifiers", () => {
    const { view, el } = makeView();
    view.render([mod()]);
    view.render([]);
    expect(el.classList.contains("hidden")).toBe(true);
    expect(el.querySelectorAll(".view-modifier")).toHaveLength(0);
  });

  it("reports a chip's × with the modifier id", () => {
    const { view, el } = makeView();
    const onDismiss = vi.fn();
    view.onDismiss = onDismiss;
    view.render([mod({ id: "vm-7" })]);
    el.querySelector<HTMLElement>(".vm-dismiss")!.click();
    expect(onDismiss).toHaveBeenCalledWith("vm-7");
  });

  it("shows the detail readout when present", () => {
    const { view, el } = makeView();
    view.render([mod({ detail: "42 hidden" })]);
    const detail = el.querySelector<HTMLElement>(".vm-detail")!;
    expect(detail.textContent).toBe("42 hidden");
    expect(detail.hidden).toBe(false);
  });

  it("gives a proximity chip a slider that reports its value", () => {
    const { view, el } = makeView();
    const onProximityInput = vi.fn();
    view.onProximityInput = onProximityInput;
    view.render([mod({ id: "vm-3", kind: "proximity", value: 40 })]);
    const range = el.querySelector<HTMLInputElement>(".vm-range")!;
    expect(range.value).toBe("40");
    range.value = "25";
    range.dispatchEvent(new Event("input", { bubbles: true }));
    expect(onProximityInput).toHaveBeenCalledWith("vm-3", 25);
  });

  it("gives an external chip a refit toggle that reports its state", () => {
    const { view, el } = makeView();
    const onExternalRefit = vi.fn();
    view.onExternalRefit = onExternalRefit;
    view.render([
      mod({ id: "vm-4", kind: "external", checked: true }),
    ]);
    const box = el.querySelector<HTMLInputElement>("[data-modifier-refit]")!;
    expect(box.checked).toBe(true);
    box.checked = false;
    box.dispatchEvent(new Event("change", { bubbles: true }));
    expect(onExternalRefit).toHaveBeenCalledWith("vm-4", false);
  });

  it("hides the refit row while externals are off the map", () => {
    const { view, el } = makeView();
    view.render([
      mod({ id: "vm-4", kind: "external", label: "Library Only" }),
    ]);
    expect(el.querySelector<HTMLElement>(".vm-refit")?.hidden).toBe(true);
    expect(el.querySelector(".vm-label")?.textContent).toBe("Library Only");
  });

  it("gives a permanent external chip a toggle and no ×", () => {
    const { view, el } = makeView();
    const onExternalToggle = vi.fn();
    view.onExternalToggle = onExternalToggle;
    view.render([
      mod({
        id: "vm-4",
        kind: "external",
        label: "Library + External",
        checked: false,
        permanent: true,
      }),
    ]);
    expect(el.querySelector(".vm-dismiss")).toBeNull();
    el.querySelector<HTMLElement>("[data-modifier-toggle]")!.click();
    expect(onExternalToggle).toHaveBeenCalledWith("vm-4");
  });

  it("shows the V and R hotkey hints on the external chip", () => {
    const { view, el } = makeView();
    view.render([
      mod({
        id: "vm-4",
        kind: "external",
        label: "Library + External",
        checked: false,
        permanent: true,
      }),
    ]);
    const hints = [...el.querySelectorAll(".vm-hotkey")].map(
      (k) => k.textContent,
    );
    expect(hints).toEqual(["V", "R"]);
  });

  it("reuses the chip element across a re-render, so a drag keeps its input", () => {
    const { view, el } = makeView();
    view.render([mod({ id: "vm-3", kind: "proximity", value: 40 })]);
    const before = el.querySelector(".view-modifier");
    view.render([mod({ id: "vm-3", kind: "proximity", value: 30 })]);
    const after = el.querySelector(".view-modifier");
    expect(after).toBe(before);
    expect(el.querySelector<HTMLInputElement>(".vm-range")!.value).toBe("30");
  });

  it("drops a chip no longer in the list", () => {
    const { view, el } = makeView();
    view.render([mod({ id: "vm-1" }), mod({ id: "vm-2" })]);
    view.render([mod({ id: "vm-2" })]);
    const chips = el.querySelectorAll<HTMLElement>(".view-modifier");
    expect(chips).toHaveLength(1);
    expect(chips[0].dataset.modifierId).toBe("vm-2");
  });

  it("renders an exclusions chip's rule in place of its label", () => {
    const { view, el } = makeView();
    view.render([
      mod({
        id: "vm-9",
        kind: "exclusions",
        label: "Solo",
        glyph: "◉",
        rule: {
          value: "House/sub",
          icon: "folder",
          colors: ["hsl(0,60%,55%)"],
          title: "House/sub",
        },
        detail: "2 tracks",
      }),
    ]);
    // The rule chip replaces the plain label, behind the action glyph.
    expect(el.querySelector(".vm-label")).toBeNull();
    expect(el.querySelector(".vm-mark")?.textContent).toBe("◉");
    expect(el.querySelector<HTMLElement>(".vm-mark")?.title).toBe("Solo");
    expect(el.querySelector(".chip-value")?.textContent).toBe("House/sub");
    expect(el.querySelector(".chip-icon-folder")).not.toBeNull();
    expect(el.querySelector<HTMLElement>(".chip")?.title).toBe("House/sub");
    expect(el.querySelector(".vm-detail")?.textContent).toBe("2 tracks");
  });

  it("keeps the plain label for a chip without a rule", () => {
    const { view, el } = makeView();
    view.render([mod({ id: "vm-9", kind: "exclusions", label: "Solo" })]);
    expect(el.querySelector(".vm-label")?.textContent).toBe("Solo");
    expect(el.querySelector(".chip")).toBeNull();
  });
});
