import { describe, expect, it } from "vitest";
import {
  HOTKEY_MAP,
  MOUSE_HINTS,
  SHORTCUT_ZONES,
  renderShortcutList,
  shortcutCoverageProblems,
} from "./hotkeys";

describe("F1 shortcut zone table", () => {
  it("lists every binding and mouse hint in exactly one zone", () => {
    expect(shortcutCoverageProblems()).toEqual([]);
  });

  it("zone actions all exist in HOTKEY_MAP", () => {
    const actions = new Set(HOTKEY_MAP.map((b) => b.action));
    for (const zone of SHORTCUT_ZONES) {
      for (const g of zone.groups) {
        for (const item of g.items) {
          if (item.action) expect(actions.has(item.action)).toBe(true);
          if (item.mouse) expect(MOUSE_HINTS.some((h) => h.id === item.mouse)).toBe(true);
        }
      }
    }
  });
});

const IS_MAC_LABEL = {
  alt: /Mac|iPhone|iPad/.test(navigator.platform) ? "⌥" : "Alt",
};

describe("renderShortcutList", () => {
  function mount(): HTMLElement {
    const container = document.createElement("div");
    renderShortcutList(container);
    return container;
  }

  it("renders every zone in order", () => {
    const root = mount();
    const titles = [...root.querySelectorAll(".shortcut-zone-title")].map((e) => e.textContent);
    expect(titles).toEqual(SHORTCUT_ZONES.map((z) => z.title));
  });

  it("clusters same-key siblings with one cap slot per distinct modifier", () => {
    const root = mount();
    const clusters = [...root.querySelectorAll(".shortcut-cluster")];
    const sKeys = clusters.find((c) => c.querySelector(".cap.key")?.textContent === "S");
    expect(sKeys).toBeDefined();
    const rows = [...sKeys!.querySelectorAll(".shortcut-row")];
    expect(rows.map((r) => r.querySelector(".shortcut-title")?.textContent)).toEqual([
      "Isolate",
      "Exclude",
      "Show all tracks",
    ]);
    // Every row reserves the same slots (alt, shift) so the caps line up.
    for (const r of rows) {
      expect(r.querySelectorAll(".cap.mod, .cap-slot.mod").length).toBe(2);
    }
    expect(rows[1].querySelectorAll(".cap.mod").length).toBe(1);
    expect(rows[1].querySelector(".cap.mod")?.textContent).toBe("⇧");
    expect(rows[2].querySelector(".cap.mod")?.textContent).toBe(IS_MAC_LABEL.alt);
  });

  it("clusters pointer-gesture siblings (Shift/Ctrl/Alt + drag) by gesture", () => {
    const root = mount();
    const drag = [...root.querySelectorAll(".shortcut-cluster")].find((c) =>
      [...c.querySelectorAll(".shortcut-title")].some((t) => t.textContent === "Lasso select"),
    );
    expect(drag).toBeDefined();
    expect(drag!.querySelectorAll(".shortcut-row").length).toBeGreaterThanOrEqual(4);
    expect(drag!.querySelector(".cap.gesture")?.textContent).toBe("drag");
  });

  it("shows a multi-key action's chords in one row", () => {
    const root = mount();
    const row = [...root.querySelectorAll(".shortcut-row")].find(
      (r) => r.querySelector(".shortcut-title")?.textContent === "Focus search",
    );
    expect(row).toBeDefined();
    expect(row!.querySelectorAll(".chord").length).toBe(2);
  });

  it("gives every row a title and a small description", () => {
    const root = mount();
    const rows = [...root.querySelectorAll(".shortcut-row")];
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) {
      expect(r.querySelector(".shortcut-title")?.textContent?.trim()).toBeTruthy();
      expect(r.querySelector(".shortcut-desc")?.textContent?.trim()).toBeTruthy();
    }
  });

  it("never explains a modifier in prose: descriptions carry no key chords", () => {
    const texts = [...HOTKEY_MAP.map((b) => b.description), ...MOUSE_HINTS.map((h) => h.description)];
    for (const t of texts) expect(t).not.toMatch(/\b(Shift|Alt|Ctrl|Cmd)\b|⌘|⇧|⌥|\([^)]*\)/);
  });

  it("renders a cap for every binding key", () => {
    const root = mount();
    const keys = new Set([...root.querySelectorAll(".cap.key")].map((k) => k.textContent));
    for (const b of HOTKEY_MAP) {
      if (b.key === "Alt") continue; // shown as the ⌥ modifier cap
      expect(keys.has(b.key === "Escape" ? "Esc" : b.key.length === 1 ? b.key.toUpperCase() : b.key)).toBe(true);
    }
  });
});
