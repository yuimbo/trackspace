// ─── RightPanelBinder ───────────────────────────────────────
//
// Owns the right panel's tab strip. Search and Properties stay pinned above it;
// the four tabs (`RIGHT_PANEL_TABS`) show and hide the sections tagged with
// `data-rp-panel`, and the choice is persisted on the model. The sections keep
// their ids, so the other coordinators are unaffected by the tab they sit in.

import { type Model, type RightPanelTab } from "../model";

export interface RightPanelDeps {
  model: Model;
}

export class RightPanelBinder {
  constructor(private deps: RightPanelDeps) {}

  /** Wire the tab buttons; call once from the controller. */
  bind(): void {
    const $tabs = document.getElementById("rp-tabs");
    if ($tabs) {
      for (const btn of $tabs.querySelectorAll<HTMLElement>("[data-rp-tab]")) {
        btn.addEventListener("click", () => {
          const tab = btn.dataset.rpTab;
          if (tab) this.setTab(tab as RightPanelTab);
        });
      }
    }
    this.sync();
  }

  /** Activate one tab and persist it on the model. */
  setTab(tab: RightPanelTab): void {
    this.deps.model.rightPanelTab = tab;
    this.deps.model.saveLS();
    this.sync();
  }

  /** Mirror `model.rightPanelTab` onto the buttons and the tagged sections. */
  sync(): void {
    const active = this.deps.model.rightPanelTab;
    const $tabs = document.getElementById("rp-tabs");
    if ($tabs) {
      for (const btn of $tabs.querySelectorAll<HTMLElement>("[data-rp-tab]")) {
        const on = btn.dataset.rpTab === active;
        btn.classList.toggle("active", on);
        btn.setAttribute("aria-selected", on ? "true" : "false");
      }
    }
    for (const el of document.querySelectorAll<HTMLElement>("[data-rp-panel]")) {
      el.hidden = el.dataset.rpPanel !== active;
    }
  }
}
