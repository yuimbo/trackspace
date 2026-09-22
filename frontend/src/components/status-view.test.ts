import { beforeEach, describe, expect, it } from "vitest";
import { Model, type QueueKindStatus, type QueueStatus } from "../model";
import { StatusView } from "./status-view";

function kind(overrides: Partial<QueueKindStatus> = {}): QueueKindStatus {
  return {
    label: "Label",
    pending: 0,
    running: 0,
    done: 0,
    failed: 0,
    total: 0,
    finished: 0,
    outstanding: 0,
    ...overrides,
  };
}

function queue(overrides: Partial<QueueStatus> = {}): QueueStatus {
  return {
    kinds: {},
    order: [],
    outstanding: 0,
    total: 0,
    finished: 0,
    active: false,
    epoch: 0,
    ...overrides,
  };
}

function statusText(): string {
  return document.getElementById("status-text")!.textContent ?? "";
}

beforeEach(() => {
  document.body.innerHTML = '<span id="status-text"></span>';
});

describe("StatusView analysis progress", () => {
  it("shows the running stage with its progress and overall percent", () => {
    const view = new StatusView();
    const model = new Model();
    model.queueStatus = queue({
      active: true,
      outstanding: 2,
      total: 10,
      finished: 4,
      order: ["features", "maest"],
      kinds: {
        features: kind({ outstanding: 1, total: 5, finished: 3 }),
        maest: kind({ outstanding: 1, total: 5, finished: 1 }),
      },
    });
    view.update(model);
    expect(statusText()).toContain("Audio features 3 / 5 · 40%");
  });

  it("shows a singleton stage's named phase instead of a bare count", () => {
    const view = new StatusView();
    const model = new Model();
    model.queueStatus = queue({
      active: true,
      outstanding: 1,
      total: 1,
      finished: 0,
      order: ["cluster"],
      kinds: {
        cluster: kind({
          label: "Clustering",
          running: 1,
          outstanding: 1,
          total: 1,
          stage: { phase: "leiden", done: 2, total: 6, at: 1 },
        }),
      },
    });
    view.update(model);
    expect(statusText()).toContain("Clustering Leiden 2 / 6 · 33%");
  });

  it("omits the percent while a tail stage reports no steps", () => {
    const view = new StatusView();
    const model = new Model();
    model.queueStatus = queue({
      active: true,
      outstanding: 1,
      total: 1,
      finished: 0,
      order: ["supervise"],
      kinds: {
        supervise: kind({
          label: "Folder supervision",
          running: 1,
          outstanding: 1,
          total: 1,
          stage: { phase: "mining", done: 0, total: 0, at: 1 },
        }),
      },
    });
    view.update(model);
    expect(statusText()).toContain("Folder supervision hard negatives");
    expect(statusText()).not.toContain("%");
  });

  it("shows the layout phase while a projection is in flight", () => {
    const view = new StatusView();
    const model = new Model();
    model.projectionPending = true;
    model.layoutProgress = { phase: "jacobian", done: 2, total: 5 };
    view.update(model);
    expect(statusText()).toContain("Layout (Tags): Jacobian 2 / 5");
  });

  it("shows no analysis segment when the queue is idle", () => {
    const view = new StatusView();
    const model = new Model();
    model.queueStatus = queue({ active: false, outstanding: 0 });
    view.update(model);
    expect(statusText()).not.toContain("Clustering");
  });

  it("shows a note while the head's predictions are being fetched", () => {
    const view = new StatusView();
    const model = new Model();
    model.folderPredictionsLoading = true;
    view.update(model);
    expect(statusText()).toContain("Predicting folders");
    model.folderPredictionsLoading = false;
    view.update(model);
    expect(statusText()).not.toContain("Predicting folders");
  });
});
