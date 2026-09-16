import { describe, expect, it } from "vitest";
import { pipelineProgress, stageDetail } from "./pipeline";
import type { QueueKindStatus, QueueStatus } from "../model";

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

describe("pipelineProgress", () => {
  it("is null when idle", () => {
    expect(pipelineProgress(null)).toBeNull();
    expect(pipelineProgress(queue({ active: false, outstanding: 0 }))).toBeNull();
  });

  it("picks the first running stage in pipeline order", () => {
    const prog = pipelineProgress(
      queue({
        active: true,
        outstanding: 2,
        total: 10,
        finished: 4,
        order: ["features", "maest"],
        kinds: {
          features: kind({ label: "Audio features", outstanding: 1, total: 5, finished: 3 }),
          maest: kind({ label: "MAEST analysis", outstanding: 1, total: 5, finished: 1 }),
        },
      }),
    );
    expect(prog).not.toBeNull();
    expect(prog!.label).toBe("Audio features");
    expect(prog!.detail).toBe("3 / 5");
    expect(prog!.fraction).toBeCloseTo(0.4);
    expect(prog!.generating).toBe(true);
    expect(prog!.tail).toBe(false);
  });

  it("reports the singleton phase with its step count", () => {
    const prog = pipelineProgress(
      queue({
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
      }),
    );
    expect(prog!.label).toBe("Clustering");
    expect(prog!.detail).toBe("Leiden 2 / 6");
    expect(prog!.generating).toBe(false);
    expect(prog!.tail).toBe(true);
    // A singleton is 1 job among thousands, so the queue-wide fraction is
    // pinned near 100 %; the tail stage's own steps must drive the bar.
    expect(prog!.fraction).toBeCloseTo(2 / 6);
    expect(prog!.indeterminate).toBe(false);
  });

  it("marks an opaque tail phase indeterminate instead of near-complete", () => {
    const prog = pipelineProgress(
      queue({
        active: true,
        outstanding: 1,
        total: 9501,
        finished: 9500,
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
      }),
    );
    expect(prog!.indeterminate).toBe(true);
    expect(prog!.detail).toBe("hard negatives");
  });

  it("shows a one-shot phase without counts", () => {
    expect(
      stageDetail(
        "supervise",
        kind({ stage: { phase: "mining", done: 0, total: 0, at: 1 } }),
      ),
    ).toBe("hard negatives");
  });

  it("falls back to finished/total for per-track stages", () => {
    expect(stageDetail("maest", kind({ finished: 7, total: 12 }))).toBe("7 / 12");
  });
});
