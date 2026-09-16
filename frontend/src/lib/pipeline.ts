// ─── Analysis pipeline summary ─────────────────────────────────
//
// The queue is the single source of truth for "what is the server doing", but its
// raw shape is per-kind counts plus, for library-wide singleton stages
// (supervise/cluster), an in-flight ``stage`` phase. Both the status bar and the
// on-canvas indicator want the same derived answer — "which stage, how far along,
// overall percent" — so the reduction lives here, pure and unit-testable.

import type { QueueKindStatus, QueueStatus, QueueStageStatus } from "../model";

export type { QueueStageStatus };

export interface PipelineProgress {
  active: boolean;
  /** Overall completion across every queued stage, 0..1. */
  fraction: number;
  /** Friendly name of the stage currently running, e.g. "MAEST embeddings". */
  label: string;
  /** The running stage's progress, e.g. "12 / 64" or "Leiden 3 / 6". */
  detail: string;
  /** Per-track embedding stages are still outstanding. */
  generating: boolean;
  /** Only the library-wide tail (supervision/clustering) remains. */
  tail: boolean;
  /** The running tail stage reported no step count, so the bar cannot be determinate. */
  indeterminate: boolean;
}

/** Friendly names for job kinds — action-oriented, unlike the server's nouns. */
const STAGE_LABELS: Record<string, string> = {
  scan: "Scanning library",
  features: "Audio features",
  structure: "Track structure",
  maest: "MAEST embeddings",
  clap: "CLAP embeddings",
  supervise: "Folder supervision",
  cluster: "Clustering",
};

/** Friendly names for the phase strings the singleton handlers emit. */
const PHASE_LABELS: Record<string, string> = {
  vectors: "gathering vectors",
  gather: "gathering vectors",
  pca: "PCA",
  graph: "kNN graph",
  leiden: "Leiden",
  mining: "hard negatives",
  training: "training",
  projecting: "projecting rows",
  done: "done",
  // Layout phases (streamed as `layout_progress`, not a queued job).
  layout: "laying out",
  chart: "landmarks",
  jacobian: "Jacobian",
};

/** Kinds that produce the per-track embeddings the map is built from. */
const GENERATING_KINDS = new Set(["features", "maest", "clap"]);

/** Library-wide singleton kinds, each one job among thousands of per-track rows. */
const TAIL_KINDS = new Set(["supervise", "cluster"]);

export function phaseLabel(phase: string): string {
  return PHASE_LABELS[phase] ?? phase;
}

/** Human text for a running stage: phase name plus counts when it has steps. */
export function stageDetail(kind: string, k: QueueKindStatus): string {
  const stage = k.stage;
  if (stage) {
    const phase = phaseLabel(stage.phase);
    return stage.total > 1 ? `${phase} ${stage.done} / ${stage.total}` : phase;
  }
  return `${k.finished} / ${k.total}`;
}

/**
 * Reduce a queue snapshot to one progress summary, or ``null`` when idle.
 *
 * The running stage is the first (pipeline order) kind with outstanding work; the
 * overall fraction is finished/total across the whole queue, so it moves steadily
 * through features → MAEST → CLAP → supervision → clustering.
 */
export function pipelineProgress(q: QueueStatus | null): PipelineProgress | null {
  if (!q || !q.active) return null;

  let label = "";
  let detail = "";
  let generating = false;
  let firstKind = "";
  for (const kind of q.order) {
    const k = q.kinds[kind];
    if (!k || k.outstanding <= 0) continue;
    if (GENERATING_KINDS.has(kind)) generating = true;
    if (!label) {
      label = STAGE_LABELS[kind] ?? k.label;
      detail = stageDetail(kind, k);
      firstKind = kind;
    }
  }
  if (!label) return null;

  // The overall fraction is finished/total across the whole queue, which a
  // library-wide singleton can never move: it is one job among thousands, so the
  // bar sits pinned near 100 % for its entire run. When the only work left is a
  // tail stage, drive the bar from that stage's own step count instead — and mark
  // it indeterminate while the stage reports no steps (gather/mining).
  const first = q.kinds[firstKind];
  const stage = first?.stage;
  let indeterminate = false;
  let fraction = q.total > 0 ? Math.min(1, q.finished / q.total) : 0;
  if (TAIL_KINDS.has(firstKind)) {
    if (stage && stage.total > 0) fraction = Math.min(1, stage.done / stage.total);
    else indeterminate = true;
  }

  return {
    active: true,
    fraction,
    label,
    detail,
    generating,
    tail: !generating,
    indeterminate,
  };
}
