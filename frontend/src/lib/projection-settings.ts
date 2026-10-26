// ─── Projection settings ──────────────────────────────────────
//
// The cheap read surface behind the View options influences panel. The backend
// parses the same projection query as the layout route and returns the effective
// configuration (including fixed/disabled blocks), so the UI never re-derives
// recipe rules client-side.

export interface ProjectionSettingsSource {
  id: string;
  /** Effective weight in the projection (0 when outside the lens). */
  weight: number;
  /** Server capability: false when an optional model block is off. */
  available: boolean;
  /** Whether this source belongs to the active lens's source set. */
  in_lens: boolean;
}

export interface ProjectionSettingsBudget {
  tsne_n_iter: number;
  tsne_early_exaggeration_iter: number;
  default_tsne_n_iter: number;
  default_tsne_early_exaggeration_iter: number;
  final_exaggeration: number;
}

export interface ProjectionSettingsFolderBasis {
  mode: string;
  boost: number;
  depth_boost: number;
}

export interface ProjectionSettingsCoarseSchedule {
  fraction_count: number;
  min_fraction: number;
  max_fraction: number;
  iteration_budget: number;
  early_exaggeration_iter: number;
  perplexity_gamma: number;
  perplexity_rule: string;
}

export interface ProjectionSettingsAudioFeatures {
  blend: number;
  mask: number[];
  dim: number;
}

export interface ProjectionSettings {
  ok: boolean;
  method: string;
  dims: number;
  recipe: string;
  sources: ProjectionSettingsSource[];
  weights_editable: boolean;
  weights_editable_keys: string[];
  budget: ProjectionSettingsBudget;
  folder_basis: ProjectionSettingsFolderBasis;
  coarse_schedule: ProjectionSettingsCoarseSchedule;
  final_perplexity: {
    rule: string;
    resolved: number | null;
    note: string;
  };
  audio_features: ProjectionSettingsAudioFeatures;
}
