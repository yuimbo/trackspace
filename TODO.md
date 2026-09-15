# Trackspace — engineering TODO

Concrete, actionable work. Structural refactors are in `architecture_plan.md`;
design rules are in `AGENTS.md`.

## Smaller cleanups

- [ ] **`TrackspaceState` lifecycle (optional).** `create()` still runs at import
      time; see `architecture_plan.md`.

## Frontend

- [ ] Extract `EmbeddingCoordinator`, `FolderTreeCoordinator`,
      `CanvasInteractionController`, and `OptionsPanelBinder`.
      See `architecture_plan.md` for the full ordering.
- [ ] Optimize / speed up t-sne? Can it maybe be done iteratively or something?
