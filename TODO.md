# Trackspace — engineering TODO

Concrete, actionable work. Structural refactors are in `architecture_plan.md`;
design rules are in `AGENTS.md`.

## Test coverage gaps

- [ ] **Frontend assertions where feasible** for the queue/cluster panel wiring.
      Deferred into the `AnalysisCoordinator` extraction below — unit-testing an
      extracted coordinator is feasible; testing `controller.ts` in place is not.
      Add vitest as part of that extraction.

## Smaller cleanups

- [ ] **`TrackspaceState` lifecycle (optional).** `create()` still runs at import
      time; see `architecture_plan.md`.

## Frontend

- [ ] Extract `AnalysisCoordinator` (queue SSE, status polling, cluster fetch and
      panel) from `controller.ts` — the newest and most self-contained block.
- [ ] Then `EmbeddingCoordinator`, `FolderTreeCoordinator`,
      `CanvasInteractionController`, `OptionsPanelBinder`, and `lib/api.ts`.
      See `architecture_plan.md` for the full ordering.
- [ ] Optimize / speed up t-sne? Can it maybe be done iteratively or something?
