# Trackspace — engineering TODO

Concrete, actionable work. Structural refactors are in `architecture_plan.md`;
design rules are in `AGENTS.md`.

## Highest value

- [ ] **De-duplicate `feature_cache.py`.** Six sources repeat the same
      `get / put / has / get_all / count / purge` block across ~790 lines. Add a
      generic descriptor (blob column, version column, LRU prefix, extra columns)
      and drive all sources from it. The MAEST pair needs one escape hatch: its
      embedding and logits share a version column and are written together by
      `put_maest()` in a single statement.

- [ ] **Extract the two remaining seams from `layout.py`** (~940 lines):
      semantic weighting (folder centroid decomposition, text fallbacks, low-rank
      application) and the projection backends (PCA / cuML / mlx-vis / sklearn).

- [ ] **Bound the PCA reference globals in `layout.py`.**
      `_pca_ref_paths` / `_pca_ref_coords` grow for the process lifetime.
      Fold them into the bounded tier-cache pattern the other caches already use,
      and add hit/miss/eviction counters.

- [ ] **Verify the EffNet ONNX download by checksum, not size.**
      `effnet.py` accepts anything ≥ `_EFFNET_MODEL_MIN_BYTES` (10 MB), so a
      truncated or corrupted download is silently used. MAEST already verifies
      sha256 against `checkpoints.json` — mirror that for EffNet.

## Correctness & resilience

- [ ] **Centralised request validation.** No helpers exist for required fields,
      types, or enum checks (`sources`, projection `method`) across `/api/*`
      POST routes. Add one module and use it at every entry point.

- [ ] **Structured JSON errors for batch IO failures.** Tag writes and file
      operations can raise out of a batch handler and surface as an uncaught 500.

- [ ] **Thread-contention tests for `feature_cache`.** Concurrent reads and writes
      under generation load are the real usage pattern; the current tests are
      single-threaded. Assert lock usage and the SQLite connection strategy hold.

- [ ] **LRU tests for prefixed keys.** Cover eviction correctness across mixed
      source access (`fp`, `effnet:fp`, `feat:fp`, `maest:fp`, `maestlog:fp`,
      `rhythm:fp`).

## Test coverage gaps

- [ ] **Filesystem watcher paths.** No tests for `TrackspaceFilesystemHandler`
      `on_created` / `on_moved`: cache remap on move, and that a new MP3 is
      queued for analysis rather than analysed inline.

- [ ] **Multi-root integration ("no hidden scope").** `test_roots_registry.py`
      covers registry logic only. Add full-library listing, and embedding
      status/projection with an empty folder, across more than one root.

- [ ] **Frontend assertions where feasible** for the queue/cluster panel wiring.

## Smaller cleanups

- [ ] **Type the scattered tuning constants.** `_BPM_LO` / `_BPM_HI`
      (`librosa_audio_features.py`), `_EFFNET_MODEL_MIN_BYTES` /
      `_DOWNLOAD_MAX_RETRIES` (`effnet.py`), and `FROZEN_FEATURES_BLEND`
      (`projection_config.py`) live in three places with no single reference.

- [ ] **Progress callback for `generate_effnet_embeddings_batch`.** Long batches
      cannot report incremental status to the caller.

- [ ] **`TrackspaceState` lifecycle (optional).** `create()` still runs at import
      time; see `architecture_plan.md`.

## Frontend

- [ ] Extract `AnalysisCoordinator` (queue SSE, status polling, cluster fetch and
      panel) from `controller.ts` — the newest and most self-contained block.
- [ ] Then `EmbeddingCoordinator`, `FolderTreeCoordinator`,
      `CanvasInteractionController`, `OptionsPanelBinder`, and `lib/api.ts`.
      See `architecture_plan.md` for the full ordering.
