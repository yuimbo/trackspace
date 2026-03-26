# Python >400 LOC analysis TODOs

## `backend/app.py` (980 lines)
- [ ] Split route handlers into focused modules (`routes_tracks.py`, `routes_tags.py`, `routes_embeddings.py`, `routes_folders.py`) and register with Flask blueprints.
- [ ] Extract embedding job state/SSE subscriber logic (`_embed_status`, `_embed_lock`, `_embed_subscribers`, `_broadcast_embed_event`) into a dedicated `EmbeddingJobManager` class to reduce shared global state.
- [ ] Add centralized request validation helpers for JSON payloads and query params (required fields, type checks, enum checks for sources/methods) and use them across all `/api/*` POST routes.
- [ ] Add per-endpoint error handling that returns structured JSON errors instead of uncaught 500s when file IO/tag writes fail in batch operations.
- [ ] Add tests for `api_embeddings_generate` covering mixed source requests, priority ordering, model-not-ready responses, and "generation already in progress" behavior.
- [ ] Add tests for filesystem watchdog ingestion paths (`on_created`, `on_moved`) to verify cache remap + auto-embed behavior for moved/new MP3s.

## `backend/embeddings.py` (1127 lines)
- [ ] Decompose into smaller modules: `clap_model.py` (load/infer), `projection_backends.py` (PCA/UMAP/t-SNE), `semantic_weighting.py` (folder/tag basis), and `projection_cache.py` (tiered caches).
- [ ] Replace unbounded process-lifetime globals (`_source_row_cache`, `_pca_ref_*`) with a bounded cache object that supports explicit reset + telemetry counters (hits/misses/evictions).
- [x] Add cache invalidation tests that verify semantic/layout cache keys change when source versions, feature mask, or blend parameters change.
- [ ] Add numerical stability tests for `_apply_low_rank_semantic_weight` and `_build_folder_directions` (zero norms, tiny clusters, degenerate siblings).
- [ ] Add backend-selection tests for cuML/mlx-vis fallback paths to ensure runtime exceptions correctly disable that backend and continue on CPU.
- [x] Add timing instrumentation logs around `compute_projection` stages (gather, semantic weighting, projection backend) for easier profiling and regression detection.

## `backend/audio_features.py` (547 lines)
- [ ] Move EffNet model download/load concerns into a small `EffnetModelManager` class and keep feature extraction logic separate from model lifecycle.
- [ ] Add checksum or ETag verification for downloaded ONNX model files (not only minimum-byte checks) to prevent silently using corrupted artifacts.
- [ ] Replace magic constants (`_BPM_LO`, `_BPM_HI`, blend limits, excerpt multipliers) with a typed config object and document defaults in one place.
- [ ] Add tests for `_load_feature_audio_excerpt` and `_load_effnet_segments` boundary behavior around threshold durations.
- [ ] Add tests for `extract_audio_features` output contract (shape, dtype, bounded ranges for tempo/energy/danceability, key unit-circle constraints).
- [ ] Add optional coarse progress callback for `generate_effnet_embeddings_batch` so long batch runs can emit incremental status to the caller.

## `backend/feature_cache.py` (404 lines)
- [ ] Introduce a small generic helper for repeated get/put/has/get_all/count/purge patterns across CLAP/EffNet/features to eliminate copy-paste SQL logic.
- [ ] Use explicit SQLite transactions (`BEGIN IMMEDIATE`) for batch writes and add `executemany` paths to reduce write amplification during generation jobs.
- [ ] Add schema migration/version table support so new columns can be added deterministically instead of best-effort `ALTER TABLE` inside try/except.
- [x] Add DB-level integrity checks at startup (`PRAGMA integrity_check`, optional) with warning logs and recovery guidance when corruption is detected.
- [ ] Add targeted unit tests for LRU behavior with prefixed keys (`fp`, `effnet:fp`, `feat:fp`) and eviction correctness across mixed source access.
- [ ] Add thread-contention tests for concurrent reads/writes to verify lock usage and sqlite connection strategy stay safe under embedding generation load.