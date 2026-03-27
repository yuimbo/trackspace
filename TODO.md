# Python >400 LOC analysis TODOs

## `backend/app.py` (980 lines)
- [ ] Split route handlers into focused modules (`routes_tracks.py`, `routes_tags.py`, `routes_embeddings.py`, `routes_folders.py`) and register with Flask blueprints.
- [ ] Extract embedding job state/SSE subscriber logic (`_embed_status`, `_embed_lock`, `_embed_subscribers`, `_broadcast_embed_event`) into a dedicated `EmbeddingJobManager` class to reduce shared global state.
- [ ] Add centralized request validation helpers for JSON payloads and query params (required fields, type checks, enum checks for sources/methods) and use them across all `/api/*` POST routes.
- [ ] Add per-endpoint error handling that returns structured JSON errors instead of uncaught 500s when file IO/tag writes fail in batch operations.
- [ ] Add tests for `api_embeddings_generate` covering mixed source requests, priority ordering, model-not-ready responses, and "generation already in progress" behavior.
- [ ] Add tests for filesystem watchdog ingestion paths (`on_created`, `on_moved`) to verify cache remap + auto-embed behavior for moved/new MP3s.

## `backend/embeddings/layout.py` (composite layout + projection)
- [ ] Decompose into smaller modules: `clap_model.py` (load/infer), `projection_backends.py` (PCA/UMAP/t-SNE), `semantic_weighting.py` (folder/tag basis), and `projection_cache.py` (tiered caches).
- [ ] Replace unbounded process-lifetime globals (`_source_row_cache`, `_pca_ref_*`) with a bounded cache object that supports explicit reset + telemetry counters (hits/misses/evictions).

## `backend/embeddings/audio_features.py` (+ root shim `backend/audio_features.py`)
- [ ] Move EffNet model download/load concerns into a small `EffnetModelManager` class and keep feature extraction logic separate from model lifecycle.
- [ ] Add checksum or ETag verification for downloaded ONNX model files (not only minimum-byte checks) to prevent silently using corrupted artifacts.
- [ ] Replace magic constants (`_BPM_LO`, `_BPM_HI`, blend limits, excerpt multipliers) with a typed config object and document defaults in one place.
- [ ] Add optional coarse progress callback for `generate_effnet_embeddings_batch` so long batch runs can emit incremental status to the caller.

## `backend/embeddings/feature_cache.py`
- [ ] Introduce a small generic helper for repeated get/put/has/get_all/count/purge patterns across CLAP/EffNet/features to eliminate copy-paste SQL logic.
- [ ] Add targeted unit tests for LRU behavior with prefixed keys (`fp`, `effnet:fp`, `feat:fp`) and eviction correctness across mixed source access.
- [ ] Add thread-contention tests for concurrent reads/writes to verify lock usage and sqlite connection strategy stay safe under embedding generation load.