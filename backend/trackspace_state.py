"""Process-wide Trackspace runtime: caches, roots, analysis queue, SSE, thread pool."""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from watchdog.observers import Observer

from backend.cache import TrackCache
from backend.embeddings.feature_cache import (
    LIBRARY_MEMORY_BYTES,
    LIBRARY_MEMORY_ENTRIES,
    FeatureCache,
)
from backend.jobs import JobQueue, WorkerPool
from backend.services.analysis_pipeline import ClusterStore
from backend.services.embedding_broadcaster import EmbeddingBroadcaster
from backend.services.embedding_status_cache import EmbeddingStatusCache
from backend.services.similarity import SimilarityIndex

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)


@dataclass
class TrackspaceState:
    """Single bag of mutable services and library roots for one server process."""

    data_dir: str
    dist_dir: str
    thread_pool: ThreadPoolExecutor
    track_cache: TrackCache
    feature_cache: FeatureCache
    embed_broadcaster: EmbeddingBroadcaster
    status_cache: EmbeddingStatusCache
    roots: OrderedDict[str, str]
    roots_state_path: str
    #: Durable analysis queue — survives restarts, so work resumes where it stopped.
    job_queue: JobQueue
    #: Background workers, one per job kind.
    workers: WorkerPool
    #: Latest multi-resolution clustering of the library.
    clusters: ClusterStore
    #: Lazily-built cosine index over the composite vectors (inspect card).
    similarity: SimilarityIndex
    observer: Observer | None = None
    root_watches: dict[str, object] = field(default_factory=dict)
    music_root: str = ""
    #: Root ids registered only to imprint an external folder (scan/analyse/fan-out)
    #: and removed when the imprint finishes — they are not part of the library.
    temp_roots: set[str] = field(default_factory=set)
    #: Per-root role: ``"library"`` (default) or ``"external"``. External roots are
    #: analysed and laid out like any other, but are kept out of the default view and
    #: the supervised head's training labels. Missing ids mean ``"library"``.
    root_roles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, data_dir: str | None = None) -> TrackspaceState:
        # ``TRACKSPACE_DATA_DIR`` lets tests and throwaway runs point caches,
        # roots, and the job queue at a scratch directory instead of the real one.
        data_dir = (
            data_dir
            or os.environ.get("TRACKSPACE_DATA_DIR")
            or os.path.join(PROJECT_DIR, "data")
        )
        os.makedirs(data_dir, exist_ok=True)
        dist_dir = os.path.join(PROJECT_DIR, "frontend", "dist")
        return cls(
            data_dir=data_dir,
            dist_dir=dist_dir,
            thread_pool=ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 4) * 4),
            ),
            # Sized to hold the whole library in memory: a smaller LRU thrashes on
            # a library-wide scan and every page load re-hits SQLite per track.
            track_cache=TrackCache(
                os.path.join(data_dir, "track_cache.db"), max_memory=20000
            ),
            feature_cache=FeatureCache(
                os.path.join(data_dir, "audio_features.db"),
                max_memory=LIBRARY_MEMORY_ENTRIES,
                max_memory_bytes=LIBRARY_MEMORY_BYTES,
            ),
            embed_broadcaster=EmbeddingBroadcaster(),
            status_cache=EmbeddingStatusCache(
                track_infos_max=8,
                status_coverage_max=48,
            ),
            roots=OrderedDict(),
            roots_state_path=os.path.join(data_dir, "roots.json"),
            job_queue=JobQueue(os.path.join(data_dir, "jobs.db")),
            workers=WorkerPool(),
            clusters=ClusterStore(),
            similarity=SimilarityIndex(),
        )
