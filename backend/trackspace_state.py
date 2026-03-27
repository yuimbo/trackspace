"""Process-wide Trackspace runtime: caches, roots, embedding/SSE state, thread pool."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from watchdog.observers import Observer

from backend.cache import TrackCache
from backend.embeddings.feature_cache import FeatureCache
from backend.services.embedding_broadcaster import EmbeddingBroadcaster
from backend.services.embedding_status_cache import EmbeddingStatusCache

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
    embed_status: dict[str, Any]
    embed_lock: threading.Lock
    embed_broadcaster: EmbeddingBroadcaster
    status_cache: EmbeddingStatusCache
    roots: OrderedDict[str, str]
    roots_state_path: str
    observer: Observer | None = None
    root_watches: dict[str, object] = field(default_factory=dict)
    music_root: str = ""

    @classmethod
    def create(cls) -> TrackspaceState:
        data_dir = os.path.join(PROJECT_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        dist_dir = os.path.join(PROJECT_DIR, "frontend", "dist")
        embed_status: dict[str, Any] = {
            "running": False,
            "done": 0,
            "total": 0,
            "error": None,
        }
        return cls(
            data_dir=data_dir,
            dist_dir=dist_dir,
            thread_pool=ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 4) * 4),
            ),
            track_cache=TrackCache(os.path.join(data_dir, "track_cache.db")),
            feature_cache=FeatureCache(os.path.join(data_dir, "audio_features.db")),
            embed_status=embed_status,
            embed_lock=threading.Lock(),
            embed_broadcaster=EmbeddingBroadcaster(),
            status_cache=EmbeddingStatusCache(
                track_infos_max=8,
                status_coverage_max=48,
            ),
            roots=OrderedDict(),
            roots_state_path=os.path.join(data_dir, "roots.json"),
        )
