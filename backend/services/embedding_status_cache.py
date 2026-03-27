from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


TrackInfosKey = tuple[str, bool, str]
StatusCoverageKey = tuple[str, int, str]


class EmbeddingStatusCache:
    """Thread-safe LRUs used by /api/embeddings/status."""

    def __init__(self, *, track_infos_max: int = 8, status_coverage_max: int = 48) -> None:
        self._track_infos_max = track_infos_max
        self._status_coverage_max = status_coverage_max
        self._track_infos_lru: OrderedDict[TrackInfosKey, list[dict[str, Any]]] = OrderedDict()
        self._status_coverage_lru: OrderedDict[StatusCoverageKey, tuple[dict[str, Any], str | None]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def get_track_infos(self, key: TrackInfosKey) -> list[dict[str, Any]] | None:
        with self._lock:
            if key not in self._track_infos_lru:
                return None
            self._track_infos_lru.move_to_end(key)
            return self._track_infos_lru[key]

    def put_track_infos(self, key: TrackInfosKey, infos: list[dict[str, Any]]) -> None:
        with self._lock:
            self._track_infos_lru[key] = infos
            self._track_infos_lru.move_to_end(key)
            while len(self._track_infos_lru) > self._track_infos_max:
                self._track_infos_lru.popitem(last=False)

    def get_status_coverage(self, key: StatusCoverageKey) -> tuple[dict[str, Any], str | None] | None:
        with self._lock:
            if key not in self._status_coverage_lru:
                return None
            self._status_coverage_lru.move_to_end(key)
            base, layout_revision = self._status_coverage_lru[key]
            return dict(base), layout_revision

    def put_status_coverage(
        self,
        key: StatusCoverageKey,
        *,
        base: dict[str, Any],
        layout_revision: str | None,
    ) -> None:
        with self._lock:
            self._status_coverage_lru[key] = (dict(base), layout_revision)
            self._status_coverage_lru.move_to_end(key)
            while len(self._status_coverage_lru) > self._status_coverage_max:
                self._status_coverage_lru.popitem(last=False)
