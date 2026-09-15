"""Filesystem watchdog handler — cache remaps on move + background CLAP ingest for new MP3s."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from watchdog.events import DirMovedEvent, FileCreatedEvent, FileMovedEvent, FileSystemEventHandler

log = logging.getLogger(__name__)


@dataclass
class FilesystemWatchDeps:
    track_cache: Any
    cached_read_all: Callable[[str], dict[str, Any]]
    virtual_from_abs: Callable[[str], str]
    is_mp3_path: Callable[[str], bool]
    #: Queue analysis for one newly-seen virtual path (durable; see backend.jobs).
    #: Model handles, cache versions, and the old embed_status flag are gone —
    #: the watcher only reports *that* a file appeared; the queue decides what
    #: to do about it and when.
    enqueue_track: Callable[[str], None] | None = None


class TrackspaceFilesystemHandler(FileSystemEventHandler):
    """Remap track cache on moves; ingest new MP3s (tags + optional CLAP embedding)."""

    _SETTLE_SECS = 3.0
    _SETTLE_POLLS = 3

    def __init__(self, deps: FilesystemWatchDeps) -> None:
        self._deps = deps

    def on_moved(self, event) -> None:
        d = self._deps
        if isinstance(event, DirMovedEvent):
            old_prefix = event.src_path.rstrip(os.sep) + os.sep
            new_prefix = event.dest_path.rstrip(os.sep) + os.sep
            d.track_cache.remap_prefix(old_prefix, new_prefix)
        elif isinstance(event, FileMovedEvent):
            d.track_cache.remap(event.src_path, event.dest_path)
            if d.is_mp3_path(event.dest_path):
                threading.Thread(
                    target=self._ingest_new,
                    args=(event.dest_path,),
                    daemon=True,
                ).start()

    def on_created(self, event) -> None:
        d = self._deps
        if isinstance(event, FileCreatedEvent) and d.is_mp3_path(event.src_path):
            threading.Thread(
                target=self._ingest_new,
                args=(event.src_path,),
                daemon=True,
            ).start()

    def _ingest_new(self, abs_path: str) -> None:
        d = self._deps
        prev_size = -1
        for _ in range(self._SETTLE_POLLS):
            time.sleep(self._SETTLE_SECS / self._SETTLE_POLLS)
            try:
                cur_size = os.path.getsize(abs_path)
            except OSError:
                return
            if cur_size == prev_size:
                break
            prev_size = cur_size

        if not os.path.isfile(abs_path):
            return

        try:
            data = d.cached_read_all(abs_path)
        except Exception:
            log.exception("Watchdog: failed to read new file %s", abs_path)
            return

        rel_path = d.virtual_from_abs(abs_path)
        log.info("Watchdog: ingested %s", rel_path)

        if not data.get("fingerprint"):
            return

        # Hand the work to the durable queue rather than spawning an analysis
        # thread here. A burst of file events therefore produces queued jobs
        # (deduped, resumable, retried) instead of racing threads guarded by a
        # single global "running" flag that dropped work when it was set.
        if d.enqueue_track is not None:
            try:
                d.enqueue_track(rel_path)
            except Exception:
                log.exception("Watchdog: failed to queue analysis for %s", rel_path)
