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
    embeddings_mod: Any
    embedding_version: int
    feature_cache: Any
    embed_lock: threading.Lock
    embed_status: dict[str, Any]
    broadcast_embed_event: Callable[[dict[str, Any]], None]
    emit_decode_warning_once: Callable[[str, str, set[str]], None]


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

        fp = data.get("fingerprint")
        if not fp or not d.embeddings_mod.is_model_ready():
            return
        if d.feature_cache.has_embedding(fp, version=d.embedding_version):
            return

        with d.embed_lock:
            if d.embed_status["running"]:
                return
            d.embed_status.update({"running": True, "done": 0, "total": 1, "error": None})

        def _run() -> None:
            try:
                warn_seen: set[str] = set()
                emb = d.embeddings_mod.generate_embedding(
                    abs_path,
                    on_decode_warning=lambda m: d.emit_decode_warning_once(
                        rel_path, m, warn_seen,
                    ),
                )
                ok = emb is not None
                if ok:
                    d.feature_cache.put_embedding(fp, emb, version=d.embedding_version)
                evt: dict[str, Any] = {
                    "type": "progress",
                    "path": rel_path,
                    "ok": ok,
                    "done": 1,
                    "total": 1,
                }
                if not ok:
                    evt["failures"] = ["clap"]
                    log.warning("Watchdog: CLAP embedding failed for %s", rel_path)
                d.broadcast_embed_event(evt)
                log.info("Watchdog: embedding %s for %s", "ok" if ok else "failed", rel_path)
            except Exception as e:
                log.exception("Watchdog: embedding generation failed for %s", rel_path)
                with d.embed_lock:
                    d.embed_status["error"] = str(e)
                d.broadcast_embed_event({"type": "error", "error": str(e)})
            finally:
                with d.embed_lock:
                    d.embed_status["running"] = False
                d.broadcast_embed_event({"type": "done"})

        threading.Thread(target=_run, daemon=True).start()
