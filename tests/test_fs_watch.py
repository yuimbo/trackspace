"""Tests for backend.fs_watch.TrackspaceFilesystemHandler.

The handler takes all its collaborators as injected callables
(``FilesystemWatchDeps``), so these tests drive the real ``on_created`` /
``on_moved`` paths against fakes — no Flask app, no models, no real fingerprinting.

Two properties matter most here:

* a directory/file move re-keys the track cache so entries survive under the new
  path instead of becoming stale orphans;
* a newly-seen MP3 is handed to the durable queue (``enqueue_track``) rather than
  analysed inline, and a file without a fingerprint is never queued.
"""

from __future__ import annotations

import os
import threading

from watchdog.events import (
    DirCreatedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileMovedEvent,
)

from backend.fs_watch import FilesystemWatchDeps, TrackspaceFilesystemHandler


class _FakeTrackCache:
    def __init__(self) -> None:
        self.remaps: list[tuple[str, str]] = []
        self.prefix_remaps: list[tuple[str, str]] = []

    def remap(self, old_path: str, new_path: str) -> int:
        self.remaps.append((old_path, new_path))
        return 1

    def remap_prefix(self, old_prefix: str, new_prefix: str) -> int:
        self.prefix_remaps.append((old_prefix, new_prefix))
        return 1


class _Harness:
    """Wires fake deps and records everything the handler does."""

    def __init__(
        self,
        *,
        read_result: dict | None = None,
        read_raises: bool = False,
        enqueue_track=None,
    ) -> None:
        self.cache = _FakeTrackCache()
        self.read_calls: list[str] = []
        self.enqueued: list[str] = []
        self.read_event = threading.Event()
        self.enqueue_event = threading.Event()
        self._read_result = read_result if read_result is not None else {
            "fingerprint": "fp-1"
        }
        self._read_raises = read_raises

        if enqueue_track is None:
            def _default_enqueue(rel: str) -> None:
                self.enqueued.append(rel)
                self.enqueue_event.set()

            enqueue_track = _default_enqueue
        self._enqueue_track = enqueue_track

        self.deps = FilesystemWatchDeps(
            track_cache=self.cache,
            cached_read_all=self._read,
            virtual_from_abs=lambda p: f"root/{os.path.basename(p)}",
            is_mp3_path=lambda name: name.lower().endswith(".mp3"),
            enqueue_track=self._enqueue_track,
        )
        self.handler = TrackspaceFilesystemHandler(self.deps)

    def _read(self, path: str) -> dict:
        self.read_calls.append(path)
        self.read_event.set()
        if self._read_raises:
            raise OSError("unreadable")
        return dict(self._read_result)


def _mp3(tmp_path, name: str = "song.mp3") -> str:
    p = tmp_path / name
    p.write_bytes(b"ID3\x00fake")
    return str(p)


def _no_settle(monkeypatch) -> None:
    """Make the settle poll finish immediately — no real 3 s waits in CI."""
    monkeypatch.setattr(TrackspaceFilesystemHandler, "_SETTLE_SECS", 0.03)
    monkeypatch.setattr(TrackspaceFilesystemHandler, "_SETTLE_POLLS", 3)


# ─── on_moved ────────────────────────────────────────────────


def test_dir_moved_remaps_prefix(tmp_path):
    h = _Harness()
    src = str(tmp_path / "old")
    dst = str(tmp_path / "new")
    h.handler.on_moved(DirMovedEvent(src, dst))

    assert h.cache.prefix_remaps == [(src + os.sep, dst + os.sep)]
    assert h.cache.remaps == []


def test_file_moved_remaps_entry(tmp_path):
    h = _Harness()
    src = str(tmp_path / "a.mp3")
    dst = str(tmp_path / "b.txt")  # non-mp3 dest: remap only, no ingest
    h.handler.on_moved(FileMovedEvent(src, dst))

    assert h.cache.remaps == [(src, dst)]
    assert h.cache.prefix_remaps == []
    assert h.read_calls == []


def test_file_moved_to_mp3_enqueues(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    h = _Harness()
    src = str(tmp_path / "a.mp3")
    dst = _mp3(tmp_path, "moved.mp3")
    h.handler.on_moved(FileMovedEvent(src, dst))

    assert h.enqueue_event.wait(2.0), "ingest never enqueued the moved file"
    assert h.cache.remaps == [(src, dst)]
    assert h.read_calls == [dst]
    assert h.enqueued == ["root/moved.mp3"]


def test_file_moved_from_non_mp3_to_mp3_enqueues(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    h = _Harness()
    dst = _mp3(tmp_path, "landed.mp3")
    h.handler.on_moved(FileMovedEvent(str(tmp_path / "incoming.part"), dst))

    assert h.enqueue_event.wait(2.0)
    assert h.enqueued == ["root/landed.mp3"]


# ─── on_created ─────────────────────────────────────────────


def test_file_created_mp3_enqueues(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    h = _Harness()
    src = _mp3(tmp_path, "fresh.mp3")
    h.handler.on_created(FileCreatedEvent(src))

    assert h.enqueue_event.wait(2.0), "ingest never enqueued the new file"
    assert h.read_calls == [src]
    assert h.enqueued == ["root/fresh.mp3"]


def test_file_created_non_mp3_ignored(tmp_path):
    h = _Harness()
    h.handler.on_created(FileCreatedEvent(str(tmp_path / "cover.jpg")))

    assert h.read_calls == []
    assert h.enqueued == []


def test_dir_created_ignored(tmp_path):
    h = _Harness()
    h.handler.on_created(DirCreatedEvent(str(tmp_path / "newdir")))

    assert h.read_calls == []
    assert h.enqueued == []


# ─── _ingest_new edge cases (driven synchronously) ───────────


def test_ingest_missing_file_never_reads(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    h = _Harness()
    h.handler._ingest_new(str(tmp_path / "gone.mp3"))

    assert h.read_calls == []
    assert h.enqueued == []


def test_ingest_without_fingerprint_not_enqueued(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    path = _mp3(tmp_path)
    h = _Harness(read_result={"tags": {}})
    h.handler._ingest_new(path)

    assert h.read_calls == [path]
    assert h.enqueued == []


def test_ingest_read_error_is_swallowed(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    path = _mp3(tmp_path)
    h = _Harness(read_raises=True)
    h.handler._ingest_new(path)

    assert h.read_calls == [path]
    assert h.enqueued == []


def test_ingest_enqueue_none_is_safe(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    path = _mp3(tmp_path)
    h = _Harness(enqueue_track=None)
    h.deps.enqueue_track = None
    h.handler._ingest_new(path)

    assert h.read_calls == [path]
    assert h.enqueued == []


def test_ingest_enqueue_error_is_swallowed(monkeypatch, tmp_path):
    _no_settle(monkeypatch)
    path = _mp3(tmp_path)

    def boom(_rel: str) -> None:
        raise RuntimeError("queue down")

    h = _Harness(enqueue_track=boom)
    # Must not propagate out of the ingest worker.
    h.handler._ingest_new(path)
