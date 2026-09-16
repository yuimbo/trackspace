"""The native libsndfile MP3 open must never run concurrently.

libsndfile's ``mpeg_decoder_init`` calls the global, non-thread-safe
``mpg123_init()`` on *every* ``sf_open``.  The ``features`` and ``maest``
analysis workers decode MP3s at the same time, so without
``decode_stderr._DECODE_LOCK`` they race in mpg123's static layer-table init
and the process dies with SIGBUS.  Reproduced standalone against the real
library: 10/10 crashes unlocked, 0/10 with the open serialised.

These tests pin the lock contract deterministically with a fake SoundFile.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from backend import decode_stderr


class _Probe:
    """Records the peak number of overlapping native opens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.inside += 1
            self.calls += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.005)

    def exit(self) -> None:
        with self._lock:
            self.inside -= 1


def _run_concurrently(fn, *, n_threads: int = 4, n_calls: int = 25) -> None:
    def run() -> None:
        for _ in range(n_calls):
            fn()

    threads = [threading.Thread(target=run) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_librosa_load_serializes_the_native_open(monkeypatch):
    probe = _Probe()

    class _FakeSoundFile:
        def __init__(self, path):
            probe.enter()

        def close(self):
            probe.exit()

    def _fake_load(desc, **kwargs):
        try:
            return np.zeros(4, dtype=np.float32), 48000
        finally:
            desc.close()

    monkeypatch.setattr(decode_stderr._soundfile, "SoundFile", _FakeSoundFile)
    monkeypatch.setattr(decode_stderr._librosa, "load", _fake_load)

    _run_concurrently(
        lambda: decode_stderr.librosa_load("a.mp3", sr=48000, mono=True)
    )

    assert probe.calls == 100
    assert probe.peak == 1


def test_librosa_load_falls_back_when_soundfile_cannot_open(monkeypatch):
    """A path soundfile rejects must still reach librosa's own loader (audioread)."""
    seen = {}

    def _boom(path):
        raise decode_stderr._soundfile.SoundFileRuntimeError("unsupported")

    def _fake_load(path, **kwargs):
        seen["path"] = path
        return np.zeros(4, dtype=np.float32), 44100

    monkeypatch.setattr(decode_stderr, "_open_soundfile", _boom)
    monkeypatch.setattr(decode_stderr._librosa, "load", _fake_load)

    y, sr = decode_stderr.librosa_load("weird.ogg", sr=44100, mono=True)

    assert sr == 44100
    assert seen["path"] == "weird.ogg"


def test_librosa_get_duration_serializes_the_native_probe(monkeypatch):
    probe = _Probe()

    def _fake_get_duration(**kwargs):
        probe.enter()
        try:
            return 1.0
        finally:
            probe.exit()

    monkeypatch.setattr(decode_stderr._librosa, "get_duration", _fake_get_duration)

    _run_concurrently(lambda: decode_stderr.librosa_get_duration(path="a.mp3"))

    assert probe.calls == 100
    assert probe.peak == 1
