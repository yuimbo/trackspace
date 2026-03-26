"""Suppress libsndfile / libmpg123 chatter on the process stderr (fd 2).

Those libraries print ID3, Xing/VBR, and resync diagnostics directly to the C
runtime stderr, bypassing Python logging.  ``contextlib.redirect_stderr`` does
not help.  We temporarily ``dup2`` fd 2 to ``/dev/null`` or to an optional log
file while decoding.

``TRACKSPACE_DECODE_STDERR_LOG``
    If set to a filesystem path, captured output is appended there instead of
    discarded.  The directory is created as needed.

Redirection is process-wide and holds a lock so nested and concurrent decode
calls do not corrupt the saved fd.  Any other thread that writes to stderr
while the lock is held may interleave or briefly follow the redirected fd;
decode windows are typically short.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator

import librosa as _librosa

_ENV_LOG = "TRACKSPACE_DECODE_STDERR_LOG"


class _Silence:
    __slots__ = ()
    lock = threading.RLock()
    depth = 0
    saved_fd: int | None = None


def _open_target_fd() -> int:
    log_path = (os.environ.get(_ENV_LOG) or "").strip()
    if log_path:
        d = os.path.dirname(os.path.abspath(log_path))
        if d:
            os.makedirs(d, exist_ok=True)
        return os.open(
            log_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
    return os.open(os.devnull, os.O_WRONLY)


@contextmanager
def silence_c_stderr() -> Iterator[None]:
    """Redirect fd 2 for the duration of the block (reentrant)."""
    with _Silence.lock:
        _Silence.depth += 1
        if _Silence.depth == 1:
            _Silence.saved_fd = os.dup(2)
            tfd = _open_target_fd()
            os.dup2(tfd, 2)
            os.close(tfd)
    try:
        yield
    finally:
        with _Silence.lock:
            _Silence.depth -= 1
            if _Silence.depth == 0:
                saved = _Silence.saved_fd
                assert saved is not None
                os.dup2(saved, 2)
                os.close(saved)
                _Silence.saved_fd = None


def librosa_load(path, **kwargs):
    """Like :func:`librosa.load` but silencing native decode stderr."""
    with silence_c_stderr():
        return _librosa.load(path, **kwargs)


def librosa_get_duration(**kwargs):
    """Like :func:`librosa.get_duration` but silencing native decode stderr."""
    with silence_c_stderr():
        return _librosa.get_duration(**kwargs)
