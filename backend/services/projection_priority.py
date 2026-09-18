"""Give user-requested projections priority over the background layout prefetch.

The layout prefetch runs t-SNE fits for lenses the user is not looking at. A fit is
seconds long and holds the shared accelerator lock, so without a gate a lens switch could
wait behind several prefetched fits (the "never-ending computing layout" case).

This gate serialises projection fits and lets a waiting user request run before any further
prefetch starts. A prefetch already mid-fit is not interruptible, so a user request waits at
most one fit; the prefetch then yields until the user is done.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_cond = threading.Condition()
#: Requests waiting to run (user only; a prefetch waits without counting as a request).
_user_waiting = 0
_user_active = False
_prefetch_active = False


@contextmanager
def user_projection() -> Iterator[None]:
    """Hold the projection slot for a user-requested fit."""
    global _user_waiting, _user_active
    with _cond:
        _user_waiting += 1
        while _user_active or _prefetch_active:
            _cond.wait()
        _user_waiting -= 1
        _user_active = True
    try:
        yield
    finally:
        with _cond:
            _user_active = False
            _cond.notify_all()


@contextmanager
def prefetch_projection() -> Iterator[None]:
    """Hold the projection slot for a background fit, yielding to user requests."""
    global _prefetch_active
    with _cond:
        while _user_active or _user_waiting > 0 or _prefetch_active:
            _cond.wait()
        _prefetch_active = True
    try:
        yield
    finally:
        with _cond:
            _prefetch_active = False
            _cond.notify_all()


def user_projection_active() -> bool:
    """True while a user projection is running or waiting (for tests/telemetry)."""
    with _cond:
        return _user_active or _user_waiting > 0
