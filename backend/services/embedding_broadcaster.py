from __future__ import annotations

import queue
import threading
from typing import Any


class EmbeddingBroadcaster:
    """Pub/sub helper for embedding SSE events."""

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def subscribe(self, *, maxsize: int = 2000) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            dead: list[queue.Queue[dict[str, Any]]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)
