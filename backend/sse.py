from __future__ import annotations

from collections.abc import Iterator

from flask import Response


def sse_response(events: Iterator[str]) -> Response:
    """Return a Flask SSE response with shared headers."""
    return Response(
        events,
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
