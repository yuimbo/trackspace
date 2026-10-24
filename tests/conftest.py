"""Shared test helpers."""


class CountingConn:
    """Delegating proxy over a ``ClosingConnection`` that counts ``execute``.

    Installed by monkeypatching ``cache._conn`` / ``queue._conn`` so tests can
    assert that a memo hit skipped the SQL round-trip. ``__enter__`` returns the
    proxy so the ``with self._conn() as conn: conn.execute(...)`` idiom is
    counted; ``__exit__`` keeps the ClosingConnection's commit-then-close
    semantics.
    """

    def __init__(self, inner, counter: list[int]) -> None:
        self._inner = inner  # ClosingConnection
        self._counter = counter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def execute(self, sql, params=()):
        self._counter[0] += 1
        return self._inner._conn.execute(sql, params)

    def __getattr__(self, name):
        # Delegate to the raw connection, like ClosingConnection.__enter__ does.
        return getattr(self._inner._conn, name)


def install_counting_conn(obj, counter: list[int]) -> None:
    """Route ``obj._conn`` through a :class:`CountingConn` counting into *counter*."""

    real = obj._conn

    def wrapped():
        return CountingConn(real(), counter)

    obj._conn = wrapped
