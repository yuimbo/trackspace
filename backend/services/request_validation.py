"""Shared request validation helpers for the HTTP API.

Routes are thin: they parse a JSON body (or query args), validate it here, and
hand plain Python values to the domain layer. :class:`ApiValidationError` is mapped
to a structured JSON 400 by the app-level error handler in ``factory.py``, so a
missing field or a wrong type never surfaces as an HTML error page or an uncaught
``KeyError`` 500.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from flask import request

_MISSING = object()


class ApiValidationError(ValueError):
    """A request payload or query parameter failed validation.

    The message is safe to return to the client.
    """


def parse_json_body(req: Any = None, *, required: bool = True) -> dict[str, Any]:
    """Return the request's JSON body as a dict.

    An empty body is allowed when ``required=False`` (the caller applies its
    defaults); a present-but-malformed body is always rejected. ``force=True``
    keeps the previous behaviour of accepting JSON regardless of the declared
    content type.
    """
    req = req if req is not None else request
    raw = req.get_data(cache=True, as_text=True)
    if not raw or not raw.strip():
        if required:
            raise ApiValidationError("expected a JSON object body")
        return {}
    data = req.get_json(silent=True, force=True, cache=True)
    if not isinstance(data, dict):
        raise ApiValidationError("expected a JSON object body")
    return data


def _require(data: Mapping[str, Any], key: str) -> Any:
    if not isinstance(data, Mapping):
        raise ApiValidationError("expected a JSON object body")
    if key not in data or data[key] is None:
        raise ApiValidationError(f"missing required field {key!r}")
    return data[key]


def require_str(
    data: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
    max_len: int | None = None,
) -> str:
    value = _require(data, key)
    if not isinstance(value, str):
        raise ApiValidationError(f"field {key!r} must be a string")
    if not allow_empty and not value.strip():
        raise ApiValidationError(f"field {key!r} must not be empty")
    if max_len is not None and len(value) > max_len:
        raise ApiValidationError(f"field {key!r} is too long (max {max_len})")
    return value


def opt_str(
    data: Mapping[str, Any],
    key: str,
    default: str = "",
    *,
    max_len: int | None = None,
) -> str:
    value = data.get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    if not isinstance(value, str):
        raise ApiValidationError(f"field {key!r} must be a string")
    if max_len is not None and len(value) > max_len:
        raise ApiValidationError(f"field {key!r} is too long (max {max_len})")
    return value


def _as_str_list(value: Any, key: str, *, max_items: int | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ApiValidationError(f"field {key!r} must be a list of strings")
    if max_items is not None and len(value) > max_items:
        raise ApiValidationError(
            f"field {key!r} has too many items (max {max_items})"
        )
    for item in value:
        if not isinstance(item, str):
            raise ApiValidationError(f"field {key!r} must be a list of strings")
    return list(value)


def require_str_list(
    data: Mapping[str, Any], key: str, *, max_items: int | None = None
) -> list[str]:
    return _as_str_list(_require(data, key), key, max_items=max_items)


def opt_str_list(
    data: Mapping[str, Any],
    key: str,
    default: Iterable[str] = (),
    *,
    max_items: int | None = None,
) -> list[str]:
    value = data.get(key, _MISSING)
    if value is _MISSING or value is None:
        return list(default)
    return _as_str_list(value, key, max_items=max_items)


def require_mapping_list(
    data: Mapping[str, Any], key: str, *, max_items: int | None = None
) -> list[dict[str, Any]]:
    value = _require(data, key)
    if not isinstance(value, (list, tuple)):
        raise ApiValidationError(f"field {key!r} must be a list of objects")
    if max_items is not None and len(value) > max_items:
        raise ApiValidationError(
            f"field {key!r} has too many items (max {max_items})"
        )
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ApiValidationError(f"field {key!r} must be a list of objects")
        items.append(dict(item))
    return items


def opt_bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ApiValidationError(f"field {key!r} must be a boolean")


def opt_int(
    data: Mapping[str, Any],
    key: str,
    default: int | None = None,
    *,
    lo: int | None = None,
    hi: int | None = None,
) -> int | None:
    value = data.get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        raise ApiValidationError(f"field {key!r} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            raise ApiValidationError(f"field {key!r} must be an integer") from None
    else:
        raise ApiValidationError(f"field {key!r} must be an integer")
    return _bounded(parsed, key, lo=lo, hi=hi)


def opt_float(
    data: Mapping[str, Any],
    key: str,
    default: float | None = None,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float | None:
    value = data.get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        raise ApiValidationError(f"field {key!r} must be a number")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            raise ApiValidationError(f"field {key!r} must be a number") from None
    else:
        raise ApiValidationError(f"field {key!r} must be a number")
    return _bounded(parsed, key, lo=lo, hi=hi)


def _bounded(value: Any, key: str, *, lo: Any, hi: Any) -> Any:
    if lo is not None and value < lo:
        raise ApiValidationError(f"field {key!r} must be >= {lo}")
    if hi is not None and value > hi:
        raise ApiValidationError(f"field {key!r} must be <= {hi}")
    return value


def one_of(value: Any, allowed: Iterable[Any], *, field: str) -> Any:
    allowed = tuple(allowed)
    if value not in allowed:
        options = ", ".join(map(str, allowed))
        raise ApiValidationError(
            f"field {field!r} must be one of: {options} (got {value!r})"
        )
    return value
