"""Folder swatch colours for the server-rendered folder tree.

Mirrors ``frontend/src/lib/colors.ts`` (``dirColor``) exactly: the sidebar partial
paints swatches server-side while the canvas paints the same tracks client-side, and
they must agree. Both test suites assert the same golden vectors in
``frontend/src/lib/dir-color-vectors.json``.

Roots and their top-level folders keep the original path-hash hue (one vivid colour
per playlist); deeper folders are tints/shades of their top-level ancestor's hue with
the deviation halving per level, so a subtree reads as one colour family that tightens
as it deepens.
"""

from __future__ import annotations

import math

_HASH_MASK = 0x3FFFF


def _hash_path(s: str, mult: int, seed: int) -> int:
    """Original 31-rolling path hash; *mult*/*seed* salt the offset channels."""
    h = seed & _HASH_MASK
    for ch in s:
        h = (h * mult + ord(ch)) & _HASH_MASK
    return h


def _signed_unit(h: int) -> float:
    """Deterministic value in [-1, 1] from a hash."""
    return (h % 2001) / 1000 - 1


def dir_color(folder_path: str) -> str:
    if not folder_path:
        return "hsl(350,60%,55%)"
    segments = folder_path.split("/")
    depth = len(segments) - 1
    base = _hash_path(folder_path, 31, 0)
    if depth <= 1:
        return f"hsl({base % 360},65%,60%)"

    hue = _hash_path(f"{segments[0]}/{segments[1]}", 31, 0) % 360
    # Depth 2 is the widest tint; each level past it halves the deviation. The
    # amplitude is deliberately wide (lightness ±30, saturation ±18 at depth 2) so
    # sibling subgenres stay separable on the map.
    f = 0.5 ** (depth - 2)
    dl = _signed_unit(_hash_path(folder_path, 7, 11)) * 30 * f
    ds = _signed_unit(_hash_path(folder_path, 13, 23)) * 18 * f
    light = math.floor(60 + dl + 0.5)
    sat = math.floor(65 + ds + 0.5)
    return f"hsl({hue},{sat}%,{light}%)"
