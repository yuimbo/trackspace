"""Perceptual re-encodings of the cached 6-D audio-feature vector.

The cache stores ``[tempo_norm, key_cos, key_sin, mode, energy_norm,
danceability]`` (``librosa_audio_features``).  Two of those
dimensions are not perceptually uniform:

- ``tempo_norm`` is linear in BPM, so a 90 BPM track and its double-time 180 BPM
  sibling — which a DJ mixes freely — sit far apart.
- key is a circle-of-fifths unit circle with ``mode`` as a separate orthogonal bit,
  so a relative major/minor pair (C major / A minor: the same key signature) is as
  far apart as two unrelated keys.

This module re-encodes the same cached values into a space where perceptual
neighbours are Euclidean neighbours, without re-running any analysis:

- :func:`camelot_xy` — major keys on an outer ring at their circle-of-fifths
  angle, minor keys on an inner ring rotated 90° CCW (three fifths), so each
  relative pair shares a radius and fifth-neighbours stay angularly adjacent.  This
  is the Camelot-wheel topology.
- :func:`bpm_spiral_xy` — the tempo wraps once per octave: the angle is the
  fractional octave of the BPM and the radius grows with log-BPM, so half/double
  tempo pairs land on the same ray at different radii.

Both are pure functions of the cached vector, so a change of encoding only needs a
``FEATURE_ENCODING_VERSION`` bump to invalidate derived caches — never a
``FEATURES_VERSION`` bump (that would discard the raw rows and force a full
re-analysis).

The encoded block keeps the same six dimensions as the raw one, as three (x, y)
pairs plus the two still-linear descriptors::

    [camelot_x, camelot_y, spiral_x, spiral_y, energy_norm, danceability]

so downstream consumers (composite gather, PCA, t-SNE, the mask) are unchanged in
shape.
"""

from __future__ import annotations

import math

import numpy as np

from backend.tuning import BPM_HI, BPM_LO

#: Bump when the encoding below changes.  Derived (prepared-row / composite)
#: caches key on it; the raw ``features`` cache does not, because the raw vector is
#: untouched.
FEATURE_ENCODING_VERSION = 1

#: Encoded block width — the same six columns as the raw vector.
ENCODED_FEATURE_DIM = 6

#: Major keys sit on the outer ring, minor on the inner one.  The radius gap keeps a
#: relative pair (same ray) closer than two unrelated keys of the same mode.
CAMELOT_MAJOR_RADIUS = 0.82
CAMELOT_MINOR_RADIUS = 0.45

#: The minor ring is rotated three fifths (90°) so A minor lands on C major's ray.
_MINOR_FIFTHS_OFFSET = 3

#: Spiral radius bounds; the low end is lifted off the origin so a 60 BPM cluster
#: stays legible instead of collapsing to a point.
BPM_SPIRAL_R_MIN = 0.18
BPM_SPIRAL_R_MAX = 0.86

#: Octaves spanned by the BPM window (``BPM_LO`` .. ``BPM_HI``).
_BPM_OCTAVES = math.log2(BPM_HI / BPM_LO)


def camelot_xy(key_cos: float, key_sin: float, mode: float) -> tuple[float, float]:
    """Camelot-wheel position from the raw circle-of-fifths key encoding.

    ``mode >= 0.5`` is major (outer ring); below is minor (inner ring, rotated
    so relative pairs align radially).
    """
    angle = math.atan2(float(key_sin), float(key_cos))
    if float(mode) < 0.5:
        angle -= _MINOR_FIFTHS_OFFSET * (2.0 * math.pi / 12.0)
        r = CAMELOT_MINOR_RADIUS
    else:
        r = CAMELOT_MAJOR_RADIUS
    return r * math.cos(angle), r * math.sin(angle)


def bpm_spiral_xy(bpm: float) -> tuple[float, float]:
    """Spiral position for a tempo in BPM; doubles share a ray."""
    bpm = max(float(bpm), 1e-6)
    octave = math.log2(bpm / BPM_LO)
    angle = 2.0 * math.pi * (octave % 1.0)
    span = max(_BPM_OCTAVES, 1e-9)
    t = min(max(octave / span, 0.0), 1.0)
    r = BPM_SPIRAL_R_MIN + t * (BPM_SPIRAL_R_MAX - BPM_SPIRAL_R_MIN)
    return r * math.cos(angle), r * math.sin(angle)


def bpm_spiral_xy_norm(tempo_norm: float) -> tuple[float, float]:
    """Spiral position from the cached ``tempo_norm`` column (inverted to BPM)."""
    bpm = float(tempo_norm) * (BPM_HI - BPM_LO) + BPM_LO
    return bpm_spiral_xy(bpm)


def encode_feature_vector(vec: np.ndarray | None) -> np.ndarray | None:
    """Re-encode a raw 6-D cached feature row into the perceptual block.

    Returns ``None`` for a missing/undersized row so callers can treat it as
    absent rather than encode garbage.  The output is finite and same-shaped.
    """
    if vec is None:
        return None
    raw = np.asarray(vec, dtype=np.float32).reshape(-1)
    if raw.size < ENCODED_FEATURE_DIM:
        return None
    tempo_norm = float(raw[0])
    key_cos = float(raw[1])
    key_sin = float(raw[2])
    mode = float(raw[3])
    energy = float(raw[4])
    dance = float(raw[5])

    cx, cy = camelot_xy(key_cos, key_sin, mode)
    sx, sy = bpm_spiral_xy_norm(tempo_norm)
    out = np.array([cx, cy, sx, sy, energy, dance], dtype=np.float32)
    if not np.isfinite(out).all():
        return None
    return out
