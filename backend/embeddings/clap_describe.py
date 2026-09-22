"""Zero-shot text descriptions and semantic coordinates for a track in CLAP's space.

CLAP has no text decoder, so it cannot *generate* a caption token by token. Two
readings are produced instead, both from the same cached CLAP audio embedding:

* **Descriptions** — zero-shot retrieval: encode a flat bank of DJ/electronic phrases
  with the text tower and rank them by cosine to the track. The result is the bank
  phrase the model considers closest, not free text.
* **Coordinates** — project the track onto contrastive axes (``sparse <-> dense``, …).
  Each axis is a difference of the two pole text embeddings; the track's position is
  z-scored against the reference distribution captured below, so the readout is
  "where in this collection" (a percentile and a leaning pole), not an absolute
  cosine that would top out on whichever axis is loudest for the whole library.

The reference mean/std/reliability per axis were measured on the development library
(15 408 cached CLAP vectors). ``reliability`` is |Cohen's d| of the best folder-genre
contrast the axis separates; it weights the axis, because pole separation alone
misleads (``dark <-> bright`` has near-identical poles yet separates genres at
d = 2.58). Re-calibrate when the collection changes materially.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .linalg_utils import l2_normalize

log = logging.getLogger(__name__)

FACET_GENRE = "Genre"
FACET_MOOD = "Mood"
FACET_DENSITY = "Density"
FACET_MOTION = "Motion"
FACET_TEXTURE = "Texture"
FACET_LOW_END = "Low-end"
FACET_INSTRUMENT = "Instruments"
FACET_VOCAL = "Vocals"
FACET_USE = "Use"


@dataclass(frozen=True)
class Prompt:
    """One candidate phrase and the facet it belongs to."""

    facet: str
    text: str


#: The candidate phrases ranked against a track's audio embedding, grouped by facet.
#: Phrased as short captions, matching the audio-caption distribution CLAP's text
#: tower was trained on. Articles are written out in full (no "a acid").
FACET_BANK: dict[str, tuple[str, ...]] = {
    FACET_GENRE: (
        "a techno track",
        "a minimal techno track",
        "a deep techno track",
        "a dub techno track",
        "a hypnotic techno track",
        "an acid techno track",
        "a hardgroove techno track",
        "a melodic techno track",
        "a progressive house track",
        "a deep house track",
        "a tech house track",
        "a minimal house track",
        "a melodic house track",
        "an afro house track",
        "an organic house track",
        "a jazzy house track",
        "a loungy house track",
        "a trance track",
        "a psytrance track",
        "a progressive trance track",
        "a breakbeat track",
        "a drum and bass track",
        "a liquid drum and bass track",
        "a jungle track",
        "a dubstep track",
        "a garage track",
        "a UK garage track",
        "an ambient track",
        "a downtempo track",
        "an IDM track",
        "an electro track",
        "a breaks track",
        "a disco track",
        "a funk track",
        "a soul track",
        "a house track",
        "a dub track",
    ),
    FACET_MOOD: (
        "a hypnotic track",
        "a driving track",
        "a euphoric track",
        "a dark track",
        "an uplifting track",
        "a melancholic track",
        "a dreamy track",
        "a tense track",
        "a playful track",
        "a meditative track",
        "a stimulating track",
        "a brooding track",
        "an ecstatic track",
        "a sleazy track",
        "a serious track",
        "a sensual track",
        "a claustrophobic track",
        "a triumphant track",
        "a mysterious track",
        "a hopeful track",
        "a jazzy track",
        "a loungy track",
        "a laidback track",
        "a soulful track",
        "a warm track",
        "a cold track",
    ),
    FACET_DENSITY: (
        "a sparse track",
        "a dense track",
        "a stripped-back track",
        "a layered track",
        "a spacious track",
        "a wall-of-sound track",
        "a minimal track",
        "a busy track",
        "a track with lots of negative space",
        "an uncluttered track",
        "a maximal track",
        "a cavernous track",
    ),
    FACET_MOTION: (
        "a rolling track",
        "a four-on-the-floor track",
        "a breakbeat track",
        "a syncopated track",
        "a steady track",
        "a pulsing track",
        "a cyclical track",
        "an evolving track",
        "a restless track",
        "a swung track",
        "a stuttering track",
        "a trance-inducing track",
        "a repetitive track",
        "a shuffling track",
    ),
    FACET_TEXTURE: (
        "a dubby track",
        "a metallic track",
        "a granular track",
        "a warm track",
        "a cold track",
        "a wet track",
        "a dry track",
        "a soft-edged track",
        "an abrasive track",
        "an organic track",
        "a mechanical track",
        "a lo-fi track",
        "a polished track",
        "a hazy track",
        "a crisp track",
    ),
    FACET_LOW_END: (
        "a track with a sub-heavy low-end",
        "a track with a rolling low-end",
        "a track with a reese bass",
        "a track with a plucky bassline",
        "a track with a deep sub-bass",
        "a track with a punchy kick",
        "a track with a warm bass",
        "a track with a growling bass",
        "a track with a clean sub",
        "a track with a distorted bass",
        "a track with a prominent bassline",
        "a track with no bass",
    ),
    FACET_INSTRUMENT: (
        "a track with saxophone",
        "a track with piano",
        "a track with rhodes",
        "a track with strings",
        "a track with brass",
        "a track with a trumpet",
        "a track with flute",
        "a track with acoustic guitar",
        "a track with electric guitar",
        "a track with distorted guitars",
        "a track with a bass guitar",
        "a track with synth pads",
        "a track with arpeggiated synths",
        "a track with a synth lead",
        "a track with a drum machine",
        "a track with live drums",
        "a track with organ",
        "a track with a harp",
        "a track with a marimba",
        "a track with a bell melody",
    ),
    FACET_VOCAL: (
        "an instrumental track",
        "a track with a talking voice",
        "a track with spoken word",
        "a track with a narration",
        "a track with a vocal sample",
        "a track with a sung topline",
        "a track with female vocals",
        "a track with male vocals",
        "a track with chanted vocals",
        "a track with chopped vocals",
        "a track with wordless vocals",
        "a track with a choir",
    ),
    FACET_USE: (
        "a warm-up track",
        "a peak-time track",
        "a closing track",
        "an afterhours track",
        "a DJ tool",
        "an intro track",
        "a build-up track",
        "a track with a breakdown",
        "a track that drops hard",
        "an ambient interlude",
        "a set opener",
        "a set closer",
        "a lounge track",
        "a background track",
    ),
}

#: The flat phrase bank the Describe tab ranks, in facet order. A phrase that
#: belongs to two facets (``a breakbeat track`` is genre and motion) is kept once,
#: under its first facet, so the flat list never shows the same term twice.
def _flat_bank(facet_bank: dict[str, tuple[str, ...]]) -> tuple[Prompt, ...]:
    seen: set[str] = set()
    out: list[Prompt] = []
    for facet, texts in facet_bank.items():
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            out.append(Prompt(facet, text))
    return tuple(out)


PROMPT_BANK: tuple[Prompt, ...] = _flat_bank(FACET_BANK)


@dataclass(frozen=True)
class Axis:
    """One contrastive semantic axis, calibrated on the development library.

    ``pos`` / ``neg`` are the pole captions. ``mean`` / ``std`` are the reference
    distribution of the track projection onto the axis direction, so a track can be
    placed by percentile. ``reliability`` is |Cohen's d| of the best folder-genre
    contrast the axis separates, used to weight the axis.
    """

    pos: str
    neg: str
    mean: float
    std: float
    reliability: float

    @property
    def label(self) -> str:
        return f"{self.pos} <-> {self.neg}"


AXES: tuple[Axis, ...] = (
    Axis("sparse", "dense", mean=-0.1597, std=0.0747, reliability=0.720),
    Axis("hypnotic", "attention-grabbing", mean=0.0916, std=0.0540, reliability=0.605),
    Axis("repetitive", "evolving", mean=-0.0537, std=0.1024, reliability=2.594),
    Axis("dark", "bright", mean=-0.0327, std=0.0883, reliability=2.577),
    Axis("dry", "spacious", mean=-0.0513, std=0.1101, reliability=1.963),
    Axis("mechanical", "organic", mean=-0.1800, std=0.0747, reliability=0.898),
    Axis("restrained", "explosive", mean=-0.0788, std=0.0594, reliability=0.775),
    Axis("beatless", "beat-driven", mean=-0.2689, std=0.0944, reliability=1.800),
    Axis("meditative", "stimulating", mean=0.0285, std=0.0855, reliability=1.562),
    Axis("instrumental", "vocal-driven", mean=-0.0886, std=0.0807, reliability=0.639),
    Axis("melancholic", "euphoric", mean=-0.1309, std=0.1013, reliability=2.411),
    Axis("smooth", "punchy", mean=-0.0547, std=0.1267, reliability=2.074),
    Axis("static", "constantly changing", mean=0.2047, std=0.0971, reliability=3.397),
    Axis("minimal", "ornate", mean=0.1754, std=0.0967, reliability=1.599),
    Axis("warm", "cold", mean=-0.0598, std=0.0873, reliability=1.300),
    Axis("jazzy", "electronic", mean=-0.2730, std=0.0997, reliability=2.123),
    Axis("loungy", "energetic", mean=-0.0467, std=0.1394, reliability=2.420),
)


class PromptBank:
    """The phrase bank plus its lazily-built, process-lifetime text-embedding matrix.

    Encoding the bank is one batched text forward pass and the result is a property of
    the bank + loaded checkpoint, so it is built at most once. *embed_fn* is injected
    (``clap.generate_text_embeddings``) to keep the bank testable.
    """

    def __init__(self, prompts: Sequence[Prompt] = PROMPT_BANK) -> None:
        self.prompts: tuple[Prompt, ...] = tuple(prompts)
        self._lock = threading.Lock()
        self._matrix: np.ndarray | None = None

    def matrix(self, embed_fn: Callable[[list[str]], np.ndarray | None]) -> np.ndarray | None:
        """Unit-normalised text matrix ``(len(prompts), dim)``, or ``None`` if unavailable."""
        if self._matrix is not None:
            return self._matrix
        with self._lock:
            if self._matrix is not None:
                return self._matrix
            vecs = embed_fn([p.text for p in self.prompts])
            if vecs is None:
                return None
            mat = np.asarray(vecs, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(self.prompts):
                log.warning(
                    "CLAP prompt bank encoded %s rows for %d prompts; disabling describe",
                    mat.shape,
                    len(self.prompts),
                )
                return None
            self._matrix = l2_normalize(mat)
        return self._matrix


class AxisDirections:
    """Unit direction per axis, from the two pole text embeddings (cached)."""

    def __init__(self, axes: Sequence[Axis] = AXES) -> None:
        self.axes: tuple[Axis, ...] = tuple(axes)
        self._lock = threading.Lock()
        self._directions: np.ndarray | None = None

    def directions(self, embed_fn: Callable[[list[str]], np.ndarray | None]) -> np.ndarray | None:
        """Unit axis matrix ``(len(axes), dim)``, or ``None`` if unavailable."""
        if self._directions is not None:
            return self._directions
        with self._lock:
            if self._directions is not None:
                return self._directions
            poles = [f"{a.pos} music" for a in self.axes] + [
                f"{a.neg} music" for a in self.axes
            ]
            vecs = embed_fn(poles)
            if vecs is None:
                return None
            mat = np.asarray(vecs, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != 2 * len(self.axes):
                return None
            pos, neg = mat[: len(self.axes)], mat[len(self.axes) :]
            diff = pos - neg
            norms = np.linalg.norm(diff, axis=-1, keepdims=True)
            np.maximum(norms, 1e-9, out=norms)
            self._directions = diff / norms
        return self._directions


def rank_descriptions(
    audio_vec: np.ndarray,
    matrix: np.ndarray,
    prompts: Sequence[Prompt] = PROMPT_BANK,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Rank *prompts* by cosine similarity to *audio_vec*, descending.

    Both sides are L2-normalised, so the dot product is the cosine. The sort is
    stable, so equal scores keep their bank order.
    """
    prompts = tuple(prompts)
    a = l2_normalize(np.asarray(audio_vec, dtype=np.float32).reshape(1, -1))[0]
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != len(prompts) or mat.shape[1] != a.size:
        return []
    sims = l2_normalize(mat) @ a
    order = np.argsort(-sims, kind="stable")
    out: list[dict[str, Any]] = []
    for i in order[: max(0, limit)]:
        idx = int(i)
        out.append(
            {
                "text": prompts[idx].text,
                "facet": prompts[idx].facet,
                "score": float(sims[idx]),
            }
        )
    return out


def coordinates(
    audio_vec: np.ndarray,
    directions: np.ndarray,
    axes: Sequence[Axis] = AXES,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Place *audio_vec* on each axis, strongest (``|z| * reliability``) first.

    The percentile is the reference-distribution CDF at the track's z-score, so the
    readout reads "higher than 97 % of the reference library", and *pole* is the side
    of the axis the track leans to. Axes are weighted by their genre reliability so a
    non-discriminative axis (beatless vs beat-driven for an all-electronic collection)
    does not crowd out the ones that carry information.
    """
    axes = tuple(axes)
    a = l2_normalize(np.asarray(audio_vec, dtype=np.float32).reshape(1, -1))[0]
    dirs = np.asarray(directions, dtype=np.float32)
    if dirs.ndim != 2 or dirs.shape[0] != len(axes) or dirs.shape[1] != a.size:
        return []
    proj = dirs @ a
    means = np.array([ax.mean for ax in axes], dtype=np.float32)
    stds = np.maximum(np.array([ax.std for ax in axes], dtype=np.float32), 1e-9)
    zs = (proj - means) / stds
    # Normal CDF, so the percentile is continuous and needs no library scan.
    pct = np.array(
        [0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))) for z in zs],
        dtype=np.float32,
    )
    weight = np.abs(zs) * np.array([ax.reliability for ax in axes], dtype=np.float32)
    order = np.argsort(-weight, kind="stable")
    out: list[dict[str, Any]] = []
    for i in order[: max(0, limit)]:
        idx = int(i)
        ax = axes[idx]
        out.append(
            {
                "axis": ax.label,
                "pos": ax.pos,
                "neg": ax.neg,
                "pole": ax.pos if pct[idx] >= 0.5 else ax.neg,
                "z": float(zs[idx]),
                "percentile": float(pct[idx]),
                "reliability": float(ax.reliability),
            }
        )
    return out
