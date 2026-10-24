from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from backend.embeddings.projection_config import (
    FROZEN_FEATURE_MASK,
    FROZEN_FOLDER_BOOST,
    FROZEN_FOLDER_DEPTH_BOOST,
    FROZEN_SCALE_FOLDERS,
    FROZEN_SOURCE_WEIGHTS,
    FROZEN_SOURCES,
    required_sources_for,
)
from backend.services.request_validation import ApiValidationError, one_of, weight_map
from backend.tuning import FROZEN_FEATURES_BLEND

#: Accepted projection methods. ``umap`` is the legacy alias for ``tsne``.
_METHODS: tuple[str, ...] = ("tsne", "pca", "umap")

#: Query param carrying client source weights: ``w=maest_logits:0.45,maest:0.25,…``.
_WEIGHTS_PARAM = "w"

#: Query param selecting the lens's source blocks: ``sources=clap``.
_SOURCES_PARAM = "sources"


@dataclass(frozen=True)
class ProjectionParams:
    method: str
    tag_names: list[str]
    explicit_folders: list[str]
    scale_folders: bool
    feature_mask: object  # np.ndarray
    features_blend: float
    folder_boost: float
    folder_depth_boost: float
    #: Source blocks this lens projects. A strict subset is a per-source lens;
    #: ``FROZEN_SOURCES`` is the combined lens.
    sources: tuple[str, ...]
    #: Sources a track must have to appear in this lens, derived from ``sources``.
    required_sources: tuple[str, ...]
    #: Client-supplied source weights (complete vector), or ``None`` for the frozen
    #: defaults. Part of ``layout_revision`` so a re-weighting refits the layout.
    source_weights: dict[str, float] | None = None
    #: Output dimensionality of the projection: 2 (default) or opt-in 3. Part of
    #: ``layout_revision`` and the fit-cache key, so 2-D and 3-D layouts never
    #: share cached positions or fitted embeddings.
    dims: int = 2


def embedding_projection_query_fingerprint(req: Any) -> str:
    """Stable hash of query args that affect coverage counts or layout_revision.

    Projection hyperparameters are server-fixed, so optional client args are ignored.
    """
    skip = frozenset({"folder", "recursive", "models_only"})
    parts: list[str] = []
    for key in sorted(req.args.keys()):
        if key in skip:
            continue
        parts.append(f"{key}={req.args.get(key, '')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _parse_weight_spec(raw: str) -> dict[str, float]:
    """Parse ``maest_logits:0.45,maest:0.25`` into a raw ``{source: float}`` map.

    Shape errors (missing ``:``, non-numeric value) raise a client-safe 400; key and
    range validation happens in :func:`weight_map`.
    """
    parsed: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition(":")
        name = name.strip()
        if not sep or not name:
            raise ApiValidationError(
                f"field {_WEIGHTS_PARAM!r} must be a comma-separated list of "
                "source:weight pairs"
            )
        try:
            parsed[name] = float(value.strip())
        except ValueError:
            raise ApiValidationError(
                f"weight for {name!r} must be a number"
            ) from None
    if not parsed:
        raise ApiValidationError(
            f"field {_WEIGHTS_PARAM!r} must list at least one source"
        )
    return parsed


def projection_params_for_lens(
    sources: tuple[str, ...],
    source_weights: dict[str, float] | None,
) -> ProjectionParams:
    """Build the frozen-default projection params for a lens.

    Shared by the HTTP route (after query parsing) and the layout-prefetch job, so a
    warmed layout is computed with exactly the parameters a later request would use.
    ``source_weights=None`` means the frozen defaults (the combined lens).
    """
    try:
        required_sources = required_sources_for(sources)
    except ValueError as e:
        raise ApiValidationError(str(e)) from None

    return ProjectionParams(
        method="tsne",
        tag_names=[],
        explicit_folders=[],
        scale_folders=FROZEN_SCALE_FOLDERS,
        feature_mask=FROZEN_FEATURE_MASK.copy(),
        features_blend=FROZEN_FEATURES_BLEND,
        folder_boost=FROZEN_FOLDER_BOOST,
        folder_depth_boost=FROZEN_FOLDER_DEPTH_BOOST,
        sources=sources,
        required_sources=required_sources,
        source_weights=source_weights,
    )


def parse_projection_params(
    req: Any,
    *,
    default_when_no_method: bool = False,
) -> ProjectionParams:
    """Parse method (``pca`` preview vs ``tsne``), sources, optional weights, dims.

    All other fields are frozen. An unknown ``method`` is rejected with a structured
    400 rather than silently falling back to t-SNE, so client bugs surface instead of
    being masked. Weights, when present, are validated against the canonical sources
    and completed from the frozen defaults. ``dims`` is ``2`` unless ``3`` is asked
    for explicitly (an unknown value is a structured 400).
    """
    del default_when_no_method
    raw_method = (req.args.get("method") or "").strip().lower()
    if not raw_method:
        method = "tsne"
    else:
        one_of(raw_method, _METHODS, field="method")
        method = "tsne" if raw_method == "umap" else raw_method

    raw_dims = (req.args.get("dims") or "2").strip()
    one_of(raw_dims, ("2", "3"), field="dims")
    dims = int(raw_dims)

    raw_sources = (req.args.get(_SOURCES_PARAM) or "").strip()
    sources = _parse_sources(raw_sources) if raw_sources else FROZEN_SOURCES

    raw_weights = (req.args.get(_WEIGHTS_PARAM) or "").strip()
    source_weights = (
        weight_map(
            _parse_weight_spec(raw_weights),
            allowed=FROZEN_SOURCE_WEIGHTS,
            defaults=FROZEN_SOURCE_WEIGHTS,
            field=_WEIGHTS_PARAM,
        )
        if raw_weights
        else None
    )

    params = projection_params_for_lens(sources, source_weights)
    if dims != 2:
        params = replace(params, dims=dims)
    return params if method == params.method else replace(params, method=method)


def _parse_sources(raw: str) -> tuple[str, ...]:
    """Parse ``sources=clap,rhythm`` into canonical frozen-source order.

    Unknown names and an empty list are client-safe 400s; a valid list is
    deduplicated and reordered to the canonical ``FROZEN_SOURCES`` order so two
    requests that name the same set share one layout revision.
    """
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ApiValidationError(
            f"field {_SOURCES_PARAM!r} must list at least one source"
        )
    for name in names:
        if name not in FROZEN_SOURCES:
            raise ApiValidationError(
                f"field {_SOURCES_PARAM!r} has unknown source {name!r} "
                f"(allowed: {', '.join(FROZEN_SOURCES)})"
            )
    named = set(names)
    return tuple(s for s in FROZEN_SOURCES if s in named)
