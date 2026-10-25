from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from backend.embeddings.projection_config import (
    DEFAULT_RECIPE_TSNE_EARLY_EXAGGERATION_ITER,
    DEFAULT_RECIPE_TSNE_N_ITER,
    FROZEN_FEATURE_MASK,
    FROZEN_FOLDER_BOOST,
    FROZEN_FOLDER_DEPTH_BOOST,
    FROZEN_SCALE_FOLDERS,
    FROZEN_SOURCE_WEIGHTS,
    FROZEN_SOURCES,
    PROJECTION_RECIPE_IDS,
    TSNE_EARLY_EXAGGERATION_ITER_HI,
    TSNE_EARLY_EXAGGERATION_ITER_LO,
    TSNE_N_ITER_HI,
    TSNE_N_ITER_LO,
    recipe_for,
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

#: Query param selecting a canonical recipe id: ``recipe=genre_full``.
_RECIPE_PARAM = "recipe"

#: Query param overriding the openTSNE full-fit iteration count.
_TSNE_N_ITER_PARAM = "tsne_n_iter"

#: Query param overriding the openTSNE early-exaggeration iteration count.
_TSNE_EARLY_PARAM = "tsne_early_exaggeration_iter"


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
    #: Canonical recipe id (``None``/``"default"`` = the frozen app defaults, ``"genre_full"``
    #: = the opt-in pure-genre lens). Part of ``layout_revision`` and the fit-cache key.
    recipe_id: str | None = None
    #: openTSNE full-fit iteration budget (effective values; default 100 for both
    #: recipes unless overridden — the gallery 500/250 is an explicit override). A
    #: deviation from the recipe default is part of ``layout_revision`` and the
    #: fit-cache key, so two iteration budgets never share a fitted embedding.
    tsne_n_iter: int = DEFAULT_RECIPE_TSNE_N_ITER
    tsne_early_exaggeration_iter: int = DEFAULT_RECIPE_TSNE_EARLY_EXAGGERATION_ITER


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


def _recipe_weight_override(
    raw: str, recipe, *, field: str
) -> dict[str, float]:
    """Complete a ``w=`` override for a canonical recipe into its full weight vector.

    Only the recipe's own canonical source keys may be overridden (``features`` is
    excluded from ``genre_full``), so an unknown source is a client-safe 400. The
    result is the recipe's *complete* weight vector with the override applied —
    the unspecified sources keep their recipe defaults (including the explicit
    zeros), so a request that happens to match the defaults hashes/caches
    identically to the bare recipe. An all-zero vector is rejected by
    :func:`weight_map`.
    """
    recipe_sources = tuple(recipe.sources)
    validated = weight_map(
        _parse_weight_spec(raw),
        allowed=recipe_sources,
        defaults={s: float(recipe.weights[s]) for s in recipe_sources},
        field=field,
    )
    # Restore the recipe's full vector (e.g. ``features: 0`` for genre_full) so
    # the completed override keeps the same key set as the canonical recipe.
    completed = dict(recipe.weights)
    completed.update(validated)
    return completed


def projection_params_for_lens(
    sources: tuple[str, ...],
    source_weights: dict[str, float] | None,
    *,
    recipe_id: str | None = None,
    tsne_n_iter: int | None = None,
    tsne_early_exaggeration_iter: int | None = None,
    default_n_iter: int = DEFAULT_RECIPE_TSNE_N_ITER,
    default_early: int = DEFAULT_RECIPE_TSNE_EARLY_EXAGGERATION_ITER,
) -> ProjectionParams:
    """Build the projection params for a lens.

    Shared by the HTTP route (after query parsing) and the layout-prefetch job, so a
    warmed layout is computed with exactly the parameters a later request would use.
    ``source_weights=None`` means the frozen defaults (the combined lens). An opt-in
    ``recipe_id`` and its openTSNE budget are threaded through; an explicit
    *tsne_n_iter* / *tsne_early_exaggeration_iter* overrides that default.
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
        recipe_id=recipe_id,
        tsne_n_iter=tsne_n_iter if tsne_n_iter is not None else default_n_iter,
        tsne_early_exaggeration_iter=(
            tsne_early_exaggeration_iter
            if tsne_early_exaggeration_iter is not None
            else default_early
        ),
    )


def projection_params_for_recipe(
    recipe_id: str | None,
    *,
    genre_full_enabled: bool = False,
    tsne_n_iter: int | None = None,
    tsne_early_exaggeration_iter: int | None = None,
    source_weights: dict[str, float] | None = None,
    for_status: bool = False,
) -> ProjectionParams:
    """Build the params for a canonical recipe id.

    Shared by the HTTP route and the layout-prefetch job so a warmed genre_full
    layout is computed with exactly the parameters a later request would use.
    ``genre_full_enabled`` gates the opt-in recipe: requesting it on a disabled
    server is a client-safe 400 (unless ``for_status``, the status capability
    bootstrap). Explicit *tsne_n_iter* / *tsne_early_exaggeration_iter*
    override the recipe's own openTSNE budget. *source_weights*, when given, is a
    validated override of the recipe's canonical weight vector (the request form
    ``recipe=genre_full + w``); it is used verbatim but the recipe still owns the
    identity/budget. The recipe's sources, weights and budget come from the
    canonical ``projection_config`` registry, not copies here.
    """
    rid = recipe_id or "default"
    if rid not in PROJECTION_RECIPE_IDS:
        raise ApiValidationError(f"unknown recipe {recipe_id!r}")
    if rid == "genre_full" and not genre_full_enabled and not for_status:
        raise ApiValidationError("genre_full recipe is not enabled on this server")

    recipe = recipe_for(rid)
    # ``default`` omits the weights (the client's frozen-defaults form); an opt-in
    # recipe carries its complete, explicit weight vector (the override replaces
    # the canonical vector but keeps every recipe-owned field).
    if rid == "default":
        weights = None
    elif source_weights is not None:
        weights = source_weights
    else:
        weights = dict(recipe.weights)
    return projection_params_for_lens(
        recipe.sources,
        weights,
        recipe_id=rid,
        tsne_n_iter=tsne_n_iter,
        tsne_early_exaggeration_iter=tsne_early_exaggeration_iter,
        default_n_iter=recipe.tsne_n_iter,
        default_early=recipe.tsne_early_exaggeration_iter,
    )


def parse_projection_params(
    req: Any,
    *,
    default_when_no_method: bool = False,
    genre_full_enabled: bool = False,
    for_status: bool = False,
) -> ProjectionParams:
    """Parse method (``pca`` preview vs ``tsne``), sources, optional weights, dims, recipe.

    All other fields are frozen. An unknown ``method`` is rejected with a structured
    400 rather than silently falling back to t-SNE, so client bugs surface instead of
    being masked. Weights, when present, are validated against the canonical sources
    and completed from the frozen defaults. ``dims`` is ``2`` unless ``3`` is asked
    for explicitly (an unknown value is a structured 400).

    ``recipe`` selects a canonical recipe: ``recipe=genre_full`` fixes the source
    set to the pure-genre lens and its 100/100 openTSNE budget (a disabled server
    rejects it with a 400). Its weights may be tuned with ``w=`` — only the
    recipe's canonical source keys are valid overrides, completed from the genre
    defaults, and the vector is still part of ``layout_revision`` so a re-weighting
    refits. ``sources=`` stays forbidden with a canonical recipe (the recipe fixes
    its source blocks). ``tsne_n_iter`` / ``tsne_early_exaggeration_iter``
    override the recipe's openTSNE budget within validated bounds.
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

    # The opt-in recipe id takes precedence over per-request sources/weights: a
    # canonical recipe is authoritative about its source set and budget.
    raw_recipe = (req.args.get(_RECIPE_PARAM) or "").strip().lower()
    if raw_recipe:
        one_of(raw_recipe, PROJECTION_RECIPE_IDS, field=_RECIPE_PARAM)
        recipe_id = raw_recipe
    else:
        recipe_id = None

    tsne_n_iter = _opt_tsne_int(
        req.args.get(_TSNE_N_ITER_PARAM),
        field=_TSNE_N_ITER_PARAM,
        lo=TSNE_N_ITER_LO, hi=TSNE_N_ITER_HI,
    )
    tsne_early = _opt_tsne_int(
        req.args.get(_TSNE_EARLY_PARAM),
        field=_TSNE_EARLY_PARAM,
        lo=TSNE_EARLY_EXAGGERATION_ITER_LO, hi=TSNE_EARLY_EXAGGERATION_ITER_HI,
    )

    if recipe_id and recipe_id != "default":
        # ``for_status`` (the ``/api/embeddings/status`` capability bootstrap)
        # tolerates a disabled genre_full so the client can still learn
        # ``genre_full_enabled`` (its absence in the payload) and reconcile away
        # from a genre map it cannot render. The projection route stays strict.
        if recipe_id == "genre_full" and not genre_full_enabled and not for_status:
            raise ApiValidationError("genre_full recipe is not enabled on this server")
        if req.args.get(_SOURCES_PARAM):
            raise ApiValidationError(
                f"recipe {recipe_id!r} fixes its source blocks; "
                f"do not also pass {_SOURCES_PARAM}"
            )
        # ``recipe=genre_full + w`` is the tunable genre weight preview: only the
        # recipe's canonical source keys may be overridden (features excluded),
        # completed from the genre defaults, and an all-zero vector is rejected.
        raw_weights = (req.args.get(_WEIGHTS_PARAM) or "").strip()
        source_weights = None
        if raw_weights:
            recipe = recipe_for(recipe_id)
            source_weights = _recipe_weight_override(
                raw_weights, recipe, field=_WEIGHTS_PARAM
            )
        params = projection_params_for_recipe(
            recipe_id,
            genre_full_enabled=genre_full_enabled,
            tsne_n_iter=tsne_n_iter,
            tsne_early_exaggeration_iter=tsne_early,
            source_weights=source_weights,
            for_status=for_status,
        )
    else:
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

        params = projection_params_for_lens(
            sources,
            source_weights,
            tsne_n_iter=tsne_n_iter,
            tsne_early_exaggeration_iter=tsne_early,
        )

    if dims != 2:
        params = replace(params, dims=dims)
    return params if method == params.method else replace(params, method=method)


def _opt_tsne_int(raw: str | None, *, field: str, lo: int, hi: int) -> int | None:
    """Parse an optional bounded openTSNE iteration count, or ``None`` when absent."""
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise ApiValidationError(f"field {field!r} must be an integer") from None
    if value < lo or value > hi:
        raise ApiValidationError(f"field {field!r} must be between {lo} and {hi}")
    return value


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
