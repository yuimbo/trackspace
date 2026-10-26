"""Budget-distribution gallery around the shallow 50→250→1000 chain (final 50/0).

Covers the ``hierarchy-distribution`` set (``HIERARCHY_DISTRIBUTION_RECIPES``):
four variants derived from the reference ``tsne_chain50_250_1000_final50`` by
``dataclasses.replace`` that redistribute a fixed +900 normal-iteration surplus
across the chain's levels (``early1000``/``middle1000``/``late1000``/``spread400``).
Each variant differs from the reference only in ``tsne_sample_levels``
(id/label apart), keeps the final 50/0 geometry and the 100/0/0 early budgets,
and adds exactly 900 normal iterations. The reference's unchanged identity, the
set-wide identity uniqueness and the set's generation/route isolation are
covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le
from tests.conftest import recipe_field_diff

REFERENCE_ID = "tsne_chain50_250_1000_final50"


def _normal_sum(levels: tuple[tuple[int, int, int], ...]) -> int:
    return sum(n for _, n, _ in levels)


# ── recipe invariants ────────────────────────────────────────────────────────


def test_variants_differ_only_in_levels():
    ref = le.RECIPES_BY_ID[REFERENCE_ID]
    expected_levels = {
        "tsne_chain50_250_1000_early1000_final50": (
            (50, 1000, 100), (250, 100, 0), (1000, 100, 0)),
        "tsne_chain50_250_1000_middle1000_final50": (
            (50, 100, 100), (250, 1000, 0), (1000, 100, 0)),
        "tsne_chain50_250_1000_late1000_final50": (
            (50, 100, 100), (250, 100, 0), (1000, 1000, 0)),
        "tsne_chain50_250_1000_spread400_final50": (
            (50, 400, 100), (250, 400, 0), (1000, 400, 0)),
    }
    assert set(expected_levels) == {r.id for r in le.HIERARCHY_DISTRIBUTION_RECIPES}
    ref_normal = _normal_sum(ref.tsne_sample_levels)
    for r in le.HIERARCHY_DISTRIBUTION_RECIPES:
        # Only the id/label and the levels may move — no other field, and no
        # new field, may differ from the reference.
        assert recipe_field_diff(ref, r) == {"id", "label", "tsne_sample_levels"}
        assert r.tsne_sample_levels == expected_levels[r.id]
        # Final fit stays 50 normal / 0 early; early budgets stay 100/0/0.
        assert (r.tsne_n_iter, r.tsne_early_exaggeration_iter) == (50, 0)
        assert tuple(e for _, _, e in r.tsne_sample_levels) == (100, 0, 0)
        # Each variant adds exactly 900 normal iterations, spread over the levels.
        assert _normal_sum(r.tsne_sample_levels) == ref_normal + 900
