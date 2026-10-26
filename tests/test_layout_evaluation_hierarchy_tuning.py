"""Budget-tuning gallery around the shallow 50→250→1000 multilevel chain.

Covers the ``hierarchy-tuning`` set (``HIERARCHY_TUNING_RECIPES``): four
variants derived from the reference ``tsne_chain50_250_1000`` by
``dataclasses.replace`` that move the final fit and/or the coarse budget (the
``_coarse10_final50`` variant moves both). The reference recipe's unchanged
identity, the set-wide identity uniqueness and the set's generation/route
isolation are covered by ``test_layout_evaluation_hierarchy``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openTSNE")

import backend.embeddings.layout_evaluation as le
from tests.conftest import recipe_field_diff

REFERENCE_ID = "tsne_chain50_250_1000"

# ── recipe invariants ────────────────────────────────────────────────────────


def test_variants_differ_only_in_intended_fields():
    # Each variant must be identical to the reference except for its id/label
    # and the single axis it probes (final fit, or the coarse budget). No other
    # field, and no new field, may move.
    ref = le.RECIPES_BY_ID[REFERENCE_ID]
    expected: dict[str, set[str]] = {
        "tsne_chain50_250_1000_final50": {"id", "label", "tsne_n_iter"},
        "tsne_chain50_250_1000_coarse10": {"id", "label", "tsne_sample_levels"},
        "tsne_chain50_250_1000_coarse10_final50": {
            "id", "label", "tsne_sample_levels", "tsne_n_iter"},
        "tsne_chain50_250_1000_final2000": {"id", "label", "tsne_n_iter"},
    }
    assert set(expected) == {r.id for r in le.HIERARCHY_TUNING_RECIPES}
    for r in le.HIERARCHY_TUNING_RECIPES:
        assert recipe_field_diff(ref, r) == expected[r.id]

    assert le.RECIPES_BY_ID["tsne_chain50_250_1000_final50"].tsne_n_iter == 50
    assert le.RECIPES_BY_ID["tsne_chain50_250_1000_coarse10"].tsne_sample_levels \
        == ((50, 10, 100), (250, 100, 0), (1000, 100, 0))
    assert le.RECIPES_BY_ID["tsne_chain50_250_1000_coarse10_final50"].tsne_n_iter \
        == 50
    assert le.RECIPES_BY_ID["tsne_chain50_250_1000_final2000"].tsne_n_iter == 2000
    for rid in ("tsne_chain50_250_1000_final2000",
                "tsne_chain50_250_1000_final50",
                "tsne_chain50_250_1000_coarse10_final50"):
        assert le.RECIPES_BY_ID[rid].tsne_early_exaggeration_iter == 0

