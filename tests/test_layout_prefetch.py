"""The layout-prefetch job warms each projection lens behind the scenes.

The route and the job must agree on ``layout_revision`` (that is what makes the
warm entry the one the client later reads), so the parity test below pins the
params the frontend sends for a per-source lens.
"""

from __future__ import annotations

import os
from typing import cast

import pytest

from backend.embeddings.projection_config import (
    PROJECTION_LENS_IDS,
    PROJECTION_LENSES,
)
from backend.jobs import Job, JobQueue, KIND_LAYOUT
from backend.services import analysis_pipeline as ap
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_layouts,
    make_layout_handler,
)
from backend.services.projection_request import (
    parse_projection_params,
    projection_params_for_lens,
)

_NO_DEPS = cast(AnalysisDeps, None)


class _Req:
    def __init__(self, args: dict[str, str]):
        self.args = args


def _job(lens: str, job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind=KIND_LAYOUT,
        dedupe_key=lens,
        payload={"lens": lens, "reason": "test"},
        attempts=0,
        priority=300,
    )


@pytest.fixture()
def queue(tmp_path):
    return JobQueue(os.path.join(tmp_path, "jobs.db"))


def test_lens_ids_cover_the_projection_lenses():
    assert PROJECTION_LENS_IDS == tuple(l.id for l in PROJECTION_LENSES)
    # The frontend registry's projection lenses; keep in sync with lenses.ts.
    assert PROJECTION_LENS_IDS == (
        "combined",
        "maest",
        "clap",
        "rhythm",
        "supervised",
    )


def test_layout_handler_prefetches_each_job():
    seen: list[str] = []
    handler = make_layout_handler(lambda lens: seen.append(lens) or 12)
    out = handler([_job("combined"), _job("clap", 2)])
    assert seen == ["combined", "clap"]
    assert out[1].ok is True
    assert out[1].detail == {"lens": "combined", "n_tracks": 12}
    assert out[2].ok is True


def test_layout_handler_reports_a_failure_without_raising():
    def _boom(lens: str) -> int:
        raise RuntimeError("no vectors")

    handler = make_layout_handler(_boom)
    out = handler([_job("clap")])
    assert out[1].ok is False
    assert "RuntimeError" in (out[1].error or "")


def test_layout_handler_emits_phase_progress():
    phases: list[tuple] = []
    handler = make_layout_handler(
        lambda lens: 7, emit=lambda phase, done, total: phases.append((phase, done, total))
    )
    handler([_job("combined"), _job("clap", 2)])
    assert phases == [("layout combined", 1, 2), ("layout clap", 2, 2)]


def test_enqueue_layouts_is_per_lens_and_deduped(queue):
    assert enqueue_layouts(queue, PROJECTION_LENS_IDS) == 5
    # A second drain collapses onto the same dedupe keys.
    assert enqueue_layouts(queue, PROJECTION_LENS_IDS) == 0
    pending = queue.claim(KIND_LAYOUT, limit=10)
    assert {j.dedupe_key for j in pending} == set(PROJECTION_LENS_IDS)
    assert all(j.priority == 300 for j in pending)


def test_per_source_lens_params_match_the_client_query():
    """The prefetch must build the exact params a per-source request parses to.

    The client sends ``sources=clap&w=clap:1,<others>:0``; a mismatch here would
    warm a different revision and the switch would still wait.
    """
    lens = next(l for l in PROJECTION_LENSES if l.id == "clap")
    warm = projection_params_for_lens(lens.sources, lens.weights)
    client = parse_projection_params(
        _Req({"sources": "clap", "w": _weights_spec(lens.weights or {})})
    )
    assert warm.sources == client.sources
    assert warm.required_sources == client.required_sources
    assert warm.source_weights == client.source_weights


def test_combined_lens_params_match_the_default_query():
    lens = next(l for l in PROJECTION_LENSES if l.id == "combined")
    warm = projection_params_for_lens(lens.sources, lens.weights)
    client = parse_projection_params(_Req({}))
    assert warm.sources == client.sources
    assert warm.source_weights is None
    assert client.source_weights is None


def _weights_spec(weights: dict[str, float]) -> str:
    return ",".join(f"{k}:{v}" for k, v in weights.items())


def test_enqueue_layouts_returns_zero_for_no_lenses(queue):
    assert enqueue_layouts(queue, []) == 0
