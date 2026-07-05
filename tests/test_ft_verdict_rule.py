"""DEC-P4-8 verdict-rule tests (P4 task 11; spec §4 ``test_ft_verdict_rule``).

Crafted metric pairs exercise every clause, including the spec-named "2-of-3 with a
regression elsewhere" edge and the margin clause that makes noise-only keeps impossible.
Committed BEFORE the GPU window: the rule text cannot move once numbers exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.finetune.verdict import (
    MARGIN,
    ColumnMetrics,
    decide,
    extract_column,
    render_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _col(**over: object) -> ColumnMetrics:
    defaults: dict[str, object] = {
        "run_id": "run-x",
        "model": "m",
        "prompt_hash": "h",
        "gs07_intent_accuracy": 0.60,
        "gs07_slot_f1": 0.60,
        "gs08_coherence": 0.60,
        "gs02_inventions": 0,
        "schema_validity": 1.0,
        "tool_call_sequence_accuracy": 0.80,
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
        "latency_p50_ms": 400.0,
        "latency_p95_ms": 900.0,
        "tokens_per_sec": 60.0,
        "gs07_n": 5,
        "coherence_n": 3,
    }
    defaults.update(over)
    return ColumnMetrics(**defaults)  # type: ignore[arg-type]


BASE = _col()


def test_clean_two_of_three_win_with_margin_keeps() -> None:
    qlora = _col(gs07_intent_accuracy=0.80, gs07_slot_f1=0.66, gs08_coherence=0.60)
    verdict = decide(BASE, qlora)
    assert verdict.keep
    assert set(verdict.improved) == {"gs07_intent_accuracy", "gs07_slot_f1"}
    assert verdict.margin_met and not verdict.regressed and not verdict.guard_failures


def test_three_tiny_wins_below_margin_cut() -> None:
    """Clause (ii): 3-of-3 improved but nothing clears +0.05 — noise cannot keep."""
    eps = MARGIN / 2
    qlora = _col(
        gs07_intent_accuracy=0.60 + eps,
        gs07_slot_f1=0.60 + eps,
        gs08_coherence=0.60 + eps,
    )
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert len(verdict.improved) == 3 and not verdict.margin_met


def test_two_wins_with_regression_elsewhere_cut() -> None:
    """The spec-named edge: 2-of-3 with margin, but the third regresses -> CUT."""
    qlora = _col(gs07_intent_accuracy=0.80, gs07_slot_f1=0.75, gs08_coherence=0.55)
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert verdict.regressed == ("gs08_coherence",)


def test_single_win_cut() -> None:
    qlora = _col(gs07_intent_accuracy=0.90)
    verdict = decide(BASE, qlora)
    assert not verdict.keep and verdict.improved == ("gs07_intent_accuracy",)


def test_ties_do_not_count_as_improvement() -> None:
    verdict = decide(BASE, _col())
    assert not verdict.keep and verdict.improved == ()


def test_gs02_hard_gate_on_either_column() -> None:
    qlora = _col(gs07_intent_accuracy=0.90, gs07_slot_f1=0.90, gs02_inventions=1)
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert any("QLORA" in g for g in verdict.guard_failures)
    dirty_base = _col(gs02_inventions=2)
    verdict2 = decide(dirty_base, _col(gs07_intent_accuracy=0.90, gs07_slot_f1=0.90))
    assert not verdict2.keep
    assert any("BASE" in g for g in verdict2.guard_failures)


def test_schema_validity_guard() -> None:
    qlora = _col(gs07_intent_accuracy=0.90, gs07_slot_f1=0.90, schema_validity=0.95)
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert any("schema validity" in g for g in verdict.guard_failures)


def test_sequence_accuracy_guard() -> None:
    qlora = _col(gs07_intent_accuracy=0.90, gs07_slot_f1=0.90, tool_call_sequence_accuracy=0.70)
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert any("sequence accuracy" in g for g in verdict.guard_failures)


def test_missing_primary_metric_cannot_keep() -> None:
    qlora = _col(gs07_intent_accuracy=0.90, gs07_slot_f1=0.90, gs08_coherence=None)
    verdict = decide(BASE, qlora)
    assert not verdict.keep
    assert any("MISSING" in r for r in verdict.reasons)


def test_decide_is_pure() -> None:
    qlora = _col(gs07_intent_accuracy=0.80, gs07_slot_f1=0.70)
    assert decide(BASE, qlora) == decide(BASE, qlora)


def test_extract_column_from_committed_artifact() -> None:
    """The reducer runs over the real committed dry-run artifact byte-derived."""
    from sutradhar.evals.generation_run import GenerationRunArtifact

    runs = sorted(
        p
        for p in (_REPO_ROOT / "evals" / "generation_runs").glob("*.json")
        if ".trace" not in p.name
    )
    assert runs, "no committed generation-run artifact"
    artifact = GenerationRunArtifact.model_validate_json(runs[-1].read_text(encoding="utf-8"))
    col = extract_column(artifact)
    assert col.run_id == artifact.run_id
    assert col.gs07_n > 0
    assert col.gs02_inventions == artifact.metrics.gs02_inventions
    if col.gs07_intent_accuracy is not None:
        assert 0.0 <= col.gs07_intent_accuracy <= 1.0


def test_render_table_shape() -> None:
    qlora = _col(gs07_intent_accuracy=0.80, gs07_slot_f1=0.70)
    verdict = decide(BASE, qlora)
    table = render_table(BASE, qlora, verdict)
    assert "GS-07 intent accuracy *" in table
    assert "VERDICT: KEEP the adapter" in table
    cut = render_table(BASE, _col(), decide(BASE, _col()))
    assert "VERDICT: CUT the adapter" in cut
    assert "no significance theater" in cut


@pytest.mark.parametrize("margin_exact", [MARGIN, MARGIN + 1e-9])
def test_margin_boundary_inclusive(margin_exact: float) -> None:
    """Clause (ii) is >= +0.05: exactly +0.05 satisfies the margin."""
    qlora = _col(gs07_intent_accuracy=0.60 + margin_exact, gs07_slot_f1=0.61, gs08_coherence=0.60)
    verdict = decide(BASE, qlora)
    assert verdict.margin_met and verdict.keep
