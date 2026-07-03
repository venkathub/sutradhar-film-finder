"""Validator-layer tests (P4 task 5): the gate re-earns what task 4 held by construction.

Covers the spec §4 rows ``test_ft_tool_calls_validate``, ``test_ft_decontamination``, and
``test_ft_teacher_lock`` (the lock verifier lives in ``validate``; task 6's teacher client
consumes it). Every negative case is a crafted corruption — the validators must catch it
with a logged reason, not silently pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.finetune.dataset import ToolCallRecord, TrainingConversation
from sutradhar.finetune.scaffold import ScaffoldConfig, generate
from sutradhar.finetune.snapshot import load_scaffold_snapshot
from sutradhar.finetune.validate import (
    compute_decontamination,
    lock_entities,
    protected_surfaces,
    unlock_entities,
    validate_dataset,
    validate_grounding,
    validate_quotas,
    validate_tool_calls,
    validate_tool_results,
    verify_locked,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def conversations() -> list[TrainingConversation]:
    snapshot = load_scaffold_snapshot(_REPO_ROOT / "finetune" / "scaffold_snapshot.json")
    return generate(snapshot, ScaffoldConfig(seed=5, size=200))


@pytest.fixture(scope="module")
def surfaces() -> dict[str, list[str]]:
    return protected_surfaces(
        golden_dir=_REPO_ROOT / "evals" / "golden",
        negatives_path=_REPO_ROOT / "evals" / "negatives" / "heldout.yaml",
        exemplars_path=_REPO_ROOT / "evals" / "prompts" / "exemplars_v1.md",
    )


# --- The full aggregate gate on a clean scaffold dataset ---


def test_clean_scaffold_dataset_passes_every_layer(
    conversations: list[TrainingConversation], surfaces: dict[str, list[str]]
) -> None:
    report = validate_dataset(
        conversations,
        surfaces=surfaces,
        entities_path=_REPO_ROOT / "finetune" / "training_slice_entities.json",
    )
    assert report.issues == []
    assert report.decontamination.violations == []
    assert report.ok


# --- test_ft_tool_calls_validate: corrupted calls/results are caught ---


def _clone(conv: TrainingConversation) -> TrainingConversation:
    return TrainingConversation.model_validate(conv.model_dump())


def test_hallucinated_tool_name_rejected(conversations: list[TrainingConversation]) -> None:
    from sutradhar.evals.driver import load_tool_schema

    schema = load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")
    bad = _clone(conversations[0])
    call_turn = next(t for t in bad.turns if t.tool_calls)
    call_turn.tool_calls = [ToolCallRecord(tool="get_movie_facts", arguments={})]
    issues = validate_tool_calls([bad], schema)
    assert issues and "hallucinated tool" in issues[0].detail


def test_hallucinated_parameter_rejected(conversations: list[TrainingConversation]) -> None:
    from sutradhar.evals.driver import load_tool_schema

    schema = load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")
    bad = _clone(conversations[0])
    call_turn = next(t for t in bad.turns if t.tool_calls)
    assert call_turn.tool_calls is not None
    call = call_turn.tool_calls[0]
    call_turn.tool_calls = [
        ToolCallRecord(tool=call.tool, arguments={**call.arguments, "vibes": "immaculate"})
    ]
    issues = validate_tool_calls([bad], schema)
    assert issues and "vibes" in issues[0].detail


def test_malformed_tool_result_rejected(conversations: list[TrainingConversation]) -> None:
    from sutradhar.evals.driver import load_tool_schema

    schema = load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")
    bad = _clone(next(c for c in conversations if c.behaviour == "find_by_title"))
    result_turn = next(t for t in bad.turns if t.tool_result is not None)
    result_turn.tool_result = {"candidates": "not-a-list", "ambiguous": False}
    issues = validate_tool_results([bad], schema)
    assert issues and issues[0].kind == "tool_result"


def test_invented_title_in_answer_rejected(conversations: list[TrainingConversation]) -> None:
    bad = _clone(next(c for c in conversations if c.behaviour == "find_by_title"))
    final = next(t for t in reversed(bad.turns) if t.role == "assistant" and t.content)
    final.content = (final.content or "") + (
        "\nAlso see **Maestro Reloaded** (2021, Hindi) — the spiritual sequel."
    )
    issues = validate_grounding([bad])
    assert issues and issues[0].kind == "invented_title"
    assert "Maestro Reloaded" in issues[0].detail


# --- test_ft_decontamination ---


def test_scaffolds_are_decontaminated(
    conversations: list[TrainingConversation], surfaces: dict[str, list[str]]
) -> None:
    report = compute_decontamination(conversations, surfaces=surfaces)
    assert report.violations == []
    assert report.max_similarity_golden < report.threshold
    assert report.max_similarity_exemplars < report.threshold
    assert report.max_similarity_negatives < report.threshold


def test_golden_query_leak_detected(
    conversations: list[TrainingConversation], surfaces: dict[str, list[str]]
) -> None:
    """Negative control: a training utterance that IS a golden query must be flagged."""
    leak = _clone(conversations[0])
    user_turn = next(t for t in leak.turns if t.role == "user")
    user_turn.content = surfaces["golden"][0]
    report = compute_decontamination([leak, *conversations[:5]], surfaces=surfaces)
    assert leak.conv_id in report.violations
    assert report.max_similarity_golden >= report.threshold


def test_negative_query_leak_detected(
    conversations: list[TrainingConversation], surfaces: dict[str, list[str]]
) -> None:
    """A decoy drifting onto a held-out negative query must be flagged too (Rev-2 bug class)."""
    leak = _clone(conversations[1])
    user_turn = next(t for t in leak.turns if t.role == "user")
    user_turn.content = surfaces["negatives"][0]
    report = compute_decontamination([leak], surfaces=surfaces)
    assert leak.conv_id in report.violations


# --- quotas ---


def test_quota_violation_detected(conversations: list[TrainingConversation]) -> None:
    skewed = [c for c in conversations if c.behaviour == "find_by_plot"]
    issues = validate_quotas(skewed)
    assert any(i.kind == "quota" for i in issues)
    assert validate_quotas(conversations) == []


# --- test_ft_teacher_lock: the placeholder-integrity verifier ---


def test_lock_and_unlock_round_trip() -> None:
    text = "Ye **Pokiri** hai! - **Pokiri** (2006, Telugu) - **Wanted** (2009, Hindi)"
    locked, mapping = lock_entities(text, ["Pokiri", "Wanted"])
    assert "Pokiri" not in locked and "Wanted" not in locked
    assert unlock_entities(locked, mapping) == text


def test_teacher_altering_locked_span_rejected() -> None:
    original = "Idhu ⟦T1⟧ padam! - **⟦T1⟧** (2006, Telugu)"
    mapping = {"⟦T1⟧": "Pokiri"}
    rewritten = "Idhu Pokkiri padam da! - **Pokkiri** (2006, Telugu)"  # sentinel replaced
    reasons = verify_locked(original, rewritten, mapping)
    assert any("locked span altered" in r for r in reasons)
    assert any("new bold title" in r for r in reasons)


def test_teacher_adding_title_rejected() -> None:
    original = "abstain answer with no titles. NO_MATCH."
    rewritten = "try **Salaar** instead! NO_MATCH."
    reasons = verify_locked(original, rewritten, {})
    assert any("new bold title" in r for r in reasons)


def test_teacher_dropping_preamble_rejected() -> None:
    original = 'INTENT: {"intent": "refine", "slots": {"language": "te"}}\n\n⟦T1⟧ only.'
    mapping = {"⟦T1⟧": "Okkadu"}
    rewritten = "⟦T1⟧ matrame unnadhi!"
    reasons = verify_locked(original, rewritten, mapping, require_preamble=True)
    assert any("preamble dropped" in r for r in reasons)
    altered = 'INTENT: {"intent": "list_versions"}\n\n⟦T1⟧ matrame!'
    reasons2 = verify_locked(original, altered, mapping, require_preamble=True)
    assert any("preamble altered" in r for r in reasons2)


def test_faithful_rewrite_accepted() -> None:
    original = (
        'INTENT: {"intent": "refine", "slots": {"language": "te"}}\n\n'
        "⟦T1⟧ version: **⟦T1⟧** (2003)."
    )
    mapping = {"⟦T1⟧": "Okkadu"}
    rewritten = (
        'INTENT: {"intent": "refine", "slots": {"language": "te"}}\n\n'
        "Telugu lo ⟦T1⟧ matrame unnadhi — **⟦T1⟧** (2003) chudandi!"
    )
    assert verify_locked(original, rewritten, mapping, require_preamble=True) == []
