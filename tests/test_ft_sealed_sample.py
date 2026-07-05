"""Tier-1 dataset-integrity gate over the COMMITTED sealed sample + card (P4 task 7).

CI never sees the full private dataset (DEC-P4-7) — it gates on the ~100-conversation
stratified sample and the card: schema-valid tool calls/results, zero invented titles,
contracts, decontamination evidence, teacher-stamp consistency, and card/hash shape.
Between GPU windows this is what keeps `sutradhar-ft-v1` honest in the repo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sutradhar.evals.driver import load_tool_schema
from sutradhar.finetune.dataset import DatasetCard, TrainingConversation, read_card, read_jsonl
from sutradhar.finetune.scaffold import BEHAVIOUR_SHARES
from sutradhar.finetune.validate import (
    validate_contracts,
    validate_grounding,
    validate_tool_calls,
    validate_tool_results,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _REPO_ROOT / "finetune" / "ft_v1_sample.jsonl"
_CARD = _REPO_ROOT / "finetune" / "dataset_card.json"
_SUMMARY = _REPO_ROOT / "finetune" / "teacher_run_summary.json"
_SNAPSHOT = _REPO_ROOT / "finetune" / "scaffold_snapshot.json"


@pytest.fixture(scope="module")
def sample() -> list[TrainingConversation]:
    return read_jsonl(_SAMPLE)


@pytest.fixture(scope="module")
def card() -> DatasetCard:
    return read_card(_CARD)


def test_sample_is_nonempty_and_stratified(sample: list[TrainingConversation]) -> None:
    assert len(sample) >= 90
    behaviours = {c.behaviour for c in sample}
    assert behaviours == set(BEHAVIOUR_SHARES), "sample must cover every behaviour class"


def test_sample_tool_calls_and_results_are_v0_valid(
    sample: list[TrainingConversation],
) -> None:
    schema = load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")
    assert validate_tool_calls(sample, schema) == []
    assert validate_tool_results(sample, schema) == []


def test_sample_has_zero_invented_titles(sample: list[TrainingConversation]) -> None:
    assert validate_grounding(sample) == []


def test_sample_contracts_hold(sample: list[TrainingConversation]) -> None:
    taxonomy = json.loads(
        (_REPO_ROOT / "evals" / "prompts" / "intent_taxonomy_v1.json").read_text()
    )
    assert validate_contracts(sample, set(taxonomy["intents"]), set(taxonomy["slot_keys"])) == []


def test_sample_teacher_stamp_matches_card(
    sample: list[TrainingConversation], card: DatasetCard
) -> None:
    assert card.teacher is not None
    for conv in sample:
        assert conv.teacher == card.teacher, f"{conv.conv_id}: stamp differs from the card"


def test_card_decontamination_is_clean(card: DatasetCard) -> None:
    decon = card.decontamination
    assert decon.violations == []
    assert decon.max_similarity_golden < decon.threshold
    assert decon.max_similarity_exemplars < decon.threshold
    assert decon.max_similarity_negatives < decon.threshold
    assert decon.threshold == 0.80


def test_card_shape_and_provenance(card: DatasetCard) -> None:
    assert card.dataset_id == "sutradhar-ft-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", card.sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", card.graph_snapshot)
    # The card's graph_snapshot pins the COMMITTED scaffold snapshot.
    import hashlib

    assert card.graph_snapshot == hashlib.sha256(_SNAPSHOT.read_bytes()).hexdigest()
    assert card.split["train"] + card.split["val"] == sum(
        n for langs in card.counts.values() for n in langs.values()
    )
    assert any("IMDb" in line for line in card.licenses)  # the private-first reason


def test_teacher_run_summary_below_escalation(card: DatasetCard) -> None:
    summary = json.loads(_SUMMARY.read_text(encoding="utf-8"))
    assert summary["escalation_triggered"] is False
    assert summary["rejection_rate"] <= 0.30  # DEC-P4-1 trigger never fired at seal time
    assert card.teacher is not None
    assert summary["stamp"] == card.teacher.model_dump()
