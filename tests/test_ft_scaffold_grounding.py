"""Tier-1 grounding tests for the scaffold generator (P4 task 4; P4_SPEC §4).

Everything runs from the COMMITTED ``finetune/scaffold_snapshot.json`` — no DB, no
models. Asserts the by-construction invariants actually hold on generated output:

- every tool call validates against frozen ``tool_schema.v0.json`` (DEC-P1-8 validator);
- every ``tool_result`` validates against the tool's v0 *result* subschema;
- every asserted title in every answer resolves to that conversation's own tool results
  (the SAME detector that gates GS-02 — ``detect_hallucinated_movies``);
- every entity id exists in the committed training-slice entity list (D3 disjointness);
- INTENT-preamble + bold-title contracts survive generation;
- out_of_catalog conversations abstain honestly (empty results, no asserted titles);
- generation is deterministic under a pinned seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from sutradhar.evals.driver import load_tool_schema, validate_emitted_call
from sutradhar.evals.generation import detect_hallucinated_movies, parse_intent_preamble
from sutradhar.finetune.dataset import TrainingConversation, conversations_to_jsonl
from sutradhar.finetune.scaffold import ScaffoldConfig, generate, refine_local
from sutradhar.finetune.snapshot import (
    load_scaffold_snapshot,
    result_subschema,
    title_perturbations,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENTITIES = _REPO_ROOT / "finetune" / "training_slice_entities.json"
_TAXONOMY = _REPO_ROOT / "evals" / "prompts" / "intent_taxonomy_v1.json"

_SEED = 20260703
_SIZE = 120


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")


@pytest.fixture(scope="module")
def conversations() -> list[TrainingConversation]:
    snapshot = load_scaffold_snapshot(_REPO_ROOT / "finetune" / "scaffold_snapshot.json")
    return generate(snapshot, ScaffoldConfig(seed=_SEED, size=_SIZE))


def _result_titles(conv: TrainingConversation) -> set[str]:
    """Grounding set: every title-bearing value in the conversation's tool results."""
    titles: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("title", "matched_title", "canonical_title") and isinstance(value, str):
                    titles.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for turn in conv.turns:
        if turn.tool_result is not None:
            walk(turn.tool_result)
    return titles


def _final_answers(conv: TrainingConversation) -> list[str]:
    """The final prose assistant answer of each user turn (no tool_calls, has content)."""
    return [
        t.content
        for t in conv.turns
        if t.role == "assistant" and t.content is not None and t.tool_calls is None
    ]


# --- Schema conformance ---


def test_every_tool_call_validates_against_v0(
    conversations: list[TrainingConversation], schema: dict[str, Any]
) -> None:
    checked = 0
    for conv in conversations:
        for turn in conv.turns:
            for call in turn.tool_calls or []:
                errors = validate_emitted_call(schema, call.tool, call.arguments)
                assert not errors, f"{conv.conv_id}: {call.tool} invalid: {errors}"
                checked += 1
    assert checked > _SIZE  # at least one call per conversation


def test_every_tool_result_validates_against_v0(
    conversations: list[TrainingConversation], schema: dict[str, Any]
) -> None:
    validators: dict[str, Draft202012Validator] = {}
    for conv in conversations:
        pending_tool: str | None = None
        for turn in conv.turns:
            if turn.tool_calls:
                pending_tool = turn.tool_calls[-1].tool
            if turn.tool_result is not None:
                assert pending_tool is not None, f"{conv.conv_id}: tool_result without a call"
                if pending_tool not in validators:
                    validators[pending_tool] = Draft202012Validator(
                        result_subschema(schema, pending_tool)
                    )
                errors = [e.message for e in validators[pending_tool].iter_errors(turn.tool_result)]
                assert not errors, f"{conv.conv_id}: {pending_tool} result invalid: {errors}"


# --- Grounding (the no-invented-films-by-construction claim) ---


def test_no_invented_titles_in_any_answer(conversations: list[TrainingConversation]) -> None:
    for conv in conversations:
        allowed = _result_titles(conv)
        for answer in _final_answers(conv):
            report = detect_hallucinated_movies(answer, allowed)
            assert report.invention_count == 0, (
                f"{conv.conv_id}: invented titles {report.inventions} (allowed: {sorted(allowed)})"
            )


def test_entity_ids_are_training_slice_only(conversations: list[TrainingConversation]) -> None:
    payload = json.loads(_ENTITIES.read_text(encoding="utf-8"))
    known = {w["work_id"] for w in payload["works"]}
    known.update(v["version_id"] for v in payload["versions"])
    for conv in conversations:
        unknown = set(conv.entity_ids) - known
        assert not unknown, f"{conv.conv_id}: entity ids not in the D3 fixture list: {unknown}"


def test_out_of_catalog_conversations_abstain_honestly(
    conversations: list[TrainingConversation],
) -> None:
    pure = [c for c in conversations if c.behaviour == "out_of_catalog"]
    assert pure, "quota should produce out_of_catalog conversations"
    for conv in pure:
        final = _final_answers(conv)[-1]
        assert "NO_MATCH" in final, f"{conv.conv_id}: abstention answer missing NO_MATCH"
        assert "**" not in final, f"{conv.conv_id}: abstention asserts a bold title"
        last_result = [t.tool_result for t in conv.turns if t.tool_result is not None][-1]
        assert last_result == {"results": [], "abstain": True}
        if len([t for t in conv.turns if t.role == "user"]) == 1:
            assert conv.entity_ids == [], f"{conv.conv_id}: pure abstention must ground nothing"


# --- Formatting contracts (DEC-P3-4 amendments) ---


def test_intent_preamble_contract(conversations: list[TrainingConversation]) -> None:
    taxonomy = json.loads(_TAXONOMY.read_text(encoding="utf-8"))
    intents = set(taxonomy["intents"])
    slot_keys = set(taxonomy["slot_keys"])
    for conv in conversations:
        answers = _final_answers(conv)
        assert len(answers) == len(conv.intent_labels), f"{conv.conv_id}: answers != user turns"
        for answer, intent in zip(answers, conv.intent_labels, strict=True):
            parsed = parse_intent_preamble(answer)
            assert parsed is not None, f"{conv.conv_id}: no parseable INTENT preamble"
            assert parsed.intent == intent, f"{conv.conv_id}: preamble/label mismatch"
            assert parsed.intent in intents
            assert set(parsed.slots) <= slot_keys, f"{conv.conv_id}: unknown slot keys"
        for turn in conv.turns:
            if turn.tool_calls is not None:
                assert turn.content is None, f"{conv.conv_id}: tool-calling turn carries prose"


def test_slot_labels_use_frozen_vocabulary(conversations: list[TrainingConversation]) -> None:
    slot_keys = set(json.loads(_TAXONOMY.read_text(encoding="utf-8"))["slot_keys"])
    for conv in conversations:
        for slots in conv.slot_labels:
            assert set(slots) <= slot_keys, f"{conv.conv_id}: slot labels outside vocabulary"


# --- Determinism ---


def test_generation_is_deterministic() -> None:
    snapshot = load_scaffold_snapshot(_REPO_ROOT / "finetune" / "scaffold_snapshot.json")
    a = generate(snapshot, ScaffoldConfig(seed=_SEED, size=40))
    b = generate(snapshot, ScaffoldConfig(seed=_SEED, size=40))
    assert conversations_to_jsonl(a) == conversations_to_jsonl(b)
    c = generate(snapshot, ScaffoldConfig(seed=_SEED + 1, size=40))
    assert conversations_to_jsonl(a) != conversations_to_jsonl(c)


def test_title_perturbations_are_pure() -> None:
    assert title_perturbations("Pokkiri") == title_perturbations("Pokkiri")
    assert "Pokiri" in title_perturbations("Pokkiri")
    assert all(p != "U Turn" for p in title_perturbations("U Turn"))


# --- refine_local mirrors the repository semantics on recorded rows ---


def test_refine_local_semantics() -> None:
    entries = [
        {
            "version_id": "v1",
            "title": "A",
            "language": "te",
            "year": 2006,
            "cast_lead": ["Ravi Teja"],
            "relationship": "is_original_of",
            "is_original": True,
            "sources": [],
            "confidence": "HIGH",
        },
        {
            "version_id": "v2",
            "title": "B",
            "language": "hi",
            "year": 2012,
            "cast_lead": ["Akshay Kumar"],
            "relationship": "is_remake_of",
            "is_original": False,
            "sources": [],
            "confidence": "HIGH",
        },
    ]
    assert [v["version_id"] for v in refine_local(entries, {"language": "hi"})["versions"]] == [
        "v2"
    ]
    assert [v["version_id"] for v in refine_local(entries, {"era": "newer"})["versions"]] == ["v2"]
    assert [v["version_id"] for v in refine_local(entries, {"actor": "ravi"})["versions"]] == ["v1"]
    assert refine_local(entries, {"language": "bn"})["versions"] == []
    projected = refine_local(entries, {"year": 2006})["versions"][0]
    assert set(projected) == {
        "version_id",
        "title",
        "language",
        "year",
        "relationship",
        "is_original",
    }
