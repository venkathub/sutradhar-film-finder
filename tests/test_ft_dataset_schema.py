"""Unit tests for the P4 training-dataset schema + card models (P4_SPEC §4 row 1).

Asserts: pydantic round-trip; ``extra="forbid"`` on every model; structural invariants
(roles vs tool fields, per-user-turn label counts); canonical-JSONL determinism; card
sha256 stability; and the behaviour-literal ↔ frozen-taxonomy sync guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from sutradhar.finetune.dataset import (
    BEHAVIOURS,
    DatasetCard,
    DecontReport,
    TeacherStamp,
    ToolCallRecord,
    TrainingConversation,
    TrainingMessage,
    canonical_json,
    dataset_sha256,
    read_card,
    read_jsonl,
    write_card,
    write_jsonl,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TAXONOMY = _REPO_ROOT / "evals" / "prompts" / "intent_taxonomy_v1.json"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _conversation(conv_id: str = "ft-0001") -> TrainingConversation:
    return TrainingConversation(
        conv_id=conv_id,
        behaviour="find_by_plot",
        query_lang="ta-latin",
        turns=[
            TrainingMessage(role="system", content="You are Sutradhar."),
            TrainingMessage(role="user", content="oru padam — police officer flashback story"),
            TrainingMessage(
                role="assistant",
                tool_calls=[
                    ToolCallRecord(
                        tool="search_by_plot",
                        arguments={"plot_description": "police officer flashback story"},
                    )
                ],
            ),
            TrainingMessage(
                role="tool",
                tool_result={"results": [{"work_id": "W1", "title": "Test Film"}]},
            ),
            TrainingMessage(
                role="assistant",
                content='INTENT: {"intent": "find_by_plot"}\nFound **Test Film** (2010).',
            ),
        ],
        entity_ids=["W1"],
        intent_labels=["find_by_plot"],
        slot_labels=[{"plot_description": "police officer flashback story"}],
        scaffold_hash="a" * 64,
        teacher=None,
    )


def _card(sha256: str = "b" * 64) -> DatasetCard:
    return DatasetCard(
        dataset_id="sutradhar-ft-v1",
        counts={"find_by_plot": {"ta-latin": 1}},
        graph_snapshot="c" * 64,
        teacher=TeacherStamp(
            model="sarvamai/sarvam-m", revision="deadbeef", prompt_sha256="d" * 64
        ),
        seed=42,
        decontamination=DecontReport(
            threshold=0.80,
            max_similarity_golden=0.41,
            max_similarity_exemplars=0.38,
            max_similarity_negatives=0.35,
        ),
        split={"train": 1, "val": 0, "split_seed": 7},
        licenses=["Wikidata: CC0", "TMDB: attribution", "IMDb: non-commercial (private-first)"],
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# Round-trip + extra="forbid"
# ---------------------------------------------------------------------------


def test_conversation_round_trip() -> None:
    conv = _conversation()
    restored = TrainingConversation.model_validate_json(canonical_json(conv))
    assert restored == conv


def test_card_round_trip(tmp_path: Path) -> None:
    card = _card()
    path = tmp_path / "card.json"
    write_card(path, card)
    assert read_card(path) == card


@pytest.mark.parametrize(
    ("model_cls", "payload"),
    [
        (ToolCallRecord, {"tool": "get_work", "arguments": {}, "surprise": 1}),
        (TrainingMessage, {"role": "user", "content": "hi", "surprise": 1}),
        (TeacherStamp, {"model": "m", "revision": "r", "prompt_sha256": "h", "surprise": 1}),
        (
            DecontReport,
            {
                "threshold": 0.8,
                "max_similarity_golden": 0.1,
                "max_similarity_exemplars": 0.1,
                "max_similarity_negatives": 0.1,
                "surprise": 1,
            },
        ),
    ],
)
def test_extra_fields_forbidden(model_cls: type[BaseModel], payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="surprise"):
        model_cls.model_validate(payload)


def test_extra_fields_forbidden_on_conversation_and_card() -> None:
    conv_payload = json.loads(canonical_json(_conversation()))
    conv_payload["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        TrainingConversation.model_validate(conv_payload)
    card_payload = json.loads(canonical_json(_card()))
    card_payload["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        DatasetCard.model_validate(card_payload)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_tool_calls_rejected_on_non_assistant_turns() -> None:
    call = ToolCallRecord(tool="get_work", arguments={"work_id": "W1"})
    with pytest.raises(ValidationError, match="tool_calls only valid on assistant"):
        TrainingMessage(role="user", content="hi", tool_calls=[call])


def test_empty_tool_calls_list_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        TrainingMessage(role="assistant", tool_calls=[])


def test_tool_result_rejected_on_non_tool_turns() -> None:
    with pytest.raises(ValidationError, match="tool_result only valid on tool"):
        TrainingMessage(role="assistant", content="x", tool_result={"results": []})


def test_tool_turn_requires_tool_result() -> None:
    with pytest.raises(ValidationError, match="must carry a tool_result"):
        TrainingMessage(role="tool", content="orphan")


def test_fully_empty_message_rejected() -> None:
    with pytest.raises(ValidationError, match="empty message"):
        TrainingMessage(role="assistant")


def test_conversation_requires_turns_and_a_user_turn() -> None:
    conv = _conversation()
    with pytest.raises(ValidationError, match="at least one turn"):
        TrainingConversation.model_validate({**conv.model_dump(), "turns": []})
    only_system = [TrainingMessage(role="system", content="sys").model_dump()]
    with pytest.raises(ValidationError, match="at least one user turn"):
        TrainingConversation.model_validate(
            {**conv.model_dump(), "turns": only_system, "intent_labels": [], "slot_labels": []}
        )


@pytest.mark.parametrize("field", ["intent_labels", "slot_labels"])
def test_labels_must_match_user_turn_count(field: str) -> None:
    conv = _conversation()
    bad = conv.model_dump()
    bad[field] = []  # one user turn in the builder => one label required
    with pytest.raises(ValidationError, match=f"{field} must have one entry per user turn"):
        TrainingConversation.model_validate(bad)


def test_unknown_behaviour_rejected() -> None:
    conv = _conversation()
    with pytest.raises(ValidationError):
        TrainingConversation.model_validate({**conv.model_dump(), "behaviour": "chitchat"})


# ---------------------------------------------------------------------------
# Taxonomy sync guard
# ---------------------------------------------------------------------------


def test_behaviours_match_frozen_intent_taxonomy() -> None:
    """The Behaviour literal must equal the frozen taxonomy's intents — no silent drift."""
    taxonomy = json.loads(_TAXONOMY.read_text(encoding="utf-8"))
    assert set(BEHAVIOURS) == set(taxonomy["intents"])
    assert len(BEHAVIOURS) == len(taxonomy["intents"])


# ---------------------------------------------------------------------------
# JSONL determinism + hashing
# ---------------------------------------------------------------------------


def test_jsonl_round_trip_and_determinism(tmp_path: Path) -> None:
    convs = [_conversation("ft-0001"), _conversation("ft-0002")]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    sha_a = write_jsonl(path_a, convs)
    sha_b = write_jsonl(path_b, convs)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert sha_a == sha_b == dataset_sha256(path_a)
    assert read_jsonl(path_a) == convs


def test_canonical_json_stable_and_utf8_verbatim() -> None:
    """Same data => same bytes, regardless of input key order; no \\u escaping of Indic text."""
    conv = _conversation()
    reordered = json.loads(json.dumps(conv.model_dump(mode="json")))
    # Rebuild from a dict with shuffled key insertion order.
    shuffled = {k: reordered[k] for k in sorted(reordered, reverse=True)}
    assert canonical_json(TrainingConversation.model_validate(shuffled)) == canonical_json(conv)
    native = _conversation("ft-native")
    native.turns[1].content = "दृश्यम जैसी फिल्म"
    assert "दृश्यम" in canonical_json(native)  # ensure_ascii=False — hash covers real bytes


def test_read_jsonl_names_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        canonical_json(_conversation()) + "\n" + '{"conv_id": "broken"}' + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        read_jsonl(path)


def test_card_write_is_deterministic(tmp_path: Path) -> None:
    card = _card()
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    write_card(path_a, card)
    write_card(path_b, card)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert dataset_sha256(path_a) == dataset_sha256(path_b)
