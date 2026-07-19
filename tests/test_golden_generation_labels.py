"""Generation-slice golden labels: taxonomy conformance + turn alignment (P3 task 4).

The generation slice (GS-02 conversational, GS-07, GS-08) carries the labels the Table 2
intent/slot metrics score against. This test pins: the confirmed fixture counts (Q1:
GS-07 = 5, GS-08 = 3, GS-02-conversational = 4); every label drawn from the FROZEN intent
taxonomy / slot vocabulary (evals/prompts/intent_taxonomy_v1.json — golden.py itself stays
taxonomy-agnostic); per-turn list alignment for multi-turn fixtures; and schema strictness
(extra="forbid" survives the extension).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sutradhar.evals.golden import GoldenFixture, load_fixtures
from sutradhar.evals.prompts import PromptArtifacts, load_prompt_artifacts

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixtures() -> dict[str, GoldenFixture]:
    return {f.id: f for f in load_fixtures(_REPO / "evals" / "golden")}


@pytest.fixture(scope="module")
def artifacts() -> PromptArtifacts:
    return load_prompt_artifacts(_REPO / "evals" / "prompts")


def _generation_slice(fixtures: dict[str, GoldenFixture]) -> list[GoldenFixture]:
    return [
        f
        for f in fixtures.values()
        if f.id.startswith(("GS-07", "GS-08")) or (f.id.startswith("GS-02") and f.id >= "GS-02d")
    ]


def test_confirmed_fixture_counts(fixtures: dict[str, GoldenFixture]) -> None:
    """Sizing per DEC-P7-4 (P7 task 15): GS-07 -> 10, GS-08 -> 10 (P3's Q1-confirmed
    5/3 expanded additively; the 12 pending-capture ids are unscored until the
    DEC-P7-7 window), GS-02-conversational -> 4 (unchanged)."""
    from sutradhar.evals.generation_run import PENDING_CAPTURE_FIXTURES

    assert sum(1 for f in fixtures if f.startswith("GS-07")) == 10
    assert sum(1 for f in fixtures if f.startswith("GS-08")) == 10
    conversational = [f for f in fixtures if f.startswith("GS-02") and f >= "GS-02d"]
    assert len(conversational) == 4
    # The retrieval-shaped GS-02 negatives are untouched (P2 artifact must keep validating).
    assert {"GS-02a", "GS-02b", "GS-02c"} <= set(fixtures)
    assert len(_generation_slice(fixtures)) == 24
    # Pending-capture set is exactly the P7 additions and every id exists.
    assert PENDING_CAPTURE_FIXTURES <= {f.id for f in _generation_slice(fixtures)}
    assert len(PENDING_CAPTURE_FIXTURES) == 12


def test_generation_slice_fully_labelled(fixtures: dict[str, GoldenFixture]) -> None:
    """Every generation-slice fixture carries intent + slots + tool-call labels."""
    for f in _generation_slice(fixtures):
        assert f.expected_intent is not None, f.id
        assert f.expected_slots is not None, f.id
        assert f.expected_tool_calls, f.id


def test_labels_conform_to_frozen_taxonomy(
    fixtures: dict[str, GoldenFixture], artifacts: PromptArtifacts
) -> None:
    """Intent labels ∈ the 6 frozen labels; slot keys ⊆ the 7 frozen keys."""
    for f in _generation_slice(fixtures):
        intents = f.expected_intent if isinstance(f.expected_intent, list) else [f.expected_intent]
        for intent in intents:
            assert intent in artifacts.intent_labels, (f.id, intent)
        slots = f.expected_slots if isinstance(f.expected_slots, list) else [f.expected_slots]
        for turn_slots in slots:
            assert turn_slots is not None
            assert set(turn_slots) <= artifacts.slot_keys, (f.id, set(turn_slots))


def test_multi_turn_label_alignment(fixtures: dict[str, GoldenFixture]) -> None:
    """Multi-turn fixtures: len(intents) == len(slots) == number of query turns;
    single-turn fixtures use the scalar forms."""
    for f in _generation_slice(fixtures):
        if isinstance(f.query, list):
            assert isinstance(f.expected_intent, list), f.id
            assert isinstance(f.expected_slots, list), f.id
            assert len(f.expected_intent) == len(f.query), f.id
            assert len(f.expected_slots) == len(f.query), f.id
        else:
            assert isinstance(f.expected_intent, str), f.id
            assert isinstance(f.expected_slots, dict), f.id


def test_retrieval_and_graph_fixtures_stay_unlabelled(
    fixtures: dict[str, GoldenFixture],
) -> None:
    """Non-generation fixtures don't grow labels silently (frozen outside the slice)."""
    generation_ids = {f.id for f in _generation_slice(fixtures)}
    for f in fixtures.values():
        if f.id not in generation_ids:
            assert f.expected_intent is None, f.id
            assert f.expected_slots is None, f.id


def test_extension_preserves_extra_forbid() -> None:
    """The additive fields must not loosen the schema (extra='forbid' survives)."""
    base = {
        "id": "GS-99z",
        "name": "x",
        "category": "x",
        "subsystem": "guardrail",
        "query": "q",
        "query_lang": "en",
        "expected": {"no_match": True},
        "gating_metric": "m",
        "must_not": ["x"],
        "verify_source": ["s"],
    }
    GoldenFixture.model_validate(base)  # baseline OK
    with pytest.raises(ValidationError):
        GoldenFixture.model_validate({**base, "expected_intents": "typo-field"})


def test_gs08c_mid_turn_no_match_shape(fixtures: dict[str, GoldenFixture]) -> None:
    """GS-08c: turn 2 is the mid-conversation NO_MATCH — labelled out_of_catalog with the
    refused slot, and NO Tamil version appears in the expected answers."""
    f = fixtures["GS-08c"]
    assert isinstance(f.expected_intent, list)
    assert f.expected_intent[1] == "out_of_catalog"
    assert isinstance(f.expected_slots, list)
    assert f.expected_slots[1] == {"language": "ta"}
    assert all(v.language != "ta" for v in f.expected.versions)
