"""P6 task 3 — the replay→turn adapter (P6_SPEC §2.2: one rendering path).

``replay_turns`` adapts the committed pinned-run transcripts into ChatResponse-shaped
turns so replayed and live turns render through the SAME UI components. Assertions run
against the real committed GS-08a artifact (the P4 base run — zero GPU, zero mocks) plus
synthetic edge cases for the honesty rules.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sutradhar.serving.degrade import load_replay, replay_turns
from sutradhar.toolcalls import load_tool_schema, validate_emitted_call


@pytest.fixture(scope="module")
def gs08a() -> dict[str, Any]:
    payload = load_replay("GS-08a")
    assert payload is not None
    return payload


def test_gs08a_one_turn_per_user_message(gs08a: dict[str, Any]) -> None:
    turns = gs08a["turns"]
    assert [t["message"] for t in turns] == [
        "the Drishyam with Ajay Devgn",
        "no, the original one",
        "is there a Telugu one?",
    ]
    assert [t["answer"] for t in turns] == gs08a["answers"]
    assert [t["intent"]["intent"] for t in turns] == ["disambiguate", "list_versions", "refine"]


def test_gs08a_versions_reconstructed_from_recorded_results_only(gs08a: dict[str, Any]) -> None:
    t0, t1, t2 = gs08a["turns"]
    # Turn 0 (resolve only) and turn 2 (no calls): no version set — live semantics.
    assert t0["versions"] == [] and t0["citations"] == []
    assert t2["versions"] == [] and t2["tool_calls"] == 0 and t2["trace"] == []
    # Turn 1: the recorded get_versions result → full Drishyam set, original flagged.
    titles = [v["title"] for v in t1["versions"]]
    assert len(titles) == 5 and titles[0] == "Drishyam"
    assert t1["versions"][0]["is_original"] is True
    assert all(v["sources"] for v in t1["versions"])  # citations pass through untouched
    assert [c["claim_ref"] for c in t1["citations"]] == titles
    # Trace summary count agrees with the surfaced set (original never counted twice).
    assert t1["trace"][0]["result_summary"]["count"] == 5


def test_gs08a_latencies_are_the_recorded_gpu_rounds(gs08a: dict[str, Any]) -> None:
    turns = gs08a["turns"]
    assert sum(t["latency_ms"] for t in turns) == pytest.approx(sum(gs08a["latencies_ms"]))
    assert turns[0]["latency_ms"] == pytest.approx(sum(gs08a["latencies_ms"][:2]))
    assert turns[2]["latency_ms"] == pytest.approx(gs08a["latencies_ms"][-1])
    # Per-call latency was not recorded — carried as 0.0, never faked.
    assert all(s["latency_ms"] == 0.0 for t in turns for s in t["trace"])


def test_gs08a_trace_steps_are_valid_v0_calls(gs08a: dict[str, Any]) -> None:
    """No hallucinated tool or parameter name can reach a rendered trace (§2.8)."""
    schema = load_tool_schema()
    steps = [s for t in gs08a["turns"] for s in t["trace"]]
    assert steps
    for step in steps:
        assert step["valid"] is True
        assert validate_emitted_call(schema, step["tool"], step["arguments"]) == []


def test_every_pinned_fixture_adapts_cleanly() -> None:
    """Adapter totality: every committed replayable fixture produces renderable turns."""
    from sutradhar.serving.degrade import available_replays

    for fixture_id in available_replays():
        payload = load_replay(fixture_id)
        assert payload is not None
        for turn in payload["turns"]:
            assert isinstance(turn["answer"], str)
            assert isinstance(turn["versions"], list)
            assert turn["latency_ms"] >= 0


def test_unparseable_tool_body_degrades_without_invention() -> None:
    payload = {
        "answers": ['INTENT: {"intent": "find_movie", "slots": {}}\nAnswer.'],
        "latencies_ms": [100.0, 200.0],
        "calls": [
            {
                "turn": 0,
                "tool": "get_versions",
                "arguments": {"work_id": "w1"},
                "schema_valid": True,
                "executed": True,
            }
        ],
        "messages": [
            {"role": "user", "content": "papanasam?"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "[SPOTLIGHTED] not json"},
            {"role": "assistant", "content": "Answer."},
        ],
    }
    (turn,) = replay_turns(payload)
    assert turn["versions"] == []  # never invented from an unreadable recording
    assert turn["trace"][0]["result_summary"] == {"kind": "unparsed"}
    assert turn["latency_ms"] == 300.0


def test_invalid_recorded_call_renders_as_invalid() -> None:
    payload = {
        "answers": ["ok"],
        "latencies_ms": [50.0],
        "calls": [
            {
                "turn": 0,
                "tool": "delete_database",
                "arguments": {"x": 1},
                "schema_valid": False,
                "executed": False,
            }
        ],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": json.dumps({"error": "hallucinated tool"})},
            {"role": "assistant", "content": "ok"},
        ],
    }
    (turn,) = replay_turns(payload)
    step = turn["trace"][0]
    assert step["valid"] is False
    assert step["validation_error"] == "hallucinated tool"
    assert step["result_summary"]["kind"] == "error"
    assert turn["versions"] == []
