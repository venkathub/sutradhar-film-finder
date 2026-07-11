"""P6 task 6 — ``test_ui_trace_tool_calls_validate`` (P6_SPEC §2.8/§4).

The tool-schema conformance gate for the rendered trace: every trace step in every
committed transcript the replay browser exposes must either (a) validate against the
frozen ``tool_schema.v0.json`` via the SAME ``validate_emitted_call`` the orchestrator
uses, or (b) carry its explicit error state. No hallucinated tool or parameter name
can reach a rendered trace, and nothing invalid renders silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sutradhar.serving.degrade import available_replays, load_replay
from sutradhar.toolcalls import load_tool_schema, validate_emitted_call

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = REPO_ROOT / "ui" / "app" / "src" / "generated" / "tool_labels.json"

SCHEMA = load_tool_schema()
LABELLED_TOOLS = set(json.loads(LABEL_MAP.read_text(encoding="utf-8"))["tools"])


def _all_replay_payloads() -> list[dict[str, Any]]:
    payloads = []
    for fixture_id in available_replays():
        payload = load_replay(fixture_id)
        assert payload is not None
        payloads.append(payload)
    return payloads


@pytest.mark.parametrize("payload", _all_replay_payloads(), ids=lambda p: str(p["fixture_id"]))
def test_ui_trace_tool_calls_validate(payload: dict[str, Any]) -> None:
    """Valid steps re-validate against v0 AND are renderable via the generated label
    map; invalid steps carry their error state (the ✗ render, never a silent row)."""
    for turn in payload["turns"]:
        for step in turn["trace"]:
            if step["valid"]:
                errors = validate_emitted_call(SCHEMA, step["tool"], step["arguments"])
                assert errors == [], (
                    f"{payload['fixture_id']}: step {step['step']} ({step['tool']}) is "
                    f"marked valid but fails v0 validation: {errors}"
                )
                assert step["tool"] in LABELLED_TOOLS, (
                    f"{payload['fixture_id']}: valid tool {step['tool']!r} missing from "
                    "the generated label map — regenerate tool_labels.json"
                )
            else:
                assert step["validation_error"] or step["result_summary"].get("kind") in (
                    "error",
                    "unparsed",
                ), (
                    f"{payload['fixture_id']}: invalid step {step['step']} has no "
                    "rendered error state"
                )


@pytest.mark.parametrize("payload", _all_replay_payloads(), ids=lambda p: str(p["fixture_id"]))
def test_recorded_schema_verdicts_agree_with_v0(payload: dict[str, Any]) -> None:
    """The run's recorded ``schema_valid`` verdicts round-trip: re-validating every raw
    recorded call against today's frozen artifact reproduces the recorded verdict —
    the artifact has not drifted since the run was captured."""
    for call in payload["calls"]:
        errors = validate_emitted_call(SCHEMA, call["tool"], call["arguments"])
        assert (errors == []) == bool(call["schema_valid"]), (
            f"{payload['fixture_id']}: recorded schema_valid={call['schema_valid']} for "
            f"{call['tool']} but re-validation says {errors or 'valid'}"
        )


def test_pinned_run_exercises_the_trace_surface() -> None:
    """Sanity: the conformance gate above is not vacuous — the pinned run contains
    validated tool calls across multiple fixtures."""
    payloads = _all_replay_payloads()
    steps = [s for p in payloads for t in p["turns"] for s in t["trace"]]
    assert len(payloads) >= 5 and len(steps) >= 10
    assert any(s["valid"] for s in steps)
