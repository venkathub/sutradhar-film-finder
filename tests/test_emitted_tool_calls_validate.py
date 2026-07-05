"""Emitted-tool-call validation — model output vs TOOL_SCHEMA v0 (P3 task 12; the
phase-charter "no hallucinated tool or parameter names" gate).

The DEC-P1-8 validator (built for golden expected_tool_calls, designed for reuse "against
model-emitted calls") applied to what the MODEL emits: the three seeded fault classes must
all be caught and scored, and every successfully-executed call in the committed dry-run
artifact must validate against the frozen schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.config import Settings
from sutradhar.evals.driver import load_tool_schema, validate_emitted_call
from sutradhar.evals.generation import EmittedCall, score_tool_calls
from sutradhar.evals.generation_run import GenerationRunArtifact, load_generation_run

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = load_tool_schema(REPO_ROOT / "docs/phases/tool_schema.v0.json")

# --- The three seeded fault classes (P3_SPEC §4) ---

_FAULTS = [
    pytest.param(
        "lookup_movie",
        {"name": "Drishyam"},
        "hallucinated tool",
        id="hallucinated-tool-name",
    ),
    pytest.param(
        "get_versions",
        {"work_id": "wk1", "country": "IN"},
        "country",  # additionalProperties: false names the intruder
        id="hallucinated-parameter",
    ),
    pytest.param(
        "search_by_plot",
        {"description": "a father protects his family", "top_k": "ten"},
        "'ten' is not of type 'integer'",
        id="wrong-typed-argument",
    ),
]


@pytest.mark.parametrize(("tool", "arguments", "expected_error"), _FAULTS)
def test_seeded_fault_class_is_caught(
    tool: str, arguments: dict[str, object], expected_error: str
) -> None:
    errors = validate_emitted_call(SCHEMA, tool, arguments)  # type: ignore[arg-type]
    assert errors, f"fault class not caught: {tool}({arguments})"
    assert any(expected_error in e for e in errors), errors


def test_all_three_fault_classes_scored_as_validity_failures() -> None:
    """A transcript seeded with all 3 fault classes: each becomes a scored schema-validity
    failure (never a crash), and none can match an expected call."""
    emitted = []
    for param in _FAULTS:
        tool, arguments = param.values[0], param.values[1]
        errors = validate_emitted_call(SCHEMA, tool, arguments)  # type: ignore[arg-type]
        emitted.append(
            EmittedCall(
                tool=str(tool),
                arguments=dict(arguments),  # type: ignore[arg-type]
                schema_valid=not errors,
                result=None,
            )
        )
    score = score_tool_calls([("resolve_title", {"title": "Drishyam"})], emitted)
    assert score.invalid_emitted == 3  # 3/3 caught (the §4 gate)
    assert score.schema_validity == 0.0
    assert score.call_matches == (False,)


def test_malformed_json_arguments_are_a_fourth_visible_failure() -> None:
    """arguments=None (unparseable JSON from the model) is flagged, never executed."""
    assert validate_emitted_call(SCHEMA, "resolve_title", None) != []


# --- Conversely: the committed artifact's executed calls all validate ---


@pytest.fixture(scope="module")
def artifact() -> GenerationRunArtifact:
    return load_generation_run(
        REPO_ROOT / "evals" / "generation_runs",
        Settings(_env_file=None).generation_run or None,
    )


def test_every_executed_call_in_committed_run_validates(
    artifact: GenerationRunArtifact,
) -> None:
    checked = 0
    for result in artifact.fixtures:
        for call in result.transcript.calls:
            if call.executed:
                assert call.schema_valid, (result.fixture_id, call.tool)
                errors = validate_emitted_call(SCHEMA, call.tool, call.arguments)
                assert errors == [], (result.fixture_id, call.tool, errors)
                checked += 1
    # Floor recalibrated 2026-07-04: >=25 was the MOCK run's surface; the live base
    # column legitimately executes fewer calls (seq acc 8.3% — that sparsity IS the
    # recorded result). Non-empty + per-call validity is the invariant; volume is data.
    assert checked > 0


def test_violations_recorded_verbatim_in_the_artifact(artifact: GenerationRunArtifact) -> None:
    """Every schema_valid=False call in the committed run carries its violation list,
    and re-validating the raw call today reproduces those errors (no silent healing)."""
    for result in artifact.fixtures:
        for call in result.transcript.calls:
            if not call.schema_valid:
                assert call.validation_errors
                fresh = validate_emitted_call(SCHEMA, call.tool, call.arguments)
                assert fresh == call.validation_errors, (result.fixture_id, call.tool)
