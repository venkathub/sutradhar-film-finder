"""Mock-endpoint unit tests (P3 task 11) — the scripted player drives through the REAL
driver with a fake executor (no DB): well-behaved flow, seeded hallucinated tool with
recovery, seeded invented movie, and abstaining negatives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx

from sutradhar.config import Settings
from sutradhar.evals.driver import load_tool_schema, run_fixture
from sutradhar.evals.generation import collect_result_titles, detect_hallucinated_movies
from sutradhar.evals.golden import load_fixtures
from sutradhar.serving import LLMClient

_REPO = Path(__file__).resolve().parents[1]


def _load_mock() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mock_llm_under_test", _REPO / "evals" / "mock_llm.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mock_llm = _load_mock()
SCHEMA = load_tool_schema(_REPO / "docs/phases/tool_schema.v0.json")
FIXTURES = {f.id: f for f in load_fixtures(_REPO / "evals" / "golden")}

_RESOLVE = {
    "candidates": [
        {
            "work_id": "wk1",
            "version_id": "vr0",
            "matched_title": "Drishyam",
            "score": 1.0,
            "sources": [{"source": "wikidata", "ref": "Q1"}],
        }
    ],
    "ambiguous": False,
}
_VERSIONS = {
    "original": {"version_id": "vr0", "title": "Drishyam", "language": "ml", "year": 2013},
    "versions": [
        {"version_id": "vr1", "title": "Drishyam", "language": "hi", "year": 2015},
        {"version_id": "vr2", "title": "Drushyam", "language": "te", "year": 2014},
        {"version_id": "vr3", "title": "Papanasam", "language": "ta", "year": 2015},
        {"version_id": "vr4", "title": "Drishya", "language": "kn", "year": 2014},
        {"version_id": "vr5", "title": "Apthamitra", "language": "kn", "year": 2004},
        {"version_id": "vr6", "title": "Manichitrathazhu", "language": "ml", "year": 1993},
    ],
}


def _fake_executor(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool == "resolve_title":
        return _RESOLVE
    if tool == "search_by_plot":
        return {
            "results": [{"work_id": "wk1", "canonical_title": "Drishyam", "score": 0.9}],
            "abstain": False,
        }
    if tool == "get_versions":
        return _VERSIONS
    if tool == "refine_filter":
        return {
            "versions": [
                dict(_VERSIONS["versions"][0], relationship="is_remake_of", is_original=False)
            ]
        }
    return {"work_id": "wk1"}


def _drive(fixture_id: str) -> Any:
    model = mock_llm.ScriptedModel(list(FIXTURES.values()))
    client = LLMClient(
        Settings(_env_file=None),
        http_client=httpx.Client(transport=httpx.MockTransport(model)),
    )
    return run_fixture(
        client,
        FIXTURES[fixture_id],
        system_prompt="S",
        prompt_hash="h",
        schema=SCHEMA,
        execute_tool=_fake_executor,
    )


def test_seeded_hallucinated_tool_recorded_then_recovers() -> None:
    transcript = _drive("GS-07a")
    assert transcript.calls[0].tool == "lookup_movie"
    assert transcript.calls[0].schema_valid is False  # caught by DEC-P1-8 validation
    # Recovery: the real expected sequence still executes and the turn gets an answer.
    executed = [c.tool for c in transcript.calls if c.executed]
    assert executed == ["search_by_plot", "get_versions"]
    assert transcript.answers[0] and "INTENT:" in transcript.answers[0]


def test_seeded_invention_lands_on_gs07e_only() -> None:
    transcript = _drive("GS-07e")
    answer = transcript.answers[0]
    assert answer is not None and "Chokher Aloy" in answer
    allowed = collect_result_titles(transcript.emitted_calls())
    report = detect_hallucinated_movies(answer, allowed)
    assert report.inventions == ("Chokher Aloy",)  # exactly the seed, nothing else


def test_multi_turn_backtracking_plays_all_turns() -> None:
    transcript = _drive("GS-08a")
    assert len(transcript.answers) == 3 and all(transcript.answers)
    tools = [c.tool for c in transcript.calls]
    assert tools == [
        "resolve_title",
        "get_versions",
        "refine_filter",
        "refine_filter",
        "refine_filter",
    ]
    # Placeholders were bound to ids the conversation actually returned.
    refine_args = transcript.calls[2].arguments
    assert refine_args is not None
    assert set(refine_args["version_set"]) <= {"vr0", "vr1", "vr2", "vr3", "vr4", "vr5", "vr6"}


def test_negative_fixture_abstains_with_no_titles() -> None:
    model = mock_llm.ScriptedModel(list(FIXTURES.values()))
    client = LLMClient(
        Settings(_env_file=None),
        http_client=httpx.Client(transport=httpx.MockTransport(model)),
    )
    transcript = run_fixture(
        client,
        FIXTURES["GS-02d"],
        system_prompt="S",
        prompt_hash="h",
        schema=SCHEMA,
        execute_tool=lambda tool, args: {"candidates": [], "ambiguous": False},
    )
    answer = transcript.answers[0]
    assert answer is not None and "NO_MATCH" in answer
    report = detect_hallucinated_movies(answer, set())
    assert report.asserted == () and report.invention_count == 0


def test_mid_turn_no_match_gs08c_abstains_then_recovers() -> None:
    transcript = _drive("GS-08c")
    assert len(transcript.answers) == 3
    assert transcript.answers[1] is not None and "NO_MATCH" in transcript.answers[1]
    assert transcript.answers[2] is not None and "INTENT:" in transcript.answers[2]
