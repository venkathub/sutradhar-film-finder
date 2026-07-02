"""Conversation-driver unit tests (P3 task 6; P3_SPEC §4 test_driver.py).

Scripted mock model via httpx.MockTransport + a fake tool executor — no DB, no network.
Covers: bounded tool rounds; invalid emitted call → validation failure recorded + error fed
back + loop continues; multi-turn message assembly (turn 2 sees turn 1 context); transcript
completeness; off-endpoint graceful abort; execution-error feedback; outbound tools
generated from the frozen artifact.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.evals.driver import (
    MAX_TOOL_ROUNDS,
    RecordedPlotSearch,
    ToolExecutionError,
    load_retrieval_run,
    load_tool_schema,
    openai_tools,
    run_fixture,
    validate_emitted_call,
)
from sutradhar.evals.golden import GoldenFixture
from sutradhar.serving import LLMClient

_SCHEMA = load_tool_schema()

_RESOLVE_RESULT = {
    "candidates": [
        {
            "work_id": "wk-1",
            "version_id": "vr-1",
            "matched_title": "Drishyam",
            "language": "ml",
            "year": 2013,
            "score": 1.0,
            "sources": [{"source": "wikidata", "ref": "Q15401703"}],
        }
    ],
    "ambiguous": False,
}


def _fixture(query: str | list[str], fixture_id: str = "GS-08a") -> GoldenFixture:
    return GoldenFixture(
        id=fixture_id,
        name="test",
        category="backtracking",
        subsystem="generation/backtrack",
        query=query,
        query_lang="en",
        expected={"canonical_work": "Drishyam", "canonical_year": 2013},
        gating_metric="test",
        must_not=["crash"],
        verify_source=["Q15401703"],
    )


def _tool_call_msg(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _response(
    *, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _ScriptedModel:
    """Pops scripted responses in order; records every request body it receives."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if not self.script:
            return httpx.Response(200, json=_response(content="(script exhausted)"))
        return httpx.Response(200, json=self.script.pop(0))


def _client(handler: Any) -> LLMClient:
    settings = Settings(_env_file=None, LLM_BASE_URL="http://localhost:8000/v1")
    return LLMClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _echo_executor(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool == "resolve_title":
        return _RESOLVE_RESULT
    return {"echo": tool, "args": args}


def _run(
    script: list[dict[str, Any]],
    fixture: GoldenFixture,
    executor: Any = _echo_executor,
    **kwargs: Any,
) -> tuple[Any, _ScriptedModel]:
    model = _ScriptedModel(script)
    transcript = run_fixture(
        _client(model),
        fixture,
        system_prompt="SYSTEM",
        prompt_hash="testhash",
        schema=_SCHEMA,
        execute_tool=executor,
        **kwargs,
    )
    return transcript, model


# --- Happy path + transcript completeness ---


def test_single_turn_tool_flow_and_transcript() -> None:
    script = [
        _response(tool_calls=[_tool_call_msg("c1", "resolve_title", '{"title": "Drishyam"}')]),
        _response(content="**Drishyam (2013, Malayalam)** is the original."),
    ]
    transcript, model = _run(script, _fixture("show me Drishyam"))
    assert transcript.chat_status == "up"
    assert transcript.answers == ["**Drishyam (2013, Malayalam)** is the original."]
    assert len(transcript.calls) == 1
    call = transcript.calls[0]
    assert (call.tool, call.schema_valid, call.executed) == ("resolve_title", True, True)
    assert call.result == _RESOLVE_RESULT
    assert call.turn == 0
    # Transcript completeness: user + assistant(tool_calls) + tool + assistant(answer).
    roles = [m["role"] for m in transcript.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert len(transcript.usage) == 2 and len(transcript.latencies_ms) == 2
    assert transcript.prompt_hash == "testhash"
    # The system prompt went out on the wire but is excluded from the stored transcript.
    assert model.requests[0]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_tools_array_sent_from_frozen_artifact() -> None:
    script = [_response(content="hi")]
    _, model = _run(script, _fixture("q"))
    sent_tools = model.requests[0]["tools"]
    assert {t["function"]["name"] for t in sent_tools} == set(_SCHEMA["tools"])
    assert model.requests[0]["tool_choice"] == "auto"
    assert model.requests[0]["temperature"] == 0.0


# --- Invalid emitted calls: recorded, fed back, loop continues ---


def test_hallucinated_tool_recorded_and_loop_continues() -> None:
    script = [
        _response(tool_calls=[_tool_call_msg("c1", "lookup_movie", '{"name": "Drishyam"}')]),
        _response(content="recovered"),
    ]
    transcript, model = _run(script, _fixture("q"))
    call = transcript.calls[0]
    assert call.schema_valid is False
    assert call.executed is False
    assert any("hallucinated tool" in e for e in call.validation_errors)
    # The error payload was fed back to the model as the tool result...
    fed_back = model.requests[1]["messages"][-1]
    assert fed_back["role"] == "tool"
    assert "hallucinated tool" in json.loads(fed_back["content"])["error"]
    # ...and the conversation continued to a final answer.
    assert transcript.answers == ["recovered"]


def test_hallucinated_parameter_and_wrong_type_caught() -> None:
    script = [
        _response(
            tool_calls=[
                _tool_call_msg("c1", "get_versions", '{"work_id": "w1", "country": "IN"}'),
                _tool_call_msg("c2", "search_by_plot", '{"description": "x", "top_k": "ten"}'),
            ]
        ),
        _response(content="done"),
    ]
    transcript, _ = _run(script, _fixture("q"))
    hallucinated_param, wrong_type = transcript.calls
    assert hallucinated_param.schema_valid is False  # additionalProperties: false
    assert wrong_type.schema_valid is False  # top_k must be an integer
    assert transcript.answers == ["done"]


def test_malformed_arguments_json_recorded_not_crashed() -> None:
    script = [
        _response(tool_calls=[_tool_call_msg("c1", "resolve_title", '{"title": "Drish')]),
        _response(content="ok"),
    ]
    transcript, _ = _run(script, _fixture("q"))
    call = transcript.calls[0]
    assert call.schema_valid is False
    assert call.arguments is None
    assert call.arguments_raw == '{"title": "Drish'


# --- Bounded rounds ---


def test_tool_rounds_bounded() -> None:
    script = [
        _response(tool_calls=[_tool_call_msg(f"c{i}", "resolve_title", '{"title": "x"}')])
        for i in range(MAX_TOOL_ROUNDS + 3)
    ]
    transcript, model = _run(script, _fixture("q"))
    assert transcript.tool_rounds_exhausted is True
    assert transcript.answers == [None]
    assert len(transcript.calls) == MAX_TOOL_ROUNDS
    assert len(model.requests) == MAX_TOOL_ROUNDS


# --- Multi-turn assembly ---


def test_multi_turn_context_carried() -> None:
    script = [
        _response(tool_calls=[_tool_call_msg("c1", "resolve_title", '{"title": "Drishyam"}')]),
        _response(content="turn one answer"),
        _response(content="turn two answer"),
    ]
    transcript, model = _run(script, _fixture(["find Drishyam", "the original one"]))
    assert transcript.answers == ["turn one answer", "turn two answer"]
    # Turn 2's request must contain the FULL turn-1 context.
    turn2_roles = [m["role"] for m in model.requests[2]["messages"]]
    assert turn2_roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert model.requests[2]["messages"][-1]["content"] == "the original one"
    # Per-turn call indexing recorded.
    assert transcript.calls[0].turn == 0


# --- Degradation paths ---


def test_endpoint_off_aborts_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transcript = run_fixture(
        _client(handler),
        _fixture(["a", "b"]),
        system_prompt="SYSTEM",
        prompt_hash="h",
        schema=_SCHEMA,
        execute_tool=_echo_executor,
    )
    assert transcript.chat_status == "off"
    assert transcript.answers == [None]  # aborted at turn 1; never crashed


def test_execution_error_fed_back_and_loop_continues() -> None:
    def failing_executor(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        raise ToolExecutionError("work_id 'w1' not found")

    script = [
        _response(tool_calls=[_tool_call_msg("c1", "get_work", '{"work_id": "w1"}')]),
        _response(content="handled"),
    ]
    transcript, model = _run(script, _fixture("q"), executor=failing_executor)
    call = transcript.calls[0]
    assert call.schema_valid is True and call.executed is False
    assert call.error == "work_id 'w1' not found"
    assert "not found" in json.loads(model.requests[1]["messages"][-1]["content"])["error"]
    assert transcript.answers == ["handled"]


# --- Outbound generation + inbound validation helpers ---


def test_openai_tools_generated_from_artifact() -> None:
    tools = openai_tools(_SCHEMA)
    names = [t["function"]["name"] for t in tools]
    assert names == list(_SCHEMA["tools"])
    resolve = next(t for t in tools if t["function"]["name"] == "resolve_title")
    assert resolve["function"]["parameters"]["required"] == ["title"]
    assert "$defs" in resolve["function"]["parameters"]  # refs resolve standalone


def test_validate_emitted_call_verdicts() -> None:
    assert validate_emitted_call(_SCHEMA, "resolve_title", {"title": "Drishyam"}) == []
    assert validate_emitted_call(_SCHEMA, "lookup_movie", {"x": 1}) != []
    assert validate_emitted_call(_SCHEMA, "resolve_title", None) != []
    assert validate_emitted_call(_SCHEMA, "get_versions", {"work_id": "w", "country": "IN"}) != []


# --- search_by_plot replay from the committed P2 artifact ---


def test_recorded_plot_search_replays_committed_run() -> None:
    plot_search = RecordedPlotSearch(load_retrieval_run())
    result = plot_search.result_for("GS-07a")
    assert result["results"], "GS-07a was in the P2 run; replay must return its works"
    assert {"work_id", "canonical_title", "score"} <= set(result["results"][0])
    # A P3-only fixture (never in the recorded run) replays as honest abstention.
    unseen = plot_search.result_for("GS-02e")
    assert unseen == {"results": [], "abstain": True}


def test_recorded_plot_search_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="no config record"):
        RecordedPlotSearch(load_retrieval_run(), config_key="nonexistent")
