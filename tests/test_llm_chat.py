"""Unit tests for LLMClient.chat() — OpenAI tool-calling round-trip (P3 task 2).

The whole HTTP surface is mocked through a single injected httpx.MockTransport (DEC-P0-4:
one shared httpx.Client covers the raw and SDK paths), so no network or model is touched.
Covers P3_SPEC §4: tool-call parsing, multi-message round-trip, usage/latency capture, and
the off/error contract extended to the chat path (connection refused -> status="off",
never a crash).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from sutradhar.config import Settings
from sutradhar.serving import ChatResult, LLMClient, ToolCall

_MODEL = "google/gemma-4-E4B"

_USAGE = {"prompt_tokens": 120, "completion_tokens": 17, "total_tokens": 137}


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_BASE_URL="http://localhost:8000/v1",
        LLM_MODEL=_MODEL,
        LLM_API_KEY="EMPTY",
    )


def _client(handler: object) -> LLMClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LLMClient(_settings(), http_client=httpx.Client(transport=transport))


def _chat_response(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": _MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _USAGE,
        },
    )


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_title",
            "description": "Resolve a film title to work/version ids.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }
]


def test_chat_content_roundtrip_with_usage_and_latency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(content="Papanasam is the Tamil remake of Drishyam.")

    result = _client(handler).chat([{"role": "user", "content": "papanasam original?"}])
    assert isinstance(result, ChatResult)
    assert result.status == "up"
    assert result.content == "Papanasam is the Tamil remake of Drishyam."
    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    assert result.usage == _USAGE
    assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
    # The raw assistant message is transcript/feedback-ready (OpenAI wire format).
    assert result.message is not None
    assert result.message["role"] == "assistant"


def test_chat_tool_call_parsing_parallel_calls() -> None:
    calls = [
        _tool_call("call_1", "resolve_title", json.dumps({"title": "Papanasam"})),
        _tool_call("call_2", "get_versions", json.dumps({"work_id": "wk_drishyam"})),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(tool_calls=calls, finish_reason="tool_calls")

    result = _client(handler).chat(
        [{"role": "user", "content": "papanasam?"}],
        tools=_TOOLS,
    )
    assert result.status == "up"
    assert result.content is None
    assert len(result.tool_calls) == 2
    first, second = result.tool_calls
    assert isinstance(first, ToolCall)
    assert (first.id, first.name) == ("call_1", "resolve_title")
    assert first.arguments == {"title": "Papanasam"}
    assert (second.name, second.arguments) == ("get_versions", {"work_id": "wk_drishyam"})
    assert result.finish_reason == "tool_calls"
    # tool_calls survive in the raw message for transcript feedback.
    assert result.message is not None and len(result.message["tool_calls"]) == 2


def test_chat_malformed_tool_arguments_never_crash() -> None:
    """Malformed JSON args -> arguments=None (a scored failure), raw string preserved."""
    calls = [
        _tool_call("call_1", "resolve_title", '{"title": "Papanasam'),  # truncated JSON
        _tool_call("call_2", "resolve_title", '"just a string"'),  # valid JSON, not an object
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(tool_calls=calls, finish_reason="tool_calls")

    result = _client(handler).chat([{"role": "user", "content": "x"}], tools=_TOOLS)
    assert result.status == "up"
    truncated, non_object = result.tool_calls
    assert truncated.arguments is None
    assert truncated.arguments_raw == '{"title": "Papanasam'
    assert non_object.arguments is None


def test_chat_multi_message_roundtrip_sends_full_history_and_tools() -> None:
    """The outbound request must carry every message role plus tools + tool_choice."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _chat_response(content="Drishyam (Malayalam, 2013) is the original.")

    history = [
        {"role": "system", "content": "You are Sutradhar."},
        {"role": "user", "content": "papanasam original?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("call_1", "resolve_title", '{"title": "Papanasam"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"work_id": "wk_drishyam"}'},
    ]
    result = _client(handler).chat(history, tools=_TOOLS, tool_choice="auto", temperature=0.0)
    assert result.status == "up"
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "tool"]
    assert seen["messages"][3]["tool_call_id"] == "call_1"
    assert seen["tools"] == _TOOLS
    assert seen["tool_choice"] == "auto"
    assert seen["temperature"] == 0.0
    assert seen["model"] == _MODEL


def test_chat_off_connection_refused_never_crashes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    result = _client(handler).chat([{"role": "user", "content": "x"}], tools=_TOOLS)
    assert result.status == "off"
    assert result.message is None
    assert result.tool_calls == ()
    assert result.usage is None
    assert result.latency_ms is None
    assert "endpoint OFF" in result.detail


def test_chat_error_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    result = _client(handler).chat([{"role": "user", "content": "x"}])
    assert result.status == "error"
    assert result.message is None
    assert "errored" in result.detail


def test_chat_result_serializes_without_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(content="ok")

    result = _client(handler).chat([{"role": "user", "content": "x"}])
    payload = json.dumps(result.to_dict())
    assert "EMPTY" not in payload  # api key never leaks
    assert set(result.to_dict()) == {
        "status",
        "message",
        "content",
        "tool_calls",
        "finish_reason",
        "usage",
        "latency_ms",
        "detail",
    }
