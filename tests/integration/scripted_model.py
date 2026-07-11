"""Scripted id-binding graph model + helpers (P5 task 12, extracted in P6 task 7).

Shared by the API golden regressions (``test_api_golden_regressions.py``) and the
Playwright E2E server (``tests/e2e/e2e_server.py``): a deterministic fake LLM that
drives a fixed v0 tool plan, binding REAL ids out of prior tool results — no LLM,
no GPU, but the true served request path end to end.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from sutradhar.serving.llm_client import ChatResult, ToolCall
from sutradhar.toolcalls import validate_emitted_call

# ``step_fn(step, messages)`` returns either tool-call dicts or the final prose answer.
StepFn = Callable[[int, list[dict[str, Any]]], "list[dict[str, Any]] | str"]


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results the model has seen — spotlight-marked, so undo the datamark to parse."""
    out = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m["content"]
        # Strip the provenance-notice first line + restore datamarked spaces (guardrails).
        body = content.split("\n", 1)[1] if content.startswith("[TOOL RESULT") else content
        out.append(json.loads(body.replace("\u02c6", " ")))
    return out


def first_work_id(messages: list[dict[str, Any]]) -> str:
    for result in tool_results(messages):
        for candidate in result.get("candidates", []):
            return str(candidate["work_id"])
    raise AssertionError("no resolve_title candidate to bind a work_id from")


def version_ids(messages: list[dict[str, Any]]) -> list[str]:
    for result in tool_results(messages):
        if "versions" in result and "original" in result:  # get_versions shape
            ids = [result["original"]["version_id"]] if result["original"] else []
            ids += [v["version_id"] for v in result["versions"]]
            return [str(i) for i in ids]
    raise AssertionError("no get_versions result to bind version ids from")


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedGraphModel:
    """Drives a fixed tool plan; ``step_fn`` returns tool calls or a final prose answer."""

    def __init__(self, step_fn: StepFn) -> None:
        self._step_fn = step_fn
        self._step = 0

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> ChatResult:
        emitted = self._step_fn(self._step, messages)
        self._step += 1
        if isinstance(emitted, str):
            return ChatResult(
                status="up",
                message={"role": "assistant", "content": emitted},
                content=emitted,
                tool_calls=(),
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=11.0,
                detail="ok",
            )
        tool_calls = tuple(
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments_raw=c["function"]["arguments"],
                arguments=json.loads(c["function"]["arguments"]),
            )
            for c in emitted
        )
        return ChatResult(
            status="up",
            message={"role": "assistant", "content": None, "tool_calls": emitted},
            content=None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=11.0,
            detail="ok",
        )


class RecordingExecutor:
    """Wraps an executor, capturing every (tool, args, result) for assertions and
    validating each call against a v0 schema (the emitted-tool-calls-validate test)."""

    def __init__(self, inner: Callable[..., dict[str, Any]], schema: dict[str, Any]) -> None:
        self._inner = inner
        self._schema = schema
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[tuple[str, dict[str, Any]]] = []
        self.validation_errors: list[list[str]] = []

    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, args))
        self.validation_errors.append(validate_emitted_call(self._schema, tool, args))
        result = self._inner(tool, args)
        self.results.append((tool, result))
        return result
