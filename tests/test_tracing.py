"""Tracing wrapper tests (P3 task 10; P3_SPEC §4 test_tracing.py, DEC-P3-6).

Covers: no-op with keys unset (no SDK import, no crash); spans emitted to a fake sink
with a client injected; driver + judge chokepoint wiring; trace export auth/shape.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any

import httpx

from sutradhar.config import Settings
from sutradhar.obs.tracing import Tracer, export_trace

# --- Fake sink (duck-types the langfuse client surface the wrapper uses) ---


class _FakeSpan:
    def __init__(self, name: str, kind: str, input: Any, metadata: Any) -> None:
        self.name = name
        self.kind = kind
        self.input = input
        self.metadata = metadata
        self.updates: list[dict[str, Any]] = []
        self.trace_id = "trace-123"

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeLangfuse:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.flushed = 0

    @contextmanager
    def start_as_current_observation(
        self, *, name: str, as_type: str = "span", input: Any = None, metadata: Any = None
    ) -> Any:
        span = _FakeSpan(name, as_type, input, metadata)
        self.spans.append(span)
        yield span

    def get_trace_url(self, *, trace_id: str) -> str:
        return f"https://langfuse.example/trace/{trace_id}"

    def flush(self) -> None:
        self.flushed += 1


# --- No-op guarantees (keys unset) ---


def test_disabled_without_keys_and_never_imports_sdk(monkeypatch: Any) -> None:
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)
    tracer = Tracer(Settings(_env_file=None))
    assert tracer.enabled is False
    with tracer.span("anything", kind="tool", input={"x": 1}) as span:
        span.update(output={"y": 2})  # accepted, does nothing
    assert tracer.trace_url() is None
    tracer.flush()  # no crash
    assert "langfuse" not in sys.modules  # lazy import never happened


def test_partial_keys_stay_disabled() -> None:
    settings = Settings(_env_file=None, LANGFUSE_PUBLIC_KEY="pk")  # secret+host missing
    assert Tracer(settings).enabled is False


# --- Enabled path against the fake sink ---


def test_spans_emitted_with_names_kinds_and_updates() -> None:
    sink = _FakeLangfuse()
    tracer = Tracer(client=sink)
    assert tracer.enabled is True
    with tracer.span("fixture:GS-08a", kind="agent", metadata={"fixture_id": "GS-08a"}):
        with tracer.span("chat", kind="generation", input={"round": 0}) as chat:
            chat.update(output={"status": "up"})
        with tracer.span("tool:resolve_title", kind="tool", input={"title": "Drishyam"}) as t:
            t.update(output={"ok": True})
    names = [(s.name, s.kind) for s in sink.spans]
    assert names == [
        ("fixture:GS-08a", "agent"),
        ("chat", "generation"),
        ("tool:resolve_title", "tool"),
    ]
    assert sink.spans[1].updates == [{"output": {"status": "up"}}]
    assert sink.spans[0].metadata == {"fixture_id": "GS-08a"}
    assert tracer.last_trace_id == "trace-123"
    assert tracer.trace_url() == "https://langfuse.example/trace/trace-123"
    tracer.flush()
    assert sink.flushed == 1


# --- Chokepoint wiring: driver ---


def test_driver_emits_fixture_chat_and_tool_spans() -> None:
    from sutradhar.evals.driver import load_tool_schema, run_fixture
    from sutradhar.evals.golden import GoldenFixture
    from sutradhar.serving import LLMClient

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not any(m["role"] == "tool" for m in body["messages"]):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "resolve_title", "arguments": '{"title": "D"}'},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "done"}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "m",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    sink = _FakeLangfuse()
    fixture = GoldenFixture(
        id="GS-07a",
        name="t",
        category="c",
        subsystem="intent/translit",
        query="q",
        query_lang="en",
        expected={"canonical_work": "D", "canonical_year": 2013},
        gating_metric="m",
        must_not=["x"],
        verify_source=["Q1"],
    )
    client = LLMClient(
        Settings(_env_file=None),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transcript = run_fixture(
        client,
        fixture,
        system_prompt="S",
        prompt_hash="h",
        schema=load_tool_schema(),
        execute_tool=lambda tool, args: {"candidates": [], "ambiguous": False},
        tracer=Tracer(client=sink),
    )
    assert transcript.answers == ["done"]
    names = [(s.name, s.kind) for s in sink.spans]
    assert names == [
        ("fixture:GS-07a", "agent"),
        ("chat", "generation"),
        ("tool:resolve_title", "tool"),
        ("chat", "generation"),
    ]
    # The fixture span closes with a summary update.
    assert sink.spans[0].updates[-1]["output"] == {"answers": 1, "calls": 1}


def test_judge_emits_evaluator_span() -> None:
    from sutradhar.evals.judge import JudgeClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "j",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"score": 1.0}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    sink = _FakeLangfuse()
    judge = JudgeClient(
        Settings(
            _env_file=None,
            JUDGE_BASE_URL="http://j:8000/v1",
            JUDGE_MODEL="openai/gpt-oss-20b",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        tracer=Tracer(client=sink),
    )
    verdict = judge.judge_coherence([{"user": "u", "assistant": "a"}])
    assert verdict.score == 1.0
    assert [(s.name, s.kind) for s in sink.spans] == [("judge", "evaluator")]
    assert sink.spans[0].updates[-1]["output"] == {"score": 1.0, "error": None}


# --- Trace export (evidence longevity) ---


def test_export_trace_uses_basic_auth_and_returns_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"id": "trace-123", "observations": []})

    settings = Settings(
        _env_file=None,
        LANGFUSE_PUBLIC_KEY="pk",
        LANGFUSE_SECRET_KEY="sk",
        LANGFUSE_HOST="https://langfuse.example",
    )
    payload = export_trace(
        "trace-123",
        settings,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert payload["id"] == "trace-123"
    assert seen["url"] == "https://langfuse.example/api/public/traces/trace-123"
    assert seen["auth"].startswith("Basic ")


def test_export_trace_requires_keys() -> None:
    import pytest

    with pytest.raises(ValueError, match="LANGFUSE"):
        export_trace("t", Settings(_env_file=None))
