"""P5 task 7 — orchestrator turn engine (scripted LLM, the test_driver.py house pattern).

The driver's proven loop lifted to live turns: per-call validation, bounded rounds,
off/error degradation as data, state carried across run_turn calls (GS-08 mechanics —
the HTTP layer lands in task 9).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.serving.llm_client import LLMClient
from sutradhar.serving.orchestrator import Orchestrator, collect_titles
from sutradhar.serving.schemas import ChatResponse, TurnAborted
from sutradhar.serving.sessions import (
    InMemorySessionStore,
    SessionLimitError,
    SessionStore,
)

SYSTEM_PROMPT = "You are Sutradhar. (frozen bundle stand-in)"
PROMPT_HASH = "testhash0000"

WORK_ID = str(uuid.uuid4())
V_ML, V_TA, V_HI = (str(uuid.uuid4()) for _ in range(3))


def _entry(vid: str, title: str, lang: str, year: int, rel: str, original: bool) -> dict[str, Any]:
    return {
        "version_id": vid,
        "title": title,
        "language": lang,
        "year": year,
        "cast_lead": ["Mohanlal"] if original else ["Kamal Haasan"],
        "relationship": rel,
        "is_original": original,
        "sources": [{"source": "wikidata", "ref": "Q15401703"}],
        "confidence": "HIGH",
    }


GET_VERSIONS_RESULT = {
    "original": _entry(V_ML, "Drishyam", "ml", 2013, "is_original_of", True),
    "versions": [
        _entry(V_TA, "Papanasam", "ta", 2015, "is_remake_of", False),
        _entry(V_HI, "Drishyam (Hindi)", "hi", 2015, "is_remake_of", False),
    ],
}

RESOLVE_RESULT = {
    "candidates": [
        {
            "work_id": WORK_ID,
            "version_id": V_TA,
            "matched_title": "Papanasam",
            "language": "ta",
            "year": 2015,
            "score": 1.0,
            "sources": [{"source": "wikidata", "ref": "Q18578149"}],
        }
    ],
    "ambiguous": False,
}

REFINE_RESULT = {
    "versions": [
        {
            "version_id": V_ML,
            "title": "Drishyam",
            "language": "ml",
            "year": 2013,
            "relationship": "is_original_of",
            "is_original": True,
        }
    ]
}

ANSWER_T1 = (
    'INTENT: {"intent": "list_versions", "slots": {"title": "Papanasam"}}\n'
    "**Papanasam** (2015, Tamil) is a remake of **Drishyam** (2013, Malayalam)."
)
ANSWER_T2 = (
    'INTENT: {"intent": "refine", "slots": {"era": "original"}}\n'
    "The original is **Drishyam** (2013, Malayalam), with Mohanlal."
)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


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


def _executor(calls: list[tuple[str, dict[str, Any]]]) -> Any:
    def execute(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((tool, args))
        if tool == "resolve_title":
            return RESOLVE_RESULT
        if tool == "get_versions":
            return GET_VERSIONS_RESULT
        if tool == "refine_filter":
            return REFINE_RESULT
        raise AssertionError(f"unexpected tool {tool}")

    return execute


def _orchestrator(
    model: _ScriptedModel,
    store: SessionStore | None = None,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    **kwargs: Any,
) -> tuple[Orchestrator, SessionStore]:
    store = store or InMemorySessionStore(3600)
    orch = Orchestrator(
        _client(model),
        store,
        _executor(calls if calls is not None else []),
        system_prompt=SYSTEM_PROMPT,
        prompt_hash=PROMPT_HASH,
        **kwargs,
    )
    return orch, store


TURN1_SCRIPT = [
    _response(tool_calls=[_tool_call("c1", "resolve_title", {"title": "Papanasam"})]),
    _response(tool_calls=[_tool_call("c2", "get_versions", {"work_id": WORK_ID})]),
    _response(content=ANSWER_T1),
]


def test_happy_path_single_turn() -> None:
    model = _ScriptedModel(list(TURN1_SCRIPT))
    calls: list[tuple[str, dict[str, Any]]] = []
    orch, store = _orchestrator(model, calls=calls)

    out = orch.run_turn(None, "which movie is papanasam a remake of?")
    assert isinstance(out, ChatResponse)

    # Version set mirrors get_versions untouched: original flagged, relationships kept.
    assert [v.title for v in out.versions] == ["Drishyam", "Papanasam", "Drishyam (Hindi)"]
    assert out.versions[0].is_original is True
    assert out.versions[1].relationship == "is_remake_of"
    assert out.versions[0].sources == [{"source": "wikidata", "ref": "Q15401703"}]
    assert out.versions[0].confidence == "HIGH"
    # Citations: one per surfaced version, claim_ref = title.
    assert [c.claim_ref for c in out.citations] == [v.title for v in out.versions]
    # Intent preamble parsed off the answer.
    assert out.intent is not None and out.intent.intent == "list_versions"
    assert out.intent.slots == {"title": "Papanasam"}
    assert out.answer == ANSWER_T1
    assert out.tool_calls == 2 and out.warnings == []
    assert out.usage.prompt_tokens == 30 and out.usage.completion_tokens == 15  # 3 rounds
    assert calls == [
        ("resolve_title", {"title": "Papanasam"}),
        ("get_versions", {"work_id": WORK_ID}),
    ]

    # State saved: full wire history, system EXCLUDED, one turn counted.
    state = store.load(out.conversation_id)
    assert state is not None and state.turn_count == 1
    assert all(m.get("role") != "system" for m in state.messages)
    assert state.messages[0]["role"] == "user"
    assert state.messages[-1]["content"] == ANSWER_T1

    # The model saw the frozen system prompt first, exactly once per request.
    for request in model.requests:
        assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert sum(1 for m in request["messages"] if m["role"] == "system") == 1


def test_backtracking_turn2_refines_within_version_set() -> None:
    """GS-08 mechanics: turn 2 sees the full history and refine_filter narrows versions[]
    through the retained entries (cast/sources/confidence preserved)."""
    store = InMemorySessionStore(3600)

    model1 = _ScriptedModel(list(TURN1_SCRIPT))
    orch1, _ = _orchestrator(model1, store=store)
    out1 = orch1.run_turn(None, "which movie is papanasam a remake of?")
    assert isinstance(out1, ChatResponse)

    model2 = _ScriptedModel(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "c3",
                        "refine_filter",
                        {"version_set": [V_ML, V_TA, V_HI], "by": {"era": "original"}},
                    )
                ]
            ),
            _response(content=ANSWER_T2),
        ]
    )
    orch2, _ = _orchestrator(model2, store=store)
    out2 = orch2.run_turn(out1.conversation_id, "no, the original one")
    assert isinstance(out2, ChatResponse)

    # The model received the FULL turn-1 history across HTTP-free process boundaries.
    first_request = model2.requests[0]["messages"]
    assert first_request[1]["content"] == "which movie is papanasam a remake of?"
    assert any(m.get("role") == "tool" for m in first_request)
    assert first_request[-1] == {"role": "user", "content": "no, the original one"}

    # refine narrowed the surfaced set; entry retained from turn 2's tracker? — turn 2
    # has no get_versions call, so the payload degrades to refine fields (no invention).
    assert [v.title for v in out2.versions] == ["Drishyam"]
    assert out2.versions[0].is_original is True
    assert out2.intent is not None and out2.intent.intent == "refine"

    state = store.load(out1.conversation_id)
    assert state is not None and state.turn_count == 2


def test_refine_after_get_versions_same_turn_keeps_rich_entries() -> None:
    model = _ScriptedModel(
        [
            _response(tool_calls=[_tool_call("c1", "get_versions", {"work_id": WORK_ID})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "refine_filter",
                        {"version_set": [V_ML, V_TA, V_HI], "by": {"era": "original"}},
                    )
                ]
            ),
            _response(content=ANSWER_T2),
        ]
    )
    orch, _ = _orchestrator(model)
    out = orch.run_turn(None, "the original one please")
    assert isinstance(out, ChatResponse)
    assert [v.title for v in out.versions] == ["Drishyam"]
    # Retained from the get_versions entry: cast/sources/confidence survive the refine.
    assert out.versions[0].cast_lead == ["Mohanlal"]
    assert out.versions[0].sources and out.versions[0].confidence == "HIGH"
    assert out.citations and out.citations[0].claim_ref == "Drishyam"


def test_hallucinated_tool_and_param_fed_back_loop_continues() -> None:
    model = _ScriptedModel(
        [
            _response(tool_calls=[_tool_call("c1", "delete_database", {"x": 1})]),
            _response(
                tool_calls=[
                    _tool_call("c2", "resolve_title", {"title": "Papanasam", "invented": True})
                ]
            ),
            _response(content=ANSWER_T1),
        ]
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    orch, _ = _orchestrator(model, calls=calls)
    out = orch.run_turn(None, "papanasam?")
    assert isinstance(out, ChatResponse)
    assert calls == []  # invalid calls NEVER reach the executor
    assert out.tool_calls == 2
    # The error payloads were fed back as tool messages (rounds 2 and 3 saw them).
    fed_back = [
        m["content"] for req in model.requests for m in req["messages"] if m.get("role") == "tool"
    ]
    assert any("hallucinated tool" in c for c in fed_back)
    assert any("invented" in c for c in fed_back)


def test_malformed_json_arguments_never_crash() -> None:
    model = _ScriptedModel(
        [
            _response(tool_calls=[_tool_call("c1", "resolve_title", "{not json")]),
            _response(content=ANSWER_T1),
        ]
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    orch, _ = _orchestrator(model, calls=calls)
    out = orch.run_turn(None, "papanasam?")
    assert isinstance(out, ChatResponse)
    assert calls == []


def test_tool_execution_error_fed_back_and_warned() -> None:
    from sutradhar.evals.driver import ToolExecutionError

    def execute(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        raise ToolExecutionError("plot search unavailable — embeddings endpoint off")

    model = _ScriptedModel(
        [
            _response(tool_calls=[_tool_call("c1", "search_by_plot", {"description": "x"})]),
            _response(content=ANSWER_T1),
        ]
    )
    store = InMemorySessionStore(3600)
    orch = Orchestrator(
        _client(model), store, execute, system_prompt=SYSTEM_PROMPT, prompt_hash=PROMPT_HASH
    )
    out = orch.run_turn(None, "wo film …")
    assert isinstance(out, ChatResponse)
    assert any("plot search unavailable" in w for w in out.warnings)


def test_llm_off_aborts_without_saving_state() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    store = InMemorySessionStore(3600)
    orch = Orchestrator(
        _client(refuse),
        store,
        _executor([]),
        system_prompt=SYSTEM_PROMPT,
        prompt_hash=PROMPT_HASH,
    )
    out = orch.run_turn("conv-1", "papanasam?")
    assert isinstance(out, TurnAborted)
    assert out.status == "off" and out.conversation_id == "conv-1"
    assert store.load("conv-1") is None  # no half-turn persisted; retry replays cleanly


def test_llm_error_aborts() -> None:
    def error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    orch = Orchestrator(
        _client(error),
        InMemorySessionStore(3600),
        _executor([]),
        system_prompt=SYSTEM_PROMPT,
        prompt_hash=PROMPT_HASH,
    )
    out = orch.run_turn(None, "hello")
    assert isinstance(out, TurnAborted) and out.status == "error"


def test_rounds_exhausted_aborts() -> None:
    looping = [
        _response(tool_calls=[_tool_call(f"c{i}", "resolve_title", {"title": "X"})])
        for i in range(10)
    ]
    orch, store = _orchestrator(_ScriptedModel(looping), max_tool_rounds=3)
    out = orch.run_turn(None, "papanasam?")
    assert isinstance(out, TurnAborted)
    assert out.status == "error" and "3 tool rounds" in out.detail


def test_turn_cap_raises_session_limit() -> None:
    store = InMemorySessionStore(3600)
    orch, _ = _orchestrator(_ScriptedModel([]), store=store)
    from sutradhar.serving.sessions import ConversationState

    state = ConversationState.new("conv-cap")
    state.turn_count = 20
    store.save(state)
    with pytest.raises(SessionLimitError, match="turn cap"):
        orch.run_turn("conv-cap", "one more")


def test_guardrail_hooks_are_applied() -> None:
    def spotlight(payload: dict[str, Any]) -> tuple[str, list[str]]:
        return "MARKED::" + json.dumps(payload, ensure_ascii=False), ["content withheld"]

    def gate(answer: str, titles: list[str]) -> tuple[str, list[str]]:
        assert "Drishyam" in titles and "Papanasam" in titles  # grounding surface fed in
        return answer + "\n[gated]", ["unverified title suppressed"]

    model = _ScriptedModel(list(TURN1_SCRIPT))
    orch, _ = _orchestrator(model, spotlight=spotlight, output_gate=gate)
    out = orch.run_turn(None, "papanasam?")
    assert isinstance(out, ChatResponse)
    assert out.answer.endswith("[gated]")
    # Spotlight warnings (per tool result) + gate warnings, in order.
    assert out.warnings == ["content withheld", "content withheld", "unverified title suppressed"]
    # Every tool message the model saw went through spotlight.
    tool_msgs = [
        m["content"] for req in model.requests for m in req["messages"] if m.get("role") == "tool"
    ]
    assert tool_msgs and all(c.startswith("MARKED::") for c in tool_msgs)


def test_collect_titles_walks_nested_results() -> None:
    titles = collect_titles(
        {
            "original": {"title": "Drishyam"},
            "versions": [{"title": "Papanasam"}],
            "candidates": [{"matched_title": "Drushyam"}],
            "results": [{"canonical_title": "Drishya"}],
            "noise": {"score": 1.0, "title_like": "not-a-title-key"},
        }
    )
    assert titles == ["Drishyam", "Papanasam", "Drushyam", "Drishya"]
