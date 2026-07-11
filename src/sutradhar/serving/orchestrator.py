"""Orchestrator: one live conversation turn for the P5 API (task 7, P5_SPEC §2.3).

The P3 driver's validate→execute→feedback agent loop (``evals/driver.py::run_fixture``),
lifted from golden fixtures to live user turns:

    state load → guardrails-in (caps) → agent loop (LLM ↔ validated v0 tools)
      → guardrails-out (output gate) → response assembly → state save

Same invariants as the driver:

- outbound ``tools`` generated from the frozen v0 artifact (``sutradhar.toolcalls``);
- every model-emitted call validated BEFORE execution — hallucinated tools/params and
  malformed arguments are fed back as tool errors and the loop continues, bounded by
  ``MAX_TOOL_ROUNDS``, never a crash;
- LLM ``off``/``error`` (or rounds exhausted without an answer) aborts the turn as a
  typed :class:`~sutradhar.serving.schemas.TurnAborted` — state is NOT persisted, so a
  retry after the GPU resumes replays cleanly; the API layer maps this to the structured
  offline payload (never a 5xx).

Guardrail hooks (P5 task 8 plugs in; defaults are honest passthroughs):

- ``spotlight(payload) -> str``: serializes a tool-result dict into the ``role:"tool"``
  message content (datamarking of untrusted content per DEC-P5-3);
- ``output_gate(answer, tool_titles) -> (answer, warnings)``: the deterministic
  no-hallucinated-movie gate applied before the answer reaches the user.

Response assembly (§2.3): ``versions[]`` mirrors the LAST ``get_versions`` result — or,
after a ``refine_filter``, the filtered ids mapped back through the retained entries so
refined turns keep cast/sources/confidence; one citation per surfaced version. Every
tool call additionally becomes a bounded :class:`TraceStep` on ``ChatResponse.trace``
(P6, DEC-P6-4) — the UI trace view renders what this loop already validated.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sutradhar.evals.driver import (
    MAX_TOOL_ROUNDS,
    ToolExecutionError,
    ToolExecutor,
)
from sutradhar.evals.generation import parse_intent_preamble
from sutradhar.obs.tracing import Tracer
from sutradhar.serving.llm_client import LLMClient
from sutradhar.serving.schemas import (
    ChatResponse,
    Citation,
    IntentPayload,
    TraceStep,
    TurnAborted,
    Usage,
    VersionPayload,
)
from sutradhar.serving.sessions import ConversationState, SessionStore, check_limits
from sutradhar.toolcalls import load_tool_schema, openai_tools, validate_emitted_call

# Hook signatures (task 8 provides the real implementations in serving.guardrails).
# spotlight returns (message content, warnings) — withheld adversarial content surfaces
# in the response warnings[] (P5_SPEC §2.5 layer 5).
Spotlight = Callable[[dict[str, Any]], tuple[str, list[str]]]
OutputGate = Callable[[str, list[str]], tuple[str, list[str]]]

_TITLE_KEYS = frozenset({"title", "matched_title", "canonical_title"})


def default_spotlight(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Identity serialization (no marking) — replaced by guardrails.spotlight in task 8."""
    return json.dumps(payload, ensure_ascii=False), []


def default_output_gate(answer: str, tool_titles: list[str]) -> tuple[str, list[str]]:
    """Passthrough — replaced by the no-hallucinated-movie gate in task 8."""
    return answer, []


def collect_titles(payload: Any) -> list[str]:
    """Every film-title string in a tool result (grounding surface for the output gate)."""
    titles: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _TITLE_KEYS and isinstance(value, str):
                    titles.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return titles


_SUMMARY_ERROR_CHARS = 200


def summarize_result(payload: dict[str, Any]) -> dict[str, Any]:
    """A BOUNDED trace digest of a tool-result payload (P6_SPEC §2.2).

    Kind + count + ids only — never titles, cast, sources, or plot text: the trace
    explains *how* the answer was assembled; the response's versions/citations carry
    the *what*. Pure function, one branch per v0 result shape.
    """
    if "error" in payload:  # validation feedback or ToolExecutionError feedback
        return {"kind": "error", "error": str(payload["error"])[:_SUMMARY_ERROR_CHARS]}
    if "candidates" in payload:  # resolve_title
        ids = [str(c["work_id"]) for c in payload["candidates"]]
        return {"kind": "candidates", "count": len(ids), "ids": ids}
    if "results" in payload:  # search_by_plot
        ids = [str(r["work_id"]) for r in payload["results"]]
        return {"kind": "results", "count": len(ids), "ids": ids}
    if "versions" in payload:  # get_versions (original + versions) / refine_filter
        entries = ([payload["original"]] if payload.get("original") else []) + list(
            payload["versions"]
        )
        ids = [str(e["version_id"]) for e in entries]
        return {"kind": "versions", "count": len(ids), "ids": ids}
    if "work_id" in payload:  # get_work
        return {"kind": "work", "count": 1, "ids": [str(payload["work_id"])]}
    return {"kind": "other", "keys": sorted(payload)[:10]}  # pragma: no cover — defensive


def _version_payload(entry: dict[str, Any]) -> VersionPayload:
    """A VersionEntry dict (model_dump json mode) → the API payload, untouched fields."""
    return VersionPayload(
        version_id=str(entry["version_id"]),
        title=entry["title"],
        language=entry.get("language"),
        year=entry.get("year"),
        relationship=entry.get("relationship"),
        is_original=bool(entry.get("is_original", False)),
        cast_lead=list(entry.get("cast_lead", [])),
        sources=list(entry.get("sources", [])),
        confidence=entry.get("confidence"),
    )


class _VersionTracker:
    """Retains every seen VersionEntry; tracks the currently-surfaced version set."""

    def __init__(self) -> None:
        self.entries: dict[str, VersionPayload] = {}
        self.current: list[str] = []

    def see_get_versions(self, payload: dict[str, Any]) -> None:
        ordered: list[str] = []
        original = payload.get("original")
        for entry in ([original] if original else []) + list(payload.get("versions", [])):
            version = _version_payload(entry)
            self.entries[version.version_id] = version
            if version.version_id not in ordered:
                ordered.append(version.version_id)
        self.current = ordered

    def see_refine_filter(self, payload: dict[str, Any]) -> None:
        ordered: list[str] = []
        for refined in payload.get("versions", []):
            version_id = str(refined["version_id"])
            if version_id not in self.entries:
                # Unseen id (refine over ids from an earlier session state): degrade to
                # the refine fields — never invent cast/sources we do not have.
                self.entries[version_id] = _version_payload(
                    {**refined, "cast_lead": [], "sources": [], "confidence": None}
                )
            ordered.append(version_id)
        self.current = ordered

    def versions(self) -> list[VersionPayload]:
        return [self.entries[vid] for vid in self.current]


class Orchestrator:
    """One conversation turn per :meth:`run_turn` call (constructed once per app)."""

    def __init__(
        self,
        client: LLMClient,
        store: SessionStore,
        execute_tool: ToolExecutor,
        *,
        system_prompt: str,
        prompt_hash: str,
        schema: dict[str, Any] | None = None,
        spotlight: Spotlight = default_spotlight,
        output_gate: OutputGate = default_output_gate,
        tracer: Tracer | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._store = store
        self._execute_tool = execute_tool
        self._system_prompt = system_prompt
        self._prompt_hash = prompt_hash
        self._schema = schema or load_tool_schema()
        self._tools = openai_tools(self._schema)
        self._spotlight = spotlight
        self._output_gate = output_gate
        self._tracer = tracer or Tracer()  # disabled no-op by default (DEC-P3-6)
        self._max_tool_rounds = max_tool_rounds
        self._temperature = temperature

    def run_turn(self, conversation_id: str | None, message: str) -> ChatResponse | TurnAborted:
        """Execute one user turn. Raises SessionLimitError on guardrail caps (→ 4xx)."""
        state = (self._store.load(conversation_id) if conversation_id else None) or (
            ConversationState.new(conversation_id)
        )
        check_limits(state, message)  # guardrails-in: turn cap + message-size cap

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            *state.messages,
            {"role": "user", "content": message},
        ]
        tracker = _VersionTracker()
        tool_titles: list[str] = []
        warnings: list[str] = []
        trace: list[TraceStep] = []
        usage = Usage()
        latency_ms = 0.0
        tool_call_count = 0
        answer: str | None = None

        with self._tracer.span(
            "turn",
            kind="agent",
            input={"conversation_id": state.conversation_id, "turn": state.turn_count},
            metadata={"prompt_hash": self._prompt_hash},
        ) as turn_span:
            for _round in range(self._max_tool_rounds):
                with self._tracer.span(
                    "chat",
                    kind="generation",
                    input={"round": _round, "messages": len(messages)},
                ) as chat_span:
                    result = self._client.chat(
                        messages,
                        tools=self._tools,
                        tool_choice="auto",
                        temperature=self._temperature,
                    )
                    chat_span.update(
                        output={
                            "status": result.status,
                            "finish_reason": result.finish_reason,
                            "tool_calls": len(result.tool_calls),
                        }
                    )
                if result.status != "up":
                    turn_span.update(output={"aborted": result.status})
                    return TurnAborted(  # degrade, never crash; state NOT saved
                        conversation_id=state.conversation_id,
                        status=result.status,  # narrowed to "off" | "error" here
                        detail=result.detail,
                    )
                if result.usage is not None:
                    usage.prompt_tokens += int(result.usage.get("prompt_tokens", 0))
                    usage.completion_tokens += int(result.usage.get("completion_tokens", 0))
                if result.latency_ms is not None:
                    latency_ms += result.latency_ms
                assert result.message is not None
                messages.append(result.message)

                if not result.tool_calls:
                    answer = result.content
                    break

                for call in result.tool_calls:
                    tool_call_count += 1
                    errors = validate_emitted_call(self._schema, call.name, call.arguments)
                    started = time.perf_counter()
                    with self._tracer.span(
                        f"tool:{call.name}", kind="tool", input=call.arguments
                    ) as tool_span:
                        if errors:
                            payload: dict[str, Any] = {"error": "; ".join(errors)}
                            tool_span.update(output=payload, level="ERROR")
                        else:
                            try:
                                assert call.arguments is not None  # guaranteed by validation
                                payload = self._execute_tool(call.name, call.arguments)
                                tool_span.update(output={"ok": True})
                            except ToolExecutionError as exc:
                                payload = {"error": str(exc)}
                                warnings.append(f"{call.name}: {exc}")
                                tool_span.update(output=payload, level="ERROR")
                    trace.append(
                        TraceStep(
                            step=tool_call_count,
                            tool=call.name,
                            arguments=call.arguments,
                            valid=not errors,
                            validation_error="; ".join(errors) if errors else None,
                            result_summary=summarize_result(payload),
                            latency_ms=round((time.perf_counter() - started) * 1000, 2),
                        )
                    )
                    if "error" not in payload:
                        tool_titles.extend(collect_titles(payload))
                        if call.name == "get_versions":
                            tracker.see_get_versions(payload)
                        elif call.name == "refine_filter":
                            tracker.see_refine_filter(payload)
                    # Untrusted content marked before it enters the prompt (D3);
                    # withheld-content warnings surface in the response.
                    content, spot_warnings = self._spotlight(payload)
                    warnings.extend(spot_warnings)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": content,
                        }
                    )

            if answer is None:
                turn_span.update(output={"aborted": "rounds_exhausted"})
                return TurnAborted(
                    conversation_id=state.conversation_id,
                    status="error",
                    detail=f"no answer within {self._max_tool_rounds} tool rounds",
                )

            gated_answer, gate_warnings = self._output_gate(answer, tool_titles)
            warnings.extend(gate_warnings)
            turn_span.update(
                output={"answer_chars": len(gated_answer), "tool_calls": tool_call_count}
            )

        parsed = parse_intent_preamble(gated_answer)
        versions = tracker.versions()
        now = datetime.now(tz=UTC)
        state.messages = messages[1:]  # system excluded; prompt_hash pins it (§2.2)
        state.turn_count += 1
        state.last_active = now
        self._store.save(state)

        return ChatResponse(
            conversation_id=state.conversation_id,
            answer=gated_answer,
            intent=(IntentPayload(intent=parsed.intent, slots=parsed.slots) if parsed else None),
            versions=versions,
            citations=[
                Citation(claim_ref=v.title, sources=v.sources) for v in versions if v.sources
            ],
            warnings=warnings,
            usage=usage,
            latency_ms=round(latency_ms, 2),
            tool_calls=tool_call_count,
            trace=trace,
            trace_id=self._tracer.last_trace_id,
        )
