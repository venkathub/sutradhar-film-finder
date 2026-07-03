"""Conversation driver: multi-turn fixture execution (P3 task 6; P3_SPEC §2.3).

Flow per fixture::

    system prompt (frozen, hashed) + tools (generated from tool_schema.v0.json, §2.8)
      └─ per user turn: chat() → validate every emitted call (DEC-P1-8 validator)
           ├─ invalid (hallucinated tool/param/type) → recorded verdict; error payload fed
           │   back as the tool result; the loop CONTINUES (bounded, never a crash)
           └─ valid → execute against sutradhar.graph.repository / the recorded plot search
         loop until the assistant answers in prose or MAX_TOOL_ROUNDS; then next turn

Design rules:

- **Outbound tools are generated, never hand-written** — the ``tools`` array is derived from
  the frozen ``tool_schema.v0.json`` artifact, so drift is impossible (§2.8).
- **Inbound calls are validated before execution** with the same jsonschema pattern as the
  DEC-P1-8 conformance test (params subschema + root ``$defs``), applied to model-emitted
  calls: hallucinated tool names, hallucinated parameters (``additionalProperties: false``)
  and wrong-typed arguments are all caught and *scored*, not raised.
- **``search_by_plot`` replays the committed P2 retrieval run** (``RETRIEVAL_RUN`` pinned):
  results come from the recorded per-fixture works so no neural op runs on the laptop and
  BOTH Table 2 columns (base in the P4 window, QLoRA after) see byte-identical tool
  behaviour. Fixtures without a recorded plot query (the P3 conversational negatives —
  out-of-catalog by construction) replay as ``abstain``. Description *quality* is scored
  separately by tool-call accuracy; the replay only fixes the grounding surface.
- **Off/error endpoints degrade, never crash** (DEC-P0-4): a fixture aborts with
  ``chat_status="off"|"error"`` recorded; remaining turns are skipped.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from sutradhar.evals.generation import EmittedCall
from sutradhar.evals.golden import GoldenFixture
from sutradhar.evals.retrieval import EvalRunArtifact, QueryRecord
from sutradhar.obs.tracing import Tracer
from sutradhar.serving.llm_client import LLMClient

TOOL_SCHEMA_PATH = Path("docs/phases/tool_schema.v0.json")
MAX_TOOL_ROUNDS = 6

# Executor signature: (tool_name, validated_arguments) -> v0-shaped result dict.
# Raises ToolExecutionError for runtime failures (unknown id, bad uuid, …) — the driver
# feeds the error back to the model and continues; it never crashes the run.
ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]


class ToolExecutionError(RuntimeError):
    """A schema-valid call that failed at execution time (fed back, loop continues)."""


# --- Tool schema artifact: outbound generation + inbound validation (DEC-P1-8 reuse) ---


def load_tool_schema(path: Path = TOOL_SCHEMA_PATH) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def params_subschema(schema: dict[str, Any], tool: str) -> dict[str, Any]:
    """The tool's params schema with the root $defs attached (same shape the DEC-P1-8
    conformance test validates golden expected_tool_calls against)."""
    sub = dict(schema["tools"][tool]["params"])
    sub["$defs"] = schema["$defs"]
    return sub


def openai_tools(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The outbound ``tools`` array, GENERATED from the frozen artifact (§2.8)."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": params_subschema(schema, name),
            },
        }
        for name, tool in schema["tools"].items()
    ]


def validate_emitted_call(
    schema: dict[str, Any],
    tool: str,
    arguments: dict[str, Any] | None,
) -> list[str]:
    """Validation errors for one model-emitted call (empty list = valid)."""
    if tool not in schema["tools"]:
        return [f"hallucinated tool: {tool!r} is not in TOOL_SCHEMA v0"]
    if arguments is None:
        return ["malformed tool-call arguments: not a JSON object"]
    validator = Draft202012Validator(params_subschema(schema, tool))
    return [e.message for e in validator.iter_errors(arguments)]


# --- search_by_plot replay (committed P2 retrieval run; DEC-P2-6 posture) ---


class RecordedPlotSearch:
    """Replays the pinned retrieval-run artifact per driven fixture (see module doc)."""

    def __init__(self, artifact: EvalRunArtifact, config_key: str | None = None) -> None:
        key = config_key or artifact.winner
        if key is None or key not in artifact.records:
            raise ValueError(f"retrieval run {artifact.run_id}: no config record {key!r}")
        record = artifact.records[key]
        self.run_id = artifact.run_id
        self.config_key = key
        self._queries: dict[str, QueryRecord] = {**record.queries, **record.negatives}

    def result_for(self, fixture_id: str, top_k: int = 10) -> dict[str, Any]:
        record = self._queries.get(fixture_id)
        if record is None:
            # Not part of the recorded run (P3 conversational negatives — out-of-catalog
            # by construction, golden-validated): honest abstention, never a fabrication.
            return {"results": [], "abstain": True}
        return {
            "results": [
                {
                    "work_id": w.work_id,
                    "canonical_title": w.title,
                    "language": w.language,
                    "year": w.year,
                    "score": w.score,
                }
                for w in record.works[:top_k]
            ],
            "abstain": record.abstain,
        }


def load_retrieval_run(runs_dir: Path = Path("evals/retrieval_runs")) -> EvalRunArtifact:
    """Latest committed retrieval-run artifact (same selection as the Tier-1 regressions)."""
    runs = [
        f
        for f in sorted(runs_dir.glob("*.json"))
        if not f.name.endswith((".meta.json", ".calibration.json"))
    ]
    if not runs:
        raise FileNotFoundError(f"no committed retrieval-run artifact under {runs_dir}")
    return EvalRunArtifact.model_validate_json(runs[-1].read_text(encoding="utf-8"))


# --- Repository-backed tool executor ---


def build_executor(
    session: Session,
    plot_search: RecordedPlotSearch,
    fixture_id_ref: dict[str, str],
) -> ToolExecutor:
    """Maps validated calls onto the five v0 repository functions.

    ``fixture_id_ref`` is a one-key mutable mapping ({"fixture_id": …}) the driver updates
    per fixture, so search_by_plot replays the right recorded record without threading the
    fixture through the executor signature.
    """
    from sutradhar.graph import repository

    def _uuid(value: Any, field: str) -> uuid.UUID:
        try:
            return uuid.UUID(str(value))
        except ValueError as exc:
            raise ToolExecutionError(f"{field}: {value!r} is not a known id") from exc

    def execute(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "resolve_title":
            return repository.resolve_title(
                session, args["title"], args.get("language")
            ).model_dump(mode="json")
        if tool == "search_by_plot":
            return plot_search.result_for(fixture_id_ref["fixture_id"], int(args.get("top_k", 10)))
        if tool == "get_work":
            result = repository.get_work(session, _uuid(args["work_id"], "work_id"))
            if result is None:
                raise ToolExecutionError(f"work_id {args['work_id']!r} not found")
            return result.model_dump(mode="json")
        if tool == "get_versions":
            return repository.get_versions(
                session,
                _uuid(args["work_id"], "work_id"),
                scope=args.get("scope", "indian"),
                include_sequels=bool(args.get("include_sequels", False)),
            ).model_dump(mode="json")
        if tool == "refine_filter":
            version_set = [_uuid(v, "version_set") for v in args["version_set"]]
            by = repository.RefineBy.model_validate(args["by"])
            return repository.refine_filter(session, version_set, by).model_dump(mode="json")
        raise ToolExecutionError(f"no executor for tool {tool!r}")  # pragma: no cover

    return execute


# --- Transcript records (embedded by the task-9 GenerationRunArtifact) ---


class EmittedCallRecord(BaseModel):
    """One emitted call: raw + validation verdict + execution outcome (P3_SPEC §2.2)."""

    model_config = ConfigDict(extra="forbid")

    turn: int  # 0-based user-turn index
    call_id: str
    tool: str
    arguments_raw: str
    arguments: dict[str, Any] | None
    schema_valid: bool
    validation_errors: list[str] = []
    executed: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_emitted_call(self) -> EmittedCall:
        return EmittedCall(
            tool=self.tool,
            arguments=self.arguments,
            schema_valid=self.schema_valid,
            result=self.result,
        )


class FixtureTranscript(BaseModel):
    """Full auditable record of one fixture conversation (mirrored by Langfuse traces)."""

    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    prompt_hash: str
    chat_status: str  # "up" | "off" | "error" (first non-up aborts the fixture)
    messages: list[dict[str, Any]]  # OpenAI wire format, system message EXCLUDED (hashed)
    calls: list[EmittedCallRecord] = []
    answers: list[str | None] = []  # final prose answer per user turn (None = no answer)
    usage: list[dict[str, int]] = []  # per chat() round
    latencies_ms: list[float] = []  # per chat() round
    tool_rounds_exhausted: bool = False

    def emitted_calls(self) -> list[EmittedCall]:
        return [c.to_emitted_call() for c in self.calls]


# --- The driver loop ---


def run_fixture(
    client: LLMClient,
    fixture: GoldenFixture,
    *,
    system_prompt: str,
    prompt_hash: str,
    schema: dict[str, Any],
    execute_tool: ToolExecutor,
    fixture_id_ref: dict[str, str] | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    temperature: float = 0.0,
    tracer: Tracer | None = None,
) -> FixtureTranscript:
    """Execute one golden fixture conversation end-to-end (see module doc for the flow)."""
    tracer = tracer or Tracer()  # default: disabled no-op (DEC-P3-6)
    if fixture_id_ref is not None:
        fixture_id_ref["fixture_id"] = fixture.id
    tools = openai_tools(schema)
    turns = fixture.query if isinstance(fixture.query, list) else [fixture.query]

    transcript = FixtureTranscript(
        fixture_id=fixture.id,
        prompt_hash=prompt_hash,
        chat_status="up",
        messages=[],
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    with tracer.span(
        f"fixture:{fixture.id}",
        kind="agent",
        metadata={"fixture_id": fixture.id, "prompt_hash": prompt_hash},
    ) as fixture_span:
        for turn_index, user_turn in enumerate(turns):
            messages.append({"role": "user", "content": user_turn})
            answer: str | None = None
            for _round in range(max_tool_rounds):
                with tracer.span(
                    "chat",
                    kind="generation",
                    input={"turn": turn_index, "round": _round, "messages": len(messages)},
                ) as chat_span:
                    result = client.chat(
                        messages, tools=tools, tool_choice="auto", temperature=temperature
                    )
                    chat_span.update(
                        output={
                            "status": result.status,
                            "finish_reason": result.finish_reason,
                            "tool_calls": len(result.tool_calls),
                        }
                    )
                if result.status != "up":
                    transcript.chat_status = result.status
                    transcript.answers.append(None)
                    transcript.messages = messages[1:]
                    fixture_span.update(output={"aborted": result.status})
                    return transcript  # degrade, never crash (DEC-P0-4)
                if result.usage is not None:
                    transcript.usage.append(result.usage)
                if result.latency_ms is not None:
                    transcript.latencies_ms.append(result.latency_ms)
                assert result.message is not None
                messages.append(result.message)

                if not result.tool_calls:
                    answer = result.content
                    break

                for call in result.tool_calls:
                    errors = validate_emitted_call(schema, call.name, call.arguments)
                    record = EmittedCallRecord(
                        turn=turn_index,
                        call_id=call.id,
                        tool=call.name,
                        arguments_raw=call.arguments_raw,
                        arguments=call.arguments,
                        schema_valid=not errors,
                        validation_errors=errors,
                    )
                    with tracer.span(
                        f"tool:{call.name}", kind="tool", input=call.arguments
                    ) as tool_span:
                        if errors:
                            payload: dict[str, Any] = {"error": "; ".join(errors)}
                            tool_span.update(output=payload, level="ERROR")
                        else:
                            try:
                                assert call.arguments is not None  # guaranteed by validation
                                payload = execute_tool(call.name, call.arguments)
                                record.executed = True
                                record.result = payload
                                tool_span.update(output={"ok": True})
                            except ToolExecutionError as exc:
                                payload = {"error": str(exc)}
                                record.error = str(exc)
                                tool_span.update(output=payload, level="ERROR")
                    transcript.calls.append(record)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(payload, ensure_ascii=False),
                        }
                    )
            else:
                transcript.tool_rounds_exhausted = True
            transcript.answers.append(answer)
        fixture_span.update(
            output={
                "answers": sum(1 for a in transcript.answers if a),
                "calls": len(transcript.calls),
            }
        )

    transcript.messages = messages[1:]  # system excluded; prompt_hash pins it
    return transcript
