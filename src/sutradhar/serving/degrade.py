"""Graceful degradation service (P5 task 9, P5_SPEC §2.4, DEC-P5-5).

GPU **off is the default state** and a first-class success path (the DEC-P0-4 posture at
the API layer): `/api/chat` short-circuits to a structured offline payload — HTTP 200,
never a 5xx — pointing at the recorded evidence, and `/api/replay/{fixture}` serves the
committed pinned-run transcripts so the Papanasam story is demonstrable with zero GPU.

The status cache (D5-minimal): one ``LLMClient.health()`` probe per TTL window (~30 s)
instead of a connect-timeout per request. Process-local by design — a single API process
serves the demo; the Redis-backed variant is the documented future-ops extension, not a
need the demo has (DEC-P5-5's cache-invalidation-discipline argument).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sutradhar.evals.generation import parse_intent_preamble
from sutradhar.evals.generation_run import GENERATION_RUNS_DIR, load_generation_run
from sutradhar.serving.llm_client import EndpointStatus

STATUS_CACHE_TTL_S = 30.0

OFFLINE_DETAIL = "Live demo offline by design — the GPU is on-demand."


class StatusCache:
    """Caches an endpoint-status probe for ``ttl_s`` (a paused GPU costs one timeout per
    window, not one per request)."""

    def __init__(
        self,
        probe: Callable[[], EndpointStatus],
        *,
        ttl_s: float = STATUS_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._ttl_s = ttl_s
        self._clock = clock
        self._cached: EndpointStatus | None = None
        self._deadline = 0.0

    def current(self) -> EndpointStatus:
        if self._cached is None or self._clock() >= self._deadline:
            self._cached = self._probe()
            self._deadline = self._clock() + self._ttl_s
        return self._cached

    def invalidate(self) -> None:
        self._cached = None


def offline_payload(
    conversation_id: str | None,
    detail: str = OFFLINE_DETAIL,
    *,
    demo_video: str | None = None,
) -> dict[str, Any]:
    """The §2.2 GPU-off response body — same route, structured, never an error.

    ``demo_video`` comes from ``DEMO_VIDEO_URL`` (P6): unset ⇒ the key is omitted
    (P6_SPEC §2.2) — the UI treats absence as "no video", never renders a dead link.
    """
    evidence: dict[str, Any] = {
        "benchmarks": "docs/BENCHMARKS.md",
        "replay": "/api/replay/GS-08a",
    }
    if demo_video:
        evidence["demo_video"] = demo_video
    return {
        "conversation_id": conversation_id,
        "status": "off",
        "detail": detail,
        "evidence": evidence,
        "request_live_demo": "see docs/RUNBOOK.md",
    }


def load_replay(
    fixture_id: str,
    *,
    run_id: str | None = None,
    runs_dir: Path = GENERATION_RUNS_DIR,
) -> dict[str, Any] | None:
    """One fixture's committed transcript from the pinned generation run (None = unknown).

    The ROADMAP §1(b) clause — "when the GPU is off, the same story replays from recorded
    evidence": messages, validated tool calls, per-turn answers and real GPU latencies,
    stamped with the run they came from.
    """
    artifact = load_generation_run(runs_dir, run_id)
    for fixture in artifact.fixtures:
        if fixture.fixture_id != fixture_id:
            continue
        transcript = fixture.transcript
        payload = {
            "fixture_id": fixture_id,
            "run_id": artifact.run_id,
            "mode": artifact.mode,
            "model": artifact.model,
            "prompt_hash": artifact.prompt_hash,
            "chat_status": transcript.chat_status,
            "messages": transcript.messages,
            "calls": [
                {
                    "turn": call.turn,
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "schema_valid": call.schema_valid,
                    "executed": call.executed,
                }
                for call in transcript.calls
            ],
            "answers": transcript.answers,
            "latencies_ms": transcript.latencies_ms,
        }
        payload["turns"] = replay_turns(payload)  # additive: the UI render path (P6)
        return payload
    return None


def available_replays(
    *,
    run_id: str | None = None,
    runs_dir: Path = GENERATION_RUNS_DIR,
) -> list[str]:
    """Fixture ids the pinned run can replay (listed in the 404 body — discoverable)."""
    artifact = load_generation_run(runs_dir, run_id)
    return [fixture.fixture_id for fixture in artifact.fixtures]


def list_replays(
    *,
    run_id: str | None = None,
    runs_dir: Path = GENERATION_RUNS_DIR,
) -> dict[str, Any]:
    """``GET /api/replays`` body (P6 task 1): the pinned run's identity + its fixtures.

    Promotes :func:`available_replays` from the 404 body to a first-class discovery
    route the UI replay browser lists from — stamped with the run it came from, same
    provenance fields as :func:`load_replay`.
    """
    artifact = load_generation_run(runs_dir, run_id)
    return {
        "run_id": artifact.run_id,
        "mode": artifact.mode,
        "model": artifact.model,
        "prompt_hash": artifact.prompt_hash,
        "available": [fixture.fixture_id for fixture in artifact.fixtures],
    }


def _parse_tool_content(raw: Any) -> dict[str, Any] | None:
    """A recorded ``role:"tool"`` message body as a dict, or None (never a crash)."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def replay_turns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt a recorded transcript into ChatResponse-shaped turns (P6 task 3, §2.2).

    One render model for live and replayed turns: each user message becomes a turn
    carrying ``answer`` / ``intent`` / ``versions`` / ``citations`` / ``trace`` /
    ``latency_ms`` with the SAME field names and semantics as a live ``ChatResponse``
    (plus ``message``, the user text). Honesty rules:

    - ``versions[]`` is reconstructed by feeding the RECORDED ``get_versions`` /
      ``refine_filter`` results through the orchestrator's own tracker — recorded data
      only, fresh per turn (live semantics); an unparseable tool body degrades to a
      turn without version cards, never invented entries;
    - ``latency_ms`` is the turn's real recorded GPU rounds; per-call latency was not
      recorded, so trace steps carry ``0.0`` (the turn number is the honest one);
    - ``trace[]`` comes from the recorded validated calls (``schema_valid`` → ``valid``).
    """
    # Local import: degrade is imported by jobs that never render turns.
    from sutradhar.serving.orchestrator import _VersionTracker, summarize_result
    from sutradhar.serving.schemas import TraceStep

    answers: list[str] = list(payload.get("answers", []))
    latencies: list[float] = list(payload.get("latencies_ms", []))
    calls_by_turn: dict[int, list[dict[str, Any]]] = {}
    for call in payload.get("calls", []):
        calls_by_turn.setdefault(int(call["turn"]), []).append(call)

    # Segment the wire messages into user turns; allocate one recorded latency per
    # assistant round; collect each turn's tool-result bodies in emission order.
    segments: list[dict[str, Any]] = []
    lat_i = 0
    for message in payload.get("messages", []):
        role = message.get("role")
        if role == "user":
            segments.append(
                {"message": message.get("content"), "latency_ms": 0.0, "tool_bodies": []}
            )
        elif not segments:
            continue  # defensive: transcripts always start with a user message
        elif role == "assistant":
            if lat_i < len(latencies):
                segments[-1]["latency_ms"] += latencies[lat_i]
                lat_i += 1
        elif role == "tool":
            segments[-1]["tool_bodies"].append(message.get("content"))

    turns: list[dict[str, Any]] = []
    for turn_no, segment in enumerate(segments):
        tracker = _VersionTracker()  # fresh per turn — matches live orchestrator turns
        steps: list[dict[str, Any]] = []
        turn_calls = calls_by_turn.get(turn_no, [])
        for step_no, (call, body) in enumerate(
            zip(turn_calls, segment["tool_bodies"], strict=False), start=1
        ):
            parsed = _parse_tool_content(body)
            valid = bool(call["schema_valid"])
            if parsed is None:
                summary: dict[str, Any] = {"kind": "unparsed"}
            else:
                summary = summarize_result(parsed)
                if valid and call.get("executed") and "error" not in parsed:
                    if call["tool"] == "get_versions":
                        tracker.see_get_versions(parsed)
                    elif call["tool"] == "refine_filter":
                        tracker.see_refine_filter(parsed)
            validation_error = None
            if not valid and parsed is not None and isinstance(parsed.get("error"), str):
                validation_error = parsed["error"]
            steps.append(
                TraceStep(
                    step=step_no,
                    tool=call["tool"],
                    arguments=call.get("arguments"),
                    valid=valid,
                    validation_error=validation_error,
                    result_summary=summary,
                    latency_ms=0.0,  # not recorded per call — never faked
                ).model_dump(mode="json")
            )
        answer = answers[turn_no] if turn_no < len(answers) else ""
        parsed_intent = parse_intent_preamble(answer)
        versions = [v.model_dump(mode="json") for v in tracker.versions()]
        turns.append(
            {
                "message": segment["message"],
                "answer": answer,
                "intent": (
                    {"intent": parsed_intent.intent, "slots": parsed_intent.slots}
                    if parsed_intent
                    else None
                ),
                "versions": versions,
                "citations": [
                    {"claim_ref": v["title"], "sources": v["sources"]}
                    for v in versions
                    if v["sources"]
                ],
                "warnings": [],
                "latency_ms": round(float(segment["latency_ms"]), 2),
                "tool_calls": len(turn_calls),
                "trace": steps,
            }
        )
    return turns
