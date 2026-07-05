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

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def offline_payload(conversation_id: str | None, detail: str = OFFLINE_DETAIL) -> dict[str, Any]:
    """The §2.2 GPU-off response body — same route, structured, never an error."""
    return {
        "conversation_id": conversation_id,
        "status": "off",
        "detail": detail,
        "evidence": {
            "benchmarks": "docs/BENCHMARKS.md",
            "replay": "/api/replay/GS-08a",
            "demo_video": None,  # lands in P6
        },
        "request_live_demo": "see docs/RUNBOOK.md (P6)",
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
        return {
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
    return None


def available_replays(
    *,
    run_id: str | None = None,
    runs_dir: Path = GENERATION_RUNS_DIR,
) -> list[str]:
    """Fixture ids the pinned run can replay (listed in the 404 body — discoverable)."""
    artifact = load_generation_run(runs_dir, run_id)
    return [fixture.fixture_id for fixture in artifact.fixtures]
