"""API request/response models for the P5 chat surface (P5_SPEC §2.2).

All models are pydantic ``extra="forbid"`` and mirror the repository result models, so
``sources[]`` / ``confidence`` / ``relationship`` / ``is_original`` flow to the client
untouched — the gating-story "every claim citing its source" clause is a passthrough
property, not a re-derivation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None  # null => new conversation
    message: str


class IntentPayload(BaseModel):
    """The parsed ``INTENT: {...}`` preamble of the model's final answer (may be null)."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    slots: dict[str, Any] = {}


class VersionPayload(BaseModel):
    """One surfaced language version — mirrors ``repository.VersionEntry`` untouched."""

    model_config = ConfigDict(extra="forbid")

    version_id: str
    title: str
    language: str | None
    year: int | None
    relationship: str | None  # is_original_of | is_remake_of | is_official_dub_of | …
    is_original: bool
    cast_lead: list[str] = []
    sources: list[dict[str, Any]] = []
    confidence: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_ref: str  # the claim the sources ground (the surfaced version's title)
    sources: list[dict[str, Any]]


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None  # wired by cost accounting (P5 task 10); None until then


class ChatResponse(BaseModel):
    """One successful conversation turn (GPU up). The §2.2 contract."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    status: Literal["up"] = "up"
    answer: str
    intent: IntentPayload | None = None
    versions: list[VersionPayload] = []
    citations: list[Citation] = []
    warnings: list[str] = []
    usage: Usage = Usage()
    latency_ms: float = 0.0
    tool_calls: int = 0
    trace_id: str | None = None


class TurnAborted(BaseModel):
    """The orchestrator's degradation signal: LLM off/error or rounds exhausted.

    Never surfaced raw — the API layer (task 9) maps it onto the structured offline
    payload (HTTP 200, DEC-P0-4 posture). State is NOT persisted for aborted turns, so a
    retry after the GPU resumes replays cleanly.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    status: Literal["off", "error"]
    detail: str
