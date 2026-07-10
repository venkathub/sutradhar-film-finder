"""Conversation-state store for the P5 API (task 6, DEC-P5-2).

Server-side session state keyed by ``conversation_id``: the driver's messages-carry-state
convention (P3 ``FixtureTranscript.messages``), given a keyed server home so multi-turn
backtracking (GS-08 "no, the original one") survives across HTTP requests.

Design rules (P5_SPEC §2.2):

- **The system prompt is never stored** — ``prompt_hash`` pins it (the frozen bundle);
  the record model REJECTS ``role:"system"`` messages so the convention is enforced at
  the type boundary, not by discipline.
- **TTL = natural session expiry** (``SESSION_TTL_S``): every ``save`` refreshes it
  (sliding window). Redis in the compose stack; in-memory with an injectable clock for
  tests/forks — one contract, one shared test suite.
- **Caps are guardrails, not config:** ``MAX_TURNS`` / ``MAX_MESSAGE_BYTES`` are the
  spec's "basic request-size/turn-count limits only" — enforced via
  :func:`check_limits`, surfaced by the orchestrator as a structured client error.
- **Degrade, never crash:** a corrupted stored record loads as ``None`` (fresh
  conversation), the DEC-P0-4 posture at the state layer.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

# Basic demo limits (P5_SPEC non-goals: no auth/rate-limit product features).
MAX_TURNS = 20
MAX_MESSAGE_BYTES = 8_000

_KEY_PREFIX = "sutradhar:session:"


class SessionLimitError(ValueError):
    """A guardrail cap was exceeded — surfaced as a structured client error, never a 5xx."""


def new_conversation_id() -> str:
    return str(uuid.uuid4())


class ConversationState(BaseModel):
    """One conversation's server-side record (P5_SPEC §2.2)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    created_at: datetime
    last_active: datetime
    # OpenAI wire messages, system EXCLUDED — prompt_hash pins it (the driver convention).
    messages: list[dict[str, Any]] = []
    turn_count: int = 0

    @field_validator("messages")
    @classmethod
    def _no_system_message(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any(m.get("role") == "system" for m in messages):
            raise ValueError(
                "system messages are never stored — the frozen prompt bundle (prompt_hash) "
                "pins the system prompt (P5_SPEC §2.2)"
            )
        return messages

    @classmethod
    def new(cls, conversation_id: str | None = None) -> ConversationState:
        now = datetime.now(tz=UTC)
        return cls(
            conversation_id=conversation_id or new_conversation_id(),
            created_at=now,
            last_active=now,
            messages=[],
            turn_count=0,
        )


def check_limits(
    state: ConversationState,
    new_message: str,
    *,
    max_turns: int = MAX_TURNS,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> None:
    """Raise :class:`SessionLimitError` when a cap would be exceeded by this turn."""
    if state.turn_count >= max_turns:
        raise SessionLimitError(
            f"conversation turn cap reached ({max_turns}) — start a new conversation"
        )
    size = len(new_message.encode("utf-8"))
    if size > max_message_bytes:
        raise SessionLimitError(
            f"message too large ({size} bytes > {max_message_bytes}) — shorten the message"
        )


class SessionStore(Protocol):
    """The store contract both implementations satisfy (one shared test suite)."""

    def load(self, conversation_id: str) -> ConversationState | None: ...

    def save(self, state: ConversationState) -> None: ...

    def delete(self, conversation_id: str) -> None: ...


class InMemorySessionStore:
    """Dict-backed store with an injectable clock — tests/forks, and the GPU-off-friendly
    fallback when Redis is unreachable (wired in the app lifespan, task 9)."""

    def __init__(self, ttl_s: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._entries: dict[str, tuple[float, str]] = {}  # id -> (deadline, payload json)

    def load(self, conversation_id: str) -> ConversationState | None:
        entry = self._entries.get(conversation_id)
        if entry is None:
            return None
        deadline, payload = entry
        if self._clock() >= deadline:
            self._entries.pop(conversation_id, None)  # expired
            return None
        return _parse_state(payload)

    def save(self, state: ConversationState) -> None:
        # TTL refreshed on every save — sliding-window expiry, same as the Redis leg.
        deadline = self._clock() + self._ttl_s
        self._entries[state.conversation_id] = (deadline, state.model_dump_json())

    def delete(self, conversation_id: str) -> None:
        self._entries.pop(conversation_id, None)


class RedisSessionStore:
    """Redis-backed store (the compose service; DEC-P5-2). ``SET … EX ttl`` per save."""

    def __init__(self, client: Any, ttl_s: int) -> None:
        self._redis = client
        self._ttl_s = ttl_s

    def load(self, conversation_id: str) -> ConversationState | None:
        payload = self._redis.get(_KEY_PREFIX + conversation_id)
        if payload is None:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        return _parse_state(payload)

    def save(self, state: ConversationState) -> None:
        self._redis.set(
            _KEY_PREFIX + state.conversation_id, state.model_dump_json(), ex=self._ttl_s
        )

    def delete(self, conversation_id: str) -> None:
        self._redis.delete(_KEY_PREFIX + conversation_id)


def _parse_state(payload: str) -> ConversationState | None:
    """Corrupted stored state loads as None (fresh conversation) — degrade, never crash."""
    try:
        return ConversationState.model_validate_json(payload)
    except (ValueError, json.JSONDecodeError):
        return None
