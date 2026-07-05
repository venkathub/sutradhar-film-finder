"""P5 task 6 — conversation session store (DEC-P5-2): one contract, two implementations.

The shared suite runs against BOTH stores (in-memory + fakeredis), so the test/fork leg
and the compose-Redis leg can never drift. TTL *expiry semantics* are proven on the
in-memory leg (injectable clock); the Redis leg proves the TTL is actually set/refreshed
on the key (fakeredis tracks real time, so expiry itself isn't sleep-tested).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import fakeredis
import pytest

from sutradhar.serving.sessions import (
    MAX_MESSAGE_BYTES,
    MAX_TURNS,
    ConversationState,
    InMemorySessionStore,
    RedisSessionStore,
    SessionLimitError,
    SessionStore,
    check_limits,
    new_conversation_id,
)

TTL_S = 3600


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _in_memory() -> tuple[SessionStore, _Clock]:
    clock = _Clock()
    return InMemorySessionStore(TTL_S, clock=clock), clock


def _redis() -> tuple[SessionStore, Any]:
    client = fakeredis.FakeRedis()
    return RedisSessionStore(client, TTL_S), client


@pytest.fixture(params=["in_memory", "redis"])
def store(request: pytest.FixtureRequest) -> SessionStore:
    return {"in_memory": _in_memory, "redis": _redis}[request.param]()[0]


def _state(messages: list[dict[str, Any]] | None = None, turns: int = 1) -> ConversationState:
    state = ConversationState.new()
    state.messages = messages or [
        {"role": "user", "content": "wo film jisme baap evidence chhupata hai"},
        {"role": "assistant", "content": "INTENT …\n**Drishyam** (2013) …"},
    ]
    state.turn_count = turns
    return state


# --- Shared contract (both implementations) ---


def test_round_trip(store: SessionStore) -> None:
    state = _state()
    store.save(state)
    loaded = store.load(state.conversation_id)
    assert loaded is not None
    assert loaded == state  # messages order, timestamps, turn_count — all preserved


def test_missing_id_is_none(store: SessionStore) -> None:
    assert store.load(new_conversation_id()) is None


def test_delete(store: SessionStore) -> None:
    state = _state()
    store.save(state)
    store.delete(state.conversation_id)
    assert store.load(state.conversation_id) is None
    store.delete(state.conversation_id)  # idempotent


def test_save_overwrites(store: SessionStore) -> None:
    state = _state(turns=1)
    store.save(state)
    state.turn_count = 2
    state.messages.append({"role": "user", "content": "no, the original one"})
    store.save(state)
    loaded = store.load(state.conversation_id)
    assert loaded is not None and loaded.turn_count == 2
    assert loaded.messages[-1]["content"] == "no, the original one"


# --- TTL semantics ---


def test_in_memory_ttl_expiry_and_refresh() -> None:
    store, clock = _in_memory()
    state = _state()
    store.save(state)

    clock.now += TTL_S - 1
    assert store.load(state.conversation_id) is not None  # still live

    store.save(state)  # refresh: sliding window restarts
    clock.now += TTL_S - 1
    assert store.load(state.conversation_id) is not None

    clock.now += 2
    assert store.load(state.conversation_id) is None  # expired


def test_redis_ttl_set_and_refreshed() -> None:
    store, client = _redis()
    state = _state()
    store.save(state)
    key = f"sutradhar:session:{state.conversation_id}"
    ttl = client.ttl(key)
    assert 0 < ttl <= TTL_S  # EX applied

    client.expire(key, 10)  # simulate an old, nearly-expired session
    store.save(state)
    assert client.ttl(key) > 10  # save refreshed the window


# --- The §2.2 convention: system prompt is NEVER stored (prompt_hash pins it) ---


def test_system_message_rejected_at_the_model() -> None:
    base = ConversationState.new().model_dump()
    base["messages"] = [{"role": "system", "content": "You are…"}]
    with pytest.raises(ValueError, match="system messages are never stored"):
        ConversationState.model_validate(base)


# --- Degrade, never crash ---


def test_corrupted_redis_payload_loads_as_none() -> None:
    store, client = _redis()
    state = _state()
    store.save(state)
    key = f"sutradhar:session:{state.conversation_id}"
    client.set(key, b"{not json", ex=TTL_S)
    assert store.load(state.conversation_id) is None  # fresh conversation, no 500

    client.set(key, b'{"unexpected": "shape"}', ex=TTL_S)
    assert store.load(state.conversation_id) is None


# --- Guardrail caps (surfaced by the orchestrator as structured client errors) ---


def test_turn_cap() -> None:
    state = _state(turns=MAX_TURNS)
    with pytest.raises(SessionLimitError, match="turn cap"):
        check_limits(state, "one more question")
    check_limits(_state(turns=MAX_TURNS - 1), "fine")  # under the cap passes


def test_message_size_cap() -> None:
    state = _state(turns=1)
    with pytest.raises(SessionLimitError, match="too large"):
        check_limits(state, "x" * (MAX_MESSAGE_BYTES + 1))
    # Byte-counted, not char-counted: multibyte scripts hit the cap honestly.
    with pytest.raises(SessionLimitError, match="too large"):
        check_limits(state, "த" * (MAX_MESSAGE_BYTES // 3 + 1))
    check_limits(state, "த" * 100)  # normal native-script message passes


def test_caps_overridable() -> None:
    check_limits(_state(turns=50), "hi", max_turns=100)
    with pytest.raises(SessionLimitError):
        check_limits(_state(turns=1), "hello", max_message_bytes=3)


# --- Misc contract details ---


def test_new_state_shape() -> None:
    state = ConversationState.new()
    assert state.messages == [] and state.turn_count == 0
    assert state.created_at == state.last_active
    assert state.created_at.tzinfo is not None  # tz-aware, comparable across processes


def test_clock_is_injectable_type() -> None:
    clock: Callable[[], float] = _Clock()
    InMemorySessionStore(10, clock=clock)  # constructor contract
