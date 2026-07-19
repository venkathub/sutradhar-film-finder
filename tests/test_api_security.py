"""P7 task 4 (DEC-P7-2) — auth + rate limiting on the paid path.

Contract under test:
- GPU-off (the default state): unauthenticated requests still get the structured
  offline payload — the degradation surface is the always-available portfolio face.
- GPU-up: bearer auth required. No tokens configured => 503 ``auth_not_configured``
  (never silently open); bad/missing token => 401; valid token => the turn runs.
- ``CHAT_AUTH=disabled`` is an explicit local/e2e opt-out.
- Rate limits key by auth token first (hashed), client IP as fallback;
  ``X-Forwarded-For`` is honored only when ``TRUST_PROXY=1``.

Tier-1: no DB, no GPU, no Redis (memory:// limit storage) — everything injected.
"""

from __future__ import annotations

from tests.test_api import (
    OFF,
    TEST_TOKEN,
    TURN1,
    TURN2,
    _client,
    _ScriptedModel,
    _settings,
)


def _chat(client, message: str = "papanasam?", **kwargs):
    return client.post("/api/chat", json={"message": message}, **kwargs)


# --- auth ---


def test_gpu_off_needs_no_token() -> None:
    client = _client(probe_status=OFF, authenticated=False)
    resp = _chat(client)
    assert resp.status_code == 200
    assert resp.json()["status"] == "off"


def test_up_path_without_token_is_401() -> None:
    client = _client(llm_handler=_ScriptedModel(list(TURN1)), authenticated=False)
    resp = _chat(client)
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"] == "unauthorized" and body["request_id"]
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_up_path_with_wrong_token_is_401() -> None:
    client = _client(llm_handler=_ScriptedModel(list(TURN1)), authenticated=False)
    resp = _chat(client, headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_up_path_with_valid_token_runs_the_turn() -> None:
    client = _client(llm_handler=_ScriptedModel(list(TURN1)))
    resp = _chat(client)
    assert resp.status_code == 200
    assert resp.json()["status"] == "up"


def test_unconfigured_auth_is_503_never_open() -> None:
    """DEC-P7-2: auth required (default) + no tokens => a refusal, not an open path."""
    client = _client(
        llm_handler=_ScriptedModel(list(TURN1)),
        settings=_settings(API_AUTH_TOKENS=""),
        authenticated=False,
    )
    resp = _chat(client)
    assert resp.status_code == 503
    assert resp.json()["error"] == "auth_not_configured"
    # Even presenting a token cannot open an unconfigured endpoint.
    resp = _chat(client, headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert resp.status_code == 503


def test_chat_auth_disabled_is_an_explicit_opt_out() -> None:
    client = _client(
        llm_handler=_ScriptedModel(list(TURN1)),
        settings=_settings(CHAT_AUTH="disabled", API_AUTH_TOKENS=""),
        authenticated=False,
    )
    assert _chat(client).status_code == 200


def test_second_configured_token_also_works() -> None:
    client = _client(
        llm_handler=_ScriptedModel(list(TURN1)),
        settings=_settings(API_AUTH_TOKENS=f"{TEST_TOKEN}, interviewer-token"),
        authenticated=False,
    )
    resp = _chat(client, headers={"Authorization": "Bearer interviewer-token"})
    assert resp.status_code == 200


def test_health_and_status_and_replays_stay_open() -> None:
    client = _client(probe_status=OFF, authenticated=False)
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/replays").status_code == 200


# --- rate limiting (memory:// storage; fresh app per test => fresh counters) ---


def _limited_client(**settings_overrides):
    script = list(TURN1) + list(TURN2) + list(TURN1) + list(TURN2)
    return _client(
        llm_handler=_ScriptedModel(script),
        settings=_settings(CHAT_RATE_LIMIT="2/minute", **settings_overrides),
    )


def test_rate_limit_trips_with_envelope_and_request_id() -> None:
    client = _limited_client()
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 200
    resp = _chat(client)
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"] == "rate_limited" and body["request_id"]
    assert "2 per 1 minute" in body["detail"]
    assert resp.headers["Retry-After"]
    assert resp.headers["X-Request-Id"] == body["request_id"]


def test_rate_limit_keys_by_token_not_ip() -> None:
    """Two tokens from the same client IP get separate buckets (token-first keying)."""
    client = _limited_client(API_AUTH_TOKENS=f"{TEST_TOKEN},other-token")
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 429  # TEST_TOKEN bucket exhausted
    resp = _chat(client, headers={"Authorization": "Bearer other-token"})
    assert resp.status_code == 200  # fresh bucket for the second token


def test_rate_limit_applies_before_auth_for_anonymous_hammering() -> None:
    """Unauthenticated requests are limited by IP — 401s cannot be farmed freely."""
    client = _limited_client()
    client.headers.pop("Authorization")
    assert _chat(client).status_code == 401
    assert _chat(client).status_code == 401
    assert _chat(client).status_code == 429


def test_rotating_garbage_tokens_cannot_mint_fresh_buckets() -> None:
    """PR #9 blocking finding 1: only a VALID token earns its own bucket. Rotating
    invalid bearer tokens must share the client-IP bucket — otherwise token
    brute-force is unthrottled and the limiter key-space grows per request."""
    client = _limited_client()
    client.headers.pop("Authorization")
    for n in (1, 2):
        resp = _chat(client, headers={"Authorization": f"Bearer garbage-{n}"})
        assert resp.status_code == 401
    # Third request with yet another fresh garbage token: SAME IP bucket => 429.
    resp = _chat(client, headers={"Authorization": "Bearer garbage-3"})
    assert resp.status_code == 429
    # And a VALID token still has its own untouched bucket.
    assert _chat(client, headers={"Authorization": f"Bearer {TEST_TOKEN}"}).status_code == 200


def test_x_forwarded_for_ignored_unless_trust_proxy() -> None:
    """A spoofable header must not let a client mint fresh IP buckets."""
    client = _limited_client()  # TRUST_PROXY defaults to False
    client.headers.pop("Authorization")
    xff_a = {"X-Forwarded-For": "203.0.113.1"}
    xff_b = {"X-Forwarded-For": "203.0.113.2"}
    assert _chat(client, headers=xff_a).status_code == 401
    assert _chat(client, headers=xff_b).status_code == 401
    # Third request still 429s: both spoofed IPs shared the real-client bucket.
    assert _chat(client, headers=xff_a).status_code == 429


def test_x_forwarded_for_honored_behind_trusted_proxy() -> None:
    client = _limited_client(TRUST_PROXY="1")
    client.headers.pop("Authorization")
    xff_a = {"X-Forwarded-For": "203.0.113.1"}
    xff_b = {"X-Forwarded-For": "203.0.113.2"}
    assert _chat(client, headers=xff_a).status_code == 401
    assert _chat(client, headers=xff_a).status_code == 401
    assert _chat(client, headers=xff_a).status_code == 429  # per-forwarded-IP bucket full
    assert _chat(client, headers=xff_b).status_code == 401  # different IP, fresh bucket
