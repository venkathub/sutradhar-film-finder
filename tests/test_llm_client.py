"""Unit tests for LLMClient.health() — up / off / error paths (P0 task 6).

The whole HTTP surface (raw /health + the OpenAI SDK's /v1/*) is mocked through a single
injected httpx.MockTransport, so no network or model is touched (Tier-1-safe).
"""

from __future__ import annotations

import json

import httpx

from sutradhar.config import Settings
from sutradhar.serving import EndpointStatus, LLMClient

_MODEL = "google/gemma-4-E4B"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_BASE_URL="http://localhost:8000/v1",
        LLM_MODEL=_MODEL,
        LLM_API_KEY="EMPTY",
    )


def _client(handler: object) -> LLMClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LLMClient(_settings(), http_client=httpx.Client(transport=transport))


def _models_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"object": "list", "data": [{"id": _MODEL, "object": "model"}]},
    )


def _completion_response(model: str, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "length",
                }
            ],
        },
    )


def _up_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/health"):
        return httpx.Response(200, text="")
    if path.endswith("/v1/models"):
        return _models_response()
    if path.endswith("/chat/completions"):
        return _completion_response(_MODEL, "pong")
    return httpx.Response(404)


def test_health_up() -> None:
    status = _client(_up_handler).health()
    assert isinstance(status, EndpointStatus)
    assert status.status == "up"
    assert status.model == _MODEL
    assert status.sample_token == "pong"
    assert isinstance(status.latency_ms, float)
    assert status.latency_ms >= 0


def test_health_off_connection_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    status = _client(handler).health()
    assert status.status == "off"
    assert status.sample_token is None
    assert status.latency_ms is None
    assert "endpoint OFF" in status.detail


def test_health_off_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    status = _client(handler).health()
    assert status.status == "off"


def test_health_error_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="")
        if request.url.path.endswith("/v1/models"):
            return _models_response()
        return httpx.Response(500, json={"error": "boom"})

    status = _client(handler).health()
    assert status.status == "error"
    assert status.sample_token is None


def test_health_error_on_health_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    status = _client(handler).health()
    assert status.status == "error"
    assert "503" in status.detail


def test_model_mismatch_still_up_but_noted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, text="")
        if path.endswith("/v1/models"):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "other/model", "object": "model"}]},
            )
        return _completion_response("other/model", "x")

    status = _client(handler).health()
    assert status.status == "up"
    assert "!=" in status.detail


def test_status_serializes_without_secret() -> None:
    status = _client(_up_handler).health()
    payload = json.dumps(status.to_dict())
    assert "EMPTY" not in payload  # api key never leaks into the status
    assert set(status.to_dict()) == {"status", "model", "sample_token", "latency_ms", "detail"}
