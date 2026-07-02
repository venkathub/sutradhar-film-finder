"""OpenAI-compatible LLM client + connectivity smoke (P0 task 6, DEC-P0-4).

The endpoint is an on-demand vLLM instance reached via the env-driven ``LLM_BASE_URL``
(JarvisLabs instance URL or an SSH-tunnelled ``localhost:8000/v1``) — normally paused.
:meth:`LLMClient.health` therefore treats "endpoint OFF" as a first-class success path
(``status="off"``, never an exception): the seed of "graceful degradation as a feature".

Probe sequence (P0_SPEC §2.3, web-verified vLLM routes §2.7):
  1. ``GET {base%/v1}/health``  → liveness (vLLM returns 200, empty body when up).
  2. ``GET /v1/models``         → confirm the served id matches ``LLM_MODEL``.
  3. ``POST /v1/chat/completions`` (``max_tokens=1``) → capture sample_token + latency_ms.

Connection-refused/timeout at any step → ``status="off"``. HTTP 5xx / malformed body →
``status="error"`` (distinct from off). Only genuinely unexpected errors propagate.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import httpx
import openai
from openai import OpenAI

from sutradhar.config import Settings

Status = Literal["up", "off", "error"]

_OFF_DETAIL = "endpoint OFF — bring up the on-demand GPU (see infra/README.md), then retry."


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON-object parse of tool-call arguments (malformed/non-object → None)."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# Errors that mean "the endpoint is not answering" → status "off" (a paused GPU, not a bug).
_OFF_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    openai.APIConnectionError,  # wraps the httpx connection/timeout errors
    openai.APITimeoutError,
)
# Errors that mean "the endpoint answered, but badly" → status "error" (distinct from off).
_ERROR_ERRORS: tuple[type[Exception], ...] = (
    openai.APIStatusError,
    openai.APIResponseValidationError,
    KeyError,
    IndexError,
    ValueError,
)


@dataclass(frozen=True)
class EndpointStatus:
    """Structured result of a connectivity probe (P0_SPEC §2.2).

    Reused by P5 graceful-degradation and P3 tracing hooks. Carries no secret.
    """

    status: Status
    model: str | None
    sample_token: str | None
    latency_ms: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    """One model-emitted tool call, parsed from the OpenAI wire format (P3 task 2).

    ``arguments`` is the best-effort JSON parse of ``arguments_raw``; ``None`` when the model
    emitted malformed JSON. Malformed arguments are a *scored* failure for the eval driver
    (schema-validity metric, P3_SPEC §2.4) — never an exception here.
    """

    id: str
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChatResult:
    """Structured result of one chat round-trip (P3_SPEC §1.3).

    Same status contract as :class:`EndpointStatus` (DEC-P0-4): ``off`` = endpoint not
    answering (paused GPU — first-class, never an exception), ``error`` = answered badly.
    ``message`` is the raw OpenAI-wire assistant message dict, so the eval driver can append
    it verbatim to the next turn's ``messages`` and record it in the transcript.
    ``usage`` carries prompt/completion/total token counts (the Table 2 tokens/sec source).
    """

    status: Status
    message: dict[str, Any] | None
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    usage: dict[str, int] | None
    latency_ms: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient:
    """Thin OpenAI-compatible client over ``LLM_BASE_URL`` (constructed from Settings)."""

    def __init__(self, settings: Settings, *, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        base = settings.llm_base_url.rstrip("/")
        # Liveness route lives at the server root, not under /v1 (vLLM-specific).
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        self._health_url = f"{root}/health"
        # Share one httpx.Client between the raw /health GET and the OpenAI SDK, so a single
        # transport mock covers the whole surface in tests ("mock the client transport", §4).
        self._http = http_client or httpx.Client(timeout=settings.llm_timeout_s)
        self._openai = OpenAI(
            base_url=base,
            api_key=settings.llm_api_key or "EMPTY",
            timeout=settings.llm_timeout_s,
            max_retries=0,
            http_client=self._http,
        )

    @property
    def model(self) -> str:
        return self._settings.llm_model

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Chat completion round-trip (health probe uses 1 token; P1 extraction passes
        ``temperature=0`` + vLLM ``guided_json`` via ``extra_body`` for schema-forced output)."""
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        resp = self._openai.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        """OpenAI tool-calling round-trip (P3 task 2; P3_SPEC §1.3).

        Sends the full ``messages`` history (system / user / assistant / tool roles) plus the
        ``tools`` array — generated from ``tool_schema.v0.json`` by the eval driver, never
        hand-written (P3_SPEC §2.8). Preserves the DEC-P0-4 contract: a paused endpoint
        returns ``status="off"``, a misbehaving one ``status="error"`` — never an exception
        for either. Tool-call arguments are parsed best-effort; malformed JSON yields
        ``ToolCall.arguments=None`` for the driver to score, not a crash.
        """
        kwargs: dict[str, Any] = {}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        try:
            start = time.perf_counter()
            resp = self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,  # type: ignore[arg-type]  # plain-dict wire format
                **kwargs,
            )
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            choice = resp.choices[0]
            raw_message = choice.message.model_dump(exclude_none=True)
            tool_calls = tuple(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_raw=tc.function.arguments,
                    arguments=_parse_arguments(tc.function.arguments),
                )
                for tc in choice.message.tool_calls or []
                if tc.type == "function"
            )
            usage: dict[str, int] | None = None
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            return ChatResult(
                status="up",
                message=raw_message,
                content=choice.message.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage=usage,
                latency_ms=latency_ms,
                detail="ok",
            )
        except _OFF_ERRORS as exc:
            return ChatResult(
                status="off",
                message=None,
                content=None,
                tool_calls=(),
                finish_reason=None,
                usage=None,
                latency_ms=None,
                detail=f"{_OFF_DETAIL} ({type(exc).__name__})",
            )
        except _ERROR_ERRORS as exc:
            return ChatResult(
                status="error",
                message=None,
                content=None,
                tool_calls=(),
                finish_reason=None,
                usage=None,
                latency_ms=None,
                detail=f"endpoint reachable but errored: {type(exc).__name__}: {exc}",
            )

    def health(self) -> EndpointStatus:
        """Probe the endpoint. Never raises for a down endpoint (returns ``status="off"``)."""
        expected = self._settings.llm_model
        try:
            # (1) liveness
            resp = self._http.get(self._health_url)
            if resp.status_code >= 500:
                return EndpointStatus(
                    status="error",
                    model=expected,
                    sample_token=None,
                    latency_ms=None,
                    detail=f"/health returned {resp.status_code}",
                )

            # (2) confirm served model id
            reported = self._reported_model(default=expected)
            detail = "endpoint UP"
            if reported != expected:
                detail = f"endpoint UP — served model {reported!r} != LLM_MODEL {expected!r}"

            # (3) one-token completion → sample_token + latency
            start = time.perf_counter()
            token = self.complete("ping", max_tokens=1)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return EndpointStatus(
                status="up",
                model=reported,
                sample_token=token,
                latency_ms=round(latency_ms, 2),
                detail=detail,
            )
        except _OFF_ERRORS as exc:
            return EndpointStatus(
                status="off",
                model=expected,
                sample_token=None,
                latency_ms=None,
                detail=f"{_OFF_DETAIL} ({type(exc).__name__})",
            )
        except _ERROR_ERRORS as exc:
            return EndpointStatus(
                status="error",
                model=expected,
                sample_token=None,
                latency_ms=None,
                detail=f"endpoint reachable but errored: {type(exc).__name__}: {exc}",
            )

    def _reported_model(self, *, default: str) -> str:
        """Return the first served model id from ``GET /v1/models`` (best-effort)."""
        listing = self._openai.models.list()
        data = getattr(listing, "data", None) or []
        if data:
            return str(data[0].id)
        return default
