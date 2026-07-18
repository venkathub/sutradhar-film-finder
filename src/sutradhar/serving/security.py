"""Bearer-token auth + token-first rate-limit keying for the paid path.

P7 task 4 (DEC-P7-2): ``POST /api/chat`` is the endpoint that burns GPU seconds —
it is never open when the GPU is up. Design:

- **Auth**: static bearer tokens from ``API_AUTH_TOKENS`` (comma-separated env;
  per-person tokens, revocation = removal). No JWT/OAuth machinery — a portfolio
  demo endpoint does not earn it (DEC-P5-1 rationale, again).
- **Never silently open**: with auth required (the default) and no tokens
  configured, the up-path returns 503 ``auth_not_configured`` — a deliberate
  refusal, not an open endpoint. Local/e2e stacks opt out *explicitly* via
  ``CHAT_AUTH=disabled``.
- **Degradation stays open**: the GPU-off path (offline payload, replays, health,
  static UI) is the always-available surface and needs no token — it costs
  nothing and is the portfolio's public face.
- **Rate-limit keying**: by auth token first (sha256 — the raw token never becomes
  a storage key), client IP as fallback. Per-IP alone is weak behind NAT/proxies;
  ``X-Forwarded-For`` is honored only when ``TRUST_PROXY=1`` (a spoofable header
  must be opt-in, only when a trusted proxy sets it).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from sutradhar.config import Settings


def parse_tokens(raw: str | None) -> frozenset[str]:
    """``API_AUTH_TOKENS`` is comma-separated; unset/blanks are ignored."""
    return frozenset(token.strip() for token in (raw or "").split(",") if token.strip())


def bearer_token(request: Request) -> str | None:
    """The ``Authorization: Bearer <token>`` value, or None."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


def client_ip(request: Request, *, trust_proxy: bool) -> str:
    """Client IP for limit keying; first ``X-Forwarded-For`` hop only when trusted."""
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def make_rate_limit_key(settings: Settings) -> Callable[[Request], str]:
    """slowapi key function: token-first (hashed), IP fallback (DEC-P7-2)."""

    def key(request: Request) -> str:
        token = bearer_token(request)
        if token:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
            return f"tok:{digest}"
        return f"ip:{client_ip(request, trust_proxy=settings.trust_proxy)}"

    return key


def authorize_chat(request: Request, settings: Settings) -> JSONResponse | None:
    """Return None when authorized, else the structured denial response.

    The caller applies this on the GPU-*up* path only: the off/degraded path is
    the always-available surface and stays unauthenticated by design.
    """
    if settings.chat_auth == "disabled":
        return None  # explicit local/e2e opt-out — never the default
    request_id = getattr(request.state, "request_id", "")
    tokens = parse_tokens(settings.api_auth_tokens)
    if not tokens:
        return JSONResponse(
            {
                "error": "auth_not_configured",
                "detail": "live chat requires auth but API_AUTH_TOKENS is not set — "
                "the paid path is never silently open (DEC-P7-2)",
                "request_id": request_id,
            },
            status_code=503,
        )
    if bearer_token(request) not in tokens:
        return JSONResponse(
            {
                "error": "unauthorized",
                "detail": "missing or invalid bearer token",
                "request_id": request_id,
            },
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None
