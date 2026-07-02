"""Thin, no-op-safe Langfuse tracing wrapper (P3 task 10; DEC-P3-6 option A).

One explicit seam — ``Tracer.span()`` context managers — around exactly four chokepoints:
the driver's fixture loop, each ``chat()`` round, each tool execution, and judge calls.
P5's FastAPI middleware reuses this same wrapper; no ``@observe`` decorator magic, no
SDK import at module import time.

No-op guarantees (test-enforced):
- ``LANGFUSE_*`` unset → ``Tracer.enabled`` is False and every ``span()`` yields a no-op
  handle — zero SDK import, zero network, zero behaviour change (Tier-1 CI / forks).
- The Langfuse client is injectable (``client=``) so tests use a fake sink; the real
  client is built lazily only when all three keys are present (self-hosted instance per
  DEC-P3-7; ``LANGFUSE_HOST`` env-driven, never hardcoded).

Trace export (evidence longevity, DEC-P3-7): benchmark-cited traces are additionally
exported as JSON (:func:`export_trace`) and committed with the run artifact, so standing
evidence never depends on VPS uptime.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from sutradhar.config import Settings

SpanKind = str  # langfuse as_type: "span" | "generation" | "tool" | "agent" | "evaluator" | …


class _NoopSpan:
    """Handle returned when tracing is off — accepts updates, does nothing."""

    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        return None


_NOOP_SPAN = _NoopSpan()


class Tracer:
    """Explicit span seam over the (injectable) Langfuse client. Safe when disabled."""

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self._client = client
        self.last_trace_id: str | None = None
        if client is None and settings is not None and _keys_present(settings):
            from langfuse import Langfuse  # lazy: only imported when tracing is ON

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        input: Any = None,  # noqa: A002 — mirrors the langfuse parameter name
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Yield a span handle (``.update(output=…)``-able). No-op when disabled."""
        if self._client is None:
            yield _NOOP_SPAN
            return
        with self._client.start_as_current_observation(
            name=name, as_type=kind, input=input, metadata=metadata
        ) as span:
            trace_id = getattr(span, "trace_id", None)
            if trace_id is not None:
                self.last_trace_id = str(trace_id)
            yield span

    def trace_url(self) -> str | None:
        if self._client is None or self.last_trace_id is None:
            return None
        url = self._client.get_trace_url(trace_id=self.last_trace_id)
        return str(url) if url else None

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


def _keys_present(settings: Settings) -> bool:
    return bool(
        settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host
    )


def export_trace(
    trace_id: str,
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch one trace (with observations) from the Langfuse API for committed evidence.

    Uses the public API with basic auth (public key : secret key). The returned JSON is
    committed alongside benchmark artifacts so trace evidence outlives the VPS.
    """
    if not _keys_present(settings):
        raise ValueError("Langfuse keys unset — cannot export a trace (set LANGFUSE_*)")
    host = str(settings.langfuse_host).rstrip("/")
    client = http_client or httpx.Client(timeout=30.0)
    response = client.get(
        f"{host}/api/public/traces/{trace_id}",
        auth=(str(settings.langfuse_public_key), str(settings.langfuse_secret_key)),
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload
