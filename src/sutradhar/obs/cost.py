"""Token/cost/latency accounting (P5 task 10, P5_SPEC §2.7, DEC-P5-6).

Cost model: the GPU is rented **by time, not by token** (DEC-0003: A100 @
``GPU_HOURLY_USD``), so a request's cost is its amortized share of GPU wall-clock —
``gpu_hourly_usd × latency_hours`` — not a per-token price. The **derived $/1k-token**
figure is still computed because Langfuse's cost dashboards need a custom model-price
definition for self-hosted models (its model-name catalog doesn't know gemma-4-E4B-it;
without the definition the dashboard renders $0 — the D6 setup step).

Null-honesty rule (the generation-run dry-run precedent): a request with no measured
latency or no token usage gets ``tokens_per_sec=None`` / ``cost_usd=None`` — mock or
absent timings must never masquerade as GPU numbers. ``0.0`` would be a lie; ``None`` is
honest.

The in-process :class:`MetricsAccumulator` backs ``GET /api/metrics`` — the JSON summary
side of the D6 dashboards (Langfuse custom dashboards are the visual side; a one-shot
vLLM ``/metrics`` snapshot is the serving-internals side, captured in the task-13 window).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestCost:
    """One request's accounting (attached to the trace + the ChatResponse)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float | None
    tokens_per_sec: float | None  # completion tokens / wall seconds
    cost_usd: float | None  # amortized GPU wall-clock share
    usd_per_1k_tokens: float | None  # derived — the Langfuse model-price input (D6)


def request_cost(
    usage: dict[str, int] | None,
    latency_ms: float | None,
    gpu_hourly_usd: float,
) -> RequestCost:
    """Amortized per-request cost from token usage + wall time (see module doc)."""
    prompt = int((usage or {}).get("prompt_tokens", 0))
    completion = int((usage or {}).get("completion_tokens", 0))
    total = prompt + completion

    if not latency_ms or latency_ms <= 0:
        return RequestCost(prompt, completion, total, None, None, None, None)

    seconds = latency_ms / 1000.0
    cost = gpu_hourly_usd * (seconds / 3600.0)
    tokens_per_sec = completion / seconds if completion else None
    usd_per_1k = (cost / total) * 1000.0 if total else None
    return RequestCost(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        latency_ms=round(latency_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
        cost_usd=round(cost, 8),
        usd_per_1k_tokens=round(usd_per_1k, 8) if usd_per_1k is not None else None,
    )


class MetricsAccumulator:
    """In-process chat-request counters for ``GET /api/metrics`` (P5_SPEC §2.2).

    Chat-scoped by design: the §2.2 contract names "requests, tokens, cost_usd,
    latency p50/p95, by status" — replay/status/health probes are not model traffic.
    Thread-safe; latencies bounded to the last ``max_samples`` (demo-window scale).
    """

    def __init__(self, *, max_samples: int = 1000, clock: Any = time.time) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._since = float(clock())
        self._by_status: Counter[str] = Counter()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0
        self._latencies: deque[float] = deque(maxlen=max_samples)

    def record(
        self,
        status: str,
        *,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
        cost_usd: float | None = None,
    ) -> None:
        with self._lock:
            self._by_status[status] += 1
            if usage:
                self._prompt_tokens += int(usage.get("prompt_tokens", 0))
                self._completion_tokens += int(usage.get("completion_tokens", 0))
            if cost_usd is not None:
                self._cost_usd += cost_usd
            if latency_ms is not None and latency_ms > 0:
                self._latencies.append(latency_ms)

    @staticmethod
    def _percentile(samples: list[float], q: float) -> float | None:
        if not samples:
            return None
        ordered = sorted(samples)
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return round(ordered[index], 2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._latencies)
            return {
                "since": self._since,
                "requests": {
                    "total": sum(self._by_status.values()),
                    "by_status": dict(self._by_status),
                },
                "tokens": {
                    "prompt": self._prompt_tokens,
                    "completion": self._completion_tokens,
                    "total": self._prompt_tokens + self._completion_tokens,
                },
                "cost_usd_total": round(self._cost_usd, 8),
                "latency_ms": {
                    "p50": self._percentile(samples, 0.50),
                    "p95": self._percentile(samples, 0.95),
                    "samples": len(samples),
                },
            }
