"""P5 task 10 — token/cost/latency accounting (P5_SPEC §2.7).

The null-honesty rule is the point: mock/absent timings must never produce numbers that
look like GPU measurements (the generation-run dry-run precedent).
"""

from __future__ import annotations

import pytest

from sutradhar.obs.cost import MetricsAccumulator, request_cost


class TestRequestCost:
    def test_amortized_gpu_wall_clock_math(self) -> None:
        # 60 s of a $0.89/h A100 = $0.014833…; 300 completion tokens / 60 s = 5 tok/s.
        cost = request_cost(
            {"prompt_tokens": 700, "completion_tokens": 300}, 60_000.0, gpu_hourly_usd=0.89
        )
        assert cost.total_tokens == 1000
        assert cost.cost_usd == pytest.approx(0.89 / 60, abs=1e-8)
        assert cost.tokens_per_sec == pytest.approx(5.0)
        # Derived $/1k-token — the Langfuse custom model-price input (D6).
        assert cost.usd_per_1k_tokens == pytest.approx(cost.cost_usd, abs=1e-8)  # 1000 tokens

    def test_rate_is_env_driven_not_hardcoded(self) -> None:
        a = request_cost({"completion_tokens": 10}, 1000.0, gpu_hourly_usd=0.89)
        b = request_cost({"completion_tokens": 10}, 1000.0, gpu_hourly_usd=1.78)
        assert a.cost_usd is not None and b.cost_usd == pytest.approx(a.cost_usd * 2)

    def test_no_latency_yields_null_gpu_numbers(self) -> None:
        for latency in (None, 0, -5.0):
            cost = request_cost({"prompt_tokens": 10, "completion_tokens": 5}, latency, 0.89)
            assert cost.tokens_per_sec is None
            assert cost.cost_usd is None
            assert cost.usd_per_1k_tokens is None
            assert cost.total_tokens == 15  # token counts are still honest facts

    def test_zero_tokens_yields_null_rates_but_real_cost(self) -> None:
        # Wall time was spent even if the model emitted nothing — cost is real,
        # token-derived rates are not.
        cost = request_cost({}, 2000.0, 0.89)
        assert cost.cost_usd == pytest.approx(0.89 * 2 / 3600, abs=1e-8)
        assert cost.tokens_per_sec is None
        assert cost.usd_per_1k_tokens is None


class TestMetricsAccumulator:
    def test_by_status_and_sums(self) -> None:
        m = MetricsAccumulator()
        m.record(
            "up",
            latency_ms=100.0,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost_usd=0.001,
        )
        m.record(
            "up",
            latency_ms=300.0,
            usage={"prompt_tokens": 20, "completion_tokens": 10},
            cost_usd=0.002,
        )
        m.record("off")
        m.record("limit")
        snap = m.snapshot()
        assert snap["requests"]["total"] == 4
        assert snap["requests"]["by_status"] == {"up": 2, "off": 1, "limit": 1}
        assert snap["tokens"] == {"prompt": 30, "completion": 15, "total": 45}
        assert snap["cost_usd_total"] == pytest.approx(0.003)
        assert snap["latency_ms"]["samples"] == 2

    def test_percentiles_on_known_distribution(self) -> None:
        m = MetricsAccumulator()
        for latency in range(1, 101):  # 1..100 ms
            m.record("up", latency_ms=float(latency))
        snap = m.snapshot()
        assert snap["latency_ms"]["p50"] == pytest.approx(50.0, abs=1.0)
        assert snap["latency_ms"]["p95"] == pytest.approx(95.0, abs=1.0)

    def test_empty_snapshot_shape(self) -> None:
        snap = MetricsAccumulator().snapshot()
        assert snap["requests"] == {"total": 0, "by_status": {}}
        assert snap["tokens"] == {"prompt": 0, "completion": 0, "total": 0}
        assert snap["cost_usd_total"] == 0.0
        assert snap["latency_ms"] == {"p50": None, "p95": None, "samples": 0}

    def test_latency_samples_bounded(self) -> None:
        m = MetricsAccumulator(max_samples=10)
        for latency in range(1, 101):
            m.record("up", latency_ms=float(latency))
        snap = m.snapshot()
        assert snap["latency_ms"]["samples"] == 10
        assert snap["latency_ms"]["p50"] == pytest.approx(95.0, abs=1.0)  # last 10 kept
        assert snap["requests"]["total"] == 100  # counts are NOT bounded, only samples
