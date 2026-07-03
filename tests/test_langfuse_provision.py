"""DEC-P3-7 bootstrap idempotency tests (P3 task 10; P3_SPEC §4 test_langfuse_provision).

Fake AIC API (httpx.MockTransport) + fake SSH transcript — no network, no spend, ever.
Covers: fresh-state runs every step in order; re-run against a configured box is a no-op;
partial state resumes at the right step; instance-exists never calls checkout; destructive
paths require the explicit flag; the Docker-in-LXC nesting gate escalates, not retries.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_PROVISION_PATH = Path(__file__).resolve().parents[1] / "infra" / "langfuse" / "provision.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("langfuse_provision_under_test", _PROVISION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provision = _load()

# --- Fake AIC API ---


class _FakeAic:
    """Scriptable AIC API behind httpx.MockTransport; records every call path."""

    def __init__(
        self,
        vps: list[dict[str, Any]] | None = None,
        balance_paise: int = 100_000,
    ) -> None:
        self.vps = vps or []
        self.balance_paise = balance_paise
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        assert request.headers["authorization"] == "Bearer test-key"
        if path == "/api/v1/vps" and request.method == "GET":
            return httpx.Response(200, json=self.vps)
        if path.endswith("/start"):
            return httpx.Response(200, json={"ok": True})
        if path == "/api/v1/billing/wallet":
            return httpx.Response(200, json={"balance": self.balance_paise})
        if path == "/api/v1/public/essential-vps-plans":
            return httpx.Response(
                200,
                json=[
                    {"slug": "essential-4gb", "price_paise": 39_900},
                    {"slug": "essential-8gb", "price_paise": 79_900},
                ],
            )
        if path == "/api/v1/vps/checkout":
            return httpx.Response(200, json={"orderId": "order_abc"})
        if path == "/api/v1/vps/checkout/verify":
            return httpx.Response(
                200,
                json={"id": 77, "ssh": {"host": "1.2.3.4", "port": 22, "user": "root"}},
            )
        return httpx.Response(404)

    def api(self) -> Any:
        return provision.AicApi(
            api_key="test-key",
            client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


# --- Fake SSH transcript ---


class _FakeSsh:
    """Simulates box state: checks consult `satisfied`; acts mutate it."""

    def __init__(self, satisfied: set[str] | None = None, fail_acts: set[str] | None = None):
        self.satisfied = satisfied if satisfied is not None else set()
        self.fail_acts = fail_acts or set()
        self.transcript: list[str] = []
        self._steps = {s.name: s for s in provision.bootstrap_steps("1.2.3.4.sslip.io")}

    def run(self, command: str) -> tuple[int, str]:
        self.transcript.append(command)
        for name, step in self._steps.items():
            if command == step.check:
                return (0, "") if name in self.satisfied else (1, "")
            if command in step.act:
                if name in self.fail_acts:
                    return 1, f"simulated failure in {name}"
                self.satisfied.add(name)
                return 0, "done"
        if "LANGFUSE_INIT_PROJECT" in command:  # final key read-back
            return (
                0,
                "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-x\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-y",
            )
        if "docker compose down -v" in command or command.startswith("rm -rf"):
            self.satisfied.clear()
            return 0, "wiped"
        return 0, ""


_ALL_STEPS = [s.name for s in provision.bootstrap_steps("d")]


# --- Phase 1: find-or-create ---


def test_existing_running_instance_never_calls_checkout() -> None:
    fake = _FakeAic(vps=[{"id": 1, "name": provision.INSTANCE_NAME, "status": "running"}])
    outcome = provision.ensure_instance(
        fake.api(), ssh_public_keys=["k"], payment_prompt=lambda o: None, log=lambda m: None
    )
    assert outcome.status == "running"
    assert not any("checkout" in c for c in fake.calls)
    assert not any("wallet" in c for c in fake.calls)  # no spend paths touched at all


def test_stopped_instance_is_started_not_recreated() -> None:
    fake = _FakeAic(vps=[{"id": 5, "name": provision.INSTANCE_NAME, "status": "stopped"}])
    outcome = provision.ensure_instance(
        fake.api(), ssh_public_keys=["k"], payment_prompt=lambda o: None, log=lambda m: None
    )
    assert outcome.status == "started"
    assert "POST /api/v1/vps/5/start" in fake.calls
    assert not any("checkout" in c for c in fake.calls)


def test_absent_instance_checks_wallet_then_checkout_with_ssh_keys() -> None:
    fake = _FakeAic(balance_paise=100_000)
    seen: dict[str, Any] = {}

    def pay(order: dict[str, Any]) -> dict[str, Any]:
        seen["order"] = order
        return {
            "razorpay_order_id": order["orderId"],
            "razorpay_payment_id": "p",
            "razorpay_signature": "s",
        }

    outcome = provision.ensure_instance(
        fake.api(), ssh_public_keys=["ssh-ed25519 AAA"], payment_prompt=pay, log=lambda m: None
    )
    assert outcome.status == "created"
    assert seen["order"]["orderId"] == "order_abc"
    assert outcome.instance is not None and outcome.instance["ssh"]["host"] == "1.2.3.4"
    # Plan resolved from the live catalogue, never hardcoded ids.
    assert "GET /api/v1/public/essential-vps-plans" in fake.calls


def test_insufficient_wallet_stops_before_checkout() -> None:
    fake = _FakeAic(balance_paise=10_000)  # ₹100 < ₹799 + fee
    outcome = provision.ensure_instance(
        fake.api(), ssh_public_keys=["k"], payment_prompt=lambda o: None, log=lambda m: None
    )
    assert outcome.status == "needs-topup"
    assert not any("checkout" in c for c in fake.calls)


def test_browser_only_payment_can_abort_and_resume() -> None:
    fake = _FakeAic()
    outcome = provision.ensure_instance(
        fake.api(), ssh_public_keys=["k"], payment_prompt=lambda o: None, log=lambda m: None
    )
    assert outcome.status == "awaiting-payment"  # safe stop; re-run resumes


# --- Phase 2: check-then-act idempotency ---


def test_fresh_state_executes_every_step_in_order() -> None:
    ssh = _FakeSsh()
    report = provision.run_bootstrap(ssh, "1.2.3.4.sslip.io", log=lambda m: None)
    assert [name for name, _ in report.results] == _ALL_STEPS
    assert all(outcome == "configured" for _, outcome in report.results)
    assert ssh.satisfied == set(_ALL_STEPS)


def test_rerun_on_configured_box_is_pure_noop() -> None:
    ssh = _FakeSsh(satisfied=set(_ALL_STEPS))
    report = provision.run_bootstrap(ssh, "1.2.3.4.sslip.io", log=lambda m: None)
    assert all(outcome == "already" for _, outcome in report.results)
    # Only the checks (and the final key read-back) ran — zero act commands.
    acts = {cmd for step in provision.bootstrap_steps("1.2.3.4.sslip.io") for cmd in step.act}
    assert not acts & set(ssh.transcript)


def test_partial_state_resumes_at_the_right_step() -> None:
    """Docker installed, Langfuse absent — the exact P3_SPEC §4 scenario."""
    ssh = _FakeSsh(satisfied={"swap-4g", "docker-installed", "docker-lxc-nesting"})
    report = provision.run_bootstrap(ssh, "1.2.3.4.sslip.io", log=lambda m: None)
    outcomes = dict(report.results)
    assert outcomes["swap-4g"] == "already"
    assert outcomes["docker-installed"] == "already"
    assert outcomes["docker-lxc-nesting"] == "already"
    assert outcomes["langfuse-cloned-pinned"] == "configured"
    assert outcomes["https-health"] == "configured"


def test_lxc_nesting_gate_escalates_never_retries() -> None:
    ssh = _FakeSsh(satisfied={"swap-4g", "docker-installed"}, fail_acts={"docker-lxc-nesting"})
    with pytest.raises(provision.BootstrapBlockedError, match="escalate to AIC support"):
        provision.run_bootstrap(ssh, "1.2.3.4.sslip.io", log=lambda m: None)


def test_destructive_wipe_requires_explicit_recreate_flag() -> None:
    ssh = _FakeSsh(satisfied=set(_ALL_STEPS))
    provision.run_bootstrap(ssh, "1.2.3.4.sslip.io", log=lambda m: None)
    assert not any("down -v" in c for c in ssh.transcript)  # never destructive by default
    ssh2 = _FakeSsh(satisfied=set(_ALL_STEPS))
    provision.run_bootstrap(ssh2, "1.2.3.4.sslip.io", recreate=True, log=lambda m: None)
    assert any("down -v" in c for c in ssh2.transcript)


# --- Hardening invariants on the step plan itself ---


def test_step_plan_pins_tag_and_hardening() -> None:
    steps = {s.name: s for s in provision.bootstrap_steps("langfuse.example.com")}
    assert provision.LANGFUSE_TAG in " ".join(steps["langfuse-cloned-pinned"].act)
    assert "AUTH_DISABLE_SIGNUP=true" in " ".join(steps["signup-disabled"].act)
    assert "reverse_proxy localhost:3000" in " ".join(steps["caddy-tls-443"].act)
    assert "langfuse.example.com" in steps["https-health"].check
    assert steps["docker-lxc-nesting"].gate and steps["https-health"].gate
    # Secrets are generated once and only when absent (check-then-act).
    assert "CHANGEME" in steps["secrets-once"].check
    assert "openssl rand" in " ".join(steps["secrets-once"].act)
