"""Idempotent from-scratch Langfuse bootstrap on AIC Cloud (P3 task 10; DEC-P3-7).

``make langfuse-up`` must set up Langfuse from scratch if not installed, be safe to
re-run, and converge from ANY intermediate state:

- **Phase 1 — instance (AIC API, find-or-create):** locate ``sutradhar-obs-01``
  (running → skip to phase 2; stopped → start; absent → wallet pre-check → resolve the
  pinned ``essential-8gb`` plan from the live catalogue (never a hardcoded id) →
  checkout. The Razorpay payment legs are browser-only BY DESIGN: the script prepares
  the order, prints instructions, and waits for the payment confirmation input.
- **Phase 2 — configuration (SSH, check-then-act):** every step probes before acting,
  so an already-satisfied step is skipped and a partial bootstrap resumes exactly where
  it stopped: swap → Docker → Docker-in-LXC nesting gate (Essential VPS is LXC; if
  nesting is blocked we STOP with an escalation message, per DEC-P3-7 caveat a) →
  pinned-tag clone → secrets generated ONCE (headless init with pinned project keys) →
  compose up → tunnel retirement (the P3 cloudflared workaround, superseded 2026-07-11
  by the AIC platform public URL: ``<slug>-3000.aiccloud.online`` with edge TLS) →
  ufw hardening → signup disabled → backup cron → HTTPS health through the public URL.

No destructive operation runs without the explicit ``--recreate`` flag. All side effects
go through injectable seams (``AicApi`` over an httpx client + an ``SshRunner``), so the
test suite drives the whole bootstrap against a fake API + fake SSH transcript — CI never
spends money or opens a connection (mock-tested like ``extract_session``).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Make sutradhar importable when run as a script (repo root on sys.path via uv run).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sutradhar.config import Settings, get_settings  # noqa: E402

INSTANCE_NAME = "sutradhar-obs-01"
PLAN_SLUG = "essential-8gb"  # DEC-P3-7: cheapest tier WITH dedicated IPv4
OS_IMAGE = "ubuntu-24.04"
LANGFUSE_TAG = "v3.203.3"  # pinned release (verified on GitHub 2026-07-03)
AIC_BASE_URL = "https://api.aiccloud.in"
REMOTE_DIR = "/root/langfuse"
# AIC platform public URL (feature launched 2026-07; verified live 2026-07-11 via
# GET /api/v1/vps/{id}/public-url → {"slug": "…", "base_domain": "aiccloud.online"}):
# <slug>-PORT.<base_domain> routes HTTPS (wildcard TLS at the platform edge) to the
# VPS port — Langfuse web is :3000, so the public host is f"{slug}-3000.{base}".
# This REPLACES the P3 cloudflared quick tunnel (DEC-P3-7 amendment 2 workaround for
# the blocked inbound 443/80): no on-box agent, no random trycloudflare URL, and the
# slug survives reboots. HTTP(S)-only, which is all Langfuse needs.
LANGFUSE_WEB_PORT = 3000
LOCAL_ASSETS = Path(__file__).resolve().parent


class BootstrapBlockedError(RuntimeError):
    """A hard stop that must NOT be retried blindly (e.g. Docker-in-LXC nesting denied)."""


# --- Phase 1: AIC Cloud API (find-or-create, no hardcoded plan ids) ---


@dataclass
class AicApi:
    api_key: str
    client: httpx.Client
    base_url: str = AIC_BASE_URL

    def _get(self, path: str) -> Any:
        r = self.client.get(f"{self.base_url}{path}", headers=self._headers())
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        r = self.client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
        r.raise_for_status()
        return r.json()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_vps(self) -> list[dict[str, Any]]:
        data = self._get("/api/v1/vps")
        if isinstance(data, list):
            return data
        # Live shape (verified 2026-07-03): {"instances": [...]}.
        return list(data.get("instances", data.get("items", [])))

    def start_vps(self, vps_id: Any) -> Any:
        return self._post(f"/api/v1/vps/{vps_id}/start")

    def wallet_balance_paise(self) -> int:
        data = self._get("/api/v1/billing/wallet")
        return int(data.get("balance", 0))

    def essential_plans(self) -> list[dict[str, Any]]:
        data = self._get("/api/v1/public/essential-vps-plans")
        return list(data if isinstance(data, list) else data.get("plans", []))

    def checkout(self, plan_slug: str, name: str, os_image: str) -> dict[str, Any]:
        payload = {"planSlug": plan_slug, "name": name, "os": os_image}
        return dict(self._post("/api/v1/vps/checkout", payload))

    def verify_checkout(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._post("/api/v1/vps/checkout/verify", payload))

    def public_url(self, vps_id: Any) -> dict[str, Any]:
        """The platform public-URL record (verified live 2026-07-11):
        {"slug": "3um6p", "base_domain": "aiccloud.online", "custom_domains": [...]}."""
        return dict(self._get(f"/api/v1/vps/{vps_id}/public-url"))

    def delete_vps(self, vps_id: Any) -> Any:
        """Permanently delete a VPS (used by the P6 close-out teardown, 2026-07-11)."""
        r = self.client.delete(f"{self.base_url}/api/v1/vps/{vps_id}", headers=self._headers())
        r.raise_for_status()
        return r.json() if r.content else {}


@dataclass
class InstanceOutcome:
    status: str  # "running" | "started" | "created" | "needs-topup" | "awaiting-payment"
    #             | "dashboard-checkout" (verified live: AIC checkout is dashboard-only)
    instance: dict[str, Any] | None = None
    detail: str = ""


def ensure_instance(
    api: AicApi,
    *,
    ssh_public_keys: list[str],
    payment_prompt: Callable[[dict[str, Any]], dict[str, Any] | None],
    log: Callable[[str], None] = print,
) -> InstanceOutcome:
    """Find-or-create the VPS. Never calls checkout when the instance exists."""
    for vps in api.list_vps():
        if vps.get("name") == INSTANCE_NAME:
            state = str(vps.get("status", "")).lower()
            if state == "running":
                log(f"[phase1] {INSTANCE_NAME} exists and is running — skipping to phase 2")
                return InstanceOutcome(status="running", instance=vps)
            log(f"[phase1] {INSTANCE_NAME} exists ({state}) — starting it")
            api.start_vps(vps["id"])
            return InstanceOutcome(status="started", instance=vps)

    plans = {p.get("slug"): p for p in api.essential_plans()}
    plan = plans.get(PLAN_SLUG)
    if plan is None:
        raise BootstrapBlockedError(
            f"plan {PLAN_SLUG!r} not in the live catalogue ({sorted(plans)}) — "
            "re-check DEC-P3-7 plan selection"
        )
    price_paise = int(
        plan.get("price_monthly_paise", plan.get("price_paise", plan.get("price", 0)))
    )
    if price_paise <= 0:
        raise BootstrapBlockedError(
            f"plan {PLAN_SLUG!r} has no readable price in the catalogue payload — "
            "re-verify the AIC plans API shape before any checkout"
        )
    balance = api.wallet_balance_paise()
    if balance < price_paise + 100:  # +₹1 security fee
        need = (price_paise + 100 - balance) / 100
        log(
            f"[phase1] wallet ₹{balance / 100:.0f} < plan ₹{price_paise / 100:.0f}+fee — "
            f"top up ≥ ₹{need:.0f} in the AIC dashboard (Razorpay is browser-only), then re-run"
        )
        return InstanceOutcome(status="needs-topup", detail=f"short ₹{need:.0f}")

    order: dict[str, Any]
    try:
        order = api.checkout(PLAN_SLUG, INSTANCE_NAME, OS_IMAGE)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            # Verified live 2026-07-03: AIC serves checkout to the DASHBOARD only
            # ("This endpoint is not available via API key"). The create leg is a
            # documented manual step; find-or-create + phase 2 remain automated.
            log(
                "[phase1] AIC checkout is dashboard-only (403 via API key). Do this once:\n"
                f"  1. dashboard -> VPS -> buy plan '{PLAN_SLUG}' (Rs.799/mo + Rs.1 fee)\n"
                f"  2. name it EXACTLY {INSTANCE_NAME!r}; OS {OS_IMAGE!r}\n"
                "  3. attach your SSH public key (key-only SSH; password auth is "
                "disabled during bootstrap)\n"
                "  4. re-run `make langfuse-up` — it will find the instance and "
                "continue with phase 2"
            )
            return InstanceOutcome(status="dashboard-checkout", detail="checkout 403 via API")
        raise
    log(
        f"[phase1] checkout order {order.get('orderId')!r} created — pay it via Razorpay "
        "in the browser (payment legs are browser-only by design)"
    )
    payment = payment_prompt(order)
    if payment is None:
        return InstanceOutcome(status="awaiting-payment", detail=str(order.get("orderId")))
    payment = {**payment, "sshKeys": ssh_public_keys}  # key-only SSH from first boot
    created = api.verify_checkout(payment)
    log(f"[phase1] instance created: {created.get('id')}")
    return InstanceOutcome(status="created", instance=created)


# --- Phase 2: SSH bootstrap (check-then-act; every step converges) ---


@dataclass
class Step:
    name: str
    check: str  # shell probe; exit 0 = already satisfied
    act: list[str]  # commands run only when the check fails
    destructive: bool = False
    gate: bool = False  # act failure => BootstrapBlockedError (no blind retry)
    optional: bool = False  # act failure => warn and continue (e.g. swap inside LXC)


def _secrets_script(public_host: str) -> str:
    """Generate the compose .env ONCE: every '# CHANGEME' secret rotated, headless-init
    org/project/user pinned so the instance is reproducible from scratch (DEC-P3-7)."""
    return (
        f"cd {REMOTE_DIR} && umask 077 && "
        "PW_SALT=$(openssl rand -hex 32) && ENC_KEY=$(openssl rand -hex 32) && "
        "NEXTAUTH=$(openssl rand -hex 32) && CH_PW=$(openssl rand -hex 16) && "
        "MINIO_PW=$(openssl rand -hex 16) && REDIS_PW=$(openssl rand -hex 16) && "
        "PG_PW=$(openssl rand -hex 16) && "
        "PUB_KEY=pk-lf-$(openssl rand -hex 16) && SEC_KEY=sk-lf-$(openssl rand -hex 16) && "
        "INIT_PW=$(openssl rand -hex 12) && "
        "cat > .env << ENVEOF\n"
        f"NEXTAUTH_URL=https://{public_host}\n"
        "NEXTAUTH_SECRET=$NEXTAUTH\n"
        "SALT=$PW_SALT\n"
        "ENCRYPTION_KEY=$ENC_KEY\n"
        "POSTGRES_PASSWORD=$PG_PW\n"
        # Verified live 2026-07-03: the compose does NOT derive DATABASE_URL from
        # POSTGRES_PASSWORD — it must be written explicitly or web/worker auth-fails
        # against the rotated-password postgres.
        "DATABASE_URL=postgresql://postgres:$PG_PW@postgres:5432/postgres\n"
        "CLICKHOUSE_PASSWORD=$CH_PW\n"
        "MINIO_ROOT_PASSWORD=$MINIO_PW\n"
        # Live finding #2 of this class: the S3_*_SECRET_ACCESS_KEY defaults are the
        # LITERAL 'miniosecret', not derived from MINIO_ROOT_PASSWORD — derive all three.
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW\n"
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW\n"
        "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO_PW\n"
        "REDIS_AUTH=$REDIS_PW\n"
        "LANGFUSE_INIT_ORG_ID=sutradhar\n"
        "LANGFUSE_INIT_PROJECT_ID=sutradhar-p3\n"
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=$PUB_KEY\n"
        "LANGFUSE_INIT_PROJECT_SECRET_KEY=$SEC_KEY\n"
        "LANGFUSE_INIT_USER_EMAIL=owner@sutradhar.local\n"
        "LANGFUSE_INIT_USER_NAME=owner\n"
        "LANGFUSE_INIT_USER_PASSWORD=$INIT_PW\n"
        "ENVEOF\n"
        "echo generated"
    )


def _backup_script_b64() -> str:
    import base64

    return base64.b64encode((LOCAL_ASSETS / "backup_langfuse.sh").read_bytes()).decode("ascii")


def bootstrap_steps(public_host: str, ssh_port: int = 22) -> list[Step]:
    """The ordered, check-then-act phase-2 plan (see module doc).

    ``public_host`` is the platform public URL host (``<slug>-3000.aiccloud.online``),
    resolved laptop-side from the AIC API — TLS terminates at the platform edge, so the
    box needs no Caddy, no ACME, and (since 2026-07-11) no cloudflared tunnel.
    """
    compose = f"cd {REMOTE_DIR} && docker compose --env-file .env"
    return [
        Step(
            # Verified live 2026-07-03: swapon is NOT PERMITTED inside a Proxmox LXC
            # container (swap is host-managed) — best-effort only, warn and continue.
            name="swap-4g",
            check="swapon --show | grep -q /swapfile",
            act=[
                "fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile "
                "&& swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab"
            ],
            optional=True,
        ),
        Step(
            name="base-packages",
            check="command -v curl && command -v git",
            act=["apt-get update && apt-get install -y curl git ca-certificates"],
        ),
        Step(
            name="docker-installed",
            check="docker --version",
            act=["curl -fsSL https://get.docker.com | sh"],
        ),
        Step(
            name="docker-lxc-nesting",  # DEC-P3-7 caveat (a): validate day-0, escalate if blocked
            check="test -f /root/.docker-nesting-ok",
            act=["docker run --rm hello-world && touch /root/.docker-nesting-ok"],
            gate=True,
        ),
        Step(
            # The AIC API exposes an ssh_password for the box — password auth must die
            # (DEC-P3-7 hardening: key-only SSH).
            name="sshd-key-only",
            check="grep -qs '^PasswordAuthentication no' /etc/ssh/sshd_config.d/99-sutradhar.conf",
            act=[
                "printf 'PasswordAuthentication no\\nPermitRootLogin prohibit-password\\n' "
                "> /etc/ssh/sshd_config.d/99-sutradhar.conf && "
                "(systemctl reload ssh || systemctl reload sshd)"
            ],
        ),
        Step(
            name="langfuse-cloned-pinned",
            check=(
                f"test -d {REMOTE_DIR}/.git && "
                f"git -C {REMOTE_DIR} describe --tags --exact-match | grep -qx {LANGFUSE_TAG}"
            ),
            act=[
                f"rm -rf {REMOTE_DIR} && git clone --depth 1 --branch {LANGFUSE_TAG} "
                f"https://github.com/langfuse/langfuse.git {REMOTE_DIR}"
            ],
        ),
        Step(
            name="secrets-once",
            # DATABASE_URL presence is part of the contract (live finding: compose does
            # not derive it); an .env missing it is healed in place — secrets are NEVER
            # rotated against an already-initialized postgres volume.
            check=(
                f"test -f {REMOTE_DIR}/.env && ! grep -q CHANGEME {REMOTE_DIR}/.env "
                f"&& grep -q '^DATABASE_URL=' {REMOTE_DIR}/.env "
                f"&& grep -q '^LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=' {REMOTE_DIR}/.env"
            ),
            act=[
                f"cd {REMOTE_DIR} && if test -f .env; then "
                "PG_PW=$(sed -n 's/^POSTGRES_PASSWORD=//p' .env) && "
                "MINIO_PW=$(sed -n 's/^MINIO_ROOT_PASSWORD=//p' .env) && "
                "{ grep -q '^DATABASE_URL=' .env || echo "
                '"DATABASE_URL=postgresql://postgres:$PG_PW@postgres:5432/postgres" >> .env; } && '
                "{ grep -q '^LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=' .env || "
                "printf '%s\\n%s\\n%s\\n' "
                '"LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW" '
                '"LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW" '
                '"LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO_PW" >> .env; } && '
                # A healed .env must reach the running containers: compose recreates
                # only what changed (a passing health check can't see env drift).
                "docker compose --env-file .env up -d; "
                f"else {_secrets_script(public_host)}; fi"
            ],
        ),
        Step(
            name="compose-up",
            check="curl -sf http://localhost:3000/api/public/health",
            act=[f"{compose} up -d"],
        ),
        Step(
            # Public HTTPS (revised 2026-07-11): the AIC platform public URL
            # (<slug>-3000.aiccloud.online → VPS :3000, wildcard TLS at the edge)
            # REPLACES the P3 cloudflared quick tunnel — the tunnel existed only
            # because inbound 443/80 were blocked (DEC-P3-7 amendment 2). Retire the
            # tunnel service if a previous bootstrap installed it (idempotent).
            name="tunnel-retired",
            check="! systemctl is-enabled --quiet cloudflared-quick 2>/dev/null",
            act=[
                "systemctl disable --now cloudflared-quick 2>/dev/null || true",
                "rm -f /etc/systemd/system/cloudflared-quick.service "
                "/usr/local/bin/cloudflared && systemctl daemon-reload",
            ],
        ),
        Step(
            # SSH (internal 22 + the NAT'd external port) and the Langfuse web port
            # for the platform proxy ONLY; MinIO, ClickHouse etc. stay unreachable
            # (DEC-P3-7 hardening). 443/80 dropped with Caddy/ACME — TLS is the
            # platform edge's job now. Verified live 2026-07-03: allow BOTH 22 and
            # the external ssh_port or ufw locks out our own session.
            name="ufw-firewall",
            check="ufw status | grep -q 'Status: active'",
            act=[
                f"apt-get install -y ufw && ufw allow 22/tcp && ufw allow {ssh_port}/tcp "
                f"&& ufw allow {LANGFUSE_WEB_PORT}/tcp && ufw --force enable"
            ],
        ),
        Step(
            name="signup-disabled",
            check=f"grep -q '^AUTH_DISABLE_SIGNUP=true' {REMOTE_DIR}/.env",
            # Service name in the pinned compose is langfuse-web (verified live 2026-07-03).
            act=[
                f"echo 'AUTH_DISABLE_SIGNUP=true' >> {REMOTE_DIR}/.env "
                f"&& {compose} up -d langfuse-web"
            ],
        ),
        Step(
            name="backup-cron",
            check=(
                "test -x /root/backup_langfuse.sh && "
                "crontab -l 2>/dev/null | grep -q backup_langfuse"
            ),
            act=[
                # The committed backup script travels inline (base64 — no scp dependency).
                "echo "
                + shlex.quote(_backup_script_b64())
                + " | base64 -d > /root/backup_langfuse.sh && chmod +x /root/backup_langfuse.sh",
                '(crontab -l 2>/dev/null; echo "15 2 * * * /bin/bash '
                '/root/backup_langfuse.sh >> /var/log/backup_langfuse.log 2>&1") | crontab -',
            ],
        ),
        Step(
            name="https-health",
            # End-to-end through the PLATFORM edge (edge TLS -> proxy -> web container),
            # not just localhost — the same posture the tunnel health-check had.
            check=f'curl -sf "https://{public_host}/api/public/health"',
            act=[f'sleep 15 && curl -sf "https://{public_host}/api/public/health"'],
            gate=True,
        ),
    ]


DESTROY_STEPS = [
    Step(
        name="recreate-wipe",
        check="false",  # always acts — guarded by the explicit --recreate flag
        act=[
            f"cd {REMOTE_DIR} 2>/dev/null && docker compose down -v || true",
            f"rm -rf {REMOTE_DIR}",
        ],
        destructive=True,
    )
]


@dataclass
class SubprocessSsh:
    """Real SSH runner (key-only auth; password auth is disabled during bootstrap)."""

    host: str
    port: int = 22
    user: str = "root"

    def run(self, command: str) -> tuple[int, str]:
        proc = subprocess.run(  # noqa: S603 — operator-invoked infra tool
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-p",
                str(self.port),
                f"{self.user}@{self.host}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        return proc.returncode, proc.stdout + proc.stderr


@dataclass
class BootstrapReport:
    results: list[tuple[str, str]] = field(default_factory=list)  # (step, already|configured)

    def add(self, step: str, outcome: str) -> None:
        self.results.append((step, outcome))


def run_bootstrap(
    ssh: Any,
    public_host: str,
    *,
    ssh_port: int = 22,
    recreate: bool = False,
    log: Callable[[str], None] = print,
) -> BootstrapReport:
    """Phase 2: converge the box to a healthy, hardened Langfuse (safe to re-run)."""
    report = BootstrapReport()
    if recreate:
        log("[phase2] --recreate: wiping the existing deployment (DESTRUCTIVE)")
        for step in DESTROY_STEPS:
            for cmd in step.act:
                ssh.run(cmd)
            report.add(step.name, "configured")
    for step in bootstrap_steps(public_host, ssh_port):
        rc, _ = ssh.run(step.check)
        if rc == 0:
            log(f"[phase2] {step.name}: already satisfied — skipping")
            report.add(step.name, "already")
            continue
        log(f"[phase2] {step.name}: configuring …")
        for cmd in step.act:
            rc, output = ssh.run(cmd)
            if rc != 0:
                if step.gate:
                    raise BootstrapBlockedError(
                        f"{step.name} failed hard (rc={rc}): {output.strip()[:400]} — "
                        "if this is the LXC nesting gate, escalate to AIC support / a "
                        "KVM-class product rather than fighting it (DEC-P3-7)"
                    )
                if step.optional:
                    log(
                        f"[phase2] {step.name}: best-effort step failed (rc={rc}) — "
                        f"continuing: {output.strip()[:160]}"
                    )
                    break
                raise RuntimeError(f"{step.name}: {cmd!r} exited {rc}: {output.strip()[:400]}")
        else:
            report.add(step.name, "configured")
            continue
        report.add(step.name, "skipped-optional")
    rc, env_dump = ssh.run(f"grep -E 'LANGFUSE_INIT_PROJECT_(PUBLIC|SECRET)_KEY' {REMOTE_DIR}/.env")
    log("[phase2] bootstrap converged. For the laptop .env:")
    log(f"  LANGFUSE_HOST=https://{public_host}")
    for line in env_dump.strip().splitlines():
        log(f"  {line.replace('LANGFUSE_INIT_PROJECT_', 'LANGFUSE_')}")
    return report


# --- CLI ---


def _default_payment_prompt(order: dict[str, Any]) -> dict[str, Any] | None:
    print(f"Pay Razorpay order {order.get('orderId')!r} in the browser, then paste the ids.")
    payment_id = input("razorpay_payment_id (empty to abort and re-run later): ").strip()
    if not payment_id:
        return None
    return {
        "razorpay_order_id": order.get("orderId"),
        "razorpay_payment_id": payment_id,
        "razorpay_signature": input("razorpay_signature: ").strip(),
    }


def _local_ssh_public_keys() -> list[str]:
    """Any *.pub under ~/.ssh (people name keys freely — verified live: Vkt-Jarvis.pub)."""
    return [
        p.read_text(encoding="utf-8").strip()
        for p in sorted((Path.home() / ".ssh").glob("*.pub"))
        if p.is_file()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--public-host",
        default="",
        help="Override the platform public host (default: <slug>-3000.aiccloud.online "
        "resolved from the AIC public-url API; or a verified custom domain)",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="DESTRUCTIVE: wipe + redeploy langfuse"
    )
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    api_key = settings.require("aiccloud_api_key")
    with httpx.Client(timeout=60.0) as http:
        api = AicApi(api_key=api_key, client=http)
        outcome = ensure_instance(
            api,
            ssh_public_keys=_local_ssh_public_keys(),
            payment_prompt=_default_payment_prompt,
        )
        public_host = args.public_host
        if not public_host and outcome.instance is not None:
            # Platform public URL (2026-07-11): <slug>-3000.<base_domain> -> VPS :3000.
            record = api.public_url(outcome.instance.get("id"))
            public_host = (
                f"{record['slug']}-{LANGFUSE_WEB_PORT}."
                f"{record.get('base_domain', 'aiccloud.online')}"
            )
    if outcome.status in ("needs-topup", "awaiting-payment", "dashboard-checkout"):
        print(f"stopping here ({outcome.status}) — re-run `make langfuse-up` afterwards")
        return 1
    assert outcome.instance is not None
    # Live instance shape (verified 2026-07-03): ip_address + ssh_port at the top level;
    # keep the nested ssh{} fallback for the (never-observed) checkout-response shape.
    ssh_info = outcome.instance.get("ssh", {})
    host = str(
        ssh_info.get("host") or outcome.instance.get("ip_address") or outcome.instance.get("ip", "")
    )
    port = int(ssh_info.get("port") or outcome.instance.get("ssh_port", 22))
    ssh = SubprocessSsh(host=host, port=port)
    run_bootstrap(ssh, public_host, ssh_port=port, recreate=args.recreate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
