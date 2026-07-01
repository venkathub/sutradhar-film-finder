"""JarvisLabs ephemeral GPU validation — create -> serve -> smoke -> destroy (P0 task 8, DEC-P0-5).

One committed, automated lifecycle: provision a *fresh* JarvisLabs GPU, `vllm serve` the
env-driven ``LLM_MODEL``, health-wait, run the existing connectivity smoke against it, capture
evidence, then **destroy** the instance — with teardown guaranteed in a ``finally`` so a failed
smoke never leaks a billing GPU. Isolated under ``infra/gpu/`` (nothing in the app serving path
knows it is JarvisLabs). Ephemeral by design: create -> destroy, no warm machine.

    make gpu-validate   # python infra/gpu/jarvis.py validate
    make gpu-nuke       # python infra/gpu/jarvis.py nuke  (destroy any stray tagged instance)

Cost: one short create->destroy cycle on an A100 (DEC-0003 envelope). Developer/workflow_dispatch
invoked only — never on a PR.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sutradhar.config import Settings, get_settings
from sutradhar.serving.llm_client import EndpointStatus, LLMClient

# Tag used to name our instances/scripts so `nuke` can find and destroy strays.
INSTANCE_NAME = "sutradhar-p0-validate"
SERVE_PORT = 8000
# Generous health-wait: a fresh container must pip-install vllm and download the weights.
DEFAULT_HEALTH_TIMEOUT_S = 1500  # 25 min hard cap
DEFAULT_POLL_INTERVAL_S = 15


@dataclass
class Evidence:
    """What the one-time run records into infra/README.md's seed mini-runbook."""

    requested_model: str
    gpu_type: str
    booted: bool = False
    status: str = "not-started"
    served_model: str | None = None
    sample_token: str | None = None
    latency_ms: float | None = None
    tokens_per_sec: float | None = None
    create_to_up_s: float | None = None
    machine_id: int | None = None
    destroyed: bool = False
    detail: str = ""
    fallback_recommended: str | None = None

    def as_lines(self) -> list[str]:
        return [
            f"- requested_model:   {self.requested_model}",
            f"- gpu_type:          {self.gpu_type}",
            f"- booted:            {self.booted}",
            f"- smoke_status:      {self.status}",
            f"- served_model:      {self.served_model}",
            f"- sample_token:      {self.sample_token!r}",
            f"- latency_ms:        {self.latency_ms}",
            f"- tokens_per_sec:    {self.tokens_per_sec}",
            f"- create_to_up_s:    {self.create_to_up_s}",
            f"- machine_id:        {self.machine_id}",
            f"- destroyed:         {self.destroyed}",
            f"- fallback_needed:   {self.fallback_recommended}",
            f"- detail:            {self.detail}",
        ]


# Base gemma-4-E4B ships no chat template, so /v1/chat/completions (the smoke's probe step 3)
# 400s without one. Verified on the live A100 run (2026-07-01): passing this Gemma template makes
# chat.completions return a token (status="up"). Kept here so `make gpu-validate` reaches "up".
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)


def build_startup_script(model: str) -> str:
    """Bash run on instance launch: install vLLM and serve the model on SERVE_PORT.

    ``LLM_MODEL`` defaults to an *ungated* model, so no HF token is embedded in this script.
    A Gemma chat template is written and passed to ``--chat-template`` because the base model
    defines none (see :data:`GEMMA_CHAT_TEMPLATE`).
    """
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "pip install -U vllm\n"
        "cat > /root/chat_template.jinja << 'JINJA'\n"
        f"{GEMMA_CHAT_TEMPLATE}\n"
        "JINJA\n"
        f"vllm serve {model} --host 0.0.0.0 --port {SERVE_PORT} "
        "--chat-template /root/chat_template.jinja\n"
    )


def candidate_base_urls(instance: Any) -> list[str]:
    """Reachable OpenAI-compatible base URLs to try, derived from the live Instance object."""
    urls: list[str] = []
    public_ip = getattr(instance, "public_ip", None)
    if public_ip:
        urls.append(f"http://{public_ip}:{SERVE_PORT}/v1")
    for ep in getattr(instance, "endpoints", None) or []:
        ep = str(ep).rstrip("/")
        urls.append(ep if ep.endswith("/v1") else f"{ep}/v1")
    # De-dup, preserve order.
    return list(dict.fromkeys(urls))


@dataclass
class Deps:
    """Injectable side effects — real implementations by default, fakes in tests."""

    create_client: Callable[[Settings], Any]
    close_client: Callable[[Any], None]
    create_instance: Callable[[Any, Settings, str], Any]
    destroy_instance: Callable[[Any, int], bool]
    list_instances: Callable[[Any], list[Any]]
    probe_health: Callable[[str, float], bool]
    run_smoke: Callable[[str, Settings], EndpointStatus]
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = lambda msg: print(msg, flush=True)


def _default_deps() -> Deps:
    from jarvislabs import Client  # imported lazily so unit tests need no SDK

    def create_client(settings: Settings) -> Any:
        return Client(api_key=settings.require("jarvislabs_api_key"))

    def create_instance(client: Any, settings: Settings, script: str) -> Any:
        client.scripts.add(script=script, name=INSTANCE_NAME)
        script_id = max(
            (s.script_id for s in client.scripts.list() if s.script_name == INSTANCE_NAME),
            default=None,
        )
        return client.instances.create(
            gpu_type=settings.gpu_type,
            num_gpus=1,
            template="pytorch",
            storage=80,
            name=INSTANCE_NAME,
            http_ports=str(SERVE_PORT),
            script_id=str(script_id) if script_id is not None else None,
        )

    def probe_health(base_url: str, timeout: float) -> bool:
        import httpx

        root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
        try:
            resp = httpx.get(f"{root}/health", timeout=timeout)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False
        return resp.status_code == 200

    def run_smoke(base_url: str, settings: Settings) -> EndpointStatus:
        probe_settings = settings.model_copy(update={"llm_base_url": base_url})
        return LLMClient(probe_settings).health()

    return Deps(
        create_client=create_client,
        close_client=lambda c: c.close(),
        create_instance=create_instance,
        destroy_instance=lambda c, mid: bool(c.instances.destroy(mid)),
        list_instances=lambda c: list(c.instances.list()),
        probe_health=probe_health,
        run_smoke=run_smoke,
    )


def _wait_for_endpoint(
    instance: Any,
    deps: Deps,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> str | None:
    """Poll candidate URLs' /health until one is 200 or the timeout elapses."""
    deadline = deps.monotonic() + timeout_s
    while deps.monotonic() < deadline:
        for url in candidate_base_urls(instance):
            if deps.probe_health(url, 10.0):
                deps.log(f"  endpoint healthy: {url}")
                return url
        deps.sleep(poll_interval_s)
    return None


def validate(
    settings: Settings | None = None,
    deps: Deps | None = None,
    *,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> Evidence:
    """Run the full create -> serve -> smoke -> destroy lifecycle. Teardown is guaranteed."""
    settings = settings or get_settings()
    deps = deps or _default_deps()
    ev = Evidence(requested_model=settings.llm_model, gpu_type=settings.gpu_type)

    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"[gpu-validate] creating {settings.gpu_type} to serve {settings.llm_model} …")
        t0 = deps.monotonic()
        script = build_startup_script(settings.llm_model)
        instance = deps.create_instance(client, settings, script)
        ev.machine_id = getattr(instance, "machine_id", None)
        deps.log(f"  instance {ev.machine_id} running; waiting for vLLM /health …")

        base_url = _wait_for_endpoint(
            instance, deps, timeout_s=health_timeout_s, poll_interval_s=poll_interval_s
        )
        if base_url is None:
            ev.status = "off"
            ev.detail = f"vLLM did not become healthy within {health_timeout_s}s"
            ev.fallback_recommended = "Qwen/Qwen3-4B-Instruct-2507"
            return ev

        ev.create_to_up_s = round(deps.monotonic() - t0, 1)
        status = deps.run_smoke(base_url, settings)
        ev.status = status.status
        ev.served_model = status.model
        ev.sample_token = status.sample_token
        ev.latency_ms = status.latency_ms
        ev.booted = status.status == "up"
        if not ev.booted:
            ev.detail = f"smoke returned {status.status}: {status.detail}"
            ev.fallback_recommended = "Qwen/Qwen3-4B-Instruct-2507"
        else:
            ev.detail = "Gemma-4-E4B + vLLM booted on the rented GPU (DEC-0001 follow-up done)."
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        ev.fallback_recommended = "Qwen/Qwen3-4B-Instruct-2507"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        deps.close_client(client)


def nuke(settings: Settings | None = None, deps: Deps | None = None) -> list[int]:
    """Destroy any stray instance carrying our tag — a leaked billing GPU is a defect."""
    settings = settings or get_settings()
    deps = deps or _default_deps()
    client = deps.create_client(settings)
    destroyed: list[int] = []
    try:
        for inst in deps.list_instances(client):
            if getattr(inst, "name", None) == INSTANCE_NAME:
                mid = getattr(inst, "machine_id", None)
                if mid is not None and deps.destroy_instance(client, mid):
                    destroyed.append(mid)
                    deps.log(f"  nuked instance {mid}")
        if not destroyed:
            deps.log("  no stray sutradhar instances found")
        return destroyed
    finally:
        deps.close_client(client)


def _write_evidence(ev: Evidence) -> None:
    """Print the evidence block for pasting into infra/README.md's seed mini-runbook."""
    print("\n=== gpu-validate evidence (paste into infra/README.md) ===")
    for line in ev.as_lines():
        print(line)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "validate"
    if cmd == "validate":
        ev = validate()
        _write_evidence(ev)
        return 0 if ev.booted and ev.destroyed else 1
    if cmd == "nuke":
        nuke()
        return 0
    print(f"usage: jarvis.py [validate|nuke]  (got {cmd!r})")
    return 2


if __name__ == "__main__":
    sys.exit(main())
