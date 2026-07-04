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

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    remove_scripts: Callable[[Any], int] = lambda client: 0
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
            storage=settings.gpu_storage_gb,
            name=INSTANCE_NAME,
            # Expose the serve port AND the embed port (P4 window step [5]: BGE-M3 on
            # :8001 for the laptop-driven RAGAS pass — P3 never needed it off-box).
            http_ports=f"{SERVE_PORT},{EMBED_SERVE_PORT}",
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

    def remove_scripts(client: Any) -> int:
        removed = 0
        for s in list(client.scripts.list()):
            if s.script_name == INSTANCE_NAME and client.scripts.remove(s.script_id):
                removed += 1
        return removed

    return Deps(
        create_client=create_client,
        close_client=lambda c: c.close(),
        create_instance=create_instance,
        destroy_instance=lambda c, mid: bool(c.instances.destroy(mid)),
        list_instances=lambda c: list(c.instances.list()),
        probe_health=probe_health,
        run_smoke=run_smoke,
        remove_scripts=remove_scripts,
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
        try:  # script-quota hygiene: a leaked startup script blocks future runs (3-slot cap)
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
        deps.close_client(client)


def extract_session(
    settings: Settings | None = None,
    deps: Deps | None = None,
    *,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> Evidence:
    """P1 task 11: create -> serve -> run candidate-edge extraction -> destroy (DEC-P1-4).

    Reuses the validated P0 lifecycle; the extraction batch runs in a subprocess with
    ``LLM_BASE_URL`` pointed at the fresh instance, so the pipeline code stays
    provider-agnostic. Teardown guaranteed in ``finally``; artifact + candidate_edges
    persist before destroy.
    """
    settings = settings or get_settings()
    deps = deps or _default_deps()
    ev = Evidence(requested_model=settings.llm_model, gpu_type=settings.gpu_type)

    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"[gpu-extract] creating {settings.gpu_type} to serve {settings.llm_model} …")
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
            return ev
        ev.create_to_up_s = round(deps.monotonic() - t0, 1)
        ev.booted = True

        deps.log(f"  running extraction batch against {base_url} …")
        import os
        import subprocess

        env = {**os.environ, "LLM_BASE_URL": base_url, "LLM_TIMEOUT_S": "180"}
        proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
            [sys.executable, "data-pipeline/extract_candidates.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        deps.log(proc.stdout)
        if proc.stderr:
            deps.log(proc.stderr)
        ev.status = "up" if proc.returncode == 0 else "error"
        ev.detail = (
            "extraction batch complete (artifact + candidate_edges persisted)"
            if proc.returncode == 0
            else f"extraction exited {proc.returncode}"
        )
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        try:  # script-quota hygiene: a leaked startup script blocks future runs (3-slot cap)
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
        deps.close_client(client)


# --- P2 embed+score session (DEC-P2-7: HF Hub relay) --------------------------------
#
# The GPU box has no repo checkout and no Postgres. The laptop uploads gpu_inputs.json +
# the self-contained rag-engine/embed_and_score.py to a PRIVATE HF dataset repo
# (HF_ARTIFACT_REPO); the instance startup script downloads both, pip-installs the gpu
# deps, runs the job, uploads the sealed run dir + an EXIT marker back to the Hub; the
# laptop polls for EXIT, downloads the run into data/artifacts/retrieval/<run_id>/,
# verifies its MANIFEST, and destroys the instance (teardown in `finally`, as always).
#
# Secret note: unlike the vLLM validate script (ungated model, no token), this startup
# script MUST embed an HF token — upload back to the Hub is impossible without one. Use a
# fine-grained token scoped to HF_ARTIFACT_REPO only; the script slot is removed in the
# same `finally` (remove_scripts), so the token never outlives the session.

EMBED_JOB_SCRIPT = Path(__file__).resolve().parent.parent.parent / "rag-engine/embed_and_score.py"
EMBED_EXPORT_SCRIPT = "rag-engine/export_gpu_inputs.py"
EMBED_INPUTS_PATH = Path("data/interim/gpu_inputs.json")
EMBED_ARTIFACTS_ROOT = Path("data/artifacts/retrieval")
DEFAULT_EMBED_TIMEOUT_S = 3600  # embed+score is minutes; 1 h is a generous hard cap


@dataclass
class HubRelay:
    """Injectable HF Hub side effects for the embed session (fakes in tests)."""

    upload_file: Callable[[Path, str], None]  # local path -> path_in_repo
    exists: Callable[[str], bool]  # path_in_repo present?
    download_dir: Callable[[str, Path], None]  # path_in_repo prefix -> local dir
    fetch_log: Callable[[str], str | None] = lambda run_id: None  # job.log tail (diagnostics)


def _default_hub(settings: Settings) -> HubRelay:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=settings.require("hf_token"))
    repo = settings.require("hf_artifact_repo")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def upload_file(local: Path, path_in_repo: str) -> None:
        api.upload_file(
            path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo, repo_type="dataset"
        )

    def exists(path_in_repo: str) -> bool:
        return bool(api.file_exists(repo, path_in_repo, repo_type="dataset"))

    def download_dir(prefix: str, dest: Path) -> None:
        api.snapshot_download(
            repo_id=repo, repo_type="dataset", allow_patterns=[f"{prefix}/*"], local_dir=dest
        )

    def fetch_log(run_id: str, tail_chars: int = 4000) -> str | None:
        try:
            path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=f"runs/{run_id}/job.log",
                token=settings.require("hf_token"),
            )
            return Path(path).read_text(encoding="utf-8", errors="replace")[-tail_chars:]
        except Exception:  # noqa: BLE001 — diagnostics only, never mask the real failure
            return None

    return HubRelay(
        upload_file=upload_file, exists=exists, download_dir=download_dir, fetch_log=fetch_log
    )


def build_embed_startup_script(token: str, repo: str, run_id: str) -> str:
    """Bash run on instance launch: pull job+inputs from the Hub, run, push results back.

    The job's stdout/stderr are captured to ``job.log`` and uploaded even on failure, so a
    failed session is debuggable from the laptop without SSH."""
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"  # no -x: the token must never echo into instance logs
        f"export HF_TOKEN={token}\n"
        # FlagEmbedding 1.3–1.4 targets the transformers 4.x API (its reranker calls
        # tokenizer.prepare_for_model, removed in v5) — pin below 5 on the box.
        "pip install -q pyarrow huggingface_hub 'transformers<5' FlagEmbedding"
        " >/root/pip.log 2>&1 || true\n"
        "python - <<'PY'\n"
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import HfApi, hf_hub_download\n"
        f"repo, run_id = {repo!r}, {run_id!r}\n"
        "kw = dict(repo_id=repo, repo_type='dataset')\n"
        "api = HfApi()\n"
        "log = Path('/root/job.log')\n"
        "try:\n"
        "    inputs = hf_hub_download(filename=f'runs/{run_id}/gpu_inputs.json', **kw)\n"
        "    job = hf_hub_download(filename=f'runs/{run_id}/embed_and_score.py', **kw)\n"
        "    with log.open('wb') as fh:\n"
        "        rc = subprocess.call([sys.executable, job, '--inputs', inputs,\n"
        "                              '--out', '/root/artifacts', '--run-id', run_id],\n"
        "                             stdout=fh, stderr=subprocess.STDOUT)\n"
        "except Exception as exc:\n"
        "    log.write_text(f'driver exception: {exc!r}')\n"
        "    rc = 98\n"
        "for extra in ('/root/pip.log',):\n"
        "    try:\n"
        "        api.upload_file(path_or_fileobj=extra,\n"
        "                        path_in_repo=f'runs/{run_id}/pip.log', **kw)\n"
        "    except Exception:\n"
        "        pass\n"
        "try:\n"
        "    api.upload_file(path_or_fileobj=str(log),\n"
        "                    path_in_repo=f'runs/{run_id}/job.log', **kw)\n"
        "except Exception:\n"
        "    pass\n"
        "if rc == 0:\n"
        "    api.upload_folder(folder_path=f'/root/artifacts/{run_id}',\n"
        "                      path_in_repo=f'runs/{run_id}/out', **kw)\n"
        "api.upload_file(path_or_fileobj=str(rc).encode(),\n"
        "                path_in_repo=f'runs/{run_id}/EXIT', **kw)\n"
        "PY\n"
    )


def _run_export(out_path: Path) -> tuple[int, str, str]:
    """Laptop-side input export (needs the local DB). Injectable for tests."""
    import subprocess

    proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
        [sys.executable, EMBED_EXPORT_SCRIPT, "--out", str(out_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, proc.stdout, proc.stderr


def embed_session(
    settings: Settings | None = None,
    deps: Deps | None = None,
    hub: HubRelay | None = None,
    *,
    job_timeout_s: float = DEFAULT_EMBED_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    run_export: Callable[[Path], tuple[int, str, str]] = _run_export,
    artifacts_root: Path = EMBED_ARTIFACTS_ROOT,
    inputs_path: Path = EMBED_INPUTS_PATH,
) -> Evidence:
    """P2 task 5/6: export inputs -> create -> embed+score on the box -> pull -> destroy."""
    from sutradhar.rag.artifacts import ArtifactRun, new_run_id

    settings = settings or get_settings()
    deps = deps or _default_deps()
    hub = hub or _default_hub(settings)
    ev = Evidence(requested_model=settings.embed_model, gpu_type=settings.gpu_type)
    run_id = new_run_id()

    deps.log(f"[gpu-embed] exporting inputs for run {run_id} …")
    returncode, stdout, stderr = run_export(inputs_path)
    deps.log(stdout)
    if returncode != 0:
        ev.status = "error"
        ev.detail = f"input export failed: {stderr.strip()[-300:]}"
        return ev

    hub.upload_file(inputs_path, f"runs/{run_id}/gpu_inputs.json")
    hub.upload_file(EMBED_JOB_SCRIPT, f"runs/{run_id}/embed_and_score.py")

    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"  creating {settings.gpu_type} for embed+score (run {run_id}) …")
        t0 = deps.monotonic()
        script = build_embed_startup_script(
            settings.require("hf_token"), settings.require("hf_artifact_repo"), run_id
        )
        instance = deps.create_instance(client, settings, script)
        ev.machine_id = getattr(instance, "machine_id", None)
        deps.log(f"  instance {ev.machine_id} running; waiting for runs/{run_id}/EXIT …")

        deadline = deps.monotonic() + job_timeout_s
        exit_seen = False
        while deps.monotonic() < deadline:
            if hub.exists(f"runs/{run_id}/EXIT"):
                exit_seen = True
                break
            deps.sleep(poll_interval_s)
        if not exit_seen:
            ev.status = "off"
            ev.detail = f"embed job did not finish within {job_timeout_s}s"
            return ev
        ev.create_to_up_s = round(deps.monotonic() - t0, 1)

        if not hub.exists(f"runs/{run_id}/out/MANIFEST.sha256"):
            ev.status = "error"
            log_tail = hub.fetch_log(run_id)
            ev.detail = "job exited without a sealed artifact run" + (
                f" — job.log tail:\n{log_tail}" if log_tail else " (no job.log uploaded)"
            )
            return ev

        staging = artifacts_root / ".staging" / run_id
        hub.download_dir(f"runs/{run_id}/out", staging)
        src = staging / "runs" / run_id / "out"
        dest = artifacts_root / run_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        ArtifactRun.open(artifacts_root, run_id)  # hard verification before we celebrate
        ev.booted = True
        ev.status = "up"
        ev.detail = (
            f"sealed artifact run pulled + verified: {dest} — set RETRIEVAL_RUN={run_id} in .env"
        )
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        try:  # script-quota hygiene + the embedded token dies with the script slot
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
        deps.close_client(client)


# --- P3 judge session (DEC-P3-1: self-hosted OSS judge + BGE-M3, ephemeral) ----------
#
# Serves the pinned judge (JUDGE_MODEL, e.g. openai/gpt-oss-20b) via vLLM on SERVE_PORT
# and BGE-M3 as an OpenAI-compatible embedding server on EMBED_SERVE_PORT in the same
# session (DEC-P3-3: RAGAS answer-relevancy embeddings, zero external eval APIs). The
# judging work is a BATCH PASS over recorded transcripts (judge-validate report and, in
# Tier-2, the RAGAS pass), so the session is create -> judge -> destroy; between windows
# CI gates on the recorded artifact (DEC-P2-6 posture). Both models fit the A100 40 GB
# workhorse together (gpt-oss-20b MoE ~14-16 GB + BGE-M3) — no SKU upgrade.

EMBED_SERVE_PORT = 8001
DEFAULT_JUDGE_TIMEOUT_S = 3600  # the batch pass is minutes; 1 h hard cap (< $1, DEC-0003)


def build_judge_startup_script(judge_model: str, embed_model: str) -> str:
    """Serve the judge (port 8000, foreground) + BGE-M3 embeddings (port 8001, background).
    Both models are ungated — no HF token is embedded (same posture as validate)."""
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "pip install -U vllm\n"
        f"nohup vllm serve {embed_model} --task embed --host 0.0.0.0 "
        f"--port {EMBED_SERVE_PORT} > /root/embed.log 2>&1 &\n"
        f"vllm serve {judge_model} --host 0.0.0.0 --port {SERVE_PORT}\n"
    )


def _run_judge_report(base_url: str, judge_model: str) -> tuple[int, str, str]:
    """Default in-session runner: the κ report against the fresh judge endpoint."""
    import os
    import subprocess

    env = {**os.environ, "JUDGE_BASE_URL": base_url, "JUDGE_MODEL": judge_model}
    proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
        [sys.executable, "evals/judge_validate.py", "report"],
        env=env,
        capture_output=True,
        text=True,
        timeout=DEFAULT_JUDGE_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout, proc.stderr


def judge_session(
    settings: Settings | None = None,
    deps: Deps | None = None,
    *,
    run_report: Callable[[str, str], tuple[int, str, str]] = _run_judge_report,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> Evidence:
    """P3 task 7/13: create -> serve judge+embedder -> run the κ report -> destroy.

    Reuses the validated P0 lifecycle; teardown guaranteed in ``finally``. The report and
    worksheet persist under evals/judge_validation/ before the instance dies.
    """
    settings = settings or get_settings()
    deps = deps or _default_deps()
    judge_model = settings.require("judge_model")  # clear var-named error when unset
    ev = Evidence(requested_model=judge_model, gpu_type=settings.gpu_type)

    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"[gpu-judge] creating {settings.gpu_type} to serve {judge_model} …")
        t0 = deps.monotonic()
        script = build_judge_startup_script(judge_model, settings.embed_model)
        instance = deps.create_instance(client, settings, script)
        ev.machine_id = getattr(instance, "machine_id", None)
        deps.log(f"  instance {ev.machine_id} running; waiting for vLLM /health …")

        base_url = _wait_for_endpoint(
            instance, deps, timeout_s=health_timeout_s, poll_interval_s=poll_interval_s
        )
        if base_url is None:
            ev.status = "off"
            ev.detail = f"vLLM did not become healthy within {health_timeout_s}s"
            return ev
        ev.create_to_up_s = round(deps.monotonic() - t0, 1)
        ev.booted = True

        deps.log(f"  running judge κ report against {base_url} …")
        returncode, stdout, stderr = run_report(base_url, judge_model)
        deps.log(stdout)
        if stderr:
            deps.log(stderr)
        ev.status = "up" if returncode == 0 else "error"
        ev.detail = (
            "judge report complete (evals/judge_validation/report.json persisted)"
            if returncode == 0
            else f"judge report exited {returncode}"
            + (" — κ gate FAILED (DEC-P3-1 escalation path)" if returncode == 3 else "")
        )
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        try:  # script-quota hygiene: a leaked startup script blocks future runs
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
        deps.close_client(client)


# --- P4 teacher session (DEC-P4-1: Sarvam-M 24B, 8-bit weight-only, ephemeral) --------
#
# Serves the synthetic-data teacher (TEACHER_MODEL, e.g. sarvamai/sarvam-m) via vLLM with
# --quantization fp8 — on the Ampere A100 40 GB this runs as W8A16 WEIGHT-ONLY via Marlin
# kernels (~24 GB weights + KV headroom; P4_SPEC §3 D1). If bring-up disappoints, the
# recorded plan-B SKU is the RTX 6000 Ada 48 GB (native FP8, DEC-0003 value alternative).
# The work is `build_dataset.py teach` running ON THE LAPTOP against the fresh endpoint
# (JudgeClient/judge_session pattern): create -> serve -> teach -> destroy. Cost ~1-2 h.

DEFAULT_TEACH_TIMEOUT_S = 3 * 3600  # ~6k short rewrites; hard cap keeps DEC-0003 honest
# 48 GB bf16 download (unauthenticated) + on-the-fly fp8 quantization outlasts the 25-min
# default health cap comfortably on a slow mirror — 45 min for the teacher bring-up.
TEACHER_HEALTH_TIMEOUT_S = 2700


def build_teacher_startup_script(teacher_model: str) -> str:
    """Serve the teacher via vLLM, 8-bit weight-only (no HF token — Sarvam-M is ungated).

    HF_HOME is redirected to /home: on JarvisLabs the purchased storage volume mounts at
    /home while /root sits on a small fixed overlay — the ~48 GB Sarvam-M bf16 download
    filled it on the first live bring-up (2026-07-03; docs.jarvislabs.ai/getting_started:
    "always use /home for storage").
    """
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "export HF_HOME=/home/hf_cache\n"
        "mkdir -p $HF_HOME\n"
        "pip install -U vllm\n"
        f"vllm serve {teacher_model} --quantization fp8 --max-model-len 8192 "
        f"--host 0.0.0.0 --port {SERVE_PORT}\n"
    )


def _run_teach(base_url: str, teacher_model: str) -> tuple[int, str, str]:
    """Default in-session runner: the surface pass against the fresh teacher endpoint."""
    import os
    import subprocess

    env = {**os.environ, "TEACHER_BASE_URL": base_url, "TEACHER_MODEL": teacher_model}
    proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
        [sys.executable, "finetune/build_dataset.py", "teach"],
        env=env,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TEACH_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout, proc.stderr


def teacher_session(
    settings: Settings | None = None,
    deps: Deps | None = None,
    *,
    run_teach: Callable[[str, str], tuple[int, str, str]] = _run_teach,
    health_timeout_s: float = TEACHER_HEALTH_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> Evidence:
    """P4 task 6/7: create -> serve Sarvam-M -> run the surface pass -> destroy.

    Teardown guaranteed in ``finally``; the taught dataset + raw-output cache persist
    under data/artifacts/finetune/ before the instance dies. Exit code 2 from the teach
    runner = the DEC-P4-1 escalation trigger (rejection rate > 30%) — surfaced, and the
    instance is still destroyed.
    """
    settings = settings or get_settings()
    deps = deps or _default_deps()
    teacher_model = settings.require("teacher_model")  # clear var-named error when unset
    # Sarvam-M fp8-on-the-fly downloads the FULL bf16 checkpoint (~48 GB) before
    # quantizing — the 80 GB default disk ran out on the first live bring-up (2026-07-03).
    settings = settings.model_copy(update={"gpu_storage_gb": max(settings.gpu_storage_gb, 150)})
    ev = Evidence(requested_model=teacher_model, gpu_type=settings.gpu_type)

    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"[gpu-teacher] creating {settings.gpu_type} to serve {teacher_model} …")
        t0 = deps.monotonic()
        script = build_teacher_startup_script(teacher_model)
        instance = deps.create_instance(client, settings, script)
        ev.machine_id = getattr(instance, "machine_id", None)
        deps.log(f"  instance {ev.machine_id} running; waiting for vLLM /health …")

        base_url = _wait_for_endpoint(
            instance, deps, timeout_s=health_timeout_s, poll_interval_s=poll_interval_s
        )
        if base_url is None:
            ev.status = "off"
            ev.detail = f"vLLM did not become healthy within {health_timeout_s}s"
            return ev
        ev.create_to_up_s = round(deps.monotonic() - t0, 1)
        ev.booted = True

        deps.log(f"  running teacher surface pass against {base_url} …")
        returncode, stdout, stderr = run_teach(base_url, teacher_model)
        deps.log(stdout)
        if stderr:
            deps.log(stderr)
        ev.status = "up" if returncode == 0 else "error"
        ev.detail = (
            "surface pass complete (taught.jsonl + raw cache persisted)"
            if returncode == 0
            else f"teach exited {returncode}"
            + (" — rejection rate > 30% (DEC-P4-1 escalation trigger)" if returncode == 2 else "")
        )
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        try:  # script-quota hygiene: a leaked startup script blocks future runs
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
        deps.close_client(client)


# --- P4 finetune window (task 10; P4_SPEC §2.4 — the ONE-TIME training + benchmark
# window). Laptop half of the marker protocol driven by finetune/gpu_window.py (on-box
# half, relay-shipped). Strict order: base capture -> train -> merged capture(s) ->
# judge/RAGAS both columns -> pull -> DESTROY (teardown in finally, as always).

FT_MERGED_SERVED_NAME = "sutradhar-qlora-merged"  # vLLM --served-model-name for phase [3]
FT_TRL_DIR = Path("data/artifacts/finetune/trl")
FT_WINDOW_ARTIFACTS = Path("data/artifacts/finetune/window")
FT_WINDOW_SCRIPT = Path(__file__).resolve().parent.parent.parent / "finetune/gpu_window.py"
FT_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent.parent / "finetune/train_qlora.py"
DEFAULT_WINDOW_TIMEOUT_S = 8 * 3600  # spec budget: ~4-8 h total
DEFAULT_MARKER_TIMEOUT_S = 5400


def build_window_startup_script(token: str, repo: str, run_id: str) -> str:
    """Download + run the on-box window driver; job.log + EXIT uploaded even on crash."""
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"  # no -x: the token must never echo into instance logs
        f"export HF_TOKEN={token}\n"
        "export HF_HOME=/home/hf_cache\n"
        "mkdir -p /home/hf_cache\n"
        "pip install -q -U huggingface_hub >/root/pip.log 2>&1 || true\n"
        "python - <<'PY'\n"
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import HfApi, hf_hub_download\n"
        f"repo, run_id = {repo!r}, {run_id!r}\n"
        "kw = dict(repo_id=repo, repo_type='dataset')\n"
        "api = HfApi()\n"
        "log = Path('/root/job.log')\n"
        "try:\n"
        "    driver = hf_hub_download(filename=f'runs/{run_id}/gpu_window.py', **kw)\n"
        "    with log.open('wb') as fh:\n"
        "        rc = subprocess.call([sys.executable, driver, '--repo', repo,\n"
        "                              '--run-id', run_id],\n"
        "                             stdout=fh, stderr=subprocess.STDOUT)\n"
        "except Exception as exc:\n"
        "    log.write_text(f'driver exception: {exc!r}')\n"
        "    rc = 98\n"
        "try:\n"
        "    api.upload_file(path_or_fileobj=str(log),\n"
        "                    path_in_repo=f'runs/{run_id}/job.log', **kw)\n"
        "except Exception:\n"
        "    pass\n"
        "api.upload_file(path_or_fileobj=str(rc).encode(),\n"
        "                path_in_repo=f'runs/{run_id}/EXIT', **kw)\n"
        "PY\n"
    )


def _run_ft_benchmark(base_url: str, variant: str) -> tuple[int, str, str]:
    """Default capture runner: the P3 harness live against the window endpoint."""
    import os
    import subprocess

    args = [
        sys.executable,
        "evals/run_generation_eval.py",
        "--mode",
        "live",
        "--serving-json",
        json.dumps({"window_column": variant}),
    ]
    if variant == "qlora_no_exemplars":
        args += ["--prompt-variant", "no_exemplars"]
    env = {**os.environ, "LLM_BASE_URL": base_url}
    if variant.startswith("qlora"):
        env["LLM_MODEL"] = FT_MERGED_SERVED_NAME  # phase [3] serves under this name
    proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
        args, env=env, capture_output=True, text=True, timeout=3600
    )
    # rc=3 = the runner's GS-02 gate fired ON THE MODEL UNDER TEST. For the window that
    # is a RESULT (the artifact is recorded; DEC-P4-8's guard judges it at verdict time),
    # not an orchestration failure — a hallucinating BASE column is exactly the honest
    # baseline the benchmark exists to document (found live 2026-07-04).
    if proc.returncode == 3:
        note = "\n[window] GS-02 gate fired on this column — recorded; verdict applies DEC-P4-8"
        return 0, proc.stdout + note, proc.stderr
    return proc.returncode, proc.stdout, proc.stderr


def _run_ft_judge(base_url: str, embed_url: str, run_ids: list[str]) -> tuple[int, str, str]:
    """Default judge runner: re-judge + RAGAS every captured column artifact."""
    import subprocess

    settings = get_settings()
    env = {
        **os.environ,
        "JUDGE_BASE_URL": base_url,
        "JUDGE_MODEL": settings.require("judge_model"),
        "EMBED_BASE_URL": embed_url,
    }
    out, err = "", ""
    for run_id in run_ids:
        proc = subprocess.run(  # noqa: S603 — our own entrypoint, controlled args
            [
                sys.executable,
                "evals/rejudge_run.py",
                "--run",
                run_id,
                "--with-ragas",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        out += proc.stdout
        err += proc.stderr
        if proc.returncode != 0:
            return proc.returncode, out, err
    return 0, out, err


def _embed_base_url(instance: Any, base_url: str) -> str:
    """The BGE-M3 endpoint (:8001) as reachable from the laptop: prefer an instance
    endpoint that names the embed port; fall back to a port swap on the serve URL."""
    for ep in getattr(instance, "endpoints", None) or []:
        ep = str(ep).rstrip("/")
        if str(EMBED_SERVE_PORT) in ep:
            return ep if ep.endswith("/v1") else f"{ep}/v1"
    return base_url.replace(str(SERVE_PORT), str(EMBED_SERVE_PORT), 1)


def _latest_run_id(runs_dir: Path = Path("evals/generation_runs")) -> str:
    runs = sorted(p.stem for p in runs_dir.glob("*.json") if ".trace" not in p.name)
    if not runs:
        raise RuntimeError("no generation-run artifact found after capture")
    return runs[-1]


def finetune_session(
    settings: Settings | None = None,
    deps: Deps | None = None,
    hub: HubRelay | None = None,
    *,
    run_benchmark: Callable[[str, str], tuple[int, str, str]] = _run_ft_benchmark,
    run_judge_pass: Callable[[str, str, list[str]], tuple[int, str, str]] = _run_ft_judge,
    latest_run_id: Callable[[], str] = _latest_run_id,
    window_timeout_s: float = DEFAULT_WINDOW_TIMEOUT_S,
    marker_timeout_s: float = DEFAULT_MARKER_TIMEOUT_S,
    health_timeout_s: float = TEACHER_HEALTH_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    trl_dir: Path = FT_TRL_DIR,
    artifacts_root: Path = FT_WINDOW_ARTIFACTS,
) -> Evidence:
    """THE one-time window (§2.4): base capture -> train -> QLoRA capture(s) -> judge ->
    pull -> destroy. Every phase failure still tears the instance down."""
    from sutradhar.finetune.train import TRAINING_PIPS

    settings = settings or get_settings()
    deps = deps or _default_deps()
    hub = hub or _default_hub(settings)
    base_model = settings.llm_model
    judge_model = settings.require("judge_model")
    settings = settings.model_copy(update={"gpu_storage_gb": max(settings.gpu_storage_gb, 150)})
    ev = Evidence(requested_model=base_model, gpu_type=settings.gpu_type)
    run_id = f"ftwin-{os.urandom(4).hex()}"

    payload = {
        "train_config.json": trl_dir / "train_config.json",
        "train_rows.jsonl": trl_dir / "train_rows.jsonl",
        "val_rows.jsonl": trl_dir / "val_rows.jsonl",
        "train_qlora.py": FT_TRAIN_SCRIPT,
        "gpu_window.py": FT_WINDOW_SCRIPT,
        "gemma4_train_template.jinja": FT_TRAIN_SCRIPT.parent / "gemma4_train_template.jinja",
        "merge_adapter.py": FT_TRAIN_SCRIPT.parent / "merge_adapter.py",
    }
    missing = [name for name, path in payload.items() if not path.exists()]
    if missing:
        ev.status = "error"
        ev.detail = f"relay payload missing: {missing} — run `make build-ft-scaffold`/export-trl"
        return ev
    window_config = {
        "base_model": base_model,
        "judge_model": judge_model,
        "embed_model": settings.embed_model,
        "serve_flags": settings.vllm_serve_flags.split(),
        "training_pips": list(TRAINING_PIPS),
        "marker_timeout_s": marker_timeout_s,
        # RESUME (2026-07-04 lesson): a relay prefix holding a finished adapter — the
        # window skips training entirely and goes straight to merge/captures.
        "resume_from": os.environ.get("FT_RESUME_FROM", ""),
        "merged_served_name": FT_MERGED_SERVED_NAME,
    }
    deps.log(f"[gpu-finetune] window {run_id}: uploading relay payload …")
    for name, path in payload.items():
        hub.upload_file(path, f"runs/{run_id}/{name}")
    config_path = trl_dir / "window_config.json"
    config_path.write_text(json.dumps(window_config, indent=2) + "\n", encoding="utf-8")
    hub.upload_file(config_path, f"runs/{run_id}/window_config.json")

    def wait_marker(name: str, timeout_s: float) -> bool:
        deadline = deps.monotonic() + timeout_s
        while deps.monotonic() < deadline:
            if hub.exists(f"runs/{run_id}/{name}"):
                return True
            if hub.exists(f"runs/{run_id}/EXIT"):  # box died early
                return False
            deps.sleep(poll_interval_s)
        return False

    def put_marker(name: str) -> None:
        marker = trl_dir / f".marker_{name}"
        marker.write_bytes(b"1")
        hub.upload_file(marker, f"runs/{run_id}/{name}")

    captured_runs: list[str] = []
    client = deps.create_client(settings)
    instance: Any = None
    try:
        deps.log(f"  creating {settings.gpu_type} for the finetune window …")
        t0 = deps.monotonic()
        script = build_window_startup_script(
            settings.require("hf_token"), settings.require("hf_artifact_repo"), run_id
        )
        instance = deps.create_instance(client, settings, script)
        ev.machine_id = getattr(instance, "machine_id", None)

        # [1] base column
        deps.log("  waiting for BASE_UP …")
        if not wait_marker("BASE_UP", health_timeout_s + marker_timeout_s):
            ev.status = "error"
            ev.detail = "BASE_UP never appeared" + (
                f" — job.log tail:\n{hub.fetch_log(run_id)}" if hub.fetch_log(run_id) else ""
            )
            return ev
        base_url = _wait_for_endpoint(
            instance, deps, timeout_s=health_timeout_s, poll_interval_s=poll_interval_s
        )
        if base_url is None:
            ev.status = "error"
            ev.detail = "BASE_UP marker present but endpoint unreachable from the laptop"
            return ev
        ev.booted = True
        ev.create_to_up_s = round(deps.monotonic() - t0, 1)
        deps.log(f"  [1] capturing BASE column against {base_url} …")
        rc, out, err = run_benchmark(base_url, "base")
        deps.log(out)
        if rc != 0:
            ev.status = "error"
            ev.detail = f"base capture failed rc={rc}: {err.strip()[-300:]}"
            return ev
        captured_runs.append(latest_run_id())
        put_marker("BASE_CAPTURED")

        # [2]+[3] train happens on-box; wait for the merged model to come up.
        deps.log("  [2] training on-box; waiting for MERGED_UP …")
        if not wait_marker("MERGED_UP", window_timeout_s):
            ev.status = "error"
            ev.detail = "MERGED_UP never appeared (training failed?)" + (
                f" — job.log tail:\n{hub.fetch_log(run_id)}" if hub.fetch_log(run_id) else ""
            )
            return ev
        deps.log("  [3] capturing QLORA column (headline, frozen prompt) …")
        rc, out, err = run_benchmark(base_url, "qlora")
        deps.log(out)
        if rc != 0:
            ev.status = "error"
            ev.detail = f"qlora capture failed rc={rc}: {err.strip()[-300:]}"
            return ev
        captured_runs.append(latest_run_id())
        deps.log("  [4] capturing QLORA no-exemplar supplementary (DEC-P4-6) …")
        rc, out, err = run_benchmark(base_url, "qlora_no_exemplars")
        deps.log(out)
        if rc != 0:
            ev.status = "error"
            ev.detail = f"no-exemplar capture failed rc={rc}: {err.strip()[-300:]}"
            return ev
        captured_runs.append(latest_run_id())
        put_marker("QLORA_CAPTURED")

        # [5] judge + RAGAS over all captured columns, same session.
        deps.log("  [5] waiting for JUDGE_UP …")
        if not wait_marker("JUDGE_UP", marker_timeout_s):
            ev.status = "error"
            ev.detail = "JUDGE_UP never appeared"
            return ev
        embed_url = _embed_base_url(instance, base_url)
        rc, out, err = run_judge_pass(base_url, embed_url, captured_runs)
        deps.log(out)
        if rc != 0:
            ev.status = "error"
            ev.detail = f"judge pass failed rc={rc}: {err.strip()[-300:]}"
            return ev
        put_marker("JUDGE_DONE")

        # [6] EXIT + pull the on-box outputs (training metrics; adapter went to HF Hub).
        if not wait_marker("EXIT", marker_timeout_s):
            ev.status = "error"
            ev.detail = "EXIT never appeared after JUDGE_DONE"
            return ev
        dest = artifacts_root / run_id
        dest.mkdir(parents=True, exist_ok=True)
        if hub.exists(f"runs/{run_id}/out/training_metrics.json"):
            hub.download_dir(f"runs/{run_id}/out", dest)
        ev.status = "up"
        ev.detail = (
            f"window complete: columns {captured_runs} captured + judged; "
            f"on-box outputs pulled to {dest}"
        )
        return ev
    except Exception as exc:  # noqa: BLE001 — record, then guarantee teardown in finally
        ev.status = "error"
        ev.detail = f"{type(exc).__name__}: {exc}"
        return ev
    finally:
        if instance is not None and ev.machine_id is not None:
            try:
                ev.destroyed = deps.destroy_instance(client, ev.machine_id)
                deps.log(f"  destroyed instance {ev.machine_id}: {ev.destroyed}")
            except Exception as exc:  # noqa: BLE001
                ev.detail += f" | TEARDOWN FAILED: {exc} — run `make gpu-nuke`"
                deps.log(f"  TEARDOWN FAILED for {ev.machine_id}: {exc}")
        try:  # script-quota hygiene + the embedded token dies with the script slot
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
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
    if cmd == "extract":
        ev = extract_session()
        _write_evidence(ev)
        return 0 if ev.status == "up" and ev.destroyed else 1
    if cmd == "embed":
        ev = embed_session()
        _write_evidence(ev)
        return 0 if ev.status == "up" and ev.destroyed else 1
    if cmd == "judge":
        ev = judge_session()
        _write_evidence(ev)
        return 0 if ev.status == "up" and ev.destroyed else 1
    if cmd == "teacher":
        ev = teacher_session()
        _write_evidence(ev)
        return 0 if ev.status == "up" and ev.destroyed else 1
    if cmd == "finetune":
        ev = finetune_session()
        _write_evidence(ev)
        return 0 if ev.status == "up" and ev.destroyed else 1
    if cmd == "nuke":
        nuke()
        return 0
    print(f"usage: jarvis.py [validate|extract|embed|judge|teacher|finetune|nuke]  (got {cmd!r})")
    return 2


if __name__ == "__main__":
    sys.exit(main())
