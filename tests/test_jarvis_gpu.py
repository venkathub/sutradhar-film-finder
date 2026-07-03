"""Offline tests for the JarvisLabs GPU lifecycle (P0 task 8).

Fully mocked — no SDK, no network, no GPU. The key assertion is that teardown (destroy) is
guaranteed even when serving/smoke fails, so a run can never leak a billing GPU.

The script lives outside the package (infra/gpu/, per P0_SPEC §2.5); load it by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from sutradhar.config import Settings
from sutradhar.serving.llm_client import EndpointStatus

_JARVIS_PATH = Path(__file__).resolve().parents[1] / "infra" / "gpu" / "jarvis.py"


def _load_jarvis() -> Any:
    spec = importlib.util.spec_from_file_location("jarvis_under_test", _JARVIS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass machinery resolves the module by name
    spec.loader.exec_module(module)
    return module


jarvis = _load_jarvis()


class _FakeInstance:
    def __init__(self, machine_id: int, public_ip: str = "10.0.0.9") -> None:
        self.machine_id = machine_id
        self.public_ip = public_ip
        self.endpoints: list[str] = []
        self.name = jarvis.INSTANCE_NAME


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.destroyed: list[int] = []
        self.instances_list: list[Any] = []


def _settings() -> Settings:
    return Settings(_env_file=None, LLM_MODEL="google/gemma-4-E4B", GPU_TYPE="A100")


def _base_deps(client: _FakeClient, **over: Any) -> Any:
    """Deps that succeed unless overridden; monotonic advances so timeouts terminate."""
    ticker = {"t": 0.0}

    def monotonic() -> float:
        ticker["t"] += 1.0
        return ticker["t"]

    def destroy(c: _FakeClient, mid: int) -> bool:
        c.destroyed.append(mid)
        return True

    defaults = dict(
        create_client=lambda s: client,
        close_client=lambda c: setattr(c, "closed", True),
        create_instance=lambda client, settings, script: _FakeInstance(4242),
        destroy_instance=destroy,
        list_instances=lambda c: c.instances_list,
        probe_health=lambda url, timeout: True,
        run_smoke=lambda url, s: EndpointStatus(
            status="up",
            model="google/gemma-4-E4B",
            sample_token="pong",
            latency_ms=11.0,
            detail="endpoint UP",
        ),
        monotonic=monotonic,
        sleep=lambda s: None,
        log=lambda m: None,
    )
    defaults.update(over)
    return jarvis.Deps(**defaults)


def test_happy_path_boots_and_destroys() -> None:
    client = _FakeClient()
    ev = jarvis.validate(_settings(), _base_deps(client), health_timeout_s=100, poll_interval_s=1)
    assert ev.booted is True
    assert ev.status == "up"
    assert ev.sample_token == "pong"
    assert ev.destroyed is True
    assert client.destroyed == [4242]
    assert client.closed is True


def test_teardown_runs_when_smoke_raises() -> None:
    client = _FakeClient()

    def boom(url: str, s: Settings) -> EndpointStatus:
        raise RuntimeError("vllm crashed mid-smoke")

    ev = jarvis.validate(
        _settings(), _base_deps(client, run_smoke=boom), health_timeout_s=100, poll_interval_s=1
    )
    assert ev.booted is False
    assert ev.status == "error"
    # The critical guarantee: the instance was still destroyed.
    assert client.destroyed == [4242]
    assert ev.destroyed is True
    assert ev.fallback_recommended == "Qwen/Qwen3-4B-Instruct-2507"


def test_teardown_runs_when_health_times_out() -> None:
    client = _FakeClient()
    ev = jarvis.validate(
        _settings(),
        _base_deps(client, probe_health=lambda url, timeout: False),
        health_timeout_s=5,
        poll_interval_s=1,
    )
    assert ev.booted is False
    assert ev.status == "off"
    assert client.destroyed == [4242]  # never leak a GPU, even on timeout


def test_nuke_destroys_only_tagged_instances() -> None:
    client = _FakeClient()
    tagged = _FakeInstance(1)
    other = _FakeInstance(2)
    other.name = "someone-elses-box"
    client.instances_list = [tagged, other]
    destroyed = jarvis.nuke(_settings(), _base_deps(client))
    assert destroyed == [1]
    assert client.destroyed == [1]


def test_candidate_base_urls_from_ip_and_endpoints() -> None:
    inst = _FakeInstance(7, public_ip="1.2.3.4")
    inst.endpoints = ["https://abc-8000.jarvislabs.net"]
    urls = jarvis.candidate_base_urls(inst)
    assert "http://1.2.3.4:8000/v1" in urls
    assert "https://abc-8000.jarvislabs.net/v1" in urls


def test_startup_script_has_no_secret_and_serves_model() -> None:
    script = jarvis.build_startup_script("google/gemma-4-E4B")
    assert "vllm serve google/gemma-4-E4B" in script
    assert "--port 8000" in script
    assert "--chat-template" in script  # base gemma needs one (verified on live run)
    assert "hf_" not in script.lower()  # ungated model => no token embedded


# --- P3 judge session (DEC-P3-1: serve judge + BGE-M3, run kappa report, destroy) ---


def _judge_settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_MODEL="google/gemma-4-E4B",
        GPU_TYPE="A100",
        JUDGE_MODEL="openai/gpt-oss-20b",
    )


def test_judge_session_runs_report_and_destroys() -> None:
    client = _FakeClient()
    seen: dict[str, str] = {}

    def fake_report(base_url: str, judge_model: str) -> tuple[int, str, str]:
        seen["base_url"] = base_url
        seen["model"] = judge_model
        return 0, "kappa=0.8 PASS", ""

    ev = jarvis.judge_session(
        _judge_settings(),
        _base_deps(client),
        run_report=fake_report,
        health_timeout_s=100,
        poll_interval_s=1,
    )
    assert ev.status == "up" and ev.booted is True
    assert seen["model"] == "openai/gpt-oss-20b"
    assert seen["base_url"].endswith("/v1")
    assert client.destroyed == [4242] and ev.destroyed is True


def test_judge_session_kappa_gate_failure_still_destroys() -> None:
    client = _FakeClient()
    ev = jarvis.judge_session(
        _judge_settings(),
        _base_deps(client),
        run_report=lambda url, model: (3, "kappa=0.4 FAIL", ""),
        health_timeout_s=100,
        poll_interval_s=1,
    )
    assert ev.status == "error"
    assert "κ gate FAILED" in ev.detail
    assert client.destroyed == [4242]  # never leak a GPU on a failed gate


def test_judge_session_requires_judge_model() -> None:
    from sutradhar.config import ConfigError

    client = _FakeClient()
    try:
        jarvis.judge_session(_settings(), _base_deps(client))
    except ConfigError as exc:
        assert "JUDGE_MODEL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unset JUDGE_MODEL")


def test_judge_startup_script_serves_both_models_no_secret() -> None:
    script = jarvis.build_judge_startup_script("openai/gpt-oss-20b", "BAAI/bge-m3")
    assert f"--port {jarvis.SERVE_PORT}" in script
    assert f"--task embed --host 0.0.0.0 --port {jarvis.EMBED_SERVE_PORT}" in script
    assert "openai/gpt-oss-20b" in script and "BAAI/bge-m3" in script
    assert "hf_" not in script and "HF_TOKEN" not in script  # ungated models, no token


# --- P4 teacher session (task 6; DEC-P4-1 Sarvam-M, DEC-P0-5 ephemeral pattern) ---


def _teacher_settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_MODEL="google/gemma-4-E4B",
        GPU_TYPE="A100",
        TEACHER_MODEL="sarvamai/sarvam-m",
    )


def test_teacher_session_runs_teach_and_destroys() -> None:
    client = _FakeClient()
    seen: dict[str, str] = {}

    def fake_teach(base_url: str, teacher_model: str) -> tuple[int, str, str]:
        seen["base_url"] = base_url
        seen["model"] = teacher_model
        return 0, "taught 2000 conversations (rate 4.00%)", ""

    ev = jarvis.teacher_session(
        _teacher_settings(),
        _base_deps(client),
        run_teach=fake_teach,
        health_timeout_s=100,
        poll_interval_s=1,
    )
    assert ev.status == "up" and ev.booted is True
    assert seen["model"] == "sarvamai/sarvam-m"
    assert seen["base_url"].endswith("/v1")
    assert client.destroyed == [4242] and ev.destroyed is True


def test_teacher_session_escalation_exit_still_destroys() -> None:
    client = _FakeClient()
    ev = jarvis.teacher_session(
        _teacher_settings(),
        _base_deps(client),
        run_teach=lambda url, model: (2, "rate 42%", ""),
        health_timeout_s=100,
        poll_interval_s=1,
    )
    assert ev.status == "error"
    assert "DEC-P4-1 escalation trigger" in ev.detail
    assert client.destroyed == [4242]  # never leak a GPU on a failed gate


def test_teacher_session_teach_crash_still_destroys() -> None:
    client = _FakeClient()

    def boom(url: str, model: str) -> tuple[int, str, str]:
        raise RuntimeError("laptop lost network mid-pass")

    ev = jarvis.teacher_session(
        _teacher_settings(),
        _base_deps(client),
        run_teach=boom,
        health_timeout_s=100,
        poll_interval_s=1,
    )
    assert ev.status == "error"
    assert client.destroyed == [4242]


def test_teacher_session_requires_teacher_model() -> None:
    from sutradhar.config import ConfigError

    client = _FakeClient()
    try:
        jarvis.teacher_session(_settings(), _base_deps(client))
    except ConfigError as exc:
        assert "TEACHER_MODEL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unset TEACHER_MODEL")


def test_teacher_startup_script_quantized_no_secret() -> None:
    script = jarvis.build_teacher_startup_script("sarvamai/sarvam-m")
    assert "vllm serve sarvamai/sarvam-m" in script
    assert "--quantization fp8" in script  # W8A16 weight-only on Ampere (P4_SPEC D1)
    assert "HF_HOME=/home/hf_cache" in script  # persistent volume — root overlay is tiny
    assert "hf_" not in script.replace("hf_cache", "")  # ungated model — no token embedded
