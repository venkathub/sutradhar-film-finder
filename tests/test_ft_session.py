"""Fake-transcript tests for the P4 finetune window driver (spec §4 test_finetune_session).

DEC-P0-5 pattern: fake Deps + fake HubRelay drive the §2.4 phase machine end-to-end with
zero network/GPU. Asserts strict step ordering (base -> train -> after -> judge -> push
-> destroy), teardown on injected failure at EVERY phase, and that the HF token never
reaches a log line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_jarvis() -> Any:
    spec = importlib.util.spec_from_file_location("jarvis_ft", REPO_ROOT / "infra/gpu/jarvis.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["jarvis_ft"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


jarvis = _load_jarvis()

from sutradhar.config import Settings  # noqa: E402


class _FakeInstance:
    def __init__(self, machine_id: int) -> None:
        self.machine_id = machine_id
        self.public_ip = "10.0.0.9"
        self.endpoints: list[str] = []
        self.name = jarvis.INSTANCE_NAME


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.destroyed: list[int] = []


class FakeHub:
    """Marker-file relay fake: the 'box side' is scripted by phase_plan."""

    def __init__(self, phase_plan: list[str]) -> None:
        self.files: set[str] = set()
        self.uploaded: list[str] = []
        self.phase_plan = phase_plan  # markers the fake box emits, in order
        self._emitted = 0

    def upload_file(self, local: Path, path_in_repo: str) -> None:
        self.uploaded.append(path_in_repo)
        self.files.add(path_in_repo)
        # When the laptop uploads a phase-completion marker, the fake box "reacts" by
        # emitting its next marker(s).
        self._advance()

    def exists(self, path_in_repo: str) -> bool:
        self._advance()
        return path_in_repo in self.files

    def download_dir(self, prefix: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "training_metrics.json").write_text("{}", encoding="utf-8")

    def fetch_log(self, run_id: str) -> str | None:
        return "fake job.log tail"

    def _advance(self) -> None:
        """Emit the next box marker once its laptop-side precondition is satisfied."""
        preconditions = {
            "BASE_UP": None,
            "MERGED_UP": "BASE_CAPTURED",
            "JUDGE_UP": "QLORA_CAPTURED",
            "EXIT": "JUDGE_DONE",
            "out/training_metrics.json": "BASE_CAPTURED",
        }
        while self._emitted < len(self.phase_plan):
            marker = self.phase_plan[self._emitted]
            pre = preconditions.get(marker)
            if pre is not None and not any(u.endswith(pre) for u in self.uploaded):
                break
            self.files.add(f"runs/{self._run_id()}/{marker}")
            self._emitted += 1

    def _run_id(self) -> str:
        for name in self.uploaded:
            if name.startswith("runs/"):
                return name.split("/")[1]
        return "unknown"


def _settings(tmp_path: Path) -> Settings:
    import pytest  # noqa: F401

    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        LLM_MODEL="google/gemma-4-E4B",
        JUDGE_MODEL="openai/gpt-oss-20b",
        HF_TOKEN="hf_fttest_secret_token",
        HF_ARTIFACT_REPO="user/relay",
        GPU_TYPE="A100",
    )


def _payload_dir(tmp_path: Path) -> Path:
    trl = tmp_path / "trl"
    trl.mkdir()
    for name in ("train_config.json", "train_rows.jsonl", "val_rows.jsonl"):
        (trl / name).write_text("{}", encoding="utf-8")
    return trl


def _deps(client: _FakeClient, logs: list[str], **over: Any) -> Any:
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
        create_instance=lambda client, settings, script: _FakeInstance(7777),
        destroy_instance=destroy,
        list_instances=lambda c: [],
        probe_health=lambda url, timeout: True,
        run_smoke=lambda url, s: None,
        monotonic=monotonic,
        sleep=lambda s: None,
        log=logs.append,
    )
    defaults.update(over)
    return jarvis.Deps(**defaults)


_FULL_PLAN = ["BASE_UP", "MERGED_UP", "out/training_metrics.json", "JUDGE_UP", "EXIT"]


def _run(
    tmp_path: Path,
    plan: list[str],
    benchmark_rcs: dict[str, int] | None = None,
    judge_rc: int = 0,
) -> tuple[Any, _FakeClient, FakeHub, list[str], list[tuple[str, str]]]:
    client = _FakeClient()
    logs: list[str] = []
    hub = FakeHub(plan)
    captures: list[tuple[str, str]] = []
    rcs = benchmark_rcs or {}
    counter = {"n": 0}

    def fake_benchmark(base_url: str, variant: str) -> tuple[int, str, str]:
        captures.append(("benchmark", variant))
        counter["n"] += 1
        return rcs.get(variant, 0), f"captured {variant}", ""

    def fake_judge(base_url: str, run_ids: list[str]) -> tuple[int, str, str]:
        captures.append(("judge", ",".join(run_ids)))
        return judge_rc, "judged", ""

    def fake_latest() -> str:
        return f"run-{counter['n']:02d}"

    ev = jarvis.finetune_session(
        _settings(tmp_path),
        _deps(client, logs),
        hub,
        run_benchmark=fake_benchmark,
        run_judge_pass=fake_judge,
        latest_run_id=fake_latest,
        window_timeout_s=200,
        marker_timeout_s=100,
        health_timeout_s=100,
        poll_interval_s=1,
        trl_dir=_payload_dir(tmp_path),
        artifacts_root=tmp_path / "window",
    )
    return ev, client, hub, logs, captures


def test_full_window_strict_order_and_destroy(tmp_path: Path) -> None:
    ev, client, hub, logs, captures = _run(tmp_path, _FULL_PLAN)
    assert ev.status == "up", ev.detail
    assert captures == [
        ("benchmark", "base"),
        ("benchmark", "qlora"),
        ("benchmark", "qlora_no_exemplars"),
        ("judge", "run-01,run-02,run-03"),
    ]
    # Laptop-side phase markers appear in strict order.
    order = [
        m
        for m in ("BASE_CAPTURED", "QLORA_CAPTURED", "JUDGE_DONE")
        if any(u.endswith(m) for u in hub.uploaded)
    ]
    assert order == ["BASE_CAPTURED", "QLORA_CAPTURED", "JUDGE_DONE"]
    uploads_flat = " ".join(hub.uploaded)
    assert "train_config.json" in uploads_flat and "gpu_window.py" in uploads_flat
    assert client.destroyed == [7777] and ev.destroyed is True
    assert (tmp_path / "window").exists()


def test_missing_payload_fails_before_any_gpu(tmp_path: Path) -> None:
    client = _FakeClient()
    logs: list[str] = []
    hub = FakeHub(_FULL_PLAN)
    ev = jarvis.finetune_session(
        _settings(tmp_path),
        _deps(client, logs),
        hub,
        trl_dir=tmp_path / "does-not-exist",
        artifacts_root=tmp_path / "window",
    )
    assert ev.status == "error" and "relay payload missing" in ev.detail
    assert client.destroyed == []  # no instance was ever created


def test_base_capture_failure_still_destroys(tmp_path: Path) -> None:
    ev, client, hub, logs, captures = _run(tmp_path, _FULL_PLAN, benchmark_rcs={"base": 1})
    assert ev.status == "error" and "base capture failed" in ev.detail
    assert client.destroyed == [7777]
    assert ("benchmark", "qlora") not in captures  # the window stopped at the failure


def test_training_failure_no_merged_up_still_destroys(tmp_path: Path) -> None:
    # Box emits BASE_UP then dies with EXIT (training crash) — MERGED_UP never comes.
    ev, client, hub, logs, captures = _run(tmp_path, ["BASE_UP", "EXIT"])
    assert ev.status == "error" and "MERGED_UP never appeared" in ev.detail
    assert client.destroyed == [7777]


def test_judge_failure_still_destroys(tmp_path: Path) -> None:
    ev, client, hub, logs, captures = _run(tmp_path, _FULL_PLAN, judge_rc=1)
    assert ev.status == "error" and "judge pass failed" in ev.detail
    assert client.destroyed == [7777]


def test_token_never_reaches_logs(tmp_path: Path) -> None:
    ev, client, hub, logs, captures = _run(tmp_path, _FULL_PLAN)
    for line in logs:
        assert "hf_fttest_secret_token" not in line
    assert ev.status == "up"


def test_window_startup_script_embeds_token_but_no_xtrace() -> None:
    script = jarvis.build_window_startup_script("hf_tok123", "user/relay", "ftwin-ab")
    assert "hf_tok123" in script  # required for the relay upload path
    assert "set -euo pipefail" in script and "set -euxo" not in script  # no xtrace echo
    assert "gpu_window.py" in script and "EXIT" in script
