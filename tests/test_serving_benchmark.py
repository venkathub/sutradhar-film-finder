"""P5 task 13 — serving-benchmark window orchestration (fake Deps, no GPU/network).

The critical guarantees: both sessions destroy their instance on every path (incl. a
capture crash), the artifact seals whatever evidence was gathered (partial evidence
beats no evidence), and the sealed shape carries the §6.1 reproducibility stamp.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sutradhar.config import Settings
from sutradhar.serving.benchmark import (
    StepResult,
    build_artifact,
    latency_capture,
    seal,
)
from sutradhar.serving.llm_client import EndpointStatus
from sutradhar.serving.schemas import ChatResponse, TurnAborted, Usage

_JARVIS_PATH = Path(__file__).resolve().parents[1] / "infra" / "gpu" / "jarvis.py"


def _load_jarvis() -> Any:
    spec = importlib.util.spec_from_file_location("jarvis_bench_under_test", _JARVIS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jarvis = _load_jarvis()


class _FakeInstance:
    def __init__(self, machine_id: int) -> None:
        self.machine_id = machine_id
        self.public_ip = ""  # proxy-only host (notebooksn), like the real box
        # Two proxy URLs (no port in them) — the LLM and the embed/rerank sidecar.
        self.endpoints = [
            f"https://llm{machine_id}.notebooksn.jarvislabs.net",
            f"https://side{machine_id}.notebooksn.jarvislabs.net",
        ]
        self.name = jarvis.INSTANCE_NAME


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.destroyed: list[int] = []


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_MODEL="google/gemma-4-E4B-it",
        HF_TOKEN="hf_test",
        JUDGE_MODEL="openai/gpt-oss-20b",
        RETRIEVAL_RUN="run-1",
    )


def _deps(client: _FakeClient, **over: Any) -> Any:
    ticker = {"t": 0.0}
    counter = {"n": 0}

    def monotonic() -> float:
        ticker["t"] += 1.0
        return ticker["t"]

    def create_instance(c: Any, s: Any, script: str) -> _FakeInstance:
        counter["n"] += 1
        return _FakeInstance(4000 + counter["n"])

    def destroy(c: _FakeClient, mid: int) -> bool:
        c.destroyed.append(mid)
        return True

    defaults = dict(
        create_client=lambda s: client,
        close_client=lambda c: setattr(c, "closed", True),
        create_instance=create_instance,
        destroy_instance=destroy,
        list_instances=lambda c: [],
        probe_health=lambda url, timeout: True,
        run_smoke=lambda url, s: EndpointStatus(
            status="up",
            model="google/gemma-4-E4B-it",
            sample_token="x",
            latency_ms=10.0,
            detail="up",
        ),
        remove_scripts=lambda c: 0,
        monotonic=monotonic,
        sleep=lambda s: None,
        log=lambda m: None,
    )
    defaults.update(over)
    return jarvis.Deps(**defaults)


def _ok_captures(llm: str, embed: str, rerank: str) -> dict[str, Any]:
    return {
        "parity": StepResult(ok=True, data={"recall@10": 1.0, "matches_table1_gate": True}),
        "injection": StepResult(ok=True, data={"defenses_on": {"asr": 0.0}}),
        "latency": StepResult(ok=True, data={"latency_p50_ms": 900.0}),
    }


def _ok_relevancy(judge: str, embed: str) -> StepResult:
    return StepResult(ok=True, data={"mean_answer_relevancy": 0.81, "n_scored": 12})


def test_happy_path_two_sessions_sealed_and_destroyed(tmp_path: Path) -> None:
    client = _FakeClient()
    urls_seen: dict[str, Any] = {}

    def captures(llm: str, embed: str, rerank: str) -> dict[str, Any]:
        urls_seen.update(llm=llm, embed=embed, rerank=rerank)
        return _ok_captures(llm, embed, rerank)

    ev = jarvis.serving_benchmark_session(
        _settings(),
        _deps(client),
        run_captures=captures,
        run_relevancy=_ok_relevancy,
        health_timeout_s=50,
        poll_interval_s=1,
        out_dir=tmp_path,
    )
    assert ev.status == "up" and ev.destroyed is True
    assert client.destroyed == [4001, 4002]  # BOTH sessions torn down, in order
    assert client.closed is True
    # Live URLs threaded into the captures: LLM and sidecar are DISTINCT proxy URLs
    # (the 2026-07-05 lesson), rerank == embed (one sidecar process).
    assert urls_seen["llm"].startswith("https://llm4001")
    assert urls_seen["embed"].startswith("https://side4001")
    assert urls_seen["embed"] == urls_seen["rerank"]

    sealed = list(tmp_path.glob("servewin-*.json"))
    assert len(sealed) == 1
    artifact = json.loads(sealed[0].read_text())
    assert artifact["parity"]["ok"] and artifact["relevancy"]["ok"]
    assert artifact["prompt_hash"].startswith("98b3ece1")  # the v1.1 serving hash
    assert artifact["tool_schema_version"] == "v0" and artifact["tool_schema_sha256"]
    assert artifact["serving"]["gpu_type"] == "A100"
    # MANIFEST seals the payload byte-for-byte.
    manifest = (tmp_path / f"{artifact['run_id']}.MANIFEST.sha256").read_text()
    import hashlib

    assert manifest.split()[0] == hashlib.sha256(sealed[0].read_bytes()).hexdigest()


def test_capture_crash_still_tears_down_and_seals_partial(tmp_path: Path) -> None:
    client = _FakeClient()

    def boom(llm: str, embed: str, rerank: str) -> dict[str, Any]:
        raise RuntimeError("parity crashed mid-capture")

    ev = jarvis.serving_benchmark_session(
        _settings(),
        _deps(client),
        run_captures=boom,
        run_relevancy=_ok_relevancy,
        health_timeout_s=50,
        poll_interval_s=1,
        out_dir=tmp_path,
    )
    assert client.destroyed == [4001, 4002]  # session A destroyed despite the crash
    assert ev.status == "error"  # partial evidence is not a green run
    artifact = json.loads(next(tmp_path.glob("servewin-*.json")).read_text())
    assert artifact["parity"]["ok"] is False and "parity crashed" in artifact["parity"]["error"]
    assert artifact["relevancy"]["ok"] is True  # session B still ran and sealed


def test_session_a_failure_still_runs_session_b(tmp_path: Path) -> None:
    client = _FakeClient()
    calls = {"b": 0}

    def relevancy(judge: str, embed: str) -> StepResult:
        calls["b"] += 1
        return _ok_relevancy(judge, embed)

    # Session A's LLM never passes the chat smoke (gemma), session B's judge does —
    # keyed on the requested model id (A smokes gemma, B smokes the judge model).
    def smoke(url: str, s: Any) -> EndpointStatus:
        up = "gpt-oss" in s.llm_model  # only the judge session resolves an LLM
        return EndpointStatus(
            status="up" if up else "error",
            model=s.llm_model,
            sample_token="x" if up else None,
            latency_ms=9.0 if up else None,
            detail="up" if up else "chat 404",
        )

    ev = jarvis.serving_benchmark_session(
        _settings(),
        _deps(client, run_smoke=smoke),
        run_captures=_ok_captures,
        run_relevancy=relevancy,
        health_timeout_s=5,
        poll_interval_s=1,
        out_dir=tmp_path,
    )
    assert calls["b"] == 1  # session B still executed after A failed
    assert client.destroyed == [4001, 4002]  # nothing leaked
    artifact = json.loads(next(tmp_path.glob("servewin-*.json")).read_text())
    assert artifact["parity"]["ok"] is False  # session A never resolved an LLM
    assert artifact["relevancy"]["ok"] is True
    assert ev.destroyed is True


def test_latency_capture_math_and_abort_handling() -> None:
    class _Orch:
        def __init__(self) -> None:
            self.n = 0

        def run_turn(self, cid: Any, msg: str) -> Any:
            self.n += 1
            if self.n == 3:
                return TurnAborted(conversation_id="x", status="off", detail="died")
            return ChatResponse(
                conversation_id="x",
                answer="a",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                latency_ms=1000.0 * self.n,  # 1s, 2s
                tool_calls=1,
            )

    result = latency_capture(
        _Orch(),  # type: ignore[arg-type]
        _settings(),
        http_get=lambda url: "vllm:num_requests_running 0\nprocess_cpu_seconds_total 1\n",
    )
    assert result.ok
    assert result.data["completed"] == 2 and result.data["turns"] == 3
    assert result.data["latency_p50_ms"] == pytest.approx(1500.0)
    assert result.data["tokens_per_sec_mean"] == pytest.approx((50 / 1 + 50 / 2) / 2)
    # Snapshot keeps vLLM lines only (D6-B partial adoption).
    assert "vllm:num_requests_running" in result.data["vllm_metrics"]
    assert "process_cpu_seconds_total" not in result.data["vllm_metrics"]


def test_artifact_all_ok_property() -> None:
    ok = StepResult(ok=True)
    bad = StepResult(ok=False, error="x")
    artifact = build_artifact(
        "servewin-test",
        _settings(),
        parity=ok,
        injection=ok,
        latency=ok,
        relevancy=bad,
    )
    assert artifact.all_ok is False
    good = artifact.model_copy(update={"relevancy": ok})
    assert good.all_ok is True


def test_seal_writes_manifest(tmp_path: Path) -> None:
    ok = StepResult(ok=True)
    artifact = build_artifact(
        "servewin-seal", _settings(), parity=ok, injection=ok, latency=ok, relevancy=ok
    )
    path = seal(artifact, tmp_path)
    assert path.exists() and (tmp_path / "servewin-seal.MANIFEST.sha256").exists()


def test_phase_serve_reruns_only_session_a_and_merges(tmp_path: Path) -> None:
    """phase=serve + merge_run: re-run session A only, keep the sealed relevancy result."""
    # Seed a prior artifact with a good relevancy (session B evidence to preserve).
    ok = StepResult(ok=True, data={"mean_answer_relevancy": 0.57})
    stale = StepResult(ok=False, error="stale pre-fix")
    prior = build_artifact(
        "servewin-merge", _settings(), parity=stale, injection=stale, latency=stale, relevancy=ok
    )
    seal(prior, tmp_path)

    client = _FakeClient()
    calls = {"a": 0, "b": 0}

    def captures(llm: str, embed: str, rerank: str) -> dict[str, Any]:
        calls["a"] += 1
        return _ok_captures(llm, embed, rerank)

    def relevancy(judge: str, embed: str) -> StepResult:
        calls["b"] += 1  # must NOT be called
        return _ok_relevancy(judge, embed)

    ev = jarvis.serving_benchmark_session(
        _settings(),
        _deps(client),
        run_captures=captures,
        run_relevancy=relevancy,
        health_timeout_s=50,
        poll_interval_s=1,
        out_dir=tmp_path,
        phase="serve",
        merge_run="servewin-merge",
    )
    assert calls["a"] == 1 and calls["b"] == 0  # only session A ran
    assert client.destroyed == [4001]  # only one instance (session A)
    artifact = json.loads((tmp_path / "servewin-merge.json").read_text())
    assert artifact["parity"]["ok"] is True  # refreshed by the re-run
    assert artifact["relevancy"]["ok"] is True  # preserved from the merge base
    assert artifact["relevancy"]["data"]["mean_answer_relevancy"] == 0.57
    assert ev.status == "up"
