"""P5 serving-benchmark capture steps (task 13, P5_SPEC §2.8 Tier-2 / §5.13).

The four laptop-side captures the GPU window runs against the live endpoints, plus the
sealed-artifact writer. Each step is a plain function so the jarvis session injects them
and tests fake them; a step failure is recorded in the artifact, never hidden — the
window seals whatever evidence it gathered and the instance is destroyed regardless.

Steps:
1. :func:`parity_check` — the 13 retrieval fixtures through the LIVE providers
   (HttpEmbeddings/HttpReranker → sidecar); Recall@10 + VSR GS-01/06 must match Table 1.
2. :func:`injection_capture` — real-model ASR defenses on/off + utility-under-attack.
3. :func:`latency_capture` — API e2e p50/p95 + tokens/sec + a vLLM /metrics snapshot.
4. :func:`relevancy_backfill` — RAGAS answer_relevancy over the pinned base generation
   run (discharges the BENCHMARKS footnote-¹ gap).

Artifact: ``evals/serving_runs/<run_id>.json`` + ``<run_id>.MANIFEST.sha256`` carrying the
§6.1 reproducibility stamp (code SHA, prompt v1.1 hash, tool-schema v0 sha256,
model@revision, vLLM flags, serving-run id).
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sutradhar.config import Settings
from sutradhar.evals.prompts import load_serving_prompt_artifacts
from sutradhar.serving.orchestrator import Orchestrator
from sutradhar.serving.schemas import ChatResponse
from sutradhar.toolcalls import load_tool_schema

SERVING_RUNS_DIR = Path("evals/serving_runs")

# A handful of real turns for the latency capture (the gating story + code-mix).
LATENCY_PROBES = (
    "which movie is Papanasam a remake of?",
    "wo film jisme baap evidence chhupa ke family ko bachata hai",
    "show me every Drishyam film",
)


class StepResult(BaseModel):
    """One capture step's outcome (ok flag + payload or error — never a raised exception)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] = {}
    error: str | None = None


def _guarded(fn: Any) -> StepResult:
    try:
        return StepResult(ok=True, data=fn())
    except Exception as exc:  # noqa: BLE001 — a step failure is evidence, not a crash
        return StepResult(ok=False, error=f"{type(exc).__name__}: {exc}"[:400])


# --- Step 1: live-path parity check ---


def parity_check(session: Any, settings: Settings, artifacts_root: Path) -> StepResult:
    """Re-validate Table 1's gate numbers through the live GPU providers (§2.8)."""
    from sutradhar.evals.retrieval import run_retrieval_eval
    from sutradhar.rag.providers import HttpEmbeddings, HttpReranker, chunk_text_lookup

    def run() -> dict[str, Any]:
        run_id = settings.require("retrieval_run")
        # Derive the winner cell from the committed artifact and re-validate ONLY that
        # config/depth through live providers (bounds GPU calls; it is the Table 1 gate).
        recorded = json.loads(
            (Path("evals/retrieval_runs") / f"{run_id}.json").read_text(encoding="utf-8")
        )
        winner_key = recorded["winner"]  # e.g. "1024tok_15pct/d20"
        chunk_config, depth_str = winner_key.split("/d")
        embedder = HttpEmbeddings(settings.require("embed_base_url"), settings.embed_model)
        reranker = HttpReranker(
            settings.require("rerank_base_url"), settings.rerank_model, chunk_text_lookup(session)
        )
        artifact = run_retrieval_eval(
            session,
            artifacts_root,
            run_id,
            providers=(embedder, reranker),
            chunk_configs=(chunk_config,),
            rerank_depths=(int(depth_str),),
        )
        winner = artifact.winner
        assert winner is not None
        m = artifact.metrics[winner]
        return {
            "retrieval_run": run_id,
            "winner": winner,
            "recall@10": m["recall@10"],
            "version_set_recall_gs01": m["version_set_recall_gs01"],
            "version_set_recall_gs06": m["version_set_recall_gs06"],
            "matches_table1_gate": (
                m["recall@10"] >= 0.90
                and m["version_set_recall_gs01"] == 1.0
                and m["version_set_recall_gs06"] == 1.0
            ),
        }

    return _guarded(run)


# --- Step 2: injection ASR (real model), defenses on/off + utility ---


def injection_capture() -> StepResult:
    """Live injection suite: ASR/FP/utility with defenses ON, plus the OFF baseline."""
    import sys
    from importlib import import_module

    def run() -> dict[str, Any]:
        # The runner is a top-level module (evals/), not under the package.
        sys.path.insert(0, str(Path("evals").resolve()))
        runner = import_module("run_injection_eval")
        on = runner.run_injection_suite("live", True)
        off = runner.run_injection_suite("live", False)
        return {"defenses_on": on, "defenses_off": off}

    return _guarded(run)


# --- Step 3: API e2e latency / tokens + vLLM /metrics snapshot ---


def latency_capture(
    orchestrator: Orchestrator,
    settings: Settings,
    *,
    probes: tuple[str, ...] = LATENCY_PROBES,
    http_get: Any = None,
) -> StepResult:
    """Drive real turns, capture p50/p95 latency + tokens/sec + a vLLM /metrics snapshot."""

    def run() -> dict[str, Any]:
        latencies: list[float] = []
        tps: list[float] = []
        completed = 0
        for probe in probes:
            outcome = orchestrator.run_turn(None, probe)
            if isinstance(outcome, ChatResponse):
                completed += 1
                latencies.append(outcome.latency_ms)
                secs = outcome.latency_ms / 1000.0
                if secs > 0 and outcome.usage.completion_tokens:
                    tps.append(outcome.usage.completion_tokens / secs)
        return {
            "turns": len(probes),
            "completed": completed,
            "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
            "latency_p95_ms": (
                round(statistics.quantiles(latencies, n=20)[-1], 2) if len(latencies) > 1 else None
            ),
            "tokens_per_sec_mean": round(statistics.mean(tps), 2) if tps else None,
            "vllm_metrics": _vllm_metrics_snapshot(settings, http_get),
        }

    return _guarded(run)


def _vllm_metrics_snapshot(settings: Settings, http_get: Any) -> str | None:
    """One-shot GET of vLLM's Prometheus /metrics (TTFT/TPOT/queue) — richer than client
    timing; a standing Prometheus stack is deliberately NOT deployed (D6-B rejection)."""
    base = settings.llm_base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    getter = http_get or (lambda url: httpx.get(url, timeout=10.0).text)
    try:
        text = getter(f"{root}/metrics")
    except Exception as exc:  # noqa: BLE001 — snapshot is best-effort evidence
        return f"unavailable: {type(exc).__name__}"
    # Keep only vLLM's own gauges/counters (drop process_* noise) to bound the artifact.
    lines = [ln for ln in str(text).splitlines() if ln.startswith("vllm:")]
    return "\n".join(lines[:200]) if lines else str(text)[:4000]


# --- Step 4: answer_relevancy backfill over the pinned base run ---


def relevancy_backfill(settings: Settings, run_id: str | None) -> StepResult:
    """RAGAS answer_relevancy over the pinned base generation run (judge+embed in-session).
    Discharges the BENCHMARKS footnote-¹ gap (embedding-backed relevancy returned null).
    Reuses the tested ``apply_ragas_scores`` path — no bespoke scoring."""
    from sutradhar.evals.generation_run import apply_ragas_scores, load_generation_run
    from sutradhar.evals.ragas_metrics import build_scorer

    def run() -> dict[str, Any]:
        artifact = load_generation_run(run_id=run_id)
        scorer, reason = build_scorer(settings)
        if scorer is None:
            raise RuntimeError(reason)
        apply_ragas_scores(artifact.fixtures, scorer)
        per_fixture = {
            r.fixture_id: (r.ragas.answer_relevancy if r.ragas else None) for r in artifact.fixtures
        }
        # Capture the per-fixture RAGAS error too — the P4 footnote-¹ null was recorded
        # WITHOUT the exception, forcing a blind re-run. Surface it so any residual null
        # is diagnosable from the sealed artifact (root-cause first, then recompute).
        errors = {
            r.fixture_id: (r.ragas.answer_relevancy_error if r.ragas else "no ragas block")
            for r in artifact.fixtures
            if not (r.ragas and r.ragas.answer_relevancy is not None)
        }
        present = [v for v in per_fixture.values() if v is not None]
        return {
            "generation_run": artifact.run_id,
            "per_fixture": per_fixture,
            "errors": errors,
            "mean_answer_relevancy": round(sum(present) / len(present), 4) if present else None,
            "n_scored": len(present),
            "n_total": len(per_fixture),
        }

    return _guarded(run)


# --- Sealed artifact ---


class ServingRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    code_sha: str | None
    prompt_hash: str  # v1.1 serving bundle
    tool_schema_version: str
    tool_schema_sha256: str
    model: str
    serving: dict[str, Any]  # vLLM flags, GPU type, endpoints
    parity: StepResult
    injection: StepResult
    latency: StepResult
    relevancy: StepResult

    @property
    def all_ok(self) -> bool:
        return all(s.ok for s in (self.parity, self.injection, self.latency, self.relevancy))


def build_artifact(
    run_id: str,
    settings: Settings,
    *,
    parity: StepResult,
    injection: StepResult,
    latency: StepResult,
    relevancy: StepResult,
    served_model: str | None = None,
    endpoints: dict[str, str] | None = None,
) -> ServingRunArtifact:
    from sutradhar.evals.generation_run import code_sha

    schema_path = Path("docs/phases/tool_schema.v0.json")
    return ServingRunArtifact(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        code_sha=code_sha(),
        prompt_hash=load_serving_prompt_artifacts().prompt_hash,
        tool_schema_version="v0",
        tool_schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        model=served_model or settings.llm_model,
        serving={
            "vllm_serve_flags": settings.vllm_serve_flags,
            "gpu_type": settings.gpu_type,
            "gpu_hourly_usd": settings.gpu_hourly_usd,
            "endpoints": endpoints or {},
            "n_tools": len(load_tool_schema()["tools"]),
        },
        parity=parity,
        injection=injection,
        latency=latency,
        relevancy=relevancy,
    )


def run_serve_captures(
    llm_base_url: str,
    embed_base_url: str,
    rerank_base_url: str,
    *,
    settings: Settings | None = None,
    artifacts_root: Path = Path("data/artifacts/retrieval"),
) -> dict[str, StepResult]:
    """Session-A captures against the serve window (Gemma + sidecar): parity, injection,
    latency. Points a Settings copy at the live endpoints; builds a real DB session +
    orchestrator. Returns the three StepResults (each already guarded)."""
    from sutradhar.graph.db import create_graph_engine, create_session_factory
    from sutradhar.rag.providers import HttpEmbeddings, HttpReranker, chunk_text_lookup
    from sutradhar.rag.retrieve import RetrievalConfig, Retriever
    from sutradhar.serving import guardrails
    from sutradhar.serving.executor import build_live_executor
    from sutradhar.serving.llm_client import LLMClient
    from sutradhar.serving.sessions import InMemorySessionStore

    base = settings or Settings()
    live = base.model_copy(
        update={
            "llm_base_url": llm_base_url,
            "embed_base_url": embed_base_url,
            "rerank_base_url": rerank_base_url,
        }
    )
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        parity = parity_check(session, live, artifacts_root)

        # Injection uses its own in-memory graph executor; only LLM_BASE_URL matters — set
        # it in the process env so run_injection_eval's LLMClient(Settings()) sees the window.
        import os

        os.environ["LLM_BASE_URL"] = llm_base_url
        injection = injection_capture()

        # Latency: real orchestrator against the seeded live graph + live retriever
        # (winner config from the committed run — identical to Table 1).
        run_id = live.require("retrieval_run")
        recorded = json.loads(
            (Path("evals/retrieval_runs") / f"{run_id}.json").read_text(encoding="utf-8")
        )
        winner_chunk, winner_depth = recorded["winner"].split("/d")
        cfg = RetrievalConfig(
            chunk_config=winner_chunk,
            embed_model=live.embed_model,
            index_version=run_id,
            rerank_depth=int(winner_depth),
        )
        retriever = Retriever(
            session,
            cfg,
            HttpEmbeddings(embed_base_url, live.embed_model),
            HttpReranker(rerank_base_url, live.rerank_model, chunk_text_lookup(session)),
        )
        artifacts = load_serving_prompt_artifacts()
        orchestrator = Orchestrator(
            LLMClient(live),
            InMemorySessionStore(3600),
            build_live_executor(session, retriever),
            system_prompt=artifacts.system_prompt(),
            prompt_hash=artifacts.prompt_hash,
            spotlight=guardrails.spotlight,
            output_gate=guardrails.output_gate,
        )
        latency = latency_capture(orchestrator, live)
    finally:
        session.close()
        engine.dispose()
    return {"parity": parity, "injection": injection, "latency": latency}


def run_relevancy_capture(
    judge_base_url: str,
    embed_base_url: str,
    *,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> StepResult:
    """Session-B capture: answer_relevancy over the pinned base run (judge + BGE-M3)."""
    base = settings or Settings()
    live = base.model_copy(
        update={"judge_base_url": judge_base_url, "embed_base_url": embed_base_url}
    )
    return relevancy_backfill(live, run_id)


def seal(artifact: ServingRunArtifact, out_dir: Path = SERVING_RUNS_DIR) -> Path:
    """Write ``<run_id>.json`` + a ``<run_id>.MANIFEST.sha256`` (the committed evidence)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact.run_id}.json"
    payload = artifact.model_dump_json(indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (out_dir / f"{artifact.run_id}.MANIFEST.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return path
