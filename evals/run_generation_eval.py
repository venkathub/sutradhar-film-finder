"""Generation benchmark runner (P3 task 9; P3_SPEC §1.1). Thin Typer CLI — models,
scoring, and aggregation live (typed + unit-tested) in ``sutradhar.evals.generation_run``.

  make generation-dryrun       # --mode dry_run: scripted mock endpoint, artifact committed
  make benchmark-generation    # --mode live: LLM_BASE_URL (the P4 GPU window)

Honesty rules baked in: dry_run artifacts carry serving=null and null latency/throughput
(validator-enforced) and are never published to Table 2; the authoritative base capture
happens at the top of the P4 GPU window with THIS harness (identical serving conditions
for both columns).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sutradhar.config import get_settings
from sutradhar.evals.driver import (
    RecordedPlotSearch,
    build_executor,
    load_retrieval_run,
    load_tool_schema,
)
from sutradhar.evals.generation_run import (
    GENERATION_RUNS_DIR,
    GenerationRunArtifact,
    aggregate_metrics,
    apply_judge_scores,
    apply_ragas_scores,
    build_stamp,
    execute_fixtures,
    make_run_id,
    select_generation_fixtures,
)
from sutradhar.evals.golden import load_fixtures
from sutradhar.evals.judge import COHERENCE_PROMPT, JudgeClient
from sutradhar.evals.prompts import load_prompt_artifacts
from sutradhar.evals.ragas_metrics import build_scorer, ragas_version
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.obs.mlflow_log import EXPERIMENT_GENERATION, log_generation_run
from sutradhar.obs.tracing import Tracer
from sutradhar.serving import LLMClient

app = typer.Typer(add_completion=False)


@app.command()
def main(
    mode: str = typer.Option("dry_run", "--mode", help="dry_run (mock) | live (LLM_BASE_URL)"),
    out_dir: Path = typer.Option(GENERATION_RUNS_DIR, "--out-dir"),  # noqa: B008 — typer idiom
    with_judge: bool = typer.Option(
        False, "--with-judge", help="Run the coherence judge pass (needs JUDGE_BASE_URL)."
    ),
    with_ragas: bool = typer.Option(
        False, "--with-ragas", help="Run the RAGAS pass (needs JUDGE_* + EMBED_BASE_URL)."
    ),
    serving_json: str = typer.Option(
        "", "--serving-json", help="Live runs: JSON blob of serving config for the stamp."
    ),
    prompt_variant: str = typer.Option(
        "full",
        "--prompt-variant",
        help="full = the frozen headline bundle; no_exemplars = the D6 supplementary "
        "capture (DEC-P4-6) — prompt_hash gets an explicit ':no_exemplars' suffix so it "
        "can never masquerade as the frozen stamp.",
    ),
) -> None:
    if mode not in ("dry_run", "live"):
        raise typer.BadParameter("mode must be dry_run or live")
    if prompt_variant not in ("full", "no_exemplars"):
        raise typer.BadParameter("prompt-variant must be full or no_exemplars")
    settings = get_settings()
    artifacts = load_prompt_artifacts()
    if prompt_variant == "no_exemplars":
        system_prompt = artifacts.system.rstrip() + "\n"
        prompt_hash = f"{artifacts.prompt_hash}:no_exemplars"
    else:
        system_prompt = artifacts.system_prompt()
        prompt_hash = artifacts.prompt_hash
    schema = load_tool_schema()
    fixtures = select_generation_fixtures(load_fixtures())
    typer.echo(f"[generation-eval] mode={mode}; {len(fixtures)} generation fixtures")

    # --- model under test ---
    if mode == "dry_run":
        try:
            import mock_llm  # evals/mock_llm.py — the canned-transcript player (task 11)
        except ImportError as exc:
            typer.echo(f"dry_run needs evals/mock_llm.py (P3 task 11): {exc}")
            raise typer.Exit(code=1) from exc
        client = mock_llm.build_mock_client(settings)
        model_id = "mock"
        serving = None
    else:
        client = LLMClient(settings)
        model_id = settings.llm_model
        serving = {
            "llm_base_url": settings.llm_base_url,
            "gpu_type": settings.gpu_type,
            "captured_by": "make benchmark-generation",
            "prompt_variant": prompt_variant,
        }
        if serving_json:
            serving.update(json.loads(serving_json))

    # --- tool surface: live graph + the pinned P2 retrieval replay (DEC-P3-8) ---
    retrieval = load_retrieval_run()
    plot_search = RecordedPlotSearch(retrieval)
    fixture_ref: dict[str, str] = {"fixture_id": ""}
    tracer = Tracer(settings)  # no-op unless LANGFUSE_* set (DEC-P3-6)
    if tracer.enabled:
        typer.echo(f"Langfuse tracing ON -> {settings.langfuse_host}")
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        executor = build_executor(session, plot_search, fixture_ref)
        results = execute_fixtures(
            fixtures,
            client,
            system_prompt=system_prompt,
            prompt_hash=prompt_hash,
            execute_tool=executor,
            fixture_id_ref=fixture_ref,
            schema=schema,
            tracer=tracer,
            log=typer.echo,
        )
    engine.dispose()

    # --- optional GPU-session enrichment passes (skip cleanly when off) ---
    judge_block = None
    if with_judge:
        judge = JudgeClient(settings, tracer=tracer)
        if not judge.available:
            typer.echo("judge off — set JUDGE_BASE_URL/JUDGE_MODEL; skipping coherence pass")
        else:
            judged = apply_judge_scores(results, judge)
            judge_block = {
                "coherence": judge.config(
                    COHERENCE_PROMPT, ragas_version=ragas_version()
                ).model_dump()
            }
            typer.echo(f"judge coherence pass: {judged} fixtures judged")
    if with_ragas:
        scorer, reason = build_scorer(settings)
        if scorer is None:
            typer.echo(f"{reason}; skipping RAGAS pass")
        else:
            scored = apply_ragas_scores(results, scorer)
            typer.echo(f"RAGAS pass: {scored} fixtures scored")

    metrics = aggregate_metrics(results, mode)  # type: ignore[arg-type]
    artifact = GenerationRunArtifact(
        run_id=make_run_id(),
        mode=mode,  # type: ignore[arg-type]
        model=model_id,
        serving=serving,
        prompt_hash=prompt_hash,
        tool_schema_version="v0",
        judge=judge_block,
        retrieval_run=retrieval.run_id,
        fixtures=results,
        metrics=metrics,
        stamp=build_stamp(prompt_hash=prompt_hash, retrieval_run=retrieval.run_id),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{artifact.run_id}.json"
    out.write_text(artifact.model_dump_json(indent=1) + "\n", encoding="utf-8")
    typer.echo(f"committed generation-run artifact: {out}")
    if mode == "live" and metrics.fixtures_completed == 0:
        typer.echo(
            f"FATAL: 0/{metrics.fixtures_total} fixtures completed against the live "
            "endpoint — the serving config is broken (tool parser? model?); artifact "
            "kept for diagnostics but this run is NOT a benchmark column",
            err=True,
        )
        raise typer.Exit(4)

    # --- observability evidence (both degrade cleanly when off/unreachable) ---
    tracer.flush()
    if tracer.enabled:
        typer.echo(f"Langfuse trace: {tracer.trace_url() or tracer.last_trace_id}")
    try:
        mlflow_run = log_generation_run(artifact, out, settings=settings)
        typer.echo(
            f"MLflow run: {mlflow_run} ({EXPERIMENT_GENERATION} @ {settings.mlflow_tracking_uri})"
        )
    except Exception as exc:  # noqa: BLE001 — observability must never fail the eval
        typer.echo(
            f"MLflow logging skipped ({type(exc).__name__}) — start it with `make mlflow-up`"
        )

    m = metrics
    typer.echo(
        f"fixtures: {m.fixtures_completed}/{m.fixtures_total} completed | "
        f"tool-call seq {m.tool_call_sequence_accuracy} (call {m.tool_call_call_level}, "
        f"schema-validity {m.schema_validity})"
    )
    typer.echo(
        f"intent {m.intent_accuracy} (code-mixed {m.code_mixed_intent_accuracy}) | "
        f"slots micro-F1 {m.slot_micro_f1}"
    )
    typer.echo(
        f"faithfulness {m.faithfulness} ({m.inventions}/{m.titles_asserted} inventions) | "
        f"GS-02 inventions = {m.gs02_inventions} (gate: 0)"
    )
    if m.gs02_inventions != 0:
        typer.echo("GATE FAILED: hallucinated movie on the GS-02 slice")
        raise typer.Exit(code=3)
    typer.echo(f"pin it: GENERATION_RUN={artifact.run_id}")


if __name__ == "__main__":
    app()
