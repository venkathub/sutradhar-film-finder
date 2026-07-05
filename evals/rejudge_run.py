"""Re-judge a committed generation-run artifact (P4 task 10; window step [5]).

The finetune window captures BOTH benchmark columns BEFORE the judge is served (base and
merged models occupy the GPU first, §2.4) — so the judge + RAGAS pass runs afterwards
over the RECORDED transcripts, in the same session, with the same frozen judge
(DEC-P3-1). This CLI loads a committed artifact, applies the judge/RAGAS passes to its
stored fixture results, recomputes the metrics block, and writes the artifact back in
place (same run_id — the judge block records what scored it).

    JUDGE_BASE_URL=… JUDGE_MODEL=… uv run python evals/rejudge_run.py --run <run_id> --with-ragas
"""

from __future__ import annotations

from pathlib import Path

import typer

from sutradhar.config import get_settings
from sutradhar.evals.generation_run import (
    GenerationRunArtifact,
    aggregate_metrics,
    apply_judge_scores,
    apply_ragas_scores,
)
from sutradhar.evals.judge import COHERENCE_PROMPT, JudgeClient
from sutradhar.evals.ragas_metrics import build_scorer, ragas_version

app = typer.Typer(add_completion=False)

RUNS_DIR = Path("evals/generation_runs")


@app.command()
def main(
    run: str = typer.Option(..., "--run", help="Run id (evals/generation_runs/<id>.json)."),
    runs_dir: Path = typer.Option(RUNS_DIR),  # noqa: B008 — typer idiom
    with_ragas: bool = typer.Option(False, "--with-ragas"),
) -> None:
    settings = get_settings()
    path = runs_dir / f"{run}.json"
    artifact = GenerationRunArtifact.model_validate_json(path.read_text(encoding="utf-8"))

    judge = JudgeClient(settings)
    if not judge.available:
        typer.echo("judge off — set JUDGE_BASE_URL/JUDGE_MODEL", err=True)
        raise typer.Exit(1)
    judged = apply_judge_scores(artifact.fixtures, judge)
    judge_block = {
        "coherence": judge.config(COHERENCE_PROMPT, ragas_version=ragas_version()).model_dump()
    }
    typer.echo(f"judge coherence pass: {judged} fixtures judged")

    if with_ragas:
        scorer, reason = build_scorer(settings)
        if scorer is None:
            typer.echo(f"{reason}; skipping RAGAS pass")
        else:
            scored = apply_ragas_scores(artifact.fixtures, scorer)
            typer.echo(f"RAGAS pass: {scored} fixtures scored")

    updated = artifact.model_copy(
        update={
            "judge": judge_block,
            "metrics": aggregate_metrics(artifact.fixtures, artifact.mode),
        }
    )
    path.write_text(updated.model_dump_json(indent=1) + "\n", encoding="utf-8")
    typer.echo(f"re-judged artifact written back: {path}")


if __name__ == "__main__":
    app()
