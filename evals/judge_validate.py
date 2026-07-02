"""Judge human-agreement CLI (P3 task 7; DEC-P3-1, P3_SPEC §2.5). Thin Typer wrapper —
all logic lives (typed + unit-tested) in ``sutradhar.evals.judge_validation``.

  uv run python evals/judge_validate.py generate   # worksheet from the committed gen run
  uv run python evals/judge_validate.py report     # judge pass + κ (needs JUDGE_BASE_URL)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from sutradhar.config import get_settings
from sutradhar.evals.driver import FixtureTranscript
from sutradhar.evals.judge import JudgeClient
from sutradhar.evals.judge_validation import (
    VALIDATION_DIR,
    build_worksheet,
    compute_report,
    load_worksheet,
    save_report,
    save_worksheet,
)

app = typer.Typer(add_completion=False)

GENERATION_RUNS_DIR = Path("evals/generation_runs")


def _load_transcripts(run_path: Path | None) -> list[FixtureTranscript]:
    if run_path is None:
        runs = sorted(GENERATION_RUNS_DIR.glob("*.json"))
        if not runs:
            typer.echo(
                "no committed generation run found — run `make generation-dryrun` first "
                f"(looked in {GENERATION_RUNS_DIR})"
            )
            raise typer.Exit(code=1)
        run_path = runs[-1]
    payload = json.loads(run_path.read_text("utf-8"))
    transcripts = [FixtureTranscript.model_validate(f["transcript"]) for f in payload["fixtures"]]
    typer.echo(f"loaded {len(transcripts)} transcripts from {run_path}")
    return transcripts


@app.command()
def generate(
    run: Path | None = typer.Option(None, help="Generation-run artifact"),  # noqa: B008
    out_dir: Path = typer.Option(VALIDATION_DIR, help="Worksheet dir"),  # noqa: B008 — typer idiom
) -> None:
    """Build the ~24-item human-labelling worksheet (+ blind foil key) from transcripts."""
    items, key = build_worksheet(_load_transcripts(run))
    path = save_worksheet(items, key, out_dir)
    foils = sum(1 for v in key.values() if v["is_foil"])
    typer.echo(f"wrote {path}: {len(items)} items ({foils} foils) — fill every human_label")


@app.command()
def report(
    worksheet_dir: Path = typer.Option(VALIDATION_DIR, help="Worksheet dir"),  # noqa: B008 — typer idiom
) -> None:
    """Score the labelled worksheet with the pinned judge; write the κ agreement report."""
    settings = get_settings()
    judge = JudgeClient(settings)
    if not judge.available:
        typer.echo(
            "judge off — set JUDGE_BASE_URL / JUDGE_MODEL (serve it via `make gpu-judge`, "
            "DEC-P3-1); skipping cleanly."
        )
        raise typer.Exit(code=0)  # first-class off path, not a failure
    items = load_worksheet(worksheet_dir)
    agreement = compute_report(items, judge)
    path = save_report(agreement, worksheet_dir)
    errors = sum(1 for v in agreement.verdicts if v.judge_error)
    typer.echo(f"judged {agreement.n_items} items ({errors} judge_error)")
    typer.echo(
        f"percent agreement = {agreement.percent_agreement:.3f}; "
        f"Cohen's kappa = {agreement.cohens_kappa:.3f}"
    )
    typer.echo(agreement.gate)
    typer.echo(f"report -> {path}")
    if agreement.cohens_kappa < 0.6:
        sys.exit(3)  # distinct code: gate failed (rubric revision / escalation per DEC-P3-1)


if __name__ == "__main__":
    app()
