"""Calibrate the NO_MATCH abstention threshold from the committed run artifact
(P2 task 11, DEC-P2-5). No DB, no GPU — pure laptop math over recorded scores.

    make calibrate-no-match      # writes evals/retrieval_runs/<run_id>.calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sutradhar.config import Settings
from sutradhar.evals.calibration import calibrate, collect_inputs
from sutradhar.evals.retrieval import EvalRunArtifact

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: str = typer.Option("", "--run-id", help="Eval run id; default RETRIEVAL_RUN."),
    runs_dir: Path = typer.Option(Path("evals/retrieval_runs"), "--runs-dir"),  # noqa: B008
    config_key: str = typer.Option("", "--config", help="Grid cell; default = the winner."),
) -> None:
    settings = Settings()
    resolved = run_id or settings.retrieval_run
    if not resolved:
        raise typer.BadParameter("no run id: pass --run-id or set RETRIEVAL_RUN in .env")
    artifact = EvalRunArtifact.model_validate_json(
        (runs_dir / f"{resolved}.json").read_text(encoding="utf-8")
    )
    cell = config_key or str(artifact.winner)
    report = calibrate(collect_inputs(artifact, cell))

    out = runs_dir / f"{resolved}.calibration.json"
    out.write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")

    typer.echo(f"cell: {cell}")
    typer.echo(
        f"θ = {report.theta}  (= (1+{report.relative_margin}) × top calibration canary "
        f"{report.max_calibration_negative[0]}={report.max_calibration_negative[1]:.5f})"
    )
    typer.echo(
        f"zero-false-reject feasible: {report.zero_false_reject_feasible}"
        + (f"  (witness: {report.infeasibility_witness})" if report.infeasibility_witness else "")
    )
    typer.echo(f"GS-02 false accepts:  {report.gs02_false_accepts or 'NONE'}")
    typer.echo(f"test false accepts:   {report.test_false_accepts or 'NONE'}")
    typer.echo(
        f"test NO_MATCH recall: {report.test_no_match_recall}   "
        f"precision: {report.test_no_match_precision}"
    )
    typer.echo(f"documented positive false rejects: {report.positive_false_rejects}")
    typer.echo(f"curve points: {len(report.curve)} (full curve in {out})")
    typer.echo(json.dumps([p.model_dump() for p in report.curve[-6:]], indent=1))


if __name__ == "__main__":
    app()
