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
    from sutradhar.config import get_settings
    from sutradhar.evals.generation_run import load_generation_run

    if run_path is not None:
        payload = json.loads(run_path.read_text("utf-8"))
        transcripts = [
            FixtureTranscript.model_validate(f["transcript"]) for f in payload["fixtures"]
        ]
        typer.echo(f"loaded {len(transcripts)} transcripts from {run_path}")
        return transcripts
    try:
        artifact = load_generation_run(GENERATION_RUNS_DIR, get_settings().generation_run or None)
    except FileNotFoundError as exc:
        typer.echo(f"no committed generation run — run `make generation-dryrun` first ({exc})")
        raise typer.Exit(code=1) from exc
    typer.echo(f"loaded {len(artifact.fixtures)} transcripts from run {artifact.run_id}")
    return [f.transcript for f in artifact.fixtures]


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


@app.command()
def blind(
    worksheet_dir: Path = typer.Option(VALIDATION_DIR, help="Worksheet dir"),  # noqa: B008 — typer idiom
) -> None:
    """P7 task 17 (DEC-P7-6): write the blind second-pass worksheet (+ id-map key).

    Ids are re-minted and order reshuffled (recorded seed) so foil provenance and
    fixture pairing are invisible; read PROTOCOL.md before labelling.
    """
    from sutradhar.evals.judge_validation import (
        build_blind_worksheet,
        load_worksheet,
        save_blind_worksheet,
    )

    items = load_worksheet(worksheet_dir)
    blind_items, id_map = build_blind_worksheet(items)
    path = save_blind_worksheet(blind_items, id_map, worksheet_dir)
    typer.echo(f"wrote {path}: {len(blind_items)} blind items — read PROTOCOL.md, then label")


@app.command()
def testretest(
    worksheet_dir: Path = typer.Option(VALIDATION_DIR, help="Worksheet dir"),  # noqa: B008 — typer idiom
) -> None:
    """P7 task 17 (DEC-P7-6): intra-rater test-retest report from the labelled blind pass.

    Additive: report.json stays frozen; the judge leg is computed offline from its
    recorded per-item verdicts (no GPU, nothing re-scored).
    """
    from sutradhar.evals.judge_validation import (
        BLIND_KEY_FILE,
        REPORT_FILE,
        compute_testretest_report,
        load_blind_worksheet,
        load_worksheet,
        save_testretest_report,
    )

    items = load_worksheet(worksheet_dir)
    blind_items = load_blind_worksheet(worksheet_dir)
    id_map = json.loads((worksheet_dir / BLIND_KEY_FILE).read_text("utf-8"))
    frozen_path = worksheet_dir / REPORT_FILE
    judge_binaries: dict[str, int] | None = None
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text("utf-8"))
        judge_binaries = {v["item_id"]: int(v["judge_binary"]) for v in frozen["verdicts"]}
    report_ = compute_testretest_report(items, blind_items, id_map, judge_binaries)
    path = save_testretest_report(report_, worksheet_dir)
    typer.echo(f"n = {report_.n_items}; percent agreement = {report_.percent_agreement:.3f}")
    typer.echo(
        f"intra-rater kappa = {report_.intra_rater_kappa:.3f} "
        f"(real items only: {report_.intra_rater_kappa_real_items_only:.3f})"
    )
    if report_.second_pass_vs_judge_kappa is not None:
        typer.echo(f"second-pass vs judge kappa = {report_.second_pass_vs_judge_kappa:.3f}")
    typer.echo("FRAMING: intra-rater test-retest proxy — NOT a human-human ceiling (DEC-P7-6)")
    typer.echo(f"report -> {path}")


if __name__ == "__main__":
    app()
