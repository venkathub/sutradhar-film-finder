"""make ft-verdict — the 30-second demo: base-vs-QLoRA table + the frozen DEC-P4-8
keep/cut verdict, computed from COMMITTED artifacts with the GPU off.

Run ids come from ``finetune/window_runs.json`` (written at window publish, task 13) or
explicit ``--base/--qlora`` flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sutradhar.evals.generation_run import GenerationRunArtifact
from sutradhar.finetune.verdict import decide, extract_column, render_table

app = typer.Typer(add_completion=False)

RUNS_DIR = Path("evals/generation_runs")
POINTER = Path("finetune/window_runs.json")


def _load(run_id: str, runs_dir: Path) -> GenerationRunArtifact:
    path = runs_dir / f"{run_id}.json"
    return GenerationRunArtifact.model_validate_json(path.read_text(encoding="utf-8"))


@app.command()
def main(
    base: str = typer.Option("", help="Base-column run id (default: window_runs.json)."),
    qlora: str = typer.Option("", help="QLoRA-column run id (default: window_runs.json)."),
    runs_dir: Path = typer.Option(RUNS_DIR),  # noqa: B008 — typer idiom
    pointer: Path = typer.Option(POINTER),  # noqa: B008 — typer idiom
) -> None:
    if not (base and qlora):
        if not pointer.exists():
            typer.echo(
                f"no {pointer} yet (written at window publish) — pass --base/--qlora", err=True
            )
            raise typer.Exit(2)
        runs = json.loads(pointer.read_text(encoding="utf-8"))
        base = base or runs["base"]
        qlora = qlora or runs["qlora"]
    base_col = extract_column(_load(base, runs_dir))
    qlora_col = extract_column(_load(qlora, runs_dir))
    verdict = decide(base_col, qlora_col)
    typer.echo(render_table(base_col, qlora_col, verdict))
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
