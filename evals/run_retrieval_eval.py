"""Run the retrieval eval + ablation grid (P2 task 10) and write the committed artifact.

make retrieval-eval        # uses RETRIEVAL_RUN; writes evals/retrieval_runs/<run_id>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sutradhar.config import Settings
from sutradhar.evals.retrieval import run_retrieval_eval
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.rag.artifacts import DEFAULT_ARTIFACTS_ROOT

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: str = typer.Option("", "--run-id", help="Artifact run id; default RETRIEVAL_RUN."),
    artifacts_root: Path = typer.Option(  # noqa: B008 — typer idiom
        DEFAULT_ARTIFACTS_ROOT, "--artifacts-root"
    ),
    out_dir: Path = typer.Option(Path("evals/retrieval_runs"), "--out-dir"),  # noqa: B008
) -> None:
    settings = Settings()
    resolved = run_id or settings.retrieval_run
    if not resolved:
        raise typer.BadParameter("no run id: pass --run-id or set RETRIEVAL_RUN in .env")

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        artifact = run_retrieval_eval(session, artifacts_root, resolved)
    engine.dispose()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{resolved}.json"
    out.write_text(artifact.model_dump_json(indent=1) + "\n", encoding="utf-8")

    typer.echo(f"committed run artifact: {out} ({out.stat().st_size / 1e6:.2f} MB)")
    typer.echo(
        f"{'config':<22} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>7} "
        f"{'VSR-01':>7} {'VSR-06':>7}"
    )
    for key, m in sorted(artifact.metrics.items()):
        slices = m["slices"]
        r1 = sum(s["recall@1"] * s["n"] for s in slices.values())
        r5 = sum(s["recall@5"] * s["n"] for s in slices.values())
        n = sum(s["n"] for s in slices.values())
        typer.echo(
            f"{key:<22} {r1 / n:>6.3f} {r5 / n:>6.3f} {m['recall@10']:>6.3f} "
            f"{m['mrr@10']:>7.3f} {m['version_set_recall_gs01']:>7.3f} "
            f"{m['version_set_recall_gs06']:>7.3f}"
        )
    typer.echo(f"winner: {artifact.winner}")
    winner_metrics = artifact.metrics[str(artifact.winner)]
    typer.echo(json.dumps(winner_metrics["slices"], indent=1))


if __name__ == "__main__":
    app()
