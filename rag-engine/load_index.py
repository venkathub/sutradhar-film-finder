"""Load the pinned artifact run into pgvector (P2 task 7).

    uv run python rag-engine/load_index.py                 # uses RETRIEVAL_RUN from .env
    uv run python rag-engine/load_index.py --run-id <id>
"""

from __future__ import annotations

from pathlib import Path

import typer

from sutradhar.config import Settings
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.rag.artifacts import DEFAULT_ARTIFACTS_ROOT
from sutradhar.rag.index import load_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: str = typer.Option("", "--run-id", help="Artifact run id; default RETRIEVAL_RUN."),
    artifacts_root: Path = typer.Option(  # noqa: B008 — typer idiom
        DEFAULT_ARTIFACTS_ROOT, "--artifacts-root"
    ),
) -> None:
    settings = Settings()
    resolved = run_id or settings.retrieval_run
    if not resolved:
        raise typer.BadParameter("no run id: pass --run-id or set RETRIEVAL_RUN in .env")

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        report = load_index(session, artifacts_root, resolved)
        session.commit()
    engine.dispose()

    typer.echo(f"index loaded: run {report.run_id} ({report.embed_model})")
    for config in report.configs:
        unused = report.unused_bank_vectors[config]
        note = f" ({unused} bank vectors unused)" if unused else ""
        typer.echo(f"  {config}: {report.rows_loaded[config]} embeddings{note}")


if __name__ == "__main__":
    app()
