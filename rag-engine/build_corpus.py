"""Build the retrieval corpus (P2 task 3): gate-visible plot chunks + metadata cards.

uv run python rag-engine/build_corpus.py            # all three ablation configs
uv run python rag-engine/build_corpus.py --config 512tok_15pct
"""

from __future__ import annotations

import typer

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.rag.chunking import CHUNK_CONFIGS
from sutradhar.rag.corpus import build_corpus

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: list[str] = typer.Option(  # noqa: B008 — typer idiom
        [], "--config", help="Chunk config name(s); default = all ablation configs."
    ),
) -> None:
    known = {c.name: c for c in CHUNK_CONFIGS}
    if unknown := [name for name in config if name not in known]:
        raise typer.BadParameter(f"unknown config(s) {unknown}; choose from {sorted(known)}")
    selected = tuple(known[name] for name in config) if config else CHUNK_CONFIGS

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        report = build_corpus(session, selected)
        session.commit()
    engine.dispose()

    typer.echo(
        f"corpus: {report.versions_seen} gate-visible versions, "
        f"{report.plot_docs} plot docs, {report.cards_written} metadata cards"
    )
    for name, count in report.plot_chunks.items():
        typer.echo(f"  {name}: {count} plot chunks")
    if report.versions_without_plots:
        typer.echo("versions without plot text (metadata card only):", err=True)
        for v in report.versions_without_plots:
            typer.echo(f"  - {v}", err=True)


if __name__ == "__main__":
    app()
