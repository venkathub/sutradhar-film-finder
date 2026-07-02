"""Validate the golden fixtures against the live graph (P1 task 14).

make golden-validate    # exit 1 on any invalid fixture
"""

from __future__ import annotations

import typer

from sutradhar.evals.golden import validate_all
from sutradhar.graph.db import create_graph_engine, create_session_factory

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        fixtures, issues = validate_all(session)
    engine.dispose()

    typer.echo(f"fixtures loaded: {len(fixtures)}")
    categories = sorted({f.id[:5] for f in fixtures})
    typer.echo(f"scenario categories covered: {', '.join(categories)}")
    if issues:
        typer.echo(f"INVALID — {len(issues)} issue(s):", err=True)
        for issue in issues:
            typer.echo(f"  {issue.fixture_id}: {issue.issue}", err=True)
        raise typer.Exit(1)
    typer.echo("all fixtures valid (golden-eligible, gate-visible, categories complete)")


if __name__ == "__main__":
    app()
