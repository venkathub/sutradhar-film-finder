"""Print the graph coverage + extraction-lift report (P1_SPEC §2.3 step 8).

make graph-report            # human-readable
uv run python data-pipeline/graph_report.py --json   # machine-readable
"""

from __future__ import annotations

import typer

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.pipeline.report import build_report, render_report, report_to_json

app = typer.Typer(add_completion=False)


@app.command()
def main(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        report = build_report(session)
    engine.dispose()
    typer.echo(report_to_json(report) if as_json else render_report(report))
    if not report.flagship_coverage_ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
