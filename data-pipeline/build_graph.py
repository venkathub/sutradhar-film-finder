"""Run the graph builder (P1_SPEC §2.3 step 5): dub-vs-remake cross-check + dub-track
edge derivation + integrity report.

    uv run python data-pipeline/build_graph.py
"""

from __future__ import annotations

import typer

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.pipeline.build import build_graph

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        report = build_graph(session)
        session.commit()
    engine.dispose()

    typer.echo(
        f"graph: {report.works} works / {report.versions} versions / "
        f"{report.edges_total} edges\n"
        f"edges rule-checked:    {report.edges_checked}\n"
        f"  rule agrees:         {report.rule_agreements}\n"
        f"  conflicts opened:    {report.rule_conflicts_opened}\n"
        f"  insufficient leads:  {report.rule_insufficient_evidence}\n"
        f"dub edges derived:     {report.dub_edges_derived} (MEDIUM, rule-sourced)"
    )
    if report.anomalies:
        typer.echo("ANOMALIES:", err=True)
        for anomaly in report.anomalies:
            typer.echo(f"  - {anomaly}", err=True)


if __name__ == "__main__":
    app()
