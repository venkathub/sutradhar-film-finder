"""The 30-second demo (P1 DoD): the Drishyam version set, straight from the ground-truth
views — original flagged, relationships labelled, per-claim sources cited.

    make graph-demo
"""

from __future__ import annotations

import typer
from sqlalchemy import text

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.repository import get_versions, resolve_title

app = typer.Typer(add_completion=False)


def _cite(sources: list[dict[str, str]]) -> str:
    return ", ".join(f"{s['source']}:{s['ref']}" for s in sources)


@app.command()
def main(
    title: str = typer.Argument("Papanasam", help="Any title, any script, any spelling."),
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        resolved = resolve_title(session, title)
        if not resolved.candidates:
            typer.echo(
                f"NO_MATCH — {title!r} resolves to nothing in the graph (and that's a feature)."
            )
            raise typer.Exit(0)
        top = resolved.candidates[0]
        typer.echo(
            f"'{title}' -> {top.matched_title} ({top.language}, {top.year}) "
            f"score={top.score}"
            + ("  [ambiguous: multiple works match — ask the user]" if resolved.ambiguous else "")
        )

        work_row = session.execute(
            text(
                "SELECT primary_title, first_release_year "
                "FROM ground_truth_works WHERE work_id = :w"
            ),
            {"w": str(top.work_id)},
        ).one()
        typer.echo(
            f"\nWork: {work_row.primary_title} ({work_row.first_release_year}) — "
            f"full franchise, Indian + foreign:\n"
        )
        result = get_versions(session, top.work_id, scope="all", include_sequels=True)
        for v in sorted(result.versions, key=lambda v: (v.year or 0, v.language)):
            flag = " ★ ORIGINAL" if v.is_original else ""
            rel = v.relationship or "(unverified)"
            cast = f"  cast: {', '.join(v.cast_lead[:3])}" if v.cast_lead else ""
            typer.echo(f"  {v.title} ({v.language}, {v.year})  [{rel}]{flag}{cast}")
            typer.echo(f"      sources: {_cite(v.sources)}")
    engine.dispose()


if __name__ == "__main__":
    app()
