"""Apply human conflict resolutions from a reviewed YAML (P4 task 3; P1_SPEC §1.5 gate).

P1 shipped zero open conflicts so no applier existed; the P4 training-slice ingest opened
the first one (Adithya Varma release_year, seed+TMDB=2019 vs stale Wikidata P577=2018).
Same batch-audit pattern as ``review_candidates.py --decisions``: the YAML is the audit
artifact of a human review session; this CLI only applies it — resolution is recorded on
the row (``status=resolved, resolution={by: human, ...}``), the row itself stays forever.

    uv run python data-pipeline/resolve_conflicts.py
    uv run python data-pipeline/resolve_conflicts.py --resolutions my_review.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from sqlalchemy import select

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.schema import Conflict, Version

app = typer.Typer(add_completion=False)

DEFAULT_RESOLUTIONS = Path("data-pipeline/conflict_resolutions.yaml")


@app.command()
def main(
    resolutions: Path = typer.Option(  # noqa: B008 — typer idiom
        DEFAULT_RESOLUTIONS, help="Reviewed resolutions YAML (the audit artifact)."
    ),
) -> None:
    payload: dict[str, Any] = yaml.safe_load(resolutions.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = payload.get("resolutions", [])
    if not entries:
        typer.echo("no resolutions in file — nothing to do")
        raise typer.Exit(0)

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    applied = skipped = 0
    with factory() as session:
        for entry in entries:
            version = session.scalars(
                select(Version).where(Version.wikidata_qid == entry["wikidata_qid"])
            ).first()
            if version is None:
                typer.echo(f"  ! no version for {entry['wikidata_qid']} — skipped", err=True)
                skipped += 1
                continue
            conflict = session.scalars(
                select(Conflict).where(
                    Conflict.entity_kind == "version",
                    Conflict.entity_id == version.version_id,
                    Conflict.field == entry["field"],
                    Conflict.status == "open",
                )
            ).first()
            if conflict is None:
                typer.echo(
                    f"  ! no open {entry['field']} conflict for {entry['wikidata_qid']} — skipped"
                )
                skipped += 1
                continue
            conflict.status = "resolved"
            conflict.resolution = {
                "by": "human",
                "reviewer": entry["reviewer"],
                "date": entry["date"],
                "chosen_value": entry["chosen_value"],
                "evidence": entry["evidence"].strip(),
            }
            if entry["field"] == "release_year":
                version.release_year = int(entry["chosen_value"])
            applied += 1
        session.commit()
    engine.dispose()
    typer.echo(f"resolutions applied: {applied} (skipped: {skipped})")


if __name__ == "__main__":
    app()
