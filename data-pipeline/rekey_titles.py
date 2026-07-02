"""Re-key + finish populating the version_title index (P1 task 8).

Idempotent finishing pass after the connectors (tasks 5–7):
1. every version's own canonical title (seed-sourced) is present in the index;
2. every row's ``match_key`` is recomputed with the full transliteration pipeline
   (tasks 5/6 wrote interim Latin-only keys);
3. every row's ``script`` column is populated by detection.

    uv run python data-pipeline/rekey_titles.py
"""

from __future__ import annotations

import typer
from sqlalchemy import select

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version, VersionTitle
from sutradhar.pipeline.normalize import detect_script, match_key
from sutradhar.pipeline.titles import upsert_version_title

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    seeded = rekeyed = scripted = 0

    with factory() as session:
        # 1. Seed canonical titles into the index (idempotent; merges if already there).
        for version in session.scalars(select(Version)).all():
            outcome = upsert_version_title(
                session,
                version.version_id,
                version.title,
                "canonical",
                version.language,
                [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
            )
            if outcome == "new":
                seeded += 1

        # 2/3. Recompute match_key + populate script on every row.
        for row in session.scalars(select(VersionTitle)).all():
            new_key = match_key(row.title)
            new_script = detect_script(row.title)
            if row.match_key != new_key:
                row.match_key = new_key
                rekeyed += 1
            if row.script != new_script:
                row.script = new_script
                scripted += 1
        session.commit()
    engine.dispose()

    typer.echo(
        f"canonical rows seeded: {seeded}\n"
        f"keys recomputed:       {rekeyed}\n"
        f"scripts populated:     {scripted}"
    )


if __name__ == "__main__":
    app()
