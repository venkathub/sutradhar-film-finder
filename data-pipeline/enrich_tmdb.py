"""Enrich seed-slice versions from TMDB (P1_SPEC §2.3 step 2).

One ``append_to_response=translations,alternative_titles,credits`` call per film (rate-limit
friendly). Snapshot-first; ``--offline`` replays the latest snapshot. Requires ``TMDB_API_KEY``
for live runs.

    uv run python data-pipeline/enrich_tmdb.py             # live fetch + snapshot + enrich
    uv run python data-pipeline/enrich_tmdb.py --offline   # replay latest snapshot
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from sqlalchemy import select

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.schema import Version
from sutradhar.pipeline.snapshots import latest_snapshot_dir, load_snapshot, write_snapshot
from sutradhar.pipeline.tmdb import TMDBClient, enrich_tmdb, parse_movie

app = typer.Typer(add_completion=False)

SNAPSHOT_ROOT = Path("data/raw/tmdb")


@app.command()
def main(
    offline: bool = typer.Option(  # noqa: B008 — typer idiom
        False, help="Replay the latest snapshot; no API calls."
    ),
    snapshot_root: Path = typer.Option(  # noqa: B008 — typer idiom
        SNAPSHOT_ROOT, help="Snapshot base directory."
    ),
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)

    with factory() as session:
        tmdb_ids = sorted(
            {v for v in session.scalars(select(Version.tmdb_id)).all() if v is not None}
        )
        if not tmdb_ids:
            typer.echo("no versions with tmdb_id — run ingest-spine first", err=True)
            raise typer.Exit(1)

        if offline:
            snap_dir = latest_snapshot_dir(snapshot_root)
            payload = load_snapshot(snap_dir, "movies")
            typer.echo(f"replaying snapshot {snap_dir}")
        else:
            snap_dir = snapshot_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            client = TMDBClient()
            try:
                payload = {str(tmdb_id): client.get_movie(tmdb_id) for tmdb_id in tmdb_ids}
            finally:
                client.close()
            write_snapshot(snap_dir, "movies", payload)
            typer.echo(f"snapshot written to {snap_dir} ({len(payload)} movies)")

        movies = {int(k): parse_movie(raw) for k, raw in payload.items()}
        report = enrich_tmdb(session, movies)
        session.commit()
    engine.dispose()

    typer.echo(
        f"versions enriched:  {report.versions_enriched}\n"
        f"titles written:     {report.titles_written} "
        f"(dub titles mapped: {report.dub_titles_mapped})\n"
        f"people upserted:    {report.people_upserted}\n"
        f"cast rows written:  {report.cast_rows_written}\n"
        f"conflicts recorded: {report.conflicts_recorded} (open: {report.conflicts_open})"
    )
    if report.missing_payloads:
        typer.echo(f"missing payloads for tmdb_ids: {report.missing_payloads}", err=True)


if __name__ == "__main__":
    app()
