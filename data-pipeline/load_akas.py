"""Load slice-filtered IMDb ``title.akas`` rows into version_title (P1_SPEC §2.3 step 3).

The raw multi-GB dump is streamed and filtered on the fly — never stored, never committed.
Only the filtered rows are snapshotted (``data/raw/imdb/<UTC-stamp>/``).

    uv run python data-pipeline/load_akas.py                       # stream from datasets.imdbws.com
    uv run python data-pipeline/load_akas.py --akas-file dump.gz   # use a local pre-downloaded dump
    uv run python data-pipeline/load_akas.py --offline             # replay latest filtered snapshot
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from sqlalchemy import select

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.schema import Version
from sutradhar.pipeline.imdb import (
    download_and_filter_akas,
    filter_akas_stream,
    load_akas,
    rows_from_jsonable,
    rows_to_jsonable,
)
from sutradhar.pipeline.seed import load_seed_slice
from sutradhar.pipeline.snapshots import latest_snapshot_dir, load_snapshot, write_snapshot

app = typer.Typer(add_completion=False)

SNAPSHOT_ROOT = Path("data/raw/imdb")


@app.command()
def main(
    offline: bool = typer.Option(  # noqa: B008 — typer idiom
        False, help="Replay the latest filtered snapshot; no download."
    ),
    akas_file: Path | None = typer.Option(  # noqa: B008 — typer idiom
        None, help="Local title.akas.tsv.gz (skips the download)."
    ),
    snapshot_root: Path = typer.Option(  # noqa: B008 — typer idiom
        SNAPSHOT_ROOT, help="Snapshot base directory."
    ),
    slice_path: Path | None = typer.Option(  # noqa: B008 — typer idiom
        None,
        "--slice",
        help="Restrict to versions whose QIDs come from this slice YAML "
        "(P4 training run; default: all versions).",
    ),
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)

    slice_qids: set[str] | None = None
    if slice_path is not None:
        slice_ = load_seed_slice(slice_path)
        slice_qids = {qid for _, qid in slice_._iter_qids()}

    with factory() as session:
        id_rows = session.execute(select(Version.imdb_id, Version.wikidata_qid)).all()
        tconsts = {
            r.imdb_id
            for r in id_rows
            if r.imdb_id is not None and (slice_qids is None or r.wikidata_qid in slice_qids)
        }
        if not tconsts:
            typer.echo("no versions with imdb_id — run ingest-spine first", err=True)
            raise typer.Exit(1)

        if offline:
            snap_dir = latest_snapshot_dir(snapshot_root)
            rows = rows_from_jsonable(load_snapshot(snap_dir, "akas_filtered"))
            typer.echo(f"replaying snapshot {snap_dir} ({len(rows)} rows)")
        else:
            if akas_file is not None:
                with akas_file.open("rb") as fh:
                    rows = filter_akas_stream(fh, tconsts)
            else:
                typer.echo(f"streaming title.akas for {len(tconsts)} tconsts …")
                rows = download_and_filter_akas(tconsts)
            snap_dir = snapshot_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            write_snapshot(snap_dir, "akas_filtered", rows_to_jsonable(rows))
            typer.echo(f"filtered snapshot written to {snap_dir} ({len(rows)} rows)")

        report = load_akas(session, rows)
        session.commit()
    engine.dispose()

    typer.echo(
        f"rows seen:           {report.rows_seen}\n"
        f"titles new:          {report.titles_new} "
        f"(dub titles mapped: {report.dub_titles_mapped})\n"
        f"titles corroborated: {report.titles_corroborated}"
    )
    if report.unmatched_tconsts:
        typer.echo(f"unmatched tconsts: {report.unmatched_tconsts}", err=True)


if __name__ == "__main__":
    app()
