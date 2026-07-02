"""Ingest the Wikidata spine for the committed seed slice (P1_SPEC §2.3 step 1).

Snapshot-first: a live run persists raw responses under ``data/raw/wikidata/<UTC-stamp>/``
(hash-recorded) before touching the DB; ``--offline`` replays the latest snapshot so rebuilds
and CI never re-hit the API.

    uv run python data-pipeline/ingest_spine.py            # live fetch + snapshot + ingest
    uv run python data-pipeline/ingest_spine.py --offline  # replay latest snapshot
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.pipeline.seed import load_seed_slice
from sutradhar.pipeline.wikidata import (
    WikidataClient,
    ingest_spine,
    load_snapshot,
    parse_entity,
    write_snapshot,
)

app = typer.Typer(add_completion=False)

SNAPSHOT_ROOT = Path("data/raw/wikidata")


@app.command()
def main(
    offline: bool = typer.Option(  # noqa: B008 — typer idiom
        False, help="Replay the latest snapshot; no API calls."
    ),
    snapshot_root: Path = typer.Option(  # noqa: B008 — typer idiom
        SNAPSHOT_ROOT, help="Snapshot base directory."
    ),
) -> None:
    slice_ = load_seed_slice()
    qids = sorted(
        {
            v.wikidata_qid
            for w in slice_.works.values()
            for v in w.versions.values()
            if v.wikidata_qid
        }
        | {w.wikidata_qid for w in slice_.works.values() if w.wikidata_qid}
    )

    if offline:
        snapshots = sorted(d for d in snapshot_root.iterdir() if d.is_dir())
        if not snapshots:
            typer.echo(f"no snapshot under {snapshot_root}; run live first", err=True)
            raise typer.Exit(1)
        snap_dir = snapshots[-1]
        payload = load_snapshot(snap_dir, "entities")
        discovered = load_snapshot(snap_dir, "sparql_backlinks").get("qids", [])
        typer.echo(f"replaying snapshot {snap_dir}")
    else:
        snap_dir = snapshot_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        client = WikidataClient()
        try:
            payload = client.get_entities(qids)
            discovered = client.discover_backlinks(qids)
        finally:
            client.close()
        write_snapshot(snap_dir, "entities", payload)
        write_snapshot(snap_dir, "sparql_backlinks", {"qids": discovered})
        typer.echo(f"snapshot written to {snap_dir}")

    entities = {qid: parse_entity(raw) for qid, raw in payload.items()}
    new_qids = sorted(set(discovered) - set(qids))

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        report = ingest_spine(session, slice_, entities)
        session.commit()
    engine.dispose()

    typer.echo(
        f"works upserted:    {report.works_upserted}\n"
        f"versions upserted: {report.versions_upserted}\n"
        f"edges written:     {report.edges_written}"
        + "".join(f"\n  + {label}" for label in report.edge_labels)
        + f"\nconflicts opened:  {report.conflicts_opened}"
    )
    if new_qids:
        typer.echo(
            "discovered (P144/P4969 backlinks) NOT in slice — review for conditional add "
            f"(§7 Q1): {', '.join(new_qids)}"
        )
    if report.discovered_unmatched_qids:
        typer.echo(f"unmatched edge targets (kept out): {report.discovered_unmatched_qids}")


if __name__ == "__main__":
    app()
