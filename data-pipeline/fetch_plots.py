"""Fetch revision-pinned Wikipedia plot prose into plot_texts (P1_SPEC §2.3 step 4).

Article titles come from the Wikidata sitelinks already captured in the task-4 snapshot
(zero extra Wikidata calls). Per version: the enwiki article + the version's own-language
wiki article, when sitelinks exist. Snapshot-first; ``--offline`` replays.

    uv run python data-pipeline/fetch_plots.py             # live fetch + snapshot + load
    uv run python data-pipeline/fetch_plots.py --offline   # replay latest snapshot
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from sqlalchemy import select

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.schema import Version
from sutradhar.pipeline.snapshots import latest_snapshot_dir, load_snapshot, write_snapshot
from sutradhar.pipeline.wikipedia import (
    WikipediaClient,
    load_plots,
    parse_page,
    sitelinks_from_entities,
)

app = typer.Typer(add_completion=False)

SNAPSHOT_ROOT = Path("data/raw/wikipedia")
WIKIDATA_SNAPSHOT_ROOT = Path("data/raw/wikidata")


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
        versions = session.scalars(select(Version)).all()
        lang_by_qid = {v.wikidata_qid: v.language for v in versions if v.wikidata_qid}
        if not lang_by_qid:
            typer.echo("no versions with QIDs — run ingest-spine first", err=True)
            raise typer.Exit(1)

        if offline:
            snap_dir = latest_snapshot_dir(snapshot_root)
            payload = load_snapshot(snap_dir, "pages")
            typer.echo(f"replaying snapshot {snap_dir}")
        else:
            wd_snap = latest_snapshot_dir(WIKIDATA_SNAPSHOT_ROOT)
            entities = load_snapshot(wd_snap, "entities")
            sitelinks = sitelinks_from_entities(entities)
            typer.echo(f"sitelinks from {wd_snap} for {len(sitelinks)} QIDs")

            client = WikipediaClient()
            payload = {}
            try:
                for qid, lang in sorted(lang_by_qid.items()):
                    links = sitelinks.get(qid, {})
                    wanted = {"en", lang} & set(links)
                    for wiki_lang in sorted(wanted):
                        payload[f"{qid}|{wiki_lang}"] = {
                            "lang": wiki_lang,
                            "title": links[wiki_lang],
                            "response": client.get_page(wiki_lang, links[wiki_lang]),
                        }
            finally:
                client.close()
            snap_dir = snapshot_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            write_snapshot(snap_dir, "pages", payload)
            typer.echo(f"snapshot written to {snap_dir} ({len(payload)} pages)")

        pages_by_qid: dict[str, list[Any]] = {}
        for key, entry in payload.items():
            qid = key.split("|", 1)[0]
            page = parse_page(entry["lang"], entry["response"])
            if page is not None:
                pages_by_qid.setdefault(qid, []).append(page)

        report = load_plots(session, pages_by_qid)
        session.commit()
    engine.dispose()

    typer.echo(
        f"pages seen:      {report.pages_seen}\n"
        f"rows new:        {report.rows_new}\n"
        f"rows re-pinned:  {report.rows_repinned}\n"
        f"rows unchanged:  {report.rows_unchanged}"
    )
    if report.versions_without_sitelink:
        typer.echo(f"no QID/sitelink (skipped): {report.versions_without_sitelink}", err=True)


if __name__ == "__main__":
    app()
