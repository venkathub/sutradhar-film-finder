"""Integration tests: Wikipedia plot loading into real Postgres (P1 task 7)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import PlotText, Version
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.pipeline.wikipedia import WikiPage, load_plots, parse_page

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WD_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "wikidata" / "entities_sample.json"
WP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "wikipedia" / "pages_sample.json"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except Exception as exc:  # noqa: BLE001 — connection probe
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    with engine.connect() as conn:
        outer = conn.begin()
        s = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
        for table in (
            "chunk_embeddings",
            "chunks",
            "candidate_edges",
            "edges",
            "conflicts",
            "plot_texts",
            "version_cast",
            "version_title",
            "version",
            "person",
            "work",
        ):
            s.execute(text(f"DELETE FROM {table}"))
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


@pytest.fixture(scope="module")
def pages_by_qid() -> dict[str, list[WikiPage]]:
    raw = json.loads(WP_FIXTURE.read_text(encoding="utf-8"))
    out: dict[str, list[WikiPage]] = {}
    for key, entry in raw.items():
        qid = key.split("|", 1)[0]
        page = parse_page(entry["lang"], entry["response"])
        assert page is not None
        out.setdefault(qid, []).append(page)
    return out


def _ingest(session: Session) -> None:
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads(WD_FIXTURE.read_text(encoding="utf-8"))
    ingest_spine(session, slice_, {qid: parse_entity(e) for qid, e in wd.items()})


def test_plots_loaded_revision_pinned_and_licensed(
    session: Session, pages_by_qid: dict[str, list[WikiPage]]
) -> None:
    _ingest(session)
    report = load_plots(session, pages_by_qid)
    assert report.rows_new == 4  # Drishyam en+ml, Baahubali en, Devadasu en
    rows = session.scalars(select(PlotText)).all()
    assert len(rows) == 4
    for row in rows:
        assert row.revision_id and row.revision_id.isdigit()
        assert row.license == "CC BY-SA 4.0"
        assert row.source_url and row.source == "wikipedia"
        assert row.retrieved_at is not None
    # Both languages attached to the SAME Drishyam version (en + ml rows).
    drishyam = session.scalars(select(Version).where(Version.wikidata_qid == "Q15401703")).one()
    langs = {r.language for r in rows if r.version_id == drishyam.version_id}
    assert langs == {"en", "ml"}


def test_plots_idempotent_and_repin_on_revision_change(
    session: Session, pages_by_qid: dict[str, list[WikiPage]]
) -> None:
    _ingest(session)
    load_plots(session, pages_by_qid)
    again = load_plots(session, pages_by_qid)
    assert again.rows_new == 0 and again.rows_repinned == 0 and again.rows_unchanged == 4

    # Simulate an article edit: same page, new revision id → text re-pinned, no new row.
    edited = {
        qid: [p.model_copy(update={"revision_id": p.revision_id + "9"}) for p in pages]
        for qid, pages in pages_by_qid.items()
    }
    repin = load_plots(session, edited)
    assert repin.rows_new == 0 and repin.rows_repinned == 4
    assert session.execute(text("SELECT count(*) FROM plot_texts")).scalar_one() == 4


def test_qidless_versions_reported_not_invented(
    session: Session, pages_by_qid: dict[str, list[WikiPage]]
) -> None:
    """Dub tracks without QIDs get no plot rows — reported as a gap, never guessed."""
    _ingest(session)
    report = load_plots(session, pages_by_qid)
    assert any("Baahubali" in v for v in report.versions_without_sitelink)
    qidless_with_plots = session.execute(
        text(
            "SELECT count(*) FROM plot_texts p JOIN version v USING (version_id) "
            "WHERE v.wikidata_qid IS NULL"
        )
    ).scalar_one()
    assert qidless_with_plots == 0
