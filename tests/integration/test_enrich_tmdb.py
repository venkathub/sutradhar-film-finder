"""Integration tests: TMDB enrichment into real Postgres (P1 task 5).

Chains the fixture-driven spine ingest (task 4) with fixture-driven TMDB enrichment —
no live API. Rollback-per-test as in the sibling integration modules.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import Conflict, Person, Version, VersionCast, VersionTitle
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.tmdb import TMDBMovie, enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WD_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "wikidata" / "entities_sample.json"
TMDB_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tmdb" / "movies_sample.json"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except OperationalError as exc:
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
def movies() -> dict[int, TMDBMovie]:
    raw = json.loads(TMDB_FIXTURE.read_text(encoding="utf-8"))
    return {int(k): parse_movie(v) for k, v in raw.items()}


def _ingest(session: Session) -> None:
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    raw = json.loads(WD_FIXTURE.read_text(encoding="utf-8"))
    ingest_spine(session, slice_, {qid: parse_entity(e) for qid, e in raw.items()})


def test_enrichment_writes_titles_and_cast(session: Session, movies: dict[int, TMDBMovie]) -> None:
    _ingest(session)
    report = enrich_tmdb(session, movies)
    # Fixture spine carries P4947 tmdb_ids for its 9 entities; 6 movie payloads committed.
    assert report.versions_enriched >= 5
    assert report.titles_written > 0 and report.people_upserted > 0
    assert report.conflicts_recorded == 0
    # Drishyam ml (tmdb 244049): canonical title indexed with a match_key, lead cast present.
    drishyam = session.scalars(select(Version).where(Version.tmdb_id == 244049)).one()
    titles = session.scalars(
        select(VersionTitle).where(VersionTitle.version_id == drishyam.version_id)
    ).all()
    assert any(t.kind == "canonical" and t.match_key for t in titles)
    cast = session.scalars(
        select(VersionCast).where(VersionCast.version_id == drishyam.version_id)
    ).all()
    assert any(c.role_kind == "lead" for c in cast)
    assert any(c.role_kind == "director" for c in cast)


def test_dub_track_receives_translation_title(
    session: Session, movies: dict[int, TMDBMovie]
) -> None:
    """Baahubali's ml translation lands on the QID-less ml dub track as kind=dub."""
    _ingest(session)
    enrich_tmdb(session, movies)
    baahubali_te = session.scalars(select(Version).where(Version.tmdb_id == 256040)).one()
    dub = session.scalars(
        select(Version).where(
            Version.work_id == baahubali_te.work_id,
            Version.language == "ml",
            Version.tmdb_id.is_(None),
        )
    ).one()
    titles = session.scalars(
        select(VersionTitle).where(VersionTitle.version_id == dub.version_id)
    ).all()
    assert len(titles) == 1 and titles[0].kind == "dub"
    # The ta co-original is NOT a dub: no TMDB ta translation exists, so no row at all —
    # and if one existed it would be kind=canonical (is_original guard).
    ta_original = session.scalars(
        select(Version).where(
            Version.work_id == baahubali_te.work_id,
            Version.language == "ta",
            Version.is_original.is_(True),
        )
    ).one()
    ta_titles = session.scalars(
        select(VersionTitle).where(VersionTitle.version_id == ta_original.version_id)
    ).all()
    assert all(t.kind != "dub" for t in ta_titles)


def test_enrichment_idempotent(session: Session, movies: dict[int, TMDBMovie]) -> None:
    _ingest(session)
    first = enrich_tmdb(session, movies)
    second = enrich_tmdb(session, movies)
    assert second.titles_written == 0
    assert second.people_upserted == 0
    assert second.cast_rows_written == 0
    assert second.conflicts_recorded == 0
    assert len(session.scalars(select(Person)).all()) == first.people_upserted


def test_year_split_opens_conflict_and_hides_version(
    session: Session, movies: dict[int, TMDBMovie]
) -> None:
    """Seed-vs-TMDB year split (n=2, no majority) → OPEN conflict → gate view hides row."""
    _ingest(session)
    doctored = dict(movies)
    doctored[244049] = movies[244049].model_copy(update={"release_year": 1999})
    report = enrich_tmdb(session, doctored)
    assert report.conflicts_open == 1
    conflict = session.scalars(
        select(Conflict).where(Conflict.field == "release_year", Conflict.status == "open")
    ).one()
    assert {v["value"] for v in conflict.values} == {2013, 1999}
    hidden = session.execute(
        text("SELECT count(*) FROM ground_truth_versions WHERE tmdb_id = 244049")
    ).scalar_one()
    assert hidden == 0
    # Re-run: the same disagreement is not re-queued.
    again = enrich_tmdb(session, doctored)
    assert again.conflicts_recorded == 0
