"""P7 task 7 (DEC-P7-1 finding 9) — DB-owned uniqueness regressions.

Proves the three a41f09c3d7e2 constraints reject duplicates at the DB layer
(previously app-discipline only), that the migration round-trips, and that the
pre-audit aborts (rather than silently deletes) on pre-existing duplicates.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import CandidateEdge, Person, Version, Work

pytestmark = pytest.mark.integration

SOURCES = [{"source": "wikidata", "ref": "Q1"}]


@pytest.fixture(scope="module")
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


def _work(session: Session) -> Work:
    work = Work(
        work_type="film",
        primary_title="Drishyam",
        original_language="ml",
        first_release_year=2013,
        confidence="HIGH",
        sources=SOURCES,
    )
    session.add(work)
    session.flush()
    return work


def _version(work: Work, *, language: str, year: int | None, qid: str | None = None) -> Version:
    return Version(
        work_id=work.work_id,
        wikidata_qid=qid,
        title=f"Drishyam ({language})",
        language=language,
        release_year=year,
        confidence="HIGH",
        sources=SOURCES,
    )


def _candidate(**overrides: object) -> CandidateEdge:
    values: dict = {
        "edge_type": "is_remake_of",
        "src_title_raw": "Papanasam",
        "dst_title_raw": "Drishyam",
        "supporting_sentence": "Papanasam is a remake of Drishyam.",
        "source_page": "en:Papanasam_(film)",
        "source_revision": "12345",
        "model_id": "test-model",
        "extraction_run": "run-1",
    }
    values.update(overrides)
    return CandidateEdge(**values)


def test_person_tmdb_id_unique(session: Session) -> None:
    session.add(Person(name="Kamal Haasan", tmdb_id=35742, sources=SOURCES))
    session.flush()
    session.add(Person(name="Kamal Hassan (dup)", tmdb_id=35742, sources=SOURCES))
    with pytest.raises(IntegrityError, match="uq_person_tmdb_id"):
        session.flush()


def test_version_fallback_key_unique(session: Session) -> None:
    work = _work(session)
    session.add(_version(work, language="ta", year=2015))
    session.flush()
    session.add(_version(work, language="ta", year=2015))
    with pytest.raises(IntegrityError, match="uq_version_work_lang_year"):
        session.flush()


def test_version_null_years_may_coexist(session: Session) -> None:
    """NULLS DISTINCT: two year-unknown dub tracks of one work are legitimate."""
    work = _work(session)
    session.add_all(
        [_version(work, language="hi", year=None), _version(work, language="hi", year=None)]
    )
    session.flush()  # no violation


def test_candidate_edges_dedup_key_unique_even_with_nulls(session: Session) -> None:
    session.add(_candidate())
    session.flush()
    session.add(_candidate(extraction_run="run-2"))  # same dedup 4-tuple, later run
    with pytest.raises(IntegrityError, match="uq_candidate_edges_dedup"):
        session.flush()
    session.rollback()
    # NULLS NOT DISTINCT: identical-up-to-NULL-titles proposals are duplicates too.
    session.add(_candidate(src_title_raw=None, dst_title_raw=None))
    session.flush()
    session.add(_candidate(src_title_raw=None, dst_title_raw=None, extraction_run="run-2"))
    with pytest.raises(IntegrityError, match="uq_candidate_edges_dedup"):
        session.flush()


def test_migration_round_trip(engine: Engine) -> None:
    """Downgrade to the pre-P7 revision and back: constraints drop and return."""
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")

    def index_names() -> set[str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes WHERE indexname IN "
                    "('uq_person_tmdb_id', 'uq_candidate_edges_dedup')"
                )
            ).fetchall()
        return {row[0] for row in rows}

    assert index_names() == {"uq_person_tmdb_id", "uq_candidate_edges_dedup"}
    command.downgrade(cfg, "bb22e78ff305")
    try:
        assert index_names() == set()
    finally:
        command.upgrade(cfg, "head")
    assert index_names() == {"uq_person_tmdb_id", "uq_candidate_edges_dedup"}


def test_pre_audit_aborts_on_existing_duplicates(engine: Engine) -> None:
    """The migration never deletes data: pre-existing duplicates ABORT the upgrade
    with the offending keys named (conflicts-queue posture)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    command.downgrade(cfg, "bb22e78ff305")  # window where duplicates are insertable
    work_id = None
    try:
        with engine.begin() as conn:
            work_id = conn.execute(
                text(
                    "INSERT INTO work (work_type, primary_title, original_language, "
                    "first_release_year, confidence, sources) VALUES "
                    "('film', 'Audit Probe', 'ml', 2013, 'HIGH', '[]'::jsonb) "
                    "RETURNING work_id"
                )
            ).scalar_one()
            for _ in range(2):
                conn.execute(
                    text(
                        "INSERT INTO version (work_id, title, language, release_year, "
                        "confidence, sources) VALUES "
                        f"('{work_id}', 'Audit Probe (ta)', 'ta', 2015, 'HIGH', '[]'::jsonb)"
                    )
                )
        with pytest.raises(Exception, match="uniqueness pre-audit failed"):
            command.upgrade(cfg, "head")
    finally:
        with engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM version WHERE work_id = '{work_id}'")
            ) if work_id else None
            conn.execute(text("DELETE FROM work WHERE primary_title = 'Audit Probe'"))
        command.upgrade(cfg, "head")
