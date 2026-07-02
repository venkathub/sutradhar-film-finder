"""Integration tests: IMDb akas loading into real Postgres (P1 task 6).

Chains the fixture-driven spine ingest + TMDB enrichment + akas load — no live API.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import Version, VersionTitle
from sutradhar.pipeline.imdb import AkaRow, load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WD_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "wikidata" / "entities_sample.json"
TMDB_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tmdb" / "movies_sample.json"
AKAS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "imdb" / "akas_sample.tsv"


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
def akas_rows() -> list[AkaRow]:
    lines = AKAS_FIXTURE.read_text(encoding="utf-8").splitlines()
    return [r for r in (parse_aka_line(line) for line in lines) if r is not None]


def _ingest_and_enrich(session: Session) -> None:
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads(WD_FIXTURE.read_text(encoding="utf-8"))
    ingest_spine(session, slice_, {qid: parse_entity(e) for qid, e in wd.items()})
    tm = json.loads(TMDB_FIXTURE.read_text(encoding="utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})


def test_akas_fill_dub_titles_and_corroborate(session: Session, akas_rows: list[AkaRow]) -> None:
    _ingest_and_enrich(session)
    report = load_akas(session, akas_rows)
    assert report.titles_new > 0
    assert report.dub_titles_mapped >= 1  # Baahubali hi dub title arrives via akas

    baahubali_te = session.scalars(select(Version).where(Version.imdb_id == "tt2631186")).one()
    hi_dub = session.scalars(
        select(Version).where(Version.work_id == baahubali_te.work_id, Version.language == "hi")
    ).one()
    hi_titles = session.scalars(
        select(VersionTitle).where(VersionTitle.version_id == hi_dub.version_id)
    ).all()
    assert any(t.kind == "dub" and t.title == "बाहुबली: एक शुरुआत" for t in hi_titles)
    # The ta co-original gets kind=canonical (is_original guard), never dub.
    ta_orig = session.scalars(
        select(Version).where(
            Version.work_id == baahubali_te.work_id,
            Version.language == "ta",
            Version.is_original.is_(True),
        )
    ).one()
    ta_titles = session.scalars(
        select(VersionTitle).where(VersionTitle.version_id == ta_orig.version_id)
    ).all()
    assert ta_titles and all(t.kind == "canonical" for t in ta_titles)


def test_union_corroboration_merges_sources(session: Session, akas_rows: list[AkaRow]) -> None:
    """A title seen by both TMDB and IMDb ends with BOTH refs (≥2 sources = HIGH per value)."""
    _ingest_and_enrich(session)
    report = load_akas(session, akas_rows)
    assert report.titles_corroborated > 0
    corroborated = session.execute(
        text("SELECT count(*) FROM version_title WHERE jsonb_array_length(sources) >= 2")
    ).scalar_one()
    assert corroborated == report.titles_corroborated
    # No row ever carries the same source twice (independence, not repetition).
    fake = session.execute(
        text(
            "SELECT count(*) FROM version_title vt WHERE ("
            " SELECT count(DISTINCT s->>'source') FROM jsonb_array_elements(vt.sources) s"
            ") < jsonb_array_length(vt.sources)"
        )
    ).scalar_one()
    assert fake == 0


def test_akas_load_idempotent(session: Session, akas_rows: list[AkaRow]) -> None:
    _ingest_and_enrich(session)
    load_akas(session, akas_rows)
    count_before = session.execute(text("SELECT count(*) FROM version_title")).scalar_one()
    again = load_akas(session, akas_rows)
    assert again.titles_new == 0 and again.titles_corroborated == 0
    count_after = session.execute(text("SELECT count(*) FROM version_title")).scalar_one()
    assert count_before == count_after


def test_same_pass_duplicate_titles_merge_not_duplicate(
    session: Session, akas_rows: list[AkaRow]
) -> None:
    """Two akas rows with identical (version,title,kind) in ONE pass → one row (flush fix)."""
    _ingest_and_enrich(session)
    dup = [r for r in akas_rows if r.titleId == "tt3417422"][:1]
    doubled = dup + [dup[0].model_copy(update={"ordering": 99})]
    report = load_akas(session, doubled)
    assert report.titles_new <= 1
    rows = session.execute(
        text(
            "SELECT count(*) FROM version_title vt JOIN version v USING (version_id) "
            "WHERE v.imdb_id = 'tt3417422' AND vt.title = :t AND vt.kind = :k"
        ),
        {"t": dup[0].title, "k": "canonical" if dup[0].isOriginalTitle else "aka"},
    ).scalar_one()
    assert rows == 1
