"""Integration tests: repository tool-backing behaviour (P1 task 10, TOOL_SCHEMA v0 §2.5).

Named GS regressions live in ``test_golden_regressions.py``; this module covers the
per-function contract: resolve_title scoring/ambiguity, get_work source_work, refine_filter
semantics, and the gate guarantee (conflicted rows invisible through every function).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.repository import (
    RefineBy,
    get_versions,
    get_work,
    refine_filter,
    resolve_title,
)
from sutradhar.graph.schema import Conflict, Version, Work
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.imdb import load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


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


@pytest.fixture()
def built(session: Session) -> Session:
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
    akas_lines = (FIXTURES / "imdb" / "akas_sample.tsv").read_text("utf-8").splitlines()
    load_akas(session, [r for r in (parse_aka_line(ln) for ln in akas_lines) if r])
    # Task-8 finishing pass: canonical titles into the index (fixture-chain equivalent
    # of `make rekey-titles`), so resolve_title sees every version.
    from sutradhar.graph.models import SourceId, SourceRef
    from sutradhar.pipeline.titles import upsert_version_title

    for v in session.scalars(select(Version)).all():
        upsert_version_title(
            session,
            v.version_id,
            v.title,
            "canonical",
            v.language,
            [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
        )
    build_graph(session)
    return session


# --- resolve_title ---


def test_resolve_title_exact_scores_one(built: Session) -> None:
    result = resolve_title(built, "Papanasam")
    assert result.candidates and result.candidates[0].score == 1.0
    assert result.candidates[0].matched_title == "Papanasam"
    assert result.candidates[0].language == "ta"
    assert result.candidates[0].sources  # provenance travels with the result


def test_resolve_title_vikram_ambiguous(built: Session) -> None:
    """GS-10 at the tool layer: 'Vikram' hits two distinct Works → ambiguous=true."""
    result = resolve_title(built, "Vikram")
    work_ids = {c.work_id for c in result.candidates}
    assert len(work_ids) == 2 and result.ambiguous is True
    years = {c.year for c in result.candidates}
    assert years == {1986, 2022}


def test_resolve_title_native_script_query(built: Session) -> None:
    """GS-11: a Malayalam-script query resolves to the ml Drishyam version."""
    result = resolve_title(built, "ദൃശ്യം")
    assert result.candidates
    top = result.candidates[0]
    assert top.language in {"ml", "hi"}  # drishyam-key family; ml canonical present
    langs = {c.language for c in result.candidates}
    assert "ml" in langs


def test_resolve_title_perturbed_query(built: Session) -> None:
    result = resolve_title(built, "Papanaasam", language="ta")
    assert result.candidates and result.candidates[0].language == "ta"
    assert result.candidates[0].score == 1.0  # vowel collapse → exact key


def test_resolve_title_language_hint_filters(built: Session) -> None:
    result = resolve_title(built, "Drishyam", language="hi")
    assert result.candidates
    assert all(c.language == "hi" for c in result.candidates)


# --- get_work ---


def test_get_work_source_work(built: Session) -> None:
    """GS-05: a sibling adaptation exposes its literary source."""
    devadasu = built.scalars(select(Work).where(Work.primary_title == "Devadasu")).one()
    result = get_work(built, devadasu.work_id)
    assert result is not None
    assert result.source_work is not None
    assert result.source_work.work_type == "literary_source"
    assert result.source_work.canonical_title == "Devdas"
    assert result.based_on == [result.source_work.work_id]


def test_get_work_unknown_id_returns_none(built: Session) -> None:
    assert get_work(built, uuid.uuid4()) is None


# --- refine_filter (GS-08 backtracking semantics) ---


@pytest.fixture()
def drishyam_indian_set(built: Session) -> list[uuid.UUID]:
    work = built.scalars(select(Work).where(Work.primary_title == "Drishyam")).one()
    result = get_versions(built, work.work_id, scope="indian")
    return [v.version_id for v in result.versions]


def test_refine_by_actor(built: Session, drishyam_indian_set: list[uuid.UUID]) -> None:
    result = refine_filter(built, drishyam_indian_set, RefineBy(actor="Ajay Devgn"))
    assert len(result.versions) == 1 and result.versions[0].language == "hi"


def test_refine_by_era_original(built: Session, drishyam_indian_set: list[uuid.UUID]) -> None:
    result = refine_filter(built, drishyam_indian_set, RefineBy(era="original"))
    assert len(result.versions) == 1
    assert result.versions[0].language == "ml" and result.versions[0].is_original


def test_refine_by_language(built: Session, drishyam_indian_set: list[uuid.UUID]) -> None:
    result = refine_filter(built, drishyam_indian_set, RefineBy(language="te"))
    assert len(result.versions) == 1 and result.versions[0].title == "Drushyam"


def test_refine_by_era_newer(built: Session, drishyam_indian_set: list[uuid.UUID]) -> None:
    result = refine_filter(built, drishyam_indian_set, RefineBy(era="newer"))
    assert {v.year for v in result.versions} == {2014, 2015}  # everything after the 2013 original


def test_refine_chained_turns(built: Session, drishyam_indian_set: list[uuid.UUID]) -> None:
    """GS-08 shape: narrow by actor, backtrack, then ask for the Telugu one."""
    turn1 = refine_filter(built, drishyam_indian_set, RefineBy(actor="Ajay Devgn"))
    assert len(turn1.versions) == 1
    turn2 = refine_filter(built, drishyam_indian_set, RefineBy(era="original"))
    assert turn2.versions[0].language == "ml"
    turn3 = refine_filter(built, drishyam_indian_set, RefineBy(language="te"))
    assert turn3.versions[0].year == 2014


def test_refine_empty_set(built: Session) -> None:
    assert refine_filter(built, [], RefineBy(language="ta")).versions == []


# --- The gate guarantee at the repository layer ---


def test_conflicted_version_invisible_through_repository(built: Session) -> None:
    """An open conflict hides a version from resolve_title AND get_versions (no bypass)."""
    hi = built.scalars(
        select(Version).where(Version.language == "hi", Version.title == "Drishyam")
    ).one()
    built.add(
        Conflict(
            entity_kind="version",
            entity_id=hi.version_id,
            field="release_year",
            values=[{"value": 2015, "source": "a"}, {"value": 2016, "source": "b"}],
        )
    )
    built.flush()
    work = built.scalars(select(Work).where(Work.primary_title == "Drishyam")).one()
    versions = get_versions(built, work.work_id, scope="indian")
    assert all(v.version_id != hi.version_id for v in versions.versions)
    resolved = resolve_title(built, "Drishyam", language="hi")
    assert all(c.version_id != hi.version_id for c in resolved.candidates)
