"""Integration tests: spine ingest into real Postgres (P1 task 4).

Fixture-driven (the committed trimmed capture of the 2026-07-02 live snapshot) — no live API.
Reuses the per-test rollback pattern from ``test_graph_schema.py``.
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
from sutradhar.graph.schema import Conflict, Edge, Version, Work
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, SeedSlice, load_seed_slice
from sutradhar.pipeline.wikidata import WikidataEntity, ingest_spine, parse_entity

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "wikidata" / "entities_sample.json"


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
        # Isolate from any previously-ingested live data: the ingest is upsert-keyed on
        # QIDs, so a clean slate makes count assertions deterministic.
        for table in (
            "candidate_edges",
            "edges",
            "conflicts",
            "plot_texts",
            "version_cast",
            "version_title",
            "version",
            "work",
        ):
            s.execute(text(f"DELETE FROM {table}"))
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


@pytest.fixture(scope="module")
def slice_() -> SeedSlice:
    return load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)


@pytest.fixture(scope="module")
def entities() -> dict[str, WikidataEntity]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {qid: parse_entity(e) for qid, e in raw.items()}


def test_ingest_writes_skeletons_and_wikidata_edges(
    session: Session, slice_: SeedSlice, entities: dict[str, WikidataEntity]
) -> None:
    report = ingest_spine(session, slice_, entities)
    assert report.works_upserted == len(slice_.works) == 15
    assert report.versions_upserted == slice_.version_count() == 31
    # Fixture carries: 3 version-remake edges (hi-Drishyam->ml via P144, Papanasam via P4969
    # inverse, Chandramukhi->ml via P144), 1 sequel edge, 2 based_on edges
    # (Devdas-2002 P144 + novella P4969 inverse -> Devadasu... exact set asserted below).
    labels = set(report.edge_labels)
    assert "Q19824636 is_remake_of Q15401703" in labels
    assert "Q18129183 is_remake_of Q15401703" in labels  # from the ml original's P4969
    assert "Q102036246.work is_sequel_of Q15401703.work" in labels
    assert "Q247854.work based_on Q11169650" in labels
    # Wikidata-honesty: the kn/te remakes have NO asserted edge — they must NOT appear.
    assert not any("Q17052094" in label or "Q16248158" in label for label in labels)
    assert report.conflicts_opened == 0


def test_ingest_is_idempotent(
    session: Session, slice_: SeedSlice, entities: dict[str, WikidataEntity]
) -> None:
    first = ingest_spine(session, slice_, entities)
    second = ingest_spine(session, slice_, entities)
    assert second.edges_written == 0 and second.conflicts_opened == 0
    assert session.scalar(select(Work.work_id).select_from(Work).limit(1)) is not None
    counts = {
        "work": len(session.scalars(select(Work)).all()),
        "version": len(session.scalars(select(Version)).all()),
        "edges": len(session.scalars(select(Edge)).all()),
    }
    assert counts == {"work": 15, "version": 31, "edges": first.edges_written}


def test_year_disagreement_opens_conflict(
    session: Session, slice_: SeedSlice, entities: dict[str, WikidataEntity]
) -> None:
    """A seed-vs-Wikidata release-year mismatch is queued, never silently resolved."""
    bad = dict(entities)
    original = bad["Q19824636"]
    bad["Q19824636"] = original.model_copy(update={"publication_years": (1999,)})
    report = ingest_spine(session, slice_, bad)
    assert report.conflicts_opened == 1
    conflict = session.scalars(select(Conflict).where(Conflict.status == "open")).one()
    values = {json.dumps(v, sort_keys=True) for v in conflict.values}
    assert any('"value": 2015' in v for v in values)  # seed side preserved
    assert any("1999" in v for v in values)  # wikidata side preserved
    # And the gated view hides the conflicted version until resolution.
    hidden = session.execute(
        text("SELECT count(*) FROM ground_truth_versions v WHERE v.wikidata_qid = 'Q19824636'")
    ).scalar_one()
    assert hidden == 0
    # Re-running with the same disagreement must not duplicate the conflict.
    again = ingest_spine(session, slice_, bad)
    assert again.conflicts_opened == 0


def test_qidless_dub_tracks_are_medium(
    session: Session, slice_: SeedSlice, entities: dict[str, WikidataEntity]
) -> None:
    """QID-less versions (Baahubali ta/hi/ml, Devadas ta) land as MEDIUM, single human source."""
    ingest_spine(session, slice_, entities)
    medium = session.scalars(select(Version).where(Version.confidence == "MEDIUM")).all()
    assert {v.language for v in medium} == {"ta", "hi", "ml"}
    assert all(v.wikidata_qid is None for v in medium)
    assert len(medium) == 4
    for v in medium:
        assert v.sources and v.sources[0]["source"] == "human"
