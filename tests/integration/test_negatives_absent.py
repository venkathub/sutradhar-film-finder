"""Integration: every held-out negative is absent-from-slice by construction (P2 task 1).

Builds the fullest title surface the title channel can see (spine + TMDB + IMDb AKAs +
canonical titles + build_graph) and asserts :func:`resolve_title` returns ZERO candidates
for every negative query — including at the rapidfuzz 0.80 radius (DEC-P1-5). A collision
here means the negative must be re-authored, not the threshold moved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.evals.negatives import NEGATIVES_PATH, validate_all_negatives
from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.imdb import load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
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
def titled(session: Session) -> Session:
    """Fixture chain with the FULL title index: canonical + TMDB + IMDb AKAs."""
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
    akas_lines = (FIXTURES / "imdb" / "akas_sample.tsv").read_text("utf-8").splitlines()
    load_akas(session, [r for r in (parse_aka_line(ln) for ln in akas_lines) if r])
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


def test_all_negatives_absent_from_slice(titled: Session) -> None:
    """The enforcement of 'absent-from-slice-by-construction' for every NEG fixture."""
    fixtures, issues = validate_all_negatives(titled, REPO_ROOT / NEGATIVES_PATH)
    assert len(fixtures) == 24
    assert issues == [], "\n".join(f"{i.fixture_id}: {i.issue}" for i in issues)


def test_validator_catches_a_planted_collision(titled: Session) -> None:
    """Sanity: if a negative title WERE in the slice, the validator would flag it."""
    import yaml

    from sutradhar.evals.negatives import NegativeFixture, validate_negative

    planted = NegativeFixture.model_validate(
        {
            **yaml.safe_load((REPO_ROOT / NEGATIVES_PATH).read_text("utf-8"))["fixtures"][1],
            "query": "Drishyam",  # a real slice title
        }
    )
    issues = validate_negative(titled, planted)
    assert issues and "resolves in the title channel" in issues[0].issue
