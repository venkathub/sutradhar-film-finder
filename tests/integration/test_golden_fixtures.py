"""Integration tests: golden fixtures validate against the graph (P1 task 14).

The fixture chain (tasks 4/5/6/9/11) + the CI-mirrored review pass (task 12 semantics)
must yield a graph in which EVERY GS-01..GS-11 fixture is golden-eligible — the P1 DoD
"seed golden set frozen, validator-clean". Also lands the last named regressions:
``test_gs01_version_set_recall`` and the strengthened GS-09B proximate assertion.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.evals.golden import load_fixtures, validate_all
from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version, Work
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

from .ci_review_pass import apply_ci_review_pass

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


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
def reviewed(session: Session) -> Session:
    """Full fixture chain + CI-mirrored review pass = the post-task-12 graph state."""
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
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
    art = json.loads((FIXTURES / "extraction" / "outputs_sample.json").read_text("utf-8"))
    load_candidates(session, art["raw_outputs"], art["pages"], art["model_id"])
    confirmed = apply_ci_review_pass(session)
    assert confirmed >= 10  # the review lift materialized in CI too
    return session


# --- The DoD requirement: every fixture validator-clean ---


def test_all_golden_fixtures_validate(reviewed: Session) -> None:
    fixtures, issues = validate_all(reviewed, GOLDEN_DIR)
    assert len(fixtures) >= 25  # ~2-3 per category (§7 Q2)
    assert issues == [], "\n".join(f"{i.fixture_id}: {i.issue}" for i in issues)
    covered = {f.id[:5] for f in fixtures}
    assert covered == {f"GS-{i:02d}" for i in range(1, 12)}  # all 11 categories


def test_fixture_validator_rejects_medium_backed(reviewed: Session) -> None:
    """DEC-P1-7 layer 3: un-verify a dub edge → the GS-04 fixture must fail validation."""
    reviewed.execute(
        text(
            "UPDATE edges SET human_verified = false "
            "WHERE edge_type = 'is_official_dub_of' AND confidence = 'MEDIUM'"
        )
    )
    _, issues = validate_all(reviewed, GOLDEN_DIR)
    assert any("golden-eligible" in i.issue or "edge" in i.issue for i in issues)
    assert any(i.fixture_id.startswith("GS-04") for i in issues)


def test_fixture_validator_rejects_conflicted(reviewed: Session) -> None:
    """An open conflict behind a fixture record → validator rejects (never frozen dirty)."""
    papanasam = reviewed.scalars(select(Version).where(Version.title == "Papanasam")).one()
    reviewed.execute(
        text(
            "INSERT INTO conflicts (entity_kind, entity_id, field, values) "
            "VALUES ('version', :v, 'release_year', '[{\"value\": 2015}, {\"value\": 2014}]')"
        ),
        {"v": str(papanasam.version_id)},
    )
    _, issues = validate_all(reviewed, GOLDEN_DIR)
    assert any("not gate-visible" in i.issue and "Papanasam" in i.issue for i in issues)


def test_expected_tool_calls_present_on_conversational_fixtures() -> None:
    fixtures = load_fixtures(GOLDEN_DIR)
    for fixture in fixtures:
        if fixture.id.startswith(("GS-07", "GS-08")):
            assert fixture.expected_tool_calls, f"{fixture.id} missing expected_tool_calls"


# --- The last named regressions ---


def test_gs01_version_set_recall(reviewed: Session) -> None:
    """GS-01: the exact Indian version set, original flagged, EVERY remake labelled —
    version-set recall = 1.0. (Unblocked by extraction+review: kn/te edges are verified.)"""
    from sutradhar.graph.repository import get_versions

    work = reviewed.scalars(select(Work).where(Work.primary_title == "Drishyam")).one()
    result = get_versions(reviewed, work.work_id, scope="indian")
    got = {(v.title, v.language, v.year, v.relationship, v.is_original) for v in result.versions}
    expected = {
        ("Drishyam", "ml", 2013, "is_original_of", True),
        ("Drishya", "kn", 2014, "is_remake_of", False),
        ("Drushyam", "te", 2014, "is_remake_of", False),
        ("Papanasam", "ta", 2015, "is_remake_of", False),
        ("Drishyam", "hi", 2015, "is_remake_of", False),
    }
    assert got == expected, f"version-set recall < 1.0: {got ^ expected}"
    assert result.original is not None and result.original.language == "ml"


def test_gs09_transitive_lineage_proximate_edge(reviewed: Session) -> None:
    """GS-09B strengthened: Chandramukhi's verified is_remake_of edge points at APTHAMITRA
    (proximate source preserved as evidence), while the Work still derives the ml original."""
    chandramukhi = reviewed.scalars(select(Version).where(Version.title == "Chandramukhi")).one()
    apthamitra = reviewed.scalars(select(Version).where(Version.title == "Apthamitra")).one()
    rows = reviewed.execute(
        text(
            "SELECT dst_id, human_verified FROM ground_truth_edges "
            "WHERE src_id = :s AND edge_type = 'is_remake_of'"
        ),
        {"s": str(chandramukhi.version_id)},
    ).all()
    targets = {r.dst_id for r in rows}
    assert apthamitra.version_id in targets  # the proximate edge exists, verified
    assert all(r.human_verified for r in rows if r.dst_id == apthamitra.version_id)
    # And the lineage's sole original stays Manichitrathazhu (edge depth irrelevant).
    from sutradhar.graph.repository import get_versions

    result = get_versions(reviewed, chandramukhi.work_id, scope="indian")
    originals = [v for v in result.versions if v.is_original]
    assert len(originals) == 1 and originals[0].title == "Manichitrathazhu"
