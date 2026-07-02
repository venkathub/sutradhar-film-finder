"""Integration test: the graph report over the fixture chain (P1 task 13)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.report import build_report, render_report
from sutradhar.pipeline.review import Decision, EndpointSpec, apply_decisions
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
    return session


def test_report_version_coverage_full_and_edge_gaps_named(built: Session) -> None:
    report = build_report(built, seed_path=REPO_ROOT / DEFAULT_SEED_PATH)
    # Version coverage: all 31 seed versions ingest from the slice → flagship gate PASS.
    assert report.flagship_coverage_ok is True
    assert all(f.coverage == 1.0 for f in report.franchises)
    # Edge coverage: fixture chain pre-review is partial — gaps are NAMED, not hidden.
    assert report.edge_coverage.coverage < 1.0
    assert any("drishya_kn" in m for m in report.edge_coverage.missing)
    # Extraction metrics present (fixture artifact slice: 5 pages, 1 genuinely malformed).
    assert report.extraction.proposed > 0 and report.extraction.pending > 0
    assert report.extraction.precision is None  # nothing decided yet


def test_report_reflects_review_lift(built: Session) -> None:
    """Confirming the proximate edge moves both edge coverage and the lift counter."""
    from sutradhar.graph.schema import CandidateEdge

    candidate = built.scalars(
        select(CandidateEdge).where(
            CandidateEdge.src_title_raw == "Chandramukhi",
            CandidateEdge.dst_title_raw == "Apthamitra",
        )
    ).one()
    before = build_report(built, seed_path=REPO_ROOT / DEFAULT_SEED_PATH)
    apply_decisions(
        built,
        [
            Decision(
                candidate_id=candidate.candidate_id,
                verdict="confirm",
                src=EndpointSpec(title="Chandramukhi", language="ta"),
                dst=EndpointSpec(title="Apthamitra", language="kn"),
            )
        ],
        reviewer="tester",
    )
    after = build_report(built, seed_path=REPO_ROOT / DEFAULT_SEED_PATH)
    assert after.edge_coverage.present == before.edge_coverage.present + 1
    assert (
        after.extraction.edges_created_beyond_wikidata
        == before.extraction.edges_created_beyond_wikidata + 1
    )
    assert after.extraction.precision == 1.0  # 1 confirmed / 1 decided
    assert not any(m.startswith("chandramukhi_ta ") for m in after.edge_coverage.missing)


def test_report_renders_with_stamp(built: Session) -> None:
    report = build_report(built, seed_path=REPO_ROOT / DEFAULT_SEED_PATH)
    rendered = render_report(report)
    assert "flagship gate (=1.0): PASS" in rendered
    assert "Reproducibility stamp" in rendered
    assert report.stamp["code_sha"] and report.stamp["seed_slice_sha"]
