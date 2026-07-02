"""Integration tests: the human review gate (P1 task 12, gate enforcement per spec §4).

The enforced properties: promotion is the only candidate→edge path; rejection never writes
edges; promoted edges are human_verified + audit-linked; corroboration merges instead of
duplicating; work-level types promote at work level; MEDIUM rule edges verify explicitly.
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
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import CandidateEdge, Version, Work
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.review import (
    BindingError,
    Decision,
    EndpointSpec,
    apply_decisions,
    list_medium_rule_edges,
    list_proposed,
    promote,
    reject,
    verify_medium_edges,
)
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
def built(session: Session) -> Session:
    """Fixture chain through build_graph + the REAL extraction artifact slice."""
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


def _candidate(built: Session, edge_type: str, src_raw: str, dst_raw: str) -> CandidateEdge:
    return built.scalars(
        select(CandidateEdge).where(
            CandidateEdge.edge_type == edge_type,
            CandidateEdge.src_title_raw == src_raw,
            CandidateEdge.dst_title_raw == dst_raw,
        )
    ).one()


def test_promotion_creates_verified_edge_with_audit_link(built: Session) -> None:
    candidate = _candidate(built, "is_remake_of", "Chandramukhi", "Apthamitra")
    edges_before = built.execute(text("SELECT count(*) FROM edges")).scalar_one()
    edge, created = promote(built, candidate, reviewer="tester")
    assert created is True
    assert edge.human_verified is True and edge.confidence == "HIGH"
    assert candidate.status == "confirmed"
    assert candidate.reviewed_by == "tester" and candidate.reviewed_at is not None
    assert candidate.promoted_edge_id == edge.edge_id
    sources = {s["source"] for s in edge.sources}
    assert sources == {"wikipedia", "human"}
    # And it is now gate-visible (the extraction lift becoming ground truth):
    visible = built.execute(
        text("SELECT count(*) FROM ground_truth_edges WHERE edge_id = :e"),
        {"e": str(edge.edge_id)},
    ).scalar_one()
    assert visible == 1
    assert built.execute(text("SELECT count(*) FROM edges")).scalar_one() == edges_before + 1


def test_promotion_corroborates_existing_edge(built: Session) -> None:
    """Apthamitra→Manichitrathazhu already exists (Wikidata P144) — confirm merges, no dup."""
    candidate = _candidate(built, "is_remake_of", "Apthamitra", "Manichithrathazhu")
    edges_before = built.execute(text("SELECT count(*) FROM edges")).scalar_one()
    edge, created = promote(
        built,
        candidate,
        reviewer="tester",
        # The model's language hint was wrong (ml for a kn film) so extraction left src
        # unbound — reviewer resolution supplies both endpoints (binding ≠ repair).
        src=EndpointSpec(title="Apthamitra", language="kn"),
        dst=EndpointSpec(title="Manichitrathazhu", language="ml"),  # spelling-variant bind
    )
    assert created is False
    assert built.execute(text("SELECT count(*) FROM edges")).scalar_one() == edges_before
    assert edge.human_verified is True
    sources = {s["source"] for s in edge.sources}
    assert {"wikidata", "wikipedia", "human"} <= sources  # provenance accumulates


def test_rejection_never_writes_edges(built: Session) -> None:
    candidate = (
        _candidate(built, "is_remake_of", "Drishyam", "Drishya")
        if _exists(built, "is_remake_of", "Drishyam", "Drishya")
        else list_proposed(built)[0]
    )
    edges_before = built.execute(text("SELECT count(*) FROM edges")).scalar_one()
    reject(built, candidate, reviewer="tester")
    assert candidate.status == "rejected"
    assert candidate.reviewed_by == "tester" and candidate.promoted_edge_id is None
    assert built.execute(text("SELECT count(*) FROM edges")).scalar_one() == edges_before


def _exists(built: Session, edge_type: str, src_raw: str, dst_raw: str) -> bool:
    return (
        built.scalars(
            select(CandidateEdge).where(
                CandidateEdge.edge_type == edge_type,
                CandidateEdge.src_title_raw == src_raw,
                CandidateEdge.dst_title_raw == dst_raw,
            )
        ).first()
        is not None
    )


def test_unbindable_confirm_refused(built: Session) -> None:
    """A confirm whose endpoints can't bind raises — never a silent partial edge."""
    candidate = list_proposed(built)[0]
    with pytest.raises(BindingError):
        promote(
            built,
            candidate,
            reviewer="tester",
            src=EndpointSpec(title="No Such Film Ever"),
            dst=EndpointSpec(title="Definitely Missing"),
        )
    assert candidate.status == "proposed"  # untouched


def test_based_on_promotes_at_work_level(built: Session) -> None:
    candidate = _candidate(built, "based_on", "Devadasu (1953 film)", "Devdas")
    edge, created = promote(
        built,
        candidate,
        reviewer="tester",
        src=EndpointSpec(title="Devadasu", language="te"),
        dst=EndpointSpec(title="Devdas", year=1917, work=True),
    )
    assert edge.src_kind == "work" and edge.dst_kind == "work"
    novella = built.scalars(select(Work).where(Work.work_type == "literary_source")).one()
    assert edge.dst_id == novella.work_id
    assert created is False  # corroborates the Wikidata based_on edge


def test_decisions_file_batch_and_precision(built: Session) -> None:
    proposed = list_proposed(built)
    confirm_one = _candidate(built, "is_remake_of", "Chandramukhi", "Apthamitra")
    decisions = [Decision(candidate_id=confirm_one.candidate_id, verdict="confirm")]
    rejected_n = 0
    for c in proposed:
        if c.candidate_id != confirm_one.candidate_id and rejected_n < 3:
            decisions.append(Decision(candidate_id=c.candidate_id, verdict="reject"))
            rejected_n += 1
    report = apply_decisions(built, decisions, reviewer="tester")
    assert report.confirmed == 1 and report.rejected == 3
    assert report.precision == 0.25
    # Already-decided candidates are not re-decidable (audit integrity).
    again = apply_decisions(built, decisions, reviewer="tester")
    assert again.confirmed == 0 and again.rejected == 0
    assert len(again.errors) == len(decisions)


def test_medium_rule_edges_verify_explicitly(built: Session) -> None:
    pending = list_medium_rule_edges(built)
    assert pending  # the Baahubali/Devadas dub tracks from the builder
    ids = [e.edge_id for e in pending]
    verified = verify_medium_edges(built, "tester", ids)
    assert verified == len(ids)
    for edge in pending:
        assert edge.human_verified is True
        assert any(s["source"] == "human" for s in edge.sources)
    # Verified rule edges are now golden-eligible; nothing re-verifies.
    assert list_medium_rule_edges(built) == []
    assert verify_medium_edges(built, "tester", ids) == 0
    # An unknown id is ignored, not invented.
    assert verify_medium_edges(built, "tester", [uuid.uuid4()]) == 0
