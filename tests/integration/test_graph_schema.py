"""Integration tests: graph-schema constraints + verification-gate views against real Postgres.

Opt-in (marker: ``integration``), following the P0 compose-stack pattern: skip cleanly when
Postgres is unreachable. Migrations are applied once per session (``alembic upgrade head``);
each test runs inside a rolled-back transaction so the database stays clean across runs.

    make up && make db-migrate
    uv run pytest -m integration
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import Conflict, Edge, Version, Work

pytestmark = pytest.mark.integration

SRC = [{"source": "wikidata", "ref": "Q1618487", "retrieved_at": "2026-07-02T00:00:00Z"}]


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except OperationalError as exc:
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")
    # Apply migrations once (idempotent), via the same entrypoint `make db-migrate` uses.
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    """A Session inside an outer transaction that is always rolled back (clean DB per test)."""
    with engine.connect() as conn:
        outer = conn.begin()
        s = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


def _work(session: Session, **overrides: Any) -> Work:
    values: dict[str, Any] = {
        "work_type": "film",
        "primary_title": "Drishyam",
        "original_language": "ml",
        "first_release_year": 2013,
        "confidence": "HIGH",
        "sources": SRC,
    }
    values.update(overrides)
    w = Work(**values)
    session.add(w)
    session.flush()
    return w


def _version(session: Session, work: Work, **overrides: Any) -> Version:
    values: dict[str, Any] = {
        "work_id": work.work_id,
        "title": "Drishyam",
        "language": "ml",
        "release_year": 2013,
        "country": "indian",
        "confidence": "HIGH",
        "sources": SRC,
    }
    values.update(overrides)
    v = Version(**values)
    session.add(v)
    session.flush()
    return v


def _edge(session: Session, src: Version, dst: Version, **overrides: Any) -> Edge:
    values: dict[str, Any] = {
        "edge_type": "is_remake_of",
        "src_kind": "version",
        "src_id": src.version_id,
        "dst_kind": "version",
        "dst_id": dst.version_id,
        "confidence": "HIGH",
        "sources": SRC,
    }
    values.update(overrides)
    e = Edge(**values)
    session.add(e)
    session.flush()
    return e


# --- Constraint enforcement (P1_SPEC §4 "Schema constraints") ---


def test_version_to_work_remake_edge_rejected(session: Session) -> None:
    """A remake edge must be version→version; version→work violates the type-shape CHECK."""
    w = _work(session)
    v = _version(session, w)
    with pytest.raises(IntegrityError, match="ck_edges_type_shape"):
        session.add(
            Edge(
                edge_type="is_remake_of",
                src_kind="version",
                src_id=v.version_id,
                dst_kind="work",
                dst_id=w.work_id,
                confidence="HIGH",
                sources=SRC,
            )
        )
        session.flush()


def test_self_edge_rejected(session: Session) -> None:
    w = _work(session)
    v = _version(session, w)
    with pytest.raises(IntegrityError, match="ck_edges_no_self_edge"):
        session.add(
            Edge(
                edge_type="is_remake_of",
                src_kind="version",
                src_id=v.version_id,
                dst_kind="version",
                dst_id=v.version_id,
                confidence="HIGH",
                sources=SRC,
            )
        )
        session.flush()


def test_duplicate_edge_rejected(session: Session) -> None:
    w = _work(session)
    v1 = _version(session, w, title="Papanasam", language="ta")
    v2 = _version(session, w, is_original=True)
    _edge(session, v1, v2)
    with pytest.raises(IntegrityError, match="uq_edges_edge_type_src_dst"):
        _edge(session, v1, v2)


def test_dangling_edge_endpoint_rejected_by_trigger(session: Session) -> None:
    """Soft polymorphic FKs are hardened by the edges_endpoints_exist trigger (DEC-P1-1)."""
    w = _work(session)
    v = _version(session, w)
    with pytest.raises(IntegrityError, match="not found in version"):
        session.add(
            Edge(
                edge_type="is_remake_of",
                src_kind="version",
                src_id=uuid.uuid4(),  # nonexistent version
                dst_kind="version",
                dst_id=v.version_id,
                confidence="HIGH",
                sources=SRC,
            )
        )
        session.flush()


def test_wikidata_qid_unique(session: Session) -> None:
    _work(session, wikidata_qid="Q1618487")
    with pytest.raises(IntegrityError, match="uq_work_wikidata_qid"):
        _work(session, wikidata_qid="Q1618487")


def test_unknown_confidence_tier_rejected(session: Session) -> None:
    """CANDIDATE is not a live-graph tier — it exists only as the candidate_edges table."""
    with pytest.raises(IntegrityError, match="ck_work_confidence"):
        _work(session, confidence="CANDIDATE")


# --- Verification-gate views (P1_SPEC §1.8 + MEDIUM-passes clarification) ---


def _gt_edge_ids(session: Session) -> set[uuid.UUID]:
    rows = session.execute(text("SELECT edge_id FROM ground_truth_edges")).scalars().all()
    return set(rows)


def test_medium_edge_passes_gate_views(session: Session) -> None:
    """MEDIUM rows are live-but-flagged (DATA_SOURCES.md tier table); the golden-fixture
    validator — not the view — enforces HIGH/human-verified."""
    w = _work(session)
    v1 = _version(session, w, title="Drushyam", language="te")
    v2 = _version(session, w, is_original=True)
    e = _edge(session, v1, v2, confidence="MEDIUM")
    assert e.edge_id in _gt_edge_ids(session)


def test_open_conflict_hides_edge_until_resolved(session: Session) -> None:
    w = _work(session)
    v1 = _version(session, w, title="Papanasam", language="ta")
    v2 = _version(session, w, is_original=True)
    e = _edge(session, v1, v2)
    conflict = Conflict(
        entity_kind="edge",
        entity_id=e.edge_id,
        field="edge_type",
        values=[
            {"value": "is_remake_of", "source": "wikidata"},
            {"value": "is_official_dub_of", "source": "derived_rule"},
        ],
    )
    session.add(conflict)
    session.flush()
    assert e.edge_id not in _gt_edge_ids(session), "open conflict must hide the edge"

    conflict.status = "resolved"
    conflict.resolution = {"by": "human", "chosen_value": "is_remake_of"}
    session.flush()
    assert e.edge_id in _gt_edge_ids(session), "resolved conflict must unhide the edge"


def test_empty_sources_excluded_from_gate_views(session: Session) -> None:
    w = _work(session)
    v1 = _version(session, w, title="Drishya", language="kn")
    v2 = _version(session, w, is_original=True)
    e = _edge(session, v1, v2, sources=[])
    assert e.edge_id not in _gt_edge_ids(session)


def test_candidate_rows_never_in_gate_views(session: Session) -> None:
    """By construction: ground_truth_edges reads the edges table only. Assert the view's
    definition doesn't reference candidate_edges (schema-level proof, not just data-level)."""
    definition = session.execute(
        text("SELECT pg_get_viewdef('ground_truth_edges'::regclass, true)")
    ).scalar_one()
    assert "candidate_edges" not in definition
    assert "FROM edges" in definition


def test_gate_views_exist_for_works_and_versions(session: Session) -> None:
    w = _work(session)
    v = _version(session, w)
    works = session.execute(text("SELECT work_id FROM ground_truth_works")).scalars().all()
    versions = session.execute(text("SELECT version_id FROM ground_truth_versions")).scalars().all()
    assert w.work_id in set(works)
    assert v.version_id in set(versions)
