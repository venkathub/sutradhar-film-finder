"""Integration tests: recorded-artifact extraction loading into candidate_edges (P1 task 11).

Runs against a recorded-output artifact fixture (CI never calls a model, ROADMAP §6.2).
Asserts the CANDIDATE quarantine: proposals land in candidate_edges only, never in edges,
and never appear through the ground-truth views or the repository.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import CandidateEdge, Edge, Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
ARTIFACT = FIXTURES / "extraction" / "outputs_sample.json"


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


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return data


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
    return session


def test_artifact_loads_candidates(built: Session, artifact: dict[str, Any]) -> None:
    report = load_candidates(
        built, artifact["raw_outputs"], artifact["pages"], artifact["model_id"]
    )
    assert report.candidates_written > 0
    assert report.run_hash and len(report.run_hash) == 16
    rows = built.scalars(select(CandidateEdge)).all()
    assert len(rows) == report.candidates_written
    for row in rows:
        assert row.status == "proposed"
        assert row.supporting_sentence and row.source_revision
        assert row.model_id == artifact["model_id"]
        assert row.extraction_run == report.run_hash


def test_candidates_never_touch_edges_or_views(built: Session, artifact: dict[str, Any]) -> None:
    edges_before = built.scalars(select(Edge)).all()
    load_candidates(built, artifact["raw_outputs"], artifact["pages"], artifact["model_id"])
    edges_after = built.scalars(select(Edge)).all()
    assert len(edges_after) == len(edges_before)  # NOT ONE edge from extraction
    gt_count = built.execute(text("SELECT count(*) FROM ground_truth_edges")).scalar_one()
    assert (
        gt_count
        == len([e for e in edges_before])
        - built.execute(
            text(
                "SELECT count(*) FROM edges e WHERE EXISTS (SELECT 1 FROM conflicts c "
                "WHERE c.entity_kind='edge' AND c.entity_id=e.edge_id AND c.status='open')"
            )
        ).scalar_one()
    )


def test_unsupported_and_malformed_counted(built: Session, artifact: dict[str, Any]) -> None:
    doctored = dict(artifact["raw_outputs"])
    first_key = sorted(doctored)[0]
    doctored["fake|page"] = "not json at all"
    pages = dict(artifact["pages"])
    pages["fake|page"] = {"title": "Fake", "text": "Nothing here.", "revision": "1"}
    # A hallucinated supporting sentence on a real page:
    doctored[first_key] = (
        '{"relationships": [{"edge_type": "is_remake_of", "src_title": "X", '
        '"dst_title": "Y", "supporting_sentence": "This sentence is not in the article.", '
        '"confidence": 0.9}]}'
    )
    report = load_candidates(built, doctored, pages, artifact["model_id"])
    # Baseline: the REAL fixture already contains one genuinely malformed model output
    # (the Baahubali page) + our doctored non-JSON page = 2.
    assert report.responses_malformed == 2
    assert report.proposals_unsupported >= 1
    assert report.parse_failure_rate > 0


def test_candidate_loading_idempotent(built: Session, artifact: dict[str, Any]) -> None:
    first = load_candidates(built, artifact["raw_outputs"], artifact["pages"], artifact["model_id"])
    again = load_candidates(built, artifact["raw_outputs"], artifact["pages"], artifact["model_id"])
    assert again.candidates_written == 0
    # P7 task 7: the recorded artifact carries 3 within-batch duplicate proposals
    # that the pre-constraint SELECT-then-skip dedup could not see (autoflush=False);
    # they are now skipped on EVERY pass, so the re-run's duplicate count covers
    # both the DB-deduped rows and the batch-deduped repeats.
    assert first.proposals_duplicate == 3  # the latent-bug evidence, now counted
    assert again.proposals_duplicate == first.candidates_written + first.proposals_duplicate


def test_conservative_binding(built: Session, artifact: dict[str, Any]) -> None:
    """Bound version_ids must be real and unambiguous; ambiguous titles keep raw strings."""
    load_candidates(built, artifact["raw_outputs"], artifact["pages"], artifact["model_id"])
    for row in built.scalars(select(CandidateEdge)).all():
        if row.src_version_id is not None:
            assert built.get(Version, row.src_version_id) is not None
        assert row.src_title_raw  # raw strings always preserved for the reviewer
        assert row.dst_title_raw
