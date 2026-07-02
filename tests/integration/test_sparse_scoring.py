"""Integration: the in-DB sparse channel (P2 task 8) — `sparse_top_chunks` ordering and
scores equal hand-computed inner products on planted vectors; zero-overlap excluded."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from pgvector import SparseVector
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import DENSE_DIM, SPARSE_DIM, Chunk, ChunkEmbedding, Version, Work
from sutradhar.rag.sparse import sparse_top_chunks

pytestmark = pytest.mark.integration

RUN = "sparse-test-run"
MODEL = "BAAI/bge-m3@test"


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
def planted(session: Session) -> Session:
    """Three chunks with hand-chosen sparse weights (config '512tok_15pct')."""
    work = Work(
        work_type="film",
        primary_title="W",
        confidence="HIGH",
        sources=[{"source": "human", "ref": "t"}],
    )
    session.add(work)
    session.flush()
    version = Version(
        work_id=work.work_id,
        title="V",
        language="ml",
        confidence="HIGH",
        sources=[{"source": "human", "ref": "t"}],
    )
    session.add(version)
    session.flush()
    weights_by_seq = {
        0: {10: 1.0, 20: 2.0},  # overlap with query on 10 and 20
        1: {10: 3.0},  # overlap on 10 only
        2: {99: 5.0},  # zero overlap with the query
    }
    for seq, weights in weights_by_seq.items():
        chunk = Chunk(
            version_id=version.version_id,
            work_id=work.work_id,
            plot_id=None,
            kind="plot",
            seq=seq,
            text=f"chunk {seq}",
            chunker="recursive_para",
            chunk_config="512tok_15pct",
            content_hash=f"{seq:064d}",
            license="CC BY-SA 4.0",
        )
        session.add(chunk)
        session.flush()
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embed_model=MODEL,
                index_version=RUN,
                dense=np.zeros(DENSE_DIM, dtype=np.float32),
                sparse=SparseVector(weights, SPARSE_DIM),
            )
        )
    session.flush()
    return session


QUERY = {10: 0.5, 20: 0.75}
# Hand-computed inner products: seq0 = 0.5*1 + 0.75*2 = 2.0; seq1 = 0.5*3 = 1.5; seq2 = 0.


def test_ordering_and_scores_match_hand_computation(planted: Session) -> None:
    hits = sparse_top_chunks(
        planted,
        QUERY,
        chunk_config="512tok_15pct",
        embed_model=MODEL,
        index_version=RUN,
        top_n=10,
    )
    assert [h.content_hash[-1] for h in hits] == ["0", "1"]  # seq2 (no overlap) excluded
    assert hits[0].score == pytest.approx(2.0)
    assert hits[1].score == pytest.approx(1.5)
    assert all(isinstance(h.chunk_id, uuid.UUID) for h in hits)


def test_top_n_truncates(planted: Session) -> None:
    hits = sparse_top_chunks(
        planted,
        QUERY,
        chunk_config="512tok_15pct",
        embed_model=MODEL,
        index_version=RUN,
        top_n=1,
    )
    assert len(hits) == 1 and hits[0].score == pytest.approx(2.0)


def test_wrong_config_or_run_returns_nothing(planted: Session) -> None:
    for kwargs in (
        {"chunk_config": "256tok_15pct", "embed_model": MODEL, "index_version": RUN},
        {"chunk_config": "512tok_15pct", "embed_model": MODEL, "index_version": "other-run"},
    ):
        assert sparse_top_chunks(planted, QUERY, top_n=10, **kwargs) == []  # type: ignore[arg-type]


def test_empty_query_weights_returns_nothing(planted: Session) -> None:
    hits = sparse_top_chunks(
        planted,
        {},
        chunk_config="512tok_15pct",
        embed_model=MODEL,
        index_version=RUN,
        top_n=10,
    )
    assert hits == []
