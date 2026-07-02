"""Integration: pgvector extension + chunk schema constraints (P2 task 2, §2.3).

Constraint-level proof that the retrieval store holds its invariants: kind CHECK,
per-config uniqueness, FK integrity, dense/sparse round-trips, and CASCADE from
chunks → chunk_embeddings. Gate-visibility of *populated* chunks is task 7's test
(index loader); here we prove the schema itself.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from pgvector import SparseVector
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import DENSE_DIM, SPARSE_DIM, Chunk, ChunkEmbedding, Version, Work

pytestmark = pytest.mark.integration


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
def version(session: Session) -> Version:
    work = Work(
        work_type="film",
        primary_title="Test Work",
        confidence="HIGH",
        sources=[{"source": "human", "ref": "test"}],
    )
    session.add(work)
    session.flush()
    v = Version(
        work_id=work.work_id,
        title="Test Version",
        language="ml",
        confidence="HIGH",
        sources=[{"source": "human", "ref": "test"}],
    )
    session.add(v)
    session.flush()
    return v


def _chunk(version: Version, **overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "version_id": version.version_id,
        "work_id": version.work_id,
        "kind": "plot",
        "seq": 0,
        "text": "Test Version (ml). A plot paragraph.",
        "chunker": "recursive_para",
        "chunk_config": "512tok_15pct",
        "content_hash": "0" * 64,
        "license": "CC BY-SA 4.0",
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_pgvector_extension_enabled(session: Session) -> None:
    installed = session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
    assert installed is not None


def test_chunk_insert_and_kind_check(session: Session, version: Version) -> None:
    session.add(_chunk(version))
    session.add(_chunk(version, kind="metadata_card", plot_id=None))
    session.flush()
    with pytest.raises(IntegrityError, match="ck_chunks_kind"):
        session.add(_chunk(version, kind="banana", seq=99))
        session.flush()


def test_chunk_uniqueness_per_config(session: Session, version: Version) -> None:
    """Same (version, kind, chunker, config, seq) twice → rejected; other config → fine."""
    session.add(_chunk(version))
    session.flush()
    session.add(_chunk(version, chunk_config="256tok_15pct"))  # ablation sibling OK
    session.flush()
    with pytest.raises(IntegrityError, match="uq_chunks_version_id"):
        session.add(_chunk(version))
        session.flush()


def test_chunk_fk_integrity(session: Session, version: Version) -> None:
    with pytest.raises(IntegrityError, match="fk_chunks_version_id_version"):
        session.add(_chunk(version, version_id=uuid.uuid4()))
        session.flush()


def test_embedding_roundtrip_dense_and_sparse(session: Session, version: Version) -> None:
    """Stored vectors come back exactly; sparsevec keeps indices/values."""
    chunk = _chunk(version)
    session.add(chunk)
    session.flush()
    dense = np.zeros(DENSE_DIM, dtype=np.float32)
    dense[0], dense[7] = 0.5, -0.25
    sparse = SparseVector({3: 0.75, 1000: 0.125, 249999: 1.0}, SPARSE_DIM)
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            embed_model="BAAI/bge-m3@testrev",
            index_version="run-test",
            dense=dense,
            sparse=sparse,
        )
    )
    session.flush()
    session.expire_all()
    row = session.scalars(select(ChunkEmbedding)).one()
    assert np.allclose(np.asarray(row.dense), dense)
    got = row.sparse
    assert dict(zip(got.indices(), got.values(), strict=True)) == {
        3: 0.75,
        1000: 0.125,
        249999: 1.0,
    }
    assert got.dimensions() == SPARSE_DIM


def test_embedding_pk_allows_ab_rows(session: Session, version: Version) -> None:
    """PK (chunk_id, embed_model, index_version): A/B + re-embeds coexist per chunk."""
    chunk = _chunk(version)
    session.add(chunk)
    session.flush()
    dense = np.zeros(DENSE_DIM, dtype=np.float32)
    sparse = SparseVector({1: 1.0}, SPARSE_DIM)
    for model, run in [("m3@a", "run-1"), ("m3@a", "run-2"), ("gemma2-9b@b", "run-1")]:
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embed_model=model,
                index_version=run,
                dense=dense,
                sparse=sparse,
            )
        )
    session.flush()
    assert len(session.scalars(select(ChunkEmbedding)).all()) == 3
    with pytest.raises(IntegrityError, match="pk_chunk_embeddings"):
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embed_model="m3@a",
                index_version="run-1",
                dense=dense,
                sparse=sparse,
            )
        )
        session.flush()


def test_delete_chunk_cascades_embeddings(session: Session, version: Version) -> None:
    chunk = _chunk(version)
    session.add(chunk)
    session.flush()
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            embed_model="m3@a",
            index_version="run-1",
            dense=np.zeros(DENSE_DIM, dtype=np.float32),
            sparse=SparseVector({1: 1.0}, SPARSE_DIM),
        )
    )
    session.flush()
    session.execute(text("DELETE FROM chunks WHERE chunk_id = :c"), {"c": str(chunk.chunk_id)})
    assert session.scalars(select(ChunkEmbedding)).all() == []


def test_sparsevec_inner_product_in_db(session: Session, version: Version) -> None:
    """The sparse leg's scoring primitive: `<#>` = negative inner product, in-DB (§2.3)."""
    chunk = _chunk(version)
    session.add(chunk)
    session.flush()
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            embed_model="m3@a",
            index_version="run-1",
            dense=np.zeros(DENSE_DIM, dtype=np.float32),
            sparse=SparseVector({10: 0.5, 20: 2.0}, SPARSE_DIM),
        )
    )
    session.flush()
    overlap = session.execute(
        text("SELECT sparse <#> :q FROM chunk_embeddings"),
        {"q": SparseVector({10: 1.0, 20: 0.25, 30: 9.9}, SPARSE_DIM).to_text()},
    ).scalar_one()
    assert overlap == pytest.approx(-(0.5 * 1.0 + 2.0 * 0.25))  # -1.0: negative inner product
    disjoint = session.execute(
        text("SELECT sparse <#> :q FROM chunk_embeddings"),
        {"q": SparseVector({99: 1.0}, SPARSE_DIM).to_text()},
    ).scalar_one()
    assert disjoint == 0.0  # zero token overlap → score 0
