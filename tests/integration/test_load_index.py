"""Integration: index loader (P2 task 7) — sealed run → chunk_embeddings, gate-visible
only, pgvector round-trips on real stored vectors, idempotent reload, hard failure on a
corpus/run mismatch."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from pgvector import SparseVector
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import DENSE_DIM, SPARSE_DIM, Chunk, Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.pipeline.wikipedia import WikiPage, load_plots, parse_page
from sutradhar.rag.artifacts import ArtifactRun, MissingArtifactError, write_embedding_bank
from sutradhar.rag.chunking import CHUNK_CONFIGS
from sutradhar.rag.corpus import build_corpus
from sutradhar.rag.index import load_index

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
RUN_ID = "stub-index-run"


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


def _stub_vector(content_hash: str) -> tuple[np.ndarray, dict[int, float]]:
    """Deterministic per-hash unit vector + tiny sparse weights (1024-dim, column-true)."""
    seed = int(content_hash[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DENSE_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec), {seed % 250_000 + 1: 0.5, seed % 997 + 1: 0.25}


def _write_run_from_db(session: Session, base: Path, run_id: str = RUN_ID) -> ArtifactRun:
    """A sealed artifact run whose banks mirror the DB chunks (stub vectors)."""
    run = ArtifactRun.create(base, run_id=run_id)
    configs = [c.name for c in CHUNK_CONFIGS]
    for config in configs:
        hashes = sorted(
            {
                row[0]
                for row in session.execute(
                    text("SELECT content_hash FROM chunks WHERE chunk_config = :c"),
                    {"c": config},
                )
            }
        )
        pairs = [_stub_vector(h) for h in hashes]
        write_embedding_bank(
            run,
            f"corpus_{config}",
            hashes,
            np.stack([p[0] for p in pairs]) if pairs else np.zeros((0, DENSE_DIM), np.float32),
            [p[1] for p in pairs],
        )
    run.write_json(
        "meta.json",
        {"run_id": run_id, "embed_model": "BAAI/bge-m3", "configs": configs, "stub": True},
    )
    run.write_manifest()
    return run


@pytest.fixture()
def indexed(session: Session, tmp_path: Path) -> tuple[Session, Path]:
    """Graph + plots + corpus + a matching sealed run, loaded into chunk_embeddings."""
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
    raw = json.loads((FIXTURES / "wikipedia" / "pages_sample.json").read_text("utf-8"))
    pages: dict[str, list[WikiPage]] = {}
    for key, entry in raw.items():
        page = parse_page(entry["lang"], entry["response"])
        assert page is not None
        pages.setdefault(key.split("|", 1)[0], []).append(page)
    load_plots(session, pages)
    build_corpus(session)
    _write_run_from_db(session, tmp_path)
    load_index(session, tmp_path, RUN_ID)
    return session, tmp_path


def test_every_chunk_gets_exactly_one_embedding_row(indexed: tuple[Session, Path]) -> None:
    session, _ = indexed
    counts = session.execute(
        text(
            "SELECT c.chunk_config, count(c.chunk_id) AS chunks, count(e.chunk_id) AS embedded "
            "FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id "
            "GROUP BY c.chunk_config"
        )
    ).all()
    assert counts and all(row.chunks == row.embedded for row in counts)


def test_embeddings_cover_only_gate_visible_versions(indexed: tuple[Session, Path]) -> None:
    """The verification gate holds through the index: every embedded chunk's version is
    gate-visible (chunks are built from the views; the join proves no leakage)."""
    session, _ = indexed
    leaked = session.execute(
        text(
            "SELECT count(*) FROM chunk_embeddings e JOIN chunks c ON c.chunk_id = e.chunk_id "
            "WHERE c.version_id NOT IN (SELECT version_id FROM ground_truth_versions)"
        )
    ).scalar_one()
    assert leaked == 0


def test_dense_cosine_roundtrip_finds_the_stored_chunk(indexed: tuple[Session, Path]) -> None:
    session, _ = indexed
    some = session.execute(
        text("SELECT content_hash FROM chunks WHERE chunk_config = '512tok_15pct' LIMIT 1")
    ).scalar_one()
    query_vec, _ = _stub_vector(some)
    top = session.execute(
        text(
            "SELECT c.content_hash FROM chunk_embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id "
            "WHERE c.chunk_config = '512tok_15pct' "
            "ORDER BY e.dense <=> :q LIMIT 1"
        ),
        {"q": json.dumps([float(x) for x in query_vec])},
    ).scalar_one()
    assert top == some  # its own vector is its nearest neighbor (cosine distance 0)


def test_sparse_inner_product_roundtrip(indexed: tuple[Session, Path]) -> None:
    session, _ = indexed
    some = session.execute(
        text("SELECT content_hash FROM chunks WHERE chunk_config = '512tok_15pct' LIMIT 1")
    ).scalar_one()
    _, weights = _stub_vector(some)
    score = session.execute(
        text(
            "SELECT e.sparse <#> :q FROM chunk_embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id WHERE c.content_hash = :h LIMIT 1"
        ),
        {"q": SparseVector(weights, SPARSE_DIM).to_text(), "h": some},
    ).scalar_one()
    expected = -sum(w * w for w in weights.values())
    assert score == pytest.approx(expected, rel=1e-5)


def test_reload_is_idempotent(indexed: tuple[Session, Path]) -> None:
    session, tmp_path = indexed
    before = session.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar_one()
    report = load_index(session, tmp_path, RUN_ID)  # second load, same run
    after = session.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar_one()
    assert before == after == sum(report.rows_loaded.values())


def test_corpus_drift_is_a_hard_failure(indexed: tuple[Session, Path]) -> None:
    """A chunk the run has never seen (corpus changed after export) refuses to load."""
    session, tmp_path = indexed
    chunk = session.scalars(select(Chunk).where(Chunk.chunk_config == "512tok_15pct")).first()
    assert chunk is not None
    session.add(
        Chunk(
            version_id=chunk.version_id,
            work_id=chunk.work_id,
            plot_id=None,
            kind="metadata_card",
            seq=999,
            text="a chunk the GPU never embedded",
            language="en",
            chunker=chunk.chunker,
            chunk_config="512tok_15pct",
            content_hash="f" * 64,
            license="CC BY-SA 4.0",
        )
    )
    session.flush()
    with pytest.raises(MissingArtifactError, match="no recorded vector"):
        load_index(session, tmp_path, RUN_ID)
