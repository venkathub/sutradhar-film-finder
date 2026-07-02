"""Sparse-channel scoring (P2 task 8, DEC-P2-2): BGE-M3 lexical weights, scored IN-DB.

The query's lexical weights (token id → weight over the 250,002-dim XLM-R vocab, 0-based
as BGE-M3 emits them) become a pgvector ``sparsevec`` literal (``{idx:w,…}/dims``,
1-based — the shift is handled by ``pgvector.SparseVector``), and Postgres computes the
inner product natively via ``<#>`` (negative inner product). No app-side sparse math:
SQL is the scorer, exactly the same operator the integration tests hand-verified.

Zero-overlap chunks score 0 and are **excluded** from the channel ranking — a no-signal
row must not collect RRF mass just for existing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pgvector import SparseVector
from sqlalchemy import text
from sqlalchemy.orm import Session

from sutradhar.graph.schema import SPARSE_DIM


def sparse_literal(weights: dict[int, float], dims: int = SPARSE_DIM) -> str:
    """0-based token-id weights → pgvector sparsevec text literal (1-based indices)."""
    if any(i < 0 or i >= dims for i in weights):
        raise ValueError(f"token id out of range for sparsevec({dims})")
    return str(SparseVector(weights, dims).to_text())


@dataclass(frozen=True)
class ChannelHit:
    """One chunk hit in a retrieval channel, with its raw channel score."""

    chunk_id: uuid.UUID
    work_id: uuid.UUID
    content_hash: str
    score: float


def sparse_top_chunks(
    session: Session,
    weights: dict[int, float],
    *,
    chunk_config: str,
    embed_model: str,
    index_version: str,
    top_n: int,
) -> list[ChannelHit]:
    """In-DB sparse channel: top-N chunks by lexical-weight inner product (desc)."""
    if not weights:
        return []
    rows = session.execute(
        text(
            "SELECT c.chunk_id, c.work_id, c.content_hash, -(e.sparse <#> :q) AS score "
            "FROM chunk_embeddings e JOIN chunks c ON c.chunk_id = e.chunk_id "
            "WHERE c.chunk_config = :cfg AND e.embed_model = :model "
            "AND e.index_version = :run AND (e.sparse <#> :q) < 0 "
            "ORDER BY e.sparse <#> :q, c.content_hash LIMIT :n"
        ),
        {
            "q": sparse_literal(weights),
            "cfg": chunk_config,
            "model": embed_model,
            "run": index_version,
            "n": top_n,
        },
    ).all()
    return [
        ChannelHit(
            chunk_id=r.chunk_id,
            work_id=r.work_id,
            content_hash=r.content_hash,
            score=float(r.score),
        )
        for r in rows
    ]
