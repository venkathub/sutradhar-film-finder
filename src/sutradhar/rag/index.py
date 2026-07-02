"""Index loader (P2 task 7): sealed artifact run → ``chunk_embeddings`` rows.

Bridges the two halves of the compute-placement rule (ROADMAP §2): the GPU session
produced vectors as verified artifacts; this loader (laptop, no models) joins them onto
the ``chunks`` table **by ``content_hash``** — the same key the corpus builder wrote and
``ArtifactEmbeddings`` serves — and materializes pgvector rows for in-DB dense cosine and
sparse ``<#>`` scoring.

Strictness contract (never silently degrade):

- the run is MANIFEST-verified before a single row is written (``ArtifactRun.open``);
- a DB chunk with **no vector in the bank** is a hard error → the corpus changed after
  the export; rebuild the corpus to match the run, or re-run ``make gpu-embed``;
- loading is idempotent per ``(embed_model, index_version)``: delete-and-reinsert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pgvector import SparseVector
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sutradhar.graph.schema import SPARSE_DIM, Chunk, ChunkEmbedding
from sutradhar.rag.artifacts import ArtifactRun, MissingArtifactError
from sutradhar.rag.chunking import CHUNKER_NAME


class IndexReport(BaseModel):
    run_id: str
    embed_model: str
    configs: list[str]
    rows_loaded: dict[str, int] = Field(default_factory=dict)  # per chunk_config
    unused_bank_vectors: dict[str, int] = Field(default_factory=dict)  # in run, not in DB


def _bank(run: ArtifactRun, bank: str) -> dict[str, tuple[np.ndarray, dict[int, float]]]:
    hashes: list[str] = json.loads(run.path(f"{bank}_hashes.json").read_text(encoding="utf-8"))
    dense = np.load(run.path(f"{bank}_dense.npy"))
    sparse: list[dict[str, float]] = json.loads(
        run.path(f"{bank}_sparse.json").read_text(encoding="utf-8")
    )
    return {
        h: (vec.astype(np.float32), {int(k): float(v) for k, v in weights.items()})
        for h, vec, weights in zip(hashes, dense, sparse, strict=True)
    }


def load_index(
    session: Session, artifacts_root: Path, run_id: str, sparse_dim: int = SPARSE_DIM
) -> IndexReport:
    """Load every config bank of a sealed run into ``chunk_embeddings``."""
    run = ArtifactRun.open(artifacts_root, run_id)  # hard MANIFEST verification
    meta = json.loads(run.path("meta.json").read_text(encoding="utf-8"))
    embed_model = str(meta["embed_model"])
    report = IndexReport(run_id=run_id, embed_model=embed_model, configs=list(meta["configs"]))

    # Idempotent per (embed_model, index_version): a reload never duplicates rows.
    session.execute(
        delete(ChunkEmbedding).where(
            ChunkEmbedding.embed_model == embed_model,
            ChunkEmbedding.index_version == run_id,
        )
    )

    for config in report.configs:
        vectors = _bank(run, f"corpus_{config}")
        chunks = session.execute(
            select(Chunk.chunk_id, Chunk.content_hash).where(
                Chunk.chunker == CHUNKER_NAME, Chunk.chunk_config == config
            )
        ).all()
        missing = sorted({c.content_hash for c in chunks} - set(vectors))
        if missing:
            raise MissingArtifactError(
                f"run {run_id}, config {config}: {len(missing)} DB chunk(s) have no recorded "
                f"vector (first: {missing[0][:12]}…) — the corpus changed after the export; "
                "rebuild the corpus to match the run or re-run `make gpu-embed`"
            )
        for chunk in chunks:
            dense_vec, sparse_weights = vectors[chunk.content_hash]
            session.add(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    embed_model=embed_model,
                    index_version=run_id,
                    dense=dense_vec,
                    sparse=SparseVector(sparse_weights, sparse_dim),
                )
            )
        report.rows_loaded[config] = len(chunks)
        report.unused_bank_vectors[config] = len(set(vectors) - {c.content_hash for c in chunks})
    session.flush()
    return report
