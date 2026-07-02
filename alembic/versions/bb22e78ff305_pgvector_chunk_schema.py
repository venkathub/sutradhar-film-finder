"""pgvector extension + chunk schema (P2_SPEC §2.3, DEC-P2-1)

Enables the vector extension (image is pgvector/pgvector:0.8.4-pg17 — ships dense
``vector`` AND ``sparsevec``, so both hybrid legs live in Postgres) and creates the
``chunks`` + ``chunk_embeddings`` tables. No vector indexes: exact scan is correct at
seed-slice scale (~10² chunks); HNSW is a recorded catalog-scale revisit (DEC-P2-1).

Revision ID: bb22e78ff305
Revises: c636e4be00a5
Create Date: 2026-07-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import SPARSEVEC, Vector

revision = "bb22e78ff305"
down_revision = "c636e4be00a5"
branch_labels = None
depends_on = None

DENSE_DIM = 1024  # BGE-M3 (DEC-0002 default); dim is per embed_model row
SPARSE_DIM = 250_002  # XLM-R vocab (BGE-M3 lexical weights)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunks",
        sa.Column(
            "chunk_id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("work_id", sa.Uuid(), nullable=False),
        sa.Column("plot_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("chunker", sa.Text(), nullable=False),
        sa.Column("chunk_config", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("license", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("kind IN ('plot', 'metadata_card')", name=op.f("ck_chunks_kind")),
        sa.ForeignKeyConstraint(
            ["plot_id"], ["plot_texts.plot_id"], name=op.f("fk_chunks_plot_id_plot_texts")
        ),
        sa.ForeignKeyConstraint(
            ["version_id"], ["version.version_id"], name=op.f("fk_chunks_version_id_version")
        ),
        sa.ForeignKeyConstraint(["work_id"], ["work.work_id"], name=op.f("fk_chunks_work_id_work")),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "version_id",
            "kind",
            "chunker",
            "chunk_config",
            "seq",
            name=op.f("uq_chunks_version_id"),
        ),
    )
    op.create_index(op.f("ix_chunks_version_id"), "chunks", ["version_id"], unique=False)
    op.create_index(op.f("ix_chunks_work_id"), "chunks", ["work_id"], unique=False)
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("embed_model", sa.Text(), nullable=False),
        sa.Column("index_version", sa.Text(), nullable=False),
        sa.Column("dense", Vector(DENSE_DIM), nullable=False),
        sa.Column("sparse", SPARSEVEC(SPARSE_DIM), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.chunk_id"],
            name=op.f("fk_chunk_embeddings_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "chunk_id", "embed_model", "index_version", name=op.f("pk_chunk_embeddings")
        ),
    )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
    op.drop_index(op.f("ix_chunks_work_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_version_id"), table_name="chunks")
    op.drop_table("chunks")
    # The extension is left installed: other objects may depend on it and the image ships it.
