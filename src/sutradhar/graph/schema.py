"""Declarative schema for the Catalog + Remake-Graph store (P1_SPEC §2.2).

The tables that encode the hard problem:

- A **Work** is one film-story lineage (or a literary source, for GS-05); the original film and
  all its remakes/dubs are **Versions of the same Work**.
- **Edges** are a single polymorphic table (DEC-P1-1): remake/dub edges connect versions,
  sequel/based_on edges connect works — shape-enforced by CHECK constraints, endpoint existence
  by a validation trigger (created in the initial Alembic migration).
- Every record and edge carries ``confidence`` + ``sources[]`` (inline jsonb, DEC-P1-3).
- ``candidate_edges`` holds LLM proposals only; it is **not** an edge table and is never read by
  the ground-truth views (verification gate by construction).
- The verification-gate views (``ground_truth_works/versions/edges``) live in the migration
  (raw SQL — Alembic owns view DDL). Gate predicate (P1_SPEC §1.8, clarified per DATA_SOURCES.md
  tier table): sources non-empty AND no open conflict. MEDIUM rows pass the views ("live graph,
  flagged"); the golden-fixture validator separately enforces HIGH/human-verified.

``is_original_of`` is deliberately **not** a stored edge type: it is derived from
``version.is_original`` + the inverse presentation of incoming remake/dub edges (one source of
truth; no flag/edge divergence possible).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy import text as sql_text  # alias: Chunk has a `text` column that shadows it
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# --- Enumerations (CHECK-enforced; kept as tuples so tests/pydantic share one source) ---

WORK_TYPES = ("film", "literary_source")
CONFIDENCE_TIERS = ("HIGH", "MEDIUM")  # CANDIDATE is a *table* (candidate_edges), not a tier here
EDGE_TYPES = (
    "is_remake_of",
    "is_official_dub_of",
    "is_unofficial_remake_of",
    "is_sequel_of",
    "based_on",
)
VERSION_EDGE_TYPES = ("is_remake_of", "is_official_dub_of", "is_unofficial_remake_of")
WORK_EDGE_TYPES = ("is_sequel_of", "based_on")
NODE_KINDS = ("version", "work")
TITLE_KINDS = ("canonical", "aka", "dub", "transliteration")
ROLE_KINDS = ("lead", "support", "director")
CANDIDATE_STATUSES = ("proposed", "confirmed", "rejected")
CONFLICT_ENTITY_KINDS = ("work", "version", "edge")
CONFLICT_STATUSES = ("open", "resolved")
PLOT_SOURCES = ("wikipedia", "tmdb")
CHUNK_KINDS = ("plot", "metadata_card")  # P2_SPEC §2.2: plot chunks + per-Version metadata cards

# Embedding dimensions (P2_SPEC §2.3). Dense dim is per embed_model row — 1024 = BGE-M3
# (DEC-0002 default); a DEC-0002 A/B challenger with another dim would land as a second
# column/migration, an accepted cost of keeping the common case typed.
DENSE_DIM = 1024
SPARSE_DIM = 250_002  # XLM-RoBERTa vocab (BGE-M3 lexical weights)

# Names of the verification-gate views (DDL lives in the initial migration).
GROUND_TRUTH_VIEWS = ("ground_truth_works", "ground_truth_versions", "ground_truth_edges")


def _sql_in(values: tuple[str, ...]) -> str:
    """Render a tuple of enum values as a SQL IN-list literal for CHECK constraints."""
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


# Deterministic constraint names → reviewable autogenerate diffs + testable error messages.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for the graph schema (single metadata, naming convention applied)."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class _ProvenanceMixin:
    """Columns every gated record/edge carries (P1_SPEC §2.2): confidence + sources + audit."""

    confidence: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    human_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=text("now()"),
    )


class Work(_ProvenanceMixin, Base):
    """Canonical film-story lineage (or literary source) — the Work node."""

    __tablename__ = "work"

    work_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    work_type: Mapped[str] = mapped_column(Text, nullable=False)
    primary_title: Mapped[str] = mapped_column(Text, nullable=False)
    original_language: Mapped[str | None] = mapped_column(Text)  # null for literary sources
    first_release_year: Mapped[int | None] = mapped_column(Integer)
    wikidata_qid: Mapped[str | None] = mapped_column(Text, unique=True)

    __table_args__ = (
        CheckConstraint(f"work_type IN {_sql_in(WORK_TYPES)}", name="work_type"),
        CheckConstraint(f"confidence IN {_sql_in(CONFIDENCE_TIERS)}", name="confidence"),
    )


class Version(_ProvenanceMixin, Base):
    """Per-language film Version of a Work (original, remake, or dub track)."""

    __tablename__ = "version"

    version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    work_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("work.work_id"), nullable=False, index=True
    )
    wikidata_qid: Mapped[str | None] = mapped_column(Text, unique=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer, index=True)
    imdb_id: Mapped[str | None] = mapped_column(Text, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)  # BCP-47-ish: ml, ta, te, hi, ...
    release_year: Mapped[int | None] = mapped_column(Integer)
    country: Mapped[str | None] = mapped_column(Text)  # drives GS-09 scope: indian | foreign
    is_original: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))

    __table_args__ = (
        CheckConstraint(f"confidence IN {_sql_in(CONFIDENCE_TIERS)}", name="confidence"),
    )


class VersionTitle(Base):
    """Cross-script title match index: canonical/AKA/dub/transliteration rows (GS-07/10/11)."""

    __tablename__ = "version_title"

    title_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("version.version_id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str | None] = mapped_column(Text)
    script: Mapped[str | None] = mapped_column(Text)  # deva | taml | mlym | latn | ...
    language: Mapped[str | None] = mapped_column(Text)
    match_key: Mapped[str] = mapped_column(Text, nullable=False)  # normalized romanized key §2.4
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        CheckConstraint(f"kind IN {_sql_in(TITLE_KINDS)}", name="kind"),
        Index("ix_version_title_match_key", "match_key"),
    )


class Person(Base):
    """Cast/crew person (entity-resolved on Wikidata QID / TMDB id, never on name)."""

    __tablename__ = "person"

    person_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    wikidata_qid: Mapped[str | None] = mapped_column(Text, unique=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)


class VersionCast(Base):
    """Cast/crew membership per Version — the dub-vs-remake rule's evidence (lead overlap)."""

    __tablename__ = "version_cast"

    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("version.version_id"), primary_key=True
    )
    person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("person.person_id"), primary_key=True)
    role_kind: Mapped[str] = mapped_column(Text, primary_key=True)
    billing_order: Mapped[int | None] = mapped_column(Integer)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (CheckConstraint(f"role_kind IN {_sql_in(ROLE_KINDS)}", name="role_kind"),)


class Edge(_ProvenanceMixin, Base):
    """Single polymorphic typed-edge table (DEC-P1-1).

    Shape rules (CHECK-enforced):
      remake / dub / unofficial-remake : version -> version
      sequel / based_on                : work    -> work
    Endpoint existence (soft polymorphic FKs) is enforced by the ``edges_endpoints_exist``
    constraint trigger created in the initial migration.
    """

    __tablename__ = "edges"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    edge_type: Mapped[str] = mapped_column(Text, nullable=False)
    src_kind: Mapped[str] = mapped_column(Text, nullable=False)
    src_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    dst_kind: Mapped[str] = mapped_column(Text, nullable=False)
    dst_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        CheckConstraint(f"edge_type IN {_sql_in(EDGE_TYPES)}", name="edge_type"),
        CheckConstraint(f"src_kind IN {_sql_in(NODE_KINDS)}", name="src_kind"),
        CheckConstraint(f"dst_kind IN {_sql_in(NODE_KINDS)}", name="dst_kind"),
        CheckConstraint(f"confidence IN {_sql_in(CONFIDENCE_TIERS)}", name="confidence"),
        CheckConstraint("src_id <> dst_id", name="no_self_edge"),
        CheckConstraint(
            f"(edge_type IN {_sql_in(VERSION_EDGE_TYPES)}"
            " AND src_kind = 'version' AND dst_kind = 'version')"
            f" OR (edge_type IN {_sql_in(WORK_EDGE_TYPES)}"
            " AND src_kind = 'work' AND dst_kind = 'work')",
            name="type_shape",
        ),
        UniqueConstraint("edge_type", "src_id", "dst_id", name="uq_edges_edge_type_src_dst"),
        Index("ix_edges_src_id", "src_id"),
        Index("ix_edges_dst_id", "dst_id"),
    )


class CandidateEdge(Base):
    """LLM-proposed edge awaiting the human gate. NEVER read by ground-truth views."""

    __tablename__ = "candidate_edges"

    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    edge_type: Mapped[str] = mapped_column(Text, nullable=False)
    src_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("version.version_id"))
    dst_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("version.version_id"))
    src_title_raw: Mapped[str | None] = mapped_column(Text)
    dst_title_raw: Mapped[str | None] = mapped_column(Text)
    supporting_sentence: Mapped[str] = mapped_column(Text, nullable=False)
    source_page: Mapped[str] = mapped_column(Text, nullable=False)  # wikipedia page (evidence pin)
    source_revision: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    model_confidence: Mapped[float | None] = mapped_column(Float)
    extraction_run: Mapped[str] = mapped_column(Text, nullable=False)  # run hash (reproducibility)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'proposed'"))
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_edge_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("edges.edge_id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(f"edge_type IN {_sql_in(EDGE_TYPES)}", name="edge_type"),
        CheckConstraint(f"status IN {_sql_in(CANDIDATE_STATUSES)}", name="status"),
    )


class Conflict(Base):
    """Multi-source disagreement queue — never silently resolved (P1_SPEC §1.5)."""

    __tablename__ = "conflicts"

    conflict_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    entity_kind: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    field: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. release_year, edge_type
    values: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)  # both sides kept
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'open'"))
    resolution: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # {rule|human, chosen_value}
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(f"entity_kind IN {_sql_in(CONFLICT_ENTITY_KINDS)}", name="entity_kind"),
        CheckConstraint(f"status IN {_sql_in(CONFLICT_STATUSES)}", name="status"),
        Index("ix_conflicts_entity", "entity_kind", "entity_id", "status"),
    )


class PlotText(Base):
    """Plot/synopsis prose per Version — the P2 embedding corpus (content, not facts)."""

    __tablename__ = "plot_texts"

    plot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("version.version_id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    revision_id: Mapped[str | None] = mapped_column(Text)  # Wikipedia revision pin (CC BY-SA)
    license: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (CheckConstraint(f"source IN {_sql_in(PLOT_SOURCES)}", name="source"),)


class Chunk(Base):
    """Embeddable retrieval unit (P2_SPEC §2.3): plot chunk or per-Version metadata card.

    ``text`` is header + body — exactly what gets embedded, so ``content_hash`` keys the
    recorded-artifact lookup (``ArtifactEmbeddings``). ``chunk_config`` is the ablation key
    (e.g. ``512tok_15pct``); the same source doc yields one chunk row set per config.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("version.version_id"), nullable=False, index=True
    )
    # Denormalized for chunk→Work aggregation (one hop, no join through version).
    work_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("work.work_id"), nullable=False, index=True
    )
    plot_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("plot_texts.plot_id")
    )  # NULL for metadata cards
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # order within source doc
    text: Mapped[str] = mapped_column(Text, nullable=False)  # header + body (what was embedded)
    language: Mapped[str | None] = mapped_column(Text)
    chunker: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. 'recursive_para'
    chunk_config: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. '512tok_15pct'
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)  # sha256(text)
    license: Mapped[str] = mapped_column(Text, nullable=False)  # carries CC BY-SA from plot_texts
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("now()")
    )

    __table_args__ = (
        CheckConstraint(f"kind IN {_sql_in(CHUNK_KINDS)}", name="kind"),
        UniqueConstraint("version_id", "kind", "chunker", "chunk_config", "seq"),
    )


class ChunkEmbedding(Base):
    """BGE-M3 dense + sparse vectors per chunk (P2_SPEC §2.3).

    Separate table so the DEC-0002 A/B and re-embeds coexist: PK is
    ``(chunk_id, embed_model, index_version)``. Sparse scoring runs IN-DB via ``<#>``
    (sparsevec negative inner product, native since pgvector 0.7.0) — no app-side math.
    Exact scan at seed-slice scale; HNSW is a recorded catalog-scale revisit, not built.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    embed_model: Mapped[str] = mapped_column(Text, primary_key=True)  # id+revision
    index_version: Mapped[str] = mapped_column(Text, primary_key=True)  # artifact run id
    dense: Mapped[Any] = mapped_column(Vector(DENSE_DIM), nullable=False)
    # BGE-M3 lexical weights over the 250,002-dim XLM-R vocab; ~10²–10³ nnz per chunk,
    # far under sparsevec's 16,000-nnz storage cap.
    sparse: Mapped[Any] = mapped_column(SPARSEVEC(SPARSE_DIM), nullable=False)
