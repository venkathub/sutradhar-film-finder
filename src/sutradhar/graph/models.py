"""Domain + provenance types for the graph store (P1 task 2, DEC-P1-3).

Pydantic models validated **at the write boundary**: every record/edge that ingestion or the
review gate writes to Postgres passes through these first, so structurally-bad data (empty
``sources[]``, unknown source ids, edge shape violations) fails before any insert — the DB
CHECKs/triggers are the backstop, not the first line.

Gate helpers mirror the two gate layers (DEC-P1-7):
- :func:`passes_gate_view` — the ``ground_truth_*`` view predicate (sources non-empty AND no
  open conflict). MEDIUM passes, flagged.
- :func:`golden_eligible` — the stricter golden-fixture rule (HIGH OR human-verified).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Enumerations (kept in sync with sutradhar.graph.schema tuples; drift is unit-tested) ---


class SourceId(StrEnum):
    """Where a claim comes from (P1_SPEC §2.2 SourceRef contract).

    ``RULE`` (DEC-P1-3 amendment, task 9): evidence produced by a documented deterministic
    rule (e.g. the dub-vs-remake lead-cast rule) — recorded honestly as its own source id,
    never disguised as a human or external source. Rule-only claims are MEDIUM by the
    tier table ("a derived rule with no corroboration").
    """

    WIKIDATA = "wikidata"
    TMDB = "tmdb"
    IMDB = "imdb"
    WIKIPEDIA = "wikipedia"
    HUMAN = "human"
    RULE = "rule"


class Confidence(StrEnum):
    """Live-graph confidence tiers. CANDIDATE is a table (candidate_edges), not a tier."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


class WorkType(StrEnum):
    FILM = "film"
    LITERARY_SOURCE = "literary_source"


class EdgeType(StrEnum):
    """The five stored edge types — is_original_of is derived, never stored."""

    IS_REMAKE_OF = "is_remake_of"
    IS_OFFICIAL_DUB_OF = "is_official_dub_of"
    IS_UNOFFICIAL_REMAKE_OF = "is_unofficial_remake_of"
    IS_SEQUEL_OF = "is_sequel_of"
    BASED_ON = "based_on"


class NodeKind(StrEnum):
    VERSION = "version"
    WORK = "work"


class TitleKind(StrEnum):
    CANONICAL = "canonical"
    AKA = "aka"
    DUB = "dub"
    TRANSLITERATION = "transliteration"


class RoleKind(StrEnum):
    LEAD = "lead"
    SUPPORT = "support"
    DIRECTOR = "director"


VERSION_EDGE_TYPES = frozenset(
    {EdgeType.IS_REMAKE_OF, EdgeType.IS_OFFICIAL_DUB_OF, EdgeType.IS_UNOFFICIAL_REMAKE_OF}
)
WORK_EDGE_TYPES = frozenset({EdgeType.IS_SEQUEL_OF, EdgeType.BASED_ON})


# --- Provenance ---


class SourceRef(BaseModel):
    """One element of a record/edge's ``sources[]`` (inline jsonb, DEC-P1-3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SourceId
    ref: str = Field(min_length=1)  # "Q1618487" | "tmdb:266856" | "tt3417422" | "<page>@<rev>"
    field: str | None = None  # which field this source vouches for (optional)
    retrieved_at: datetime | None = None

    def to_jsonb(self) -> dict[str, Any]:
        """jsonb-compatible dict (ISO datetimes, nulls dropped) for the sources[] column."""
        return self.model_dump(mode="json", exclude_none=True)


def sources_to_jsonb(sources: list[SourceRef]) -> list[dict[str, Any]]:
    """Serialize a validated sources[] list for insertion into a jsonb column."""
    return [s.to_jsonb() for s in sources]


class _GatedRecord(BaseModel):
    """Base for every record/edge behind the verification gate: provenance is mandatory."""

    model_config = ConfigDict(extra="forbid")

    confidence: Confidence
    sources: list[SourceRef] = Field(min_length=1)
    human_verified: bool = False

    @field_validator("sources")
    @classmethod
    def _sources_non_empty(cls, v: list[SourceRef]) -> list[SourceRef]:
        # min_length already enforces this; kept explicit so the error message names the rule.
        if not v:
            raise ValueError("sources[] must not be empty (verification gate, DATA_SOURCES.md)")
        return v


# --- Node records ---


class WorkRecord(_GatedRecord):
    """Canonical Work node (film lineage or literary source)."""

    work_id: uuid.UUID | None = None  # None until inserted (DB default gen_random_uuid())
    work_type: WorkType
    primary_title: str = Field(min_length=1)
    original_language: str | None = None
    first_release_year: int | None = None
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q\d+$")

    @model_validator(mode="after")
    def _literary_source_has_no_language(self) -> Self:
        if self.work_type is WorkType.LITERARY_SOURCE and self.original_language is not None:
            raise ValueError("literary_source works carry no original_language (P1_SPEC §2.2)")
        return self


class VersionRecord(_GatedRecord):
    """Per-language film Version of a Work."""

    version_id: uuid.UUID | None = None
    work_id: uuid.UUID
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q\d+$")
    tmdb_id: int | None = None
    imdb_id: str | None = Field(default=None, pattern=r"^tt\d+$")
    title: str = Field(min_length=1)
    language: str = Field(min_length=2)  # BCP-47-ish: ml, ta, te, hi, kn, si, zh, ...
    release_year: int | None = None
    country: str | None = None  # indian | foreign (drives GS-09 scope)
    is_original: bool = False


class VersionTitleRecord(BaseModel):
    """Cross-script title-index row (not confidence-gated; provenance still mandatory)."""

    model_config = ConfigDict(extra="forbid")

    title_id: uuid.UUID | None = None
    version_id: uuid.UUID
    title: str = Field(min_length=1)
    kind: TitleKind | None = None
    script: str | None = None
    language: str | None = None
    match_key: str = Field(min_length=1)
    sources: list[SourceRef] = Field(min_length=1)


class PersonRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: uuid.UUID | None = None
    name: str = Field(min_length=1)
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q\d+$")
    tmdb_id: int | None = None
    sources: list[SourceRef] = Field(min_length=1)


class VersionCastRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: uuid.UUID
    person_id: uuid.UUID
    role_kind: RoleKind
    billing_order: int | None = None
    sources: list[SourceRef] = Field(min_length=1)


# --- Edge record (shape rules mirrored from the DB CHECKs — fail before insert) ---


class EdgeRecord(_GatedRecord):
    """Typed edge. Shape: remake/dub = version→version; sequel/based_on = work→work."""

    edge_id: uuid.UUID | None = None
    edge_type: EdgeType
    src_kind: NodeKind
    src_id: uuid.UUID
    dst_kind: NodeKind
    dst_id: uuid.UUID

    @model_validator(mode="after")
    def _shape_and_self_edge(self) -> Self:
        if self.src_id == self.dst_id:
            raise ValueError("self-edge: src_id and dst_id must differ")
        required = NodeKind.VERSION if self.edge_type in VERSION_EDGE_TYPES else NodeKind.WORK
        if self.src_kind is not required or self.dst_kind is not required:
            raise ValueError(
                f"edge shape violation: {self.edge_type} requires "
                f"{required.value}->{required.value}, got "
                f"{self.src_kind.value}->{self.dst_kind.value}"
            )
        return self


# --- Gate predicate helpers (DEC-P1-7 layers) ---


def passes_gate_view(sources: list[SourceRef], has_open_conflict: bool) -> bool:
    """The ``ground_truth_*`` view predicate: sources non-empty AND no open conflict.

    MEDIUM rows pass (live-but-flagged); CANDIDATE never reaches this predicate — it lives
    in candidate_edges, which no view reads.
    """
    return len(sources) > 0 and not has_open_conflict


def golden_eligible(confidence: Confidence, human_verified: bool) -> bool:
    """The golden-fixture layer: HIGH or human-verified only (GOLDEN_SET_SCENARIOS.md)."""
    return confidence is Confidence.HIGH or human_verified
