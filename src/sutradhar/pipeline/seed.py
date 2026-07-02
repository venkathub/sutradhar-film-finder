"""Typed loader for ``data-pipeline/seed_slice.yaml`` (P1 task 3).

The seed slice is the committed input to ingestion AND the curated-truth denominator for the
graph-coverage metric (P1_SPEC §1.10). The pydantic models here reject structural nonsense at
load time: dangling relationship targets, works without originals, duplicate QIDs, literary
sources with version lists, originals that claim to be remakes.

Backlog entries (§7 Q1 conditional adds) are names + reasons only — they carry no QIDs, create
no versions, and count in no denominator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sutradhar.graph.models import EdgeType, WorkType

# Version-level relationships only: sequel/based_on are work-level YAML keys, not versions.
SeedRelType = Literal["is_remake_of", "is_official_dub_of", "is_unofficial_remake_of"]

# Repo-relative default location of the committed slice.
DEFAULT_SEED_PATH = Path("data-pipeline/seed_slice.yaml")


class SeedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: SeedRelType
    of: str  # version key within the same work (proximate source, §2.7)


class SeedVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    language: str = Field(min_length=2)
    release_year: int
    country: Literal["indian", "foreign"]
    is_original: bool = False
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q\d+$")
    relationship: SeedRelationship | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _original_xor_relationship(self) -> Self:
        if self.is_original and self.relationship is not None:
            raise ValueError("an original version cannot also carry a relationship edge")
        if not self.is_original and self.relationship is None:
            raise ValueError("a non-original version must state its relationship")
        return self


class SeedWork(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    franchise: str = Field(min_length=1)
    work_type: WorkType
    primary_title: str = Field(min_length=1)
    original_language: str | None = None
    first_release_year: int
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q\d+$")
    is_sequel_of: str | None = None  # work key
    based_on: str | None = None  # work key (literary_source)
    versions: dict[str, SeedVersion] = Field(default_factory=dict)
    note: str | None = None

    @model_validator(mode="after")
    def _shape_by_work_type(self) -> Self:
        if self.work_type is WorkType.LITERARY_SOURCE:
            if self.versions:
                raise ValueError("a literary_source work has no film versions")
            if self.original_language is not None:
                raise ValueError("a literary_source work carries no original_language")
            if self.is_sequel_of or self.based_on:
                raise ValueError("a literary_source work cannot be a sequel or adaptation")
        else:
            if not self.versions:
                raise ValueError("a film work must list at least one version")
            if not any(v.is_original for v in self.versions.values()):
                raise ValueError("a film work must flag at least one original version")
            for key, version in self.versions.items():
                rel = version.relationship
                if rel is not None:
                    if rel.of not in self.versions:
                        raise ValueError(
                            f"version {key!r}: relationship target {rel.of!r} not in this work"
                        )
                    if rel.of == key:
                        raise ValueError(f"version {key!r}: relationship targets itself")
        return self


class SeedBacklogEntry(BaseModel):
    """Reported-but-unconfirmed version (§7 Q1): a name + why it was excluded. No QID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SeedMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_at: str
    description: str


class SeedSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    meta: SeedMeta
    works: dict[str, SeedWork]
    backlog: list[SeedBacklogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_work_integrity(self) -> Self:
        # Work-level edge targets must exist and be shape-correct.
        for key, work in self.works.items():
            if work.is_sequel_of is not None:
                target = self.works.get(work.is_sequel_of)
                if target is None:
                    raise ValueError(f"work {key!r}: is_sequel_of {work.is_sequel_of!r} unknown")
                if target.work_type is not WorkType.FILM:
                    raise ValueError(f"work {key!r}: is_sequel_of must target a film work")
            if work.based_on is not None:
                target = self.works.get(work.based_on)
                if target is None:
                    raise ValueError(f"work {key!r}: based_on {work.based_on!r} unknown")
                if target.work_type is not WorkType.LITERARY_SOURCE:
                    raise ValueError(f"work {key!r}: based_on must target a literary_source")
        # QIDs are the entity-resolution hub: unique across the whole slice.
        seen: dict[str, str] = {}
        for owner, qid in self._iter_qids():
            if qid in seen:
                raise ValueError(f"duplicate wikidata_qid {qid}: {seen[qid]} and {owner}")
            seen[qid] = owner
        # Version keys are globally unique (they name curated-truth rows and fixture refs).
        version_owner: dict[str, str] = {}
        for wkey, work in self.works.items():
            for vkey in work.versions:
                if vkey in version_owner:
                    raise ValueError(
                        f"duplicate version key {vkey!r} in {version_owner[vkey]!r} and {wkey!r}"
                    )
                version_owner[vkey] = wkey
        return self

    def _iter_qids(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for wkey, work in self.works.items():
            if work.wikidata_qid:
                pairs.append((f"work:{wkey}", work.wikidata_qid))
            for vkey, version in work.versions.items():
                if version.wikidata_qid:
                    pairs.append((f"version:{vkey}", version.wikidata_qid))
        return pairs

    # --- Curated-truth helpers (graph-coverage denominators, P1_SPEC §1.10) ---

    def franchises(self) -> dict[str, list[str]]:
        """franchise -> list of work keys."""
        result: dict[str, list[str]] = {}
        for key, work in self.works.items():
            result.setdefault(work.franchise, []).append(key)
        return result

    def version_count(self, franchise: str | None = None) -> int:
        """Number of curated-truth versions (optionally per franchise). Backlog excluded."""
        return sum(
            len(w.versions)
            for w in self.works.values()
            if franchise is None or w.franchise == franchise
        )


class EdgeTypeMismatchError(ValueError):
    """Raised if a seed relationship type falls outside the frozen edge-type enum."""


def load_seed_slice(path: Path | str = DEFAULT_SEED_PATH) -> SeedSlice:
    """Load + validate the committed seed slice. Raises on any structural violation."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    slice_ = SeedSlice.model_validate(raw)
    # Belt-and-braces: relationship types must be valid stored edge types (schema sync).
    valid = {e.value for e in EdgeType}
    for work in slice_.works.values():
        for version in work.versions.values():
            if version.relationship and version.relationship.type not in valid:
                raise EdgeTypeMismatchError(version.relationship.type)
    return slice_
