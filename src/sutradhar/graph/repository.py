"""Read-only repository backing the graph tools (P1 task 10, TOOL_SCHEMA v0 §2.5).

Plain Python functions implementing the **frozen tool contract** over the **ground-truth
views only** — `ground_truth_works` / `ground_truth_versions` / `ground_truth_edges` — so
CANDIDATE-tier rows and conflicted records are structurally invisible here. P5 wraps these
as served tools; P4 generates synthetic calls against the same signatures; this module is
the "contract is satisfiable" proof.

v0 semantics implemented (wording-level tightenings, pinned at the task-15 freeze):
- ``resolve_title.candidates[].score`` = the rapidfuzz-normalized 0–1 value (exact key = 1.0);
  ``ambiguous`` = candidates span more than one Work (GS-10).
- ``scope`` maps to ``version.country`` (``indian`` | ``foreign``; ``all`` = no filter).
- ``include_sequels`` traverses work-level ``is_sequel_of`` edges (both directions,
  transitively — a franchise walk). A sequel work's own original is labelled
  ``is_sequel_of`` relative to the queried work; its remakes keep ``is_remake_of``.
- ``era`` in ``refine_filter`` resolves against the set's original version's year
  (``newer`` = strictly later, ``older`` = strictly earlier, ``original`` = the flag).
- ``search_by_plot`` is NOT here — it needs P2 retrieval + calibrated abstain.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, Column, Integer, MetaData, Table, Text, Uuid, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from sutradhar.graph.schema import Person, VersionCast, VersionTitle
from sutradhar.pipeline.normalize import best_matches, match_key

# --- The verification-gate views, mapped read-only (separate MetaData: never migrated) ---

_views = MetaData()

gt_works = Table(
    "ground_truth_works",
    _views,
    Column("work_id", Uuid),
    Column("work_type", Text),
    Column("primary_title", Text),
    Column("original_language", Text),
    Column("first_release_year", Integer),
    Column("wikidata_qid", Text),
    Column("confidence", Text),
    Column("sources", JSONB),
    Column("human_verified", Boolean),
)

gt_versions = Table(
    "ground_truth_versions",
    _views,
    Column("version_id", Uuid),
    Column("work_id", Uuid),
    Column("wikidata_qid", Text),
    Column("title", Text),
    Column("language", Text),
    Column("release_year", Integer),
    Column("country", Text),
    Column("is_original", Boolean),
    Column("confidence", Text),
    Column("sources", JSONB),
)

gt_edges = Table(
    "ground_truth_edges",
    _views,
    Column("edge_id", Uuid),
    Column("edge_type", Text),
    Column("src_kind", Text),
    Column("src_id", Uuid),
    Column("dst_kind", Text),
    Column("dst_id", Uuid),
    Column("confidence", Text),
    Column("sources", JSONB),
    Column("human_verified", Boolean),
)

VERSION_EDGE_LABELS = ("is_remake_of", "is_official_dub_of", "is_unofficial_remake_of")
Scope = Literal["indian", "all", "foreign"]
Era = Literal["original", "newer", "older"]


# --- Result models (mirror TOOL_SCHEMA v0 result shapes) ---


class ResolvedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    work_id: uuid.UUID
    version_id: uuid.UUID | None
    matched_title: str
    language: str | None
    year: int | None
    score: float
    sources: list[dict[str, Any]]


class ResolveTitleResult(BaseModel):
    candidates: list[ResolvedCandidate]
    ambiguous: bool


class SourceWorkRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    work_id: uuid.UUID
    canonical_title: str
    work_type: str


class GetWorkResult(BaseModel):
    work_id: uuid.UUID
    canonical_title: str
    work_type: str
    original_language: str | None
    first_release_year: int | None
    source_work: SourceWorkRef | None
    based_on: list[uuid.UUID]
    sources: list[dict[str, Any]]
    confidence: str


class VersionEntry(BaseModel):
    version_id: uuid.UUID
    title: str
    language: str
    year: int | None
    cast_lead: list[str]
    relationship: str | None  # is_original_of | is_remake_of | is_official_dub_of | …
    is_original: bool
    sources: list[dict[str, Any]]
    confidence: str


class GetVersionsResult(BaseModel):
    original: VersionEntry | None
    versions: list[VersionEntry]


class RefineBy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str | None = None
    language: str | None = None
    year: int | None = None
    era: Era | None = None
    relationship: str | None = None


class RefinedVersion(BaseModel):
    version_id: uuid.UUID
    title: str
    language: str
    year: int | None
    relationship: str | None
    is_original: bool


class RefineFilterResult(BaseModel):
    versions: list[RefinedVersion]


# --- Internal helpers ---


def _lead_names(session: Session, version_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    if not version_ids:
        return {}
    rows = session.execute(
        select(VersionCast.version_id, Person.name, VersionCast.billing_order)
        .join(Person, Person.person_id == VersionCast.person_id)
        .where(VersionCast.version_id.in_(version_ids), VersionCast.role_kind == "lead")
        .order_by(VersionCast.billing_order)
    ).all()
    leads: dict[uuid.UUID, list[str]] = {}
    for version_id, name, _ in rows:
        leads.setdefault(version_id, []).append(name)
    return leads


def _relationship_labels(
    session: Session, version_ids: list[uuid.UUID], root_work_id: uuid.UUID
) -> dict[uuid.UUID, str | None]:
    """Version → relationship label. Original of the ROOT work → derived ``is_original_of``;
    a sequel work's original → ``is_sequel_of``; otherwise the version's outgoing gate-visible
    remake/dub edge label; None when no verified edge exists (an honest gap, GS-01 pre-review)."""
    if not version_ids:
        return {}
    edge_rows = session.execute(
        select(gt_edges.c.src_id, gt_edges.c.edge_type).where(
            gt_edges.c.src_id.in_(version_ids),
            gt_edges.c.edge_type.in_(VERSION_EDGE_LABELS),
        )
    ).all()
    by_src = {row.src_id: row.edge_type for row in edge_rows}
    version_rows = session.execute(
        select(gt_versions.c.version_id, gt_versions.c.work_id, gt_versions.c.is_original).where(
            gt_versions.c.version_id.in_(version_ids)
        )
    ).all()
    labels: dict[uuid.UUID, str | None] = {}
    for row in version_rows:
        if row.version_id in by_src:
            labels[row.version_id] = by_src[row.version_id]
        elif row.is_original:
            labels[row.version_id] = (
                "is_original_of" if row.work_id == root_work_id else "is_sequel_of"
            )
        else:
            labels[row.version_id] = None
    return labels


def _version_entry(
    row: Any, leads: dict[uuid.UUID, list[str]], labels: dict[uuid.UUID, str | None]
) -> VersionEntry:
    return VersionEntry(
        version_id=row.version_id,
        title=row.title,
        language=row.language,
        year=row.release_year,
        cast_lead=leads.get(row.version_id, []),
        relationship=labels.get(row.version_id),
        is_original=bool(row.is_original),
        sources=list(row.sources),
        confidence=row.confidence,
    )


# --- Tool-backing functions (signatures per TOOL_SCHEMA v0) ---


def resolve_title(session: Session, title: str, language: str | None = None) -> ResolveTitleResult:
    """Cross-script/fuzzy title resolution over the gate-visible title index (GS-07/10/11)."""
    query_key = match_key(title)
    rows = session.execute(
        select(
            VersionTitle.match_key,
            VersionTitle.title,
            gt_versions.c.version_id,
            gt_versions.c.work_id,
            gt_versions.c.language,
            gt_versions.c.release_year,
            gt_versions.c.sources,
        ).join_from(VersionTitle, gt_versions, VersionTitle.version_id == gt_versions.c.version_id)
    ).all()
    if language is not None:
        rows = [r for r in rows if r.language == language]

    scores = dict(best_matches(query_key, sorted({r.match_key for r in rows}), limit=50))
    best_per_version: dict[uuid.UUID, ResolvedCandidate] = {}
    for r in rows:
        score = scores.get(r.match_key)
        if score is None:
            continue
        current = best_per_version.get(r.version_id)
        if current is None or score > current.score:
            best_per_version[r.version_id] = ResolvedCandidate(
                work_id=r.work_id,
                version_id=r.version_id,
                matched_title=r.title,
                language=r.language,
                year=r.release_year,
                score=score,
                sources=list(r.sources),
            )
    candidates = sorted(best_per_version.values(), key=lambda c: (-c.score, c.matched_title))[:10]
    ambiguous = len({c.work_id for c in candidates}) > 1
    return ResolveTitleResult(candidates=candidates, ambiguous=ambiguous)


def get_work(session: Session, work_id: uuid.UUID) -> GetWorkResult | None:
    """Canonical Work + its literary source when one exists (GS-05)."""
    row = session.execute(select(gt_works).where(gt_works.c.work_id == work_id)).first()
    if row is None:
        return None
    based_on_rows = session.execute(
        select(gt_edges.c.dst_id).where(
            gt_edges.c.edge_type == "based_on", gt_edges.c.src_id == work_id
        )
    ).all()
    based_on = [r.dst_id for r in based_on_rows]
    source_work: SourceWorkRef | None = None
    if based_on:
        source_row = session.execute(
            select(gt_works).where(gt_works.c.work_id == based_on[0])
        ).first()
        if source_row is not None:
            source_work = SourceWorkRef(
                work_id=source_row.work_id,
                canonical_title=source_row.primary_title,
                work_type=source_row.work_type,
            )
    return GetWorkResult(
        work_id=row.work_id,
        canonical_title=row.primary_title,
        work_type=row.work_type,
        original_language=row.original_language,
        first_release_year=row.first_release_year,
        source_work=source_work,
        based_on=based_on,
        sources=list(row.sources),
        confidence=row.confidence,
    )


def _franchise_work_ids(session: Session, work_id: uuid.UUID) -> set[uuid.UUID]:
    """Transitive closure over work-level is_sequel_of edges, both directions (GS-06)."""
    seen = {work_id}
    frontier = {work_id}
    while frontier:
        rows = session.execute(
            select(gt_edges.c.src_id, gt_edges.c.dst_id).where(
                gt_edges.c.edge_type == "is_sequel_of",
                (gt_edges.c.src_id.in_(frontier)) | (gt_edges.c.dst_id.in_(frontier)),
            )
        ).all()
        found = {r.src_id for r in rows} | {r.dst_id for r in rows}
        frontier = found - seen
        seen |= found
    return seen


def get_versions(
    session: Session,
    work_id: uuid.UUID,
    scope: Scope = "indian",
    include_sequels: bool = False,
) -> GetVersionsResult:
    """All gate-visible Versions of a Work (optionally its franchise), typed + flagged."""
    work_ids = _franchise_work_ids(session, work_id) if include_sequels else {work_id}
    stmt = select(gt_versions).where(gt_versions.c.work_id.in_(work_ids))
    if scope != "all":
        stmt = stmt.where(gt_versions.c.country == scope)
    rows = session.execute(stmt.order_by(gt_versions.c.release_year)).all()

    ids = [r.version_id for r in rows]
    leads = _lead_names(session, ids)
    labels = _relationship_labels(session, ids, work_id)
    entries = [_version_entry(r, leads, labels) for r in rows]

    original = next(
        (
            e
            for r, e in zip(rows, entries, strict=True)
            if e.is_original and r.work_id == work_id and r.wikidata_qid is not None
        ),
        next(
            (
                e
                for r, e in zip(rows, entries, strict=True)
                if e.is_original and r.work_id == work_id
            ),
            None,
        ),
    )
    return GetVersionsResult(original=original, versions=entries)


def refine_filter(
    session: Session, version_set: list[uuid.UUID], by: RefineBy
) -> RefineFilterResult:
    """Narrow a conversational version set (GS-08 backtracking)."""
    if not version_set:
        return RefineFilterResult(versions=[])
    rows = session.execute(
        select(gt_versions).where(gt_versions.c.version_id.in_(version_set))
    ).all()
    ids = [r.version_id for r in rows]
    leads = _lead_names(session, ids)
    root_work = rows[0].work_id if rows else uuid.uuid4()
    labels = _relationship_labels(session, ids, root_work)

    original_years = [r.release_year for r in rows if r.is_original and r.release_year]
    pivot_year = (
        min(original_years)
        if original_years
        else min((r.release_year for r in rows if r.release_year), default=None)
    )

    kept = []
    for r in rows:
        if by.language is not None and r.language != by.language:
            continue
        if by.year is not None and r.release_year != by.year:
            continue
        if by.actor is not None:
            names = " ".join(leads.get(r.version_id, [])).casefold()
            if by.actor.casefold() not in names:
                continue
        if by.relationship is not None and labels.get(r.version_id) != by.relationship:
            continue
        if by.era is not None:
            if by.era == "original" and not r.is_original:
                continue
            if by.era == "newer" and (
                pivot_year is None or r.release_year is None or r.release_year <= pivot_year
            ):
                continue
            if by.era == "older" and (
                pivot_year is None or r.release_year is None or r.release_year >= pivot_year
            ):
                continue
        kept.append(r)

    return RefineFilterResult(
        versions=[
            RefinedVersion(
                version_id=r.version_id,
                title=r.title,
                language=r.language,
                year=r.release_year,
                relationship=labels.get(r.version_id),
                is_original=bool(r.is_original),
            )
            for r in kept
        ]
    )
