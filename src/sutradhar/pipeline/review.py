"""Human review gate (P1 task 12, DEC-P1-6): confirm/reject candidates, promote to edges.

Gate semantics (the enforced property):
- **Promotion** is the ONLY path from ``candidate_edges`` to ``edges``: a confirmed candidate
  becomes a ``human_verified=true``, HIGH edge (or corroborates an existing edge — sources
  merged, verified flag set), with ``promoted_edge_id`` linking the audit trail.
- **Rejection** records ``reviewed_by``/``reviewed_at`` and never touches ``edges``.
- **Skip** leaves the candidate ``proposed`` (e.g. out-of-slice truths — counted separately,
  excluded from the precision denominator).
- Confirm-as-proposed: the reviewer confirms the *relationship as stated* (type + direction).
  Supplying endpoint **bindings** (raw title → graph node) is resolution, not repair — the
  model's own bindings may be wrong (observed live: a bad language hint mis-bound a title).
- Work-level types (``based_on`` / ``is_sequel_of``) promote at the **work** level: version
  endpoints are mapped to their works; ``based_on`` may bind straight to a literary work.
- Rule-derived MEDIUM edges (dub tracks) have their own verification queue:
  :func:`verify_medium_edges` sets ``human_verified`` + a human source ref — the same
  human-gate semantics, applied to the builder's derivations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.models import Confidence, SourceId, SourceRef, sources_to_jsonb
from sutradhar.graph.schema import CandidateEdge, Edge, Version, Work

WORK_LEVEL_TYPES = {"is_sequel_of", "based_on"}
Verdict = Literal["confirm", "reject", "skip"]


class EndpointSpec(BaseModel):
    """How a decisions file names a graph node (exact match, not fuzzy — reviewer intent)."""

    model_config = ConfigDict(extra="forbid")

    title: str
    language: str | None = None
    year: int | None = None
    work: bool = False  # True → bind a Work (literary sources have no versions)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: uuid.UUID
    verdict: Verdict
    src: EndpointSpec | None = None
    dst: EndpointSpec | None = None
    note: str | None = None


@dataclass
class ReviewReport:
    confirmed: int = 0
    rejected: int = 0
    skipped: int = 0
    edges_created: int = 0
    edges_corroborated: int = 0
    medium_verified: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.confirmed + self.rejected

    @property
    def precision(self) -> float | None:
        return round(self.confirmed / self.decided, 4) if self.decided else None


class BindingError(ValueError):
    """A confirm verdict whose endpoints cannot be bound — promotion refused."""


def find_version(
    session: Session, title: str, language: str | None = None, year: int | None = None
) -> Version | None:
    stmt = select(Version).where(Version.title == title)
    if language is not None:
        stmt = stmt.where(Version.language == language)
    if year is not None:
        stmt = stmt.where(Version.release_year == year)
    matches = session.scalars(stmt).all()
    return matches[0] if len(matches) == 1 else None


def find_work(session: Session, title: str, year: int | None = None) -> Work | None:
    stmt = select(Work).where(Work.primary_title == title)
    if year is not None:
        stmt = stmt.where(Work.first_release_year == year)
    matches = session.scalars(stmt).all()
    return matches[0] if len(matches) == 1 else None


def _bind_endpoint(
    session: Session,
    spec: EndpointSpec | None,
    stored_version_id: uuid.UUID | None,
    work_level: bool,
) -> tuple[str, uuid.UUID] | None:
    """Resolve an endpoint to ('version'|'work', id). None = unbindable."""
    if spec is not None:
        if spec.work or (work_level and spec.work):
            work = find_work(session, spec.title, spec.year)
            return ("work", work.work_id) if work else None
        version = find_version(session, spec.title, spec.language, spec.year)
        if version is None:
            return None
        if work_level:
            return ("work", version.work_id)
        return ("version", version.version_id)
    if stored_version_id is not None:
        if work_level:
            version = session.get(Version, stored_version_id)
            return ("work", version.work_id) if version else None
        return ("version", stored_version_id)
    return None


def promote(
    session: Session,
    candidate: CandidateEdge,
    reviewer: str,
    src: EndpointSpec | None = None,
    dst: EndpointSpec | None = None,
    reviewed_at: datetime | None = None,
) -> tuple[Edge, bool]:
    """Confirm a candidate → create (or corroborate) a human-verified edge.

    Returns ``(edge, created)``. Raises :class:`BindingError` when endpoints can't bind.
    """
    reviewed_at = reviewed_at or datetime.now(tz=UTC)
    work_level = candidate.edge_type in WORK_LEVEL_TYPES
    src_bound = _bind_endpoint(session, src, candidate.src_version_id, work_level)
    dst_bound = _bind_endpoint(session, dst, candidate.dst_version_id, work_level)
    if src_bound is None or dst_bound is None:
        raise BindingError(
            f"candidate {candidate.candidate_id}: cannot bind "
            f"src={candidate.src_title_raw!r} dst={candidate.dst_title_raw!r}"
        )
    (src_kind, src_id), (dst_kind, dst_id) = src_bound, dst_bound
    if src_id == dst_id:
        raise BindingError(f"candidate {candidate.candidate_id}: self-edge after binding")

    refs = [
        SourceRef(
            source=SourceId.WIKIPEDIA,
            ref=f"{candidate.source_page}@{candidate.source_revision}",
            retrieved_at=reviewed_at,
        ),
        SourceRef(source=SourceId.HUMAN, ref=reviewer, retrieved_at=reviewed_at),
    ]

    existing = session.scalars(
        select(Edge).where(
            Edge.edge_type == candidate.edge_type,
            Edge.src_id == src_id,
            Edge.dst_id == dst_id,
        )
    ).first()
    if existing is not None:
        seen = {(s.get("source"), s.get("ref")) for s in existing.sources}
        new_refs = [r for r in refs if (r.source.value, r.ref) not in seen]
        if new_refs:
            existing.sources = existing.sources + sources_to_jsonb(new_refs)
        existing.human_verified = True
        existing.confidence = Confidence.HIGH.value
        edge, created = existing, False
    else:
        edge = Edge(
            edge_type=candidate.edge_type,
            src_kind=src_kind,
            src_id=src_id,
            dst_kind=dst_kind,
            dst_id=dst_id,
            confidence=Confidence.HIGH.value,
            sources=sources_to_jsonb(refs),
            human_verified=True,
        )
        session.add(edge)
        session.flush()
        created = True

    candidate.status = "confirmed"
    candidate.reviewed_by = reviewer
    candidate.reviewed_at = reviewed_at
    candidate.promoted_edge_id = edge.edge_id
    session.flush()
    return edge, created


def reject(
    session: Session,
    candidate: CandidateEdge,
    reviewer: str,
    reviewed_at: datetime | None = None,
) -> None:
    """Record the rejection. Never writes to edges — by construction and by test."""
    candidate.status = "rejected"
    candidate.reviewed_by = reviewer
    candidate.reviewed_at = reviewed_at or datetime.now(tz=UTC)
    session.flush()


def list_proposed(session: Session) -> list[CandidateEdge]:
    return list(
        session.scalars(
            select(CandidateEdge)
            .where(CandidateEdge.status == "proposed")
            .order_by(CandidateEdge.edge_type, CandidateEdge.src_title_raw)
        )
    )


def apply_decisions(session: Session, decisions: list[Decision], reviewer: str) -> ReviewReport:
    """Batch-apply a reviewed decisions file (the audit artifact of a review session)."""
    report = ReviewReport()
    for decision in decisions:
        candidate = session.get(CandidateEdge, decision.candidate_id)
        if candidate is None:
            report.errors.append(f"unknown candidate {decision.candidate_id}")
            continue
        if candidate.status != "proposed":
            report.errors.append(f"{decision.candidate_id} already {candidate.status}")
            continue
        if decision.verdict == "confirm":
            try:
                _edge, created = promote(session, candidate, reviewer, decision.src, decision.dst)
            except BindingError as exc:
                report.errors.append(str(exc))
                continue
            report.confirmed += 1
            if created:
                report.edges_created += 1
            else:
                report.edges_corroborated += 1
        elif decision.verdict == "reject":
            reject(session, candidate, reviewer)
            report.rejected += 1
        else:
            report.skipped += 1
    return report


def list_medium_rule_edges(session: Session) -> list[Edge]:
    """Rule-derived MEDIUM edges awaiting human verification (the builder's dub tracks)."""
    edges = session.scalars(
        select(Edge).where(
            Edge.confidence == Confidence.MEDIUM.value, Edge.human_verified.is_(False)
        )
    ).all()
    return [e for e in edges if any(s.get("source") == "rule" for s in e.sources)]


def verify_medium_edges(
    session: Session,
    reviewer: str,
    edge_ids: list[uuid.UUID],
    reviewed_at: datetime | None = None,
) -> int:
    """Human-verify explicitly listed rule-derived edges (same gate semantics).

    Verifying an edge also verifies its MEDIUM endpoint versions: confirming "this dub
    track's edge is real" is confirming the track itself (one human act, both records).
    """
    reviewed_at = reviewed_at or datetime.now(tz=UTC)
    human_ref = SourceRef(source=SourceId.HUMAN, ref=reviewer, retrieved_at=reviewed_at)
    verified = 0
    by_id = {e.edge_id: e for e in list_medium_rule_edges(session)}
    for edge_id in edge_ids:
        edge = by_id.get(edge_id)
        if edge is None:
            continue
        edge.human_verified = True
        edge.sources = edge.sources + sources_to_jsonb([human_ref])
        for endpoint_id in (edge.src_id, edge.dst_id):
            version = session.get(Version, endpoint_id)
            if (
                version is not None
                and version.confidence == "MEDIUM"
                and not version.human_verified
            ):
                version.human_verified = True
                version.sources = version.sources + sources_to_jsonb([human_ref])
        verified += 1
    session.flush()
    return verified


def list_medium_versions(session: Session) -> list[Version]:
    """MEDIUM, unverified version rows (e.g. a QID-less bilingual co-original)."""
    return list(
        session.scalars(
            select(Version).where(
                Version.confidence == Confidence.MEDIUM.value,
                Version.human_verified.is_(False),
            )
        )
    )


def verify_medium_versions(
    session: Session,
    reviewer: str,
    version_ids: list[uuid.UUID],
    reviewed_at: datetime | None = None,
) -> int:
    """Human-verify explicitly listed MEDIUM version rows (seed-curated tracks)."""
    reviewed_at = reviewed_at or datetime.now(tz=UTC)
    human_ref = SourceRef(source=SourceId.HUMAN, ref=reviewer, retrieved_at=reviewed_at)
    verified = 0
    by_id = {v.version_id: v for v in list_medium_versions(session)}
    for version_id in version_ids:
        version = by_id.get(version_id)
        if version is None:
            continue
        version.human_verified = True
        version.sources = version.sources + sources_to_jsonb([human_ref])
        verified += 1
    session.flush()
    return verified


def load_decisions(payload: dict[str, Any]) -> list[Decision]:
    return [Decision.model_validate(d) for d in payload["decisions"]]
