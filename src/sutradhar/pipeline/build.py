"""Graph builder: dub-vs-remake rule, rule-vs-edge cross-check, dub-track edge derivation,
integrity checks (P1 task 9, §2.3 step 5).

The **dub-vs-remake rule** (DATA_SOURCES.md, verbatim): same lead cast carrying across
language versions → ``is_official_dub_of``; disjoint cast → ``is_remake_of``. Here it runs
two ways:

1. **Cross-check** every existing version-level edge against the rule. Agreement is counted
   (corroboration evidence); disagreement opens a ``conflicts`` row on the edge — **never a
   silent re-type** — which hides the edge from the ground-truth views until a human resolves.
2. **Derivation** of ``is_official_dub_of`` edges for QID-less, non-original tracks (Baahubali
   hi/ml, Devadas ta): these versions exist *only* as tracks of their film (no external record
   of their own — same film, same cast by construction), so the dub edge makes the modelling
   explicit. Confidence **MEDIUM** ("derived rule with no corroboration", tier table) with an
   honest ``rule`` source ref — the human gate (task 12) promotes them.

**What the builder never does:** write remake edges from seed curation. Missing remake edges
(the Wikidata gap) belong to extraction + human review — that separation keeps the lift
metric honest. Edge origins are separable via ``sources[0].source``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.models import Confidence, EdgeType, SourceId, SourceRef, sources_to_jsonb
from sutradhar.graph.schema import Conflict, Edge, Version, VersionCast, Work

# Overlap (relative to the smaller lead set) at/above which shared casting means "same film".
DUB_OVERLAP_THRESHOLD = 0.5

RULE_REF_DUB_TRACK = "dub-track-rule"
RULE_REF_CAST_OVERLAP = "lead-cast-overlap-rule"


def classify_dub_vs_remake(
    src_leads: set[uuid.UUID], dst_leads: set[uuid.UUID]
) -> Literal["is_official_dub_of", "is_remake_of"] | None:
    """The pure rule. ``None`` = insufficient evidence (either side has no known leads)."""
    if not src_leads or not dst_leads:
        return None
    overlap = len(src_leads & dst_leads) / min(len(src_leads), len(dst_leads))
    return "is_official_dub_of" if overlap >= DUB_OVERLAP_THRESHOLD else "is_remake_of"


@dataclass
class BuildReport:
    edges_checked: int = 0
    rule_agreements: int = 0
    rule_conflicts_opened: int = 0
    rule_insufficient_evidence: int = 0
    dub_edges_derived: int = 0
    anomalies: list[str] = field(default_factory=list)
    works: int = 0
    versions: int = 0
    edges_total: int = 0


def _lead_sets(session: Session) -> dict[uuid.UUID, set[uuid.UUID]]:
    leads: dict[uuid.UUID, set[uuid.UUID]] = {}
    for row in session.scalars(select(VersionCast).where(VersionCast.role_kind == "lead")):
        leads.setdefault(row.version_id, set()).add(row.person_id)
    return leads


def _open_edge_conflict_exists(session: Session, edge_id: uuid.UUID) -> bool:
    return (
        session.scalars(
            select(Conflict).where(
                Conflict.entity_kind == "edge",
                Conflict.entity_id == edge_id,
                Conflict.field == "edge_type",
            )
        ).first()
        is not None
    )


def build_graph(session: Session, retrieved_at: datetime | None = None) -> BuildReport:
    """Run the rule cross-check + dub-track derivation + integrity checks. Idempotent."""
    retrieved_at = retrieved_at or datetime.now(tz=UTC)
    report = BuildReport()

    works = session.scalars(select(Work)).all()
    versions = session.scalars(select(Version)).all()
    edges = session.scalars(select(Edge)).all()
    leads = _lead_sets(session)
    by_work: dict[uuid.UUID, list[Version]] = {}
    for v in versions:
        by_work.setdefault(v.work_id, []).append(v)

    # --- 1. Rule-vs-edge cross-check (never a silent re-type) ---
    version_edge_types = {"is_remake_of", "is_official_dub_of", "is_unofficial_remake_of"}
    for edge in edges:
        if edge.edge_type not in version_edge_types:
            continue
        report.edges_checked += 1
        verdict = classify_dub_vs_remake(
            leads.get(edge.src_id, set()), leads.get(edge.dst_id, set())
        )
        if verdict is None:
            report.rule_insufficient_evidence += 1
        elif verdict == edge.edge_type or (
            verdict == "is_remake_of" and edge.edge_type == "is_unofficial_remake_of"
        ):
            report.rule_agreements += 1
        else:
            if not _open_edge_conflict_exists(session, edge.edge_id):
                session.add(
                    Conflict(
                        entity_kind="edge",
                        entity_id=edge.edge_id,
                        field="edge_type",
                        values=[
                            {"value": edge.edge_type, "source": "edge.sources (as stored)"},
                            {"value": verdict, "source": f"rule:{RULE_REF_CAST_OVERLAP}"},
                        ],
                    )
                )
                report.rule_conflicts_opened += 1

    # --- 2. Dub-track edge derivation (QID-less, non-original tracks → primary original) ---
    existing_pairs = {(e.edge_type, e.src_id, e.dst_id) for e in edges}
    for work in works:
        members = by_work.get(work.work_id, [])
        primary = next((v for v in members if v.is_original and v.wikidata_qid is not None), None)
        if primary is None:
            continue
        for track in members:
            if (
                track.wikidata_qid is not None
                or track.tmdb_id is not None
                or track.is_original
                or track.version_id == primary.version_id
            ):
                continue
            key = ("is_official_dub_of", track.version_id, primary.version_id)
            if key in existing_pairs:
                continue
            session.add(
                Edge(
                    edge_type=EdgeType.IS_OFFICIAL_DUB_OF.value,
                    src_kind="version",
                    src_id=track.version_id,
                    dst_kind="version",
                    dst_id=primary.version_id,
                    confidence=Confidence.MEDIUM.value,  # derived rule, no corroboration
                    sources=sources_to_jsonb(
                        [
                            SourceRef(
                                source=SourceId.RULE,
                                ref=RULE_REF_DUB_TRACK,
                                field="edge_type",
                                retrieved_at=retrieved_at,
                            )
                        ]
                    ),
                )
            )
            existing_pairs.add(key)
            report.dub_edges_derived += 1
    session.flush()

    # --- 3. Integrity / anomaly checks (QID hub is the merge key — GS-10 by construction) ---
    for work in works:
        members = by_work.get(work.work_id, [])
        originals = [v for v in members if v.is_original]
        if work.work_type == "film" and not originals:
            report.anomalies.append(f"work {work.primary_title!r} has no original version")
        if len(originals) > 2:
            report.anomalies.append(f"work {work.primary_title!r} flags {len(originals)} originals")

    report.works = len(works)
    report.versions = len(versions)
    report.edges_total = len(session.scalars(select(Edge)).all())
    return report
