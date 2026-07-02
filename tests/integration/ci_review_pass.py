"""Shared CI review-pass helper: mirror the real 2026-07-02 review semantics on the
fixture chain, keyed by (edge_type, src_raw, dst_raw) instead of live candidate UUIDs.

The real pass is the committed ``data-pipeline/review_decisions_20260702.yaml`` (applied to
the live DB); CI rebuilds the same verified graph from fixtures so the golden validator and
named regressions test identical semantics.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.schema import CandidateEdge
from sutradhar.pipeline.review import (
    EndpointSpec,
    list_medium_rule_edges,
    list_medium_versions,
    promote,
    verify_medium_edges,
    verify_medium_versions,
)

V = EndpointSpec

# (edge_type, src_title_raw, dst_title_raw) -> (src spec, dst spec); mirrors the confirmed
# subset of review_decisions_20260702.yaml that the CI fixture pages can propose.
CI_CONFIRMATIONS: dict[tuple[str, str, str], tuple[EndpointSpec, EndpointSpec]] = {
    ("is_remake_of", "Drishya", "Drishyam"): (
        V(title="Drishya", language="kn"),
        V(title="Drishyam", language="ml"),
    ),
    ("is_remake_of", "Drushyam", "Drishyam"): (
        V(title="Drushyam", language="te"),
        V(title="Drishyam", language="ml"),
    ),
    ("is_remake_of", "Papanasam", "Drishyam"): (
        V(title="Papanasam", language="ta"),
        V(title="Drishyam", language="ml"),
    ),
    ("is_remake_of", "Dharmayuddhaya", "Drishyam"): (
        V(title="Dharmayuddhaya", language="si"),
        V(title="Drishyam", language="ml"),
    ),
    ("is_remake_of", "Sheep Without a Shepherd", "Drishyam"): (
        V(title="Sheep Without a Shepherd", language="zh"),
        V(title="Drishyam", language="ml"),
    ),
    ("is_remake_of", "Drishya 2", "Drishyam 2"): (
        V(title="Drishya 2", language="kn"),
        V(title="Drishyam 2", language="ml"),
    ),
    ("is_remake_of", "Drushyam 2", "Drishyam 2"): (
        V(title="Drushyam 2", language="te"),
        V(title="Drishyam 2", language="ml"),
    ),
    ("is_remake_of", "Drishyam 2", "Drishyam"): (
        V(title="Drishyam 2", language="hi"),
        V(title="Drishyam 2", language="ml"),
    ),
    ("is_remake_of", "Chandramukhi", "Apthamitra"): (
        V(title="Chandramukhi", language="ta"),
        V(title="Apthamitra", language="kn"),
    ),
    ("is_remake_of", "Chandramukhi", "Manichitrathazhu"): (
        V(title="Chandramukhi", language="ta"),
        V(title="Manichitrathazhu", language="ml"),
    ),
    ("is_remake_of", "Apthamitra", "Manichithrathazhu"): (
        V(title="Apthamitra", language="kn"),
        V(title="Manichitrathazhu", language="ml"),
    ),
    ("is_remake_of", "Apthamitra", "Manichitrathazhu"): (
        V(title="Apthamitra", language="kn"),
        V(title="Manichitrathazhu", language="ml"),
    ),
    ("is_remake_of", "Bhool Bhulaiyaa", "Manichitrathazhu"): (
        V(title="Bhool Bhulaiyaa", language="hi"),
        V(title="Manichitrathazhu", language="ml"),
    ),
    ("is_remake_of", "Rajmohol", "Manichitrathazhu"): (
        V(title="Rajmohol", language="bn"),
        V(title="Manichitrathazhu", language="ml"),
    ),
    ("based_on", "Devadasu (1953 film)", "Devdas"): (
        V(title="Devadasu", language="te"),
        V(title="Devdas", year=1917, work=True),
    ),
}


def apply_ci_review_pass(session: Session, reviewer: str = "ci-mirror") -> int:
    """Promote the CI-mirrored confirmations + verify MEDIUM dub edges. Returns confirms."""
    confirmed = 0
    for candidate in session.scalars(
        select(CandidateEdge).where(CandidateEdge.status == "proposed")
    ).all():
        key = (candidate.edge_type, candidate.src_title_raw or "", candidate.dst_title_raw or "")
        specs = CI_CONFIRMATIONS.get(key)
        if specs is None:
            continue
        promote(session, candidate, reviewer, src=specs[0], dst=specs[1])
        confirmed += 1
    pending = list_medium_rule_edges(session)
    verify_medium_edges(session, reviewer, [e.edge_id for e in pending])
    # Remaining MEDIUM versions (e.g. the QID-less bilingual co-original) verified explicitly,
    # mirroring the real pass.
    leftover = list_medium_versions(session)
    verify_medium_versions(session, reviewer, [v.version_id for v in leftover])
    return confirmed
