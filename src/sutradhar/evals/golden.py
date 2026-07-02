"""Golden fixture schema + validator (P1 task 14).

Fixture schema per ``GOLDEN_SET_SCENARIOS.md``; the validator enforces the golden gate
(stricter than the ground-truth views, DEC-P1-7 layer 3):

1. Schema validity (pydantic; ``expected_tool_calls`` validate against TOOL_SCHEMA v0 in the
   task-15 conformance test).
2. **Graph verification**: every expected version/relationship must exist gate-visibly AND be
   golden-eligible (HIGH or human-verified) — a MEDIUM-unverified-backed or conflict-hidden
   fixture is rejected. ``NO_MATCH`` fixtures verify the *absence* of any resolvable record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

GOLDEN_DIR = Path("evals/golden")

Subsystem = Literal["retrieval", "graph", "intent/translit", "guardrail", "generation/backtrack"]


class ExpectedVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    language: str
    year: int
    relationship: str | None = None  # is_original_of | is_remake_of | is_official_dub_of | …
    is_original: bool = False


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any]


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_match: bool = False
    canonical_work: str | None = None  # primary_title of the expected Work
    canonical_year: int | None = None
    versions: list[ExpectedVersion] = Field(default_factory=list)
    works: list[str] = Field(default_factory=list)  # multi-work expectations (GS-10)
    source_work: str | None = None  # literary source title (GS-05)
    ambiguous: bool | None = None


class GoldenFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^GS-\d{2}[a-z]$")
    name: str
    category: str
    subsystem: Subsystem
    query: str | list[str]  # ordered list for multi-turn
    query_lang: str
    expected: Expected
    gating_metric: str
    must_not: list[str] = Field(min_length=1)
    verify_source: list[str] = Field(min_length=1)  # QIDs / tmdb ids / page@revision
    expected_tool_calls: list[ExpectedToolCall] | None = None
    scope: str | None = None  # for scoping fixtures (GS-09)


class ValidationIssue(BaseModel):
    fixture_id: str
    issue: str


def load_fixtures(directory: Path = GOLDEN_DIR) -> list[GoldenFixture]:
    fixtures = []
    for path in sorted(directory.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for raw in payload["fixtures"]:
            fixtures.append(GoldenFixture.model_validate(raw))
    return fixtures


# --- Graph verification (the fixture gate) ---


def _version_row(session: Session, v: ExpectedVersion) -> Any | None:
    return session.execute(
        text(
            "SELECT gv.version_id, gv.work_id, gv.is_original, gv.confidence, gv.human_verified "
            "FROM ground_truth_versions gv "
            "WHERE gv.title = :t AND gv.language = :l AND gv.release_year = :y"
        ),
        {"t": v.title, "l": v.language, "y": v.year},
    ).first()


def _has_golden_edge(session: Session, src_id: Any, edge_type: str) -> bool:
    row = session.execute(
        text(
            "SELECT confidence, human_verified FROM ground_truth_edges "
            "WHERE src_id = :s AND edge_type = :t"
        ),
        {"s": src_id, "t": edge_type},
    ).first()
    if row is None:
        return False
    return row.confidence == "HIGH" or bool(row.human_verified)


def validate_fixture(session: Session, fixture: GoldenFixture) -> list[ValidationIssue]:
    """Verify a fixture against the live graph. Empty list = valid (freezable)."""
    issues: list[ValidationIssue] = []

    def issue(msg: str) -> None:
        issues.append(ValidationIssue(fixture_id=fixture.id, issue=msg))

    if fixture.expected.no_match:
        # NO_MATCH fixtures: nothing in the graph may be resolvable for the query terms.
        from sutradhar.graph.repository import resolve_title

        queries = fixture.query if isinstance(fixture.query, list) else [fixture.query]
        for q in queries:
            hits = resolve_title(session, q)
            if hits.candidates:
                issue(f"NO_MATCH fixture resolves to {hits.candidates[0].matched_title!r}")
        return issues

    for v in fixture.expected.versions:
        row = _version_row(session, v)
        if row is None:
            issue(f"expected version not gate-visible: {v.title} ({v.language}, {v.year})")
            continue
        if not (row.confidence == "HIGH" or row.human_verified):
            issue(f"version not golden-eligible (MEDIUM, unverified): {v.title} ({v.language})")
        if v.is_original and not row.is_original:
            issue(f"expected original flag missing on {v.title} ({v.language})")
        if v.relationship and v.relationship != "is_original_of":
            # is_sequel_of is a WORK-level relationship: the sequel work's original carries
            # the label in presentation; the stored edge hangs off the version's work.
            src_id = row.work_id if v.relationship == "is_sequel_of" else row.version_id
            if not _has_golden_edge(session, src_id, v.relationship):
                issue(f"no golden-eligible {v.relationship} edge from {v.title} ({v.language})")
        if v.relationship == "is_original_of" and not row.is_original:
            issue(f"{v.title}: labelled is_original_of but is_original is false")

    for work_title in fixture.expected.works:
        count = session.execute(
            text("SELECT count(*) FROM ground_truth_works WHERE primary_title = :t"),
            {"t": work_title},
        ).scalar_one()
        if count == 0:
            issue(f"expected work not gate-visible: {work_title}")

    if fixture.expected.source_work:
        count = session.execute(
            text(
                "SELECT count(*) FROM ground_truth_works "
                "WHERE primary_title = :t AND work_type = 'literary_source'"
            ),
            {"t": fixture.expected.source_work},
        ).scalar_one()
        if count == 0:
            issue(f"literary source not gate-visible: {fixture.expected.source_work}")

    return issues


def validate_all(
    session: Session, directory: Path = GOLDEN_DIR
) -> tuple[list[GoldenFixture], list[ValidationIssue]]:
    fixtures = load_fixtures(directory)
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for fixture in fixtures:
        if fixture.id in seen_ids:
            issues.append(ValidationIssue(fixture_id=fixture.id, issue="duplicate fixture id"))
        seen_ids.add(fixture.id)
        issues.extend(validate_fixture(session, fixture))
    covered = {f.id[:5] for f in fixtures}
    for gs in [f"GS-{i:02d}" for i in range(1, 12)]:
        if gs not in covered:
            issues.append(ValidationIssue(fixture_id=gs, issue="scenario category uncovered"))
    return fixtures, issues
