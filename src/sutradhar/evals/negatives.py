"""Held-out negative set for NO_MATCH abstention calibration (P2 task 1, DEC-P2-5).

These are deliberately **not** golden fixtures: they exist to tune the abstention
threshold θ, and tuning on the golden set would contaminate the GS-02 gate. They share
GS-02's field shape (so the same absence semantics apply) plus a ``split`` marker that
divides them 50/50 into a calibration half (θ is tuned here) and a test half (NO_MATCH
precision/recall is *reported* here, never tuned).

Absence is verified against the live graph exactly like the golden validator verifies
GS-02: :func:`sutradhar.graph.repository.resolve_title` must return zero candidates —
"absent from slice" is enforced by measurement, not asserted by authoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

NEGATIVES_PATH = Path("evals/negatives/heldout.yaml")

NegativeKind = Literal["plot", "title"]
Split = Literal["calibration", "test"]


class NegativeExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_match: bool

    @field_validator("no_match")
    @classmethod
    def _must_be_no_match(cls, v: bool) -> bool:
        if not v:
            raise ValueError("negative fixtures must expect no_match: true")
        return v


class NegativeFixture(BaseModel):
    """GS-02-schema-compatible negative query, plus ``kind`` and ``split``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^NEG-\d{2}$")
    name: str
    kind: NegativeKind
    split: Split
    query: str = Field(min_length=1)
    query_lang: str
    expected: NegativeExpected
    must_not: list[str] = Field(min_length=1)
    verify_source: list[str] = Field(min_length=1)


class NegativeValidationIssue(BaseModel):
    fixture_id: str
    issue: str


def load_negatives(path: Path = NEGATIVES_PATH) -> list[NegativeFixture]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [NegativeFixture.model_validate(raw) for raw in payload["fixtures"]]


def validate_negative(session: Session, fixture: NegativeFixture) -> list[NegativeValidationIssue]:
    """Verify absence-from-slice against the live graph (same check as golden GS-02)."""
    from sutradhar.graph.repository import resolve_title

    issues: list[NegativeValidationIssue] = []
    hits = resolve_title(session, fixture.query)
    if hits.candidates:
        top = hits.candidates[0]
        issues.append(
            NegativeValidationIssue(
                fixture_id=fixture.id,
                issue=(
                    f"negative resolves in the title channel: query {fixture.query!r} -> "
                    f"{top.matched_title!r} (score {top.score:.2f})"
                ),
            )
        )
    return issues


def validate_all_negatives(
    session: Session, path: Path = NEGATIVES_PATH
) -> tuple[list[NegativeFixture], list[NegativeValidationIssue]]:
    fixtures = load_negatives(path)
    issues: list[NegativeValidationIssue] = []
    seen: set[str] = set()
    for fixture in fixtures:
        if fixture.id in seen:
            issues.append(
                NegativeValidationIssue(fixture_id=fixture.id, issue="duplicate fixture id")
            )
        seen.add(fixture.id)
        issues.extend(validate_negative(session, fixture))
    return fixtures, issues
