"""Unit tests for the dub-vs-remake rule (P1 task 9) — pure function, no DB."""

from __future__ import annotations

import uuid

from sutradhar.pipeline.build import DUB_OVERLAP_THRESHOLD, classify_dub_vs_remake

A, B, C, D, E = (uuid.uuid4() for _ in range(5))


def test_same_lead_cast_is_dub() -> None:
    assert classify_dub_vs_remake({A, B}, {A, B}) == "is_official_dub_of"


def test_disjoint_cast_is_remake() -> None:
    """Mohanlal-vs-Kamal-Haasan case: different leads → different film → remake."""
    assert classify_dub_vs_remake({A, B}, {C, D}) == "is_remake_of"


def test_insufficient_evidence_is_none() -> None:
    assert classify_dub_vs_remake(set(), {A}) is None
    assert classify_dub_vs_remake({A}, set()) is None
    assert classify_dub_vs_remake(set(), set()) is None


def test_threshold_boundary() -> None:
    # overlap relative to the SMALLER set: {A} ∩ {A,B,C} = 1/1 = 1.0 → dub
    assert classify_dub_vs_remake({A}, {A, B, C}) == "is_official_dub_of"
    # {A,B} vs {B,C,D}: 1/2 = 0.5 → at threshold → dub
    assert DUB_OVERLAP_THRESHOLD == 0.5
    assert classify_dub_vs_remake({A, B}, {B, C, D}) == "is_official_dub_of"
    # {A,B,E} vs {B,C,D}: 1/3 < 0.5 → remake
    assert classify_dub_vs_remake({A, B, E}, {B, C, D}) == "is_remake_of"


def test_rule_is_symmetric_for_disjoint_and_identical() -> None:
    assert classify_dub_vs_remake({A}, {B}) == classify_dub_vs_remake({B}, {A})
    assert classify_dub_vs_remake({A}, {A}) == classify_dub_vs_remake({A}, {A})
