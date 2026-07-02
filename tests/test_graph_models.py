"""Hermetic unit tests for the domain + provenance types (P1 task 2, no DB).

Covers the P1_SPEC §4 unit rows: SourceRef/confidence validation (empty sources[], unknown
source ids rejected), edge shape rules mirrored in Python, gate-predicate helpers, and
enum↔schema drift.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sutradhar.graph import schema
from sutradhar.graph.models import (
    Confidence,
    EdgeRecord,
    EdgeType,
    NodeKind,
    RoleKind,
    SourceId,
    SourceRef,
    TitleKind,
    VersionRecord,
    WorkRecord,
    WorkType,
    golden_eligible,
    passes_gate_view,
    sources_to_jsonb,
)

SRC = SourceRef(source=SourceId.WIKIDATA, ref="Q1618487")
UID_A = uuid.uuid4()
UID_B = uuid.uuid4()


def _work(**overrides: object) -> WorkRecord:
    values: dict[str, object] = {
        "work_type": WorkType.FILM,
        "primary_title": "Drishyam",
        "original_language": "ml",
        "confidence": Confidence.HIGH,
        "sources": [SRC],
    }
    values.update(overrides)
    return WorkRecord.model_validate(values)


def _edge(**overrides: object) -> EdgeRecord:
    values: dict[str, object] = {
        "edge_type": EdgeType.IS_REMAKE_OF,
        "src_kind": NodeKind.VERSION,
        "src_id": UID_A,
        "dst_kind": NodeKind.VERSION,
        "dst_id": UID_B,
        "confidence": Confidence.HIGH,
        "sources": [SRC],
    }
    values.update(overrides)
    return EdgeRecord.model_validate(values)


# --- SourceRef ---


def test_source_ref_valid_roundtrip() -> None:
    ref = SourceRef(
        source=SourceId.WIKIPEDIA,
        ref="Drishyam@1234567",
        field="release_year",
        retrieved_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    blob = ref.to_jsonb()
    assert blob == {
        "source": "wikipedia",
        "ref": "Drishyam@1234567",
        "field": "release_year",
        "retrieved_at": "2026-07-02T00:00:00Z",
    }


def test_source_ref_rejects_unknown_source_id() -> None:
    with pytest.raises(ValidationError):
        SourceRef.model_validate({"source": "fandom_wiki", "ref": "x"})


def test_source_ref_rejects_empty_ref() -> None:
    with pytest.raises(ValidationError):
        SourceRef(source=SourceId.TMDB, ref="")


def test_sources_to_jsonb_drops_nulls() -> None:
    assert sources_to_jsonb([SRC]) == [{"source": "wikidata", "ref": "Q1618487"}]


# --- Gated records ---


def test_empty_sources_rejected_on_records_and_edges() -> None:
    with pytest.raises(ValidationError, match="sources"):
        _work(sources=[])
    with pytest.raises(ValidationError, match="sources"):
        _edge(sources=[])


def test_candidate_is_not_a_confidence_tier() -> None:
    with pytest.raises(ValidationError):
        _work(confidence="CANDIDATE")


def test_qid_and_imdb_id_patterns_enforced() -> None:
    with pytest.raises(ValidationError):
        _work(wikidata_qid="1618487")  # missing Q prefix
    with pytest.raises(ValidationError):
        VersionRecord(
            work_id=UID_A,
            title="Papanasam",
            language="ta",
            imdb_id="3417422",  # missing tt prefix
            confidence=Confidence.HIGH,
            sources=[SRC],
        )


def test_literary_source_carries_no_language() -> None:
    with pytest.raises(ValidationError, match="literary_source"):
        _work(work_type=WorkType.LITERARY_SOURCE, primary_title="Devdas (novella)")
    ok = _work(
        work_type=WorkType.LITERARY_SOURCE,
        primary_title="Devdas (novella)",
        original_language=None,
    )
    assert ok.work_type is WorkType.LITERARY_SOURCE


# --- Edge shape rules (Python mirror of ck_edges_type_shape / ck_edges_no_self_edge) ---


def test_edge_shape_remake_requires_version_to_version() -> None:
    with pytest.raises(ValidationError, match="edge shape violation"):
        _edge(dst_kind=NodeKind.WORK)


def test_edge_shape_sequel_requires_work_to_work() -> None:
    with pytest.raises(ValidationError, match="edge shape violation"):
        _edge(edge_type=EdgeType.IS_SEQUEL_OF)  # kinds default to version->version
    ok = _edge(
        edge_type=EdgeType.IS_SEQUEL_OF,
        src_kind=NodeKind.WORK,
        dst_kind=NodeKind.WORK,
    )
    assert ok.edge_type is EdgeType.IS_SEQUEL_OF


def test_self_edge_rejected() -> None:
    with pytest.raises(ValidationError, match="self-edge"):
        _edge(dst_id=UID_A)


def test_is_original_of_is_not_an_edge_type() -> None:
    with pytest.raises(ValidationError):
        _edge(edge_type="is_original_of")


# --- Gate predicate helpers (DEC-P1-7 layers) ---


def test_passes_gate_view_truth_table() -> None:
    assert passes_gate_view([SRC], has_open_conflict=False) is True
    assert passes_gate_view([SRC], has_open_conflict=True) is False
    assert passes_gate_view([], has_open_conflict=False) is False


def test_golden_eligible_truth_table() -> None:
    assert golden_eligible(Confidence.HIGH, human_verified=False) is True
    assert golden_eligible(Confidence.MEDIUM, human_verified=True) is True
    assert golden_eligible(Confidence.MEDIUM, human_verified=False) is False


# --- Enum ↔ schema drift guards (one source of truth per enum, tested not assumed) ---


@pytest.mark.parametrize(
    ("enum_cls", "schema_values"),
    [
        (Confidence, schema.CONFIDENCE_TIERS),
        (EdgeType, schema.EDGE_TYPES),
        (NodeKind, schema.NODE_KINDS),
        (WorkType, schema.WORK_TYPES),
        (TitleKind, schema.TITLE_KINDS),
        (RoleKind, schema.ROLE_KINDS),
    ],
)
def test_model_enums_match_schema_tuples(enum_cls: type, schema_values: tuple[str, ...]) -> None:
    assert {e.value for e in enum_cls} == set(schema_values)
