"""Unit tests for the held-out negative set (P2 task 1, DEC-P2-5).

Schema/shape checks only — no DB, no model. Graph absence is enforced by
``tests/integration/test_negatives_absent.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sutradhar.evals.negatives import NEGATIVES_PATH, NegativeFixture, load_negatives

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def negatives() -> list[NegativeFixture]:
    return load_negatives(REPO_ROOT / NEGATIVES_PATH)


def test_loads_and_validates(negatives: list[NegativeFixture]) -> None:
    assert len(negatives) == 24  # spec §1.7: ~24 queries, enough for a slice-scale θ


def test_ids_unique_and_patterned(negatives: list[NegativeFixture]) -> None:
    ids = [f.id for f in negatives]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("NEG-") for i in ids)


def test_split_is_50_50(negatives: list[NegativeFixture]) -> None:
    """Calibration/test halves are equal — θ is tuned on one, reported on the other."""
    calibration = [f for f in negatives if f.split == "calibration"]
    test = [f for f in negatives if f.split == "test"]
    assert len(calibration) == len(test) == 12


def test_both_kinds_in_each_split(negatives: list[NegativeFixture]) -> None:
    """Each half sees plot-negatives AND title-negatives (no distribution skew)."""
    for split in ("calibration", "test"):
        kinds = [f.kind for f in negatives if f.split == split]
        assert kinds.count("plot") == 6, split
        assert kinds.count("title") == 6, split


def test_all_expect_no_match(negatives: list[NegativeFixture]) -> None:
    assert all(f.expected.no_match for f in negatives)


def test_code_mixed_and_native_scripts_present(negatives: list[NegativeFixture]) -> None:
    """The negative surface mirrors the golden query registers (GS-02/07 style)."""
    langs = {f.query_lang for f in negatives}
    assert "en" in langs
    assert langs & {"ta-latin", "hi-latin", "te-latin"}  # code-mixed romanized
    assert "hi" in langs  # native script


def test_no_match_false_rejected() -> None:
    with pytest.raises(ValidationError, match="no_match"):
        NegativeFixture.model_validate(
            {
                "id": "NEG-99",
                "name": "bad",
                "kind": "plot",
                "split": "test",
                "query": "q",
                "query_lang": "en",
                "expected": {"no_match": False},
                "must_not": ["x"],
                "verify_source": ["absent-from-seed-slice-by-construction"],
            }
        )


def test_golden_id_pattern_rejected() -> None:
    """Negatives must never masquerade as golden fixtures (contamination boundary)."""
    with pytest.raises(ValidationError, match="id"):
        NegativeFixture.model_validate(
            {
                "id": "GS-02a",
                "name": "bad",
                "kind": "title",
                "split": "test",
                "query": "q",
                "query_lang": "en",
                "expected": {"no_match": True},
                "must_not": ["x"],
                "verify_source": ["absent-from-seed-slice-by-construction"],
            }
        )
