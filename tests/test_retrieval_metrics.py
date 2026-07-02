"""Hermetic unit tests for the retrieval eval metrics (P2 task 10) — hand-computed, no
DB, no artifacts. These are the same functions Tier-1 CI uses to score the committed
run artifact, so their correctness IS the gate's correctness."""

from __future__ import annotations

from typing import Any

from sutradhar.evals.golden import GoldenFixture
from sutradhar.evals.retrieval import (
    RecordedVersion,
    RecordedWork,
    expected_rank,
    fixture_slice,
    needs_sequel_walk,
    pick_winner,
    recall_at_k,
    reciprocal_rank,
    version_set_recall,
)


def _fixture(**expected: Any) -> GoldenFixture:
    return GoldenFixture.model_validate(
        {
            "id": "GS-01a",
            "name": "t",
            "category": "flagship",
            "subsystem": "graph",
            "query": "q",
            "query_lang": "en",
            "expected": {"canonical_work": "Drishyam", "canonical_year": 2013, **expected},
            "gating_metric": "m",
            "must_not": ["x"],
            "verify_source": ["Q1"],
        }
    )


def _work(title: str, year: int | None = 2013, score: float = 0.5) -> RecordedWork:
    return RecordedWork(work_id="w", title=title, language="ml", year=year, score=score)


def _version(
    title: str,
    language: str = "ml",
    year: int | None = 2013,
    relationship: str | None = None,
    is_original: bool = False,
) -> RecordedVersion:
    return RecordedVersion(
        title=title, language=language, year=year, relationship=relationship,
        is_original=is_original,
    )


# --- rank / recall / MRR ---


def test_expected_rank_uses_title_and_year() -> None:
    fixture = _fixture()
    works = [_work("Drishyam 2", 2021), _work("Drishyam", 2013)]
    assert expected_rank(fixture, works) == 2
    # Same title, wrong year (the two-Vikram trap) does not match.
    assert expected_rank(fixture, [_work("Drishyam", 2015)]) is None


def test_recall_and_mrr_hand_computed() -> None:
    fixture = _fixture()
    works = [_work("Sholay"), _work("Magadheera"), _work("Drishyam", 2013)]
    assert recall_at_k(fixture, works, 1) == 0.0
    assert recall_at_k(fixture, works, 5) == 1.0
    assert reciprocal_rank(fixture, works) == 1 / 3
    assert reciprocal_rank(fixture, []) == 0.0
    assert recall_at_k(fixture, works, 2) == 0.0


def test_mrr_cutoff_at_10() -> None:
    fixture = _fixture()
    works = [_work(f"Other {i}", 2000 + i) for i in range(10)] + [_work("Drishyam", 2013)]
    assert reciprocal_rank(fixture, works) == 0.0  # rank 11 > cutoff


# --- version-set recall (the Papanasam metric) ---

EXPECTED_FAMILY = [
    {"title": "Drishyam", "language": "ml", "year": 2013, "is_original": True,
     "relationship": "is_original_of"},
    {"title": "Papanasam", "language": "ta", "year": 2015, "relationship": "is_remake_of"},
]


def test_version_set_recall_full_match() -> None:
    fixture = _fixture(versions=EXPECTED_FAMILY)
    got = [
        _version("Drishyam", "ml", 2013, "is_original_of", is_original=True),
        _version("Papanasam", "ta", 2015, "is_remake_of"),
        _version("Drishya", "kn", 2014, "is_remake_of"),  # extras never hurt recall
    ]
    assert version_set_recall(fixture.expected.versions, got) == 1.0


def test_version_set_recall_wrong_relationship_is_a_miss() -> None:
    """A dub labelled as remake (or vice versa) must NOT count — GS-04 semantics."""
    fixture = _fixture(versions=EXPECTED_FAMILY)
    got = [
        _version("Drishyam", "ml", 2013, "is_original_of", is_original=True),
        _version("Papanasam", "ta", 2015, "is_official_dub_of"),  # mislabelled
    ]
    assert version_set_recall(fixture.expected.versions, got) == 0.5


def test_version_set_recall_missing_original_flag_is_a_miss() -> None:
    fixture = _fixture(versions=EXPECTED_FAMILY)
    got = [
        _version("Drishyam", "ml", 2013, "is_original_of", is_original=False),  # flag lost
        _version("Papanasam", "ta", 2015, "is_remake_of"),
    ]
    assert version_set_recall(fixture.expected.versions, got) == 0.5


def test_version_set_recall_empty_cases() -> None:
    fixture = _fixture(versions=EXPECTED_FAMILY)
    assert version_set_recall(fixture.expected.versions, None) == 0.0
    assert version_set_recall(fixture.expected.versions, []) == 0.0
    assert version_set_recall([], []) == 1.0


# --- slice map / sequel detection / winner ---


def test_slice_map_covers_spec_slices() -> None:
    assert fixture_slice("GS-01a") == "flagship"
    assert fixture_slice("GS-03b") == "plot_only"
    assert fixture_slice("GS-06a") == "franchise"
    assert fixture_slice("GS-07a") == "code_mixed"
    assert fixture_slice("GS-11c") == "fuzzy_title"
    assert fixture_slice("GS-02a") == "negative"
    assert fixture_slice("GS-08a") is None  # backtracking is P5's problem
    assert fixture_slice("GS-09a") is None  # scoping: graph concern, no retrieval fixture


def test_needs_sequel_walk_detects_gs06_shape() -> None:
    with_sequel = _fixture(
        versions=[*EXPECTED_FAMILY,
                  {"title": "Drishyam 2", "language": "ml", "year": 2021,
                   "relationship": "is_sequel_of"}]
    )
    assert needs_sequel_walk(with_sequel) is True
    assert needs_sequel_walk(_fixture(versions=EXPECTED_FAMILY)) is False


def test_pick_winner_ordering() -> None:
    def m(r10: float, vsr: float, mrr: float) -> dict[str, Any]:
        return {
            "recall@10": r10,
            "version_set_recall_gs01": vsr,
            "version_set_recall_gs06": vsr,
            "mrr@10": mrr,
        }

    metrics = {
        "256tok_15pct/d50": m(0.9, 1.0, 0.8),
        "512tok_15pct/d50": m(1.0, 1.0, 0.7),
        "512tok_15pct/d20": m(1.0, 1.0, 0.9),  # same R@10+VSR, better MRR → wins
    }
    assert pick_winner(metrics) == "512tok_15pct/d20"
    # Full tie → deterministic key order.
    tied = {"b/d50": m(1, 1, 1), "a/d50": m(1, 1, 1)}
    assert pick_winner(tied) == "a/d50"
