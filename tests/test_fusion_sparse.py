"""Hermetic unit tests: RRF fusion + max-aggregation on hand-computed fixtures, and the
sparsevec literal builder's serialization edge cases (P2 task 8)."""

from __future__ import annotations

import pytest

from sutradhar.rag.fusion import RRF_K, aggregate_max, fuse, rrf_scores
from sutradhar.rag.sparse import sparse_literal

# --- RRF (hand-computed, k=60) ---


def test_rrf_hand_computed_two_channels() -> None:
    dense = ["a", "b", "c"]
    sparse = ["b", "a"]
    scores = rrf_scores([dense, sparse], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)


def test_fuse_orders_by_mass_then_key() -> None:
    # a and b end up with identical mass (see above) → deterministic key tie-break.
    fused = fuse([["a", "b", "c"], ["b", "a"]])
    assert [item for item, _ in fused] == ["a", "b", "c"]  # a before b: tie broken on key
    assert fused[0][1] == fused[1][1]


def test_single_channel_preserves_order() -> None:
    fused = fuse([["x", "y", "z"]])
    assert [i for i, _ in fused] == ["x", "y", "z"]
    assert [s for _, s in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_item_missing_from_one_channel_still_ranks() -> None:
    fused = dict(fuse([["a"], ["b"]]))
    assert fused["a"] == fused["b"] == pytest.approx(1 / 61)


def test_default_k_is_the_dec_p2_2_pin() -> None:
    assert RRF_K == 60


def test_rrf_rejects_duplicates_within_a_channel() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        rrf_scores([["a", "a"]])


def test_rrf_rejects_bad_k_and_handles_empty() -> None:
    with pytest.raises(ValueError, match="k must be"):
        rrf_scores([["a"]], k=0)
    assert fuse([]) == []
    assert fuse([[], []]) == []


# --- chunk→Work max aggregation ---


def test_aggregate_max_keeps_best_chunk_per_work() -> None:
    chunk_work = {"c1": "drishyam", "c2": "drishyam", "c3": "papanasam"}
    scored = [("c1", 0.4), ("c2", 0.9), ("c3", 0.7)]
    assert aggregate_max(scored, chunk_work.__getitem__) == [
        ("drishyam", 0.9),
        ("papanasam", 0.7),
    ]


def test_aggregate_max_tie_breaks_on_group_key() -> None:
    scored = [("c1", 0.5), ("c2", 0.5)]
    groups = {"c1": "b-work", "c2": "a-work"}
    assert aggregate_max(scored, groups.__getitem__) == [("a-work", 0.5), ("b-work", 0.5)]


def test_aggregate_max_empty() -> None:
    assert aggregate_max([], lambda item: item) == []


# --- sparsevec literal builder ---


def test_literal_shifts_to_one_based_and_orders() -> None:
    # BGE-M3 token ids are 0-based; the pgvector text literal is 1-based.
    assert sparse_literal({0: 0.5, 3: 1.0}, dims=10) == "{1:0.5,4:1.0}/10"


def test_literal_empty_weights() -> None:
    assert sparse_literal({}, dims=10) == "{}/10"


def test_literal_boundary_indices() -> None:
    lit = sparse_literal({250_001: 1.0})  # last valid XLM-R token id
    assert lit.startswith("{250002:") and lit.endswith("/250002")
    with pytest.raises(ValueError, match="out of range"):
        sparse_literal({250_002: 1.0})
    with pytest.raises(ValueError, match="out of range"):
        sparse_literal({-1: 1.0})


def test_literal_preserves_float_precision() -> None:
    lit = sparse_literal({7: 0.123456}, dims=100)
    assert "0.123456" in lit
