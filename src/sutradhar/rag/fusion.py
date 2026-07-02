"""Rank fusion + chunk→Work aggregation (P2 task 8, DEC-P2-2/P2-4).

Pure math over channel rankings — no DB, no model:

- **RRF, k=60** (the cross-industry default; deliberately untuned): an item at 1-based
  rank ``r`` in a channel contributes ``1/(k+r)``; contributions sum across channels.
  Rank-based, so heterogeneous channel scores (cosine vs inner product vs fuzzy ratio)
  need no normalization.
- **Deterministic ordering everywhere**: ties break on the item key, so the same inputs
  produce the same fused list on every run/platform (the committed-artifact CI gate
  depends on this).
- **chunk→Work aggregation = max chunk score per work** (P2_SPEC §2.4): a Work is as
  relevant as its best chunk.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence

RRF_K = 60  # DEC-P2-2


def rrf_scores[K: Hashable](rankings: Sequence[Sequence[K]], k: int = RRF_K) -> dict[K, float]:
    """Reciprocal-rank-fusion mass per item across channels (1-based ranks)."""
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    scores: dict[K, float] = {}
    for ranking in rankings:
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                raise ValueError(f"duplicate item {item!r} within one channel ranking")
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def fuse[K: Hashable](rankings: Sequence[Sequence[K]], k: int = RRF_K) -> list[tuple[K, float]]:
    """Fused ranking, best first; ties broken deterministically on the item key."""
    scores = rrf_scores(rankings, k)
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def aggregate_max[K: Hashable](
    scored: Iterable[tuple[K, float]],
    group_of: Callable[[K], Hashable],
) -> list[tuple[Hashable, float]]:
    """Group items (chunks) and keep the max score per group (Work), best first."""
    best: dict[Hashable, float] = {}
    for item, score in scored:
        group = group_of(item)
        if group not in best or score > best[group]:
            best[group] = score
    return sorted(best.items(), key=lambda pair: (-pair[1], str(pair[0])))
