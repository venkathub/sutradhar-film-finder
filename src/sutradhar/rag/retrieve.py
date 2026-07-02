"""Hybrid retriever (P2 task 9): the §2.4 flow, laptop-side over stored artifacts.

```
query ─┬─ title channel: resolve_title (match_key + rapidfuzz θ=0.80, DEC-P1-5)
       │      → matched versions' METADATA-CARD chunks (fuzzy-score order)
       ├─ dense channel: recorded query vector → pgvector cosine top-N
       └─ sparse channel: recorded lexical weights → in-DB <#> top-N
             ↓ RRF k=60 (DEC-P2-2, chunk-level)
             ↓ cross-encoder rerank of the top `rerank_depth` (DEC-P2-4;
               sigmoid scores from the recorded full matrix)
             ↓ chunk→Work max aggregation
             ↓ abstain if top score < θ_no_match (DEC-P2-5; θ calibrated in task 11)
       Work results (search_by_plot v0 shape)
```

The title channel joins chunk-level fusion through each matched version's metadata card
(`kind='metadata_card'`) — one embedded unit per version that then flows through the same
rerank/aggregation path as any other chunk, so no separate work-level merge is needed.

Every knob is data (`RetrievalConfig`) → ablatable and stampable. No neural op runs here:
embeddings come from an ``EmbeddingProvider`` (recorded artifacts on laptop/CI), rerank
scores from a ``RerankProvider`` (recorded matrix). The live GPU path (P5) swaps
providers, not code.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass

from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from sutradhar.graph.repository import resolve_title
from sutradhar.rag.artifacts import EmbeddingProvider, RerankProvider
from sutradhar.rag.fusion import RRF_K, aggregate_max, fuse
from sutradhar.rag.sparse import ChannelHit, sparse_top_chunks

# DEC-P2-5 calibrated NO_MATCH threshold (P2 task 11, run 20260702T135315Z-f6583183):
# θ = 1.35 × top calibration-canary score (NEG-17 = 0.11241). Zero false accepts on
# GS-02 + the untouched test half (NO_MATCH recall 1.0); the zero-false-reject
# constraint measured INFEASIBLE (witness GS-07a = 0.00084 ≤ NEG-17) — four weak-scoring
# positives (GS-03a/c, GS-07a/b) return abstain=true WITH their (correct) results,
# degrading to "low confidence", never to a hallucinated match. Curve + report:
# evals/retrieval_runs/<run>.calibration.json; rationale in docs/DECISIONS.md.
CALIBRATED_NO_MATCH_THRESHOLD = 0.151747


@dataclass(frozen=True)
class RetrievalConfig:
    """Every retrieval knob, as data (P2_SPEC §2.5) — the reproducibility-stamp unit."""

    chunk_config: str
    embed_model: str
    index_version: str
    fusion: str = "rrf"
    rrf_k: int = RRF_K
    dense_top_n: int = 50
    sparse_top_n: int = 50
    rerank_depth: int = 50  # DEC-P2-4 default; ablated {20, 50}
    top_k: int = 10
    no_match_threshold: float = CALIBRATED_NO_MATCH_THRESHOLD  # DEC-P2-5 θ

    def stamp(self) -> str:
        """Deterministic JSON for BENCHMARKS/DECISIONS reproducibility stamps."""
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class WorkHit:
    work_id: uuid.UUID
    canonical_title: str
    language: str | None
    year: int | None
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    works: list[WorkHit]
    abstain: bool
    # Eval/trace surface (not part of the tool result): fused + reranked chunk detail.
    reranked_chunks: list[tuple[ChannelHit, float]]
    channel_sizes: dict[str, int]


def dense_top_chunks(
    session: Session,
    dense_vector: list[float],
    *,
    chunk_config: str,
    embed_model: str,
    index_version: str,
    top_n: int,
) -> list[ChannelHit]:
    """In-DB dense channel: pgvector cosine top-N (score = cosine similarity)."""
    rows = session.execute(
        sqltext(
            "SELECT c.chunk_id, c.work_id, c.content_hash, "
            "1 - (e.dense <=> :q) AS score "
            "FROM chunk_embeddings e JOIN chunks c ON c.chunk_id = e.chunk_id "
            "WHERE c.chunk_config = :cfg AND e.embed_model = :model "
            "AND e.index_version = :run "
            "ORDER BY e.dense <=> :q, c.content_hash LIMIT :n"
        ),
        {
            "q": json.dumps([float(x) for x in dense_vector]),
            "cfg": chunk_config,
            "model": embed_model,
            "run": index_version,
            "n": top_n,
        },
    ).all()
    return [
        ChannelHit(
            chunk_id=r.chunk_id,
            work_id=r.work_id,
            content_hash=r.content_hash,
            score=float(r.score),
        )
        for r in rows
    ]


def title_card_chunks(session: Session, query: str, *, chunk_config: str) -> list[ChannelHit]:
    """Title channel → the matched versions' metadata-card chunks, fuzzy-score order."""
    resolved = resolve_title(session, query)
    hits: list[ChannelHit] = []
    seen: set[uuid.UUID] = set()
    for candidate in resolved.candidates:
        if candidate.version_id is None or candidate.version_id in seen:
            continue
        seen.add(candidate.version_id)
        row = session.execute(
            sqltext(
                "SELECT chunk_id, work_id, content_hash FROM chunks "
                "WHERE version_id = :v AND kind = 'metadata_card' AND chunk_config = :cfg"
            ),
            {"v": str(candidate.version_id), "cfg": chunk_config},
        ).first()
        if row is not None:
            hits.append(
                ChannelHit(
                    chunk_id=row.chunk_id,
                    work_id=row.work_id,
                    content_hash=row.content_hash,
                    score=candidate.score,
                )
            )
    return hits


class Retriever:
    """The wired §2.4 pipeline. Construct once per (config, providers); retrieve many."""

    def __init__(
        self,
        session: Session,
        config: RetrievalConfig,
        embedder: EmbeddingProvider,
        reranker: RerankProvider,
    ) -> None:
        self.session = session
        self.config = config
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        embedded = self.embedder.embed([query])[0]

        dense = dense_top_chunks(
            self.session,
            [float(x) for x in embedded.dense],
            chunk_config=cfg.chunk_config,
            embed_model=cfg.embed_model,
            index_version=cfg.index_version,
            top_n=cfg.dense_top_n,
        )
        sparse = sparse_top_chunks(
            self.session,
            embedded.sparse,
            chunk_config=cfg.chunk_config,
            embed_model=cfg.embed_model,
            index_version=cfg.index_version,
            top_n=cfg.sparse_top_n,
        )
        title = title_card_chunks(self.session, query, chunk_config=cfg.chunk_config)

        by_id: dict[uuid.UUID, ChannelHit] = {
            h.chunk_id: h for channel in (dense, sparse, title) for h in channel
        }
        fused = fuse(
            [[h.chunk_id for h in channel] for channel in (dense, sparse, title) if channel],
            k=cfg.rrf_k,
        )
        candidates = [by_id[chunk_id] for chunk_id, _ in fused[: cfg.rerank_depth]]

        scores = self.reranker.score(query, [h.content_hash for h in candidates])
        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].content_hash),
        )

        work_scores = aggregate_max(
            [(hit.chunk_id, score) for hit, score in reranked],
            lambda chunk_id: by_id[chunk_id].work_id,
        )
        top_works = work_scores[: cfg.top_k]
        abstain = not top_works or top_works[0][1] < cfg.no_match_threshold

        works = self._work_hits([(w, s) for w, s in top_works])
        return RetrievalResult(
            works=works,
            abstain=abstain,
            reranked_chunks=reranked,
            channel_sizes={"dense": len(dense), "sparse": len(sparse), "title": len(title)},
        )

    def _work_hits(self, scored: list[tuple[object, float]]) -> list[WorkHit]:
        if not scored:
            return []
        ids = [str(work_id) for work_id, _ in scored]
        rows = self.session.execute(
            sqltext(
                "SELECT work_id, primary_title, original_language, first_release_year "
                "FROM ground_truth_works WHERE work_id = ANY(:ids)"
            ),
            {"ids": ids},
        ).all()
        by_id = {row.work_id: row for row in rows}
        hits: list[WorkHit] = []
        for work_id, score in scored:
            row = by_id.get(work_id)
            if row is None:  # not gate-visible (should be impossible via views) — skip
                continue
            hits.append(
                WorkHit(
                    work_id=row.work_id,
                    canonical_title=row.primary_title,
                    language=row.original_language,
                    year=row.first_release_year,
                    score=round(float(score), 6),
                )
            )
        return hits
