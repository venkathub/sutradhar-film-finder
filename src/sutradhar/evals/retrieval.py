"""Retrieval eval harness (P2 task 10): metrics, ablation grid, committed run artifact.

Two halves, deliberately separable:

- **Pure metric functions + artifact models** (no DB, no GPU): Recall@k, MRR@10,
  version-set recall, per-slice aggregation, winner selection. Tier-1 CI recomputes every
  gating metric from the committed artifact (``evals/retrieval_runs/<run_id>.json``,
  DEC-P2-6) with these same functions — one implementation, no drift.
- **Runner** (`run_retrieval_eval`, laptop + DB + recorded artifacts): executes the full
  §2.4 pipeline for every eval query across the ablation grid (chunk config ×
  rerank depth — fusion stays RRF per DEC-P2-2) and records observations only; expected
  values live in the golden fixtures, never duplicated into the artifact.

Expected-work matching is by ``(canonical_title, year)`` against the recorded result rows
(both come from ``ground_truth_works``), so CI needs no database to score a run.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sutradhar.evals.golden import ExpectedVersion, GoldenFixture, load_fixtures
from sutradhar.evals.negatives import NegativeFixture, load_negatives
from sutradhar.graph.repository import get_versions
from sutradhar.rag.artifacts import ArtifactEmbeddings, ArtifactReranker, ArtifactRun
from sutradhar.rag.chunking import CHUNK_CONFIGS, content_hash
from sutradhar.rag.retrieve import RetrievalConfig, Retriever

# Slice map (P2_SPEC §1.6). GS-02 gates abstention; GS-04/05/10 feed named regressions.
RETRIEVAL_SLICES = {
    "GS-01": "flagship",
    "GS-03": "plot_only",
    "GS-06": "franchise",
    "GS-07": "code_mixed",
    "GS-11": "fuzzy_title",
}
REGRESSION_SLICES = {"GS-02": "negative", "GS-04": "dub_regression",
                     "GS-05": "source_regression", "GS-10": "collision_regression"}
SKIPPED = ("GS-08", "GS-09")  # multi-turn backtracking (P5) / graph-scoping (no retrieval fixture)

RERANK_DEPTHS = (20, 50)  # DEC-P2-4 ablation
CHUNK_CONFIG_NAMES: tuple[str, ...] = tuple(config.name for config in CHUNK_CONFIGS)


def fixture_slice(fixture_id: str) -> str | None:
    prefix = fixture_id[:5]
    return RETRIEVAL_SLICES.get(prefix) or REGRESSION_SLICES.get(prefix)


# --- Artifact models (committed JSON, DEC-P2-6) ---


class RecordedWork(BaseModel):
    work_id: str
    title: str
    language: str | None
    year: int | None
    score: float


class RecordedVersion(BaseModel):
    title: str
    language: str
    year: int | None
    relationship: str | None
    is_original: bool


class QueryRecord(BaseModel):
    query_id: str
    slice: str
    query_sha256: str
    abstain: bool
    top_rerank_score: float | None  # the DEC-P2-5 abstention signal, raw
    channel_sizes: dict[str, int]
    works: list[RecordedWork]  # ranked, top 10
    version_set: list[RecordedVersion] | None = None  # get_versions(top-1) when it exists


class ConfigRecord(BaseModel):
    retrieval_config: dict[str, Any]  # RetrievalConfig asdict — the reproducibility stamp
    queries: dict[str, QueryRecord]
    negatives: dict[str, QueryRecord]


class EvalRunArtifact(BaseModel):
    run_id: str
    embed_model: str
    rerank_model: str
    code_sha: str | None
    golden_fixture_count: int
    negative_count: int
    records: dict[str, ConfigRecord]  # key: "<chunk_config>/d<depth>"
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    winner: str | None = None


# --- Pure metrics (shared by the runner and Tier-1 CI) ---


def expected_rank(fixture: GoldenFixture, works: list[RecordedWork]) -> int | None:
    """1-based rank of the fixture's canonical work in the recorded ranking."""
    for i, work in enumerate(works, start=1):
        if work.title == fixture.expected.canonical_work and (
            fixture.expected.canonical_year is None or work.year == fixture.expected.canonical_year
        ):
            return i
    return None


def recall_at_k(fixture: GoldenFixture, works: list[RecordedWork], k: int) -> float:
    rank = expected_rank(fixture, works)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(fixture: GoldenFixture, works: list[RecordedWork], cutoff: int = 10) -> float:
    rank = expected_rank(fixture, works)
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def _version_matches(expected: ExpectedVersion, got: RecordedVersion) -> bool:
    if (expected.title, expected.language, expected.year) != (got.title, got.language, got.year):
        return False
    if expected.relationship is not None and expected.relationship != got.relationship:
        return False
    return not (expected.is_original and not got.is_original)


def version_set_recall(
    expected: list[ExpectedVersion], got: list[RecordedVersion] | None
) -> float:
    """Fraction of expected versions present with correct labels (the Papanasam metric)."""
    if not expected:
        return 1.0
    if not got:
        return 0.0
    return sum(1 for e in expected if any(_version_matches(e, g) for g in got)) / len(expected)


def needs_sequel_walk(fixture: GoldenFixture) -> bool:
    return any(v.relationship == "is_sequel_of" for v in fixture.expected.versions)


def compute_metrics(
    fixtures: list[GoldenFixture], record: ConfigRecord
) -> dict[str, Any]:
    """Per-slice + overall metrics for one config cell, from recorded observations only."""
    by_slice: dict[str, list[GoldenFixture]] = {}
    for fixture in fixtures:
        slice_name = fixture_slice(fixture.id)
        if (
            slice_name is not None
            and slice_name in RETRIEVAL_SLICES.values()
            and fixture.id in record.queries
        ):
            by_slice.setdefault(slice_name, []).append(fixture)

    metrics: dict[str, Any] = {"slices": {}}
    gate_fixtures: list[tuple[GoldenFixture, QueryRecord]] = []
    for slice_name, slice_fixtures in sorted(by_slice.items()):
        rows = [(f, record.queries[f.id]) for f in slice_fixtures]
        gate_fixtures.extend(rows)
        metrics["slices"][slice_name] = {
            "n": len(rows),
            "recall@1": _mean([recall_at_k(f, q.works, 1) for f, q in rows]),
            "recall@5": _mean([recall_at_k(f, q.works, 5) for f, q in rows]),
            "recall@10": _mean([recall_at_k(f, q.works, 10) for f, q in rows]),
            "mrr@10": _mean([reciprocal_rank(f, q.works) for f, q in rows]),
            "version_set_recall": _mean(
                [
                    version_set_recall(f.expected.versions, q.version_set)
                    for f, q in rows
                    if f.expected.versions
                ]
            ),
        }
    metrics["recall@10"] = _mean([recall_at_k(f, q.works, 10) for f, q in gate_fixtures])
    metrics["mrr@10"] = _mean([reciprocal_rank(f, q.works) for f, q in gate_fixtures])
    for gs, key in (("GS-01", "version_set_recall_gs01"), ("GS-06", "version_set_recall_gs06")):
        rows = [
            (f, record.queries[f.id])
            for f in fixtures
            if f.id.startswith(gs) and f.id in record.queries and f.expected.versions
        ]
        metrics[key] = _mean(
            [version_set_recall(f.expected.versions, q.version_set) for f, q in rows]
        )
    return metrics


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def pick_winner(metrics: dict[str, dict[str, Any]]) -> str:
    """Deterministic: Recall@10 → version-set recall (GS-01+GS-06) → MRR@10 → key."""

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, str]:
        key, m = item
        return (
            -float(m["recall@10"]),
            -(float(m["version_set_recall_gs01"]) + float(m["version_set_recall_gs06"])),
            -float(m["mrr@10"]),
            key,
        )

    return sorted(metrics.items(), key=sort_key)[0][0]


# --- Runner (DB + recorded artifacts) ---


def _record_query(
    session: Session,
    retriever: Retriever,
    query_id: str,
    slice_name: str,
    query: str,
    *,
    version_set_for: GoldenFixture | None = None,
) -> QueryRecord:
    outcome = retriever.retrieve(query)
    version_set: list[RecordedVersion] | None = None
    if version_set_for is not None and outcome.works and version_set_for.expected.versions:
        versions = get_versions(
            session,
            outcome.works[0].work_id,
            include_sequels=needs_sequel_walk(version_set_for),
        )
        version_set = [
            RecordedVersion(
                title=v.title,
                language=v.language,
                year=v.year,
                relationship=v.relationship,
                is_original=v.is_original,
            )
            for v in versions.versions
        ]
    return QueryRecord(
        query_id=query_id,
        slice=slice_name,
        query_sha256=content_hash(query),
        abstain=outcome.abstain,
        top_rerank_score=outcome.reranked_chunks[0][1] if outcome.reranked_chunks else None,
        channel_sizes=outcome.channel_sizes,
        works=[
            RecordedWork(
                work_id=str(w.work_id),
                title=w.canonical_title,
                language=w.language,
                year=w.year,
                score=w.score,
            )
            for w in outcome.works[:10]
        ],
        version_set=version_set,
    )


def run_retrieval_eval(
    session: Session,
    artifacts_root: Path,
    run_id: str,
    *,
    golden_dir: Path | None = None,
    negatives_path: Path | None = None,
    chunk_configs: tuple[str, ...] = CHUNK_CONFIG_NAMES,
    rerank_depths: tuple[int, ...] = RERANK_DEPTHS,
) -> EvalRunArtifact:
    import json as _json

    run = ArtifactRun.open(artifacts_root, run_id)
    meta = _json.loads(run.path("meta.json").read_text(encoding="utf-8"))
    fixtures = load_fixtures(golden_dir) if golden_dir else load_fixtures()
    negatives: list[NegativeFixture] = (
        load_negatives(negatives_path) if negatives_path else load_negatives()
    )
    eval_fixtures = [
        f for f in fixtures if fixture_slice(f.id) is not None and not f.id.startswith(SKIPPED)
    ]

    artifact = EvalRunArtifact(
        run_id=run_id,
        embed_model=str(meta["embed_model"]),
        rerank_model=str(meta["rerank_model"]),
        code_sha=meta.get("code_sha"),
        golden_fixture_count=len(eval_fixtures),
        negative_count=len(negatives),
        records={},
    )

    for chunk_config in chunk_configs:
        embedder = ArtifactEmbeddings(run, banks=("queries", f"corpus_{chunk_config}"))
        reranker = ArtifactReranker(run, chunk_config)
        for depth in rerank_depths:
            config = RetrievalConfig(
                chunk_config=chunk_config,
                embed_model=str(meta["embed_model"]),
                index_version=run_id,
                rerank_depth=depth,
            )
            retriever = Retriever(session, config, embedder, reranker)
            record = ConfigRecord(retrieval_config=asdict(config), queries={}, negatives={})
            for fixture in eval_fixtures:
                query = fixture.query if isinstance(fixture.query, str) else fixture.query[0]
                slice_name = fixture_slice(fixture.id)
                assert slice_name is not None
                record.queries[fixture.id] = _record_query(
                    session, retriever, fixture.id, slice_name, query,
                    version_set_for=fixture,
                )
            for negative in negatives:
                record.negatives[negative.id] = _record_query(
                    session, retriever, negative.id, f"heldout_{negative.split}", negative.query
                )
            key = f"{chunk_config}/d{depth}"
            artifact.records[key] = record
            artifact.metrics[key] = compute_metrics(eval_fixtures, record)

    artifact.winner = pick_winner(artifact.metrics)
    return artifact
