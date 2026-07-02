"""Generation-run artifact + orchestration (P3 task 9; P3_SPEC §2.2).

The committed, versioned evidence unit for Table 2 (mirrors the P2 retrieval-run pattern,
DEC-P2-6): ``evals/generation_runs/<run_id>.json`` embeds every fixture's full transcript,
its deterministic scores, and the aggregates — so Tier-1 CI can **recompute** the metrics
from the recorded transcripts with the same scorer bytes and gate without any model call.

Split of duties: this module = models + scoring orchestration + aggregation (typed,
unit-tested); ``evals/run_generation_eval.py`` = the thin Typer CLI; the judge/RAGAS
**enrichment passes** (:func:`apply_judge_scores`, :func:`apply_ragas_scores`) run inside
the ephemeral GPU session over the already-recorded transcripts and fill the supplementary
fields in place — deterministic metrics never depend on them.

Honesty invariants (enforced by validator, per P3_SPEC §4):
- ``mode="dry_run"`` ⇒ ``serving`` is null and latency/throughput metrics are null —
  a mock's timings are meaningless and must never look like GPU numbers.
- Table 2 publishes only ``mode="live"`` runs (the dry-run is machinery evidence).
"""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from sutradhar.evals.driver import FixtureTranscript, load_tool_schema, run_fixture
from sutradhar.evals.generation import (
    HallucinationReport,
    SlotCounts,
    ToolCallScore,
    collect_result_titles,
    detect_hallucinated_movies,
    micro_f1,
    score_intents,
    score_slots_per_turn,
    score_tool_calls,
)
from sutradhar.evals.golden import GoldenFixture
from sutradhar.evals.ragas_metrics import RagasScorer, RagasScores, ragas_version

GENERATION_RUNS_DIR = Path("evals/generation_runs")
TOOL_SCHEMA_VERSION = "v0"

_SLICES = {"GS-07": "code_mixed", "GS-08": "backtracking", "GS-02": "negative"}


def generation_slice(fixture_id: str) -> str | None:
    """The generation-slice name for a fixture id (None = not a generation fixture).
    GS-02a/b/c are retrieval-shaped and excluded; d+ are the conversational negatives."""
    prefix = fixture_id[:5]
    if prefix == "GS-02" and fixture_id[5:] < "d":
        return None
    return _SLICES.get(prefix)


def select_generation_fixtures(fixtures: list[GoldenFixture]) -> list[GoldenFixture]:
    return [f for f in fixtures if generation_slice(f.id) is not None]


# --- Per-fixture result ---


class ToolCallScoreRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_total: int
    call_matches: list[bool]
    call_level: float
    sequence_match: bool
    invalid_emitted: int
    emitted_total: int
    schema_validity: float

    @classmethod
    def from_score(cls, score: ToolCallScore) -> ToolCallScoreRecord:
        return cls(
            expected_total=score.expected_total,
            call_matches=list(score.call_matches),
            call_level=score.call_level,
            sequence_match=score.sequence_match,
            invalid_emitted=score.invalid_emitted,
            emitted_total=score.emitted_total,
            schema_validity=score.schema_validity,
        )


class TurnHallucination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    asserted: list[str]
    inventions: list[str]


class FixtureResult(BaseModel):
    """One fixture: full transcript + deterministic scores + (later) judge/RAGAS fields."""

    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    slice: str
    transcript: FixtureTranscript
    tool_calls: ToolCallScoreRecord | None = None
    intent_matches: list[bool] | None = None
    slot_counts: dict[str, int] | None = None  # {tp, fp, fn}
    hallucination: list[TurnHallucination] = []
    # Supplementary, filled by the GPU-session enrichment passes (null until then;
    # Tier-1 checks presence/shape only, never re-judges — P3_SPEC §2.7).
    judge_coherence: float | None = None
    judge_error: str | None = None
    ragas: RagasScores | None = None


def score_fixture(fixture: GoldenFixture, transcript: FixtureTranscript) -> FixtureResult:
    """All deterministic scores for one driven fixture (same bytes Tier-1 recomputes)."""
    slice_name = generation_slice(fixture.id) or "other"
    result = FixtureResult(fixture_id=fixture.id, slice=slice_name, transcript=transcript)
    if fixture.expected_tool_calls:
        expected = [(c.tool, c.arguments) for c in fixture.expected_tool_calls]
        result.tool_calls = ToolCallScoreRecord.from_score(
            score_tool_calls(expected, transcript.emitted_calls())
        )
    if fixture.expected_intent is not None:
        result.intent_matches = score_intents(fixture.expected_intent, transcript.answers)
    if fixture.expected_slots is not None:
        counts = score_slots_per_turn(fixture.expected_slots, transcript.answers)
        result.slot_counts = {"tp": counts.tp, "fp": counts.fp, "fn": counts.fn}
    allowed = collect_result_titles(transcript.emitted_calls())
    for turn, answer in enumerate(transcript.answers):
        if answer:
            report: HallucinationReport = detect_hallucinated_movies(answer, allowed)
            result.hallucination.append(
                TurnHallucination(
                    turn=turn,
                    asserted=list(report.asserted),
                    inventions=list(report.inventions),
                )
            )
    return result


# --- Aggregates (the Table 2 metrics block) ---


class MetricsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixtures_total: int
    fixtures_completed: int  # chat_status == "up" with every turn answered
    tool_call_sequence_accuracy: float | None  # Table 2 headline (DEC-P3-5)
    tool_call_call_level: float | None
    schema_validity: float | None
    intent_accuracy: float | None  # all labelled turns
    code_mixed_intent_accuracy: float | None  # the GS-07 slice (§2.4 headline)
    slot_micro_f1: float | None
    titles_asserted: int
    inventions: int
    hallucinated_movie_rate: float | None  # inventions / titles_asserted
    faithfulness: float | None  # 1 - rate (Table 2 headline)
    gs02_inventions: int  # HARD GATE: must be 0
    backtracking_coherence: float | None = None  # judge pass (GS-08)
    ragas_faithfulness: float | None = None
    answer_relevancy: float | None = None
    latency_p50_ms: float | None = None  # live runs only
    latency_p95_ms: float | None = None
    tokens_per_sec: float | None = None
    slices: dict[str, dict[str, Any]] = {}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_metrics(
    results: list[FixtureResult], mode: Literal["dry_run", "live"]
) -> MetricsBlock:
    tool_scored = [r for r in results if r.tool_calls is not None]
    intent_flags = [m for r in results if r.intent_matches for m in r.intent_matches]
    gs07_flags = [
        m for r in results if r.slice == "code_mixed" and r.intent_matches for m in r.intent_matches
    ]
    slot_total = SlotCounts()
    slot_seen = False
    for r in results:
        if r.slot_counts is not None:
            slot_seen = True
            slot_total = slot_total + SlotCounts(**r.slot_counts)
    asserted = sum(len(t.asserted) for r in results for t in r.hallucination)
    inventions = sum(len(t.inventions) for r in results for t in r.hallucination)
    rate = (inventions / asserted) if asserted else (None if not results else 0.0)
    gs02_inventions = sum(
        len(t.inventions) for r in results if r.slice == "negative" for t in r.hallucination
    )

    coherence = _mean([r.judge_coherence for r in results if r.judge_coherence is not None])
    ragas_f = _mean(
        [r.ragas.faithfulness for r in results if r.ragas and r.ragas.faithfulness is not None]
    )
    ragas_r = _mean(
        [
            r.ragas.answer_relevancy
            for r in results
            if r.ragas and r.ragas.answer_relevancy is not None
        ]
    )

    latency_p50 = latency_p95 = tokens_per_sec = None
    if mode == "live":
        latencies = [ms for r in results for ms in r.transcript.latencies_ms]
        if latencies:
            latency_p50 = round(statistics.quantiles(latencies, n=100)[49], 2)
            latency_p95 = round(statistics.quantiles(latencies, n=100)[94], 2)
        completion_tokens = sum(
            u.get("completion_tokens", 0) for r in results for u in r.transcript.usage
        )
        total_s = sum(ms for r in results for ms in r.transcript.latencies_ms) / 1000.0
        if completion_tokens and total_s > 0:
            tokens_per_sec = round(completion_tokens / total_s, 2)

    slices: dict[str, dict[str, Any]] = {}
    for slice_name in sorted({r.slice for r in results}):
        sub = [r for r in results if r.slice == slice_name]
        sub_tool = [r for r in sub if r.tool_calls is not None]
        sub_intents = [m for r in sub if r.intent_matches for m in r.intent_matches]
        slices[slice_name] = {
            "n": len(sub),
            "tool_call_sequence_accuracy": _mean(
                [1.0 if r.tool_calls and r.tool_calls.sequence_match else 0.0 for r in sub_tool]
            ),
            "intent_accuracy": _mean([1.0 if m else 0.0 for m in sub_intents]),
            "inventions": sum(len(t.inventions) for r in sub for t in r.hallucination),
        }

    return MetricsBlock(
        fixtures_total=len(results),
        fixtures_completed=sum(
            1
            for r in results
            if r.transcript.chat_status == "up" and all(a for a in r.transcript.answers)
        ),
        tool_call_sequence_accuracy=_mean(
            [1.0 if r.tool_calls and r.tool_calls.sequence_match else 0.0 for r in tool_scored]
        ),
        tool_call_call_level=_mean([r.tool_calls.call_level for r in tool_scored if r.tool_calls]),
        schema_validity=_mean([r.tool_calls.schema_validity for r in tool_scored if r.tool_calls]),
        intent_accuracy=_mean([1.0 if m else 0.0 for m in intent_flags]),
        code_mixed_intent_accuracy=_mean([1.0 if m else 0.0 for m in gs07_flags]),
        slot_micro_f1=micro_f1(slot_total) if slot_seen else None,
        titles_asserted=asserted,
        inventions=inventions,
        hallucinated_movie_rate=rate,
        faithfulness=(1.0 - rate) if rate is not None else None,
        gs02_inventions=gs02_inventions,
        backtracking_coherence=coherence,
        ragas_faithfulness=ragas_f,
        answer_relevancy=ragas_r,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        tokens_per_sec=tokens_per_sec,
        slices=slices,
    )


# --- Stamp + artifact ---


class ReproStamp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    code_sha: str | None
    golden_set_hash: str
    prompt_hash: str
    tool_schema_version: str
    tool_schema_sha256: str
    retrieval_run: str
    ragas_version: str


class GenerationRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: Literal["dry_run", "live"]
    model: str  # LLM_MODEL (+revision) or "mock"
    serving: dict[str, Any] | None  # vLLM version/flags, GPU type, decode params; null in dry_run
    prompt_hash: str
    tool_schema_version: str
    judge: dict[str, Any] | None  # JudgeConfig dump(s); null until the judge pass
    retrieval_run: str
    fixtures: list[FixtureResult]
    metrics: MetricsBlock
    stamp: ReproStamp

    @model_validator(mode="after")
    def _dry_run_invariants(self) -> GenerationRunArtifact:
        """§4 invariant: a mock's timings must never look like GPU numbers."""
        if self.mode == "dry_run":
            if self.serving is not None:
                raise ValueError("dry_run artifact must have serving=null")
            m = self.metrics
            if any(v is not None for v in (m.latency_p50_ms, m.latency_p95_ms, m.tokens_per_sec)):
                raise ValueError("dry_run artifact must have null latency/throughput metrics")
        return self


def code_sha() -> str | None:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        out = subprocess.run(  # noqa: S603, S607 — read-only local git query
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def golden_set_hash(directory: Path = Path("evals/golden")) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(tz=UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    salt = hashlib.sha256(stamp.encode() + os.urandom(8)).hexdigest()[:8]
    return f"{stamp}-{salt}"


def build_stamp(
    *,
    prompt_hash: str,
    retrieval_run: str,
    golden_dir: Path = Path("evals/golden"),
    schema_path: Path = Path("docs/phases/tool_schema.v0.json"),
) -> ReproStamp:
    return ReproStamp(
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        code_sha=code_sha(),
        golden_set_hash=golden_set_hash(golden_dir),
        prompt_hash=prompt_hash,
        tool_schema_version=TOOL_SCHEMA_VERSION,
        tool_schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        retrieval_run=retrieval_run,
        ragas_version=ragas_version(),
    )


# --- Orchestration (used by the CLI; unit-tested with fakes) ---


def execute_fixtures(
    fixtures: list[GoldenFixture],
    client: Any,  # LLMClient (duck-typed for test fakes)
    *,
    system_prompt: str,
    prompt_hash: str,
    execute_tool: Any,
    fixture_id_ref: dict[str, str],
    schema: dict[str, Any] | None = None,
    tracer: Any = None,  # sutradhar.obs.tracing.Tracer; None = disabled
    log: Any = print,
) -> list[FixtureResult]:
    schema = schema or load_tool_schema()
    results: list[FixtureResult] = []
    for fixture in fixtures:
        log(f"  driving {fixture.id} …")
        transcript = run_fixture(
            client,
            fixture,
            system_prompt=system_prompt,
            prompt_hash=prompt_hash,
            schema=schema,
            execute_tool=execute_tool,
            fixture_id_ref=fixture_id_ref,
            tracer=tracer,
        )
        results.append(score_fixture(fixture, transcript))
    return results


# --- GPU-session enrichment passes (judge + RAGAS over recorded transcripts) ---


def apply_judge_scores(results: list[FixtureResult], judge: Any) -> int:
    """Fill judge_coherence for multi-turn (backtracking) fixtures; returns judged count.
    ``judge`` = JudgeClient (duck-typed). Errors land in judge_error, never raised."""
    judged = 0
    for result in results:
        if result.slice != "backtracking":
            continue
        users = [m["content"] for m in result.transcript.messages if m.get("role") == "user"]
        turns = [
            {"user": str(u), "assistant": result.transcript.answers[i] or "(no answer)"}
            for i, u in enumerate(users)
            if i < len(result.transcript.answers)
        ]
        verdict = judge.judge_coherence(turns)
        if verdict.error is not None:
            result.judge_error = verdict.error
        else:
            result.judge_coherence = verdict.score
            judged += 1
    return judged


def apply_ragas_scores(results: list[FixtureResult], scorer: RagasScorer) -> int:
    """Fill RAGAS supplementary scores per fixture final answer; returns scored count."""
    scored = 0
    for result in results:
        answers = [a for a in result.transcript.answers if a]
        if not answers:
            continue
        users = [m["content"] for m in result.transcript.messages if m.get("role") == "user"]
        question = " / ".join(str(u) for u in users)
        contexts = [
            c.model_dump_json(include={"tool", "result"})
            for c in result.transcript.calls
            if c.result is not None
        ] or ["(no tool results — out-of-catalog query)"]
        result.ragas = scorer.score(question, answers[-1], contexts)
        scored += 1
    return scored
