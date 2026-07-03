"""Generation-run artifact tests (P3 task 9; P3_SPEC §4 schema round-trip + invariants).

Covers: GenerationRunArtifact serialize/parse; the dry_run ⇒ serving/latency-null
invariant; slice selection; score_fixture wiring; aggregation math (incl. live latency
percentiles + tokens/sec); judge/RAGAS enrichment passes over recorded results.
"""

from __future__ import annotations

from typing import Any

import pytest

from sutradhar.evals.driver import EmittedCallRecord, FixtureTranscript
from sutradhar.evals.generation_run import (
    FixtureResult,
    GenerationRunArtifact,
    MetricsBlock,
    ReproStamp,
    aggregate_metrics,
    apply_judge_scores,
    apply_ragas_scores,
    build_stamp,
    generation_slice,
    golden_set_hash,
    make_run_id,
    score_fixture,
)
from sutradhar.evals.golden import GoldenFixture
from sutradhar.evals.judge import JudgeVerdict
from sutradhar.evals.ragas_metrics import RagasScores

# --- Helpers ---


def _fixture(
    fixture_id: str = "GS-07a",
    query: str | list[str] = "q",
    **extra: Any,
) -> GoldenFixture:
    return GoldenFixture(
        id=fixture_id,
        name="t",
        category="code-mixed",
        subsystem="intent/translit",
        query=query,
        query_lang="en",
        expected={"canonical_work": "Drishyam", "canonical_year": 2013},
        gating_metric="m",
        must_not=["x"],
        verify_source=["Q1"],
        **extra,
    )


def _call_record(
    tool: str = "resolve_title",
    result: dict[str, Any] | None = None,
    valid: bool = True,
) -> EmittedCallRecord:
    return EmittedCallRecord(
        turn=0,
        call_id="c1",
        tool=tool,
        arguments_raw='{"title": "Drishyam"}',
        arguments={"title": "Drishyam"},
        schema_valid=valid,
        executed=result is not None,
        result=result,
    )


_RESOLVE = {
    "candidates": [{"work_id": "wk1", "matched_title": "Drishyam"}],
    "ambiguous": False,
}


def _transcript(
    fixture_id: str = "GS-07a",
    answers: list[str | None] | None = None,
    calls: list[EmittedCallRecord] | None = None,
    users: list[str] | None = None,
    latencies: list[float] | None = None,
    usage: list[dict[str, int]] | None = None,
) -> FixtureTranscript:
    answers = answers if answers is not None else ["**Drishyam (2013)** original."]
    users = users or ["q"] * len(answers)
    return FixtureTranscript(
        fixture_id=fixture_id,
        prompt_hash="h",
        chat_status="up",
        messages=[{"role": "user", "content": u} for u in users],
        calls=calls if calls is not None else [_call_record(result=_RESOLVE)],
        answers=answers,
        latencies_ms=latencies or [],
        usage=usage or [],
    )


def _result(fixture_id: str = "GS-07a", **kwargs: Any) -> FixtureResult:
    fixture = _fixture(fixture_id)
    transcript = kwargs.pop("transcript", _transcript(fixture_id))
    result = score_fixture(fixture, transcript)
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


# --- Slice selection ---


def test_generation_slice_mapping() -> None:
    assert generation_slice("GS-07a") == "code_mixed"
    assert generation_slice("GS-08c") == "backtracking"
    assert generation_slice("GS-02d") == "negative"
    assert generation_slice("GS-02a") is None  # retrieval-shaped, excluded
    assert generation_slice("GS-01a") is None


# --- score_fixture wiring ---


def test_score_fixture_all_deterministic_scores() -> None:
    fixture = _fixture(
        expected_tool_calls=[{"tool": "resolve_title", "arguments": {"title": "Drishyam"}}],
        expected_intent="find_by_title",
        expected_slots={"title": "Drishyam"},
    )
    transcript = _transcript(
        answers=[
            'INTENT: {"intent": "find_by_title", "slots": {"title": "Drishyam"}}\n\n'
            "**Drishyam (2013)** is the original."
        ]
    )
    result = score_fixture(fixture, transcript)
    assert result.slice == "code_mixed"
    assert result.tool_calls is not None and result.tool_calls.sequence_match is True
    assert result.intent_matches == [True]
    assert result.slot_counts == {"tp": 1, "fp": 0, "fn": 0}
    assert len(result.hallucination) == 1
    assert result.hallucination[0].inventions == []  # Drishyam grounded in tool result


def test_score_fixture_catches_invention() -> None:
    fixture = _fixture("GS-02d")
    transcript = _transcript(
        "GS-02d",
        answers=["Sure! **Salaar (2023)** has versions in five languages."],
        calls=[_call_record(result={"candidates": [], "ambiguous": False})],
    )
    result = score_fixture(fixture, transcript)
    assert result.slice == "negative"
    assert result.hallucination[0].inventions == ["Salaar"]


# --- Aggregation ---


def test_aggregate_headlines_and_gate() -> None:
    good = _result("GS-07a")
    bad = _result(
        "GS-02d",
        transcript=_transcript(
            "GS-02d",
            answers=["**Salaar (2023)** exists!"],
            calls=[_call_record(result={"candidates": [], "ambiguous": False})],
        ),
    )
    metrics = aggregate_metrics([good, bad], "dry_run")
    assert metrics.fixtures_total == 2
    assert metrics.inventions == 1
    assert metrics.gs02_inventions == 1  # the gate counter sees the negative-slice invention
    assert metrics.faithfulness == pytest.approx(1.0 - 1 / 2)
    assert metrics.slices["negative"]["inventions"] == 1
    # dry_run: no latency/throughput ever.
    assert metrics.latency_p50_ms is None and metrics.tokens_per_sec is None


def test_aggregate_live_latency_and_throughput() -> None:
    result = _result(
        transcript=_transcript(
            latencies=[100.0, 200.0, 300.0, 400.0],
            usage=[{"completion_tokens": 50, "prompt_tokens": 1, "total_tokens": 51}] * 4,
        )
    )
    metrics = aggregate_metrics([result], "live")
    assert metrics.latency_p50_ms is not None and 200.0 <= metrics.latency_p50_ms <= 300.0
    assert metrics.latency_p95_ms is not None and metrics.latency_p95_ms >= 380.0
    assert metrics.tokens_per_sec == pytest.approx(200 / 1.0)  # 200 tokens / 1000 ms


def test_aggregate_code_mixed_slice_intent() -> None:
    gs07 = _result(
        transcript=_transcript(answers=['INTENT: {"intent": "find_by_plot", "slots": {}}\n\nok'])
    )
    gs07_fixture = _fixture(expected_intent="find_by_plot", expected_slots={})
    scored = score_fixture(gs07_fixture, gs07.transcript)
    metrics = aggregate_metrics([scored], "dry_run")
    assert metrics.code_mixed_intent_accuracy == 1.0
    assert metrics.intent_accuracy == 1.0


# --- Artifact invariants + round-trip ---


def _artifact(mode: str, metrics: MetricsBlock, serving: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "run_id": make_run_id(),
        "mode": mode,
        "model": "mock" if mode == "dry_run" else "google/gemma-4-E4B",
        "serving": serving,
        "prompt_hash": "p",
        "tool_schema_version": "v0",
        "judge": None,
        "retrieval_run": "r1",
        "fixtures": [_result().model_dump()],
        "metrics": metrics.model_dump(),
        "stamp": ReproStamp(
            created_at="2026-07-03T00:00:00+00:00",
            code_sha="abc",
            golden_set_hash="g",
            prompt_hash="p",
            tool_schema_version="v0",
            tool_schema_sha256="t",
            retrieval_run="r1",
            ragas_version="0.4.3",
        ).model_dump(),
    }


def test_artifact_round_trip() -> None:
    metrics = aggregate_metrics([_result()], "dry_run")
    payload = GenerationRunArtifact.model_validate(_artifact("dry_run", metrics, None))
    reparsed = GenerationRunArtifact.model_validate_json(payload.model_dump_json())
    assert reparsed == payload
    assert reparsed.fixtures[0].transcript.answers  # transcripts survive the round trip


def test_dry_run_invariant_rejects_serving_and_latency() -> None:
    metrics = aggregate_metrics([_result()], "dry_run")
    with pytest.raises(ValueError, match="serving=null"):
        GenerationRunArtifact.model_validate(_artifact("dry_run", metrics, {"gpu": "A100"}))
    live_metrics = metrics.model_copy(update={"latency_p50_ms": 12.0})
    with pytest.raises(ValueError, match="latency"):
        GenerationRunArtifact.model_validate(_artifact("dry_run", live_metrics, None))


def test_live_mode_allows_serving_and_latency() -> None:
    result = _result(
        transcript=_transcript(latencies=[100.0, 200.0], usage=[{"completion_tokens": 10}] * 2)
    )
    metrics = aggregate_metrics([result], "live")
    artifact = GenerationRunArtifact.model_validate(
        _artifact("live", metrics, {"gpu_type": "A100"})
    )
    assert artifact.serving == {"gpu_type": "A100"}


# --- Stamp ---


def test_build_stamp_is_complete_and_current() -> None:
    from sutradhar.evals.prompts import load_prompt_artifacts

    prompt_hash = load_prompt_artifacts().prompt_hash
    stamp = build_stamp(prompt_hash=prompt_hash, retrieval_run="r1")
    assert stamp.prompt_hash == prompt_hash
    assert stamp.golden_set_hash == golden_set_hash()
    assert stamp.tool_schema_version == "v0"
    assert len(stamp.tool_schema_sha256) == 64
    assert stamp.ragas_version
    assert stamp.created_at.startswith("20")


def test_run_id_format() -> None:
    run_id = make_run_id()
    ts, _, salt = run_id.partition("-")
    assert ts.endswith("Z") and len(salt) == 8


# --- Enrichment passes ---


class _FakeJudge:
    def __init__(self, score: float | None = 1.0, error: str | None = None) -> None:
        self.score = score
        self.error = error
        self.judged: list[list[dict[str, str]]] = []

    def judge_coherence(self, conversation: list[dict[str, str]]) -> JudgeVerdict:
        self.judged.append(conversation)
        return JudgeVerdict(score=self.score, error=self.error)


def test_apply_judge_scores_backtracking_only() -> None:
    gs08 = _result(
        "GS-08a",
        transcript=_transcript("GS-08a", answers=["a1", "a2"], users=["u1", "u2"]),
    )
    gs07 = _result("GS-07a")
    judge = _FakeJudge(score=0.67)
    judged = apply_judge_scores([gs08, gs07], judge)
    assert judged == 1
    assert gs08.judge_coherence == 0.67
    assert gs07.judge_coherence is None  # single-turn slices are not coherence-judged
    assert judge.judged[0][0] == {"user": "u1", "assistant": "a1"}
    metrics = aggregate_metrics([gs08, gs07], "dry_run")
    assert metrics.backtracking_coherence == 0.67


def test_apply_judge_error_recorded_not_raised() -> None:
    gs08 = _result(
        "GS-08a", transcript=_transcript("GS-08a", answers=["a1", "a2"], users=["u1", "u2"])
    )
    judged = apply_judge_scores([gs08], _FakeJudge(score=None, error="judge_error: boom"))
    assert judged == 0
    assert gs08.judge_error == "judge_error: boom"
    assert gs08.judge_coherence is None


class _FakeScorer:
    def score(self, question: str, answer: str, contexts: list[str]) -> RagasScores:
        assert contexts, "contexts must never be empty"
        return RagasScores(faithfulness=0.9, answer_relevancy=0.8)


def test_apply_ragas_scores_and_aggregate() -> None:
    result = _result()
    scored = apply_ragas_scores([result], _FakeScorer())  # type: ignore[arg-type]
    assert scored == 1
    assert result.ragas is not None and result.ragas.faithfulness == 0.9
    metrics = aggregate_metrics([result], "dry_run")
    assert metrics.ragas_faithfulness == 0.9
    assert metrics.answer_relevancy == 0.8
