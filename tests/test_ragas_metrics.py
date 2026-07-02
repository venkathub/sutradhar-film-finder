"""RAGAS adapter tests (P3 task 8; P3_SPEC §4 — faked judge/embeddings backends).

The fakes subclass the REAL ragas base contracts (`InstructorBaseRagasLLM`,
`BaseRagasEmbedding`) and the tests drive the REAL collection metrics through them —
so ragas' faithfulness/relevancy pipelines execute end-to-end with zero network and a
version-pinned dependency (DEC-P3-3).
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevanceOutput
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)

from sutradhar.config import Settings
from sutradhar.evals.ragas_metrics import RagasScorer, build_scorer, ragas_version

T = TypeVar("T")


class FakeJudgeLLM(InstructorBaseRagasLLM):
    """Deterministic judge: dispatches on the response model ragas asks for."""

    def __init__(
        self,
        *,
        statements: list[str] | None = None,
        verdicts: list[int] | None = None,
        question: str = "what is the original of Papanasam?",
        noncommittal: bool = False,
        fail: bool = False,
    ) -> None:
        self.statements = statements if statements is not None else ["s1", "s2"]
        self.verdicts = verdicts if verdicts is not None else [1, 0]
        self.question = question
        self.noncommittal = noncommittal
        self.fail = fail
        self.calls: list[str] = []

    def generate(self, prompt: str, response_model: type[T]) -> T:
        if self.fail:
            raise RuntimeError("judge endpoint exploded")
        self.calls.append(response_model.__name__)
        if response_model is StatementGeneratorOutput:
            return StatementGeneratorOutput(statements=self.statements)  # type: ignore[return-value]
        if response_model is NLIStatementOutput:
            return NLIStatementOutput(  # type: ignore[return-value]
                statements=[
                    StatementFaithfulnessAnswer(statement=s, reason="r", verdict=v)
                    for s, v in zip(self.statements, self.verdicts, strict=True)
                ]
            )
        if response_model is AnswerRelevanceOutput:
            return AnswerRelevanceOutput(  # type: ignore[return-value]
                question=self.question, noncommittal=int(self.noncommittal)
            )
        raise AssertionError(f"unexpected response model {response_model.__name__}")

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self.generate(prompt, response_model)


class FakeEmbeddings(BaseRagasEmbedding):
    """Unit vectors: identical for question and generated questions -> cosine 1.0."""

    def __init__(self, question_vec: list[float] | None = None) -> None:
        self.question_vec = question_vec or [1.0, 0.0]

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return list(self.question_vec)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts)


def _scorer(llm: FakeJudgeLLM, embeddings: FakeEmbeddings | None = None) -> RagasScorer:
    return RagasScorer(llm, embeddings or FakeEmbeddings(), strictness=3)


# --- Real ragas pipelines over the fakes ---


def test_faithfulness_ratio_through_real_pipeline() -> None:
    scorer = _scorer(FakeJudgeLLM(statements=["a", "b"], verdicts=[1, 0]))
    scores = scorer.score("q", "answer", ["context"])
    assert scores.faithfulness == 0.5  # 1 of 2 statements supported
    assert scores.faithfulness_error is None


def test_faithfulness_all_supported_is_one() -> None:
    scores = _scorer(FakeJudgeLLM(verdicts=[1, 1])).score("q", "a", ["ctx"])
    assert scores.faithfulness == 1.0


def test_faithfulness_no_statements_records_nan_error() -> None:
    scores = _scorer(FakeJudgeLLM(statements=[], verdicts=[])).score("q", "a", ["ctx"])
    assert scores.faithfulness is None
    assert scores.faithfulness_error is not None and "NaN" in scores.faithfulness_error


def test_answer_relevancy_cosine_one_when_aligned() -> None:
    scores = _scorer(FakeJudgeLLM()).score("q", "a", ["ctx"])
    assert scores.answer_relevancy == pytest.approx(1.0)
    assert scores.answer_relevancy_error is None


def test_answer_relevancy_zero_when_all_noncommittal() -> None:
    scores = _scorer(FakeJudgeLLM(noncommittal=True)).score("q", "a", ["ctx"])
    assert scores.answer_relevancy == pytest.approx(0.0)


# --- Batch-safety: failures recorded, never raised ---


def test_metric_failures_recorded_not_raised() -> None:
    scores = _scorer(FakeJudgeLLM(fail=True)).score("q", "a", ["ctx"])
    assert scores.faithfulness is None and scores.answer_relevancy is None
    assert scores.faithfulness_error is not None
    assert "judge endpoint exploded" in scores.faithfulness_error
    assert scores.answer_relevancy_error is not None


def test_empty_contexts_fail_faithfulness_only() -> None:
    """ragas rejects empty retrieved_contexts; relevancy (no contexts needed) still scores."""
    scores = _scorer(FakeJudgeLLM()).score("q", "a", [])
    assert scores.faithfulness is None and scores.faithfulness_error is not None
    assert scores.answer_relevancy == pytest.approx(1.0)


def test_version_stamped_on_every_scores_object() -> None:
    scores = _scorer(FakeJudgeLLM()).score("q", "a", ["ctx"])
    assert scores.ragas_version == ragas_version() != ""


# --- Env wiring / off-path (DoD: skip cleanly with a clear message) ---


def test_build_scorer_off_when_judge_unset() -> None:
    scorer, reason = build_scorer(Settings(_env_file=None))
    assert scorer is None
    assert "JUDGE_BASE_URL" in reason


def test_build_scorer_off_when_embeddings_unset() -> None:
    settings = Settings(
        _env_file=None,
        JUDGE_BASE_URL="http://gpu:8000/v1",
        JUDGE_MODEL="openai/gpt-oss-20b",
    )
    scorer, reason = build_scorer(settings)
    assert scorer is None
    assert "EMBED_BASE_URL" in reason


def test_build_scorer_constructs_offline_when_fully_wired() -> None:
    """Construction is network-free; both endpoints come from env (never hardcoded)."""
    settings = Settings(
        _env_file=None,
        JUDGE_BASE_URL="http://gpu:8000/v1",
        JUDGE_MODEL="openai/gpt-oss-20b",
        JUDGE_API_KEY="k",
        EMBED_BASE_URL="http://gpu:8001/v1",
    )
    scorer, reason = build_scorer(settings)
    assert isinstance(scorer, RagasScorer)
    assert "ready" in reason
