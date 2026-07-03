"""RAGAS adapter: faithfulness + answer relevancy through the pinned judge (P3 task 8).

DEC-P3-3 option A: the RAGAS library (version pinned in every artifact stamp — RAGAS
internal prompts evolve between releases) with **no external eval API**:

- LLM = the DEC-P3-1 self-hosted judge endpoint (``JUDGE_BASE_URL``/``JUDGE_MODEL``),
  via ``ragas.llms.llm_factory`` over an OpenAI-compatible client;
- embeddings = **BGE-M3 served in the same ephemeral GPU session** (``EMBED_BASE_URL``,
  DEC-0002's embedder reused for answer-relevancy question similarity).

These are the *supplementary* Table 2 signals (P3_SPEC §2.4): RAGAS faithfulness catches
prose-level unfaithfulness the deterministic detector's formatting contract cannot; the
**gating** faithfulness signal remains the no-hallucinated-movie detector. Scores are
computed as a batch pass over recorded transcripts inside the judge GPU session; between
windows CI gates on the recorded artifact (DEC-P2-6 posture).

Off/error posture (DoD): judge or embeddings unset → :func:`build_scorer` returns
``(None, reason)`` and callers skip cleanly; a per-sample metric failure is captured in
the returned :class:`RagasScores` (``*_error``), never raised into the batch.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

# ragas imports are module-level (the SDK is a runtime dep); model construction is lazy
# and network-free — nothing calls out until .score() runs inside a GPU session.
import ragas
from pydantic import BaseModel, ConfigDict
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from sutradhar.config import Settings


def ragas_version() -> str:
    """The pinned ragas version, recorded in the artifact stamp + judge block."""
    return str(ragas.__version__)


class RagasScores(BaseModel):
    """Supplementary RAGAS signals for one (question, answer, contexts) sample."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    faithfulness_error: str | None = None
    answer_relevancy_error: str | None = None
    ragas_version: str = str(ragas.__version__)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


class RagasScorer:
    """Thin orchestration over the two collection metrics (injectable for tests)."""

    def __init__(self, llm: Any, embeddings: Any, *, strictness: int = 3) -> None:
        self._faithfulness = Faithfulness(llm=llm)
        self._relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=strictness)

    def score(self, question: str, answer: str, contexts: list[str]) -> RagasScores:
        """Score one sample; per-metric failures are recorded, never raised (batch-safe)."""
        scores = RagasScores()
        try:
            result = _run_async(
                self._faithfulness.ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts
                )
            )
            value = float(result.value)
            if math.isnan(value):
                scores.faithfulness_error = "ragas_error: no statements generated (NaN)"
            else:
                scores.faithfulness = value
        except Exception as exc:  # noqa: BLE001 — batch pass must survive any sample
            scores.faithfulness_error = f"ragas_error: {type(exc).__name__}: {exc}"
        try:
            result = _run_async(self._relevancy.ascore(user_input=question, response=answer))
            scores.answer_relevancy = float(result.value)
        except Exception as exc:  # noqa: BLE001
            scores.answer_relevancy_error = f"ragas_error: {type(exc).__name__}: {exc}"
        return scores


def build_scorer(settings: Settings) -> tuple[RagasScorer | None, str]:
    """Construct the env-wired scorer, or ``(None, reason)`` when a backend is off.

    Judge LLM ← ``JUDGE_BASE_URL``/``JUDGE_MODEL``/``JUDGE_API_KEY``; embeddings ←
    ``EMBED_BASE_URL`` + ``EMBED_MODEL`` (BGE-M3 on :data:`EMBED_SERVE_PORT` in the same
    judge session). Construction is network-free; calls happen at :meth:`RagasScorer.score`.
    """
    if not (settings.judge_base_url and settings.judge_model):
        return None, "ragas off — set JUDGE_BASE_URL / JUDGE_MODEL (DEC-P3-1 judge session)"
    if not settings.embed_base_url:
        return None, "ragas off — set EMBED_BASE_URL (BGE-M3 serves in the judge session)"

    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    llm = llm_factory(
        settings.judge_model,
        provider="openai",
        client=AsyncOpenAI(
            base_url=settings.judge_base_url,
            api_key=settings.judge_api_key or "EMPTY",
        ),
    )
    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(base_url=settings.embed_base_url, api_key="EMPTY"),
        model=settings.embed_model,
    )
    return RagasScorer(llm, embeddings), "ragas ready (judge + BGE-M3 endpoints wired)"
