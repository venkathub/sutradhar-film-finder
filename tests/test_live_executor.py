"""P5 task 7 — live tool executor (no DB: search_by_plot only touches the retriever;
uuid/provider failures are raised before any session use)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from sutradhar.evals.driver import ToolExecutionError
from sutradhar.rag.providers import ProviderUnavailableError
from sutradhar.rag.retrieve import RetrievalResult, WorkHit
from sutradhar.serving.executor import build_live_executor

WORK_ID = uuid.uuid4()


class _StubRetriever:
    def __init__(self, outcome: RetrievalResult | Exception) -> None:
        self._outcome = outcome
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _result(abstain: bool = False) -> RetrievalResult:
    return RetrievalResult(
        works=[
            WorkHit(
                work_id=WORK_ID,
                canonical_title="Drishyam",
                language="ml",
                year=2013,
                score=0.93,
            )
        ],
        abstain=abstain,
        reranked_chunks=[],
        channel_sizes={"dense": 10, "sparse": 10, "title": 1},
    )


def test_search_by_plot_maps_to_v0_shape() -> None:
    retriever = _StubRetriever(_result())
    execute = build_live_executor(session=None, retriever=retriever)  # type: ignore[arg-type]
    payload = execute("search_by_plot", {"description": "a man buries a body", "top_k": 5})
    assert retriever.queries == ["a man buries a body"]
    assert payload == {
        "results": [
            {
                "work_id": str(WORK_ID),
                "canonical_title": "Drishyam",
                "language": "ml",
                "year": 2013,
                "score": 0.93,
            }
        ],
        "abstain": False,
    }


def test_search_by_plot_abstain_passthrough() -> None:
    execute = build_live_executor(session=None, retriever=_StubRetriever(_result(abstain=True)))  # type: ignore[arg-type]
    payload = execute("search_by_plot", {"description": "weak match"})
    assert payload["abstain"] is True
    assert payload["results"]  # v0 allows results WITH abstain => "low confidence"


def test_provider_off_becomes_tool_error_feedback() -> None:
    """Sidecar off ≠ turn dead: fed back so the model can use the graph tools."""
    off = ProviderUnavailableError("embeddings", "off", "endpoint OFF")
    execute = build_live_executor(session=None, retriever=_StubRetriever(off))  # type: ignore[arg-type]
    with pytest.raises(ToolExecutionError) as exc:
        execute("search_by_plot", {"description": "anything"})
    assert "plot search unavailable" in str(exc.value)
    assert "title/graph tools still work" in str(exc.value)


def test_bad_uuid_is_tool_error() -> None:
    execute = build_live_executor(session=None, retriever=_StubRetriever(_result()))  # type: ignore[arg-type]
    for tool, args in (
        ("get_work", {"work_id": "not-a-uuid"}),
        ("get_versions", {"work_id": "nope"}),
        ("refine_filter", {"version_set": ["bad"], "by": {"era": "original"}}),
    ):
        with pytest.raises(ToolExecutionError) as exc:
            execute(tool, dict[str, Any](args))
        assert "is not a known id" in str(exc.value)
