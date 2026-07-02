"""Integration: driver executes GS-08a end-to-end against the live graph (P3 task 6).

Scripted mock MODEL (reads real ids out of prior tool results — no LLM), REAL everything
else: live Postgres graph, real repository executor, the committed P2 retrieval artifact
for search_by_plot. Proves: all five v0 tools exercised; every result conforms to the
frozen result shapes; placeholder-bound tool-call scoring = 1.0; zero hallucinated movies.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from jsonschema import Draft202012Validator
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.config import Settings
from sutradhar.evals.driver import (
    RecordedPlotSearch,
    build_executor,
    load_retrieval_run,
    load_tool_schema,
    run_fixture,
)
from sutradhar.evals.generation import (
    collect_result_titles,
    detect_hallucinated_movies,
    score_tool_calls,
)
from sutradhar.evals.golden import load_fixtures
from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.serving import LLMClient

from .ci_review_pass import apply_ci_review_pass

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
SCHEMA = load_tool_schema(REPO_ROOT / "docs/phases/tool_schema.v0.json")


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except Exception as exc:  # noqa: BLE001 — connection probe
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    with engine.connect() as conn:
        outer = conn.begin()
        s = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
        for table in (
            "chunk_embeddings",
            "chunks",
            "candidate_edges",
            "edges",
            "conflicts",
            "plot_texts",
            "version_cast",
            "version_title",
            "version",
            "person",
            "work",
        ):
            s.execute(text(f"DELETE FROM {table}"))
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


@pytest.fixture()
def reviewed(session: Session) -> Session:
    """Full fixture chain + CI-mirrored review pass (same as test_golden_fixtures)."""
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
    for v in session.scalars(select(Version)).all():
        upsert_version_title(
            session,
            v.version_id,
            v.title,
            "canonical",
            v.language,
            [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
        )
    build_graph(session)
    art = json.loads((FIXTURES / "extraction" / "outputs_sample.json").read_text("utf-8"))
    load_candidates(session, art["raw_outputs"], art["pages"], art["model_id"])
    apply_ci_review_pass(session)
    return session


class _Gs08aModel:
    """Scripted GS-08a model: emits the expected sequence, binding REAL ids it reads from
    the tool results in the request messages — plus two benign schema-valid extras
    (search_by_plot, get_work) so all FIVE v0 tools are exercised end-to-end."""

    def __init__(self) -> None:
        self.step = 0

    # -- helpers over the request messages --

    @staticmethod
    def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]

    def _work_id(self, messages: list[dict[str, Any]]) -> str:
        for result in self._tool_results(messages):
            for candidate in result.get("candidates", []):
                return str(candidate["work_id"])
        raise AssertionError("no resolve_title result to bind from")

    def _version_ids(self, messages: list[dict[str, Any]]) -> list[str]:
        for result in self._tool_results(messages):
            if "versions" in result and "original" in result:  # get_versions shape
                ids = [result["original"]["version_id"]] if result["original"] else []
                ids += [v["version_id"] for v in result["versions"]]
                return [str(i) for i in ids]
        raise AssertionError("no get_versions result to bind from")

    @staticmethod
    def _refined_title(messages: list[dict[str, Any]]) -> str:
        results = [
            r
            for r in _Gs08aModel._tool_results(messages)
            if "versions" in r and "original" not in r  # refine_filter shape
        ]
        assert results and results[-1]["versions"], "refine returned nothing"
        v = results[-1]["versions"][-1]
        return f"**{v['title']} ({v['year']}, {v['language']})**"

    def _calls_for_step(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        step = self.step
        if step == 0:  # turn 1: resolve + benign extra plot search
            return [
                self._call("resolve_title", {"title": "Drishyam"}),
                self._call(
                    "search_by_plot",
                    {"description": "father protects family after a crime", "top_k": 10},
                ),
            ]
        if step == 1:
            return [
                self._call("get_versions", {"work_id": self._work_id(messages), "scope": "indian"}),
                self._call("get_work", {"work_id": self._work_id(messages)}),  # benign extra
            ]
        if step == 2:
            return [
                self._call(
                    "refine_filter",
                    {"version_set": self._version_ids(messages), "by": {"actor": "Ajay Devgn"}},
                )
            ]
        if step == 3:
            return (
                'INTENT: {"intent": "find_by_title", "slots": '
                '{"title": "Drishyam", "actor": "Ajay Devgn"}}\n\n'
                f"That's {self._refined_title(messages)}."
            )
        if step == 4:  # turn 2: "no, the original one"
            return [
                self._call(
                    "refine_filter",
                    {"version_set": self._version_ids(messages), "by": {"era": "original"}},
                )
            ]
        if step == 5:
            return (
                'INTENT: {"intent": "refine", "slots": {"era": "original"}}\n\n'
                f"The original is {self._refined_title(messages)}."
            )
        if step == 6:  # turn 3: "is there a Telugu one?"
            return [
                self._call(
                    "refine_filter",
                    {"version_set": self._version_ids(messages), "by": {"language": "te"}},
                )
            ]
        return (
            'INTENT: {"intent": "refine", "slots": {"language": "te"}}\n\n'
            f"Yes — {self._refined_title(messages)}."
        )

    @staticmethod
    def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        outcome = self._calls_for_step(body["messages"])
        self.step += 1
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "tool_calls"
        if isinstance(outcome, str):
            message["content"] = outcome
            finish = "stop"
        else:
            message["tool_calls"] = outcome
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{self.step}",
                "object": "chat.completion",
                "model": "scripted-gs08a",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )


def test_gs08a_end_to_end_all_five_tools(reviewed: Session) -> None:
    fixtures = {f.id: f for f in load_fixtures(GOLDEN_DIR)}
    gs08a = fixtures["GS-08a"]

    fixture_ref: dict[str, str] = {"fixture_id": gs08a.id}
    executor = build_executor(
        reviewed,
        RecordedPlotSearch(load_retrieval_run(REPO_ROOT / "evals/retrieval_runs")),
        fixture_ref,
    )
    client = LLMClient(
        Settings(_env_file=None, LLM_BASE_URL="http://localhost:8000/v1"),
        http_client=httpx.Client(transport=httpx.MockTransport(_Gs08aModel())),
    )
    transcript = run_fixture(
        client,
        gs08a,
        system_prompt="SYSTEM (frozen prompt not needed by the scripted model)",
        prompt_hash="integration",
        schema=SCHEMA,
        execute_tool=executor,
        fixture_id_ref=fixture_ref,
    )

    # Conversation completed: one prose answer per turn, endpoint stayed "up".
    assert transcript.chat_status == "up"
    assert len(transcript.answers) == 3 and all(transcript.answers)

    # All FIVE v0 tools were exercised and executed against the live graph.
    executed = {c.tool for c in transcript.calls if c.executed}
    assert executed == {
        "resolve_title",
        "search_by_plot",
        "get_work",
        "get_versions",
        "refine_filter",
    }

    # Every executed result conforms to the frozen v0 result shape.
    for call in transcript.calls:
        assert call.schema_valid, (call.tool, call.validation_errors)
        result_schema = dict(SCHEMA["tools"][call.tool]["result"])
        result_schema["$defs"] = SCHEMA["$defs"]
        errors = [e.message for e in Draft202012Validator(result_schema).iter_errors(call.result)]
        assert errors == [], (call.tool, errors)

    # DEC-P3-5 scoring over the transcript: expected sequence matched, extras tolerated.
    assert gs08a.expected_tool_calls is not None
    expected = [(c.tool, c.arguments) for c in gs08a.expected_tool_calls]
    score = score_tool_calls(expected, transcript.emitted_calls())
    assert score.sequence_match is True
    assert score.call_level == 1.0
    assert score.schema_validity == 1.0

    # Zero hallucinated movies: every asserted title grounds in the tool results.
    allowed = collect_result_titles(transcript.emitted_calls())
    for answer in transcript.answers:
        assert answer is not None
        report = detect_hallucinated_movies(answer, allowed)
        assert report.invention_count == 0, report.inventions
    # Per-turn answers name the per-turn expected versions (GS-08a's behaviour).
    assert "Drishyam (2015, hi)" in transcript.answers[0]
    assert "Drishyam (2013, ml)" in transcript.answers[1]
    assert "Drushyam (2014, te)" in transcript.answers[2]
