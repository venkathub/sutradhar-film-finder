"""Playwright E2E app server (P6 task 7, P6_SPEC §4 "End-to-end").

Serves the REAL app (built UI + API + real guardrails + v1.1 prompt bundle) with a
deterministic, message-keyed scripted graph model — the pinned expected tool flows
from the golden fixtures, ids bound from real prior tool results. No LLM, no GPU.

Modes (E2E_MODE):
- ``up``  — live-mode server: seeds the graph into a HELD, never-committed transaction
            over the dev Postgres (the test_api_golden_regressions seeding chain) and
            reports the LLM endpoint as up. Rollback is implicit on process exit.
- ``off`` — degradation-mode server: no DB, status probe reports off; the offline
            payload + replay browser carry the experience (the zero-GPU story).

Launched by ``ui/app/playwright.config.ts`` (webServer) with cwd = repo root; also
runnable by hand: ``E2E_MODE=up E2E_PORT=8765 uv run python tests/e2e/e2e_server.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # test scaffolding: make `tests.*` importable

import uvicorn  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from sutradhar.config import Settings  # noqa: E402
from sutradhar.evals.driver import (  # noqa: E402
    RecordedPlotSearch,
    build_executor,
    load_retrieval_run,
)
from sutradhar.evals.prompts import load_serving_prompt_artifacts  # noqa: E402
from sutradhar.graph.db import postgres_url  # noqa: E402
from sutradhar.graph.models import SourceId, SourceRef  # noqa: E402
from sutradhar.graph.schema import Version  # noqa: E402
from sutradhar.pipeline.build import build_graph  # noqa: E402
from sutradhar.pipeline.extract import load_candidates  # noqa: E402
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice  # noqa: E402
from sutradhar.pipeline.titles import upsert_version_title  # noqa: E402
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie  # noqa: E402
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity  # noqa: E402
from sutradhar.serving.app import create_app  # noqa: E402
from sutradhar.serving.degrade import StatusCache  # noqa: E402
from sutradhar.serving.llm_client import ChatResult, EndpointStatus  # noqa: E402
from sutradhar.serving.sessions import InMemorySessionStore  # noqa: E402
from tests.integration.ci_review_pass import apply_ci_review_pass  # noqa: E402
from tests.integration.scripted_model import (  # noqa: E402
    ScriptedGraphModel,
    call,
    first_work_id,
    version_ids,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures"
UP = EndpointStatus(
    status="up", model="e2e-scripted", sample_token="x", latency_ms=9.0, detail="up"
)
OFF = EndpointStatus(status="off", model=None, sample_token=None, latency_ms=None, detail="paused")


# --- The pinned expected tool flows, keyed by the EXACT query each spec sends ---

Plan = Any  # (round, messages) -> list[tool-call dict] | str


def _answer(intent: str, slots: dict[str, Any], prose: str) -> str:
    return f"INTENT: {json.dumps({'intent': intent, 'slots': slots})}\n{prose}"


def gs01(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Drishyam", "language": "ml"})]
    if round_no == 1:
        return [call("get_versions", {"work_id": first_work_id(messages), "scope": "indian"})]
    return _answer(
        "list_versions",
        {"title": "Drishyam"},
        "Drishyam (2013, Malayalam) is the original; it was remade in Kannada, Telugu, "
        "Tamil and Hindi.",
    )


def gs06(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Drishyam", "language": "ml"})]
    if round_no == 1:
        return [
            call(
                "get_versions",
                {"work_id": first_work_id(messages), "scope": "indian", "include_sequels": True},
            )
        ]
    return _answer(
        "list_versions",
        {"title": "Drishyam"},
        "The full franchise: the Malayalam original, four remakes, and the sequel Drishyam 2.",
    )


def gs02(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Kaithi"})]
    # The pinned expected behaviour: honest abstention, never a fabricated title.
    return _answer(
        "out_of_catalog",
        {"title": "Kaithi"},
        "I checked the catalog but nothing resolves for that title. NO_MATCH.",
    )


def gs04(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Baahubali: The Beginning"})]
    if round_no == 1:
        return [call("get_versions", {"work_id": first_work_id(messages), "scope": "indian"})]
    return _answer(
        "list_versions",
        {"title": "Baahubali: The Beginning"},
        "Baahubali: The Beginning was released in Telugu and dubbed into other languages.",
    )


def gs05(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Devadasu", "language": "te"})]
    if round_no == 1:
        return [call("get_work", {"work_id": first_work_id(messages)})]
    if round_no == 2:
        return [call("get_versions", {"work_id": first_work_id(messages), "scope": "indian"})]
    return _answer(
        "find_by_title",
        {"title": "Devadasu"},
        "Devadasu (1953, Telugu) is one of several sibling adaptations of the novel "
        "Devdas — adaptations of a shared literary source, not remakes of each other.",
    )


def gs10(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Vikram"})]
    return _answer(
        "disambiguate",
        {"title": "Vikram"},
        "Two distinct films match: Vikram (1986) and Vikram (2022) — both Tamil, both "
        "with Kamal Haasan. Which one do you mean?",
    )


def gs08_turn1(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [call("resolve_title", {"title": "Drishyam"})]
    if round_no == 1:
        return [call("get_versions", {"work_id": first_work_id(messages), "scope": "indian"})]
    if round_no == 2:
        return [
            call(
                "refine_filter",
                {"version_set": version_ids(messages), "by": {"actor": "Ajay Devgn"}},
            )
        ]
    return _answer(
        "find_by_title",
        {"title": "Drishyam", "actor": "Ajay Devgn"},
        "That's Drishyam (2015, Hindi) with Ajay Devgn.",
    )


def gs08_turn2(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [
            call(
                "refine_filter",
                {"version_set": version_ids(messages), "by": {"era": "original"}},
            )
        ]
    return _answer(
        "refine",
        {"era": "original"},
        "The original is Drishyam (2013, Malayalam).",
    )


def gs08_turn3(round_no: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
    if round_no == 0:
        return [
            call(
                "refine_filter",
                {"version_set": version_ids(messages), "by": {"language": "te"}},
            )
        ]
    return _answer(
        "refine",
        {"language": "te"},
        "Yes — the Telugu version is Drushyam (2014).",
    )


SCENARIOS: dict[str, Plan] = {
    "show me every version of Drishyam": gs01,
    "the whole Drishyam franchise, sequels included": gs06,
    "Kaithi": gs02,
    "all language versions of Baahubali: The Beginning": gs04,
    "the Devadasu movie": gs05,
    "Vikram Kamal Haasan": gs10,
    "the Drishyam with Ajay Devgn": gs08_turn1,
    "no, the original one": gs08_turn2,
    "is there a Telugu one?": gs08_turn3,
}


class MessageKeyedModel:
    """Stateless across requests: the LAST user message picks the plan; the round
    within the current turn = assistant messages emitted since that user message.
    An unknown query fails loudly — the specs and the script must stay in lockstep."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        last_user_idx = max(i for i, m in enumerate(messages) if m.get("role") == "user")
        query = str(messages[last_user_idx]["content"])
        plan = SCENARIOS.get(query)
        if plan is None:
            raise AssertionError(f"e2e server: no scripted plan for query {query!r}")
        round_no = sum(1 for m in messages[last_user_idx:] if m.get("role") == "assistant")
        return ScriptedGraphModel(lambda _step, msgs: plan(round_no, msgs)).chat(messages, **kwargs)


def seed(session: Session) -> None:
    """The test_api_golden_regressions `reviewed` chain, verbatim (held transaction)."""
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


def main() -> None:
    mode = os.environ.get("E2E_MODE", "up")
    port = int(os.environ.get("E2E_PORT", "8765"))
    # CHAT_AUTH=disabled: the explicit local/e2e opt-out (DEC-P7-2) — the Playwright
    # UI has no token; production keeps auth required. Generous limit for UI runs.
    settings = Settings(_env_file=None, CHAT_AUTH="disabled", CHAT_RATE_LIMIT="1000/minute")

    if mode == "off":
        app = create_app(
            settings,
            status_cache=StatusCache(lambda: OFF),
            session_store=InMemorySessionStore(3600),
            session_factory=lambda: (_ for _ in ()).throw(  # off path never touches a DB
                AssertionError("off-mode e2e server must not open a DB session")
            ),
            prompt_artifacts=load_serving_prompt_artifacts(),
        )
    else:
        engine = create_engine(postgres_url())
        conn = engine.connect()
        conn.begin()  # HELD transaction — rolled back on process exit, never committed
        session = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
        from sqlalchemy import text as sqltext

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
            session.execute(sqltext(f"DELETE FROM {table}"))
        seed(session)
        session.close = lambda: None  # type: ignore[method-assign]  # app closes per request
        plot_search = RecordedPlotSearch(load_retrieval_run(REPO_ROOT / "evals" / "retrieval_runs"))
        executor = build_executor(session, plot_search, {"fixture_id": ""})
        app = create_app(
            settings,
            llm_client=MessageKeyedModel(),  # type: ignore[arg-type]  # duck-typed .chat
            status_cache=StatusCache(lambda: UP),
            session_store=InMemorySessionStore(3600),
            session_factory=lambda: session,
            make_executor=lambda _db: executor,
            prompt_artifacts=load_serving_prompt_artifacts(),
        )

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
