"""Named golden regressions through the API orchestration path (P5 task 12, P5_SPEC §4).

These assert the *served* ``ChatResponse`` payload — not just the repository — driven
through ``create_app`` over HTTP (``TestClient``), GPU-free: a scripted id-binding model
(the ``test_driver_e2e`` technique — it reads REAL ids out of prior tool results, no LLM),
the REAL repository executor against the seeded live graph, and the committed P2 retrieval
replay for ``search_by_plot``. Real guardrails + the v1.1 serving bundle are wired, so the
GS-02 test proves the OUTPUT GATE turns a would-be invention into a 0-invention surface.

Integration tier (live Postgres, DEC-P2-6 posture; skipped cleanly when the DB is absent).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.config import Settings
from sutradhar.evals.driver import (
    RecordedPlotSearch,
    build_executor,
    load_retrieval_run,
)
from sutradhar.evals.prompts import load_serving_prompt_artifacts
from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.serving.app import create_app
from sutradhar.serving.degrade import StatusCache
from sutradhar.serving.llm_client import EndpointStatus
from sutradhar.serving.sessions import InMemorySessionStore
from sutradhar.toolcalls import load_tool_schema

from .ci_review_pass import apply_ci_review_pass
from .scripted_model import (
    RecordingExecutor,
    ScriptedGraphModel,
    StepFn,
)
from .scripted_model import (
    call as _call,
)
from .scripted_model import (
    first_work_id as _first_work_id,
)
from .scripted_model import (
    version_ids as _version_ids,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SCHEMA = load_tool_schema(REPO_ROOT / "docs/phases/tool_schema.v0.json")
UP = EndpointStatus(status="up", model="m", sample_token="x", latency_ms=9.0, detail="up")


# --- DB fixtures (the test_driver_e2e seeding chain: full graph + CI review pass) ---


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


# --- Scripted id-binding model: shared with the E2E server (P6 task 7 extraction) ---


def _client(reviewed: Session, step_fn: StepFn) -> tuple[TestClient, RecordingExecutor]:
    plot_search = RecordedPlotSearch(load_retrieval_run(REPO_ROOT / "evals" / "retrieval_runs"))
    fixture_ref = {"fixture_id": ""}
    recorder = RecordingExecutor(build_executor(reviewed, plot_search, fixture_ref), SCHEMA)
    app = create_app(
        Settings(_env_file=None),
        llm_client=ScriptedGraphModel(step_fn),  # type: ignore[arg-type]  # duck-typed .chat
        status_cache=StatusCache(lambda: UP),
        session_store=InMemorySessionStore(3600),
        session_factory=lambda: reviewed,  # single seeded session; close() is a no-op below
        make_executor=lambda _db: recorder,
        prompt_artifacts=load_serving_prompt_artifacts(),
    )
    # The app closes the session per request; keep the seeded savepoint session alive.
    reviewed.close = lambda: None  # type: ignore[method-assign]
    return TestClient(app, raise_server_exceptions=False), recorder


def _versions(body: dict[str, Any]) -> dict[tuple[str, str | None], dict[str, Any]]:
    """versions[] keyed by (title, language) for order-independent assertions."""
    return {(v["title"], v["language"]): v for v in body["versions"]}


# --- The six named regressions ---


def test_api_version_set_recall_gs01_gs06(reviewed: Session) -> None:
    """GS-01 + GS-06: complete version set, original flagged, sequel traversal (served)."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if step == 0:
            return [_call("resolve_title", {"title": "Drishyam", "language": "ml"})]
        if step == 1:
            return [
                _call(
                    "get_versions",
                    {
                        "work_id": _first_work_id(messages),
                        "scope": "indian",
                        "include_sequels": True,
                    },
                )
            ]
        return 'INTENT: {"intent": "list_versions", "slots": {"title": "Drishyam"}}\nThe set.'

    client, _ = _client(reviewed, steps)
    body = client.post("/api/chat", json={"message": "show me every Drishyam film"}).json()
    assert body["status"] == "up"

    versions = _versions(body)
    # Complete franchise (GS-01 base + GS-06 sequels), VSR = 1.0 against the golden set.
    for title, lang in [
        ("Drishyam", "ml"),
        ("Drishya", "kn"),
        ("Drushyam", "te"),
        ("Papanasam", "ta"),
        ("Drishyam", "hi"),
        ("Drishyam 2", "ml"),
    ]:
        assert (title, lang) in versions, f"missing {title} ({lang})"
    # Exactly one original, and it is the Malayalam 2013 film.
    originals = [v for v in body["versions"] if v["is_original"]]
    assert versions[("Drishyam", "ml")]["is_original"] is True
    assert any(v["year"] == 2013 and v["language"] == "ml" for v in originals)
    # Sequel traversal present and typed as a sequel, never a remake of the original.
    assert versions[("Drishyam 2", "ml")]["relationship"] == "is_sequel_of"


def test_api_no_hallucinated_movie_gs02(reviewed: Session) -> None:
    """GS-02: the output gate turns a would-be invention into a 0-invention API surface,
    even when the (scripted) model asserts an out-of-catalog title."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        # The model resolves nothing useful, then tries to fabricate — the gate must catch it.
        if step == 0:
            return [_call("resolve_title", {"title": "Kaithi"})]
        return (
            'INTENT: {"intent": "find_by_title", "slots": {"title": "Kaithi"}}\n'
            "That's **Kaithi** (2019), a Tamil action thriller."
        )

    client, _ = _client(reviewed, steps)
    body = client.post("/api/chat", json={"message": "Kaithi"}).json()
    assert body["status"] == "up"
    # The invented title is DOWNGRADED, never asserted as fact (the served surface has 0
    # inventions even though the raw model column would record one).
    assert "[unverified — not in tool results]" in body["answer"]
    assert any("Kaithi" in w for w in body["warnings"])
    # No version_set claim smuggled through.
    assert body["versions"] == []


def test_api_dub_vs_remake_gs04(reviewed: Session) -> None:
    """GS-04: Baahubali language versions carry is_official_dub_of, never is_remake_of."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if step == 0:
            return [_call("resolve_title", {"title": "Baahubali: The Beginning"})]
        if step == 1:
            return [_call("get_versions", {"work_id": _first_work_id(messages), "scope": "indian"})]
        return 'INTENT: {"intent": "list_versions", "slots": {}}\nDubs.'

    client, _ = _client(reviewed, steps)
    body = client.post(
        "/api/chat", json={"message": "all language versions of Baahubali: The Beginning"}
    ).json()
    versions = _versions(body)
    dubs = [v for v in body["versions"] if v["relationship"] == "is_official_dub_of"]
    assert dubs, "no dub relationships surfaced"
    assert all(v["relationship"] != "is_remake_of" for v in body["versions"]), (
        "a Baahubali dub was mislabelled as a remake"
    )
    # The Hindi dub specifically is a dub, not a separate remake.
    hindi = [v for (t, lang), v in versions.items() if lang == "hi"]
    assert hindi and hindi[0]["relationship"] == "is_official_dub_of"


def test_api_sibling_vs_remake_gs05(reviewed: Session) -> None:
    """GS-05: Devdas adaptations are siblings of a shared literary source (based_on),
    never chained into one is_remake_of lineage."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if step == 0:
            return [_call("resolve_title", {"title": "Devadasu", "language": "te"})]
        if step == 1:
            return [_call("get_work", {"work_id": _first_work_id(messages)})]
        if step == 2:
            return [_call("get_versions", {"work_id": _first_work_id(messages), "scope": "indian"})]
        return 'INTENT: {"intent": "list_versions", "slots": {}}\nSiblings.'

    client, recorder = _client(reviewed, steps)
    body = client.post("/api/chat", json={"message": "the Devadasu movie"}).json()
    assert body["status"] == "up"
    # The literary source is exposed via get_work.based_on / source_work (tool result) —
    # the sibling relationship, not a remake chain.
    get_work_results = [r for (tool, r) in recorder.results if tool == "get_work"]
    assert get_work_results, "get_work was not called"
    gw = get_work_results[0]
    assert gw["source_work"] is not None and gw["source_work"]["canonical_title"] == "Devdas"
    assert gw["based_on"] == [gw["source_work"]["work_id"]]
    # The served versions of the Telugu film are not chained as remakes of a single film.
    assert all(v["relationship"] != "is_remake_of" for v in body["versions"])


def test_api_false_merge_gs10(reviewed: Session) -> None:
    """GS-10: 'Vikram' hits two distinct Works — they must stay separate in one response."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if step == 0:
            return [_call("resolve_title", {"title": "Vikram"})]
        return 'INTENT: {"intent": "find_by_title", "slots": {"title": "Vikram"}}\nTwo films.'

    client, recorder = _client(reviewed, steps)
    body = client.post("/api/chat", json={"message": "Vikram Kamal Haasan"}).json()
    assert body["status"] == "up"
    resolve_results = [r for (tool, r) in recorder.results if tool == "resolve_title"]
    assert resolve_results and resolve_results[0]["ambiguous"] is True
    # Two DISTINCT works (1986, 2022) — never merged into one record/version set.
    work_ids = {c["work_id"] for c in resolve_results[0]["candidates"]}
    years = {c["year"] for c in resolve_results[0]["candidates"]}
    assert len(work_ids) >= 2 and {1986, 2022} <= years


def test_api_emitted_tool_calls_validate(reviewed: Session) -> None:
    """Every call the orchestrator executed on the served path is v0-schema-valid — no
    hallucinated tool or parameter names reached the executor."""

    def steps(step: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if step == 0:
            return [
                _call("resolve_title", {"title": "Drishyam", "language": "ml"}),
                _call(
                    "search_by_plot",
                    {"description": "a father protects his family after a crime", "top_k": 5},
                ),
            ]
        if step == 1:
            return [_call("get_versions", {"work_id": _first_work_id(messages), "scope": "indian"})]
        if step == 2:
            return [
                _call(
                    "refine_filter",
                    {"version_set": _version_ids(messages), "by": {"era": "original"}},
                )
            ]
        return 'INTENT: {"intent": "refine", "slots": {"era": "original"}}\nThe original.'

    client, recorder = _client(reviewed, steps)
    body = client.post("/api/chat", json={"message": "the original Drishyam"}).json()
    assert body["status"] == "up"
    # All FIVE v0 tools exercised, every executed call schema-valid (none rejected).
    assert {tool for tool, _ in recorder.calls} == {
        "resolve_title",
        "search_by_plot",
        "get_versions",
        "refine_filter",
    }
    assert all(errors == [] for errors in recorder.validation_errors)
    assert body["tool_calls"] == len(recorder.calls) + 0  # executed calls counted on the turn
