"""FastAPI orchestration service (P5 task 9, P5_SPEC §2.2–§2.4, DEC-P5-1).

The single Python runtime (D1: Spring Boot gateway deliberately cut): one app factory
wiring the proven seams together —

- ``POST /api/chat``     — the orchestrator turn (GPU up) or the structured offline
                            payload (GPU off/error — **HTTP 200, never a 5xx**);
- ``GET  /api/health``   — aggregate: api / db / redis / llm / embed / rerank;
- ``GET  /api/status``   — the cached degradation state only (cheap, D5 TTL);
- ``GET  /api/replay/{fixture_id}`` — committed pinned-run transcripts (zero-GPU story);
- ``GET  /api/metrics``  — lands with cost accounting (task 10).

Wiring posture: everything is constructor-injectable for tests; production defaults come
from ``Settings``. The DB engine is **lazy** — the GPU-off experience (offline payload +
replay) works on a fresh clone with no database and no GPU (the DoD clause). Redis
unreachable ⇒ in-memory session store (logged fallback, DEC-P5-2). The live retriever is
built only when ``EMBED_BASE_URL`` + ``RERANK_BASE_URL`` + ``RETRIEVAL_RUN`` are all set
(the `make gpu-serve` exports); otherwise ``search_by_plot`` degrades to tool-error
feedback and the graph tools still answer (see serving.executor).

Run: ``make api-up`` → ``uvicorn --factory sutradhar.serving.app:create_app``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from sutradhar.config import Settings, get_settings
from sutradhar.evals.driver import ToolExecutor
from sutradhar.evals.prompts import PromptArtifacts, load_serving_prompt_artifacts
from sutradhar.obs.tracing import Tracer
from sutradhar.rag.providers import (
    HttpEmbeddings,
    HttpReranker,
    ProviderUnavailableError,
    chunk_text_lookup,
)
from sutradhar.rag.retrieve import RetrievalConfig, RetrievalResult, Retriever
from sutradhar.serving import guardrails
from sutradhar.serving.degrade import (
    OFFLINE_DETAIL,
    StatusCache,
    available_replays,
    load_replay,
    offline_payload,
)
from sutradhar.serving.executor import build_live_executor
from sutradhar.serving.llm_client import LLMClient
from sutradhar.serving.orchestrator import Orchestrator
from sutradhar.serving.schemas import ChatRequest, TurnAborted
from sutradhar.serving.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionLimitError,
    SessionStore,
)
from sutradhar.toolcalls import load_tool_schema

logger = logging.getLogger("sutradhar.serving")

RETRIEVAL_RUNS_DIR = Path("evals/retrieval_runs")


class _OfflineRetriever:
    """Stands in when the GPU sidecar env is not configured: search_by_plot degrades to
    tool-error feedback (executor) while the DB-backed graph tools keep answering."""

    def retrieve(self, query: str) -> RetrievalResult:
        raise ProviderUnavailableError(
            "embeddings",
            "off",
            "EMBED_BASE_URL/RERANK_BASE_URL/RETRIEVAL_RUN not configured — "
            "live plot retrieval needs the on-demand GPU sidecar (make gpu-serve)",
        )


def _default_session_store(settings: Settings) -> SessionStore:
    """Redis when reachable, in-memory fallback otherwise (DEC-P5-2)."""
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        return RedisSessionStore(client, settings.session_ttl_s)
    except Exception as exc:  # noqa: BLE001 — any Redis failure degrades, never crashes
        logger.warning("redis unreachable (%s) — using in-memory session store", exc)
        return InMemorySessionStore(settings.session_ttl_s)


def _winner_retrieval_config(settings: Settings, runs_dir: Path) -> RetrievalConfig | None:
    """The pinned Table 1 winner cell as a RetrievalConfig (demo.py's _winner_cell logic).
    None when RETRIEVAL_RUN is unset or the artifact is absent."""
    run_id = settings.retrieval_run
    if not run_id:
        return None
    path = runs_dir / f"{run_id}.json"
    if not path.exists():
        logger.warning("RETRIEVAL_RUN %s not found under %s — plot search offline", run_id, path)
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    winner = artifact.get("winner")
    record = artifact["records"][winner]
    return RetrievalConfig(
        chunk_config=str(record["retrieval_config"]["chunk_config"]),
        embed_model=settings.embed_model,
        index_version=run_id,
        rerank_depth=int(record["retrieval_config"]["rerank_depth"]),
    )


def _default_make_executor(settings: Settings) -> Callable[[Session], ToolExecutor]:
    """Per-request executor factory: live providers when the serve window is up."""
    config = _winner_retrieval_config(settings, RETRIEVAL_RUNS_DIR)
    live = bool(settings.embed_base_url and settings.rerank_base_url and config)

    def make(db: Session) -> ToolExecutor:
        if not live:
            return build_live_executor(db, _OfflineRetriever())  # type: ignore[arg-type]
        assert config is not None
        retriever = Retriever(
            db,
            config,
            HttpEmbeddings(str(settings.embed_base_url), settings.embed_model),
            HttpReranker(
                str(settings.rerank_base_url), settings.rerank_model, chunk_text_lookup(db)
            ),
        )
        return build_live_executor(db, retriever)

    return make


def _default_session_factory() -> Callable[[], Session]:
    """LAZY DB wiring: nothing connects until the first up-path chat request."""
    from sutradhar.graph.db import create_graph_engine, create_session_factory

    factory: Any = None

    def make() -> Session:
        nonlocal factory
        if factory is None:
            factory = create_session_factory(create_graph_engine())
        session: Session = factory()
        return session

    return make


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    status_cache: StatusCache | None = None,
    session_store: SessionStore | None = None,
    session_factory: Callable[[], Session] | None = None,
    make_executor: Callable[[Session], ToolExecutor] | None = None,
    prompt_artifacts: PromptArtifacts | None = None,
    tracer: Tracer | None = None,
    runs_dir: Path | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    llm = llm_client or LLMClient(settings)
    cache = status_cache or StatusCache(llm.health)
    store = session_store or _default_session_store(settings)
    get_db = session_factory or _default_session_factory()
    executor_for = make_executor or _default_make_executor(settings)
    artifacts = prompt_artifacts or load_serving_prompt_artifacts()
    trace = tracer or Tracer(settings)  # no-op when LANGFUSE_* unset (DEC-P3-6)
    schema = load_tool_schema()
    replay_kwargs: dict[str, Any] = {"runs_dir": runs_dir} if runs_dir else {}

    app = FastAPI(title="Sutradhar API", docs_url=None, redoc_url=None)
    app.state.status_cache = cache

    @app.middleware("http")
    async def trace_and_envelope(request: Request, call_next: Any) -> Response:
        """One request span (the DEC-P3-6 seam) + a structured error envelope."""
        with trace.span(
            "request",
            kind="agent",
            input={"method": request.method, "path": request.url.path},
        ) as span:
            try:
                response: Response = await call_next(request)
            except Exception as exc:  # noqa: BLE001 — envelope, never a bare traceback
                logger.exception("unhandled error on %s", request.url.path)
                span.update(output={"error": type(exc).__name__}, level="ERROR")
                return JSONResponse(
                    {"error": "internal error", "detail": f"{type(exc).__name__}: {exc}"[:300]},
                    status_code=500,
                )
            span.update(output={"status_code": response.status_code})
            return response

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> JSONResponse:
        status = cache.current()
        if status.status != "up":
            return JSONResponse(
                offline_payload(
                    request.conversation_id, f"{OFFLINE_DETAIL} (endpoint {status.status})"
                )
            )
        try:
            db = get_db()
            try:
                orchestrator = Orchestrator(
                    llm,
                    store,
                    executor_for(db),
                    system_prompt=artifacts.system_prompt(),
                    prompt_hash=artifacts.prompt_hash,
                    schema=schema,
                    spotlight=guardrails.spotlight,
                    output_gate=guardrails.output_gate,
                    tracer=trace,
                )
                outcome = orchestrator.run_turn(request.conversation_id, request.message)
            finally:
                db.close()
        except SessionLimitError as exc:
            return JSONResponse(
                {"error": "limit", "detail": str(exc)},
                status_code=400,
            )
        if isinstance(outcome, TurnAborted):
            cache.invalidate()  # the endpoint died mid-turn; re-probe on the next request
            return JSONResponse(
                offline_payload(
                    outcome.conversation_id,
                    f"{OFFLINE_DETAIL} (turn aborted: {outcome.detail})",
                )
            )
        return JSONResponse(outcome.model_dump(mode="json"))

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        status = cache.current()
        body: dict[str, Any] = {"status": status.status, "detail": status.detail}
        if status.status != "up":
            body["evidence"] = offline_payload(None)["evidence"]
        return body

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return {
            "api": {"status": "up"},
            "db": _db_health(get_db),
            "redis": _redis_health(settings),
            "llm": cache.current().to_dict(),
            "embed": _provider_health("embeddings", settings.embed_base_url, settings.embed_model),
            "rerank": _provider_health("rerank", settings.rerank_base_url, settings.rerank_model),
        }

    @app.get("/api/replay/{fixture_id}")
    def api_replay(fixture_id: str) -> JSONResponse:
        payload = load_replay(fixture_id, **replay_kwargs)
        if payload is None:
            return JSONResponse(
                {
                    "error": "unknown fixture",
                    "detail": f"{fixture_id!r} is not in the pinned run",
                    "available": available_replays(**replay_kwargs),
                },
                status_code=404,
            )
        return JSONResponse(payload)

    return app


# --- health probes (small, independently testable) ---


def _db_health(get_db: Callable[[], Session]) -> dict[str, Any]:
    try:
        db = get_db()
        try:
            db.execute(sqltext("SELECT 1"))
        finally:
            db.close()
        return {"status": "up"}
    except Exception as exc:  # noqa: BLE001 — health reports, never raises
        return {"status": "off", "detail": f"{type(exc).__name__}: {exc}"[:200]}


def _redis_health(settings: Settings) -> dict[str, Any]:
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        return {"status": "up"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "off", "detail": f"{type(exc).__name__}: {exc}"[:200]}


def _provider_health(name: str, base_url: str | None, model: str) -> dict[str, Any]:
    if not base_url:
        return {"status": "off", "detail": f"{name} endpoint not configured (GPU off)"}
    provider = (
        HttpEmbeddings(base_url, model)
        if name == "embeddings"
        else HttpReranker(base_url, model, lambda h: list(h))
    )
    return provider.health().to_dict()
