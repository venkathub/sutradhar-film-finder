"""FastAPI orchestration service (P5 task 9, P5_SPEC §2.2–§2.4, DEC-P5-1).

The single Python runtime (D1: Spring Boot gateway deliberately cut): one app factory
wiring the proven seams together —

- ``POST /api/chat``     — the orchestrator turn (GPU up) or the structured offline
                            payload (GPU off/error — **HTTP 200, never a 5xx**);
- ``GET  /api/health``   — aggregate: api / db / redis / llm / embed / rerank;
- ``GET  /api/status``   — the cached degradation state only (cheap, D5 TTL);
- ``GET  /api/replays``  — replay discovery: pinned run id + replayable fixtures (P6);
- ``GET  /api/replay/{fixture_id}`` — committed pinned-run transcripts (zero-GPU story);
- ``GET  /api/metrics``  — lands with cost accounting (task 10).
- ``GET  /``             — the built chat UI (static, same-origin; P6) when
                            ``ui/app/dist`` exists — otherwise API-only mode.

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
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from sutradhar.config import Settings, get_settings
from sutradhar.evals.driver import ToolExecutor
from sutradhar.evals.prompts import PromptArtifacts, load_serving_prompt_artifacts
from sutradhar.obs.cost import MetricsAccumulator, request_cost
from sutradhar.obs.tracing import Tracer
from sutradhar.rag.calibration import assert_calibration_matches
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
    list_replays,
    load_replay,
    offline_payload,
)
from sutradhar.serving.executor import build_live_executor
from sutradhar.serving.llm_client import LLMClient
from sutradhar.serving.orchestrator import Orchestrator
from sutradhar.serving.schemas import ChatRequest, TurnAborted
from sutradhar.serving.security import authorize_chat, make_rate_limit_key
from sutradhar.serving.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionLimitError,
    SessionStore,
)
from sutradhar.toolcalls import load_tool_schema

logger = logging.getLogger("sutradhar.serving")

RETRIEVAL_RUNS_DIR = Path("evals/retrieval_runs")
UI_DIST_DIR = Path("ui/app/dist")


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


def _limiter_storage(settings: Settings) -> str:
    """Redis-backed limits when Redis is reachable, in-memory otherwise.

    Mirrors the DEC-P5-2 session-store posture: degrade, never crash. Memory
    storage is per-process — fine for the single-worker demo topology (DEC-P6-5);
    the compose stack has Redis and gets shared counters automatically.
    """
    try:
        import redis

        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5).ping()
        return settings.redis_url
    except Exception:  # noqa: BLE001 — any Redis failure degrades, never crashes
        logger.warning("redis unreachable for rate limiting — using in-memory limits")
        return "memory://"


def _rate_limited_handler(request: Request, exc: Exception) -> JSONResponse:
    """429 in the standard envelope (P7 task 4); detail carries the limit only."""
    detail = getattr(exc, "detail", "rate limit exceeded")
    return JSONResponse(
        {
            "error": "rate_limited",
            "detail": f"rate limit exceeded: {detail}",
            "request_id": getattr(request.state, "request_id", ""),
        },
        status_code=429,
        headers={"Retry-After": "60"},
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
    config = RetrievalConfig(
        chunk_config=str(record["retrieval_config"]["chunk_config"]),
        embed_model=settings.embed_model,
        index_version=run_id,
        rerank_depth=int(record["retrieval_config"]["rerank_depth"]),
    )
    # P7 task 8 (DEC-P7-3): θ staleness gate — abstention goes live ONLY against
    # the exact stack it was calibrated on; any drift hard-fails at wiring time.
    assert_calibration_matches(
        embed_model=config.embed_model,
        index_version=config.index_version,
        chunk_config=config.chunk_config,
        rerank_depth=config.rerank_depth,
    )
    return config


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
    metrics: MetricsAccumulator | None = None,
    ui_dist: Path | None = None,
    rate_limit_storage: str | None = None,
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
    counters = metrics or MetricsAccumulator()
    replay_kwargs: dict[str, Any] = {"runs_dir": runs_dir} if runs_dir else {}

    def _offline(conversation_id: str | None, detail: str = OFFLINE_DETAIL) -> dict[str, Any]:
        """offline_payload with the DEMO_VIDEO_URL evidence link wired in (P6 task 1)."""
        return offline_payload(conversation_id, detail, demo_video=settings.demo_video_url)

    app = FastAPI(title="Sutradhar API", docs_url=None, redoc_url=None)
    app.state.status_cache = cache
    app.state.metrics = counters

    # DEC-P7-2: rate limiting on the paid path — token-first key, IP fallback.
    limiter = Limiter(
        key_func=make_rate_limit_key(settings),
        storage_uri=rate_limit_storage or _limiter_storage(settings),
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limited_handler)
    if settings.chat_auth == "disabled":
        logger.warning("CHAT_AUTH=disabled — /api/chat auth is OFF (local/e2e only)")

    @app.middleware("http")
    async def trace_and_envelope(request: Request, call_next: Any) -> Response:
        """One request span (the DEC-P3-6 seam) + a scrubbed error envelope.

        P7 task 3 (DEC-P7-1 finding 10): the client sees a generic message plus a
        request id only; ``str(exc)`` (which can carry DSNs, paths, internal state)
        goes to the server log, keyed by the same id. The id is echoed on every
        response as ``X-Request-Id`` so users can quote it and operators can grep
        logs / Langfuse traces for the exact request.
        """
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        with trace.span(
            "request",
            kind="agent",
            input={"method": request.method, "path": request.url.path},
            metadata={"request_id": request_id},
        ) as span:
            request.state.trace_span = span  # routes attach cost metadata here
            try:
                response: Response = await call_next(request)
            except Exception as exc:  # noqa: BLE001 — envelope, never a bare traceback
                # Full detail server-side ONLY (logger.exception includes the traceback).
                logger.exception(
                    "unhandled error on %s [request_id=%s]", request.url.path, request_id
                )
                span.update(
                    output={"error": type(exc).__name__, "request_id": request_id},
                    level="ERROR",
                )
                return JSONResponse(
                    {"error": "internal_error", "request_id": request_id},
                    status_code=500,
                    headers={"X-Request-Id": request_id},
                )
            span.update(output={"status_code": response.status_code})
            response.headers["X-Request-Id"] = request_id
            return response

    @app.post("/api/chat")
    @limiter.limit(settings.chat_rate_limit)
    def chat(payload: ChatRequest, request: Request) -> JSONResponse:
        status = cache.current()
        if status.status != "up":
            counters.record("off")
            return JSONResponse(
                _offline(payload.conversation_id, f"{OFFLINE_DETAIL} (endpoint {status.status})")
            )
        # Auth gates the GPU-up path only (DEC-P7-2): degradation stays open,
        # the endpoint that burns GPU seconds never is.
        denied = authorize_chat(request, settings)
        if denied is not None:
            counters.record("denied")
            return denied
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
                outcome = orchestrator.run_turn(payload.conversation_id, payload.message)
            finally:
                db.close()
        except SessionLimitError as exc:
            counters.record("limit")
            return JSONResponse(
                {"error": "limit", "detail": str(exc)},
                status_code=400,
            )
        if isinstance(outcome, TurnAborted):
            cache.invalidate()  # the endpoint died mid-turn; re-probe on the next request
            counters.record("aborted")
            return JSONResponse(
                _offline(
                    outcome.conversation_id,
                    f"{OFFLINE_DETAIL} (turn aborted: {outcome.detail})",
                )
            )
        # Per-request accounting (P5_SPEC §2.7): amortized GPU cost + tokens/sec,
        # on the response, the trace, and the /api/metrics accumulator.
        cost = request_cost(
            {
                "prompt_tokens": outcome.usage.prompt_tokens,
                "completion_tokens": outcome.usage.completion_tokens,
            },
            outcome.latency_ms,
            settings.gpu_hourly_usd,
        )
        outcome.usage.cost_usd = cost.cost_usd
        request.state.trace_span.update(
            metadata={
                "cost_usd": cost.cost_usd,
                "tokens_per_sec": cost.tokens_per_sec,
                "usd_per_1k_tokens": cost.usd_per_1k_tokens,
                "gpu_hourly_usd": settings.gpu_hourly_usd,
            }
        )
        counters.record(
            "up",
            latency_ms=outcome.latency_ms,
            usage={
                "prompt_tokens": outcome.usage.prompt_tokens,
                "completion_tokens": outcome.usage.completion_tokens,
            },
            cost_usd=cost.cost_usd,
        )
        return JSONResponse(outcome.model_dump(mode="json"))

    @app.get("/api/metrics")
    def api_metrics() -> dict[str, Any]:
        """The §2.2 JSON summary (chat-scoped). Self-describing for committed evidence."""
        return {
            "model": settings.llm_model,
            "gpu_hourly_usd": settings.gpu_hourly_usd,
            **counters.snapshot(),
        }

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        status = cache.current()
        body: dict[str, Any] = {"status": status.status, "detail": status.detail}
        if status.status != "up":
            body["evidence"] = _offline(None)["evidence"]
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

    @app.get("/api/replays")
    def api_replays() -> dict[str, Any]:
        """Replay discovery (P6 task 1): the pinned run + the fixtures it can replay."""
        return list_replays(**replay_kwargs)

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

    # Built UI served same-origin at / (P6 task 2, DEC-P6-5) — API routes above win
    # (mounts match after routes). Missing dist ⇒ API-only mode, never a crash: the
    # fresh-clone zero-node experience keeps working (`make ui-build` adds the UI).
    dist = ui_dist if ui_dist is not None else UI_DIST_DIR
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    else:
        logger.info("ui dist not found at %s — API-only mode (run `make ui-build`)", dist)

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
        # P7 task 3: exception *class* only — str(exc) can embed the DSN. Full
        # detail goes to the server log.
        logger.warning("db health probe failed: %s: %s", type(exc).__name__, exc)
        return {"status": "off", "detail": type(exc).__name__}


def _redis_health(settings: Settings) -> dict[str, Any]:
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        return {"status": "up"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis health probe failed: %s: %s", type(exc).__name__, exc)
        return {"status": "off", "detail": type(exc).__name__}


def _provider_health(name: str, base_url: str | None, model: str) -> dict[str, Any]:
    if not base_url:
        return {"status": "off", "detail": f"{name} endpoint not configured (GPU off)"}
    provider = (
        HttpEmbeddings(base_url, model)
        if name == "embeddings"
        else HttpReranker(base_url, model, lambda h: list(h))
    )
    return provider.health().to_dict()
