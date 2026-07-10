"""Live HTTP embedding/rerank providers for the P5 GPU path (P5_SPEC §2.1/§2.3).

The P2 promise — "the live path swaps providers, not code" — lands here: these classes
implement the frozen ``EmbeddingProvider`` / ``RerankProvider`` protocols (P2_SPEC §2.5),
so ``Retriever``/``search_by_plot`` run unchanged against the on-demand GPU sidecar
(``rag-engine/serve_embed_rerank.py``, D4: FlagEmbedding for artifact parity).

Sidecar contract (client side locked HERE; the sidecar implements to match, task 4):

- ``POST {EMBED_BASE_URL}/embeddings`` — OpenAI-compatible ``{"model", "input": [texts]}``;
  each ``data[i]`` carries the standard dense ``"embedding"`` PLUS the sparse extension
  ``"sparse": {"<token_id>": weight}`` (BGE-M3 lexical weights — vLLM/Infinity cannot
  produce these; the FlagEmbedding sidecar can, DEC-P5-4).
- ``POST {RERANK_BASE_URL}/rerank`` — ``{"model", "query", "documents": [texts]}`` →
  ``{"results": [{"index": i, "relevance_score": s}, …]}`` (TEI/Jina/vLLM convention),
  scores **sigmoid** to match the recorded P2 rerank matrix.

The frozen ``RerankProvider.score(query, chunk_hashes)`` protocol passes content HASHES
(that is what ``Retriever`` sends); a live cross-encoder needs the chunk *texts*, so
``HttpReranker`` takes a ``text_lookup`` dependency — :func:`chunk_text_lookup` resolves
hashes from the ``chunks`` table (``Chunk.text`` is byte-exactly what P2 embedded/scored).

Failure taxonomy reuses DEC-P0-4: connection/timeout ⇒ ``status="off"`` (a paused GPU is
a first-class state, not a bug); HTTP 5xx / malformed body ⇒ ``status="error"``. Because
the protocols must return values, failures surface as the *typed*
:class:`ProviderUnavailableError` (never a raw httpx exception) — the orchestrator
catches it and degrades. ``health()`` mirrors ``LLMClient.health()``'s
:class:`EndpointStatus` shape for the ``/api/health`` aggregate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import httpx
import numpy as np
from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from sutradhar.rag.artifacts import DenseSparse
from sutradhar.serving.llm_client import EndpointStatus

Status = Literal["off", "error"]

# Connection-level failures ⇒ "off" (DEC-P0-4: the on-demand GPU is normally paused).
_OFF_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.TimeoutException,
)


class ProviderUnavailableError(RuntimeError):
    """A neural provider endpoint is off or erroring — typed, never a raw httpx leak."""

    def __init__(self, provider: str, status: Status, detail: str) -> None:
        super().__init__(f"{provider}: endpoint {status} — {detail}")
        self.provider = provider
        self.status: Status = status
        self.detail = detail


def _health_url(base_url: str) -> str:
    """Liveness route at the server root (the LLMClient convention: ``{base%/v1}/health``)."""
    base = base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    return f"{root}/health"


def _post_json(
    http: httpx.Client, provider: str, url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST + parse with the DEC-P0-4 taxonomy applied (off | error, typed)."""
    try:
        resp = http.post(url, json=payload)
    except _OFF_ERRORS as exc:
        raise ProviderUnavailableError(
            provider, "off", f"endpoint OFF ({type(exc).__name__}) — bring up the GPU"
        ) from exc
    if resp.status_code != 200:
        raise ProviderUnavailableError(
            provider, "error", f"endpoint reachable but returned HTTP {resp.status_code}"
        )
    try:
        body = resp.json()
    except ValueError as exc:
        raise ProviderUnavailableError(provider, "error", "malformed JSON body") from exc
    if not isinstance(body, dict):
        raise ProviderUnavailableError(provider, "error", "response body is not an object")
    return body


def _probe_health(http: httpx.Client, provider: str, base_url: str, model: str) -> EndpointStatus:
    """GET the root /health route → EndpointStatus (never raises for a down endpoint)."""
    try:
        resp = http.get(_health_url(base_url))
    except _OFF_ERRORS as exc:
        return EndpointStatus(
            status="off",
            model=model,
            sample_token=None,
            latency_ms=None,
            detail=f"{provider} endpoint OFF ({type(exc).__name__})",
        )
    if resp.status_code >= 400:
        return EndpointStatus(
            status="error",
            model=model,
            sample_token=None,
            latency_ms=None,
            detail=f"{provider} /health returned {resp.status_code}",
        )
    return EndpointStatus(
        status="up",
        model=model,
        sample_token=None,
        latency_ms=None,
        detail=f"{provider} endpoint UP",
    )


class HttpEmbeddings:
    """Live ``EmbeddingProvider`` over the sidecar's OpenAI-compatible ``/embeddings``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = http_client or httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str]) -> list[DenseSparse]:
        body = _post_json(
            self._http,
            "embeddings",
            f"{self._base}/embeddings",
            {"model": self._model, "input": texts},
        )
        try:
            data = sorted(body["data"], key=lambda item: int(item["index"]))
            out = [
                DenseSparse(
                    dense=np.asarray(item["embedding"], dtype=np.float32),
                    sparse={int(k): float(v) for k, v in item["sparse"].items()},
                )
                for item in data
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderUnavailableError(
                "embeddings",
                "error",
                f"response missing dense+sparse contract fields ({type(exc).__name__}: {exc})",
            ) from exc
        if len(out) != len(texts):
            raise ProviderUnavailableError(
                "embeddings", "error", f"expected {len(texts)} embeddings, got {len(out)}"
            )
        return out

    def health(self) -> EndpointStatus:
        return _probe_health(self._http, "embeddings", self._base, self._model)


# Resolves chunk content hashes → the exact texts that were embedded/scored (P2 parity).
TextLookup = Callable[[list[str]], list[str]]


def chunk_text_lookup(session: Session) -> TextLookup:
    """DB-backed hash→text resolver over the ``chunks`` table (``content_hash`` keyed)."""

    def lookup(chunk_hashes: list[str]) -> list[str]:
        if not chunk_hashes:
            return []
        rows = session.execute(
            sqltext("SELECT DISTINCT content_hash, text FROM chunks WHERE content_hash = ANY(:h)"),
            {"h": list(chunk_hashes)},
        ).all()
        by_hash = {row.content_hash: row.text for row in rows}
        missing = [h for h in chunk_hashes if h not in by_hash]
        if missing:
            raise ProviderUnavailableError(
                "rerank",
                "error",
                f"no chunk text for content_hash(es) {', '.join(h[:12] for h in missing)}…",
            )
        return [by_hash[h] for h in chunk_hashes]

    return lookup


class HttpReranker:
    """Live ``RerankProvider`` over the sidecar's ``/rerank`` (sigmoid scores, P2 parity)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        text_lookup: TextLookup,
        *,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._lookup = text_lookup
        self._http = http_client or httpx.Client(timeout=timeout_s)

    def score(self, query: str, chunk_hashes: list[str]) -> list[float]:
        if not chunk_hashes:
            return []
        documents = self._lookup(chunk_hashes)
        body = _post_json(
            self._http,
            "rerank",
            f"{self._base}/rerank",
            {"model": self._model, "query": query, "documents": documents},
        )
        try:
            scores: dict[int, float] = {
                int(r["index"]): float(r["relevance_score"]) for r in body["results"]
            }
            out = [scores[i] for i in range(len(chunk_hashes))]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderUnavailableError(
                "rerank",
                "error",
                f"response missing rerank contract fields ({type(exc).__name__}: {exc})",
            ) from exc
        return out

    def health(self) -> EndpointStatus:
        return _probe_health(self._http, "rerank", self._base, self._model)
