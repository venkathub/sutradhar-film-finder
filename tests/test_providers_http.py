"""P5 task 3 — live HTTP embedding/rerank providers (P5_SPEC §2.1, mocked transport).

Locks the client side of the GPU sidecar contract (dense+sparse ``/embeddings``,
``/rerank``) and the DEC-P0-4 failure taxonomy: connection failure ⇒ typed off-state,
HTTP/shape failure ⇒ typed error-state — never a raw httpx exception, never a crash.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from sutradhar.rag.artifacts import DenseSparse
from sutradhar.rag.providers import (
    HttpEmbeddings,
    HttpReranker,
    ProviderUnavailableError,
    chunk_text_lookup,
)

BASE = "http://gpu.example:8001/v1"


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _embed_response(request: httpx.Request) -> httpx.Response:
    """Sidecar contract: OpenAI shape + per-item sparse extension; order by index."""
    payload = json.loads(request.content)
    assert request.url.path.endswith("/embeddings")
    data = [
        {
            "index": i,
            "object": "embedding",
            "embedding": [float(i) + 0.25, float(i) + 0.5],
            "sparse": {"1042": 0.31, "7": 0.02},
        }
        for i, _ in enumerate(payload["input"])
    ]
    # Deliberately out of order: the client must sort by index.
    return httpx.Response(200, json={"object": "list", "data": list(reversed(data))})


class TestHttpEmbeddings:
    def test_request_shape_and_dense_sparse_parse(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _embed_response(request)

        provider = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(handler))
        out = provider.embed(["query one", "query two"])

        assert seen == {"model": "BAAI/bge-m3", "input": ["query one", "query two"]}
        assert len(out) == 2 and all(isinstance(e, DenseSparse) for e in out)
        # Order restored from index despite the reversed wire order.
        assert out[0].dense.tolist() == pytest.approx([0.25, 0.5])
        assert out[1].dense.tolist() == pytest.approx([1.25, 1.5])
        assert out[0].dense.dtype == np.float32
        # Sparse token ids parsed to int keys (artifact shape).
        assert out[0].sparse == {1042: 0.31, 7: 0.02}

    def test_connection_failure_is_typed_off(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        provider = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(handler))
        with pytest.raises(ProviderUnavailableError) as exc:
            provider.embed(["q"])
        assert exc.value.status == "off"
        assert exc.value.provider == "embeddings"

    def test_http_500_is_typed_error(self) -> None:
        provider = HttpEmbeddings(
            BASE, "BAAI/bge-m3", http_client=_client(lambda r: httpx.Response(500, text="boom"))
        )
        with pytest.raises(ProviderUnavailableError) as exc:
            provider.embed(["q"])
        assert exc.value.status == "error"

    def test_missing_sparse_field_is_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1]}]},  # no sparse ext
            )

        provider = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(handler))
        with pytest.raises(ProviderUnavailableError) as exc:
            provider.embed(["q"])
        assert exc.value.status == "error"
        assert "contract" in exc.value.detail

    def test_count_mismatch_is_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1], "sparse": {}}]},
            )

        provider = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(handler))
        with pytest.raises(ProviderUnavailableError) as exc:
            provider.embed(["a", "b"])
        assert exc.value.status == "error"

    def test_health_up_and_off(self) -> None:
        up = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(lambda r: httpx.Response(200)))
        status = up.health()
        assert status.status == "up" and status.model == "BAAI/bge-m3"

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        down = HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(refuse))
        assert down.health().status == "off"  # never raises (DEC-P0-4)

    def test_health_probes_server_root_not_v1(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200)

        HttpEmbeddings(BASE, "BAAI/bge-m3", http_client=_client(handler)).health()
        assert paths == ["/health"]  # {base%/v1}/health, the LLMClient convention


class TestHttpReranker:
    HASHES = ["a" * 64, "b" * 64, "c" * 64]
    TEXTS = {"a" * 64: "text A", "b" * 64: "text B", "c" * 64: "text C"}

    def _lookup(self, hashes: list[str]) -> list[str]:
        return [self.TEXTS[h] for h in hashes]

    def test_documents_resolved_and_scores_in_hash_order(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            # Out-of-order results: the client must map back via index.
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 2, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                        {"index": 1, "relevance_score": 0.5},
                    ]
                },
            )

        reranker = HttpReranker(
            BASE, "BAAI/bge-reranker-v2-m3", self._lookup, http_client=_client(handler)
        )
        scores = reranker.score("the query", self.HASHES)

        assert seen == {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "the query",
            "documents": ["text A", "text B", "text C"],  # hash order preserved
        }
        assert scores == [0.1, 0.5, 0.9]  # input-hash order, not wire order

    def test_empty_hashes_short_circuits(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no HTTP call expected for empty input")

        reranker = HttpReranker(BASE, "m", self._lookup, http_client=_client(handler))
        assert reranker.score("q", []) == []

    def test_connection_failure_is_typed_off(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow")

        reranker = HttpReranker(BASE, "m", self._lookup, http_client=_client(handler))
        with pytest.raises(ProviderUnavailableError) as exc:
            reranker.score("q", self.HASHES)
        assert exc.value.status == "off"
        assert exc.value.provider == "rerank"

    def test_missing_result_index_is_typed_error(self) -> None:
        handler = lambda r: httpx.Response(  # noqa: E731
            200,
            json={"results": [{"index": 0, "relevance_score": 0.4}]},  # 1 of 3
        )
        reranker = HttpReranker(BASE, "m", self._lookup, http_client=_client(handler))
        with pytest.raises(ProviderUnavailableError) as exc:
            reranker.score("q", self.HASHES)
        assert exc.value.status == "error"


class TestChunkTextLookup:
    class _StubSession:
        """Stub of Session.execute for the DISTINCT content_hash → text query."""

        def __init__(self, rows: dict[str, str]) -> None:
            self._rows = rows

        def execute(self, _stmt: Any, params: dict[str, Any]) -> Any:
            rows = self._rows

            class Result:
                def all(self) -> list[Any]:
                    class Row:
                        def __init__(self, h: str, t: str) -> None:
                            self.content_hash = h
                            self.text = t

                    return [Row(h, rows[h]) for h in params["h"] if h in rows]

            return Result()

    def test_resolves_in_hash_order(self) -> None:
        lookup = chunk_text_lookup(self._StubSession({"h1": "one", "h2": "two"}))  # type: ignore[arg-type]
        assert lookup(["h2", "h1"]) == ["two", "one"]
        assert lookup([]) == []

    def test_missing_hash_is_typed_error(self) -> None:
        lookup = chunk_text_lookup(self._StubSession({"h1": "one"}))  # type: ignore[arg-type]
        with pytest.raises(ProviderUnavailableError) as exc:
            lookup(["h1", "h2"])
        assert exc.value.status == "error"
        assert "h2"[:12] in exc.value.detail
