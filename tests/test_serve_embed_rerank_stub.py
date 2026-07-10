"""P5 task 4 — sidecar contract dry-run (DEC-P2-7 pattern, no models, no network).

The lock that keeps client and server from drifting: the SAME ``HttpEmbeddings`` /
``HttpReranker`` the live path uses (task 3) round-trip against the sidecar app
in-process (``fastapi.testclient.TestClient`` injected as their ``http_client``).
Stub backends are asserted byte-compatible with ``embed_and_score.py``'s stubs, so
stub serving and stub artifacts agree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sutradhar.rag.artifacts import DenseSparse
from sutradhar.rag.providers import HttpEmbeddings, HttpReranker, ProviderUnavailableError

REPO_ROOT = Path(__file__).resolve().parent.parent

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BASE = "http://testserver/v1"


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "rag-engine" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sidecar() -> ModuleType:
    return _load_script("serve_embed_rerank")


@pytest.fixture(scope="module")
def http_client(sidecar: ModuleType) -> TestClient:
    app = sidecar.build_app(
        sidecar.StubEmbedder(),
        sidecar.StubScorer(),
        embed_model=EMBED_MODEL,
        rerank_model=RERANK_MODEL,
    )
    # raise_server_exceptions=False: a server bug must surface as an HTTP status the
    # typed-provider taxonomy handles, never as an in-process exception leak.
    return TestClient(app, raise_server_exceptions=False)


def test_health_roundtrip_via_provider(http_client: TestClient) -> None:
    provider = HttpEmbeddings(BASE, EMBED_MODEL, http_client=http_client)
    status = provider.health()
    assert status.status == "up"


def test_embeddings_roundtrip_via_provider(http_client: TestClient) -> None:
    provider = HttpEmbeddings(BASE, EMBED_MODEL, http_client=http_client)
    out = provider.embed(["a man buries a body", "wo film jisme baap evidence chhupata hai"])

    assert len(out) == 2 and all(isinstance(e, DenseSparse) for e in out)
    for emb in out:
        assert emb.dense.shape == (1024,)  # BGE-M3 dim — loads into the pgvector column
        assert emb.dense.dtype == np.float32
        assert emb.sparse and all(isinstance(k, int) for k in emb.sparse)
    # Distinct texts → distinct vectors; deterministic per text across calls.
    assert not np.allclose(out[0].dense, out[1].dense)
    again = provider.embed(["a man buries a body"])[0]
    assert np.array_equal(again.dense, out[0].dense)
    assert again.sparse == out[0].sparse


def test_stub_embedder_matches_embed_and_score_stub(sidecar: ModuleType) -> None:
    """Byte-parity with the P2 job's stub: stub serving == stub artifacts."""
    p2_job = _load_script("embed_and_score")
    text = "Papanasam (Tamil, 2015) — remake of Drishyam (Malayalam, 2013)."
    dense_a, sparse_a = sidecar.StubEmbedder().embed([text])
    dense_b, sparse_b = p2_job.StubEmbedder().embed([text])
    assert np.array_equal(dense_a, dense_b)
    assert sparse_a == sparse_b
    assert sidecar.StubScorer().score([("q", text)]) == p2_job.StubScorer().score([("q", text)])


def test_rerank_roundtrip_via_provider(http_client: TestClient) -> None:
    texts = {"h1": "doc one", "h2": "doc two", "h3": "doc three"}
    reranker = HttpReranker(
        BASE, RERANK_MODEL, lambda hashes: [texts[h] for h in hashes], http_client=http_client
    )
    scores = reranker.score("which doc", ["h1", "h2", "h3"])

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)  # sigmoid semantics (DEC-P2-5)
    # Deterministic + input-order aligned: reversed hashes → reversed scores.
    assert reranker.score("which doc", ["h3", "h2", "h1"]) == scores[::-1]


def test_wrong_model_id_is_400_typed_error(http_client: TestClient) -> None:
    """Env misconfiguration surfaces as a typed client error, never silent drift."""
    provider = HttpEmbeddings(BASE, "some/other-model", http_client=http_client)
    with pytest.raises(ProviderUnavailableError) as exc:
        provider.embed(["q"])
    assert exc.value.status == "error"

    reranker = HttpReranker(BASE, "some/other-reranker", lambda h: list(h), http_client=http_client)
    with pytest.raises(ProviderUnavailableError) as exc:
        reranker.score("q", ["h"])
    assert exc.value.status == "error"


def test_empty_input_is_400(http_client: TestClient) -> None:
    resp = http_client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": []})
    assert resp.status_code == 400
    resp = http_client.post(
        "/v1/rerank", json={"model": RERANK_MODEL, "query": "q", "documents": []}
    )
    assert resp.status_code == 400


def test_openai_compat_string_input_and_encoding_format(http_client: TestClient) -> None:
    """OpenAI-compat (the 2026-07-05 relevancy root cause): clients like RAGAS's
    OpenAIEmbeddings send `input` as a BARE STRING plus `encoding_format=base64`; the
    sidecar must accept both and always answer with float arrays."""
    resp = http_client.post(
        "/v1/embeddings",
        json={"model": EMBED_MODEL, "input": "single string", "encoding_format": "base64"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1 and isinstance(data[0]["embedding"], list)
    assert len(data[0]["embedding"]) == 1024

    # Equivalent to the list form (same text, same vector).
    as_list = http_client.post(
        "/v1/embeddings", json={"model": EMBED_MODEL, "input": ["single string"]}
    ).json()["data"]
    assert as_list[0]["embedding"] == data[0]["embedding"]


def test_rerank_unknown_field_still_422(http_client: TestClient) -> None:
    """The rerank route keeps extra='forbid' — it has no OpenAI-compat obligation."""
    resp = http_client.post(
        "/v1/rerank",
        json={"model": RERANK_MODEL, "query": "q", "documents": ["d"], "injected": 1},
    )
    assert resp.status_code == 422
