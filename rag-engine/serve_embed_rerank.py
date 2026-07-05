"""GPU embed/rerank sidecar (P5 task 4, D4/DEC-P5-4) — SELF-CONTAINED: stdlib + numpy +
fastapi/uvicorn, plus FlagEmbedding in real mode. No sutradhar import: the ``serve``
session (DEC-P2-7 relay pattern) ships this single file to the instance, so it must run
without the repo. The HTTP contract is locked laptop-side by
``tests/test_serve_embed_rerank_stub.py``, which round-trips the P5 live providers
(``sutradhar.rag.providers``) against this app in-process.

Why FlagEmbedding and not a second vLLM / Infinity process (DEC-P5-4): the P2 index and
the calibrated θ were produced by FlagEmbedding's BGE-M3 dense+sparse pair; only the same
library reproduces those lexical weights bit-for-bit — parity beats elegance.

Routes (one A100 sidecar on :8001, next to vLLM on :8000):

- ``GET  /health``          → 200 once models are loaded (health-gate for the session).
- ``POST /v1/embeddings``   → OpenAI-compatible ``{"model", "input": [texts]}`` →
  standard dense ``"embedding"`` PLUS the sparse extension ``"sparse": {token_id: w}``.
- ``POST /v1/rerank``       → ``{"model", "query", "documents"}`` →
  ``{"results": [{"index", "relevance_score"}]}`` — sigmoid scores (``normalize=True``),
  the same score semantics as the recorded P2 rerank matrix (DEC-P2-5).

A request naming a different ``model`` than the served one is a 400 — an env
misconfiguration must surface as a typed client error, never as silent scoring drift.

    # on the GPU instance (real):
    python serve_embed_rerank.py --port 8001
    # anywhere (deterministic stub, no models — tests + dry runs):
    python serve_embed_rerank.py --port 8001 --stub

Model ids come from --embed-model/--rerank-model or the EMBED_MODEL/RERANK_MODEL env
vars (DEC-0002 pins as fallback defaults) — never hardcoded call sites.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

DENSE_DIM = 1024  # BGE-M3; the stub matches it so stub runs exercise the same shapes

# DEC-0002 pins — fallback defaults only; env/CLI always win (CLAUDE.md: env-swappable).
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# --- Backends (Protocols identical to embed_and_score.py — the P2 job this pairs with) ---


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]: ...


class PairScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StubEmbedder:
    """Deterministic no-model embeddings (BYTE-COMPATIBLE with embed_and_score.py's stub:
    hash-seeded dense + word-hash sparse), so stub serving and stub artifacts agree."""

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        dense = np.zeros((len(texts), DENSE_DIM), dtype=np.float32)
        sparse: list[dict[int, float]] = []
        for i, text in enumerate(texts):
            seed = int(_sha(text)[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(DENSE_DIM).astype(np.float32)
            dense[i] = vec / np.linalg.norm(vec)
            weights: dict[int, float] = {}
            for word in text.lower().split()[:64]:
                token = int(_sha(word)[:8], 16) % 250_000 + 1
                weights[token] = round(min(1.0, weights.get(token, 0.0) + 0.2), 4)
            sparse.append(weights or {seed % 250_000 + 1: 0.5})
        return dense, sparse


class StubScorer:
    """Deterministic [0,1] pair scores from sha256(query||chunk) — embed_and_score parity."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [int(_sha(q + "\x00" + c)[:8], 16) / 0xFFFFFFFF for q, c in pairs]


class BgeM3Embedder:
    """Real leg: BGE-M3 dense + sparse lexical weights in one pass (FlagEmbedding)."""

    def __init__(self, model_id: str) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_id, use_fp16=True)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        out = self.model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = [
            {int(token_id): float(weight) for token_id, weight in row.items()}
            for row in out["lexical_weights"]
        ]
        return dense, sparse


class BgeReranker:
    """Real leg: cross-encoder scores, sigmoid-normalized (normalize=True, DEC-P2-5)."""

    def __init__(self, model_id: str) -> None:
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(model_id, use_fp16=True)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.compute_score([list(p) for p in pairs], normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


# --- Wire models (extra="forbid": a malformed client request is a 422, never a guess) ---


class EmbeddingsRequest(BaseModel):
    # OpenAI-compatible: `input` may be a single string OR a list, and clients (e.g. RAGAS's
    # OpenAIEmbeddings) send `encoding_format` — accept and ignore extra standard fields
    # rather than 422 (the 2026-07-05 relevancy root cause). We always return float arrays.
    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[str]

    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    query: str
    documents: list[str]


# --- App factory (tests build it with stubs; main() with the real legs) ---


def build_app(
    embedder: Embedder,
    scorer: PairScorer,
    *,
    embed_model: str,
    rerank_model: str,
) -> FastAPI:
    app = FastAPI(title="sutradhar embed/rerank sidecar", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, str]:
        # Reached only after model construction in main() → loaded == serving.
        return {"status": "ok", "embed_model": embed_model, "rerank_model": rerank_model}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest) -> dict[str, Any]:
        if req.model != embed_model:
            raise HTTPException(400, f"served embed model is {embed_model!r}, not {req.model!r}")
        texts = req.texts()
        if not texts:
            raise HTTPException(400, "input must be a non-empty string or list of texts")
        dense, sparse = embedder.embed(texts)
        return {
            "object": "list",
            "model": embed_model,
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [float(x) for x in dense[i]],
                    # Sparse extension: BGE-M3 lexical weights (token id → weight) —
                    # the field vLLM pooling / Infinity cannot serve (DEC-P5-4).
                    "sparse": {str(k): float(v) for k, v in sparse[i].items()},
                }
                for i in range(len(texts))
            ],
        }

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> dict[str, Any]:
        if req.model != rerank_model:
            raise HTTPException(400, f"served rerank model is {rerank_model!r}, not {req.model!r}")
        if not req.documents:
            raise HTTPException(400, "documents must be a non-empty list of texts")
        scores = scorer.score([(req.query, doc) for doc in req.documents])
        return {
            "model": rerank_model,
            "results": [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)],
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument(
        "--rerank-model", default=os.environ.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)
    )
    parser.add_argument("--stub", action="store_true", help="deterministic no-model dry run")
    args = parser.parse_args(argv)

    if args.stub:
        embedder: Embedder = StubEmbedder()
        scorer: PairScorer = StubScorer()
    else:
        # Load BEFORE binding the port: /health answering 200 == models are serving.
        embedder = BgeM3Embedder(args.embed_model)
        scorer = BgeReranker(args.rerank_model)

    app = build_app(embedder, scorer, embed_model=args.embed_model, rerank_model=args.rerank_model)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
