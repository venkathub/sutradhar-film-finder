"""GPU embed+score job (P2 task 5) — SELF-CONTAINED: stdlib + numpy + pyarrow only,
plus FlagEmbedding in real mode. No sutradhar import: the HF-relay driver (DEC-P2-7)
ships this single file to the instance, so it must run without the repo. The output
format is locked to ``sutradhar.rag.artifacts`` by the laptop-side dry-run test
(``ArtifactRun.open`` + ``ArtifactEmbeddings`` must load what this writes).

    # on the GPU instance (real):
    python embed_and_score.py --inputs gpu_inputs.json --out artifacts/ --run-id <id>
    # anywhere (deterministic stub, no models — used by tests + `--stub` dry runs):
    python embed_and_score.py --inputs gpu_inputs.json --out artifacts/ --run-id <id> --stub

Writes ``<out>/<run_id>/``: ``meta.json``, a ``queries`` embedding bank, one
``corpus_<config>`` bank per chunk config, one ``rerank_scores_<config>.parquet``
(columns ``query_hash``, ``chunk_hash``, ``score`` — sigmoid-normalized [0,1], the
official BGE reranker score semantics, DEC-P2-5) holding the FULL query×chunk matrix,
and a sealing ``MANIFEST.sha256``. Embeddings are cached per content-hash across configs
(metadata cards repeat identically in every config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DENSE_DIM = 1024  # BGE-M3; the stub matches it so stub runs load into the same pgvector column


# --- Providers ---


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]: ...


class PairScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StubEmbedder:
    """Deterministic no-model embeddings: hash-seeded dense + WORD-HASH sparse weights.

    Sparse token ids are derived from the text's actual words, so genuine word overlap
    between query and chunk produces genuine sparse-channel overlap — the in-DB sparse
    leg is meaningfully exercised by stub runs, not just present."""

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
    """Deterministic [0,1] pair scores from sha256(query||chunk)."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [
            int(_sha(q + "\x00" + c)[:8], 16) / 0xFFFFFFFF  # uniform [0,1]
            for q, c in pairs
        ]


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
    """Real leg: cross-encoder scores, sigmoid-normalized (normalize=True)."""

    def __init__(self, model_id: str) -> None:
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(model_id, use_fp16=True)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.compute_score([list(p) for p in pairs], normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


# --- Artifact writing (format-locked to sutradhar.rag.artifacts by the dry-run test) ---


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )


def _write_bank(
    root: Path, bank: str, hashes: list[str], dense: np.ndarray, sparse: list[dict[int, float]]
) -> None:
    assert len(hashes) == dense.shape[0] == len(sparse), f"bank {bank} misaligned"
    _write_json(root / f"{bank}_hashes.json", hashes)
    np.save(root / f"{bank}_dense.npy", np.ascontiguousarray(dense, dtype=np.float32))
    _write_json(root / f"{bank}_sparse.json", [{str(k): v for k, v in r.items()} for r in sparse])


def _write_scores(root: Path, config: str, rows: list[tuple[str, str, float]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "query_hash": pa.array([r[0] for r in rows], pa.string()),
            "chunk_hash": pa.array([r[1] for r in rows], pa.string()),
            "score": pa.array([r[2] for r in rows], pa.float32()),
        }
    )
    pq.write_table(table, root / f"rerank_scores_{config}.parquet")


def _write_manifest(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    lines = []
    for p in files:
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {p.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- The job ---


class _CachedEmbedder:
    """Per-hash cache: identical texts (cards across configs) embed once."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.cache: dict[str, tuple[np.ndarray, dict[int, float]]] = {}

    def embed_records(
        self, records: list[dict[str, str]]
    ) -> tuple[list[str], np.ndarray, list[dict[int, float]]]:
        missing = [r for r in records if r["hash"] not in self.cache]
        if missing:
            dense, sparse = self.inner.embed([r["text"] for r in missing])
            for record, vec, weights in zip(missing, dense, sparse, strict=True):
                self.cache[record["hash"]] = (vec, weights)
        hashes = [r["hash"] for r in records]
        stacked = np.stack([self.cache[h][0] for h in hashes]) if hashes else np.zeros((0, 1))
        return hashes, stacked.astype(np.float32), [self.cache[h][1] for h in hashes]


def run_job(
    inputs: dict[str, Any],
    out_dir: Path,
    run_id: str,
    embedder: Embedder,
    scorer: PairScorer,
    *,
    stub: bool,
    batch_size: int = 8192,
) -> Path:
    root = out_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    cached = _CachedEmbedder(embedder)

    queries: list[dict[str, str]] = inputs["queries"]
    q_hashes, q_dense, q_sparse = cached.embed_records(queries)
    _write_bank(root, "queries", q_hashes, q_dense, q_sparse)

    counts: dict[str, int] = {}
    for config, chunks in sorted(inputs["configs"].items()):
        c_hashes, c_dense, c_sparse = cached.embed_records(chunks)
        _write_bank(root, f"corpus_{config}", c_hashes, c_dense, c_sparse)

        rows: list[tuple[str, str, float]] = []
        pairs = [(q, c) for q in queries for c in chunks]  # the FULL matrix (P2_SPEC §2.6)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            scores = scorer.score([(q["text"], c["text"]) for q, c in batch])
            rows.extend(
                (q["hash"], c["hash"], float(s))
                for (q, c), s in zip(batch, scores, strict=True)
            )
        _write_scores(root, config, rows)
        counts[config] = len(chunks)

    _write_json(
        root / "meta.json",
        {
            "run_id": run_id,
            "run_kind": inputs.get("run_kind", "retrieval_embed_v1"),
            # noqa comment: the GPU box may run Python 3.10 (datetime.UTC is 3.11+).
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
            "stub": stub,
            "embed_model": inputs["embed_model"],
            "rerank_model": inputs["rerank_model"],
            "score_transform": "sigmoid",
            "dense_dim": DENSE_DIM,
            "configs": sorted(inputs["configs"]),
            "corpus_counts": counts,
            "query_count": len(queries),
            "unique_texts_embedded": len(cached.cache),
            "inputs_sha256": inputs["_inputs_sha256"],
            "code_sha": inputs.get("code_sha"),
        },
    )
    _write_manifest(root)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stub", action="store_true", help="deterministic no-model dry run")
    args = parser.parse_args(argv)

    blob = args.inputs.read_text(encoding="utf-8")
    inputs: dict[str, Any] = json.loads(blob)
    inputs["_inputs_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    if args.stub:
        embedder: Embedder = StubEmbedder()
        scorer: PairScorer = StubScorer()
    else:
        embedder = BgeM3Embedder(inputs["embed_model"])
        scorer = BgeReranker(inputs["rerank_model"])

    root = run_job(inputs, args.out, args.run_id, embedder, scorer, stub=args.stub)
    print(f"sealed artifact run: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
