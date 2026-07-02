"""Format-lock tests for the self-contained GPU job (P2 task 5): run the CLI with
``--stub`` (deterministic, no models) and prove ``sutradhar.rag.artifacts`` can load
everything it writes — the contract that keeps the shipped script and the laptop reader
from drifting apart."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from sutradhar.rag.artifacts import ArtifactEmbeddings, ArtifactRun
from sutradhar.rag.chunking import content_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "rag-engine" / "embed_and_score.py"

CARD = "Papanasam (Tamil, 2015) — remake of Drishyam (Malayalam, 2013). A Tamil film."
INPUTS = {
    "run_kind": "retrieval_embed_v1",
    "embed_model": "BAAI/bge-m3",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "code_sha": "deadbeef",
    "configs": {
        "256tok_15pct": [
            {"hash": content_hash("chunk one text."), "text": "chunk one text."},
            {"hash": content_hash("chunk two text."), "text": "chunk two text."},
            {"hash": content_hash(CARD), "text": CARD},  # card repeats across configs
        ],
        "512tok_15pct": [
            {
                "hash": content_hash("chunk one text. chunk two text."),
                "text": "chunk one text. chunk two text.",
            },
            {"hash": content_hash(CARD), "text": CARD},
        ],
    },
    "queries": [
        {
            "id": "GS-03a",
            "hash": content_hash("a man buries a body"),
            "text": "a man buries a body",
        },
        {"id": "NEG-01", "hash": content_hash("a mermaid in Kochi"), "text": "a mermaid in Kochi"},
    ],
}


def _run(tmp_path: Path, run_id: str = "stub-run") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = tmp_path / "gpu_inputs.json"
    inputs.write_text(json.dumps(INPUTS), encoding="utf-8")
    out = tmp_path / "artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inputs",
            str(inputs),
            "--out",
            str(out),
            "--run-id",
            run_id,
            "--stub",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return out


def test_stub_run_is_a_sealed_loadable_artifact(tmp_path: Path) -> None:
    out = _run(tmp_path)
    run = ArtifactRun.open(out, "stub-run")  # MANIFEST verifies
    provider = ArtifactEmbeddings(
        run, banks=("queries", "corpus_256tok_15pct", "corpus_512tok_15pct")
    )
    # Every input text is servable by lookup — incl. the card shared across configs.
    rows = provider.embed(["a man buries a body", CARD, "chunk one text."])
    assert all(np.isclose(np.linalg.norm(r.dense), 1.0, atol=1e-5) for r in rows)
    assert all(r.sparse for r in rows)


def test_full_rerank_matrix_per_config(tmp_path: Path) -> None:
    out = _run(tmp_path)
    for config, chunks in INPUTS["configs"].items():
        table = pq.read_table(out / "stub-run" / f"rerank_scores_{config}.parquet")
        assert table.num_rows == len(INPUTS["queries"]) * len(chunks)  # FULL matrix
        pairs = set(
            zip(table["query_hash"].to_pylist(), table["chunk_hash"].to_pylist(), strict=True)
        )
        assert len(pairs) == table.num_rows  # no dupes
        scores = table["score"].to_pylist()
        assert all(0.0 <= s <= 1.0 for s in scores)  # sigmoid-normalized domain


def test_meta_records_reproducibility_stamp(tmp_path: Path) -> None:
    out = _run(tmp_path)
    meta = json.loads((out / "stub-run" / "meta.json").read_text(encoding="utf-8"))
    assert meta["embed_model"] == "BAAI/bge-m3"
    assert meta["rerank_model"] == "BAAI/bge-reranker-v2-m3"
    assert meta["code_sha"] == "deadbeef"
    assert meta["score_transform"] == "sigmoid"
    assert meta["stub"] is True
    assert meta["configs"] == ["256tok_15pct", "512tok_15pct"]
    assert meta["query_count"] == 2
    assert len(meta["inputs_sha256"]) == 64
    # The dedupe cache worked: 4 unique corpus texts + 2 queries, not 5 + 2.
    assert meta["unique_texts_embedded"] == 6


def test_stub_outputs_are_deterministic(tmp_path: Path) -> None:
    out_a = _run(tmp_path / "a")
    out_b = _run(tmp_path / "b")
    for name in ("queries_dense.npy", "corpus_256tok_15pct_dense.npy"):
        assert (out_a / "stub-run" / name).read_bytes() == (out_b / "stub-run" / name).read_bytes()
    scores_a = pq.read_table(out_a / "stub-run" / "rerank_scores_512tok_15pct.parquet")
    scores_b = pq.read_table(out_b / "stub-run" / "rerank_scores_512tok_15pct.parquet")
    assert scores_a.equals(scores_b)


def test_script_is_self_contained() -> None:
    """DEC-P2-7: the shipped file must not import sutradhar (no repo on the box)."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import sutradhar" not in source and "from sutradhar" not in source


def test_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    _run(tmp_path)
    inputs = tmp_path / "gpu_inputs.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inputs",
            str(inputs),
            "--out",
            str(tmp_path / "artifacts"),
            "--run-id",
            "stub-run",
            "--stub",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0  # run dirs are immutable once written


@pytest.mark.parametrize("bank", ["queries", "corpus_256tok_15pct"])
def test_bank_rows_are_hash_aligned(tmp_path: Path, bank: str) -> None:
    out = _run(tmp_path)
    hashes = json.loads((out / "stub-run" / f"{bank}_hashes.json").read_text("utf-8"))
    dense = np.load(out / "stub-run" / f"{bank}_dense.npy")
    sparse = json.loads((out / "stub-run" / f"{bank}_sparse.json").read_text("utf-8"))
    assert len(hashes) == dense.shape[0] == len(sparse)
    assert all(len(h) == 64 for h in hashes)
