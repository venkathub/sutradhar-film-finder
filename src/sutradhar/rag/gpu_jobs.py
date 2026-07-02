"""GPU-job input export (P2 task 5): everything the embed+score session needs, as one JSON.

The GPU instance has no Postgres and no repo checkout (DEC-P2-7 HF-relay): the laptop
exports ``gpu_inputs.json`` — corpus texts per chunk config (from the gate-visible
``chunks`` table) + every golden-fixture query turn + every held-out negative query —
and the self-contained ``rag-engine/embed_and_score.py`` consumes it on the box.
Text records are ``{hash, text}`` with ``hash = sha256(text)``: the same key
``ArtifactEmbeddings`` uses at lookup time, so exported rows and recorded vectors can
never drift apart.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.config import Settings
from sutradhar.evals.golden import GOLDEN_DIR, load_fixtures
from sutradhar.evals.negatives import NEGATIVES_PATH, load_negatives
from sutradhar.graph.schema import Chunk
from sutradhar.rag.chunking import content_hash

RUN_KIND = "retrieval_embed_v1"


def git_sha() -> str | None:
    """Current commit SHA for the reproducibility stamp (None outside a git checkout)."""
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no user input
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def collect_query_records(
    golden_dir: Path = GOLDEN_DIR, negatives_path: Path = NEGATIVES_PATH
) -> list[dict[str, str]]:
    """Every golden query turn + every held-out negative, deterministically ordered.

    All GS categories are exported (not just the retrieval slices): GS-02 gates
    abstention, GS-04/05/10 run named regressions *through* retrieval, and embedding a
    handful of extra queries is free next to the corpus pass.
    """
    records: list[dict[str, str]] = []
    for fixture in load_fixtures(golden_dir):
        turns = fixture.query if isinstance(fixture.query, list) else [fixture.query]
        for i, turn in enumerate(turns):
            query_id = fixture.id if len(turns) == 1 else f"{fixture.id}#turn{i}"
            records.append({"id": query_id, "hash": content_hash(turn), "text": turn})
    for negative in load_negatives(negatives_path):
        records.append(
            {"id": negative.id, "hash": content_hash(negative.query), "text": negative.query}
        )
    records.sort(key=lambda r: r["id"])
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate query ids in golden+negative export")
    return records


def collect_corpus_records(session: Session) -> dict[str, list[dict[str, str]]]:
    """Per-config chunk texts (plot chunks + metadata cards), deduped and hash-ordered."""
    rows = session.execute(
        select(Chunk.chunk_config, Chunk.content_hash, Chunk.text).order_by(
            Chunk.chunk_config, Chunk.content_hash
        )
    ).all()
    configs: dict[str, dict[str, str]] = {}
    for config, chunk_hash, text in rows:
        if content_hash(text) != chunk_hash:
            raise ValueError(f"chunk {chunk_hash[:12]}… text/hash mismatch — rebuild the corpus")
        configs.setdefault(config, {})[chunk_hash] = text
    return {
        config: [{"hash": h, "text": t} for h, t in sorted(texts.items())]
        for config, texts in configs.items()
    }


def export_gpu_inputs(session: Session, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    configs = collect_corpus_records(session)
    if not configs:
        raise ValueError("no chunks in the DB — run `make build-corpus` first")
    return {
        "run_kind": RUN_KIND,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "code_sha": git_sha(),
        "configs": configs,
        "queries": collect_query_records(),
    }


def write_gpu_inputs(
    session: Session, path: Path, settings: Settings | None = None
) -> tuple[Path, dict[str, Any]]:
    inputs = export_gpu_inputs(session, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )
    return path, inputs
