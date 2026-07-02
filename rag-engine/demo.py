"""The 30-second P2 demo (task 15): recorded golden queries → the full hybrid pipeline →
a cited, relationship-labelled version set. **Runs with the GPU OFF** — query embeddings
and reranker scores come from the sealed artifact run (the graceful-degradation story).

    make rag-demo          # replays GS-07a (Tanglish) + GS-03a (plot-only)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from sutradhar.config import Settings
from sutradhar.evals.golden import load_fixtures
from sutradhar.evals.retrieval import EvalRunArtifact
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.repository import get_versions, search_by_plot
from sutradhar.rag.artifacts import (
    DEFAULT_ARTIFACTS_ROOT,
    ArtifactEmbeddings,
    ArtifactReranker,
    ArtifactRun,
)
from sutradhar.rag.retrieve import RetrievalConfig, Retriever

app = typer.Typer(add_completion=False)

DEMO_QUERIES = ("GS-07a", "GS-03a")
REL_LABELS = {
    "is_original_of": "ORIGINAL",
    "is_remake_of": "remake",
    "is_official_dub_of": "official dub",
    "is_unofficial_remake_of": "unofficial remake",
    "is_sequel_of": "sequel",
}


def _winner_cell(runs_dir: Path, run_id: str) -> tuple[str, int]:
    artifact = EvalRunArtifact.model_validate_json(
        (runs_dir / f"{run_id}.json").read_text(encoding="utf-8")
    )
    assert artifact.winner is not None
    record = artifact.records[artifact.winner]
    return (
        str(record.retrieval_config["chunk_config"]),
        int(record.retrieval_config["rerank_depth"]),
    )


def _citation(session: Session, chunk_hash: str) -> str:
    row = session.execute(
        sqltext(
            "SELECT p.source, p.source_url, p.revision_id, p.license "
            "FROM chunks c JOIN plot_texts p ON p.plot_id = c.plot_id "
            "WHERE c.content_hash = :h LIMIT 1"
        ),
        {"h": chunk_hash},
    ).first()
    if row is None:  # metadata card — graph-derived
        return "graph record (Wikidata CC0 + TMDB attribution)"
    return f"{row.source_url} @ rev {row.revision_id} ({row.license})"


@app.command()
def main(
    run_id: str = typer.Option("", "--run-id", help="Artifact run id; default RETRIEVAL_RUN."),
    runs_dir: Path = typer.Option(Path("evals/retrieval_runs"), "--runs-dir"),  # noqa: B008
    artifacts_root: Path = typer.Option(DEFAULT_ARTIFACTS_ROOT, "--artifacts-root"),  # noqa: B008
) -> None:
    settings = Settings()
    resolved = run_id or settings.retrieval_run
    if not resolved:
        raise typer.BadParameter("no run id: pass --run-id or set RETRIEVAL_RUN in .env")

    chunk_config, depth = _winner_cell(runs_dir, resolved)
    run = ArtifactRun.open(artifacts_root, resolved)
    meta = json.loads(run.path("meta.json").read_text(encoding="utf-8"))
    fixtures = {f.id: f for f in load_fixtures()}

    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        retriever = Retriever(
            session,
            RetrievalConfig(
                chunk_config=chunk_config,
                embed_model=str(meta["embed_model"]),
                index_version=resolved,
                rerank_depth=depth,
            ),
            ArtifactEmbeddings(run, banks=("queries", f"corpus_{chunk_config}")),
            ArtifactReranker(run, chunk_config),
        )
        typer.echo(
            f"sutradhar rag-demo — run {resolved} · {chunk_config}/d{depth} · GPU: OFF "
            "(recorded artifacts)\n"
        )
        for fixture_id in DEMO_QUERIES:
            fixture = fixtures[fixture_id]
            query = fixture.query if isinstance(fixture.query, str) else fixture.query[0]
            typer.echo(f"❯ [{fixture_id}, {fixture.query_lang}] {query}")

            outcome = retriever.retrieve(query)
            result = search_by_plot(session, query, top_k=3, retriever=retriever)
            top = result.results[0]
            confidence = "LOW CONFIDENCE (below calibrated θ — shown, never invented)" \
                if result.abstain else "confident"
            typer.echo(f"  → {top.canonical_title} ({top.year}) — {confidence}")

            versions = get_versions(session, top.work_id)
            for v in versions.versions:
                label = REL_LABELS.get(v.relationship or "", v.relationship or "unlabelled")
                flag = "  ★" if v.is_original else ""
                typer.echo(f"     {label:<16} {v.title} ({v.language}, {v.year}){flag}")

            best_chunk = outcome.reranked_chunks[0][0]
            typer.echo(f"  cited: {_citation(session, best_chunk.content_hash)}\n")
    engine.dispose()


if __name__ == "__main__":
    app()
