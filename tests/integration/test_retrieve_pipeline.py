"""Integration: the full §2.4 pipeline round-trip (P2 task 9) — reviewed graph + corpus
→ stub artifact run (the real embed_and_score.py CLI in --stub mode) → load_index →
Retriever → search_by_plot v0 conformance → get_versions join. Stub vectors are
hash-random, so these tests assert FLOW, SHAPE, and DETERMINISM — ranking quality is
task 10's job, measured on the real recorded run."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.repository import get_versions, search_by_plot
from sutradhar.graph.schema import Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.imdb import load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.pipeline.wikipedia import WikiPage, load_plots, parse_page
from sutradhar.rag.artifacts import (
    ArtifactEmbeddings,
    ArtifactReranker,
    ArtifactRun,
    MissingArtifactError,
)
from sutradhar.rag.corpus import build_corpus
from sutradhar.rag.gpu_jobs import write_gpu_inputs
from sutradhar.rag.index import load_index
from sutradhar.rag.retrieve import RetrievalConfig, Retriever

from .ci_review_pass import apply_ci_review_pass

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SCHEMA = json.loads((REPO_ROOT / "docs/phases/tool_schema.v0.json").read_text("utf-8"))
RUN_ID = "stub-pipeline-run"
CONFIG = "512tok_15pct"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except Exception as exc:  # noqa: BLE001 — connection probe
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    with engine.connect() as conn:
        outer = conn.begin()
        s = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
        for table in (
            "chunk_embeddings",
            "chunks",
            "candidate_edges",
            "edges",
            "conflicts",
            "plot_texts",
            "version_cast",
            "version_title",
            "version",
            "person",
            "work",
        ):
            s.execute(text(f"DELETE FROM {table}"))
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


@pytest.fixture()
def retriever(session: Session, tmp_path: Path) -> Retriever:
    """Reviewed graph + plots + corpus → stub run via the REAL CLI → loaded index."""
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
    akas_lines = (FIXTURES / "imdb" / "akas_sample.tsv").read_text("utf-8").splitlines()
    load_akas(session, [r for r in (parse_aka_line(ln) for ln in akas_lines) if r])
    for v in session.scalars(select(Version)).all():
        upsert_version_title(
            session,
            v.version_id,
            v.title,
            "canonical",
            v.language,
            [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
        )
    build_graph(session)
    art = json.loads((FIXTURES / "extraction" / "outputs_sample.json").read_text("utf-8"))
    load_candidates(session, art["raw_outputs"], art["pages"], art["model_id"])
    apply_ci_review_pass(session)
    raw = json.loads((FIXTURES / "wikipedia" / "pages_sample.json").read_text("utf-8"))
    pages: dict[str, list[WikiPage]] = {}
    for key, entry in raw.items():
        page = parse_page(entry["lang"], entry["response"])
        assert page is not None
        pages.setdefault(key.split("|", 1)[0], []).append(page)
    load_plots(session, pages)
    build_corpus(session)

    inputs_path, _ = write_gpu_inputs(session, tmp_path / "gpu_inputs.json")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "rag-engine/embed_and_score.py"),
            "--inputs",
            str(inputs_path),
            "--out",
            str(tmp_path / "artifacts"),
            "--run-id",
            RUN_ID,
            "--stub",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    load_index(session, tmp_path / "artifacts", RUN_ID)

    run = ArtifactRun.open(tmp_path / "artifacts", RUN_ID)
    meta = json.loads(run.path("meta.json").read_text("utf-8"))
    return Retriever(
        session,
        RetrievalConfig(chunk_config=CONFIG, embed_model=meta["embed_model"], index_version=RUN_ID),
        ArtifactEmbeddings(run, banks=("queries", f"corpus_{CONFIG}")),
        ArtifactReranker(run, CONFIG),
    )


GS03A = (
    "a schoolteacher with no formal education outwits the police to protect his family "
    "after his daughter accidentally kills a blackmailer"
)


def _golden_query(fixture_id: str) -> str:
    from sutradhar.evals.golden import load_fixtures

    fixture = next(f for f in load_fixtures(REPO_ROOT / "evals/golden") if f.id == fixture_id)
    return fixture.query if isinstance(fixture.query, str) else fixture.query[0]


def test_full_pipeline_roundtrip_shape_and_conformance(retriever: Retriever) -> None:
    query = _golden_query("GS-03a")
    result = search_by_plot(retriever.session, query, top_k=10, retriever=retriever)
    assert result.results, "recorded golden query must retrieve something"
    sub = dict(SCHEMA["tools"]["search_by_plot"]["result"])
    sub["$defs"] = SCHEMA["$defs"]
    payload = json.loads(result.model_dump_json())
    errors = [e.message for e in Draft202012Validator(sub).iter_errors(payload)]
    assert errors == [], errors  # the emitted call validates against frozen v0


def test_top_work_joins_get_versions(retriever: Retriever) -> None:
    """§2.4 tail: retrieval → get_versions → typed, original-flagged version set."""
    result = search_by_plot(
        retriever.session, _golden_query("GS-03a"), top_k=1, retriever=retriever
    )
    assert len(result.results) == 1
    versions = get_versions(retriever.session, result.results[0].work_id)
    assert versions.versions  # the joined Work has a version set


def test_pipeline_is_deterministic(retriever: Retriever) -> None:
    query = _golden_query("GS-07a")
    first = search_by_plot(retriever.session, query, retriever=retriever)
    second = search_by_plot(retriever.session, query, retriever=retriever)
    assert first == second


def test_all_channels_contribute(retriever: Retriever) -> None:
    """A title-ish query lights the title channel; a plot query lights dense+sparse."""
    outcome = retriever.retrieve(_golden_query("GS-11a"))
    assert outcome.channel_sizes["title"] > 0
    plot_outcome = retriever.retrieve(_golden_query("GS-03a"))
    assert plot_outcome.channel_sizes["dense"] > 0
    assert plot_outcome.channel_sizes["sparse"] > 0


def test_unrecorded_query_never_degrades(retriever: Retriever) -> None:
    with pytest.raises(MissingArtifactError, match="no recorded embedding"):
        retriever.retrieve("a query the GPU session never saw")


def test_rerank_depth_bounds_rerank_calls(retriever: Retriever) -> None:
    outcome = retriever.retrieve(_golden_query("GS-03a"))
    assert len(outcome.reranked_chunks) <= retriever.config.rerank_depth
    scores = [s for _, s in outcome.reranked_chunks]
    assert scores == sorted(scores, reverse=True)  # rerank order is score-desc
