"""Integration: corpus builder over the gate-visible graph (P2 task 3).

Full fixture chain (spine + TMDB cast + IMDb AKAs + canonical titles + build + CI review
pass + Wikipedia plots) → ``build_corpus`` → chunks for every ablation config, headers
carrying lineage, metadata cards, gate-visibility (conflict-hidden version → zero chunks),
and rebuild idempotency.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Chunk, Version
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.imdb import load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity
from sutradhar.pipeline.wikipedia import WikiPage, load_plots, parse_page
from sutradhar.rag.chunking import CHUNK_CONFIGS
from sutradhar.rag.corpus import build_corpus

from .ci_review_pass import apply_ci_review_pass

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


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
def corpus_ready(session: Session) -> Session:
    """Reviewed graph + plots: the state `make build-corpus` runs against."""
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
        qid = key.split("|", 1)[0]
        page = parse_page(entry["lang"], entry["response"])
        assert page is not None
        pages.setdefault(qid, []).append(page)
    load_plots(session, pages)
    return session


def _version(session: Session, title: str, language: str) -> Version:
    return session.scalars(
        select(Version).where(Version.title == title, Version.language == language)
    ).one()


def test_corpus_covers_every_config_and_version(corpus_ready: Session) -> None:
    report = build_corpus(corpus_ready)
    assert report.versions_seen > 20  # the seed slice is gate-visible
    for config in CHUNK_CONFIGS:
        cards = corpus_ready.execute(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE kind = 'metadata_card' AND chunk_config = :c"
            ),
            {"c": config.name},
        ).scalar_one()
        assert cards == report.versions_seen  # one card per gate-visible version per config
        plots = corpus_ready.execute(
            text("SELECT count(*) FROM chunks WHERE kind = 'plot' AND chunk_config = :c"),
            {"c": config.name},
        ).scalar_one()
        assert plots == report.plot_chunks[config.name] > 0
    # Smaller targets ⇒ at least as many chunks (ablation grid is really varying size).
    n256, n512, n1024 = (report.plot_chunks[c.name] for c in CHUNK_CONFIGS)
    assert n256 >= n512 >= n1024


def test_every_plot_chunk_carries_its_header(corpus_ready: Session) -> None:
    build_corpus(corpus_ready)
    drishyam = _version(corpus_ready, "Drishyam", "ml")
    rows = corpus_ready.scalars(
        select(Chunk).where(
            Chunk.version_id == drishyam.version_id,
            Chunk.kind == "plot",
            Chunk.chunk_config == "512tok_15pct",
        )
    ).all()
    assert rows  # en + ml Wikipedia plots exist for Drishyam in the fixture
    for chunk in rows:
        assert chunk.text.startswith("Drishyam (Malayalam, 2013). ")
        assert chunk.license == "CC BY-SA 4.0"  # carried from plot_texts
        assert chunk.plot_id is not None
    assert {c.language for c in rows} == {"en", "ml"}  # native-script plots embedded as-is


def test_remake_lineage_in_header_and_card(corpus_ready: Session) -> None:
    """The Papanasam→Drishyam lineage rides every embedded unit (P2_SPEC §2.2)."""
    build_corpus(corpus_ready)
    papanasam = _version(corpus_ready, "Papanasam", "ta")
    card = corpus_ready.scalars(
        select(Chunk).where(
            Chunk.version_id == papanasam.version_id,
            Chunk.kind == "metadata_card",
            Chunk.chunk_config == "512tok_15pct",
        )
    ).one()
    assert "— remake of Drishyam (Malayalam, 2013)" in card.text
    assert card.plot_id is None and card.seq == 0
    # A dub track must say dub, never remake (GS-04 semantics reach the corpus).
    hindi_dub = _version(corpus_ready, "Baahubali - The Beginning", "hi")
    dub_card = corpus_ready.scalars(
        select(Chunk).where(
            Chunk.version_id == hindi_dub.version_id,
            Chunk.kind == "metadata_card",
            Chunk.chunk_config == "512tok_15pct",
        )
    ).one()
    assert "official dub of" in dub_card.text
    assert "remake of" not in dub_card.text


def test_conflict_hidden_version_yields_zero_chunks(corpus_ready: Session) -> None:
    """The verification gate holds through the corpus: conflict-hidden → not embedded."""
    drishyam = _version(corpus_ready, "Drishyam", "ml")
    corpus_ready.execute(
        text(
            "INSERT INTO conflicts (entity_kind, entity_id, field, values) "
            "VALUES ('version', :v, 'release_year', '[{\"value\": 2013}, {\"value\": 2014}]')"
        ),
        {"v": str(drishyam.version_id)},
    )
    build_corpus(corpus_ready)
    count = corpus_ready.execute(
        text("SELECT count(*) FROM chunks WHERE version_id = :v"),
        {"v": str(drishyam.version_id)},
    ).scalar_one()
    assert count == 0  # excluded by construction — the view never surfaced it


def test_gpu_inputs_export_matches_db(corpus_ready: Session) -> None:
    """P2 task 5: the exported inputs mirror the chunks table, hash-verified."""
    from sutradhar.rag.gpu_jobs import export_gpu_inputs

    build_corpus(corpus_ready)
    inputs = export_gpu_inputs(corpus_ready)
    assert set(inputs["configs"]) == {c.name for c in CHUNK_CONFIGS}
    db_hashes = {
        row[0]
        for row in corpus_ready.execute(
            text("SELECT content_hash FROM chunks WHERE chunk_config = '512tok_15pct'")
        )
    }
    exported = {r["hash"] for r in inputs["configs"]["512tok_15pct"]}
    assert exported == db_hashes
    assert inputs["embed_model"] and inputs["rerank_model"]
    assert any(r["id"].startswith("NEG-") for r in inputs["queries"])


def test_rebuild_is_idempotent(corpus_ready: Session) -> None:
    build_corpus(corpus_ready)

    def snapshot() -> list[tuple[str, str, int, str]]:
        return sorted(
            corpus_ready.execute(
                text("SELECT kind, chunk_config, seq, content_hash FROM chunks")
            ).all()  # type: ignore[arg-type]
        )

    first = snapshot()
    build_corpus(corpus_ready)
    assert snapshot() == first  # delete-and-reinsert reproduces identical content
