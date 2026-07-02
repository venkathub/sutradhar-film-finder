"""Named golden-set graph regressions (P1_SPEC §4) — the tests this phase must ship.

Task 9 lands GS-04 / GS-05 / GS-10 (+ the rule-conflict gate); GS-01/02/06/09 join in
task 10 with the repository layer. Full fixture-driven chain: spine → TMDB → akas → build.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import Edge, Version, VersionCast, Work
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.imdb import load_akas, parse_aka_line
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

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
def built(session: Session) -> Session:
    """The fixture-driven pipeline through build_graph (tasks 4/5/6 + 9)."""
    slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
    wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
    ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
    tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
    enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
    akas_lines = (FIXTURES / "imdb" / "akas_sample.tsv").read_text("utf-8").splitlines()
    load_akas(session, [r for r in (parse_aka_line(ln) for ln in akas_lines) if r])
    build_graph(session)
    return session


def _work_versions(session: Session, primary_title: str) -> tuple[Work, list[Version]]:
    work = session.scalars(select(Work).where(Work.primary_title == primary_title)).one()
    versions = session.scalars(select(Version).where(Version.work_id == work.work_id)).all()
    return work, list(versions)


def _edges_within(session: Session, version_ids: set[uuid.UUID]) -> list[Edge]:
    return [
        e
        for e in session.scalars(select(Edge)).all()
        if e.src_id in version_ids and e.dst_id in version_ids
    ]


# --- GS-04: dub vs remake never conflated ---


def test_gs04_dub_vs_remake(built: Session) -> None:
    work, versions = _work_versions(built, "Baahubali: The Beginning")
    vids = {v.version_id for v in versions}
    inside = _edges_within(built, vids)
    # Every language track is a dub edge; ZERO remake edges inside the Work.
    assert inside and all(e.edge_type == "is_official_dub_of" for e in inside)
    # Bilingual double-original encoded (te + ta), dubs point at the primary (te).
    originals = {v.language for v in versions if v.is_original}
    assert originals == {"te", "ta"}
    primary = next(v for v in versions if v.is_original and v.wikidata_qid)
    assert all(e.dst_id == primary.version_id for e in inside)
    # Dub edges are honest about their origin: rule-sourced, MEDIUM, gate-visible.
    for e in inside:
        assert e.confidence == "MEDIUM" and e.sources[0]["source"] == "rule"


# --- GS-05: shared literary source ≠ remake chain ---


def test_gs05_sibling_vs_remake(built: Session) -> None:
    novella = built.scalars(select(Work).where(Work.work_type == "literary_source")).one()
    sibling_works = built.scalars(
        select(Work).where(Work.primary_title.in_(["Devdas", "Devadasu"]))
    ).all()
    sibling_works = [w for w in sibling_works if w.work_type == "film"]
    assert len(sibling_works) >= 2  # fixture carries Devdas 2002 + Devadasu 1953
    # Each sibling work is based_on the novella (work-level edge)…
    based_on = [
        e
        for e in built.scalars(select(Edge).where(Edge.edge_type == "based_on")).all()
        if e.dst_id == novella.work_id
    ]
    assert {e.src_id for e in based_on} >= {w.work_id for w in sibling_works}
    # …and ZERO is_remake_of edges exist between versions of different sibling works.
    all_sibling_versions: dict[uuid.UUID, uuid.UUID] = {}
    for w in sibling_works:
        for v in built.scalars(select(Version).where(Version.work_id == w.work_id)):
            all_sibling_versions[v.version_id] = w.work_id
    for e in built.scalars(select(Edge).where(Edge.edge_type == "is_remake_of")).all():
        if e.src_id in all_sibling_versions and e.dst_id in all_sibling_versions:
            assert all_sibling_versions[e.src_id] == all_sibling_versions[e.dst_id], (
                "remake edge crosses sibling adaptations — GS-05 violation"
            )
    # The dub edge composes INSIDE a sibling (Devadas ta → Devadasu te).
    devadasu_versions = {
        v.version_id
        for w in sibling_works
        if w.primary_title == "Devadasu"
        for v in built.scalars(select(Version).where(Version.work_id == w.work_id))
    }
    assert any(e.edge_type == "is_official_dub_of" for e in _edges_within(built, devadasu_versions))


# --- GS-10: same title + same actor ≠ same Work (false-merge rate = 0) ---


def test_gs10_false_merge(built: Session) -> None:
    vikram_works = built.scalars(select(Work).where(Work.primary_title == "Vikram")).all()
    assert len(vikram_works) == 2  # 1986 + 2022 stay distinct
    years = {w.first_release_year for w in vikram_works}
    assert years == {1986, 2022}
    version_sets = [
        {v.version_id for v in built.scalars(select(Version).where(Version.work_id == w.work_id))}
        for w in vikram_works
    ]
    assert version_sets[0].isdisjoint(version_sets[1])
    # No edges of any type between the two Vikram works' versions.
    assert _edges_within(built, version_sets[0] | version_sets[1]) == []


# --- Rule-vs-edge disagreement gate (never a silent re-type) ---


def test_rule_disagreement_opens_conflict_and_hides_edge(built: Session) -> None:
    """Doctor a remake edge to share lead cast → builder must open an edge_type conflict
    (not re-type it), and the gate view must hide the edge until resolution."""
    # Deterministic pick: a remake edge whose dst actually has lead cast (unordered
    # .first() depended on heap order and went flaky once more tables joined the DB).
    remake = next(
        (
            e
            for e in built.scalars(
                select(Edge).where(Edge.edge_type == "is_remake_of").order_by(Edge.edge_id)
            )
            if built.scalars(
                select(VersionCast).where(
                    VersionCast.version_id == e.dst_id, VersionCast.role_kind == "lead"
                )
            ).first()
            is not None
        ),
        None,
    )
    assert remake is not None
    dst_leads = built.scalars(
        select(VersionCast).where(
            VersionCast.version_id == remake.dst_id, VersionCast.role_kind == "lead"
        )
    ).all()
    assert dst_leads
    # Copy the dst's leads onto the src version → rule now says "dub".
    for lead in dst_leads:
        if built.get(VersionCast, (remake.src_id, lead.person_id, "lead")) is None:
            built.add(
                VersionCast(
                    version_id=remake.src_id,
                    person_id=lead.person_id,
                    role_kind="lead",
                    sources=[{"source": "human", "ref": "test-doctored"}],
                )
            )
    # Remove the src's own leads so overlap dominates.
    for row in built.scalars(
        select(VersionCast).where(
            VersionCast.version_id == remake.src_id, VersionCast.role_kind == "lead"
        )
    ).all():
        if row.person_id not in {lead.person_id for lead in dst_leads}:
            built.delete(row)
    built.flush()

    report = build_graph(built)
    assert report.rule_conflicts_opened == 1
    assert remake.edge_type == "is_remake_of"  # NOT silently re-typed
    visible = built.execute(
        text("SELECT count(*) FROM ground_truth_edges WHERE edge_id = :e"),
        {"e": str(remake.edge_id)},
    ).scalar_one()
    assert visible == 0  # hidden until a human resolves
    # Idempotent: the same disagreement is not re-queued.
    again = build_graph(built)
    assert again.rule_conflicts_opened == 0


def test_build_graph_idempotent(built: Session) -> None:
    first = build_graph(built)  # `built` already ran it once
    assert first.dub_edges_derived == 0 and first.rule_conflicts_opened == 0
    assert first.rule_agreements > 0  # wikidata remake edges confirmed by disjoint casts
    assert first.anomalies == []


# --- Task 10: repository-backed named regressions (GS-02 / GS-06 / GS-09) ---


def test_gs02_no_hallucinated_movie(built: Session) -> None:
    """Decoy titles resolve to NOTHING in the graph — nothing is returnable for them."""
    from sutradhar.graph.repository import resolve_title

    for decoy in ("Inception", "Kaithi", "Pather Panchali", "Minnal Murali"):
        result = resolve_title(built, decoy)
        assert result.candidates == [], f"{decoy!r} must not resolve"
        assert result.ambiguous is False


def test_gs06_franchise_version_set_recall(built: Session) -> None:
    """include_sequels traverses is_sequel_of; sequel vs remake labels never conflated."""
    from sutradhar.graph.repository import get_versions

    work, _ = _work_versions(built, "Drishyam")
    result = get_versions(built, work.work_id, scope="indian", include_sequels=True)
    titles_years = {(v.title, v.year) for v in result.versions}
    # Curated truth: 5 Drishyam-1 Indian versions + 4 Drishyam-2 versions → recall = 1.0.
    expected = {
        ("Drishyam", 2013),
        ("Drishya", 2014),
        ("Drushyam", 2014),
        ("Papanasam", 2015),
        ("Drishyam", 2015),
        ("Drishyam 2", 2021),
        ("Drishya 2", 2021),
        ("Drushyam 2", 2021),
        ("Drishyam 2", 2022),
    }
    assert titles_years == expected, f"version-set recall < 1.0: {titles_years ^ expected}"
    by_key = {(v.title, v.year): v for v in result.versions}
    # The queried work's original is is_original_of; the SEQUEL work's own original is
    # is_sequel_of; the sequel's hi remake keeps is_remake_of (never conflated).
    assert by_key[("Drishyam", 2013)].relationship == "is_original_of"
    assert by_key[("Drishyam 2", 2021)].relationship == "is_sequel_of"
    assert by_key[("Drishyam 2", 2022)].relationship == "is_remake_of"
    assert result.original is not None and result.original.year == 2013


def test_gs09_scoping(built: Session) -> None:
    """Foreign versions exist but are excluded at scope=indian, returned at scope=foreign."""
    from sutradhar.graph.repository import get_versions

    work, _ = _work_versions(built, "Drishyam")
    indian = get_versions(built, work.work_id, scope="indian")
    foreign = get_versions(built, work.work_id, scope="foreign")
    everything = get_versions(built, work.work_id, scope="all")
    assert {v.language for v in indian.versions} == {"ml", "kn", "te", "ta", "hi"}
    assert {v.language for v in foreign.versions} == {"si", "zh"}
    assert len(everything.versions) == len(indian.versions) + len(foreign.versions)
    # The foreign zh remake carries its verified label (Wikidata P144 in the fixture).
    zh = next(v for v in foreign.versions if v.language == "zh")
    assert zh.relationship == "is_remake_of" and zh.is_original is False


def test_gs09_transitive_lineage(built: Session) -> None:
    """Full Manichitrathazhu lineage returned with the ml original SOLE is_original —
    'the original of Bhool Bhulaiyaa' resolves regardless of edge depth. (The proximate
    Chandramukhi→Apthamitra edge itself arrives via extraction+review; asserted in the
    task-14 golden fixtures.)"""
    from sutradhar.graph.repository import get_versions

    work, _ = _work_versions(built, "Manichitrathazhu")
    result = get_versions(built, work.work_id, scope="indian")
    assert {v.title for v in result.versions} == {
        "Manichitrathazhu",
        "Apthamitra",
        "Chandramukhi",
        "Rajmohol",
        "Bhool Bhulaiyaa",
    }
    originals = [v for v in result.versions if v.is_original]
    assert len(originals) == 1 and originals[0].title == "Manichitrathazhu"
    assert result.original is not None and result.original.language == "ml"
    # Chandramukhi's fixture edge (Wikidata, direct-to-ml) is remake-typed — never original.
    chandramukhi = next(v for v in result.versions if v.title == "Chandramukhi")
    assert chandramukhi.relationship == "is_remake_of" and not chandramukhi.is_original
