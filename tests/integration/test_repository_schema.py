"""Integration half of tool-schema conformance (P1 task 15): repository RESULT shapes
round-trip through the frozen JSON Schema — real calls on the reviewed fixture chain,
serialized, validated. The "contract is satisfiable" proof."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from sutradhar.graph.db import postgres_url
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.repository import (
    RefineBy,
    get_versions,
    get_work,
    refine_filter,
    resolve_title,
)
from sutradhar.graph.schema import Version, Work
from sutradhar.pipeline.build import build_graph
from sutradhar.pipeline.extract import load_candidates
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice
from sutradhar.pipeline.titles import upsert_version_title
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity

from .ci_review_pass import apply_ci_review_pass

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SCHEMA = json.loads(
    (REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json").read_text(encoding="utf-8")
)


def _result_validator(tool: str) -> Draft202012Validator:
    sub = dict(SCHEMA["tools"][tool]["result"])
    sub["$defs"] = SCHEMA["$defs"]
    return Draft202012Validator(sub)


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
def reviewed(engine: Engine) -> Iterator[Session]:
    with engine.connect() as conn:
        outer = conn.begin()
        s = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
        for table in (
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
        slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
        wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
        ingest_spine(s, slice_, {q: parse_entity(e) for q, e in wd.items()})
        tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
        enrich_tmdb(s, {int(k): parse_movie(v) for k, v in tm.items()})
        for v in s.scalars(select(Version)).all():
            upsert_version_title(
                s,
                v.version_id,
                v.title,
                "canonical",
                v.language,
                [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
            )
        build_graph(s)
        art = json.loads((FIXTURES / "extraction" / "outputs_sample.json").read_text("utf-8"))
        load_candidates(s, art["raw_outputs"], art["pages"], art["model_id"])
        apply_ci_review_pass(s)
        try:
            yield s
        finally:
            s.close()
            outer.rollback()


def _assert_valid(tool: str, payload: dict[str, Any]) -> None:
    errors = [e.message for e in _result_validator(tool).iter_errors(payload)]
    assert errors == [], f"{tool} result violates frozen schema: {errors[:3]}"


def test_resolve_title_result_conforms(reviewed: Session) -> None:
    for query in ("Papanasam", "Vikram", "ദൃശ്യം"):
        result = resolve_title(reviewed, query)
        _assert_valid("resolve_title", result.model_dump(mode="json"))


def test_get_work_result_conforms(reviewed: Session) -> None:
    for title in ("Devadasu", "Drishyam", "Vikram"):
        work = reviewed.scalars(select(Work).where(Work.primary_title == title)).first()
        assert work is not None
        result = get_work(reviewed, work.work_id)
        assert result is not None
        _assert_valid("get_work", result.model_dump(mode="json"))


def test_get_versions_result_conforms(reviewed: Session) -> None:
    drishyam = reviewed.scalars(select(Work).where(Work.primary_title == "Drishyam")).one()
    for scope, sequels in (("indian", False), ("foreign", False), ("all", True)):
        result = get_versions(reviewed, drishyam.work_id, scope=scope, include_sequels=sequels)  # type: ignore[arg-type]
        _assert_valid("get_versions", result.model_dump(mode="json"))


def test_refine_filter_result_conforms(reviewed: Session) -> None:
    drishyam = reviewed.scalars(select(Work).where(Work.primary_title == "Drishyam")).one()
    versions = get_versions(reviewed, drishyam.work_id, scope="indian").versions
    ids = [v.version_id for v in versions]
    for by in (RefineBy(actor="Ajay Devgn"), RefineBy(era="original"), RefineBy(language="te")):
        result = refine_filter(reviewed, ids, by)
        _assert_valid("refine_filter", result.model_dump(mode="json"))
