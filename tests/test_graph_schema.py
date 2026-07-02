"""Hermetic unit tests for the graph schema module (no DB, no Docker).

The real constraint behaviour is exercised against Postgres in
``tests/integration/test_graph_schema.py``; here we pin the schema *inventory* (tables, enums,
CHECK/unique constraints, gate-relevant columns) so an accidental model edit fails fast in
unit CI, and we verify the env-driven DSN builder.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from sutradhar.config import Settings
from sutradhar.graph import postgres_url
from sutradhar.graph.schema import (
    EDGE_TYPES,
    GROUND_TRUTH_VIEWS,
    VERSION_EDGE_TYPES,
    WORK_EDGE_TYPES,
    Base,
)

EXPECTED_TABLES = {
    "work",
    "version",
    "version_title",
    "person",
    "version_cast",
    "edges",
    "candidate_edges",
    "conflicts",
    "plot_texts",
}


def _table(name: str) -> Table:
    return Base.metadata.tables[name]


def _constraint_names(table: Table) -> set[str]:
    return {c.name for c in table.constraints if isinstance(c, CheckConstraint | UniqueConstraint)}


def test_all_spec_tables_present() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_edge_type_enum_matches_spec() -> None:
    """The five stored edge types — is_original_of is derived, never stored (P1_SPEC §2.2)."""
    assert set(EDGE_TYPES) == {
        "is_remake_of",
        "is_official_dub_of",
        "is_unofficial_remake_of",
        "is_sequel_of",
        "based_on",
    }
    assert "is_original_of" not in EDGE_TYPES
    assert set(VERSION_EDGE_TYPES) | set(WORK_EDGE_TYPES) == set(EDGE_TYPES)
    assert not set(VERSION_EDGE_TYPES) & set(WORK_EDGE_TYPES)


def test_edges_table_constraints() -> None:
    names = _constraint_names(_table("edges"))
    for expected in (
        "ck_edges_edge_type",
        "ck_edges_type_shape",
        "ck_edges_no_self_edge",
        "ck_edges_confidence",
        "uq_edges_edge_type_src_dst",
    ):
        assert expected in names, f"missing constraint {expected}"


def test_gated_tables_carry_provenance_columns() -> None:
    """Every gated record/edge carries confidence + sources + human_verified (P1_SPEC §2.2)."""
    for table_name in ("work", "version", "edges"):
        cols = _table(table_name).columns
        for col in ("confidence", "sources", "human_verified"):
            assert col in cols, f"{table_name} missing {col}"
            assert not cols[col].nullable or col == "human_verified"


def test_qid_uniqueness_declared() -> None:
    for table_name in ("work", "version", "person"):
        assert f"uq_{table_name}_wikidata_qid" in _constraint_names(_table(table_name))


def test_candidate_edges_is_not_an_edge_table() -> None:
    """candidate_edges has no confidence tier / human_verified — it cannot masquerade as edges."""
    cols = _table("candidate_edges").columns
    assert "confidence" not in cols  # model_confidence only — a different, ungated concept
    assert "human_verified" not in cols
    assert "status" in cols and "supporting_sentence" in cols


def test_ground_truth_view_names_pinned() -> None:
    assert GROUND_TRUTH_VIEWS == (
        "ground_truth_works",
        "ground_truth_versions",
        "ground_truth_edges",
    )


def test_postgres_url_is_env_driven() -> None:
    """The DSN comes from Settings (POSTGRES_*), never a hardcoded string."""
    s = Settings(
        POSTGRES_HOST="db.example",
        POSTGRES_PORT=5433,
        POSTGRES_DB="graphdb",
        POSTGRES_USER="alice",
        POSTGRES_PASSWORD="s3cret",
    )
    url = postgres_url(s)
    assert url.drivername == "postgresql+psycopg"
    assert (url.host, url.port, url.database, url.username) == (
        "db.example",
        5433,
        "graphdb",
        "alice",
    )
    # Secrets never leak via repr (SQLAlchemy masks the password).
    assert "s3cret" not in repr(url)
