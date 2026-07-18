"""Uniqueness hardening: DB-owned keys the app previously enforced by discipline.

P7 task 7 (DEC-P7-1 finding 9). Three constraints:

1. ``uq_person_tmdb_id`` — person entity-resolution keys on ``tmdb_id``; a duplicate
   would silently merge two people's filmographies (GS-10 false-merge risk).
2. ``uq_version_work_lang_year`` — the exact QID-less fallback key
   ``_upsert_version`` looks up on; without DB backing, a race or bug mints
   duplicate versions the upsert then matches nondeterministically.
   NULLS DISTINCT (default): year-unknown dub tracks may coexist.
3. ``uq_candidate_edges_dedup`` — the ``extract.py`` SELECT-then-skip dedup key
   ``(edge_type, src_title_raw, dst_title_raw, source_page)``, now constraint-backed.
   NULLS NOT DISTINCT: proposals identical up to NULL titles ARE duplicates.

A pre-audit runs first: existing violations ABORT the migration with the offending
keys listed — duplicates are resolved by a human (conflicts-queue posture: never
silently deleted by a migration).

Revision ID: a41f09c3d7e2
Revises: bb22e78ff305
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a41f09c3d7e2"
down_revision = "bb22e78ff305"
branch_labels = None
depends_on = None


def _audit(what: str, sql: str) -> None:
    rows = op.get_bind().execute(sa.text(sql)).fetchall()
    if rows:
        offenders = "; ".join(str(tuple(row)) for row in rows[:20])
        raise RuntimeError(
            f"uniqueness pre-audit failed — {what} has {len(rows)} duplicate key(s): "
            f"{offenders}. Resolve manually (conflicts-queue posture: a migration "
            "never deletes data), then re-run `alembic upgrade head`."
        )


def upgrade() -> None:
    _audit(
        "person.tmdb_id",
        """
        SELECT tmdb_id, count(*) FROM person
        WHERE tmdb_id IS NOT NULL GROUP BY tmdb_id HAVING count(*) > 1
        """,
    )
    _audit(
        "version (work_id, language, release_year)",
        """
        SELECT work_id::text, language, release_year, count(*) FROM version
        WHERE release_year IS NOT NULL
        GROUP BY work_id, language, release_year HAVING count(*) > 1
        """,
    )
    _audit(
        "candidate_edges (edge_type, src_title_raw, dst_title_raw, source_page)",
        """
        SELECT edge_type, src_title_raw, dst_title_raw, source_page, count(*)
        FROM candidate_edges
        GROUP BY edge_type, src_title_raw, dst_title_raw, source_page
        HAVING count(*) > 1
        """,
    )
    op.create_index("uq_person_tmdb_id", "person", ["tmdb_id"], unique=True)
    op.create_unique_constraint(
        "uq_version_work_lang_year", "version", ["work_id", "language", "release_year"]
    )
    op.create_index(
        "uq_candidate_edges_dedup",
        "candidate_edges",
        ["edge_type", "src_title_raw", "dst_title_raw", "source_page"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.drop_index("uq_candidate_edges_dedup", table_name="candidate_edges")
    op.drop_constraint("uq_version_work_lang_year", "version", type_="unique")
    op.drop_index("uq_person_tmdb_id", table_name="person")
