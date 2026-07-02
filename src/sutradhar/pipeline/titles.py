"""Shared ``version_title`` upsert with source merging (P1 tasks 5/6).

Materializes the ``union`` precedence strategy (DATA_SOURCES.md "AKA / dub titles: union,
normalize, dedupe"): a title row is keyed on ``(version_id, title, kind)``; re-observing it
from a *different* source merges that source's ref into ``sources[]`` — a title vouched for
by ≥2 sources is thereby HIGH per the tier table, visible in its provenance.
"""

from __future__ import annotations

import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.models import SourceRef, sources_to_jsonb
from sutradhar.graph.schema import VersionTitle
from sutradhar.pipeline.normalize import match_key

UpsertOutcome = Literal["new", "merged", "unchanged"]


def upsert_version_title(
    session: Session,
    version_id: uuid.UUID,
    title: str,
    kind: str,
    language: str | None,
    refs: list[SourceRef],
) -> UpsertOutcome:
    """Insert a title row, or merge new source refs into an existing one (union semantics)."""
    existing = session.scalars(
        select(VersionTitle).where(
            VersionTitle.version_id == version_id,
            VersionTitle.title == title,
            VersionTitle.kind == kind,
        )
    ).first()
    if existing is None:
        session.add(
            VersionTitle(
                version_id=version_id,
                title=title,
                kind=kind,
                language=language,
                match_key=match_key(title),
                sources=sources_to_jsonb(refs),
            )
        )
        # Sessions run with autoflush=False: flush now so a second observation of the same
        # (version, title, kind) in the same pass MERGES instead of duplicating the row.
        session.flush()
        return "new"

    # Merge: dedupe on (source, ref) pairs; corroboration accumulates provenance.
    seen = {(s.get("source"), s.get("ref")) for s in existing.sources}
    added = [r for r in refs if (r.source.value, r.ref) not in seen]
    if not added:
        return "unchanged"
    existing.sources = existing.sources + sources_to_jsonb(added)
    if existing.language is None and language is not None:
        existing.language = language
    return "merged"
