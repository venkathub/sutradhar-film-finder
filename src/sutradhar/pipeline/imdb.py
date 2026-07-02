"""IMDb ``title.akas`` loader (P1 task 6).

The multi-GB dump is **streamed** from ``datasets.imdbws.com`` (env-driven
``IMDB_DATASETS_URL``) and filtered on the fly to the slice's tconsts — the raw file is never
stored and never committed; only the filtered rows land in a hash-recorded snapshot
(``data/raw/imdb/``). Column contract per developer.imdb.com (verified §2.9):
``titleId, ordering, title, region, language, types, attributes, isOriginalTitle``; ``\\N`` = null.

Loading (idempotent, union semantics via ``sutradhar.pipeline.titles``):
- ``isOriginalTitle=1`` rows corroborate the version's **canonical** title;
- language-tagged rows matching a QID-less sibling map onto that sibling
  (kind=dub, or canonical for a bilingual co-original — same guard as TMDB);
- everything else lands as ``kind=aka`` on the tconst's version.

**License: IMDb non-commercial datasets — personal/non-commercial use ONLY**
(docs/LICENSING.md). Fine for this portfolio demo; never commercialize.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, Any, Protocol, cast

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.config import Settings, get_settings
from sutradhar.graph.models import SourceId, SourceRef
from sutradhar.graph.schema import Version
from sutradhar.pipeline.titles import upsert_version_title

AKAS_FILENAME = "title.akas.tsv.gz"
_COLUMNS = (
    "titleId",
    "ordering",
    "title",
    "region",
    "language",
    "types",
    "attributes",
    "isOriginalTitle",
)


class AkaRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    titleId: str  # noqa: N815 — mirrors the documented TSV header exactly
    ordering: int
    title: str
    region: str | None = None
    language: str | None = None
    types: str | None = None
    attributes: str | None = None
    isOriginalTitle: bool = False  # noqa: N815


def _null(value: str) -> str | None:
    return None if value == r"\N" else value


def parse_aka_line(line: str) -> AkaRow | None:
    """Parse one TSV line (header contract §2.9). Returns None for malformed lines."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(_COLUMNS) or parts[0] == "titleId":
        return None
    return AkaRow(
        titleId=parts[0],
        ordering=int(parts[1]) if parts[1].isdigit() else 0,
        title=parts[2],
        region=_null(parts[3]),
        language=_null(parts[4]),
        types=_null(parts[5]),
        attributes=_null(parts[6]),
        isOriginalTitle=parts[7] == "1",
    )


class ReadableBytes(Protocol):
    """Anything with a bytes read() — a real file or the httpx streaming adapter."""

    def read(self, size: int = ..., /) -> bytes: ...  # pragma: no cover


def filter_akas_stream(stream: ReadableBytes, tconsts: set[str]) -> list[AkaRow]:
    """Stream-decompress + filter a ``title.akas.tsv.gz`` byte stream to slice tconsts."""
    rows: list[AkaRow] = []
    with gzip.open(cast(IO[bytes], stream), mode="rt", encoding="utf-8") as fh:
        for line in fh:
            # Cheap prefix check before the full parse (the dump has ~50M lines).
            tconst = line[: line.find("\t")]
            if tconst in tconsts:
                row = parse_aka_line(line)
                if row is not None:
                    rows.append(row)
    return rows


def download_and_filter_akas(tconsts: set[str], settings: Settings | None = None) -> list[AkaRow]:
    """Stream the dump from datasets.imdbws.com; only filtered rows survive (never raw)."""
    s = settings if settings is not None else get_settings()
    url = f"{s.imdb_datasets_url.rstrip('/')}/{AKAS_FILENAME}"
    with (
        httpx.Client(
            headers={"User-Agent": s.http_user_agent}, timeout=None, follow_redirects=True
        ) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        # httpx exposes the raw (still-gzipped) byte stream; gzip.open decompresses lazily.
        return filter_akas_stream(_HTTPXByteStream(response), tconsts)


class _HTTPXByteStream:
    """Minimal file-like adapter over httpx's streaming response for gzip.open."""

    def __init__(self, response: httpx.Response) -> None:
        self._iterator = response.iter_raw()
        self._buffer = b""

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            try:
                self._buffer += next(self._iterator)
            except StopIteration:
                break
        if size < 0:
            out, self._buffer = self._buffer, b""
        else:
            out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out


# --- Loading into version_title ---


@dataclass
class AkasReport:
    rows_seen: int = 0
    titles_new: int = 0
    titles_corroborated: int = 0  # merged into an existing (e.g. TMDB-sourced) row → ≥2 sources
    dub_titles_mapped: int = 0
    unmatched_tconsts: list[str] = field(default_factory=list)


def rows_to_jsonable(rows: list[AkaRow]) -> dict[str, Any]:
    return {"rows": [r.model_dump() for r in rows]}


def rows_from_jsonable(payload: dict[str, Any]) -> list[AkaRow]:
    return [AkaRow.model_validate(r) for r in payload["rows"]]


def load_akas(
    session: Session,
    rows: list[AkaRow],
    retrieved_at: datetime | None = None,
) -> AkasReport:
    """Load filtered akas rows into ``version_title`` (idempotent; union/merge semantics)."""
    retrieved_at = retrieved_at or datetime.now(tz=UTC)
    report = AkasReport()

    versions = session.scalars(select(Version)).all()
    by_tconst = {v.imdb_id: v for v in versions if v.imdb_id is not None}
    by_work: dict[Any, list[Version]] = {}
    for v in versions:
        by_work.setdefault(v.work_id, []).append(v)

    for row in rows:
        report.rows_seen += 1
        version = by_tconst.get(row.titleId)
        if version is None:
            report.unmatched_tconsts.append(row.titleId)
            continue
        # NOTE: no per-row ordering in the ref — corroboration means *independent sources*;
        # two akas rows from IMDb must dedupe to ONE imdb ref, never fake a 2-source HIGH.
        ref = SourceRef(
            source=SourceId.IMDB,
            ref=f"{row.titleId}#title.akas",
            retrieved_at=retrieved_at,
        )

        target = version
        kind = "aka"
        if row.isOriginalTitle:
            kind = "canonical"  # corroborates the canonical row (union → 2 sources → HIGH)
        elif row.language:
            sibling = next(
                (
                    s
                    for s in by_work[version.work_id]
                    if s.version_id != version.version_id
                    and s.imdb_id is None
                    and s.language == row.language
                ),
                None,
            )
            if sibling is not None:
                target = sibling
                kind = "canonical" if sibling.is_original else "dub"

        outcome = upsert_version_title(
            session, target.version_id, row.title, kind, row.language, [ref]
        )
        if outcome == "new":
            report.titles_new += 1
            if kind == "dub":
                report.dub_titles_mapped += 1
        elif outcome == "merged":
            report.titles_corroborated += 1

    report.unmatched_tconsts = sorted(set(report.unmatched_tconsts))
    session.flush()
    return report
