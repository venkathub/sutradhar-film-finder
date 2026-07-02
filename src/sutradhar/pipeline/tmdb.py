"""TMDB connector + enrichment (P1 task 5).

One ``append_to_response=translations,alternative_titles,credits`` call per film (verified v3
contract, P1_SPEC §2.9) — auth auto-detects a v4 Bearer token (``eyJ…``) vs a v3 api_key.
Snapshot-first like every connector.

Enrichment writes (idempotent):
- ``version_title`` rows: the film's canonical title per its own language, alternative titles
  (kind=aka), and translation titles mapped onto QID-less sibling dub tracks (kind=dub) — the
  only way a dub track acquires its display title from a structured source.
- ``person`` / ``version_cast``: credits (order < LEAD_BILLING_CUTOFF → lead, else support;
  crew job=Director → director) — the dub-vs-remake rule's evidence base (task 9).
- Precedence application (``sutradhar.pipeline.precedence``): release-year and
  original-language disagreements are recorded as conflicts (resolved-by-rule or open),
  never silent.

TMDB attribution (required): this product uses the TMDB API but is not endorsed or certified
by TMDB. (Recorded in docs/LICENSING.md, task 16.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.config import Settings, get_settings
from sutradhar.graph.models import SourceId, SourceRef, sources_to_jsonb
from sutradhar.graph.schema import Conflict, Person, Version, VersionCast
from sutradhar.pipeline.precedence import Observation, resolve_field
from sutradhar.pipeline.titles import upsert_version_title

LEAD_BILLING_CUTOFF = 5  # TMDB `order` < 5 → lead (top billing), else support
CAST_LIMIT = 15  # keep the slice lean; supports the dub-vs-remake overlap rule


# --- Typed payload views ---


class TMDBTitle(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    language: str | None = None  # iso_639_1
    region: str | None = None  # iso_3166_1
    kind: str  # canonical | aka | translation


class TMDBCredit(BaseModel):
    model_config = ConfigDict(frozen=True)

    tmdb_person_id: int
    name: str
    role_kind: str  # lead | support | director
    billing_order: int | None = None


class TMDBMovie(BaseModel):
    model_config = ConfigDict(frozen=True)

    tmdb_id: int
    imdb_id: str | None = None
    original_title: str
    original_language: str
    release_year: int | None = None
    titles: tuple[TMDBTitle, ...] = ()
    credits: tuple[TMDBCredit, ...] = ()


def parse_movie(raw: dict[str, Any]) -> TMDBMovie:
    """Parse one ``/movie/{id}?append_to_response=…`` payload into the enrichment view."""
    titles: list[TMDBTitle] = [
        TMDBTitle(
            title=raw["original_title"], language=raw.get("original_language"), kind="canonical"
        )
    ]
    for alt in raw.get("alternative_titles", {}).get("titles", []):
        if alt.get("title"):
            titles.append(TMDBTitle(title=alt["title"], region=alt.get("iso_3166_1"), kind="aka"))
    for tr in raw.get("translations", {}).get("translations", []):
        tr_title = tr.get("data", {}).get("title")
        if tr_title:
            titles.append(
                TMDBTitle(title=tr_title, language=tr.get("iso_639_1"), kind="translation")
            )

    credits: list[TMDBCredit] = []
    for member in raw.get("credits", {}).get("cast", [])[:CAST_LIMIT]:
        credits.append(
            TMDBCredit(
                tmdb_person_id=member["id"],
                name=member["name"],
                role_kind="lead" if member.get("order", 99) < LEAD_BILLING_CUTOFF else "support",
                billing_order=member.get("order"),
            )
        )
    for member in raw.get("credits", {}).get("crew", []):
        if member.get("job") == "Director":
            credits.append(
                TMDBCredit(tmdb_person_id=member["id"], name=member["name"], role_kind="director")
            )

    release_date = raw.get("release_date") or ""
    return TMDBMovie(
        tmdb_id=raw["id"],
        imdb_id=raw.get("imdb_id"),
        original_title=raw["original_title"],
        original_language=raw["original_language"],
        release_year=int(release_date[:4]) if len(release_date) >= 4 else None,
        titles=tuple(titles),
        credits=tuple(credits),
    )


# --- Client ---


class TMDBClient:
    """One-call-per-film TMDB v3 client (Bearer v4 token or v3 api_key, auto-detected)."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        s = settings if settings is not None else get_settings()
        api_key = s.require("tmdb_api_key")
        self._base_url = s.tmdb_api_url.rstrip("/")
        self._max_retries = max_retries
        headers = {"User-Agent": s.http_user_agent, "Accept": "application/json"}
        self._params: dict[str, str] = {}
        if api_key.startswith("eyJ"):  # v4 Read Access Token → Bearer
            headers["Authorization"] = f"Bearer {api_key}"
        else:  # classic v3 key → query param
            self._params["api_key"] = api_key
        self._client = httpx.Client(headers=headers, timeout=30.0, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            response = self._client.get(url, params={**self._params, **params})
            if response.status_code == 429 and attempt < self._max_retries:
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_movie(self, tmdb_id: int) -> dict[str, Any]:
        """ONE call per film: movie + translations + alternative_titles + credits."""
        response = self._get(
            f"{self._base_url}/movie/{tmdb_id}",
            params={"append_to_response": "translations,alternative_titles,credits"},
        )
        payload: dict[str, Any] = response.json()
        return payload


# --- Enrichment ---


@dataclass
class EnrichReport:
    versions_enriched: int = 0
    titles_written: int = 0
    dub_titles_mapped: int = 0
    people_upserted: int = 0
    cast_rows_written: int = 0
    conflicts_recorded: int = 0
    conflicts_open: int = 0
    missing_payloads: list[int] = field(default_factory=list)


def _tmdb_ref(tmdb_id: int, retrieved_at: datetime, field_name: str | None = None) -> SourceRef:
    return SourceRef(
        source=SourceId.TMDB,
        ref=f"tmdb:{tmdb_id}",
        field=field_name,
        retrieved_at=retrieved_at,
    )


def _upsert_title(
    session: Session,
    version: Version,
    title: str,
    kind: str,
    language: str | None,
    sources: list[SourceRef],
    report: EnrichReport,
) -> None:
    outcome = upsert_version_title(session, version.version_id, title, kind, language, sources)
    if outcome == "new":
        report.titles_written += 1


def _record_conflict(
    session: Session,
    version: Version,
    field_name: str,
    values: list[dict[str, Any]],
    status: str,
    resolution: dict[str, Any] | None,
    report: EnrichReport,
) -> None:
    existing = session.scalars(
        select(Conflict).where(
            Conflict.entity_id == version.version_id, Conflict.field == field_name
        )
    ).first()
    if existing is not None:
        return
    session.add(
        Conflict(
            entity_kind="version",
            entity_id=version.version_id,
            field=field_name,
            values=values,
            status=status,
            resolution=resolution,
        )
    )
    report.conflicts_recorded += 1
    if status == "open":
        report.conflicts_open += 1


def enrich_tmdb(
    session: Session,
    movies: dict[int, TMDBMovie],
    retrieved_at: datetime | None = None,
) -> EnrichReport:
    """Enrich DB versions from TMDB payloads. Idempotent (re-run adds nothing)."""
    retrieved_at = retrieved_at or datetime.now(tz=UTC)
    report = EnrichReport()

    versions = session.scalars(select(Version)).all()
    by_tmdb_id = {v.tmdb_id: v for v in versions if v.tmdb_id is not None}
    by_work: dict[Any, list[Version]] = {}
    for v in versions:
        by_work.setdefault(v.work_id, []).append(v)

    for tmdb_id, version in sorted(by_tmdb_id.items()):
        movie = movies.get(tmdb_id)
        if movie is None:
            report.missing_payloads.append(tmdb_id)
            continue
        report.versions_enriched += 1
        ref = _tmdb_ref(tmdb_id, retrieved_at)

        # imdb_id gap-fill (hub rule: Wikidata already set it where it could; TMDB fills).
        if version.imdb_id is None and movie.imdb_id:
            version.imdb_id = movie.imdb_id

        # --- release_year precedence (majority; split → open conflict) ---
        if movie.release_year is not None and version.release_year is not None:
            resolution = resolve_field(
                "release_year",
                [
                    Observation(version.release_year, "human"),
                    Observation(movie.release_year, "tmdb"),
                ],
            )
            if resolution.conflict != "none":
                _record_conflict(
                    session,
                    version,
                    "release_year",
                    resolution.conflict_values,
                    "open" if resolution.conflict == "open" else "resolved",
                    resolution.resolution,
                    report,
                )

        # --- original_language precedence (TMDB primary; disagreement recorded) ---
        if version.is_original:
            resolution = resolve_field(
                "original_language",
                [
                    Observation(version.language, "human"),
                    Observation(movie.original_language, "tmdb"),
                ],
            )
            if resolution.conflict != "none":
                _record_conflict(
                    session,
                    version,
                    "original_language",
                    resolution.conflict_values,
                    "open" if resolution.conflict == "open" else "resolved",
                    resolution.resolution,
                    report,
                )

        # --- titles → version_title ---
        siblings = by_work[version.work_id]
        qidless_siblings_by_language = {
            s.language: s
            for s in siblings
            if s.tmdb_id is None and s.version_id != version.version_id
        }
        for t in movie.titles:
            if t.kind == "canonical" or (
                t.kind == "translation" and t.language == version.language
            ):
                _upsert_title(
                    session,
                    version,
                    t.title,
                    "canonical",
                    t.language or version.language,
                    [ref],
                    report,
                )
            elif t.kind == "translation" and t.language in qidless_siblings_by_language:
                # A translation in a QID-less sibling's language = that sibling's display
                # title: kind=dub for dub tracks, canonical for a co-original (bilingual).
                sibling = qidless_siblings_by_language[t.language]
                sibling_kind = "canonical" if sibling.is_original else "dub"
                before = report.titles_written
                _upsert_title(session, sibling, t.title, sibling_kind, t.language, [ref], report)
                if report.titles_written > before and sibling_kind == "dub":
                    report.dub_titles_mapped += 1
            elif t.kind == "aka":
                _upsert_title(session, version, t.title, "aka", t.language, [ref], report)
            # other-language translations: not title-index material for this version

        # --- credits → person + version_cast ---
        for credit in movie.credits:
            person = session.scalars(
                select(Person).where(Person.tmdb_id == credit.tmdb_person_id)
            ).first()
            if person is None:
                person = Person(
                    name=credit.name,
                    tmdb_id=credit.tmdb_person_id,
                    sources=sources_to_jsonb([ref]),
                )
                session.add(person)
                session.flush()
                report.people_upserted += 1
            cast_row = session.get(
                VersionCast, (version.version_id, person.person_id, credit.role_kind)
            )
            if cast_row is None:
                session.add(
                    VersionCast(
                        version_id=version.version_id,
                        person_id=person.person_id,
                        role_kind=credit.role_kind,
                        billing_order=credit.billing_order,
                        sources=sources_to_jsonb([ref]),
                    )
                )
                report.cast_rows_written += 1

    session.flush()
    return report
