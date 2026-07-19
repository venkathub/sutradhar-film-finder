"""Wikidata connector + spine ingest (P1 task 4).

Two-phase access per the verified best practice (P1_SPEC §2.9): a single SPARQL discovery query
returns **QIDs only** (P144/P4969 backlinks of the slice), then entity detail comes from batched
``wbgetentities`` calls. Descriptive User-Agent (WMF policy), gzip, and 429/503 ``Retry-After``
backoff. Raw responses are persisted as hash-recorded snapshots under ``data/raw/wikidata/`` so
the graph build is reproducible without re-hitting the API (CI parses committed fixture samples).

Metric-honesty rule: :func:`ingest_spine` writes **only edges Wikidata actually asserts**
(P144/P4969 → remake / based_on, P155/P156 → sequel). Seed ``relationship:`` entries are curated
*truth* (coverage denominators), never an edge source — the gap between the two is exactly the
lift the extraction layer (task 11) + human gate (task 12) must earn.
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
from sutradhar.graph.models import (
    Confidence,
    EdgeType,
    SourceId,
    SourceRef,
    merge_sources_jsonb,
    sources_to_jsonb,
)
from sutradhar.graph.schema import Conflict, Edge, Version, Work
from sutradhar.pipeline.seed import SeedSlice, SeedVersion, SeedWork
from sutradhar.pipeline.snapshots import load_snapshot, write_snapshot

__all__ = [  # snapshot helpers re-exported for entrypoint/back-compat
    "WikidataClient",
    "WikidataEntity",
    "ingest_spine",
    "load_snapshot",
    "parse_entity",
    "write_snapshot",
]

# --- Wikidata property/entity constants (verified 2026-07-02, P1_SPEC §2.9) ---

P_BASED_ON = "P144"
P_DERIVATIVE_WORK = "P4969"
P_AFTER_A_WORK_BY = "P1877"
P_FOLLOWS = "P155"
P_FOLLOWED_BY = "P156"
P_PART_OF_SERIES = "P179"
P_DIRECTOR = "P57"
P_PUBLICATION_DATE = "P577"
P_ORIGINAL_LANGUAGE = "P364"
P_COUNTRY_OF_ORIGIN = "P495"
P_IMDB_ID = "P345"
P_TMDB_MOVIE_ID = "P4947"

Q_INDIA = "Q668"

# Language-item → BCP-47-ish code, for corroborating seed `language` (slice languages only).
LANGUAGE_QIDS = {
    "Q36236": "ml",
    "Q5885": "ta",
    "Q8097": "te",
    "Q1568": "hi",
    "Q33673": "kn",
    "Q9610": "bn",
    "Q13267": "si",
    "Q7850": "zh",
    "Q9186": "ta",  # some items use Q9186 for Tamil historically; normalized to ta
}

WBGETENTITIES_BATCH = 50  # API maximum


# --- Typed entity (what the spine needs; everything else stays in the raw snapshot) ---


class WikidataEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    qid: str
    label_en: str | None = None
    description_en: str | None = None
    publication_years: tuple[int, ...] = ()
    original_language_qids: tuple[str, ...] = ()
    country_qids: tuple[str, ...] = ()
    imdb_id: str | None = None
    tmdb_id: int | None = None
    based_on: tuple[str, ...] = ()  # P144 targets
    derivative_works: tuple[str, ...] = ()  # P4969 targets
    follows: tuple[str, ...] = ()  # P155
    followed_by: tuple[str, ...] = ()  # P156
    director_qids: tuple[str, ...] = ()


def _claim_items(entity: dict[str, Any], prop: str) -> list[str]:
    """Extract item-QID targets of a property's statements (skips novalue/somevalue)."""
    items: list[str] = []
    for claim in entity.get("claims", {}).get(prop, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue")
        if datavalue and datavalue.get("type") == "wikibase-entityid":
            items.append(datavalue["value"]["id"])
    return items


def _claim_strings(entity: dict[str, Any], prop: str) -> list[str]:
    values: list[str] = []
    for claim in entity.get("claims", {}).get(prop, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue")
        if datavalue and datavalue.get("type") == "string":
            values.append(datavalue["value"])
    return values


def _claim_years(entity: dict[str, Any], prop: str) -> list[int]:
    years: list[int] = []
    for claim in entity.get("claims", {}).get(prop, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue")
        if datavalue and datavalue.get("type") == "time":
            raw = datavalue["value"]["time"]  # e.g. "+2013-12-19T00:00:00Z"
            try:
                years.append(int(raw.lstrip("+")[:4]))
            except ValueError:  # malformed time — skip, never guess
                continue
    return years


def parse_entity(raw: dict[str, Any]) -> WikidataEntity:
    """Parse one ``wbgetentities`` entity JSON into the typed spine view."""
    imdb_ids = _claim_strings(raw, P_IMDB_ID)
    tmdb_ids = _claim_strings(raw, P_TMDB_MOVIE_ID)
    tmdb_id: int | None = None
    if tmdb_ids:
        try:
            tmdb_id = int(tmdb_ids[0])
        except ValueError:
            tmdb_id = None
    return WikidataEntity(
        qid=raw["id"],
        label_en=raw.get("labels", {}).get("en", {}).get("value"),
        description_en=raw.get("descriptions", {}).get("en", {}).get("value"),
        publication_years=tuple(sorted(set(_claim_years(raw, P_PUBLICATION_DATE)))),
        original_language_qids=tuple(_claim_items(raw, P_ORIGINAL_LANGUAGE)),
        country_qids=tuple(_claim_items(raw, P_COUNTRY_OF_ORIGIN)),
        imdb_id=imdb_ids[0] if imdb_ids else None,
        tmdb_id=tmdb_id,
        based_on=tuple(_claim_items(raw, P_BASED_ON)),
        derivative_works=tuple(_claim_items(raw, P_DERIVATIVE_WORK)),
        follows=tuple(_claim_items(raw, P_FOLLOWS)),
        followed_by=tuple(_claim_items(raw, P_FOLLOWED_BY)),
        director_qids=tuple(_claim_items(raw, P_DIRECTOR)),
    )


# --- Client (descriptive UA, gzip, Retry-After backoff) ---


class WikidataClient:
    """Thin Wikimedia-etiquette-compliant client (WMF UA policy + rate-limit backoff)."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        s = settings if settings is not None else get_settings()
        self._api_url = s.wikidata_api_url
        self._sparql_url = s.wikidata_sparql_url
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers={
                "User-Agent": s.http_user_agent,
                "Accept-Encoding": "gzip",
            },
            timeout=30.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        """GET with 429/503 Retry-After backoff (WMF etiquette: honor the header)."""
        for attempt in range(self._max_retries + 1):
            response = self._client.get(url, params=params)
            if response.status_code in (429, 503) and attempt < self._max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0 * (attempt + 1)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_entities(self, qids: list[str]) -> dict[str, dict[str, Any]]:
        """Batched ``wbgetentities`` (phase 2): full entity JSON keyed by QID."""
        entities: dict[str, dict[str, Any]] = {}
        for i in range(0, len(qids), WBGETENTITIES_BATCH):
            batch = qids[i : i + WBGETENTITIES_BATCH]
            response = self._get(
                self._api_url,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels|descriptions|claims|sitelinks",
                    "format": "json",
                },
            )
            payload = response.json()
            for qid, entity in payload.get("entities", {}).items():
                if "missing" not in entity:
                    entities[qid] = entity
        return entities

    def discover_backlinks(self, qids: list[str]) -> list[str]:
        """Phase 1 SPARQL: QIDs only — items claiming P144/P4969 to/from slice items.

        Used to *discover* remake versions the slice may be missing (§7 Q1 conditional adds);
        results are reported for human review, never auto-ingested.
        """
        values = " ".join(f"wd:{qid}" for qid in qids)
        query = (
            "SELECT DISTINCT ?item WHERE { "
            f"VALUES ?seed {{ {values} }} "
            "{ ?item wdt:P144 ?seed . } UNION { ?seed wdt:P4969 ?item . } }"
        )
        response = self._get(self._sparql_url, params={"query": query, "format": "json"})
        bindings = response.json().get("results", {}).get("bindings", [])
        found: list[str] = []
        for row in bindings:
            uri = row.get("item", {}).get("value", "")
            if uri.rsplit("/", 1)[-1].startswith("Q"):
                found.append(uri.rsplit("/", 1)[-1])
        return sorted(set(found))


# --- Snapshots: shared helpers live in sutradhar.pipeline.snapshots (re-exported here) ---


# --- Spine ingest (idempotent upsert; Wikidata-asserted edges only) ---


@dataclass
class SpineReport:
    """What one ingest pass did — printed by the entrypoint, asserted by tests."""

    works_upserted: int = 0
    versions_upserted: int = 0
    edges_written: int = 0
    conflicts_opened: int = 0
    discovered_unmatched_qids: list[str] = field(default_factory=list)
    edge_labels: list[str] = field(default_factory=list)


def _wikidata_ref(
    qid: str, prop: str | None = None, retrieved_at: datetime | None = None
) -> SourceRef:
    return SourceRef(
        source=SourceId.WIKIDATA,
        ref=f"{qid}#{prop}" if prop else qid,
        retrieved_at=retrieved_at,
    )


def _seed_ref(slice_: SeedSlice) -> SourceRef:
    return SourceRef(source=SourceId.HUMAN, ref=f"seed_slice@{slice_.meta.verified_at}")


def _upsert_work(
    session: Session,
    key: str,
    seed_work: SeedWork,
    anchor_qid: str | None,
    sources: list[SourceRef],
) -> Work:
    existing = None
    if anchor_qid:
        existing = session.scalars(select(Work).where(Work.wikidata_qid == anchor_qid)).first()
    if existing is None:
        existing = session.scalars(
            select(Work).where(
                Work.primary_title == seed_work.primary_title,
                Work.first_release_year == seed_work.first_release_year,
            )
        ).first()
    if existing is None:
        existing = Work(
            work_type=seed_work.work_type.value,
            primary_title=seed_work.primary_title,
            original_language=seed_work.original_language,
            first_release_year=seed_work.first_release_year,
            wikidata_qid=anchor_qid,
            confidence=(Confidence.HIGH if anchor_qid else Confidence.MEDIUM).value,
            sources=sources_to_jsonb(sources),
        )
        session.add(existing)
        session.flush()
    else:
        # P7 task 6 (DEC-P7-1 finding 7): provenance is append-only — merge, never
        # replace — and a human-verified record's curated fields are not overwritten
        # by a pipeline re-ingest.
        existing.sources = merge_sources_jsonb(existing.sources, sources_to_jsonb(sources))
        if not existing.human_verified:
            existing.original_language = seed_work.original_language
            existing.wikidata_qid = anchor_qid
        elif existing.wikidata_qid is None and anchor_qid:
            existing.wikidata_qid = anchor_qid  # filling a gap is a raise, not an overwrite
    return existing


def _upsert_version(
    session: Session,
    work: Work,
    seed_version: SeedVersion,
    entity: WikidataEntity | None,
    sources: list[SourceRef],
) -> tuple[Version, bool]:
    """Upsert one version (QID key; ``(work_id, language, release_year)`` fallback for
    QID-less dub tracks — release_year disambiguates same-language remakes inside one
    work, e.g. Don (hi 1978) vs Don (hi 2006) in the P4 training slice).

    Returns ``(version, year_conflict)`` — a seed-vs-Wikidata year disagreement is surfaced,
    never silently resolved (the caller writes the conflicts row).
    """
    existing = None
    if seed_version.wikidata_qid:
        existing = session.scalars(
            select(Version).where(Version.wikidata_qid == seed_version.wikidata_qid)
        ).first()
    if existing is None:
        existing = session.scalars(
            select(Version).where(
                Version.work_id == work.work_id,
                Version.language == seed_version.language,
                Version.release_year == seed_version.release_year,
            )
        ).first()

    year = seed_version.release_year
    year_conflict = False
    if entity and entity.publication_years and year not in entity.publication_years:
        year_conflict = True  # both values preserved via the conflicts row

    confidence = Confidence.HIGH if seed_version.wikidata_qid else Confidence.MEDIUM
    if existing is None:
        existing = Version(
            work_id=work.work_id,
            wikidata_qid=seed_version.wikidata_qid,
            tmdb_id=entity.tmdb_id if entity else None,
            imdb_id=entity.imdb_id if entity else None,
            title=seed_version.title,
            language=seed_version.language,
            release_year=year,
            country=seed_version.country,
            is_original=seed_version.is_original,
            confidence=confidence.value,
            sources=sources_to_jsonb(sources),
        )
        session.add(existing)
        session.flush()
    else:
        # P7 task 6 (DEC-P7-1 finding 7): merge sources[] (append-only provenance);
        # confidence is raise-only on re-ingest (a QID-less fallback match must not
        # downgrade a HIGH record); human-verified records keep their curated values.
        existing.sources = merge_sources_jsonb(existing.sources, sources_to_jsonb(sources))
        if existing.human_verified:
            existing.tmdb_id = existing.tmdb_id or (entity.tmdb_id if entity else None)
            existing.imdb_id = existing.imdb_id or (entity.imdb_id if entity else None)
        else:
            existing.tmdb_id = entity.tmdb_id if entity else existing.tmdb_id
            existing.imdb_id = entity.imdb_id if entity else existing.imdb_id
            existing.title = seed_version.title
            existing.release_year = year
            existing.country = seed_version.country
            existing.is_original = seed_version.is_original
            if confidence is Confidence.HIGH:
                existing.confidence = Confidence.HIGH.value
    return existing, year_conflict


def _edge_exists(session: Session, edge_type: str, src_id: Any, dst_id: Any) -> bool:
    return (
        session.scalars(
            select(Edge).where(
                Edge.edge_type == edge_type, Edge.src_id == src_id, Edge.dst_id == dst_id
            )
        ).first()
        is not None
    )


def _open_conflict_exists(session: Session, entity_id: Any, field_name: str) -> bool:
    return (
        session.scalars(
            select(Conflict).where(
                Conflict.entity_id == entity_id,
                Conflict.field == field_name,
                Conflict.status == "open",
            )
        ).first()
        is not None
    )


def ingest_spine(
    session: Session,
    slice_: SeedSlice,
    entities: dict[str, WikidataEntity],
    retrieved_at: datetime | None = None,
) -> SpineReport:
    """Upsert work/version skeletons + Wikidata-asserted edges. Idempotent (re-run = upsert)."""
    retrieved_at = retrieved_at or datetime.now(tz=UTC)
    report = SpineReport()
    seed_ref = _seed_ref(slice_)

    works_by_key: dict[str, Work] = {}
    versions_by_qid: dict[str, Version] = {}
    version_work_key: dict[str, str] = {}  # qid -> seed work key

    # 1. Works + versions.
    for wkey, seed_work in slice_.works.items():
        original_qids = [
            v.wikidata_qid for v in seed_work.versions.values() if v.is_original and v.wikidata_qid
        ]
        anchor_qid = seed_work.wikidata_qid or (original_qids[0] if original_qids else None)
        work_sources = [seed_ref] + (
            [_wikidata_ref(anchor_qid, retrieved_at=retrieved_at)] if anchor_qid else []
        )
        work = _upsert_work(session, wkey, seed_work, anchor_qid, work_sources)
        works_by_key[wkey] = work
        report.works_upserted += 1

        for seed_version in seed_work.versions.values():
            entity = entities.get(seed_version.wikidata_qid) if seed_version.wikidata_qid else None
            v_sources = [seed_ref] + (
                [_wikidata_ref(seed_version.wikidata_qid, retrieved_at=retrieved_at)]
                if seed_version.wikidata_qid
                else []
            )
            version, year_conflict = _upsert_version(session, work, seed_version, entity, v_sources)
            report.versions_upserted += 1
            if seed_version.wikidata_qid:
                versions_by_qid[seed_version.wikidata_qid] = version
                version_work_key[seed_version.wikidata_qid] = wkey
            if year_conflict and entity is not None:
                if not _open_conflict_exists(session, version.version_id, "release_year"):
                    session.add(
                        Conflict(
                            entity_kind="version",
                            entity_id=version.version_id,
                            field="release_year",
                            values=[
                                {"value": seed_version.release_year, "source": "human:seed_slice"},
                                {
                                    "value": list(entity.publication_years),
                                    "source": f"wikidata:{entity.qid}#{P_PUBLICATION_DATE}",
                                },
                            ],
                        )
                    )
                    report.conflicts_opened += 1
    session.flush()

    work_qid_to_key = {w.wikidata_qid: k for k, w in works_by_key.items() if w.wikidata_qid}

    # In-pass dedup: two Wikidata statements can assert the same edge (e.g. P155 on the
    # sequel AND P156 on the original both derive one is_sequel_of). With autoflush off,
    # the pending row is invisible to _edge_exists — dedup within the pass explicitly.
    written_in_pass: set[tuple[str, Any, Any]] = set()

    def _write_edge(
        edge_type: EdgeType,
        src_kind: str,
        src_id: Any,
        dst_kind: str,
        dst_id: Any,
        qid: str,
        prop: str,
        label: str,
    ) -> None:
        key = (edge_type.value, src_id, dst_id)
        if key in written_in_pass or _edge_exists(session, edge_type.value, src_id, dst_id):
            return
        written_in_pass.add(key)
        session.add(
            Edge(
                edge_type=edge_type.value,
                src_kind=src_kind,
                src_id=src_id,
                dst_kind=dst_kind,
                dst_id=dst_id,
                confidence=Confidence.HIGH.value,  # authoritative structured source
                sources=sources_to_jsonb([_wikidata_ref(qid, prop, retrieved_at)]),
            )
        )
        report.edges_written += 1
        report.edge_labels.append(label)

    # 2. Edges — only what Wikidata asserts (P144/P4969/P155/P156).
    for qid, entity in entities.items():
        src_version = versions_by_qid.get(qid)
        # P144 based-on (and P4969 inverse): remake edge if both ends are slice versions;
        # work-level based_on if the target is a slice work (literary source).
        for target in entity.based_on:
            dst_version = versions_by_qid.get(target)
            if src_version is not None and dst_version is not None:
                _write_edge(
                    EdgeType.IS_REMAKE_OF,
                    "version",
                    src_version.version_id,
                    "version",
                    dst_version.version_id,
                    qid,
                    P_BASED_ON,
                    f"{qid} is_remake_of {target}",
                )
            elif src_version is not None and target in work_qid_to_key:
                src_work = works_by_key[version_work_key[qid]]
                dst_work = works_by_key[work_qid_to_key[target]]
                if src_work.work_id != dst_work.work_id:
                    _write_edge(
                        EdgeType.BASED_ON,
                        "work",
                        src_work.work_id,
                        "work",
                        dst_work.work_id,
                        qid,
                        P_BASED_ON,
                        f"{qid}.work based_on {target}",
                    )
            elif target not in versions_by_qid and target not in work_qid_to_key:
                report.discovered_unmatched_qids.append(target)
        for target in entity.derivative_works:  # inverse: target is derived from qid
            dst_version = versions_by_qid.get(target)
            if src_version is not None and dst_version is not None:
                _write_edge(
                    EdgeType.IS_REMAKE_OF,
                    "version",
                    dst_version.version_id,
                    "version",
                    src_version.version_id,
                    qid,
                    P_DERIVATIVE_WORK,
                    f"{target} is_remake_of {qid}",
                )
            elif qid in work_qid_to_key and target in versions_by_qid:
                # P4969 on a slice *work* (e.g. the novella): target's work is based_on it.
                src_work_b = works_by_key[work_qid_to_key[qid]]
                dst_work_b = works_by_key[version_work_key[target]]
                if src_work_b.work_id != dst_work_b.work_id:
                    _write_edge(
                        EdgeType.BASED_ON,
                        "work",
                        dst_work_b.work_id,
                        "work",
                        src_work_b.work_id,
                        qid,
                        P_DERIVATIVE_WORK,
                        f"{target}.work based_on {qid}",
                    )
            elif target not in versions_by_qid and target not in work_qid_to_key:
                report.discovered_unmatched_qids.append(target)
        # P155/P156 sequel ordering → work-level is_sequel_of.
        if src_version is not None:
            src_work = works_by_key[version_work_key[qid]]
            for target in entity.follows:
                if target in versions_by_qid:
                    prev_work = works_by_key[version_work_key[target]]
                    if prev_work.work_id != src_work.work_id:
                        _write_edge(
                            EdgeType.IS_SEQUEL_OF,
                            "work",
                            src_work.work_id,
                            "work",
                            prev_work.work_id,
                            qid,
                            P_FOLLOWS,
                            f"{qid}.work is_sequel_of {target}.work",
                        )
            for target in entity.followed_by:
                if target in versions_by_qid:
                    next_work = works_by_key[version_work_key[target]]
                    if next_work.work_id != src_work.work_id:
                        _write_edge(
                            EdgeType.IS_SEQUEL_OF,
                            "work",
                            next_work.work_id,
                            "work",
                            src_work.work_id,
                            qid,
                            P_FOLLOWED_BY,
                            f"{target}.work is_sequel_of {qid}.work",
                        )

    report.discovered_unmatched_qids = sorted(set(report.discovered_unmatched_qids))
    session.flush()
    return report
