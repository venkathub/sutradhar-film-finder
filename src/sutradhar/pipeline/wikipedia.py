"""Wikipedia plot fetch (P1 task 7) — the P2 embedding corpus + task-11 extraction input.

MediaWiki **Action API only** (never HTML scraping; DATA_SOURCES.md): one call per page with
``prop=extracts|revisions|info`` (TextExtracts plaintext with ``== wiki ==`` section markers,
latest revision id, canonical URL). Article titles come from the Wikidata **sitelinks already
captured** in the task-4 snapshot — zero extra Wikidata calls.

Every ``plot_texts`` row is revision-pinned and carries ``license='CC BY-SA 4.0'`` + the page
URL (attribution obligations recorded per row; docs/LICENSING.md). Text stored = lead + a
Plot/Synopsis-type section when one is found, else the full extract (content, not facts).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.config import Settings, get_settings
from sutradhar.graph.schema import PlotText, Version

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"

# Plot-ish section headings (casefolded). English + the slice's native wikis where the
# heading is reliably known; unmatched articles fall back to the full extract.
PLOT_HEADINGS = {
    "plot",
    "plot summary",
    "synopsis",
    "story",
    "storyline",
    "കഥ",  # ml
    "കഥാസംഗ്രഹം",  # ml
    "கதை",  # ta
    "కథ",  # te
    "कथानक",  # hi
    "कथा",  # hi
    "ಕಥಾವಸ್ತು",  # kn
    "কাহিনি",  # bn
    "কাহিনী",  # bn
}

_HEADING_RE = re.compile(r"^==\s*(?P<title>[^=].*?)\s*==\s*$", re.MULTILINE)


class WikiPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    lang: str  # wiki language code (en, ml, ta, …)
    title: str
    extract: str
    revision_id: str
    url: str


def parse_page(lang: str, raw: dict[str, Any]) -> WikiPage | None:
    """Parse one Action-API ``query`` response (single page). None for missing pages."""
    pages = raw.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or "extract" not in page:
        return None
    revisions = page.get("revisions", [])
    if not revisions:
        return None
    return WikiPage(
        lang=lang,
        title=page["title"],
        extract=page["extract"],
        revision_id=str(revisions[0]["revid"]),
        url=page.get("canonicalurl") or page.get("fullurl") or "",
    )


def split_lead_and_plot(extract: str) -> tuple[str, str | None]:
    """Split a wiki-marked plaintext extract into (lead, plot-section-or-None)."""
    matches = list(_HEADING_RE.finditer(extract))
    lead = extract[: matches[0].start()].strip() if matches else extract.strip()
    for i, m in enumerate(matches):
        # Only top-level "== X ==" headings match the regex (=== sub-sections don't).
        if m.group("title").casefold() in PLOT_HEADINGS:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(extract)
            return lead, extract[m.end() : end].strip() or None
    return lead, None


def plot_text_for_storage(page: WikiPage) -> str:
    """Lead + plot section when found; full extract otherwise (P2 chunks it anyway)."""
    lead, plot = split_lead_and_plot(page.extract)
    if plot is not None:
        return f"{lead}\n\n{plot}".strip()
    return page.extract.strip()


def sitelinks_from_entities(
    raw_entities: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """QID → {wiki language: article title} from a task-4 wbgetentities snapshot."""
    result: dict[str, dict[str, str]] = {}
    for qid, entity in raw_entities.items():
        links: dict[str, str] = {}
        for site, link in entity.get("sitelinks", {}).items():
            if site.endswith("wiki") and site.count("wiki") == 1:  # e.g. enwiki, mlwiki
                links[site.removesuffix("wiki")] = link["title"]
        if links:
            result[qid] = links
    return result


class WikipediaClient:
    """Per-language-wiki Action API client (WMF UA policy + Retry-After etiquette)."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        s = settings if settings is not None else get_settings()
        self._url_template = s.wikipedia_api_url  # contains a {lang} placeholder
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers={"User-Agent": s.http_user_agent, "Accept-Encoding": "gzip"},
            timeout=30.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def get_page(self, lang: str, title: str) -> dict[str, Any]:
        """One call: plaintext extract (wiki section markers) + latest revision + URL."""
        url = self._url_template.format(lang=lang)
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|revisions|info",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "rvprop": "ids",
            "inprop": "url",
            "formatversion": "2",
            "format": "json",
            "redirects": "1",
        }
        for attempt in range(self._max_retries + 1):
            response = self._client.get(url, params=params)
            if response.status_code in (429, 503) and attempt < self._max_retries:
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload
        raise RuntimeError("unreachable")  # pragma: no cover


# --- Loading into plot_texts ---


@dataclass
class PlotsReport:
    pages_seen: int = 0
    rows_new: int = 0
    rows_repinned: int = 0  # revision changed → text + revision updated
    rows_unchanged: int = 0
    versions_without_sitelink: list[str] = field(default_factory=list)


def load_plots(
    session: Session,
    pages_by_qid: dict[str, list[WikiPage]],
    retrieved_at: datetime | None = None,
) -> PlotsReport:
    """Upsert plot rows keyed ``(version_id, 'wikipedia', page-language)``. Idempotent."""
    retrieved_at = retrieved_at or datetime.now(tz=UTC)
    report = PlotsReport()

    versions = session.scalars(select(Version)).all()
    by_qid = {v.wikidata_qid: v for v in versions if v.wikidata_qid is not None}
    for v in versions:
        if v.wikidata_qid is None:
            report.versions_without_sitelink.append(f"{v.title} ({v.language})")

    for qid, pages in pages_by_qid.items():
        version = by_qid.get(qid)
        if version is None:
            continue
        for page in pages:
            report.pages_seen += 1
            text = plot_text_for_storage(page)
            if not text:
                continue
            existing = session.scalars(
                select(PlotText).where(
                    PlotText.version_id == version.version_id,
                    PlotText.source == "wikipedia",
                    PlotText.language == page.lang,
                )
            ).first()
            if existing is None:
                session.add(
                    PlotText(
                        version_id=version.version_id,
                        source="wikipedia",
                        language=page.lang,
                        text=text,
                        source_url=page.url,
                        revision_id=page.revision_id,
                        license=WIKIPEDIA_LICENSE,
                        retrieved_at=retrieved_at,
                    )
                )
                session.flush()
                report.rows_new += 1
            elif existing.revision_id != page.revision_id:
                existing.text = text
                existing.revision_id = page.revision_id
                existing.source_url = page.url
                existing.retrieved_at = retrieved_at
                report.rows_repinned += 1
            else:
                report.rows_unchanged += 1

    session.flush()
    return report
