"""Unit tests for the Wikipedia connector (P1 task 7) — no live API, no DB.

Parse/section tests run against ``tests/fixtures/wikipedia/pages_sample.json``, a trimmed
capture of the real 2026-07-02 live snapshot (4 pages, en + ml).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.pipeline.wikipedia import (
    WIKIPEDIA_LICENSE,
    WikipediaClient,
    parse_page,
    plot_text_for_storage,
    sitelinks_from_entities,
    split_lead_and_plot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "wikipedia" / "pages_sample.json"
WD_FIXTURE = Path(__file__).parent / "fixtures" / "wikidata" / "entities_sample.json"


@pytest.fixture(scope="module")
def sample() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


# --- parse_page / plot extraction ---


def test_parse_real_page_pins_revision(sample: dict[str, dict[str, Any]]) -> None:
    entry = sample["Q15401703|en"]
    page = parse_page(entry["lang"], entry["response"])
    assert page is not None
    assert page.lang == "en" and page.title == "Drishyam"
    assert page.revision_id.isdigit()
    assert page.url.startswith("https://en.wikipedia.org/")
    assert "== Plot ==" in page.extract


def test_parse_native_script_page(sample: dict[str, dict[str, Any]]) -> None:
    entry = sample["Q15401703|ml"]
    page = parse_page(entry["lang"], entry["response"])
    assert page is not None and page.lang == "ml"
    assert page.extract  # native-script prose present


def test_parse_missing_page_returns_none() -> None:
    raw = {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    assert parse_page("en", raw) is None


def test_split_lead_and_plot_english(sample: dict[str, dict[str, Any]]) -> None:
    entry = sample["Q15401703|en"]
    page = parse_page(entry["lang"], entry["response"])
    assert page is not None
    lead, plot = split_lead_and_plot(page.extract)
    assert "Drishyam" in lead and "==" not in lead
    assert plot is not None and len(plot) > 200


def test_split_without_plot_heading_falls_back() -> None:
    text = "Lead paragraph.\n\n== Cast ==\nSomeone.\n\n== Reception ==\nGood."
    lead, plot = split_lead_and_plot(text)
    assert lead == "Lead paragraph." and plot is None
    # storage falls back to the full extract
    from sutradhar.pipeline.wikipedia import WikiPage

    page = WikiPage(lang="en", title="X", extract=text, revision_id="1", url="u")
    assert plot_text_for_storage(page) == text


def test_split_synonym_headings() -> None:
    text = "Lead.\n\n== Synopsis ==\nThe story goes.\n\n== Music ==\nSongs."
    lead, plot = split_lead_and_plot(text)
    assert plot == "The story goes."


def test_subheadings_do_not_match() -> None:
    text = "Lead.\n\n=== Plot ===\nnested\n\n== Music ==\nSongs."
    _, plot = split_lead_and_plot(text)
    assert plot is None  # === Plot === is a sub-section, not a top-level heading


# --- sitelinks from the task-4 snapshot ---


def test_sitelinks_from_entities_snapshot() -> None:
    entities: dict[str, dict[str, Any]] = json.loads(WD_FIXTURE.read_text(encoding="utf-8"))
    # The trimmed wikidata fixture strips sitelinks; verify against a synthetic entity too.
    synthetic = {
        "Q1": {
            "sitelinks": {
                "enwiki": {"title": "Drishyam"},
                "mlwiki": {"title": "ദൃശ്യം"},
                "enwikiquote": {"title": "ignored"},  # not a *wiki article site
                "commonswiki": {"title": "kept-by-suffix-rule"},
            }
        }
    }
    links = sitelinks_from_entities({**entities, **synthetic})
    assert links["Q1"]["en"] == "Drishyam" and links["Q1"]["ml"] == "ദൃശ്യം"
    assert "enwikiquote" not in links["Q1"] and "en" in links["Q1"]


def test_license_constant() -> None:
    assert WIKIPEDIA_LICENSE == "CC BY-SA 4.0"


# --- client URL templating + etiquette ---


def test_client_fills_lang_placeholder() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, json={"query": {"pages": []}})

    client = WikipediaClient(
        Settings(
            WIKIPEDIA_API_URL="https://{lang}.wp.test/w/api.php",
            HTTP_USER_AGENT="SutradharBot/test",
        ),
        transport=httpx.MockTransport(handler),
    )
    client.get_page("ml", "ദൃശ്യം")
    client.close()
    assert seen["host"] == "ml.wp.test"
    assert seen["ua"] == "SutradharBot/test"


def test_client_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("sutradhar.pipeline.wikipedia.time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"query": {"pages": []}})

    client = WikipediaClient(
        Settings(WIKIPEDIA_API_URL="https://{lang}.wp.test/w/api.php"),
        transport=httpx.MockTransport(handler),
    )
    client.get_page("en", "X")
    client.close()
    assert sleeps == [2.0]
