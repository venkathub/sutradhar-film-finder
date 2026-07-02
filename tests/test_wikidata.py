"""Unit tests for the Wikidata connector (P1 task 4) — no live API, no DB.

Parse tests run against ``tests/fixtures/wikidata/entities_sample.json``, a trimmed capture
of the real 2026-07-02 live-run snapshot (labels/descriptions + spine claims only).
Client etiquette (User-Agent, Retry-After backoff) is tested with ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.pipeline.wikidata import (
    WikidataClient,
    load_snapshot,
    parse_entity,
    write_snapshot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "wikidata" / "entities_sample.json"


@pytest.fixture(scope="module")
def sample() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


# --- parse_entity against real captured claims ---


def test_parse_drishyam_hi_full_claims(sample: dict[str, dict[str, Any]]) -> None:
    """Q19824636 (Drishyam 2015 hi): P144 -> ml original, external ids, year, language."""
    e = parse_entity(sample["Q19824636"])
    assert e.qid == "Q19824636"
    assert "Q15401703" in e.based_on
    assert 2015 in e.publication_years
    assert "Q1568" in e.original_language_qids  # Hindi
    assert e.imdb_id is not None and e.imdb_id.startswith("tt")
    assert e.tmdb_id is None or isinstance(e.tmdb_id, int)


def test_parse_derivative_work_inverse(sample: dict[str, dict[str, Any]]) -> None:
    """Q15401703 (Drishyam ml): P4969 covers Papanasam even though Papanasam lacks P144."""
    original = parse_entity(sample["Q15401703"])
    assert "Q18129183" in original.derivative_works
    papanasam = parse_entity(sample["Q18129183"])
    assert papanasam.based_on == ()  # the gap is real; the inverse property fills it


def test_parse_sequel_ordering(sample: dict[str, dict[str, Any]]) -> None:
    """Drishyam 2 ml follows Drishyam ml (P155) — the is_sequel_of derivation input."""
    sequel = parse_entity(sample["Q102036246"])
    assert "Q15401703" in sequel.follows


def test_parse_novella_is_not_a_film(sample: dict[str, dict[str, Any]]) -> None:
    e = parse_entity(sample["Q11169650"])
    assert e.imdb_id is None
    assert 1917 in e.publication_years


def test_parse_devdas_2002_based_on_novella(sample: dict[str, dict[str, Any]]) -> None:
    e = parse_entity(sample["Q247854"])
    assert "Q11169650" in e.based_on  # work-level based_on, not a remake edge


def test_parse_chandramukhi_asserts_direct_original(sample: dict[str, dict[str, Any]]) -> None:
    """Wikidata P144 on Chandramukhi points at Manichitrathazhu directly — the curated
    proximate source (Apthamitra) is NOT asserted; that's extraction+review's job."""
    e = parse_entity(sample["Q1193810"])
    assert "Q3530081" in e.based_on
    assert "Q4782241" not in e.based_on


def test_parse_missing_claims_yield_empty() -> None:
    e = parse_entity({"id": "Q1", "labels": {}, "descriptions": {}, "claims": {}})
    assert e.based_on == () and e.publication_years == () and e.imdb_id is None


# --- Client etiquette (MockTransport; zero live calls) ---


def _settings() -> Settings:
    return Settings(
        WIKIDATA_API_URL="https://wd.test/w/api.php",
        WIKIDATA_SPARQL_URL="https://wd.test/sparql",
        HTTP_USER_AGENT="SutradharBot/test (contact@example)",
    )


def test_client_sends_descriptive_user_agent() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, json={"entities": {}})

    client = WikidataClient(_settings(), transport=httpx.MockTransport(handler))
    client.get_entities(["Q1"])
    client.close()
    assert seen["ua"] == "SutradharBot/test (contact@example)"  # WMF UA policy


def test_client_honors_retry_after_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("sutradhar.pipeline.wikidata.time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"entities": {"Q1": {"id": "Q1", "claims": {}}}})

    client = WikidataClient(_settings(), transport=httpx.MockTransport(handler))
    entities = client.get_entities(["Q1"])
    client.close()
    assert calls["n"] == 2 and sleeps == [3.0]
    assert "Q1" in entities


def test_client_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sutradhar.pipeline.wikidata.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = WikidataClient(_settings(), transport=httpx.MockTransport(handler), max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_entities(["Q1"])
    client.close()


def test_sparql_backlinks_returns_qids_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "P144" in request.url.params["query"]  # phase-1 discovery query
        return httpx.Response(
            200,
            json={
                "results": {
                    "bindings": [
                        {"item": {"value": "http://www.wikidata.org/entity/Q999"}},
                        {"item": {"value": "http://www.wikidata.org/entity/Q999"}},  # dupe
                    ]
                }
            },
        )

    client = WikidataClient(_settings(), transport=httpx.MockTransport(handler))
    assert client.discover_backlinks(["Q15401703"]) == ["Q999"]
    client.close()


def test_wbgetentities_batches_over_50(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = request.url.params["ids"].split("|")
        batch_sizes.append(len(ids))
        return httpx.Response(200, json={"entities": {i: {"id": i, "claims": {}} for i in ids}})

    client = WikidataClient(_settings(), transport=httpx.MockTransport(handler))
    entities = client.get_entities([f"Q{i}" for i in range(1, 61)])
    client.close()
    assert batch_sizes == [50, 10] and len(entities) == 60


# --- Snapshot hash-recording round-trip ---


def test_snapshot_roundtrip_verifies_hash(tmp_path: Path) -> None:
    payload = {"entities": {"Q1": {"id": "Q1"}}}
    write_snapshot(tmp_path, "entities", payload)
    assert load_snapshot(tmp_path, "entities") == payload


def test_snapshot_tamper_detected(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "entities", {"a": 1})
    (tmp_path / "entities.json").write_text('{"a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_snapshot(tmp_path, "entities")
