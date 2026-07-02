"""Unit tests for the TMDB connector (P1 task 5) — no live API, no DB.

Parse tests run against ``tests/fixtures/tmdb/movies_sample.json``, a trimmed capture of the
real 2026-07-02 live snapshot (6 movies). Client auth modes tested via MockTransport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.pipeline.normalize import match_key
from sutradhar.pipeline.tmdb import TMDBClient, parse_movie

FIXTURE = Path(__file__).parent / "fixtures" / "tmdb" / "movies_sample.json"


@pytest.fixture(scope="module")
def sample() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


# --- parse_movie ---


def test_parse_drishyam_ml(sample: dict[str, dict[str, Any]]) -> None:
    m = parse_movie(sample["244049"])
    assert m.tmdb_id == 244049
    assert m.original_language == "ml"
    assert m.release_year == 2013
    assert m.imdb_id is not None and m.imdb_id.startswith("tt")
    assert any(t.kind == "canonical" for t in m.titles)
    directors = [c for c in m.credits if c.role_kind == "director"]
    assert any("Jeethu" in d.name for d in directors)


def test_parse_lead_vs_support_billing(sample: dict[str, dict[str, Any]]) -> None:
    m = parse_movie(sample["244049"])
    leads = [c for c in m.credits if c.role_kind == "lead"]
    assert leads and all(c.billing_order is not None and c.billing_order < 5 for c in leads)
    assert any("Mohanlal" in c.name for c in leads)


def test_parse_baahubali_has_ml_translation(sample: dict[str, dict[str, Any]]) -> None:
    """The ml translation is what maps onto the QID-less ml dub track (task-5 mapping)."""
    m = parse_movie(sample["256040"])
    ml = [t for t in m.titles if t.kind == "translation" and t.language == "ml"]
    assert len(ml) == 1


def test_parse_alternative_titles_kind_aka(sample: dict[str, dict[str, Any]]) -> None:
    m = parse_movie(sample["352173"])  # Drishyam 2015 hi
    assert any(t.kind == "aka" for t in m.titles)


def test_parse_missing_release_date_yields_none() -> None:
    m = parse_movie({"id": 1, "original_title": "X", "original_language": "ta", "release_date": ""})
    assert m.release_year is None and m.titles[0].kind == "canonical"


# --- client auth modes ---


def _settings(key: str) -> Settings:
    return Settings(
        TMDB_API_URL="https://tmdb.test/3",
        TMDB_API_KEY=key,
        HTTP_USER_AGENT="SutradharBot/test",
    )


def test_client_v4_bearer_auth() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"id": 1})

    client = TMDBClient(
        _settings("eyJhbGciOiJIUzI1NiJ9.x.y"), transport=httpx.MockTransport(handler)
    )
    client.get_movie(1)
    client.close()
    assert seen["auth"] == "Bearer eyJhbGciOiJIUzI1NiJ9.x.y"
    assert "api_key" not in seen["params"]
    assert seen["params"]["append_to_response"] == "translations,alternative_titles,credits"


def test_client_v3_api_key_param() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"id": 1})

    client = TMDBClient(_settings("abc123def456"), transport=httpx.MockTransport(handler))
    client.get_movie(1)
    client.close()
    assert seen["auth"] is None
    assert seen["params"]["api_key"] == "abc123def456"


def test_client_requires_key() -> None:
    from sutradhar.config import ConfigError

    with pytest.raises(ConfigError, match="TMDB_API_KEY"):
        TMDBClient(Settings(TMDB_API_KEY=None, _env_file=None))  # type: ignore[call-arg]


def test_client_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("sutradhar.pipeline.tmdb.time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"id": 7})

    client = TMDBClient(_settings("k"), transport=httpx.MockTransport(handler))
    assert client.get_movie(7)["id"] == 7
    client.close()
    assert sleeps == [1.0]


# --- interim match_key (upgraded in task 8) ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Papanasam", "papanasam"),
        ("K.G.F: Chapter 1", "k g f chapter 1"),
        ("Bāhubali: The Beginning", "bahubali the beginning"),
        ("  Drishyam   2  ", "drishyam 2"),
    ],
)
def test_interim_match_key(raw: str, expected: str) -> None:
    assert match_key(raw) == expected


def test_match_key_idempotent() -> None:
    key = match_key("Bāhubali: The Beginning")
    assert match_key(key) == key
