"""Unit tests for the precedence table implementation (P1 task 5, DATA_SOURCES.md rows)."""

from __future__ import annotations

import pytest

from sutradhar.pipeline.precedence import Observation, resolve_field, union_values


def test_external_id_hub_wikidata_wins() -> None:
    r = resolve_field(
        "external_id",
        [Observation("tt3417422", "wikidata"), Observation("tt9999999", "tmdb")],
    )
    assert r.value == "tt3417422" and r.confidence == "HIGH"
    assert r.conflict == "resolved" and r.resolution == {
        "by": "rule",
        "rule": "hub:wikidata",
        "chosen_value": "tt3417422",
    }


def test_canonical_title_tmdb_primary_corroborated() -> None:
    r = resolve_field(
        "canonical_title",
        [Observation("Drishyam", "tmdb"), Observation("Drishyam", "wikidata")],
    )
    assert r.value == "Drishyam" and r.confidence == "HIGH" and r.conflict == "none"


def test_canonical_title_single_source_is_medium() -> None:
    r = resolve_field("canonical_title", [Observation("Drishyam", "tmdb")])
    assert r.confidence == "MEDIUM" and r.conflict == "none"


def test_original_language_tmdb_wins_disagreement_recorded() -> None:
    """Table rule: 'TMDB' — the rule decides, but both values are preserved (resolved)."""
    r = resolve_field(
        "original_language",
        [Observation("ml", "human"), Observation("te", "tmdb")],
    )
    assert r.value == "te"  # TMDB primary
    assert r.conflict == "resolved"
    assert {v["value"] for v in r.conflict_values} == {"ml", "te"}
    assert r.resolution is not None and r.resolution["rule"] == "primary:tmdb"


def test_release_year_all_agree_high() -> None:
    r = resolve_field(
        "release_year",
        [Observation(2015, "tmdb"), Observation(2015, "wikidata"), Observation(2015, "human")],
    )
    assert r.value == 2015 and r.confidence == "HIGH" and r.conflict == "none"


def test_release_year_majority_resolves_but_flags() -> None:
    r = resolve_field(
        "release_year",
        [Observation(2015, "tmdb"), Observation(2015, "wikidata"), Observation(2014, "imdb")],
    )
    assert r.value == 2015 and r.conflict == "resolved" and r.confidence == "MEDIUM"
    assert r.resolution is not None and r.resolution["rule"] == "majority"


def test_release_year_split_opens_conflict() -> None:
    """The table's 'if split → conflict queue': rule-undecidable stays OPEN (view-hiding)."""
    r = resolve_field(
        "release_year",
        [Observation(2015, "tmdb"), Observation(2014, "human")],
    )
    assert r.conflict == "open"
    assert {v["value"] for v in r.conflict_values} == {2014, 2015}


def test_director_tmdb_primary() -> None:
    r = resolve_field(
        "director",
        [Observation("Jeethu Joseph", "wikidata"), Observation("Jeethu Joseph", "tmdb")],
    )
    assert r.confidence == "HIGH" and r.conflict == "none"


def test_aka_union_dedupes_and_grades() -> None:
    values = union_values(
        [
            Observation("Papanasam", "imdb"),
            Observation("Papanasam", "tmdb"),
            Observation("Paapanaasam", "imdb"),
        ]
    )
    assert values == [("Papanasam", "HIGH"), ("Paapanaasam", "MEDIUM")]


def test_union_via_resolve_field_rejected() -> None:
    with pytest.raises(ValueError, match="multi-valued"):
        resolve_field("aka_title", [Observation("x", "imdb")])


def test_empty_observations_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        resolve_field("release_year", [])
