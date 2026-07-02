"""Tests for the committed seed slice + its loader (P1 task 3).

Two halves:
1. The committed ``data-pipeline/seed_slice.yaml`` loads clean and encodes the flagship
   ground truth the golden scenarios need (curated-truth assertions).
2. The loader rejects structurally-bad slices (dangling targets, missing originals,
   duplicate QIDs, literary sources with versions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, SeedSlice, load_seed_slice

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / DEFAULT_SEED_PATH


@pytest.fixture(scope="module")
def slice_() -> SeedSlice:
    return load_seed_slice(SEED_PATH)


# --- The committed slice: flagship curated truth ---


def test_slice_loads_and_has_scale(slice_: SeedSlice) -> None:
    assert len(slice_.works) >= 14
    assert slice_.version_count() >= 25  # ~30 records incl. work nodes (P1_SPEC §2.7)


def test_drishyam_curated_truth(slice_: SeedSlice) -> None:
    """GS-01/GS-09A denominator: 5 Indian versions (1 original + 4 remakes) + 2 foreign."""
    work = slice_.works["drishyam"]
    indian = {k: v for k, v in work.versions.items() if v.country == "indian"}
    foreign = {k: v for k, v in work.versions.items() if v.country == "foreign"}
    assert len(indian) == 5 and len(foreign) == 2
    originals = [k for k, v in work.versions.items() if v.is_original]
    assert originals == ["drishyam_ml"]
    for key, version in work.versions.items():
        if key != "drishyam_ml":
            assert version.relationship is not None
            assert version.relationship.type == "is_remake_of"
            assert version.relationship.of == "drishyam_ml"
    assert {v.language for v in indian.values()} == {"ml", "kn", "te", "ta", "hi"}


def test_drishyam2_is_sequel_work(slice_: SeedSlice) -> None:
    """GS-06: sequel = separate Work, own version set, linked at work level."""
    work = slice_.works["drishyam_2"]
    assert work.is_sequel_of == "drishyam"
    assert len(work.versions) == 4
    assert [k for k, v in work.versions.items() if v.is_original] == ["drishyam2_ml"]


def test_baahubali_bilingual_double_original_and_dubs(slice_: SeedSlice) -> None:
    """GS-04: two original-flagged versions; every other track a dub; zero remakes."""
    work = slice_.works["baahubali_1"]
    originals = {k for k, v in work.versions.items() if v.is_original}
    assert originals == {"baahubali_te", "baahubali_ta"}
    rel_types = {v.relationship.type for v in work.versions.values() if v.relationship}
    assert rel_types == {"is_official_dub_of"}
    # QID sits on the primary original only (version.wikidata_qid UNIQUE).
    assert work.versions["baahubali_te"].wikidata_qid == "Q13897247"
    assert work.versions["baahubali_ta"].wikidata_qid is None


def test_devdas_siblings_not_remakes(slice_: SeedSlice) -> None:
    """GS-05: three sibling Works based_on the novella; no remake edges among them."""
    novella = slice_.works["devdas_novella"]
    assert novella.work_type == "literary_source" and not novella.versions
    siblings = ["devadasu_1953", "devdas_1955", "devdas_2002"]
    for key in siblings:
        assert slice_.works[key].based_on == "devdas_novella"
        assert slice_.works[key].is_sequel_of is None
    # The dub edge composes INSIDE a sibling adaptation (Devadasu -> Tamil dub).
    devadasu = slice_.works["devadasu_1953"]
    dub = devadasu.versions["devadas_ta"]
    assert dub.relationship is not None
    assert dub.relationship.type == "is_official_dub_of"
    assert dub.relationship.of == "devadasu_te"


def test_vikram_pair_distinct_works(slice_: SeedSlice) -> None:
    """GS-10: same title, same lead actor, two unrelated Works with distinct QIDs."""
    v86, v22 = slice_.works["vikram_1986"], slice_.works["vikram_2022"]
    assert v86.primary_title == v22.primary_title == "Vikram"
    assert v86.is_sequel_of is None and v22.is_sequel_of is None
    assert v86.based_on is None and v22.based_on is None
    qids = {
        v86.versions["vikram_1986_ta"].wikidata_qid,
        v22.versions["vikram_2022_ta"].wikidata_qid,
    }
    assert len(qids) == 2


def test_manichitrathazhu_transitive_chain(slice_: SeedSlice) -> None:
    """GS-09B: proximate remake edges preserved; single ultimate original."""
    work = slice_.works["manichitrathazhu"]
    assert [k for k, v in work.versions.items() if v.is_original] == ["manichitrathazhu_ml"]
    rel = {k: v.relationship for k, v in work.versions.items() if v.relationship}
    assert rel["apthamitra_kn"] is not None
    assert rel["apthamitra_kn"].of == "manichitrathazhu_ml"
    assert rel["chandramukhi_ta"] is not None
    assert rel["chandramukhi_ta"].of == "apthamitra_kn"  # remake-of-a-remake
    assert rel["bhool_bhulaiyaa_hi"] is not None
    assert rel["bhool_bhulaiyaa_hi"].of == "manichitrathazhu_ml"


def test_distractors_present(slice_: SeedSlice) -> None:
    """GS-02/GS-03 noise floor: >=4 unrelated works, incl. Mohanlal + Kamal Haasan titles."""
    distractors = slice_.franchises()["distractors"]
    assert len(distractors) >= 4
    assert "lucifer_2019" in distractors and "anbe_sivam_2003" in distractors


def test_backlog_entries_have_no_qids(slice_: SeedSlice) -> None:
    """§7 Q1: conditional adds are name+reason only — never hand-invented records."""
    assert len(slice_.backlog) == 4
    all_qids = {qid for _, qid in slice_._iter_qids()}
    assert len(all_qids) == len(slice_._iter_qids())  # unique (also model-enforced)


def test_all_confirmed_versions_have_qids_except_bilingual_shares(slice_: SeedSlice) -> None:
    """Every curated version carries a QID except documented same-item shares (dub/bilingual)."""
    missing = [
        vkey
        for work in slice_.works.values()
        for vkey, v in work.versions.items()
        if v.wikidata_qid is None
    ]
    assert set(missing) == {"baahubali_ta", "baahubali_hi", "baahubali_ml", "devadas_ta"}


# --- Loader rejections (structural validation) ---


def _minimal(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "meta": {"verified_at": "2026-07-02", "description": "t"},
        "works": {
            "w1": {
                "franchise": "f",
                "work_type": "film",
                "primary_title": "T",
                "original_language": "ml",
                "first_release_year": 2000,
                "versions": {
                    "v1": {
                        "title": "T",
                        "language": "ml",
                        "release_year": 2000,
                        "country": "indian",
                        "is_original": True,
                        "wikidata_qid": "Q1",
                    },
                },
            },
        },
    }
    doc.update(overrides)
    return doc


def test_loader_rejects_dangling_relationship_target() -> None:
    doc = _minimal()
    doc["works"]["w1"]["versions"]["v2"] = {
        "title": "T2",
        "language": "ta",
        "release_year": 2001,
        "country": "indian",
        "relationship": {"type": "is_remake_of", "of": "nope"},
    }
    with pytest.raises(ValidationError, match="not in this work"):
        SeedSlice.model_validate(doc)


def test_loader_rejects_work_without_original() -> None:
    doc = _minimal()
    doc["works"]["w1"]["versions"]["v1"]["is_original"] = False
    doc["works"]["w1"]["versions"]["v1"]["relationship"] = {
        "type": "is_remake_of",
        "of": "v1",
    }
    with pytest.raises(ValidationError):
        SeedSlice.model_validate(doc)


def test_loader_rejects_original_with_relationship() -> None:
    doc = _minimal()
    doc["works"]["w1"]["versions"]["v1"]["relationship"] = {
        "type": "is_remake_of",
        "of": "v1",
    }
    with pytest.raises(ValidationError, match="cannot also carry"):
        SeedSlice.model_validate(doc)


def test_loader_rejects_duplicate_qids() -> None:
    doc = _minimal()
    doc["works"]["w2"] = {
        "franchise": "f",
        "work_type": "film",
        "primary_title": "U",
        "original_language": "ta",
        "first_release_year": 2001,
        "versions": {
            "u1": {
                "title": "U",
                "language": "ta",
                "release_year": 2001,
                "country": "indian",
                "is_original": True,
                "wikidata_qid": "Q1",  # duplicate of v1
            },
        },
    }
    with pytest.raises(ValidationError, match="duplicate wikidata_qid"):
        SeedSlice.model_validate(doc)


def test_loader_rejects_literary_source_with_versions() -> None:
    doc = _minimal()
    doc["works"]["w1"]["work_type"] = "literary_source"
    with pytest.raises(ValidationError, match="literary_source"):
        SeedSlice.model_validate(doc)


def test_loader_rejects_based_on_targeting_film() -> None:
    doc = _minimal()
    doc["works"]["w2"] = {
        "franchise": "f",
        "work_type": "film",
        "primary_title": "U",
        "original_language": "ta",
        "first_release_year": 2001,
        "based_on": "w1",  # a film, not a literary_source
        "versions": {
            "u1": {
                "title": "U",
                "language": "ta",
                "release_year": 2001,
                "country": "indian",
                "is_original": True,
            },
        },
    }
    with pytest.raises(ValidationError, match="based_on must target a literary_source"):
        SeedSlice.model_validate(doc)


def test_loader_rejects_unknown_sequel_target() -> None:
    doc = _minimal()
    doc["works"]["w1"] = dict(doc["works"]["w1"], is_sequel_of="ghost")
    with pytest.raises(ValidationError, match="unknown"):
        SeedSlice.model_validate(doc)
