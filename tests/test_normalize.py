"""GS-11 unit suite: cross-script match_key + fuzzy resolution (P1 task 8, DEC-P1-5).

The named golden scenario this gates: a user types a perturbed/romanized/native-script title
and it must resolve to the intended film's key — while distractors stay below threshold.
All deterministic; no neural op (laptop/CI-safe).
"""

from __future__ import annotations

import pytest

from sutradhar.pipeline.normalize import (
    MATCH_THRESHOLD,
    best_matches,
    detect_script,
    match_key,
)

# The slice's canonical keys (what the version_title index holds after rekey).
INDEX = [
    match_key(t)
    for t in (
        "Drishyam",
        "Drishya",
        "Drushyam",
        "Papanasam",
        "Baahubali: The Beginning",
        "Chandramukhi",
        "Manichitrathazhu",
        "Bhool Bhulaiyaa",
        "Vikram",
        "Devdas",
        "Sholay",
        "Magadheera",
        "Anbe Sivam",
        "Lucifer",
        "K.G.F: Chapter 1",
    )
]


# --- match_key determinism + folding ---


def test_match_key_idempotent() -> None:
    for title in ("Papanasam", "பாபநாசம்", "दृश्यम", "K.G.F: Chapter 1"):
        key = match_key(title)
        assert match_key(key) == key


def test_vowel_length_collapse() -> None:
    assert match_key("Paapanaasam") == match_key("Papanasam") == "papanasam"
    assert match_key("Baahubali") == match_key("Bahubali")


def test_non_indic_passthrough() -> None:
    assert match_key("Sheep Without a Shepherd") == "shep without a shepherd"
    assert match_key("Sholay") == "sholay"


# --- script detection ---


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("दृश्यम", "deva"),
        ("பாபநாசம்", "taml"),
        ("ദൃശ്യം", "mlym"),
        ("దృశ్యం", "telu"),
        ("ದೃಶ್ಯ", "knda"),
        ("দেবদাস", "beng"),
        ("误杀", "hani"),
        ("Drishyam", "latn"),
        ("Drishyam 2", "latn"),
    ],
)
def test_detect_script(text: str, script: str) -> None:
    assert detect_script(text) == script


# --- native script → key equivalence (the heart of GS-11) ---


@pytest.mark.parametrize(
    ("native", "english"),
    [
        ("ദൃശ്യം", "Drishyam"),  # ml — exact key equality
        ("दृश्यम", "Drishyam"),  # hi
        ("ದೃಶ್ಯ", "Drishya"),  # kn
    ],
)
def test_native_script_exact_key_equivalence(native: str, english: str) -> None:
    assert match_key(native) == match_key(english)


@pytest.mark.parametrize(
    ("native", "english_key"),
    [
        ("பாபநாசம்", "papanasam"),  # ta — 1-char romanization drift, fuzzy resolves
        ("சந்திரமுகி", "chandramukhi"),
        ("భూల్ భులయ్యా", "bhol bhulaiya"),
    ],
)
def test_native_script_resolves_via_fuzzy(native: str, english_key: str) -> None:
    matches = best_matches(match_key(native), INDEX)
    assert matches, f"no match above threshold for {native}"
    assert matches[0][0] == english_key


def test_telugu_native_hits_whole_drishyam_family() -> None:
    """దృశ్యం romanizes to drishyam (the ml key) while the te popular spelling is
    drushyam — BOTH must surface above threshold; disambiguation is the graph's job
    (repository qualifies by language/year), not the key's."""
    matches = best_matches(match_key("దృశ్యం"), INDEX, limit=10)
    keys = {k for k, _ in matches}
    assert {"drishyam", "drushyam"} <= keys


# --- GS-11 perturbations (typos, vowel stretch, popular variants) ---


@pytest.mark.parametrize(
    ("query", "expected_key"),
    [
        ("Papanaasam", "papanasam"),  # vowel stretch → exact after collapse
        ("Papansam", "papanasam"),  # dropped char
        ("Papanasm", "papanasam"),  # transposed/dropped
        ("Drishyaam", "drishyam"),
        ("Drushyam", "drushyam"),  # exact (te popular spelling is in the index)
        ("Drishyam", "drishyam"),
        ("Chandramuki", "chandramukhi"),  # 1-char typo
        ("Manichitrathazu", "manichitrathazhu"),  # dropped h
        ("Bhool Bhulaiya", "bhol bhulaiya"),
        ("KGF Chapter 1", "k g f chapter 1"),
    ],
)
def test_perturbation_resolves(query: str, expected_key: str) -> None:
    matches = best_matches(match_key(query), INDEX)
    assert matches, f"{query!r} found nothing above {MATCH_THRESHOLD}"
    assert matches[0][0] == expected_key, f"{query!r} → {matches}"


def test_unrelated_query_stays_below_threshold() -> None:
    """GS-02 support: a decoy title must not fuzzy-attach to anything in the index."""
    for decoy in ("Inception", "Kaithi", "Pather Panchali"):
        assert best_matches(match_key(decoy), INDEX) == []


def test_exact_hit_scores_one_and_ranks_first() -> None:
    matches = best_matches("drishyam", INDEX)
    assert matches[0] == ("drishyam", 1.0)
    assert all(score <= 1.0 for _, score in matches)


def test_ambiguity_surface() -> None:
    """Drishyam-family keys are mutual near-matches — the resolver must return ALL of them
    above threshold (repository layer turns multi-hit into ambiguous=true, GS-10)."""
    matches = best_matches("drishyam", INDEX, limit=10)
    keys = {k for k, _ in matches}
    assert {"drishyam", "drishya", "drushyam"} <= keys
