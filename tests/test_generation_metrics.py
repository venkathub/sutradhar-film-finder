"""Pure generation-metric scorer tests (P3 task 5; P3_SPEC §4, DEC-P3-5).

No DB, no network, no model — string math over hand-built transcripts. Covers: two-level
tool-call scoring incl. benign-extra and wrong-order cases; placeholder binding (incl. an
id the conversation never saw); intent exact-match per turn; slot micro-F1 with match_key
normalization; the hallucinated-movie detector (catches a seeded invention, does not flag
the "Papanaasam" fuzzy variant, respects the abstain path).
"""

from __future__ import annotations

from typing import Any

from sutradhar.evals.generation import (
    EmittedCall,
    SlotCounts,
    collect_result_ids,
    collect_result_titles,
    detect_hallucinated_movies,
    extract_asserted_titles,
    micro_f1,
    parse_intent_preamble,
    score_intents,
    score_slots,
    score_slots_per_turn,
    score_tool_calls,
)

# --- Shared fixtures: a GS-08a-like conversation ---

_RESOLVE_RESULT = {
    "candidates": [
        {
            "work_id": "wk_drishyam",
            "version_id": "vr_drishyam_ml",
            "matched_title": "Drishyam",
            "language": "ml",
            "year": 2013,
            "score": 1.0,
            "sources": [{"source": "wikidata", "ref": "Q15401703"}],
        }
    ],
    "ambiguous": False,
}

_VERSIONS_RESULT = {
    "original": {"version_id": "vr_drishyam_ml", "title": "Drishyam", "language": "ml"},
    "versions": [
        {"version_id": "vr_drishyam_hi", "title": "Drishyam", "language": "hi"},
        {"version_id": "vr_papanasam_ta", "title": "Papanasam", "language": "ta"},
        {"version_id": "vr_drushyam_te", "title": "Drushyam", "language": "te"},
    ],
}


def _call(
    tool: str,
    arguments: dict[str, Any] | None,
    *,
    valid: bool = True,
    result: dict[str, Any] | None = None,
) -> EmittedCall:
    return EmittedCall(tool=tool, arguments=arguments, schema_valid=valid, result=result)


_EXPECTED = [
    ("resolve_title", {"title": "Drishyam"}),
    ("get_versions", {"work_id": "$work_id", "scope": "indian"}),
    ("refine_filter", {"version_set": ["$version_set"], "by": {"actor": "Ajay Devgn"}}),
]


def _good_emitted() -> list[EmittedCall]:
    return [
        _call("resolve_title", {"title": "drishyam"}, result=_RESOLVE_RESULT),
        _call(
            "get_versions", {"work_id": "wk_drishyam", "scope": "indian"}, result=_VERSIONS_RESULT
        ),
        _call(
            "refine_filter",
            {
                "version_set": ["vr_drishyam_hi", "vr_papanasam_ta"],
                "by": {"actor": "ajay devgn"},
            },
            result={"versions": [{"version_id": "vr_drishyam_hi", "title": "Drishyam"}]},
        ),
    ]


# --- Tool-call accuracy (DEC-P3-5) ---


def test_perfect_sequence_scores_full() -> None:
    score = score_tool_calls(_EXPECTED, _good_emitted())
    assert score.call_matches == (True, True, True)
    assert score.call_level == 1.0
    assert score.sequence_match is True
    assert score.schema_validity == 1.0


def test_benign_extra_call_tolerated() -> None:
    """An extra schema-valid get_work between expected calls must not zero the fixture."""
    emitted = _good_emitted()
    emitted.insert(
        2,
        _call("get_work", {"work_id": "wk_drishyam"}, result={"work_id": "wk_drishyam"}),
    )
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.sequence_match is True
    assert score.call_level == 1.0


def test_wrong_order_breaks_sequence_but_partial_credit_remains() -> None:
    """Order IS the behaviour (GS-08): refine before get_versions breaks the sequence."""
    good = _good_emitted()
    emitted = [good[0], good[2], good[1]]  # resolve, refine, get_versions
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.sequence_match is False
    # resolve matches; refine can't (version ids not yet seen -> binding fails);
    # get_versions still matches after it (partial credit, DEC-P3-5).
    assert score.call_matches == (True, True, False)
    assert 0 < score.call_level < 1.0


def test_placeholder_binding_rejects_unseen_id() -> None:
    """get_versions with a work_id the conversation never returned = scored mismatch."""
    emitted = _good_emitted()
    emitted[1] = _call(
        "get_versions", {"work_id": "wk_hallucinated", "scope": "indian"}, result=_VERSIONS_RESULT
    )
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.call_matches[1] is False
    assert score.sequence_match is False


def test_version_set_binding_requires_all_ids_seen() -> None:
    emitted = _good_emitted()
    emitted[2] = _call(
        "refine_filter",
        {"version_set": ["vr_drishyam_hi", "vr_invented"], "by": {"actor": "Ajay Devgn"}},
        result={"versions": []},
    )
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.call_matches[2] is False


def test_schema_invalid_call_never_matches_and_is_counted() -> None:
    emitted = _good_emitted()
    emitted[0] = _call("resolve_title", {"title": "Drishyam"}, valid=False, result=None)
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.call_matches[0] is False
    assert score.invalid_emitted == 1
    assert score.schema_validity == 1.0 - 1 / 3


def test_missing_expected_argument_fails_call() -> None:
    emitted = _good_emitted()
    emitted[1] = _call("get_versions", {"work_id": "wk_drishyam"}, result=_VERSIONS_RESULT)
    score = score_tool_calls(_EXPECTED, emitted)
    assert score.call_matches[1] is False  # scope: indian expected, absent


def test_free_text_description_matches_by_token_overlap() -> None:
    expected = [("search_by_plot", {"description": "father hides evidence to save his family"})]
    emitted = [
        _call(
            "search_by_plot",
            {"description": "a father hides the evidence to protect his family", "top_k": 10},
            result={"results": [], "abstain": True},
        )
    ]
    assert score_tool_calls(expected, emitted).sequence_match is True
    off_topic = [
        _call(
            "search_by_plot",
            {"description": "astronauts explore a wormhole in space"},
            result={"results": [], "abstain": True},
        )
    ]
    assert score_tool_calls(expected, off_topic).sequence_match is False


def test_title_argument_matches_across_scripts() -> None:
    """match_key normalization: native-script emitted title == romanized expected."""
    expected = [("resolve_title", {"title": "ಅಪ್ತಮಿತ್ರ"})]
    emitted = [_call("resolve_title", {"title": "Apthamitra"}, result=_RESOLVE_RESULT)]
    assert score_tool_calls(expected, emitted).sequence_match is True


# --- Intent preamble parsing + intent accuracy ---


def test_parse_intent_preamble_roundtrip() -> None:
    text = 'INTENT: {"intent": "refine", "slots": {"language": "te"}}\n\nTelugu version: ...'
    parsed = parse_intent_preamble(text)
    assert parsed is not None
    assert parsed.intent == "refine"
    assert parsed.slots == {"language": "te"}


def test_parse_intent_preamble_malformed_never_crashes() -> None:
    assert parse_intent_preamble(None) is None
    assert parse_intent_preamble("") is None
    assert parse_intent_preamble("Papanasam is the remake.") is None  # missing preamble
    assert parse_intent_preamble('INTENT: {"intent": "refine"') is None  # truncated JSON
    assert parse_intent_preamble('INTENT: ["refine"]') is None  # not an object
    assert parse_intent_preamble('INTENT: {"slots": {}}') is None  # intent missing
    assert parse_intent_preamble('INTENT: {"intent": "x", "slots": ["y"]}') is None


def test_score_intents_per_turn() -> None:
    answers = [
        'INTENT: {"intent": "find_by_title", "slots": {"title": "Drishyam"}}\n\nFound it.',
        'INTENT: {"intent": "refine", "slots": {"era": "original"}}\n\nThe original.',
        "no preamble here",  # malformed third turn
    ]
    assert score_intents(["find_by_title", "refine", "refine"], answers) == [True, True, False]
    # Missing answer (model never answered the turn) = wrong, not a crash.
    assert score_intents(["find_by_title", "refine"], answers[:1]) == [True, False]


# --- Slot micro-F1 ---


def test_slot_micro_f1_exact_and_fuzzy_title() -> None:
    counts = score_slots(
        {"title": "Papanasam", "actor": "Kamal Haasan"},
        {"title": "Papanaasam", "actor": "kamal haasan"},  # fuzzy title + case difference
    )
    assert counts == SlotCounts(tp=2, fp=0, fn=0)
    assert micro_f1(counts) == 1.0


def test_slot_micro_f1_counts_misses_and_extras() -> None:
    counts = score_slots(
        {"title": "Drishyam", "language": "te"},
        {"title": "Drishyam", "year": 2013},  # language missed, year invented
    )
    assert counts == SlotCounts(tp=1, fp=1, fn=1)
    assert micro_f1(counts) == 2 / 4


def test_slot_score_per_turn_accumulates_and_handles_missing_preamble() -> None:
    expected = [{"title": "Drishyam 2", "language": "te"}, {"language": "ta"}]
    answers = [
        'INTENT: {"intent": "find_by_title", "slots": '
        '{"title": "Drushyam 2", "language": "te"}}\n\nx',
        "no preamble",  # all turn-2 expected pairs become FN
    ]
    counts = score_slots_per_turn(expected, answers)
    assert counts == SlotCounts(tp=2, fp=0, fn=1)


def test_native_script_slot_value_matches_romanized() -> None:
    counts = score_slots({"title": "ಅಪ್ತಮಿತ್ರ"}, {"title": "Apthamitra"})
    assert counts.tp == 1 and counts.fn == 0


# --- Hallucinated-movie detector (the GS-02 gate) ---

_ALLOWED = {"Drishyam", "Papanasam", "Drushyam", "ദൃശ്യം"}


def test_detector_passes_grounded_answer() -> None:
    answer = (
        'INTENT: {"intent": "list_versions", "slots": {"title": "Papanasam"}}\n\n'
        "**Papanasam (2015, Tamil)** is a remake of **Drishyam (2013, Malayalam)**, "
        "the original (Wikidata Q15401703)."
    )
    report = detect_hallucinated_movies(answer, _ALLOWED)
    assert report.invention_count == 0
    assert set(report.asserted) == {"Papanasam", "Drishyam"}


def test_detector_catches_seeded_invention() -> None:
    answer = "**Papanasam (2015)** and its Bengali remake **Chokher Aloy (2016)** ..."
    report = detect_hallucinated_movies(answer, _ALLOWED)
    assert report.inventions == ("Chokher Aloy",)


def test_detector_does_not_flag_fuzzy_variant() -> None:
    """'Papanaasam' (double a) fuzzy-resolves to the returned 'Papanasam' — no invention."""
    report = detect_hallucinated_movies("**Papanaasam (2015, Tamil)** — Kamal Haasan.", _ALLOWED)
    assert report.invention_count == 0


def test_detector_matches_native_script_against_romanized() -> None:
    report = detect_hallucinated_movies("**దృశ్యం (2014)** is the Telugu version.", {"Drushyam"})
    assert report.invention_count == 0


def test_detector_respects_abstain_path() -> None:
    """An abstaining answer asserts no titles — zero inventions with empty tool results."""
    answer = (
        'INTENT: {"intent": "out_of_catalog", "slots": {"title": "Salaar"}}\n\n'
        "I checked the catalog — that film is not in it. NO_MATCH."
    )
    report = detect_hallucinated_movies(answer, set())
    assert report.asserted == ()
    assert report.invention_count == 0


def test_detector_catches_unbolded_title_year_invention() -> None:
    """Fallback pattern: an invented 'Title (year)' without bold is still caught."""
    report = detect_hallucinated_movies(
        "You should watch Chokher Aloy (2016), the Bengali remake.", _ALLOWED
    )
    assert "Chokher Aloy" in report.inventions


def test_detector_ignores_bare_language_year_phrases() -> None:
    """'the Telugu (2014) version' is not a title claim (fallback-pattern guard)."""
    report = detect_hallucinated_movies(
        "**Drushyam (2014)** — the Telugu (2014) version of the story.", _ALLOWED
    )
    assert report.invention_count == 0


def test_extract_asserted_titles_dedupes_by_match_key() -> None:
    titles = extract_asserted_titles("**Papanasam (2015)** ... **Papanaasam** again")
    assert titles == ["Papanasam"]


# --- Result-walking helpers ---


def test_collect_result_ids_and_titles() -> None:
    calls = _good_emitted()
    ids = collect_result_ids(calls)
    assert {"wk_drishyam", "vr_drishyam_ml", "vr_papanasam_ta"} <= ids
    titles = collect_result_titles(calls)
    assert {"Drishyam", "Papanasam", "Drushyam"} <= titles
