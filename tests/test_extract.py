"""Unit tests for the extraction layer (P1 task 11) — no model call, no DB.

Covers the P1_SPEC §4 row: valid JSON → CandidateEdge proposals; malformed → dropped +
counted (never repaired); the verbatim-evidence guard; the reproducibility hash.
"""

from __future__ import annotations

from sutradhar.pipeline.extract import (
    PROMPT_TEMPLATE,
    ExtractionResponse,
    extraction_run_hash,
    is_supported,
    parse_extraction_output,
)

VALID = """{"relationships": [{"edge_type": "is_remake_of", "src_title": "Papanasam",
"src_language": "ta", "dst_title": "Drishyam", "dst_language": "ml",
"supporting_sentence": "The film was remade in Tamil as Papanasam.", "confidence": 0.95}]}"""


def test_valid_json_parses() -> None:
    response = parse_extraction_output(VALID)
    assert isinstance(response, ExtractionResponse)
    assert len(response.relationships) == 1
    proposal = response.relationships[0]
    assert proposal.edge_type == "is_remake_of"
    assert proposal.src_title == "Papanasam" and proposal.dst_title == "Drishyam"
    assert proposal.confidence == 0.95


def test_code_fenced_json_parses() -> None:
    fenced = f"```json\n{VALID}\n```"
    response = parse_extraction_output(fenced)
    assert response is not None and len(response.relationships) == 1


def test_empty_relationships_is_valid() -> None:
    response = parse_extraction_output('{"relationships": []}')
    assert response is not None and response.relationships == []


def test_malformed_json_dropped() -> None:
    assert parse_extraction_output("The film was remade in Tamil.") is None
    assert parse_extraction_output('{"relationships": [{}]') is None  # truncated


def test_unknown_edge_type_dropped_never_repaired() -> None:
    bad = VALID.replace("is_remake_of", "is_inspired_by")
    assert parse_extraction_output(bad) is None


def test_out_of_range_confidence_dropped() -> None:
    bad = VALID.replace("0.95", "1.7")
    assert parse_extraction_output(bad) is None


def test_short_supporting_sentence_dropped() -> None:
    bad = VALID.replace("The film was remade in Tamil as Papanasam.", "Yes.")
    assert parse_extraction_output(bad) is None


def test_extra_fields_ignored_not_fatal() -> None:
    extra = VALID.replace('"confidence": 0.95}', '"confidence": 0.95, "note": "extra"}')
    assert parse_extraction_output(extra) is not None


# --- verbatim-evidence guard ---

SOURCE = (
    "Drishyam was released on 19 December 2013.\n"
    "The film was remade in Tamil as   Papanasam. It stars Kamal Haasan."
)


def test_supported_sentence_passes_whitespace_normalized() -> None:
    assert is_supported("The film was remade in Tamil as Papanasam.", SOURCE)


def test_hallucinated_sentence_fails() -> None:
    assert not is_supported("The film was remade in Korean in 2021.", SOURCE)


def test_paraphrased_sentence_fails() -> None:
    """Near-miss paraphrase is NOT verbatim — the guard is strict by design."""
    assert not is_supported("The film was remade into Tamil as Papanasam.", SOURCE)


# --- reproducibility stamp ---


def test_run_hash_deterministic_and_sensitive() -> None:
    revisions = {"Drishyam": "123", "Papanasam": "456"}
    h1 = extraction_run_hash("google/gemma-4-E4B", revisions)
    h2 = extraction_run_hash("google/gemma-4-E4B", dict(reversed(revisions.items())))
    assert h1 == h2  # order-insensitive
    assert h1 != extraction_run_hash("other-model", revisions)  # model-sensitive
    assert h1 != extraction_run_hash(
        "google/gemma-4-E4B", {"Drishyam": "124"}
    )  # revision-sensitive
    assert len(h1) == 16


def test_prompt_names_all_four_edge_types() -> None:
    for edge_type in ("is_remake_of", "is_official_dub_of", "is_sequel_of", "based_on"):
        assert edge_type in PROMPT_TEMPLATE
    assert "is_unofficial_remake_of" not in PROMPT_TEMPLATE  # reviewer-assigned, not extracted
    assert "VERBATIM" in PROMPT_TEMPLATE
