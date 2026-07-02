"""Hermetic unit tests for the recursive chunker + header/card rendering (P2 task 3).

No DB, no model: determinism, boundary respect, size/overlap bounds, metadata-header
correctness (incl. remake lineage), native-script passthrough, edge cases (DEC-P2-3).
"""

from __future__ import annotations

import uuid

import pytest

from sutradhar.rag.chunking import (
    CHUNK_CONFIGS,
    CHUNKER_NAME,
    ChunkConfig,
    chunk_text,
    content_hash,
    estimate_tokens,
    split_paragraphs,
    split_sentences,
)
from sutradhar.rag.corpus import VersionMeta, build_header, metadata_card_text

CFG = ChunkConfig("64tok_15pct", 64)  # small target → exercises packing without huge fixtures


def _para(sentences: int, stem: str) -> str:
    return " ".join(f"The {stem} scene number {i} unfolds slowly." for i in range(sentences))


def _meta(**overrides: object) -> VersionMeta:
    fields: dict[str, object] = {
        "version_id": uuid.uuid4(),
        "work_id": uuid.uuid4(),
        "title": "Papanasam",
        "language": "ta",
        "year": 2015,
        "is_original": False,
        "relationship": "is_remake_of",
        "original_title": "Drishyam",
        "original_language": "ml",
        "original_year": 2013,
        "aka_titles": ("பாபநாசம்",),
        "leads": ("Kamal Haasan", "Gautami"),
        "directors": ("Jeethu Joseph",),
    }
    fields.update(overrides)
    return VersionMeta(**fields)  # type: ignore[arg-type]


# --- Config inventory (DEC-P2-3) ---


def test_ablation_grid_pinned() -> None:
    assert [c.name for c in CHUNK_CONFIGS] == ["256tok_15pct", "512tok_15pct", "1024tok_15pct"]
    assert [c.target_tokens for c in CHUNK_CONFIGS] == [256, 512, 1024]
    assert all(c.overlap_fraction == 0.15 for c in CHUNK_CONFIGS)
    assert CHUNKER_NAME == "recursive_para"


# --- Token estimator ---


def test_estimate_tokens_latin_vs_native() -> None:
    latin = "A quiet village family drama."  # 29 chars → ~7 tokens
    assert estimate_tokens(latin) == pytest.approx(len(latin) / 4, abs=1)
    native = "കുടുംബനാഥന്റെ കഥ"  # Malayalam: 2 chars/token, costlier than latin
    assert estimate_tokens(native) == pytest.approx(len(native) / 2, abs=1)
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("a") == 1  # floor: non-empty text is ≥1 token


# --- Chunker: determinism / boundaries / bounds ---


def test_chunking_is_deterministic() -> None:
    text = "\n\n".join(_para(6, f"act{i}") for i in range(5))
    first = chunk_text(text, CFG)
    second = chunk_text(text, CFG)
    assert first == second
    assert [content_hash(c) for c in first] == [content_hash(c) for c in second]


def test_short_text_single_chunk_verbatim() -> None:
    text = "A man buries a secret. His family keeps it."
    assert chunk_text(text, CFG) == [text]


def test_empty_and_whitespace_yield_nothing() -> None:
    assert chunk_text("", CFG) == []
    assert chunk_text("  \n\n  ", CFG) == []


def test_paragraph_boundaries_respected() -> None:
    """Paragraphs that fit the target are never split across chunks."""
    paragraphs = [_para(3, f"thread{i}") for i in range(8)]
    chunks = chunk_text("\n\n".join(paragraphs), CFG)
    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in chunk for chunk in chunks), paragraph[:40]


def test_size_bounds_hold() -> None:
    text = "\n\n".join(_para(5, f"plot{i}") for i in range(10))
    for chunk in chunk_text(text, CFG):
        assert (
            0
            < estimate_tokens(chunk)
            <= CFG.target_tokens + int(CFG.target_tokens * CFG.overlap_fraction)
        )


def test_overlap_carries_trailing_units() -> None:
    """Consecutive chunks share the previous tail (≤15% budget) — and only the tail."""
    text = "\n\n".join(_para(3, f"chapter{i}") for i in range(12))
    chunks = chunk_text(text, CFG)
    assert len(chunks) >= 3
    budget = int(CFG.target_tokens * CFG.overlap_fraction)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_units = prev.split("\n\n")
        nxt_units = nxt.split("\n\n")
        shared = [u for u in nxt_units if u in prev_units]
        if shared:  # overlap units are exactly a suffix of prev / prefix of next
            assert prev_units[-len(shared) :] == shared == nxt_units[: len(shared)]
            assert sum(estimate_tokens(u) for u in shared) <= budget


def test_oversized_paragraph_splits_at_sentences() -> None:
    monster = _para(40, "unbroken")  # one paragraph, way over target
    chunks = chunk_text(monster, CFG)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk) <= CFG.target_tokens + int(
            CFG.target_tokens * CFG.overlap_fraction
        )


def test_oversized_sentence_falls_back_to_word_windows() -> None:
    run_on = "word " * 600  # a single 600-word "sentence", no terminator
    chunks = chunk_text(run_on.strip(), CFG)
    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= CFG.target_tokens * 1.2 for c in chunks)


def test_native_script_passthrough() -> None:
    """Malayalam/danda-punctuated text chunks losslessly (order + content preserved)."""
    sentences = [f"ഗ്രാമീണ കുടുംബത്തിന്റെ കഥ ഭാഗം {i} തുടരുന്നു।" for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, ChunkConfig("32tok_15pct", 32))
    assert len(chunks) > 1
    joined = " ".join(chunks)
    for sentence in sentences:
        assert sentence in joined
    assert split_sentences(text) == sentences  # danda (।) is a sentence boundary


def test_split_paragraphs_normalizes_blank_lines() -> None:
    assert split_paragraphs("one\n\ntwo\n   \nthree") == ["one", "two", "three"]


# --- Header + metadata card (P2_SPEC §2.2) ---


def test_header_carries_remake_lineage() -> None:
    header = build_header(_meta())
    assert header == "Papanasam (Tamil, 2015) — remake of Drishyam (Malayalam, 2013). "


def test_header_official_dub_wording() -> None:
    header = build_header(
        _meta(
            title="Baahubali",
            language="hi",
            year=2015,
            relationship="is_official_dub_of",
            original_title="Baahubali: The Beginning",
            original_language="te",
            original_year=2015,
        )
    )
    assert "— official dub of Baahubali: The Beginning (Telugu, 2015)" in header
    assert "remake" not in header  # dub ≠ remake, even in a header


def test_header_original_has_no_lineage_suffix() -> None:
    header = build_header(
        _meta(
            title="Drishyam",
            language="ml",
            year=2013,
            is_original=True,
            relationship=None,
            original_title=None,
            original_language=None,
            original_year=None,
        )
    )
    assert header == "Drishyam (Malayalam, 2013). "


def test_header_missing_year_degrades_gracefully() -> None:
    assert build_header(_meta(year=None, relationship=None, original_title=None)).startswith(
        "Papanasam (Tamil)."
    )


def test_metadata_card_contents() -> None:
    card = metadata_card_text(_meta())
    assert card.startswith("Papanasam (Tamil, 2015) — remake of Drishyam (Malayalam, 2013).")
    assert "Also known as: பாபநாசம்." in card  # native-script cross-lingual anchor
    assert "Directed by Jeethu Joseph." in card
    assert "Starring Kamal Haasan, Gautami." in card
    assert "original version" not in card  # not the original


def test_metadata_card_flags_original() -> None:
    card = metadata_card_text(
        _meta(
            title="Drishyam",
            language="ml",
            year=2013,
            is_original=True,
            relationship=None,
            original_title=None,
            aka_titles=(),
            leads=("Mohanlal",),
        )
    )
    assert "This is the original version of this story." in card
    assert "Starring Mohanlal." in card
