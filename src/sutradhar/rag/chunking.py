"""Deterministic recursive paragraph-boundary chunker (P2_SPEC §2.2, DEC-P2-3).

Runs on the laptop with zero neural dependencies (ROADMAP §2): chunk *sizes* are counted
with a deterministic token **estimator**, not the XLM-R tokenizer — pulling HF tokenizers
into the laptop env is exactly what §2.7 excludes. The estimate is character-class based:

- Latin-script text ≈ 4 chars/token (the common English SentencePiece ratio);
- all other scripts (Indic, CJK, …) ≈ 2 chars/token — deliberately conservative for
  native-script plots, so estimated sizes err toward SMALLER chunks and every embedded
  unit stays far inside BGE-M3's 8192-token window.

Sizes are ablation *brackets* ({256, 512, 1024}, DEC-P2-3), not exact budgets, so an
estimator is sufficient; determinism (same text → same chunks → same ``content_hash``)
is the property the artifact store and CI actually depend on.

Recursive strategy: paragraphs are the atomic unit; an oversized paragraph is split at
sentence boundaries (incl. the Devanagari danda); an oversized sentence falls back to
word windows. Units are packed greedily up to the target, and each new chunk is seeded
with the previous chunk's trailing units up to a 15% overlap budget.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

CHUNKER_NAME = "recursive_para"

_LATIN_RE = re.compile(r"[\u0000-\u024F]")  # ASCII + Latin-1 + Latin Extended A/B
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

_LATIN_CHARS_PER_TOKEN = 4.0
_OTHER_CHARS_PER_TOKEN = 2.0


@dataclass(frozen=True)
class ChunkConfig:
    """One chunking-ablation cell (DEC-P2-3); ``name`` is the ``chunk_config`` DB key."""

    name: str
    target_tokens: int
    overlap_fraction: float = 0.15


CHUNK_CONFIGS: tuple[ChunkConfig, ...] = (
    ChunkConfig("256tok_15pct", 256),
    ChunkConfig("512tok_15pct", 512),  # DEC-P2-3 default
    ChunkConfig("1024tok_15pct", 1024),
)


def estimate_tokens(text: str) -> int:
    """Deterministic XLM-R token-count estimate (see module docstring)."""
    latin = len(_LATIN_RE.findall(text))
    other = len(text) - latin
    if not text.strip():
        return 0
    return max(1, round(latin / _LATIN_CHARS_PER_TOKEN + other / _OTHER_CHARS_PER_TOKEN))


def content_hash(text: str) -> str:
    """sha256 over the exact embedded text — the artifact-store lookup key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _word_windows(sentence: str, target_tokens: int) -> list[str]:
    """Last-resort split for a single oversized sentence: greedy word windows."""
    words = sentence.split()
    windows: list[str] = []
    current: list[str] = []
    for word in words:
        if current and estimate_tokens(" ".join([*current, word])) > target_tokens:
            windows.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        windows.append(" ".join(current))
    return windows


def _units(text: str, target_tokens: int) -> list[str]:
    """Recursive descent: paragraph → sentences → word windows, until units fit."""
    units: list[str] = []
    for paragraph in split_paragraphs(text):
        if estimate_tokens(paragraph) <= target_tokens:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if estimate_tokens(sentence) <= target_tokens:
                units.append(sentence)
            else:
                units.extend(_word_windows(sentence, target_tokens))
    return units


def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    """Deterministic chunks: greedy unit packing + trailing-unit overlap seeding.

    Every returned chunk is non-empty; a text that fits the target yields exactly one
    chunk; empty/whitespace input yields no chunks. Paragraph boundaries are respected
    (units never split below sentence level unless a single sentence exceeds target).
    """
    units = _units(text, config.target_tokens)
    if not units:
        return []

    overlap_budget = int(config.target_tokens * config.overlap_fraction)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(current)
        # Seed the next chunk with trailing units of this one, newest-first within budget.
        seed: list[str] = []
        seed_tokens = 0
        for unit in reversed(current[1:]):  # never re-carry the whole chunk → progress
            unit_tokens = estimate_tokens(unit)
            if seed_tokens + unit_tokens > overlap_budget:
                break
            seed.insert(0, unit)
            seed_tokens += unit_tokens
        current = seed
        current_tokens = seed_tokens

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > config.target_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
    if [u for u in current if u]:
        # Drop a pure-overlap tail (all units already emitted) — no new content.
        if not chunks or current != chunks[-1][-len(current) :]:
            chunks.append(current)

    return ["\n\n".join(c) for c in chunks]
