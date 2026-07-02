"""Cross-script title normalization + fuzzy resolution (P1 task 8, DEC-P1-5).

``match_key`` — the single indexed romanized key every title row carries:

1. NFC → script detection (Unicode block majority);
2. native Indic scripts → **ITRANS** romanization via deterministic ``indic-transliteration``
   (measured on real slice titles 2026-07-02: ITRANS avg-similarity 87.4 vs IAST/ISO 80.9
   against popular English spellings — DEC-P1-5 amendment in DECISIONS.md);
3. Tamil digraph normalization (sanscript's Tamil scheme picks Sanskrit-positional
   aspirates: ``ப→bha``, ``ச→jha`` — folded back to the popular ``p/ch/k/d``);
4. casefold → strip diacritics → drop punctuation → collapse character runs
   (vowel-length + gemination: ``paapanaasam → papanasam``) → collapse whitespace.

Resolution = exact key hit, then ``best_matches`` (rapidfuzz ratio, 0–1) over the key index
with a threshold tuned on the GS-11 perturbation suite (0.80). No neural op anywhere —
laptop-safe, reproducible (ROADMAP §2 compute placement).
"""

from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from rapidfuzz import fuzz

# Fuzzy-resolution threshold, tuned on the GS-11 perturbation suite (see tests).
MATCH_THRESHOLD = 0.80

# Unicode block → ISO-15924-ish script code (slice languages + han).
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("deva", 0x0900, 0x097F),
    ("beng", 0x0980, 0x09FF),
    ("taml", 0x0B80, 0x0BFF),
    ("telu", 0x0C00, 0x0C7F),
    ("knda", 0x0C80, 0x0CFF),
    ("mlym", 0x0D00, 0x0D7F),
    ("sinh", 0x0D80, 0x0DFF),
    ("hani", 0x4E00, 0x9FFF),
)

_SANSCRIPT_SCHEMES: dict[str, str] = {
    "deva": sanscript.DEVANAGARI,
    "beng": sanscript.BENGALI,
    "taml": sanscript.TAMIL,
    "telu": sanscript.TELUGU,
    "knda": sanscript.KANNADA,
    "mlym": sanscript.MALAYALAM,
}

# sanscript's Tamil scheme romanizes with Sanskrit-positional aspirated/voiced consonants;
# popular Tamil romanization uses the plain series. Applied casefolded, order matters.
_TAMIL_DIGRAPH_FIXES: tuple[tuple[str, str], ...] = (
    ("bh", "p"),
    ("jh", "ch"),
    ("gh", "k"),
    ("dh", "d"),
)

_RUN_RE = re.compile(r"(.)\1+")

# Word-final inherent-schwa deletion for Indo-Aryan scripts (Hindi/Bengali popular
# romanization: दृश्यम → drishyam, एक → ek). Applied BEFORE casefold so ITRANS long "A"
# (ā, e.g. भुलैया → bhulaiyA) is preserved. Dravidian scripts keep final vowels.
_SCHWA_DROP_SCRIPTS = {"deva", "beng"}
_FINAL_SCHWA_RE = re.compile(r"(?<=[^aAeEiIoOuU\s])a\b")


def detect_script(text: str) -> str:
    """Majority Unicode block of the text's letters; 'latn' when no Indic/Han block wins."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "latn"
    return max(counts, key=lambda c: counts[c])


def _fold(text: str) -> str:
    """casefold → strip diacritics → alnum-only → collapse runs → collapse whitespace."""
    text = text.casefold()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    text = _RUN_RE.sub(r"\1", text)
    return " ".join(text.split())


def match_key(title: str) -> str:
    """Normalized romanized key for cross-source/cross-script title matching."""
    text = unicodedata.normalize("NFC", title)
    script = detect_script(text)
    scheme = _SANSCRIPT_SCHEMES.get(script)
    if scheme is not None:
        text = sanscript.transliterate(text, scheme, sanscript.ITRANS)
        if script in _SCHWA_DROP_SCRIPTS:
            text = _FINAL_SCHWA_RE.sub("", text)
        text = text.casefold()
        if script == "taml":
            for src, dst in _TAMIL_DIGRAPH_FIXES:
                text = text.replace(src, dst)
    # sinh/hani (and anything unmapped): no deterministic romanizer — fold as-is;
    # their Latin AKA rows from TMDB/IMDb carry the matchable keys instead.
    return _fold(text)


def best_matches(
    query_key: str,
    candidate_keys: list[str],
    limit: int = 5,
    threshold: float = MATCH_THRESHOLD,
) -> list[tuple[str, float]]:
    """Scored fuzzy matches (rapidfuzz ratio, normalized 0–1) over a match-key index.

    Exact hits score 1.0 and always rank first; results below ``threshold`` are dropped.
    This is the scoring backing ``resolve_title`` (TOOL_SCHEMA v0): the tool's
    ``candidates[].score`` is exactly this value.
    """
    scored: list[tuple[str, float]] = []
    for key in candidate_keys:
        score = fuzz.ratio(query_key, key) / 100.0
        if score >= threshold:
            scored.append((key, round(score, 4)))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:limit]
