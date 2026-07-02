"""Title normalization (P1 task 5 interim; upgraded with transliteration in task 8).

``match_key`` is the single indexed romanized key every title row carries (P1_SPEC §2.4).
This interim version handles Latin-script titles (NFC → casefold → strip diacritics →
drop punctuation → collapse whitespace); task 8 adds deterministic Indic transliteration
(DEC-P1-5) and vowel-length collapsing, then re-keys existing rows (population is idempotent).
"""

from __future__ import annotations

import unicodedata


def match_key(title: str) -> str:
    """Normalized romanized key for cross-source title matching (interim, Latin-focused)."""
    text = unicodedata.normalize("NFC", title)
    text = text.casefold()
    # Strip diacritics: decompose, drop combining marks.
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    # Keep letters/digits/spaces only (punctuation like ":" "-" "." folds away).
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())
