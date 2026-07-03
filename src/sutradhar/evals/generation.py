"""Pure generation-metric scorers (P3 task 5; P3_SPEC §2.4, DEC-P3-5).

Laptop/CI-safe string math only — no DB, no network, no model. The task-6 driver produces
transcripts; these functions score them; the task-9 artifact records both. Tier-1 CI re-runs
the SAME functions over the committed run artifact, so recorded metrics can never drift from
what the code computes (DEC-P2-6 posture).

Scorers:

- **Tool-call accuracy (DEC-P3-5, BFCL-style two-level):** call-level AST match (tool name +
  placeholder-bound, normalized arguments; per-expected-call fraction) and sequence-level
  (the expected sequence appears in order as a subsequence; benign extra schema-valid calls
  tolerated — the Table 2 headline). Schema validity is the driver's verdict, reported as an
  independent third number, never blended in.
- **Placeholder binding:** golden ``$work_id`` / ``[$version_set]`` bind to ids actually
  returned by earlier successful tool calls in the same conversation; an id the conversation
  never saw is a scored mismatch, not a crash (P3_SPEC §2.3).
- **Intent accuracy:** exact-match per turn of the ``INTENT:`` preamble label the frozen
  prompt requires on each final prose answer (DEC-P3-4 amendment).
- **Slot accuracy:** micro-F1 over expected (key, value) pairs; title-ish values normalized
  via ``match_key`` (DEC-P1-5), everything else casefolded.
- **Hallucinated-movie detector (the GS-02 gate):** every title asserted in a final answer
  must fuzzy-resolve (match_key + rapidfuzz ≥ MATCH_THRESHOLD = 0.80) to a title present in
  that conversation's tool results; anything else is an invention. Extraction is
  contract-driven: the frozen prompt requires titles in **bold** (and nothing else bold);
  a secondary ``Title (year)`` pattern catches unbolded year-carrying assertions. A prose
  invention that carries neither marker is out of deterministic reach — the RAGAS
  faithfulness judge is the documented supplementary net (P3_SPEC §2.4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from sutradhar.evals.prompts import INTENT_PREAMBLE_PREFIX
from sutradhar.pipeline.normalize import MATCH_THRESHOLD, match_key

# Slot keys whose values are film titles (cross-script/misspelling-tolerant comparison).
_TITLE_SLOT_KEYS = frozenset({"title"})
# Free-text argument names compared by fuzzy token overlap, not equality (a paraphrased
# plot description is correct behaviour; wording is not the behaviour under test).
_FREE_TEXT_ARGS = frozenset({"description"})
_FREE_TEXT_THRESHOLD = 0.6

_WORK_ID_PLACEHOLDER = "$work_id"
_VERSION_SET_PLACEHOLDER = "$version_set"


# --- Transcript-side containers (produced by the driver, consumed here) ---


@dataclass(frozen=True)
class EmittedCall:
    """One model-emitted tool call after driver-side validation (P3_SPEC §2.3)."""

    tool: str
    arguments: dict[str, Any] | None  # None = malformed argument JSON
    schema_valid: bool
    result: dict[str, Any] | None = None  # tool result fed back (None if not executed)


def collect_result_ids(calls: list[EmittedCall]) -> set[str]:
    """All work_id / version_id values any successful tool result returned."""
    ids: set[str] = set()
    for call in calls:
        if call.result is not None:
            _walk_ids(call.result, ids)
    return ids


def _walk_ids(node: Any, ids: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("work_id", "version_id") and isinstance(value, str):
                ids.add(value)
            else:
                _walk_ids(value, ids)
    elif isinstance(node, list):
        for item in node:
            _walk_ids(item, ids)


def collect_result_titles(calls: list[EmittedCall]) -> set[str]:
    """All title strings any successful tool result returned (the grounding set)."""
    titles: set[str] = set()
    for call in calls:
        if call.result is not None:
            _walk_titles(call.result, titles)
    return titles


def _walk_titles(node: Any, titles: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("title", "matched_title", "canonical_title") and isinstance(value, str):
                titles.add(value)
            else:
                _walk_titles(value, titles)
    elif isinstance(node, list):
        for item in node:
            _walk_titles(item, titles)


# --- Tool-call accuracy (DEC-P3-5) ---


@dataclass(frozen=True)
class ToolCallScore:
    """Two-level DEC-P3-5 score for one fixture + the independent validity count."""

    expected_total: int
    call_matches: tuple[bool, ...]  # per expected call, greedy in-order matching
    sequence_match: bool  # ALL expected calls matched, in order (Table 2 headline)
    invalid_emitted: int  # emitted calls failing schema validation (driver verdict)
    emitted_total: int

    @property
    def call_level(self) -> float:
        if not self.call_matches:
            return 0.0
        return sum(self.call_matches) / len(self.call_matches)

    @property
    def schema_validity(self) -> float:
        if self.emitted_total == 0:
            return 1.0
        return 1.0 - self.invalid_emitted / self.emitted_total


def _values_match(key: str, expected: Any, emitted: Any) -> bool:
    """Normalized single-argument comparison (see module docstring for the rules)."""
    if isinstance(expected, dict) and isinstance(emitted, dict):
        # e.g. refine_filter.by — compare per-key with the same rules.
        return all(k in emitted and _values_match(k, v, emitted[k]) for k, v in expected.items())
    if key in _FREE_TEXT_ARGS and isinstance(expected, str) and isinstance(emitted, str):
        return fuzz.token_set_ratio(expected.casefold(), emitted.casefold()) / 100.0 >= (
            _FREE_TEXT_THRESHOLD
        )
    if key in _TITLE_SLOT_KEYS and isinstance(expected, str) and isinstance(emitted, str):
        return _titles_equivalent(expected, emitted)
    if isinstance(expected, str) and isinstance(emitted, str):
        return expected.casefold() == emitted.casefold()
    return bool(expected == emitted)


def _titles_equivalent(a: str, b: str) -> bool:
    ka, kb = match_key(a), match_key(b)
    return ka == kb or fuzz.ratio(ka, kb) / 100.0 >= MATCH_THRESHOLD


def _call_matches_expected(
    expected_tool: str,
    expected_args: dict[str, Any],
    emitted: EmittedCall,
    seen_ids: set[str],
) -> bool:
    """Call-level AST match: name + every expected argument (placeholder-bound, normalized).

    Extra emitted arguments are tolerated iff the call is schema-valid (benign optional
    params, DEC-P3-5); a schema-invalid call never matches.
    """
    if emitted.tool != expected_tool or not emitted.schema_valid or emitted.arguments is None:
        return False
    for key, expected_value in expected_args.items():
        if key not in emitted.arguments:
            return False
        emitted_value = emitted.arguments[key]
        if expected_value == _WORK_ID_PLACEHOLDER:
            # Binds to any id an earlier successful call in THIS conversation returned.
            if not (isinstance(emitted_value, str) and emitted_value in seen_ids):
                return False
        elif expected_value == [_VERSION_SET_PLACEHOLDER]:
            if not (
                isinstance(emitted_value, list)
                and emitted_value
                and all(isinstance(v, str) and v in seen_ids for v in emitted_value)
            ):
                return False
        elif not _values_match(key, expected_value, emitted_value):
            return False
    return True


def score_tool_calls(
    expected: list[tuple[str, dict[str, Any]]],
    emitted: list[EmittedCall],
) -> ToolCallScore:
    """DEC-P3-5 two-level scoring: greedy in-order subsequence match of the expected
    sequence against the emitted calls; benign schema-valid extras are skipped over."""
    seen_ids: set[str] = set()
    matches: list[bool] = []
    cursor = 0
    for expected_tool, expected_args in expected:
        matched = False
        while cursor < len(emitted):
            candidate = emitted[cursor]
            # Binding set grows as the conversation progresses: ids become bindable
            # once returned by any call BEFORE the one being matched.
            if _call_matches_expected(expected_tool, expected_args, candidate, seen_ids):
                matched = True
            if candidate.result is not None:
                _walk_ids(candidate.result, seen_ids)
            cursor += 1
            if matched:
                break
        matches.append(matched)
    return ToolCallScore(
        expected_total=len(expected),
        call_matches=tuple(matches),
        sequence_match=all(matches) if matches else True,
        invalid_emitted=sum(1 for c in emitted if not c.schema_valid),
        emitted_total=len(emitted),
    )


# --- Intent preamble parsing + intent accuracy ---


@dataclass(frozen=True)
class ParsedPreamble:
    intent: str
    slots: dict[str, Any]


def parse_intent_preamble(text: str | None) -> ParsedPreamble | None:
    """Parse the frozen ``INTENT: {...}`` first line of a final answer (None = missing
    or malformed — a scored failure for intent/slot accuracy, never an exception)."""
    if not text:
        return None
    first_line = text.strip().splitlines()[0].strip()
    if not first_line.startswith(INTENT_PREAMBLE_PREFIX):
        return None
    try:
        payload = json.loads(first_line[len(INTENT_PREAMBLE_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("intent"), str):
        return None
    slots = payload.get("slots", {})
    if not isinstance(slots, dict):
        return None
    return ParsedPreamble(intent=payload["intent"], slots=slots)


def score_intents(
    expected: str | list[str],
    answers: list[str | None],
) -> list[bool]:
    """Per-turn exact match of the predicted intent label (missing/malformed = wrong)."""
    expected_list = [expected] if isinstance(expected, str) else expected
    results: list[bool] = []
    for i, expected_intent in enumerate(expected_list):
        answer = answers[i] if i < len(answers) else None
        parsed = parse_intent_preamble(answer)
        results.append(parsed is not None and parsed.intent == expected_intent)
    return results


# --- Slot micro-F1 ---


@dataclass(frozen=True)
class SlotCounts:
    """Micro-F1 accumulator (sum across turns/fixtures, then :func:`micro_f1`)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: SlotCounts) -> SlotCounts:
        return SlotCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def _slot_values_match(key: str, expected: Any, predicted: Any) -> bool:
    if key in _TITLE_SLOT_KEYS and isinstance(expected, str) and isinstance(predicted, str):
        return _titles_equivalent(expected, predicted)
    if isinstance(expected, str) and isinstance(predicted, str):
        return expected.casefold() == predicted.casefold()
    return bool(expected == predicted)


def score_slots(expected: dict[str, Any], predicted: dict[str, Any]) -> SlotCounts:
    """One turn's slot (key, value) pairs vs the prediction (micro counts)."""
    tp = fp = fn = 0
    for key, expected_value in expected.items():
        if key in predicted and _slot_values_match(key, expected_value, predicted[key]):
            tp += 1
        else:
            fn += 1
    for key in predicted:
        if key not in expected or not _slot_values_match(key, expected[key], predicted[key]):
            fp += 1
    return SlotCounts(tp=tp, fp=fp, fn=fn)


def score_slots_per_turn(
    expected: dict[str, Any] | list[dict[str, Any]],
    answers: list[str | None],
) -> SlotCounts:
    """All turns of one fixture: parse each answer's preamble, accumulate micro counts.
    A missing/malformed preamble predicts NO slots (all expected pairs become FN)."""
    expected_list = [expected] if isinstance(expected, dict) else expected
    total = SlotCounts()
    for i, expected_slots in enumerate(expected_list):
        answer = answers[i] if i < len(answers) else None
        parsed = parse_intent_preamble(answer)
        predicted = parsed.slots if parsed is not None else {}
        total = total + score_slots(expected_slots, predicted)
    return total


def micro_f1(counts: SlotCounts) -> float:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    if denominator == 0:
        return 1.0  # nothing expected, nothing predicted
    return 2 * counts.tp / denominator


# --- Deterministic no-hallucinated-movie detector (the GS-02 gate) ---

_BOLD_SPAN_RE = re.compile(r"\*\*(.+?)\*\*")
# Unbolded "Some Title (2015" — a capitalized phrase directly before a year parenthetical.
_TITLE_YEAR_RE = re.compile(r"(?<!\*)\b([A-Z][\w'.:-]*(?:\s+[\w'.:-]+){0,6})\s*\((\d{4})")
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
# Secondary-pattern guard: bare language/meta words before "(2014)" are not title claims
# ("the Telugu (2014) version"). Applies ONLY to the unbolded fallback pattern — bold spans
# are titles by contract and are never filtered.
_NON_TITLE_WORDS = frozenset(
    {
        "telugu",
        "tamil",
        "hindi",
        "malayalam",
        "kannada",
        "bengali",
        "sinhala",
        "chinese",
        "english",
        "indian",
        "original",
        "remake",
        "dub",
        "version",
        "wikidata",
        "tmdb",
        "imdb",
    }
)


@dataclass(frozen=True)
class HallucinationReport:
    """Asserted titles vs the conversation's tool-result grounding set."""

    asserted: tuple[str, ...]
    inventions: tuple[str, ...]
    allowed: frozenset[str] = field(default=frozenset())

    @property
    def invention_count(self) -> int:
        return len(self.inventions)


_CONNECTOR_WORDS = frozenset({"a", "an", "the", "of", "and", "ka", "ki", "ke", "e", "o"})


def _trim_title_phrase(phrase: str) -> str:
    """Keep the maximal title-like token run ending at the year parenthetical: walking
    right-to-left, keep capitalized/digit tokens and short connectors ("Sheep Without a
    Shepherd"), stop at the first plain lowercase word ("You should watch Chokher Aloy"
    -> "Chokher Aloy"); leading connectors are stripped."""
    kept: list[str] = []
    for token in reversed(phrase.split()):
        if token[:1].isupper() or token[:1].isdigit() or token.casefold() in _CONNECTOR_WORDS:
            kept.append(token)
        else:
            break
    while kept and kept[-1].casefold() in _CONNECTOR_WORDS:
        kept.pop()
    return " ".join(reversed(kept))


def extract_asserted_titles(answer: str) -> list[str]:
    """Title assertions in a final answer, per the frozen formatting contract:
    every **bold** span (titles are the only bold content) + any unbolded
    capitalized phrase immediately preceding a ``(year`` parenthetical."""
    titles: list[str] = []
    for span in _BOLD_SPAN_RE.findall(answer):
        cleaned = _TRAILING_PAREN_RE.sub("", span).strip().strip("\"'")
        if cleaned:
            titles.append(cleaned)
    without_bold = _BOLD_SPAN_RE.sub(" ", answer)
    for phrase, _year in _TITLE_YEAR_RE.findall(without_bold):
        cleaned = _trim_title_phrase(phrase.strip())
        # Guard the fallback pattern: skip phrases made only of language/meta words.
        words = {w.casefold() for w in cleaned.split()}
        if cleaned and not words <= _NON_TITLE_WORDS:
            titles.append(cleaned)
    # De-dup preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in titles:
        key = match_key(t)
        if key and key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def detect_hallucinated_movies(
    answer: str,
    allowed_titles: set[str],
) -> HallucinationReport:
    """Every asserted title must fuzzy-resolve (match_key + rapidfuzz ≥ 0.80, DEC-P1-5)
    to a tool-result title from THIS conversation; anything else is an invention.
    An abstaining answer asserts no titles → zero inventions (the abstain path)."""
    allowed_keys = {match_key(t) for t in allowed_titles}
    asserted = extract_asserted_titles(answer)
    inventions = []
    for title in asserted:
        key = match_key(title)
        grounded = key in allowed_keys or any(
            fuzz.ratio(key, ak) / 100.0 >= MATCH_THRESHOLD for ak in allowed_keys
        )
        if not grounded:
            inventions.append(title)
    return HallucinationReport(
        asserted=tuple(asserted),
        inventions=tuple(inventions),
        allowed=frozenset(allowed_titles),
    )
