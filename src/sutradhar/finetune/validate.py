"""Dataset validators (P4 task 5, P4_SPEC §2.1/§4; DEC-P4-2/P4-3).

The formal gate a dataset must pass before it can be sealed (``build_dataset.py
validate`` / ``make validate-dataset``). Task 4's generator holds these properties by
construction on FRESH output; the validators re-earn them on ANY dataset — including one
that has been through the teacher pass (task 6), where "by construction" no longer holds.

Layers (each reuses the frozen machinery — no new scoring semantics):
- **Tool calls / results**: DEC-P1-8 jsonschema validation against ``tool_schema.v0.json``
  (``validate_emitted_call`` + the result-side mirror) — a hallucinated tool or parameter
  is not *trainable*, by CI construction.
- **Grounding**: the GS-02 invented-title detector over every final answer vs that
  conversation's own tool results.
- **Contracts**: INTENT preamble on every final answer (parseable, taxonomy-valid,
  label-consistent); tool-calling turns carry no prose.
- **Decontamination**: max rapidfuzz similarity (over ``match_key``, threshold 0.80 —
  DEC-P1-5) of every training USER utterance vs golden queries ∪ frozen exemplar user
  turns ∪ ALL negative surfaces (GS-02 + held-out) → the card's :class:`DecontReport`.
  Entity-level: every ``entity_ids`` member must be in the committed D3 fixture list
  (``finetune/training_slice_entities.json``), which is disjoint from golden entities by
  the task-3 structural exclusion rule.
- **Mix quotas**: behaviour shares ±tolerance, language thresholds, NO_MATCH share.
- **Teacher lock** (consumed by task 6): sentinel-locking of entity spans + a verifier
  that rejects rewrites which alter locked spans, add titles, or drop the preamble.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

from sutradhar.evals.driver import load_tool_schema, validate_emitted_call
from sutradhar.evals.generation import detect_hallucinated_movies, parse_intent_preamble
from sutradhar.evals.golden import GOLDEN_DIR, load_fixtures
from sutradhar.evals.negatives import NEGATIVES_PATH, load_negatives
from sutradhar.finetune.dataset import DecontReport, TrainingConversation
from sutradhar.finetune.scaffold import BEHAVIOUR_SHARES, CODE_MIXED_LANGS, mix_stats
from sutradhar.finetune.snapshot import result_subschema
from sutradhar.pipeline.normalize import MATCH_THRESHOLD, match_key

EXEMPLARS_PATH = Path("evals/prompts/exemplars_v1.md")
ENTITIES_PATH = Path("finetune/training_slice_entities.json")

_USER_LINE_RE = re.compile(r"^User:\s*(.+)$", re.MULTILINE)


class Issue(BaseModel):
    """One validation failure, with enough context to fix or reject the row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conv_id: str
    kind: str
    detail: str


# --- Layer 1: tool calls + results against frozen v0 ---


def validate_tool_calls(
    conversations: list[TrainingConversation], schema: dict[str, Any]
) -> list[Issue]:
    issues: list[Issue] = []
    for conv in conversations:
        for turn in conv.turns:
            for call in turn.tool_calls or []:
                for error in validate_emitted_call(schema, call.tool, call.arguments):
                    issues.append(Issue(conv_id=conv.conv_id, kind="tool_call", detail=error))
    return issues


def validate_tool_results(
    conversations: list[TrainingConversation], schema: dict[str, Any]
) -> list[Issue]:
    validators: dict[str, Draft202012Validator] = {}
    issues: list[Issue] = []
    for conv in conversations:
        pending: str | None = None
        for turn in conv.turns:
            if turn.tool_calls:
                pending = turn.tool_calls[-1].tool
            if turn.tool_result is None:
                continue
            if pending is None:
                issues.append(
                    Issue(conv_id=conv.conv_id, kind="tool_result", detail="result without a call")
                )
                continue
            if pending not in validators:
                validators[pending] = Draft202012Validator(result_subschema(schema, pending))
            for error in validators[pending].iter_errors(turn.tool_result):
                issues.append(
                    Issue(
                        conv_id=conv.conv_id,
                        kind="tool_result",
                        detail=f"{pending}: {error.message}",
                    )
                )
    return issues


# --- Layer 2: grounding + formatting contracts ---


def _result_titles(conv: TrainingConversation) -> set[str]:
    titles: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("title", "matched_title", "canonical_title") and isinstance(value, str):
                    titles.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for turn in conv.turns:
        if turn.tool_result is not None:
            walk(turn.tool_result)
    return titles


def final_answers(conv: TrainingConversation) -> list[str]:
    """The final prose assistant answer of each user turn."""
    return [
        t.content
        for t in conv.turns
        if t.role == "assistant" and t.content is not None and t.tool_calls is None
    ]


def validate_grounding(conversations: list[TrainingConversation]) -> list[Issue]:
    issues: list[Issue] = []
    for conv in conversations:
        allowed = _result_titles(conv)
        for answer in final_answers(conv):
            report = detect_hallucinated_movies(answer, allowed)
            if report.invention_count:
                issues.append(
                    Issue(
                        conv_id=conv.conv_id,
                        kind="invented_title",
                        detail=f"asserted but ungrounded: {list(report.inventions)}",
                    )
                )
    return issues


def validate_contracts(
    conversations: list[TrainingConversation], intents: set[str], slot_keys: set[str]
) -> list[Issue]:
    issues: list[Issue] = []
    for conv in conversations:
        answers = final_answers(conv)
        if len(answers) != len(conv.intent_labels):
            issues.append(
                Issue(
                    conv_id=conv.conv_id,
                    kind="contract",
                    detail=f"{len(answers)} final answers for {len(conv.intent_labels)} user turns",
                )
            )
            continue
        for answer, intent in zip(answers, conv.intent_labels, strict=True):
            parsed = parse_intent_preamble(answer)
            if parsed is None:
                issues.append(
                    Issue(conv_id=conv.conv_id, kind="contract", detail="missing INTENT preamble")
                )
                continue
            if parsed.intent != intent:
                issues.append(
                    Issue(
                        conv_id=conv.conv_id,
                        kind="contract",
                        detail=f"preamble intent {parsed.intent!r} != label {intent!r}",
                    )
                )
            if parsed.intent not in intents:
                issues.append(
                    Issue(
                        conv_id=conv.conv_id,
                        kind="contract",
                        detail=f"intent {parsed.intent!r} outside the frozen taxonomy",
                    )
                )
            if not set(parsed.slots) <= slot_keys:
                issues.append(
                    Issue(
                        conv_id=conv.conv_id,
                        kind="contract",
                        detail="slot keys outside vocabulary: "
                        f"{sorted(set(parsed.slots) - slot_keys)}",
                    )
                )
        for turn in conv.turns:
            if turn.tool_calls is not None and turn.content is not None:
                issues.append(
                    Issue(
                        conv_id=conv.conv_id, kind="contract", detail="tool-calling turn has prose"
                    )
                )
    return issues


# --- Layer 3: decontamination (DEC-P4-3 scope: golden ∪ exemplars ∪ ALL negatives) ---


def protected_surfaces(
    golden_dir: Path = GOLDEN_DIR,
    negatives_path: Path = NEGATIVES_PATH,
    exemplars_path: Path = EXEMPLARS_PATH,
) -> dict[str, list[str]]:
    """The three protected query surfaces, as raw strings."""
    golden: list[str] = []
    for fixture in load_fixtures(golden_dir):
        if isinstance(fixture.query, str):
            golden.append(fixture.query)
        else:
            golden.extend(fixture.query)
    exemplars = _USER_LINE_RE.findall(exemplars_path.read_text(encoding="utf-8"))
    negatives = [f.query for f in load_negatives(negatives_path)]
    # GS-02 queries live in the golden dir (category=negative) AND gate the abstention
    # calibration — count them on the negatives surface too (DEC-P4-3 scope wording).
    gs02 = [
        q
        for fixture in load_fixtures(golden_dir)
        if fixture.category == "negative"
        for q in ([fixture.query] if isinstance(fixture.query, str) else fixture.query)
    ]
    return {"golden": golden, "exemplars": exemplars, "negatives": negatives + gs02}


def user_utterances(conversations: list[TrainingConversation]) -> list[tuple[str, str]]:
    return [
        (conv.conv_id, turn.content)
        for conv in conversations
        for turn in conv.turns
        if turn.role == "user" and turn.content
    ]


def compute_decontamination(
    conversations: list[TrainingConversation],
    surfaces: dict[str, list[str]] | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> DecontReport:
    surfaces = surfaces if surfaces is not None else protected_surfaces()
    keys = {name: [match_key(q) for q in queries] for name, queries in surfaces.items()}
    maxima = {name: 0.0 for name in surfaces}
    violations: set[str] = set()
    for conv_id, utterance in user_utterances(conversations):
        u_key = match_key(utterance)
        for name, surface_keys in keys.items():
            for s_key in surface_keys:
                score = fuzz.ratio(u_key, s_key) / 100.0
                if score > maxima[name]:
                    maxima[name] = round(score, 4)
                if score >= threshold:
                    violations.add(conv_id)
    return DecontReport(
        threshold=threshold,
        max_similarity_golden=maxima.get("golden", 0.0),
        max_similarity_exemplars=maxima.get("exemplars", 0.0),
        max_similarity_negatives=maxima.get("negatives", 0.0),
        violations=sorted(violations),
    )


def validate_entities(
    conversations: list[TrainingConversation], entities_path: Path = ENTITIES_PATH
) -> list[Issue]:
    """Every grounded entity id must be in the committed D3 fixture list (which the task-3
    exclusion rule keeps disjoint from every golden fixture entity)."""
    import json

    payload = json.loads(entities_path.read_text(encoding="utf-8"))
    known = {w["work_id"] for w in payload["works"]}
    known.update(v["version_id"] for v in payload["versions"])
    issues: list[Issue] = []
    for conv in conversations:
        unknown = sorted(set(conv.entity_ids) - known)
        if unknown:
            issues.append(
                Issue(
                    conv_id=conv.conv_id,
                    kind="entity",
                    detail=f"entity ids outside the D3 training-slice fixture list: {unknown}",
                )
            )
    return issues


# --- Layer 4: mix quotas ---


def validate_quotas(
    conversations: list[TrainingConversation],
    shares: dict[str, float] | None = None,
    tolerance: float = 0.03,
    code_mixed_min: float = 0.40,
    native_min: float = 0.10,
) -> list[Issue]:
    shares = shares or BEHAVIOUR_SHARES
    total = len(conversations)
    issues: list[Issue] = []
    if total == 0:
        return [Issue(conv_id="-", kind="quota", detail="empty dataset")]
    stats = mix_stats(conversations)
    for behaviour, share in shares.items():
        count = sum(stats.get(behaviour, {}).values())
        if abs(count / total - share) > tolerance:
            issues.append(
                Issue(
                    conv_id="-",
                    kind="quota",
                    detail=f"{behaviour}: {count}/{total} vs target share {share}",
                )
            )
    langs = [c.query_lang for c in conversations]
    code_mixed = sum(1 for lang in langs if lang in CODE_MIXED_LANGS) / total
    native = sum(1 for lang in langs if lang == "native") / total
    if code_mixed < code_mixed_min:
        issues.append(
            Issue(
                conv_id="-",
                kind="quota",
                detail=f"code-mixed {code_mixed:.2%} < {code_mixed_min:.0%}",
            )
        )
    if native < native_min:
        issues.append(
            Issue(conv_id="-", kind="quota", detail=f"native {native:.2%} < {native_min:.0%}")
        )
    return issues


# --- Layer 5: teacher placeholder lock (consumed by task 6) ---

SENTINEL_RE = re.compile(r"\[\[T(\d+)\]\]")


def lock_entities(text: str, entities: list[str]) -> tuple[str, dict[str, str]]:
    """Replace entity spans with [[Tn]] sentinels (longest-first, case-insensitive;
    ASCII-safe — rare glyphs like ⟦⟧ get mangled by byte-level BPE teachers).

    Returns ``(locked_text, mapping)`` where mapping is sentinel -> original span. The
    teacher may rewrite everything EXCEPT sentinels; :func:`verify_locked` enforces it.
    """
    mapping: dict[str, str] = {}
    locked = text
    for i, entity in enumerate(sorted(set(entities), key=len, reverse=True), start=1):
        sentinel = f"[[T{i}]]"
        if entity.isdigit():  # years: don't lock digit runs inside larger numbers/ids
            pattern = re.compile(rf"(?<!\d){re.escape(entity)}(?!\d)")
        else:
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
        if pattern.search(locked):
            locked = pattern.sub(sentinel, locked)
            mapping[sentinel] = entity
    return locked, mapping


def verify_locked(
    original: str,
    rewritten: str,
    mapping: dict[str, str],
    require_preamble: bool = False,
) -> list[str]:
    """Rejection reasons for a teacher rewrite of a locked text (empty list = accept).

    Rejects when: a locked sentinel was dropped or duplicated-away, an unknown sentinel
    appears, a NEW title assertion (bold span or Title (year) pattern) was added, a
    bolded sentinel lost its bold, list lines were collapsed, the NO_MATCH token was
    dropped, words were glued together, Indic script was flipped to Latin, or the INTENT
    preamble was dropped/altered while required. (The last five are 2026-07-03 live-pilot
    failure modes — deterministic checks, not vibes.)
    """
    reasons: list[str] = []
    for sentinel in mapping:
        if original.count(sentinel) != rewritten.count(sentinel):
            reasons.append(f"locked span altered: {sentinel} ({mapping[sentinel]!r})")
        bold = f"**{sentinel}**"
        if original.count(bold) > rewritten.count(bold):
            reasons.append(f"bold stripped from locked title: {sentinel} ({mapping[sentinel]!r})")
    for found in set(SENTINEL_RE.findall(rewritten)):
        if f"[[T{found}]]" not in mapping:
            reasons.append(f"unknown sentinel introduced: [[T{found}]]")
    # No new title assertions: any bold span in the rewrite must be a sentinel.
    for span in re.findall(r"\*\*(.+?)\*\*", rewritten):
        if not SENTINEL_RE.fullmatch(span.strip()):
            reasons.append(f"new bold title asserted: {span!r}")
    # Structure: list lines survive as list lines (rule 4 of the rewrite prompt).
    original_items = sum(1 for line in original.splitlines() if line.startswith("- "))
    rewritten_items = sum(1 for line in rewritten.splitlines() if line.startswith("- "))
    if original_items and rewritten_items != original_items:
        reasons.append(f"list structure changed: {original_items} items -> {rewritten_items}")
    # Abstention token survives (rule 5) — a NO_MATCH answer may not stop abstaining.
    if "NO_MATCH" in original and "NO_MATCH" not in rewritten:
        reasons.append("NO_MATCH token dropped")
    # Whitespace-collapse guard (2026-07-03 pilot: the teacher occasionally glues words —
    # "amanwithphasmophobia"): a rewrite may not contain absurdly longer "words" than the
    # original had.
    def _max_wordlen(text: str) -> int:
        return max((len(w) for w in text.split()), default=0)

    if rewritten and _max_wordlen(rewritten) > max(_max_wordlen(original) + 10, 30):
        reasons.append("whitespace collapsed (words glued together)")

    # Script preservation: a native-script text may not flip to Latin wholesale.
    def _indic_share(text: str) -> float:
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        return sum(1 for c in chars if ord(c) >= 0x0900) / len(chars)

    if _indic_share(original) >= 0.20 and _indic_share(rewritten) < 0.10:
        reasons.append("native script flipped to Latin")

    # Meta-leak guard (2026-07-03 pilot 5: the teacher sometimes answers the REWRITE
    # PROMPT itself — "Please provide the input text you'd like me to rewrite…").
    lowered = rewritten.casefold()
    if any(
        phrase in lowered
        for phrase in (
            "rewrite",
            "rewritten text",
            "placeholder",
            "guidelines",
            "provide the input",
            "input text",
            "surface realization",
        )
    ):
        reasons.append("meta leak: rewrite talks about the rewriting task")
    # Length-ratio guard: a rewrite is a register change, not an essay or a stub.
    if original.strip() and rewritten.strip():
        ratio = len(rewritten) / max(len(original), 1)
        if ratio > 2.5 or ratio < 0.3:
            reasons.append(f"length ratio out of bounds ({ratio:.2f}x)")
    if require_preamble:
        original_head = original.split("\n", 1)[0]
        rewritten_head = rewritten.split("\n", 1)[0]
        if not rewritten_head.startswith("INTENT: "):
            reasons.append("INTENT preamble dropped")
        elif rewritten_head != original_head:
            reasons.append("INTENT preamble altered")
    return reasons


def unlock_entities(text: str, mapping: dict[str, str]) -> str:
    for sentinel, entity in mapping.items():
        text = text.replace(sentinel, entity)
    return text


# --- Aggregate ---


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations: int
    issues: list[Issue]
    decontamination: DecontReport

    @property
    def ok(self) -> bool:
        return not self.issues and not self.decontamination.violations


def validate_dataset(
    conversations: list[TrainingConversation],
    schema: dict[str, Any] | None = None,
    intents: set[str] | None = None,
    slot_keys: set[str] | None = None,
    surfaces: dict[str, list[str]] | None = None,
    entities_path: Path = ENTITIES_PATH,
) -> ValidationReport:
    """Run every layer; the sealed dataset requires ``report.ok``."""
    import json

    schema = schema or load_tool_schema()
    if intents is None or slot_keys is None:
        taxonomy = json.loads(
            Path("evals/prompts/intent_taxonomy_v1.json").read_text(encoding="utf-8")
        )
        intents = intents or set(taxonomy["intents"])
        slot_keys = slot_keys or set(taxonomy["slot_keys"])
    issues = [
        *validate_tool_calls(conversations, schema),
        *validate_tool_results(conversations, schema),
        *validate_grounding(conversations),
        *validate_contracts(conversations, intents, slot_keys),
        *validate_entities(conversations, entities_path),
        *validate_quotas(conversations),
    ]
    report = compute_decontamination(conversations, surfaces=surfaces)
    return ValidationReport(conversations=len(conversations), issues=issues, decontamination=report)
