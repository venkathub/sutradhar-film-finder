"""Judge human-agreement validation: worksheet + κ report (P3 task 7; P3_SPEC §2.5).

Methodology = the "Judge's Verdict" agreement approach (arXiv 2510.09738): ~24 items
(GS-08 coherence conversations + a faithfulness sample, each with **deterministic foils** —
a re-answering/context-reset variant and an invented-movie variant) are labelled
independently by the human reviewer; the frozen judge scores the same items; we report
percent agreement + Cohen's κ. Gate: **κ ≥ 0.6** (DEC-P3-1 — one rubric revision allowed,
then the frontier escalation).

Flow (task 13 runs this; this module is the machinery):

1. ``make judge-worksheet`` → :func:`build_worksheet` from the committed generation-run
   transcripts → ``evals/judge_validation/worksheet.yaml`` (labels blank; foil provenance
   goes to a separate key file so labelling stays blind).
2. Human fills every ``human_label`` (1 = coherent/faithful, 0 = not).
3. ``make judge-validate`` (inside the ephemeral judge GPU session) → :func:`compute_report`
   → per-item verdicts + κ → ``evals/judge_validation/report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from sutradhar.evals.driver import FixtureTranscript
from sutradhar.evals.generation import collect_result_titles
from sutradhar.evals.judge import (
    COHERENCE_PROMPT,
    FAITHFULNESS_PROMPT,
    JudgeClient,
    JudgeVerdict,
    binarize,
    cohens_kappa,
    percent_agreement,
)

VALIDATION_DIR = Path("evals/judge_validation")
WORKSHEET_FILE = "worksheet.yaml"
KEY_FILE = "worksheet.key.json"
REPORT_FILE = "report.json"

# Deterministic invented film injected into faithfulness foils (never in any catalog).
FOIL_SENTENCE = " Its acclaimed Bengali remake **Chokher Aloy (2016)** is also worth watching."


class WorksheetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: Literal["coherence", "faithfulness"]
    fixture_id: str
    # coherence payload: (user, assistant) turns; faithfulness payload: answer + titles.
    conversation: list[dict[str, str]] | None = None
    answer: str | None = None
    allowed_titles: list[str] | None = None
    human_label: int | None = None  # 1 = coherent/faithful, 0 = not (human fills)


def _conversation_turns(transcript: FixtureTranscript) -> list[dict[str, str]]:
    users = [m["content"] for m in transcript.messages if m.get("role") == "user"]
    turns: list[dict[str, str]] = []
    for i, user in enumerate(users):
        answer = transcript.answers[i] if i < len(transcript.answers) else None
        turns.append({"user": str(user), "assistant": answer or "(no answer)"})
    return turns


def coherence_foil(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministic incoherent variant: every later turn re-answers turn 1 (the exact
    failure mode GS-08's must_not forbids — context reset after a correction)."""
    if not turns:
        return turns
    first_answer = turns[0]["assistant"]
    return [
        {"user": t["user"], "assistant": first_answer if i > 0 else t["assistant"]}
        for i, t in enumerate(turns)
    ]


def faithfulness_foil(answer: str) -> str:
    """Deterministic hallucinated variant: an invented film appended to a real answer."""
    return answer.rstrip() + FOIL_SENTENCE


def build_worksheet(
    transcripts: list[FixtureTranscript],
) -> tuple[list[WorksheetItem], dict[str, Any]]:
    """Items + the (separate) provenance key. Every eligible real item gets a foil twin:
    multi-turn transcripts → coherence pairs; answered transcripts → faithfulness pairs."""
    items: list[WorksheetItem] = []
    key: dict[str, Any] = {}

    def add(item: WorksheetItem, *, is_foil: bool) -> None:
        items.append(item)
        key[item.item_id] = {"fixture_id": item.fixture_id, "is_foil": is_foil}

    for transcript in transcripts:
        answered = [a for a in transcript.answers if a]
        if len(transcript.answers) > 1 and len(answered) > 1:
            turns = _conversation_turns(transcript)
            add(
                WorksheetItem(
                    item_id=f"coh-{transcript.fixture_id}",
                    kind="coherence",
                    fixture_id=transcript.fixture_id,
                    conversation=turns,
                ),
                is_foil=False,
            )
            add(
                WorksheetItem(
                    item_id=f"coh-{transcript.fixture_id}-foil",
                    kind="coherence",
                    fixture_id=transcript.fixture_id,
                    conversation=coherence_foil(turns),
                ),
                is_foil=True,
            )
        if answered:
            final = answered[-1]
            titles = sorted(collect_result_titles(transcript.emitted_calls()))
            add(
                WorksheetItem(
                    item_id=f"fai-{transcript.fixture_id}",
                    kind="faithfulness",
                    fixture_id=transcript.fixture_id,
                    answer=final,
                    allowed_titles=titles,
                ),
                is_foil=False,
            )
            add(
                WorksheetItem(
                    item_id=f"fai-{transcript.fixture_id}-foil",
                    kind="faithfulness",
                    fixture_id=transcript.fixture_id,
                    answer=faithfulness_foil(final),
                    allowed_titles=titles,
                ),
                is_foil=True,
            )
    return items, key


def save_worksheet(
    items: list[WorksheetItem],
    key: dict[str, Any],
    directory: Path = VALIDATION_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WORKSHEET_FILE
    payload = {
        "instructions": (
            "Label EVERY item independently: human_label = 1 when the conversation is "
            "coherent / the answer is faithful to its allowed_titles, else 0. Do not "
            "consult the key file before labelling (it records foil provenance)."
        ),
        "items": [item.model_dump(exclude_none=True) for item in items],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")
    (directory / KEY_FILE).write_text(json.dumps(key, indent=2) + "\n", "utf-8")
    return path


def load_worksheet(directory: Path = VALIDATION_DIR) -> list[WorksheetItem]:
    payload = yaml.safe_load((directory / WORKSHEET_FILE).read_text("utf-8"))
    return [WorksheetItem.model_validate(raw) for raw in payload["items"]]


class ItemVerdict(BaseModel):
    item_id: str
    kind: str
    judge_score: float | None
    judge_binary: int
    human_label: int
    judge_error: str | None = None


class AgreementReport(BaseModel):
    n_items: int
    percent_agreement: float
    cohens_kappa: float
    per_kind: dict[str, dict[str, float]]
    judge: dict[str, Any]  # JudgeConfig dumps per rubric
    verdicts: list[ItemVerdict]
    gate: str  # human-readable κ-gate outcome


def compute_report(items: list[WorksheetItem], judge: JudgeClient) -> AgreementReport:
    missing = [i.item_id for i in items if i.human_label is None]
    if missing:
        raise ValueError(f"worksheet has unlabelled items (fill human_label): {missing}")
    verdicts: list[ItemVerdict] = []
    for item in items:
        if item.kind == "coherence":
            assert item.conversation is not None
            verdict: JudgeVerdict = judge.judge_coherence(item.conversation)
        else:
            assert item.answer is not None
            verdict = judge.judge_faithfulness(item.answer, item.allowed_titles or [])
        assert item.human_label is not None
        verdicts.append(
            ItemVerdict(
                item_id=item.item_id,
                kind=item.kind,
                judge_score=verdict.score,
                judge_binary=binarize(verdict.score),
                human_label=item.human_label,
                judge_error=verdict.error,
            )
        )

    human = [v.human_label for v in verdicts]
    machine = [v.judge_binary for v in verdicts]
    kappa = cohens_kappa(machine, human)
    per_kind: dict[str, dict[str, float]] = {}
    for kind in ("coherence", "faithfulness"):
        sub = [v for v in verdicts if v.kind == kind]
        if sub:
            h = [v.human_label for v in sub]
            m = [v.judge_binary for v in sub]
            per_kind[kind] = {
                "n": float(len(sub)),
                "percent_agreement": percent_agreement(m, h),
                "cohens_kappa": cohens_kappa(m, h),
            }
    return AgreementReport(
        n_items=len(verdicts),
        percent_agreement=percent_agreement(machine, human),
        cohens_kappa=kappa,
        per_kind=per_kind,
        judge={
            "coherence": judge.config(COHERENCE_PROMPT).model_dump(),
            "faithfulness": judge.config(FAITHFULNESS_PROMPT).model_dump(),
        },
        verdicts=verdicts,
        gate=(
            f"kappa={kappa:.3f} — "
            + ("PASS (>= 0.6, judge freezable)" if kappa >= 0.6 else "FAIL (< 0.6, DEC-P3-1)")
        ),
    )


def save_report(report: AgreementReport, directory: Path = VALIDATION_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_FILE
    path.write_text(report.model_dump_json(indent=2) + "\n", "utf-8")
    return path


# --- P7 task 17 (DEC-P7-6): blind test-retest second pass ---------------------------
#
# The ROADMAP asked for a second annotator (human-human kappa ceiling); the available
# second pass is the SAME human who labelled the original worksheet, so it is measured
# and reported as INTRA-RATER (test-retest) reliability -- an upper-bound *proxy*,
# explicitly never presented as a human-human ceiling. Protocol committed BEFORE
# labelling: evals/judge_validation/PROTOCOL.md. The frozen report.json is never
# modified; the new report is additive (report_testretest.json), and the judge leg
# is computed OFFLINE from the frozen report's recorded judge_binary verdicts.

BLIND_WORKSHEET_FILE = "worksheet.blind.yaml"
BLIND_KEY_FILE = "worksheet.blind.key.json"
TESTRETEST_REPORT_FILE = "report_testretest.json"
BLIND_SHUFFLE_SEED = 20260718  # deterministic, recorded — the blinding is reproducible


class BlindItem(BaseModel):
    """A worksheet item stripped for blind relabelling: no label, no foil-revealing id,
    no fixture pairing. Only the content the rater needs."""

    model_config = ConfigDict(extra="forbid")

    blind_id: str
    kind: Literal["coherence", "faithfulness"]
    conversation: list[dict[str, str]] | None = None
    answer: str | None = None
    allowed_titles: list[str] | None = None
    human_label: int | None = None  # the rater fills this in the SECOND pass


def build_blind_worksheet(
    items: list[WorksheetItem], seed: int = BLIND_SHUFFLE_SEED
) -> tuple[list[BlindItem], dict[str, str]]:
    """Blind copy + id map. Original ids embed foil provenance (``…-foil``) and fixture
    pairing, so blind ids are re-minted after a seeded shuffle; the map lives in a
    separate key file the rater never opens."""
    import random

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    blind: list[BlindItem] = []
    id_map: dict[str, str] = {}
    for index, item in enumerate(shuffled, start=1):
        blind_id = f"blind-{index:03d}"
        id_map[blind_id] = item.item_id
        blind.append(
            BlindItem(
                blind_id=blind_id,
                kind=item.kind,
                conversation=item.conversation,
                answer=item.answer,
                allowed_titles=item.allowed_titles,
                human_label=None,
            )
        )
    return blind, id_map


def save_blind_worksheet(
    blind: list[BlindItem], id_map: dict[str, str], directory: Path = VALIDATION_DIR
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BLIND_WORKSHEET_FILE
    payload = {
        "instructions": (
            "SECOND PASS (blind test-retest, DEC-P7-6). Read PROTOCOL.md FIRST. "
            "Label EVERY item independently: human_label = 1 when the conversation is "
            "coherent / the answer is faithful to its allowed_titles, else 0. Do NOT "
            "open worksheet.yaml, worksheet.key.json, worksheet.blind.key.json, or "
            "report*.json until every label is filled."
        ),
        "items": [item.model_dump(exclude_none=True) for item in blind],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")
    (directory / BLIND_KEY_FILE).write_text(json.dumps(id_map, indent=2) + "\n", "utf-8")
    return path


def load_blind_worksheet(directory: Path = VALIDATION_DIR) -> list[BlindItem]:
    payload = yaml.safe_load((directory / BLIND_WORKSHEET_FILE).read_text("utf-8"))
    return [BlindItem.model_validate(raw) for raw in payload["items"]]


class TestRetestReport(BaseModel):
    """Additive agreement report for the blind second pass. FRAMING IS PART OF THE
    ARTIFACT: intra-rater kappa is an upper-bound proxy, not a human-human ceiling."""

    model_config = ConfigDict(extra="forbid")

    framing: str
    n_items: int
    percent_agreement: float
    intra_rater_kappa: float
    intra_rater_kappa_real_items_only: float  # foils excluded
    per_kind: dict[str, dict[str, float]]
    second_pass_vs_judge_kappa: float | None  # offline, from the FROZEN report.json
    first_pass_file: str
    protocol_file: str


def compute_testretest_report(
    original: list[WorksheetItem],
    blind: list[BlindItem],
    id_map: dict[str, str],
    frozen_judge_binaries: dict[str, int] | None = None,
) -> TestRetestReport:
    """Pure computation — reads nothing, writes nothing, mutates nothing frozen."""
    unlabelled = [b.blind_id for b in blind if b.human_label is None]
    if unlabelled:
        raise ValueError(f"blind worksheet has unlabelled items: {unlabelled}")
    by_original_id = {item.item_id: item for item in original}
    pairs: list[tuple[str, int, int]] = []  # (original_id, first, second)
    for blind_item in blind:
        original_id = id_map[blind_item.blind_id]
        first = by_original_id[original_id].human_label
        if first is None:
            raise ValueError(f"original worksheet item {original_id} is unlabelled")
        assert blind_item.human_label is not None
        pairs.append((original_id, first, blind_item.human_label))

    firsts = [p[1] for p in pairs]
    seconds = [p[2] for p in pairs]
    real = [p for p in pairs if not p[0].endswith("-foil")]
    per_kind: dict[str, dict[str, float]] = {}
    for kind in ("coherence", "faithfulness"):
        sub = [p for p in pairs if by_original_id[p[0]].kind == kind]
        if sub:
            per_kind[kind] = {
                "n": float(len(sub)),
                "percent_agreement": percent_agreement([p[1] for p in sub], [p[2] for p in sub]),
                "cohens_kappa": cohens_kappa([p[1] for p in sub], [p[2] for p in sub]),
            }

    judge_kappa: float | None = None
    if frozen_judge_binaries is not None:
        judged = [p for p in pairs if p[0] in frozen_judge_binaries]
        if judged:
            judge_kappa = cohens_kappa(
                [frozen_judge_binaries[p[0]] for p in judged], [p[2] for p in judged]
            )

    return TestRetestReport(
        framing=(
            "INTRA-RATER (test-retest) reliability: the same human relabelled blind "
            "(ids re-minted, order reshuffled, >= 14-day gap). An upper-bound PROXY "
            "for label stability — NOT a human-human inter-rater ceiling (DEC-P7-6); "
            "a genuine second human would add that ceiling additively."
        ),
        n_items=len(pairs),
        percent_agreement=percent_agreement(firsts, seconds),
        intra_rater_kappa=cohens_kappa(firsts, seconds),
        intra_rater_kappa_real_items_only=cohens_kappa([p[1] for p in real], [p[2] for p in real]),
        per_kind=per_kind,
        second_pass_vs_judge_kappa=judge_kappa,
        first_pass_file=WORKSHEET_FILE,
        protocol_file="PROTOCOL.md",
    )


def save_testretest_report(report: TestRetestReport, directory: Path = VALIDATION_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TESTRETEST_REPORT_FILE
    path.write_text(report.model_dump_json(indent=2) + "\n", "utf-8")
    return path
