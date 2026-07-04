"""The frozen keep/cut verdict (P4 task 11; DEC-P4-8 — committed BEFORE the GPU window).

A PURE function over two committed :class:`GenerationRunArtifact`s. The rule, verbatim
(user-confirmed 2026-07-03; the margins cannot move after the numbers exist):

    KEEP the adapter iff
      (i)   strict improvement on >= 2 of the 3 primary metrics
            {GS-07 intent accuracy, GS-07 slot F1, GS-08 coherence};
      (ii)  at least one improving primary metric clears >= +0.05 absolute
            (judge/small-n noise alone can never trigger a keep at n = 12 fixtures);
      (iii) no primary metric regresses;
      (iv)  all guards hold: GS-02 inventions = 0 on BOTH columns,
            schema validity >= base, tool-call sequence accuracy >= base.
    Anything else -> CUT, finding recorded (the DEC-0001 pre-commitment).

Small-n honesty: every number is reported as-is with per-slice counts; there is no
significance theater and no rounding before comparison.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sutradhar.evals.generation import SlotCounts, micro_f1
from sutradhar.evals.generation_run import GenerationRunArtifact

MARGIN = 0.05  # DEC-P4-8 clause (ii); frozen
PRIMARY_METRICS = ("gs07_intent_accuracy", "gs07_slot_f1", "gs08_coherence")


class ColumnMetrics(BaseModel):
    """One Table-2 column, reduced to what the rule + the demo table need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    model: str
    prompt_hash: str
    # Primary (D8 clause i-iii)
    gs07_intent_accuracy: float | None
    gs07_slot_f1: float | None
    gs08_coherence: float | None
    # Guards (D8 clause iv)
    gs02_inventions: int
    schema_validity: float | None
    tool_call_sequence_accuracy: float | None
    # Reported (never part of the rule)
    faithfulness: float | None
    answer_relevancy: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    tokens_per_sec: float | None
    gs07_n: int
    coherence_n: int


def extract_column(artifact: GenerationRunArtifact) -> ColumnMetrics:
    """Reduce a committed run artifact to a column (byte-derived, no re-scoring)."""
    gs07 = [r for r in artifact.fixtures if r.slice == "code_mixed"]
    gs07_intents = [m for r in gs07 if r.intent_matches for m in r.intent_matches]
    intent_acc = sum(1.0 for m in gs07_intents if m) / len(gs07_intents) if gs07_intents else None
    slot_total = SlotCounts()
    slot_seen = False
    for r in gs07:
        if r.slot_counts is not None:
            slot_seen = True
            slot_total = slot_total + SlotCounts(**r.slot_counts)
    coherent = [r.judge_coherence for r in artifact.fixtures if r.judge_coherence is not None]
    m = artifact.metrics
    return ColumnMetrics(
        run_id=artifact.run_id,
        model=artifact.model,
        prompt_hash=artifact.prompt_hash,
        gs07_intent_accuracy=intent_acc,
        gs07_slot_f1=micro_f1(slot_total) if slot_seen else None,
        gs08_coherence=(sum(coherent) / len(coherent)) if coherent else None,
        gs02_inventions=m.gs02_inventions,
        schema_validity=m.schema_validity,
        tool_call_sequence_accuracy=m.tool_call_sequence_accuracy,
        faithfulness=m.faithfulness,
        answer_relevancy=m.answer_relevancy,
        latency_p50_ms=m.latency_p50_ms,
        latency_p95_ms=m.latency_p95_ms,
        tokens_per_sec=m.tokens_per_sec,
        gs07_n=len(gs07),
        coherence_n=len(coherent),
    )


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keep: bool
    improved: tuple[str, ...]  # primary metrics strictly improved
    regressed: tuple[str, ...]  # primary metrics strictly regressed
    margin_met: bool  # clause (ii)
    guard_failures: tuple[str, ...]  # clause (iv) violations
    reasons: tuple[str, ...]  # human-readable, one line per clause outcome


def decide(base: ColumnMetrics, qlora: ColumnMetrics) -> Verdict:
    """DEC-P4-8, verbatim. Pure; no I/O; no rounding before comparison."""
    reasons: list[str] = []
    improved: list[str] = []
    regressed: list[str] = []
    margin_met = False
    incomplete = False
    for name in PRIMARY_METRICS:
        b = getattr(base, name)
        q = getattr(qlora, name)
        if b is None or q is None:
            incomplete = True
            reasons.append(f"{name}: MISSING on {'base' if b is None else 'qlora'} — cannot keep")
            continue
        if q > b:
            improved.append(name)
            if q - b >= MARGIN:
                margin_met = True
            reasons.append(f"{name}: {b:.4f} -> {q:.4f} (+{q - b:.4f})")
        elif q < b:
            regressed.append(name)
            reasons.append(f"{name}: {b:.4f} -> {q:.4f} (REGRESSION {q - b:.4f})")
        else:
            reasons.append(f"{name}: {b:.4f} -> {q:.4f} (tie)")

    guard_failures: list[str] = []
    if base.gs02_inventions != 0:
        guard_failures.append(f"GS-02 inventions on BASE = {base.gs02_inventions} (must be 0)")
    if qlora.gs02_inventions != 0:
        guard_failures.append(f"GS-02 inventions on QLORA = {qlora.gs02_inventions} (must be 0)")
    if (
        base.schema_validity is not None
        and qlora.schema_validity is not None
        and qlora.schema_validity < base.schema_validity
    ):
        guard_failures.append(
            f"schema validity {qlora.schema_validity:.4f} < base {base.schema_validity:.4f}"
        )
    if (
        base.tool_call_sequence_accuracy is not None
        and qlora.tool_call_sequence_accuracy is not None
        and qlora.tool_call_sequence_accuracy < base.tool_call_sequence_accuracy
    ):
        guard_failures.append(
            f"tool-call sequence accuracy {qlora.tool_call_sequence_accuracy:.4f} "
            f"< base {base.tool_call_sequence_accuracy:.4f}"
        )
    if qlora.schema_validity is None or qlora.tool_call_sequence_accuracy is None:
        guard_failures.append("guard metric missing on the QLORA column")

    keep = (
        not incomplete
        and len(improved) >= 2  # clause (i)
        and margin_met  # clause (ii)
        and not regressed  # clause (iii)
        and not guard_failures  # clause (iv)
    )
    reasons.append(
        f"clause(i) >=2 improved: {len(improved)}/3; clause(ii) margin>= {MARGIN}: "
        f"{margin_met}; clause(iii) regressions: {len(regressed)}; "
        f"clause(iv) guard failures: {len(guard_failures)}"
    )
    return Verdict(
        keep=keep,
        improved=tuple(improved),
        regressed=tuple(regressed),
        margin_met=margin_met,
        guard_failures=tuple(guard_failures),
        reasons=tuple(reasons),
    )


def render_table(base: ColumnMetrics, qlora: ColumnMetrics, verdict: Verdict) -> str:
    """The 30-second-demo table (`make ft-verdict`), GPU off."""

    def fmt(v: float | None, pct: bool = False) -> str:
        if v is None:
            return "—"
        return f"{v:.1%}" if pct else f"{v:.3f}"

    rows = [
        (
            "GS-07 intent accuracy *",
            fmt(base.gs07_intent_accuracy, True),
            fmt(qlora.gs07_intent_accuracy, True),
        ),
        ("GS-07 slot F1 *", fmt(base.gs07_slot_f1), fmt(qlora.gs07_slot_f1)),
        ("GS-08 coherence *", fmt(base.gs08_coherence), fmt(qlora.gs08_coherence)),
        ("GS-02 inventions (=0 gate)", str(base.gs02_inventions), str(qlora.gs02_inventions)),
        (
            "schema validity (guard)",
            fmt(base.schema_validity, True),
            fmt(qlora.schema_validity, True),
        ),
        (
            "tool-call seq accuracy (guard)",
            fmt(base.tool_call_sequence_accuracy, True),
            fmt(qlora.tool_call_sequence_accuracy, True),
        ),
        ("faithfulness", fmt(base.faithfulness), fmt(qlora.faithfulness)),
        ("answer relevancy", fmt(base.answer_relevancy), fmt(qlora.answer_relevancy)),
        ("latency p50 (ms)", fmt(base.latency_p50_ms), fmt(qlora.latency_p50_ms)),
        ("latency p95 (ms)", fmt(base.latency_p95_ms), fmt(qlora.latency_p95_ms)),
        ("tokens/sec", fmt(base.tokens_per_sec), fmt(qlora.tokens_per_sec)),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [
        f"{'metric'.ljust(width)}  {'BASE':>12}  {'QLORA':>12}",
        f"{'-' * width}  {'-' * 12}  {'-' * 12}",
        *[f"{name.ljust(width)}  {b:>12}  {q:>12}" for name, b, q in rows],
        "",
        f"base:  {base.run_id} ({base.model})",
        f"qlora: {qlora.run_id} ({qlora.model})",
        f"(* = DEC-P4-8 primary; GS-07 n={qlora.gs07_n}, coherence n={qlora.coherence_n} — "
        "exact fractions, no significance theater)",
        "",
        *verdict.reasons,
        "",
        f"VERDICT: {'KEEP the adapter' if verdict.keep else 'CUT the adapter'} (DEC-P4-8)",
    ]
    return "\n".join(lines)
