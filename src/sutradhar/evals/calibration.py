"""NO_MATCH abstention calibration (P2 task 11, DEC-P2-5).

Operates entirely on the committed eval-run artifact (no DB, no GPU): the abstention
signal is the recorded ``top_rerank_score`` (sigmoid-normalized top-1 cross-encoder
score) per query.

Protocol (canary + margin, per DEC-P2-5):

1. Tune on the **calibration half** of the held-out negatives only:
   ``θ = (1 + RELATIVE_MARGIN) × max(calibration-negative scores)`` — above the top
   canary with a safety margin.
2. **Feasibility check**: the "zero false rejects on positive golden fixtures"
   constraint is verified against the data, not assumed. When positives and negatives
   interleave (measured: they do — code-mixed positives score below out-of-catalog plot
   negatives on the raw cross-encoder), the constraint is reported INFEASIBLE and the
   hard gate (zero false accepts, GS-02 + test half) takes precedence; the resulting
   false rejects are enumerated by fixture id, never hidden.
3. **Validate** on the untouched test half + the GS-02 fixtures: report NO_MATCH
   precision/recall; any false accept there is a phase-gate failure.

``abstain=true`` still carries results (v0 allows both): a falsely-rejected positive
degrades to "low confidence", not to a wrong answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sutradhar.evals.retrieval import EvalRunArtifact

# A-priori safety margin over the top calibration canary (recorded before validation;
# the test half is untouched during tuning).
RELATIVE_MARGIN = 0.35


class CalibrationInputs(BaseModel):
    config_key: str
    positives: dict[str, float]  # retrieval-fixture id -> top_rerank_score
    gs02: dict[str, float]
    calibration_negatives: dict[str, float]
    test_negatives: dict[str, float]


class CurvePoint(BaseModel):
    theta: float
    calib_true_abstains: int
    calib_false_accepts: int
    positive_false_rejects: int
    f1: float


class CalibrationReport(BaseModel):
    config_key: str
    theta: float
    max_calibration_negative: tuple[str, float]
    relative_margin: float
    zero_false_reject_feasible: bool
    infeasibility_witness: str | None  # "<positive> <= <negative>" when infeasible
    positive_false_rejects: list[str]  # fixture ids scoring below θ (documented, not hidden)
    gs02_false_accepts: list[str]
    test_false_accepts: list[str]
    test_no_match_recall: float
    test_no_match_precision: float
    curve: list[CurvePoint] = Field(default_factory=list)


def collect_inputs(artifact: EvalRunArtifact, config_key: str) -> CalibrationInputs:
    record = artifact.records[config_key]
    positives: dict[str, float] = {}
    gs02: dict[str, float] = {}
    for qid, query in record.queries.items():
        if query.top_rerank_score is None:
            continue
        if query.slice == "negative":
            gs02[qid] = query.top_rerank_score
        elif query.slice in ("flagship", "plot_only", "franchise", "code_mixed", "fuzzy_title"):
            positives[qid] = query.top_rerank_score
    calibration = {
        qid: q.top_rerank_score
        for qid, q in record.negatives.items()
        if q.slice == "heldout_calibration" and q.top_rerank_score is not None
    }
    test = {
        qid: q.top_rerank_score
        for qid, q in record.negatives.items()
        if q.slice == "heldout_test" and q.top_rerank_score is not None
    }
    return CalibrationInputs(
        config_key=config_key,
        positives=positives,
        gs02=gs02,
        calibration_negatives=calibration,
        test_negatives=test,
    )


def _f1(true_abstains: int, false_accepts: int, false_rejects: int) -> float:
    denom = 2 * true_abstains + false_accepts + false_rejects
    return round(2 * true_abstains / denom, 4) if denom else 0.0


def calibrate(
    inputs: CalibrationInputs, relative_margin: float = RELATIVE_MARGIN
) -> CalibrationReport:
    if not inputs.calibration_negatives or not inputs.positives:
        raise ValueError("calibration needs both positives and calibration negatives")

    top_neg_id, top_neg = max(inputs.calibration_negatives.items(), key=lambda kv: (kv[1], kv[0]))
    theta = round((1 + relative_margin) * top_neg, 6)

    # Feasibility of the DEC-P2-5 zero-false-reject constraint, measured not assumed.
    min_pos_id, min_pos = min(inputs.positives.items(), key=lambda kv: (kv[1], kv[0]))
    feasible = min_pos > top_neg
    witness = None if feasible else f"{min_pos_id}={min_pos:.5f} <= {top_neg_id}={top_neg:.5f}"

    false_rejects = sorted(qid for qid, s in inputs.positives.items() if s < theta)
    gs02_false_accepts = sorted(qid for qid, s in inputs.gs02.items() if s >= theta)
    test_false_accepts = sorted(qid for qid, s in inputs.test_negatives.items() if s >= theta)
    test_abstained = len(inputs.test_negatives) - len(test_false_accepts)
    # Precision over the validation population: test negatives + all positives.
    abstained_total = test_abstained + len(false_rejects)
    precision = round(test_abstained / abstained_total, 4) if abstained_total else 1.0

    curve: list[CurvePoint] = []
    candidates = sorted({*inputs.positives.values(), *inputs.calibration_negatives.values(), theta})
    for point in candidates:
        true_abstains = sum(1 for s in inputs.calibration_negatives.values() if s < point)
        false_accepts = len(inputs.calibration_negatives) - true_abstains
        rejects = sum(1 for s in inputs.positives.values() if s < point)
        curve.append(
            CurvePoint(
                theta=round(point, 6),
                calib_true_abstains=true_abstains,
                calib_false_accepts=false_accepts,
                positive_false_rejects=rejects,
                f1=_f1(true_abstains, false_accepts, rejects),
            )
        )

    return CalibrationReport(
        config_key=inputs.config_key,
        theta=theta,
        max_calibration_negative=(top_neg_id, top_neg),
        relative_margin=relative_margin,
        zero_false_reject_feasible=feasible,
        infeasibility_witness=witness,
        positive_false_rejects=false_rejects,
        gs02_false_accepts=gs02_false_accepts,
        test_false_accepts=test_false_accepts,
        test_no_match_recall=round(test_abstained / len(inputs.test_negatives), 4)
        if inputs.test_negatives
        else 1.0,
        test_no_match_precision=precision,
        curve=curve,
    )
