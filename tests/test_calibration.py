"""Hermetic unit tests for abstention calibration (P2 task 11) — synthetic score
distributions with hand-computed outcomes, plus a regression on the real committed
artifact (repo file, no DB)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.evals.calibration import CalibrationInputs, calibrate, collect_inputs
from sutradhar.evals.retrieval import EvalRunArtifact

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "evals" / "retrieval_runs"


def _inputs(**over: object) -> CalibrationInputs:
    fields: dict[str, object] = {
        "config_key": "test/d20",
        "positives": {"GS-01a": 0.9, "GS-03a": 0.5, "GS-07a": 0.02},
        "gs02": {"GS-02a": 0.008, "GS-02b": 0.004},
        "calibration_negatives": {"NEG-01": 0.01, "NEG-05": 0.04},
        "test_negatives": {"NEG-03": 0.02, "NEG-07": 0.05},
    }
    fields.update(over)
    return CalibrationInputs.model_validate(fields)


def test_theta_is_margin_over_top_canary() -> None:
    report = calibrate(_inputs(), relative_margin=0.35)
    assert report.theta == pytest.approx(0.04 * 1.35)
    assert report.max_calibration_negative == ("NEG-05", 0.04)


def test_feasible_when_positives_clear_negatives() -> None:
    report = calibrate(_inputs(positives={"GS-01a": 0.9, "GS-03a": 0.5}))
    assert report.zero_false_reject_feasible is True
    assert report.infeasibility_witness is None
    assert report.positive_false_rejects == []


def test_infeasibility_is_witnessed_not_hidden() -> None:
    report = calibrate(_inputs())  # GS-07a=0.02 < NEG-05=0.04
    assert report.zero_false_reject_feasible is False
    assert report.infeasibility_witness is not None
    assert "GS-07a" in report.infeasibility_witness and "NEG-05" in report.infeasibility_witness
    assert report.positive_false_rejects == ["GS-07a"]


def test_gate_counters_hand_computed() -> None:
    # θ = 0.054: test negs 0.02 (abstained) / 0.05 (abstained); add a 0.06 accept.
    report = calibrate(_inputs(test_negatives={"NEG-03": 0.02, "NEG-07": 0.06}))
    assert report.theta == pytest.approx(0.054)
    assert report.test_false_accepts == ["NEG-07"]
    assert report.test_no_match_recall == 0.5
    assert report.gs02_false_accepts == []  # both GS-02 scores < θ


def test_curve_is_monotone_in_theta() -> None:
    report = calibrate(_inputs())
    thetas = [p.theta for p in report.curve]
    assert thetas == sorted(thetas)
    abstains = [p.calib_true_abstains for p in report.curve]
    assert abstains == sorted(abstains)  # raising θ can only abstain more


def test_requires_data() -> None:
    with pytest.raises(ValueError, match="calibration needs"):
        calibrate(_inputs(calibration_negatives={}))


# --- Regression on the real committed artifact (the recorded DEC-P2-5 outcome) ---


def _real_artifact() -> EvalRunArtifact | None:
    files = sorted(RUNS_DIR.glob("*.json"))
    runs = [f for f in files if not f.name.endswith((".meta.json", ".calibration.json"))]
    return EvalRunArtifact.model_validate_json(runs[0].read_text("utf-8")) if runs else None


def test_recorded_calibration_outcome_holds() -> None:
    artifact = _real_artifact()
    assert artifact is not None, "committed run artifact missing"
    report = calibrate(collect_inputs(artifact, str(artifact.winner)))
    # The phase gate: zero false accepts on GS-02 AND the untouched test half.
    assert report.gs02_false_accepts == []
    assert report.test_false_accepts == []
    assert report.test_no_match_recall == 1.0
    # The recorded, documented trade-off (DEC-P2-5): exactly these four flagged positives.
    assert report.positive_false_rejects == ["GS-03a", "GS-03c", "GS-07a", "GS-07b"]
    # θ wired into the retriever matches the recomputed value (no silent drift).
    from sutradhar.rag.retrieve import CALIBRATED_NO_MATCH_THRESHOLD

    assert report.theta == pytest.approx(CALIBRATED_NO_MATCH_THRESHOLD)
