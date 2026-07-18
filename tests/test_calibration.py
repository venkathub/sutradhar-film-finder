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
    # P7 task 8 (DEC-P7-3): the wired value now COMES FROM the pinned artifact.
    from sutradhar.rag.calibration import calibrated_threshold

    assert report.theta == pytest.approx(calibrated_threshold())


# --- P7 task 8 (DEC-P7-3): θ binding + staleness gate ---


def test_theta_round_trips_from_the_pinned_artifact() -> None:
    from sutradhar.rag.calibration import load_calibration
    from sutradhar.rag.retrieve import RetrievalConfig

    binding = load_calibration()
    assert binding.theta == pytest.approx(0.151747)  # DEC-P2-5 value, unchanged
    assert binding.config_key == "1024tok_15pct/d20"
    assert binding.embed_model == "BAAI/bge-m3"
    # RetrievalConfig's default is the artifact value — no independent constant.
    config = RetrievalConfig(
        chunk_config=binding.chunk_config,
        embed_model=binding.embed_model,
        index_version=binding.run_id,
        rerank_depth=binding.rerank_depth,
    )
    assert config.no_match_threshold == pytest.approx(binding.theta)


def test_no_theta_literal_remains_in_rag_source() -> None:
    """Grep tripwire: the calibrated value must never be re-hardcoded in code."""
    rag_dir = Path("src/sutradhar/rag")
    offenders = [
        path for path in rag_dir.rglob("*.py") if "0.151747" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"θ literal re-hardcoded in {offenders} — it lives in the artifact"


def test_stale_calibration_hard_fails() -> None:
    from sutradhar.rag.calibration import (
        StaleCalibrationError,
        assert_calibration_matches,
        load_calibration,
    )

    binding = load_calibration()
    ok = dict(
        embed_model=binding.embed_model,
        index_version=binding.run_id,
        chunk_config=binding.chunk_config,
        rerank_depth=binding.rerank_depth,
    )
    assert assert_calibration_matches(**ok) == binding  # exact match passes
    for drift in (
        {"embed_model": "BAAI/bge-multilingual-gemma2"},  # embedder swap (DEC-0002 9B leg)
        {"index_version": "20990101T000000Z-deadbeef"},  # rebuilt index
        {"chunk_config": "256tok_15pct"},  # different ablation cell
        {"rerank_depth": 50},  # different rerank depth
    ):
        with pytest.raises(StaleCalibrationError, match="never silently reused"):
            assert_calibration_matches(**{**ok, **drift})


def test_tampered_artifact_changes_are_detected(tmp_path: Path) -> None:
    """The binding follows the artifact: a mutated embed_model in a copied run is
    faithfully loaded — and then trips the staleness gate against the real stack."""
    import json as _json

    from sutradhar.rag.calibration import (
        StaleCalibrationError,
        assert_calibration_matches,
        load_calibration,
    )

    src = Path("evals/retrieval_runs")
    run = "20260702T135315Z-f6583183"
    for suffix in (".calibration.json", ".meta.json"):
        (tmp_path / f"{run}{suffix}").write_text(
            (src / f"{run}{suffix}").read_text(encoding="utf-8"), encoding="utf-8"
        )
    meta_path = tmp_path / f"{run}.meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    meta["meta"]["embed_model"] = "tampered/embedder"
    meta_path.write_text(_json.dumps(meta), encoding="utf-8")

    tampered = load_calibration(run, tmp_path)
    assert tampered.embed_model == "tampered/embedder"
    with pytest.raises(StaleCalibrationError):
        assert_calibration_matches(
            embed_model="BAAI/bge-m3",
            index_version=run,
            chunk_config=tampered.chunk_config,
            rerank_depth=tampered.rerank_depth,
            binding=tampered,
        )
