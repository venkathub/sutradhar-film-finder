"""Named golden retrieval regressions — the P2_SPEC §4 table, wired as the Tier-1 CI
gate (P2 task 12, DEC-P2-6).

Every test recomputes its gating metric from the COMMITTED run artifact
(``evals/retrieval_runs/<run_id>.json``) + the golden fixtures, using the same pure
functions as the eval harness — no DB, no GPU, no network. A regression in any gate
blocks merge; ``test_recall_gate`` additionally pins the recomputed metrics against the
artifact's own metrics block, so the committed numbers can never drift from what the
code computes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sutradhar.evals.calibration import calibrate, collect_inputs
from sutradhar.evals.golden import GoldenFixture, load_fixtures
from sutradhar.evals.retrieval import (
    ConfigRecord,
    EvalRunArtifact,
    compute_metrics,
    fixture_slice,
    version_set_recall,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "evals" / "retrieval_runs"
SCHEMA = json.loads((REPO_ROOT / "docs/phases/tool_schema.v0.json").read_text("utf-8"))


def _committed_run() -> EvalRunArtifact:
    runs = [
        f
        for f in sorted(RUNS_DIR.glob("*.json"))
        if not f.name.endswith((".meta.json", ".calibration.json"))
    ]
    assert runs, "no committed retrieval-run artifact — run `make retrieval-eval`"
    return EvalRunArtifact.model_validate_json(runs[-1].read_text("utf-8"))


@pytest.fixture(scope="module")
def artifact() -> EvalRunArtifact:
    return _committed_run()


@pytest.fixture(scope="module")
def winner(artifact: EvalRunArtifact) -> ConfigRecord:
    assert artifact.winner is not None
    return artifact.records[artifact.winner]


@pytest.fixture(scope="module")
def fixtures() -> dict[str, GoldenFixture]:
    return {f.id: f for f in load_fixtures(REPO_ROOT / "evals" / "golden")}


def _vsr(fixtures: dict[str, GoldenFixture], winner: ConfigRecord, fixture_id: str) -> float:
    fixture = fixtures[fixture_id]
    record = winner.queries[fixture_id]
    return version_set_recall(fixture.expected.versions, record.version_set)


# --- §4 named regressions ---


def test_version_set_recall_gs01(fixtures: dict[str, GoldenFixture], winner: ConfigRecord) -> None:
    """GS-01a/b: the Papanasam/Drishyam family, complete and correctly labelled — = 1.0."""
    assert _vsr(fixtures, winner, "GS-01a") == 1.0
    assert _vsr(fixtures, winner, "GS-01b") == 1.0


def test_version_set_recall_gs06(fixtures: dict[str, GoldenFixture], winner: ConfigRecord) -> None:
    """GS-06a/b: franchise walk (include_sequels) — sequel × remake crossed — = 1.0."""
    assert _vsr(fixtures, winner, "GS-06a") == 1.0
    assert _vsr(fixtures, winner, "GS-06b") == 1.0


def test_no_hallucinated_movie_gs02(artifact: EvalRunArtifact, winner: ConfigRecord) -> None:
    """GS-02a/b/c + the untouched test-half negatives: abstain on ALL — false-accept = 0."""
    gs02 = {qid: q for qid, q in winner.queries.items() if qid.startswith("GS-02")}
    assert len(gs02) == 3
    assert all(q.abstain for q in gs02.values()), {qid: q.abstain for qid, q in gs02.items()}
    test_half = {qid: q for qid, q in winner.negatives.items() if q.slice == "heldout_test"}
    assert len(test_half) == 12
    assert all(q.abstain for q in test_half.values())
    # And the calibration math agrees with the recorded flags (one source of truth).
    report = calibrate(collect_inputs(artifact, str(artifact.winner)))
    assert report.gs02_false_accepts == [] and report.test_false_accepts == []


def test_dub_vs_remake_gs04(winner: ConfigRecord) -> None:
    """GS-04 end-to-end through retrieval: every Baahubali track is a DUB, never a remake."""
    for fixture_id in ("GS-04a", "GS-04b"):
        record = winner.queries[fixture_id]
        assert record.works[0].title == "Baahubali: The Beginning"
        version_set = record.version_set or []
        assert version_set, fixture_id
        labels = {v.relationship for v in version_set}
        assert "is_remake_of" not in labels
        assert "is_official_dub_of" in labels
        # Bilingual double-original: BOTH te and ta tracks flagged original.
        originals = {v.language for v in version_set if v.is_original}
        assert {"te", "ta"} <= originals


def test_sibling_vs_remake_gs05(winner: ConfigRecord) -> None:
    """GS-05: Devdas adaptations are SIBLING works (based_on), never an is_remake_of chain."""
    record = winner.queries["GS-05a"]
    devdas_works = {
        (w.title, w.year): w.work_id for w in record.works if w.title in ("Devdas", "Devadasu")
    }
    # ≥2 sibling adaptations surface as DISTINCT works (never collapsed into one lineage).
    assert len(devdas_works) >= 2
    assert len(set(devdas_works.values())) == len(devdas_works)
    # No adaptation's version set chains a remake edge; each is its own original.
    for version in record.version_set or []:
        assert version.relationship != "is_remake_of"
    # GS-05b: the Tamil track inside a sibling is a DUB (edge types compose).
    b = winner.queries["GS-05b"]
    assert b.works[0].title == "Devadasu"
    tamil = [v for v in b.version_set or [] if v.language == "ta"]
    assert tamil and all(v.relationship == "is_official_dub_of" for v in tamil)


def test_false_merge_gs10(winner: ConfigRecord) -> None:
    """GS-10 'Vikram Kamal Haasan': two DISTINCT Works in the results — false-merge = 0."""
    record = winner.queries["GS-10a"]
    vikrams = {(w.year, w.work_id) for w in record.works if w.title == "Vikram"}
    years = {year for year, _ in vikrams}
    assert {1986, 2022} <= years
    assert len({wid for _, wid in vikrams}) >= 2  # distinct work_ids, never merged


def test_tool_calls_validate_v0(fixtures: dict[str, GoldenFixture], winner: ConfigRecord) -> None:
    """GS-07 expected_tool_calls + every recorded search_by_plot-shaped result: 100% valid v0."""
    params_schema = dict(SCHEMA["tools"]["search_by_plot"]["params"])
    params_schema["$defs"] = SCHEMA["$defs"]
    gs07_calls = [
        call
        for f in fixtures.values()
        if f.id.startswith("GS-07") and f.expected_tool_calls
        for call in f.expected_tool_calls
        if call.tool == "search_by_plot"
    ]
    assert gs07_calls, "GS-07 must carry search_by_plot expected_tool_calls"
    for call in gs07_calls:
        params_validator = Draft202012Validator(params_schema)
        errors = [e.message for e in params_validator.iter_errors(call.arguments)]
        assert errors == [], errors

    result_schema = dict(SCHEMA["tools"]["search_by_plot"]["result"])
    result_schema["$defs"] = SCHEMA["$defs"]
    validator = Draft202012Validator(result_schema)
    for record in winner.queries.values():
        payload = {
            "results": [
                {
                    "work_id": w.work_id,
                    "canonical_title": w.title,
                    "language": w.language,
                    "year": w.year,
                    "score": w.score,
                }
                for w in record.works
            ],
            "abstain": record.abstain,
        }
        errors = [e.message for e in validator.iter_errors(payload)]
        assert errors == [], (record.query_id, errors)


def test_recall_gate(artifact: EvalRunArtifact, fixtures: dict[str, GoldenFixture]) -> None:
    """THE exit gate: Recall@10 ≥ 0.90 (all retrieval fixtures) + VSR gates, recomputed
    from records; recomputation must equal the committed metrics block (no drift)."""
    eval_fixtures = [f for f in fixtures.values() if fixture_slice(f.id) is not None]
    for key, record in artifact.records.items():
        recomputed = compute_metrics(eval_fixtures, record)
        assert recomputed == artifact.metrics[key], f"metrics drift in cell {key}"
    assert artifact.winner is not None
    winner_metrics = artifact.metrics[artifact.winner]
    assert winner_metrics["recall@10"] >= 0.90  # the P4 green light
    assert winner_metrics["version_set_recall_gs01"] == 1.0
    assert winner_metrics["version_set_recall_gs06"] == 1.0


def test_fuzzy_title_holds_through_pipeline(
    fixtures: dict[str, GoldenFixture], winner: ConfigRecord
) -> None:
    """GS-11 (4 perturbations) = 1.0 through the FULL pipeline, not just the repo layer."""
    from sutradhar.evals.retrieval import recall_at_k

    gs11 = [f for f in fixtures.values() if f.id.startswith("GS-11")]
    assert len(gs11) == 4
    for fixture in gs11:
        assert recall_at_k(fixture, winner.queries[fixture.id].works, 10) == 1.0, fixture.id
