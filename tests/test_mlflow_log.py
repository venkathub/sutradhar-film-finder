"""MLflow logging tests (P3 task 10; P3_SPEC §4 test_mlflow_log.py).

Stamp completeness (every §6.1 field present as a param) and metric/artifact logging,
against a temp **file store** — no server, no network (the compose server is integration).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.config import Settings
from sutradhar.evals.generation_run import (
    GenerationRunArtifact,
    ReproStamp,
    aggregate_metrics,
)
from sutradhar.obs.mlflow_log import (
    backfill_retrieval_run,
    latest_retrieval_artifact,
    log_generation_run,
)

_STAMP_FIELDS = {
    "created_at",
    "code_sha",
    "golden_set_hash",
    "prompt_hash",
    "tool_schema_version",
    "tool_schema_sha256",
    "retrieval_run",
    "ragas_version",
}


def _artifact() -> GenerationRunArtifact:
    from tests.test_generation_run import _result  # reuse the scored-fixture builder

    results = [_result()]
    return GenerationRunArtifact(
        run_id="20260703T000000Z-abcd1234",
        mode="dry_run",
        model="mock",
        serving=None,
        prompt_hash="p" * 8,
        tool_schema_version="v0",
        judge={"coherence": {"model": "openai/gpt-oss-20b", "prompt_hash": "j" * 8}},
        retrieval_run="r1",
        fixtures=results,
        metrics=aggregate_metrics(results, "dry_run"),
        stamp=ReproStamp(
            created_at="2026-07-03T00:00:00+00:00",
            code_sha="abc123",
            golden_set_hash="g" * 8,
            prompt_hash="p" * 8,
            tool_schema_version="v0",
            tool_schema_sha256="t" * 8,
            retrieval_run="r1",
            ragas_version="0.4.3",
        ),
    )


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    # mlflow >= 3.14 put the plain file store in maintenance mode; sqlite is the
    # supported local backend (and matches the DB-backed DEC-P3-2 posture).
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_generation_run_logs_stamp_metrics_artifact(tmp_path: Path, tracking_uri: str) -> None:
    import mlflow

    artifact = _artifact()
    artifact_path = tmp_path / f"{artifact.run_id}.json"
    artifact_path.write_text(artifact.model_dump_json(), "utf-8")

    run_id = log_generation_run(
        artifact, artifact_path, settings=Settings(_env_file=None), tracking_uri=tracking_uri
    )

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    params = run.data.params
    # §6.1 stamp completeness: every stamp field is a logged param.
    for field in _STAMP_FIELDS:
        assert f"stamp.{field}" in params, f"stamp.{field} missing from MLflow params"
    assert params["mode"] == "dry_run"
    assert params["model"] == "mock"
    assert params["retrieval_run"] == "r1"
    assert params["judge.coherence.prompt_hash"] == "j" * 8
    # Table 2 aggregates as metrics (numeric only; None skipped).
    metrics = run.data.metrics
    assert metrics["fixtures_total"] == 1.0
    assert "faithfulness" in metrics
    assert "gs02_inventions" in metrics
    assert any(k.startswith("slice.") for k in metrics)
    assert "latency_p50_ms" not in metrics  # dry_run: null => not logged
    # The committed run JSON rides along as an MLflow artifact.
    artifacts = [f.path for f in mlflow.artifacts.list_artifacts(run_id=run_id)]
    assert f"{artifact.run_id}.json" in artifacts


def test_retrieval_backfill_from_committed_p2_artifact(tracking_uri: str) -> None:
    """The Table 1 backfill: log the REAL committed P2 run; recall@10 must survive."""
    import mlflow

    artifact, path = latest_retrieval_artifact(Path("evals/retrieval_runs"))
    run_id = backfill_retrieval_run(
        artifact, path, settings=Settings(_env_file=None), tracking_uri=tracking_uri
    )
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["run_id"] == artifact.run_id
    assert run.data.params["winner"] == artifact.winner
    assert run.data.params["embed_model"] == artifact.embed_model
    winner_metrics = artifact.metrics[str(artifact.winner)]
    assert run.data.metrics["recall_at_10"] == pytest.approx(winner_metrics["recall@10"])
    assert run.data.metrics["version_set_recall_gs01"] == pytest.approx(
        winner_metrics["version_set_recall_gs01"]
    )
