"""MLflow experiment tracking (P3 task 10; DEC-P3-2 self-hosted compose service).

Experiments (naming convention, P3_SPEC §2.6):
- ``sutradhar/generation`` — every generation run (dry-run machinery evidence + the
  authoritative live captures), with the full §6.1 reproducibility stamp as params,
  the Table 2 aggregates as metrics, and the committed run JSON as an artifact.
- ``sutradhar/retrieval`` — the P2 Table 1 backfill (discharges the Table 1 stamp's
  "(MLflow wiring lands in P3)" note) and future retrieval runs.

Runnable for the backfill::

    uv run python -m sutradhar.obs.mlflow_log backfill-retrieval   # make mlflow-backfill

Degrades cleanly: callers wrap :func:`log_generation_run` and print a warning when the
tracking server is down — an eval run never fails because MLflow is off.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sutradhar.config import Settings, get_settings
from sutradhar.evals.generation_run import GenerationRunArtifact
from sutradhar.evals.retrieval import EvalRunArtifact

EXPERIMENT_GENERATION = "sutradhar/generation"
EXPERIMENT_RETRIEVAL = "sutradhar/retrieval"
EXPERIMENT_SERVING = "sutradhar/serving"

RETRIEVAL_RUNS_DIR = Path("evals/retrieval_runs")


def _flatten(prefix: str, node: Any, out: dict[str, float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(node, bool):
        out[prefix] = float(node)
    elif isinstance(node, int | float):
        out[prefix] = float(node)


def log_generation_run(
    artifact: GenerationRunArtifact,
    artifact_path: Path,
    *,
    settings: Settings | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log one generation run (params = §6.1 stamp; metrics = Table 2 aggregates).

    Returns the MLflow run id (recorded in docs/BENCHMARKS.md as the run link).
    """
    import mlflow  # lazy: heavy import kept out of eval hot paths

    settings = settings or get_settings()
    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_GENERATION)
    with mlflow.start_run(run_name=artifact.run_id) as run:
        params: dict[str, Any] = {
            "run_id": artifact.run_id,
            "mode": artifact.mode,
            "model": artifact.model,
            "tool_schema_version": artifact.tool_schema_version,
            "retrieval_run": artifact.retrieval_run,
            **{f"stamp.{k}": v for k, v in artifact.stamp.model_dump().items()},
        }
        if artifact.serving:
            params.update({f"serving.{k}": v for k, v in artifact.serving.items()})
        if artifact.judge:
            for rubric, config in artifact.judge.items():
                params[f"judge.{rubric}.model"] = config.get("model")
                params[f"judge.{rubric}.prompt_hash"] = config.get("prompt_hash")
        mlflow.log_params(params)

        metrics: dict[str, float] = {}
        _flatten("", artifact.metrics.model_dump(exclude={"slices"}), metrics)
        _flatten("slice", artifact.metrics.slices, metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path))
        return str(run.info.run_id)


def backfill_retrieval_run(
    artifact: EvalRunArtifact,
    artifact_path: Path,
    *,
    settings: Settings | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log the committed P2 retrieval run's Table 1 metrics (winner config) — the
    backfill that discharges the Table 1 stamp note. A log, not a re-run."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_RETRIEVAL)
    if artifact.winner is None:
        raise ValueError(f"retrieval run {artifact.run_id} has no winner config")
    winner = artifact.records[artifact.winner]
    with mlflow.start_run(run_name=artifact.run_id) as run:
        mlflow.log_params(
            {
                "run_id": artifact.run_id,
                "embed_model": artifact.embed_model,
                "rerank_model": artifact.rerank_model,
                "code_sha": artifact.code_sha,
                "winner": artifact.winner,
                "golden_fixture_count": artifact.golden_fixture_count,
                "negative_count": artifact.negative_count,
                **{f"config.{k}": v for k, v in winner.retrieval_config.items()},
            }
        )
        metrics: dict[str, float] = {}
        _flatten("", artifact.metrics[artifact.winner], metrics)
        # MLflow metric keys must not contain '@'.
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
        mlflow.log_artifact(str(artifact_path))
        return str(run.info.run_id)


def log_serving_run(
    artifact: Any,
    artifact_path: Path,
    *,
    settings: Settings | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log one P5 serving-benchmark window (§6 DoD: eval thresholds recorded to MLflow).

    Params = the §6.1 reproducibility stamp + serving config; metrics = the four capture
    legs flattened (parity Recall@10/VSR, injection ASR on/off + FP + utility, latency
    p50/p95 + tokens/sec, answer_relevancy). Replayable GPU-free from the sealed artifact
    (``sutradhar.serving.benchmark.ServingRunArtifact``). Degrades if MLflow is off."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_SERVING)
    with mlflow.start_run(run_name=artifact.run_id) as run:
        params: dict[str, Any] = {
            "run_id": artifact.run_id,
            "created_at": artifact.created_at,
            "code_sha": artifact.code_sha,
            "model": artifact.model,
            "prompt_hash": artifact.prompt_hash,  # v1.1 serving bundle
            "tool_schema_version": artifact.tool_schema_version,
            "tool_schema_sha256": artifact.tool_schema_sha256,
            "all_ok": artifact.all_ok,
        }
        params.update(
            {f"serving.{k}": v for k, v in artifact.serving.items() if not isinstance(v, dict)}
        )
        mlflow.log_params(params)

        metrics: dict[str, float] = {}
        for leg in ("parity", "injection", "latency", "relevancy"):
            step = getattr(artifact, leg)
            metrics[f"{leg}.ok"] = float(step.ok)
            _flatten(leg, step.data, metrics)
        # MLflow metric keys allow only [A-Za-z0-9_./ -]; drop '@' and 'defenses_on/off'
        # dotted booleans already coerced by _flatten.
        clean = {k.replace("@", "_at_"): v for k, v in metrics.items()}
        mlflow.log_metrics(clean)
        mlflow.log_artifact(str(artifact_path))
        return str(run.info.run_id)


def latest_serving_artifact(runs_dir: Path = Path("evals/serving_runs")) -> tuple[Any, Path]:
    from sutradhar.serving.benchmark import ServingRunArtifact

    runs = [f for f in sorted(runs_dir.glob("*.json"))]
    if not runs:
        raise FileNotFoundError(f"no committed serving run under {runs_dir}")
    path = runs[-1]
    return ServingRunArtifact.model_validate_json(path.read_text("utf-8")), path


def latest_retrieval_artifact(runs_dir: Path = RETRIEVAL_RUNS_DIR) -> tuple[EvalRunArtifact, Path]:
    runs = [
        f
        for f in sorted(runs_dir.glob("*.json"))
        if not f.name.endswith((".meta.json", ".calibration.json"))
    ]
    if not runs:
        raise FileNotFoundError(f"no committed retrieval run under {runs_dir}")
    path = runs[-1]
    return EvalRunArtifact.model_validate_json(path.read_text("utf-8")), path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "backfill-retrieval":
        artifact, path = latest_retrieval_artifact()
        run_id = backfill_retrieval_run(artifact, path)
        print(f"backfilled Table 1 run {artifact.run_id} -> MLflow run {run_id}")
        print(f"experiment: {EXPERIMENT_RETRIEVAL} @ {get_settings().mlflow_tracking_uri}")
        return 0
    if cmd == "log-serving":
        artifact, path = latest_serving_artifact()
        run_id = log_serving_run(artifact, path)
        print(f"logged serving window {artifact.run_id} -> MLflow run {run_id}")
        print(f"experiment: {EXPERIMENT_SERVING} @ {get_settings().mlflow_tracking_uri}")
        return 0
    print("usage: python -m sutradhar.obs.mlflow_log [backfill-retrieval|log-serving]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
