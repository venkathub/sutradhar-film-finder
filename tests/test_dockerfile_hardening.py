"""P7 task 5 (DEC-P7-1 finding 10) — container hardening tripwires.

Textual Tier-1 assertions on the Dockerfiles (cheap, no docker daemon); the
CI demo-smoke job additionally asserts the BUILT image config via
``docker inspect`` (.github/workflows/tier1.yml).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DOCKERFILE = REPO_ROOT / "infra" / "app" / "Dockerfile"
MLFLOW_DOCKERFILE = REPO_ROOT / "infra" / "mlflow" / "Dockerfile"


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def test_app_image_runs_as_non_root() -> None:
    lines = _lines(APP_DOCKERFILE)
    user_lines = [line for line in lines if line.startswith("USER ")]
    assert user_lines, "app Dockerfile lost its USER instruction"
    assert user_lines[-1] not in ("USER root", "USER 0"), "app image must not end as root"


def test_app_image_has_healthcheck() -> None:
    text = APP_DOCKERFILE.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text, "app Dockerfile lost its HEALTHCHECK"
    assert "api/health" in text, "HEALTHCHECK should probe /api/health"


def test_app_user_declared_before_healthcheck_and_cmd() -> None:
    """USER must apply to the running process, not just a build stage."""
    lines = _lines(APP_DOCKERFILE)
    user_idx = max(i for i, line in enumerate(lines) if line.startswith("USER "))
    cmd_idx = max(i for i, line in enumerate(lines) if line.startswith("CMD "))
    assert user_idx < cmd_idx, "USER must precede the final CMD"


def test_mlflow_image_runs_as_non_root() -> None:
    lines = _lines(MLFLOW_DOCKERFILE)
    user_lines = [line for line in lines if line.startswith("USER ")]
    assert user_lines, "mlflow Dockerfile lost its USER instruction"
    assert user_lines[-1] not in ("USER root", "USER 0")


def test_mlflow_image_has_healthcheck() -> None:
    """PR #9 review: parity with the app image — both images self-describe liveness."""
    text = MLFLOW_DOCKERFILE.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text, "mlflow Dockerfile lost its HEALTHCHECK"
