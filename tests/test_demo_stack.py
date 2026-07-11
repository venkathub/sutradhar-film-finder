"""P6 task 8 — the demo stack shape (DEC-P6-5), pinned as Tier-1 meta-tests.

The real from-scratch proof is the CI ``demo-smoke`` job (make demo-up on a fresh
checkout); these tests keep the pieces it depends on from silently drifting.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "infra" / "app" / "Dockerfile"
COMPOSE = REPO_ROOT / "infra" / "docker-compose.yml"
MAKEFILE = REPO_ROOT / "Makefile"


def test_dockerfile_is_the_d5_multistage_shape() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM node:24-slim AS ui-build" in text  # build-time only (DEC-P6-1/Q3)
    assert "ghcr.io/astral-sh/uv" in text  # the official Astral pattern
    assert "--frozen" in text and "--no-dev" in text  # same uv.lock as laptop/CI
    assert "python:3.12-slim" in text  # runtime carries no uv/node/compilers
    assert "ui/app/dist" in text  # the built UI ships in the image
    assert 'CMD ["uvicorn", "--factory", "sutradhar.serving.app:create_app"' in text
    # No model endpoint or secret baked into the image.
    for forbidden in ("LLM_BASE_URL=http", "HF_TOKEN", "API_KEY"):
        assert forbidden not in text


def test_compose_app_service_is_demo_profile_and_env_driven() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    app = compose["services"]["app"]
    assert app["profiles"] == ["demo"]  # plain `make up` stays two-container
    assert app["build"]["dockerfile"] == "infra/app/Dockerfile"
    env = app["environment"]
    # In-network overrides + env-driven model endpoints, EMPTY by default (off
    # is first-class): the gpu-serve live flip is exports, never a rebuild.
    assert env["POSTGRES_HOST"] == "postgres"
    assert env["REDIS_URL"] == "redis://redis:6379/0"
    for var in ("EMBED_BASE_URL", "RERANK_BASE_URL", "RETRIEVAL_RUN", "DEMO_VIDEO_URL"):
        assert env[var] == "${" + var + ":-}", f"{var} must default to empty"
    assert "healthcheck" in app
    deps = app["depends_on"]
    assert deps["postgres"]["condition"] == "service_healthy"
    assert deps["redis"]["condition"] == "service_healthy"


def test_makefile_demo_targets() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "demo-up:" in text and "demo-down:" in text
    # The one-command path: env template -> compose (demo profile) -> migrate -> seed.
    assert "cp .env.example .env" in text
    assert "--profile demo up -d --build --wait" in text
    assert "alembic upgrade head" in text
    assert "seed_graph_ci.py" in text
