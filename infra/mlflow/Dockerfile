# Self-hosted MLflow tracking server + model registry (P3, DEC-P3-2 option A).
# The official image ships no Postgres driver; this adds the one runtime dep the
# backend-store URI needs. Image tag pinned (verified on ghcr.io 2026-07-03).
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir "psycopg2-binary~=2.9"

# P7 task 5 (DEC-P7-1 finding 10): non-root runtime — MLflow only talks to
# Postgres and serves HTTP. uid/gid 1000 matches the typical dev-host owner of
# the bind-mounted ../data/mlflow-artifacts dir (this profile is local-only;
# CI never starts it).
RUN groupadd --gid 1000 mlflow && useradd --uid 1000 --gid mlflow --home-dir /tmp mlflow
USER mlflow
# PR #9 review: image-level healthcheck (parity with the app image) — MLflow's
# canonical liveness route; stdlib urllib, no curl in the base image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=2)"]
