# Self-hosted MLflow tracking server + model registry (P3, DEC-P3-2 option A).
# The official image ships no Postgres driver; this adds the one runtime dep the
# backend-store URI needs. Image tag pinned (verified on ghcr.io 2026-07-03).
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir "psycopg2-binary~=2.9"
