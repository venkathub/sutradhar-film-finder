"""Integration tests for the local compose stack (P0 task 4).

Opt-in (marker: ``integration``). Each test skips cleanly when the service is not
reachable, so running the suite without Docker never fails — it just skips.

    docker compose -f infra/docker-compose.yml up -d --wait
    uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from sutradhar.config import Settings

pytestmark = pytest.mark.integration


def test_postgres_has_pgvector() -> None:
    """Postgres is reachable and the pgvector extension can be created (image proof)."""
    psycopg = pytest.importorskip("psycopg")
    s = Settings()
    try:
        conn = psycopg.connect(
            host=s.postgres_host,
            port=s.postgres_port,
            dbname=s.postgres_db,
            user=s.postgres_user,
            password=s.postgres_password,
            connect_timeout=5,
        )
    except psycopg.OperationalError as exc:  # service not up → skip, don't fail
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")

    with conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT '[1,2,3]'::vector;")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "[1,2,3]"
    conn.close()


def test_redis_ping() -> None:
    """Redis is reachable and answers PING with PONG."""
    redis = pytest.importorskip("redis")
    s = Settings()
    client = redis.Redis.from_url(s.redis_url, socket_connect_timeout=5)
    try:
        assert client.ping() is True
    except redis.exceptions.ConnectionError as exc:  # service not up → skip
        pytest.skip(f"Redis not reachable ({exc}); run `make up` first.")
    finally:
        client.close()
