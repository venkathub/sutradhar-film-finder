"""Engine/session plumbing for the graph store.

The DSN is built from :class:`sutradhar.config.Settings` (env-driven POSTGRES_* — never
hardcoded). Alembic's ``env.py`` and application code share this single URL builder.
"""

from __future__ import annotations

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from sutradhar.config import Settings, get_settings


def postgres_url(settings: Settings | None = None) -> URL:
    """Build the SQLAlchemy URL for the graph Postgres from env-driven settings."""
    s = settings if settings is not None else get_settings()
    return URL.create(
        "postgresql+psycopg",
        username=s.postgres_user,
        password=s.postgres_password,
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_db,
    )


def create_graph_engine(settings: Settings | None = None) -> Engine:
    """Create an Engine for the graph store (pool defaults; callers own disposal)."""
    return create_engine(postgres_url(settings))


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Standard session factory: explicit commits, no autoflush surprises in pipelines."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
