"""Alembic environment for the Sutradhar graph store.

The connection URL comes from :func:`sutradhar.graph.db.postgres_url` (env-driven
``POSTGRES_*`` settings) — never from ``alembic.ini`` and never hardcoded.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from sutradhar.graph.db import postgres_url
from sutradhar.graph.schema import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a DB connection (`alembic upgrade --sql`)."""
    context.configure(
        url=postgres_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live env-configured Postgres."""
    engine = create_engine(postgres_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
