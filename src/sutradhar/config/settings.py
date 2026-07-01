"""Typed, env-driven ``Settings`` — the only place the environment is read.

Design rules (CLAUDE.md / P0_SPEC §2.2):
- All runtime knobs come from env vars; model/endpoint ids are swappable, never hardcoded.
- Secrets are never logged: ``repr``/``str`` redact any field whose env var ends in
  ``_KEY`` / ``_TOKEN`` / ``_PASSWORD``.
- A required-but-missing var yields a clear error naming the var (via :meth:`Settings.require`),
  not a stack trace. Secret tokens are *contextually* required (only for the operation that needs
  them, e.g. ``HF_TOKEN`` for ``make hf-check``), so ``make smoke`` never crashes for lack of one.

Non-secret defaults mirror ``.env.example`` (dev/localhost values + the DEC-0001/DEC-0002 model-id
pins). Env always overrides these, so the config stays fully swappable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Env-var suffixes that mark a value as secret and must be redacted when printed/logged.
_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_PASSWORD")
_REDACTED = "***REDACTED***"


class ConfigError(RuntimeError):
    """Raised when a contextually-required env var is missing or empty."""


class Settings(BaseSettings):
    """All Sutradhar runtime configuration, loaded once from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM endpoint (on-demand vLLM, reached via env-driven URL; normally OFF) ---
    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="google/gemma-4-E4B", validation_alias="LLM_MODEL")
    llm_api_key: str = Field(default="EMPTY", validation_alias="LLM_API_KEY")
    llm_timeout_s: float = Field(default=10.0, validation_alias="LLM_TIMEOUT_S")

    # --- Retrieval models (env defaults = DEC-0002 pins; not loaded in P0) ---
    embed_model: str = Field(default="BAAI/bge-m3", validation_alias="EMBED_MODEL")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias="RERANK_MODEL",
    )

    # --- Postgres (+pgvector) ---
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="sutradhar", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="sutradhar", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="sutradhar", validation_alias="POSTGRES_PASSWORD")

    # --- Redis (optional cache) ---
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    # --- Secrets: contextually required (validate via .require at point of use) ---
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    jarvislabs_api_key: str | None = Field(default=None, validation_alias="JARVISLABS_API_KEY")
    tmdb_api_key: str | None = Field(default=None, validation_alias="TMDB_API_KEY")
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    # --- On-demand GPU ---
    gpu_type: str = Field(default="A100", validation_alias="GPU_TYPE")

    def require(self, field: str) -> str:
        """Return ``field``'s value or raise a clear, var-named :class:`ConfigError`.

        Used by operations that need a contextually-required secret (e.g. ``hf_whoami`` needs
        ``HF_TOKEN``; ``gpu-validate`` needs ``JARVISLABS_API_KEY``). Keeps global load lenient so
        ``make smoke`` works without every secret set.
        """
        value = getattr(self, field)
        if value is None or value == "":
            env_name = self._env_name(field)
            raise ConfigError(
                f"Missing required env var {env_name!r}: set it in your .env (see .env.example)."
            )
        return str(value)

    @classmethod
    def _env_name(cls, field: str) -> str:
        """Map a model field name to its env-var name (the validation alias)."""
        info = cls.model_fields.get(field)
        alias = getattr(info, "validation_alias", None) if info else None
        return str(alias) if alias else field.upper()

    def _redacted_items(self) -> list[str]:
        items: list[str] = []
        for name in self.__class__.model_fields:
            env_name = self._env_name(name)
            value = getattr(self, name)
            if value is not None and env_name.endswith(_SECRET_SUFFIXES):
                value = _REDACTED
            items.append(f"{name}={value!r}")
        return items

    def __repr__(self) -> str:
        return f"Settings({', '.join(self._redacted_items())})"

    __str__ = __repr__


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load :class:`Settings` once (cached) from the environment / ``.env``."""
    return Settings()
