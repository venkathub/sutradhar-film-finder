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
_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_TOKENS", "_PASSWORD")
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
    llm_model: str = Field(default="google/gemma-4-E4B-it", validation_alias="LLM_MODEL")
    llm_api_key: str = Field(default="EMPTY", validation_alias="LLM_API_KEY")
    # vLLM serving flags for the model under test (P4 window finding, 2026-07-04: tool
    # calling needs --enable-auto-tool-choice + a model-family parser or every tools
    # request 400s). Model-family-specific => env-swappable, never hardcoded in sessions.
    vllm_serve_flags: str = Field(
        default="--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4",
        validation_alias="VLLM_SERVE_FLAGS",
    )
    llm_timeout_s: float = Field(default=10.0, validation_alias="LLM_TIMEOUT_S")

    # --- Retrieval models (env defaults = DEC-0002 pins; not loaded in P0) ---
    embed_model: str = Field(default="BAAI/bge-m3", validation_alias="EMBED_MODEL")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias="RERANK_MODEL",
    )
    # P5 live path: OpenAI-compatible /v1/embeddings on the on-demand GPU (DEC-P0-4
    # status:"off" contract). Unset by default — P2 runs from recorded artifacts.
    embed_base_url: str | None = Field(default=None, validation_alias="EMBED_BASE_URL")
    # Pinned artifact run id (data/artifacts/retrieval/<run_id>) that CI/demo read.
    retrieval_run: str | None = Field(default=None, validation_alias="RETRIEVAL_RUN")

    # --- Data-source endpoints (P1 ingestion; env-swappable, never hardcoded in code) ---
    wikidata_api_url: str = Field(
        default="https://www.wikidata.org/w/api.php",
        validation_alias="WIKIDATA_API_URL",
    )
    wikidata_sparql_url: str = Field(
        default="https://query.wikidata.org/sparql",
        validation_alias="WIKIDATA_SPARQL_URL",
    )
    tmdb_api_url: str = Field(
        default="https://api.themoviedb.org/3",
        validation_alias="TMDB_API_URL",
    )
    imdb_datasets_url: str = Field(
        default="https://datasets.imdbws.com",
        validation_alias="IMDB_DATASETS_URL",
    )
    # Per-language MediaWiki Action API; {lang} is filled per wiki (en, ml, ta, …).
    wikipedia_api_url: str = Field(
        default="https://{lang}.wikipedia.org/w/api.php",
        validation_alias="WIKIPEDIA_API_URL",
    )
    # WMF User-Agent policy: descriptive UA with a contact; override with your fork/contact.
    http_user_agent: str = Field(
        default="SutradharBot/0.1 (https://github.com/sutradhar/sutradhar; data-pipeline)",
        validation_alias="HTTP_USER_AGENT",
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
    # DEC-P2-7 HF-relay: private dataset repo that ferries GPU-job inputs/artifacts.
    hf_artifact_repo: str | None = Field(default=None, validation_alias="HF_ARTIFACT_REPO")
    jarvislabs_api_key: str | None = Field(default=None, validation_alias="JARVISLABS_API_KEY")
    tmdb_api_key: str | None = Field(default=None, validation_alias="TMDB_API_KEY")
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")
    # DEC-P3-7: AIC Cloud VPS provisioning for the self-hosted Langfuse (make langfuse-up).
    aiccloud_api_key: str | None = Field(default=None, validation_alias="AICCLOUD_API_KEY")

    # --- MLflow tracking + registry (P3, DEC-P3-2: self-hosted compose service) ---
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias="MLFLOW_TRACKING_URI",
    )

    # --- LLM-as-judge endpoint (P3, DEC-P3-1: self-hosted OSS judge on the ephemeral
    # GPU; frontier API escalation is the same client with different env values).
    # Unset by default — judge-dependent steps skip cleanly, never crash. ---
    judge_base_url: str | None = Field(default=None, validation_alias="JUDGE_BASE_URL")
    judge_model: str | None = Field(default=None, validation_alias="JUDGE_MODEL")
    judge_api_key: str | None = Field(default=None, validation_alias="JUDGE_API_KEY")

    # Pinned generation-run artifact id (evals/generation_runs/<run_id>.json) that
    # Tier-1 gates on between GPU windows (DEC-P2-6 posture, generation surface).
    generation_run: str | None = Field(default=None, validation_alias="GENERATION_RUN")

    # --- Synthetic-data teacher endpoint (P4, DEC-P4-1: Sarvam-M 24B self-hosted on
    # the ephemeral GPU; frontier-API escalation is the same client with different
    # env values — A<->B is config, never code). Unset by default — teacher-dependent
    # steps skip cleanly, never crash (same posture as the judge). ---
    teacher_base_url: str | None = Field(default=None, validation_alias="TEACHER_BASE_URL")
    teacher_model: str | None = Field(default=None, validation_alias="TEACHER_MODEL")
    teacher_api_key: str | None = Field(default=None, validation_alias="TEACHER_API_KEY")

    # --- Fine-tune artifacts (P4, DEC-P4-7: HF Hub hosting with cards) ---
    # Adapter repo (public at publish time), e.g. <user>/sutradhar-gemma4-e4b-qlora-v1.
    hf_adapter_repo: str | None = Field(default=None, validation_alias="HF_ADAPTER_REPO")
    # Dataset repo (PRIVATE-first pending the LICENSING review), e.g. <user>/sutradhar-ft-v1.
    ft_dataset_repo: str | None = Field(default=None, validation_alias="FT_DATASET_REPO")
    # Dataset id stamped on the card + sealed JSONL (non-secret, defaultable).
    ft_dataset_id: str = Field(default="sutradhar-ft-v1", validation_alias="FT_DATASET_ID")

    # --- On-demand GPU ---
    gpu_type: str = Field(default="A100", validation_alias="GPU_TYPE")
    # Instance disk (GB). Sessions that download large bf16 checkpoints for on-the-fly
    # quantization (P4 teacher: Sarvam-M ~48 GB) raise this themselves; 80 GB suits the
    # 4B-class serve/train sessions.
    gpu_storage_gb: int = Field(default=80, validation_alias="GPU_STORAGE_GB")

    # --- P5 serving (P5_SPEC §2.2) ---
    # Live reranker endpoint (/rerank on the GPU sidecar). Mirrors EMBED_BASE_URL exactly:
    # unset = GPU off = first-class degraded state, never an error (DEC-P0-4 posture).
    rerank_base_url: str | None = Field(default=None, validation_alias="RERANK_BASE_URL")
    # FastAPI orchestration service port (make api-up).
    api_port: int = Field(default=8080, validation_alias="API_PORT")
    # Conversation-state TTL in Redis (DEC-P5-2); natural session expiry.
    session_ttl_s: int = Field(default=3600, validation_alias="SESSION_TTL_S")
    # A100 40GB hourly rate (DEC-0003) — cost accounting: tokens/sec + amortized $/request.
    gpu_hourly_usd: float = Field(default=0.89, validation_alias="GPU_HOURLY_USD")
    # serve-session hold TTL (DEC-P5-4): bounded live window, then destroy-in-finally.
    serve_hold_minutes: int = Field(default=60, validation_alias="SERVE_HOLD_MINUTES")

    # --- P7 serving security (DEC-P7-2): the paid path is never silently open ---
    # "required" (default) enforces bearer auth on the GPU-up chat path;
    # "disabled" is the explicit local/e2e opt-out (logged, never implicit).
    chat_auth: str = Field(default="required", validation_alias="CHAT_AUTH")
    # Comma-separated static bearer tokens (per-person; revocation = removal).
    # Redacted in logs via the _TOKENS suffix. Unset + auth required => 503.
    api_auth_tokens: str | None = Field(default=None, validation_alias="API_AUTH_TOKENS")
    # slowapi limit string for POST /api/chat, keyed token-first then IP.
    chat_rate_limit: str = Field(default="10/minute", validation_alias="CHAT_RATE_LIMIT")
    # Honor X-Forwarded-For for IP keying ONLY behind a trusted proxy.
    trust_proxy: bool = Field(default=False, validation_alias="TRUST_PROXY")

    # --- P6 UI & portfolio surface (P6_SPEC §2.2) ---
    # Recorded demo video (GitHub Release asset, DEC-P6-3). Unset ⇒ the offline
    # payload omits the link; set after the P6 task-11 rehearsal window.
    demo_video_url: str | None = Field(default=None, validation_alias="DEMO_VIDEO_URL")
    # Canonical URL of the static always-available surface (GitHub Pages, DEC-P6-3).
    # Docs/link-check use only — the app never depends on it.
    site_base_url: str | None = Field(default=None, validation_alias="SITE_BASE_URL")

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
