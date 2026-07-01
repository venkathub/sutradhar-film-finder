"""Unit tests for the env-driven Settings subsystem (P0 task 3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sutradhar.config import ConfigError, Settings

_ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any Sutradhar env vars so defaults are exercised deterministically."""
    for name in Settings.model_fields:
        env_name = Settings._env_name(name)
        monkeypatch.delenv(env_name, raising=False)


def _parse_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def test_defaults_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://localhost:8000/v1"
    assert s.llm_model == "google/gemma-4-E4B"
    assert s.llm_api_key == "EMPTY"
    assert s.llm_timeout_s == 10.0
    assert s.embed_model == "BAAI/bge-m3"
    assert s.postgres_port == 5432
    assert s.hf_token is None


def test_defaults_match_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every non-empty value in .env.example must equal the corresponding Settings default."""
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    example = _parse_env_example()
    for name in Settings.model_fields:
        env_name = Settings._env_name(name)
        assert env_name in example, f"{env_name} missing from .env.example"
        documented = example[env_name]
        if documented == "":
            # Blank in the template => secret the user fills; default should be None.
            assert getattr(s, name) is None, f"{env_name} blank in template but has a default"
            continue
        actual = getattr(s, name)
        # Compare numerically when the field is numeric so 10 == 10.0, else compare as strings.
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            assert float(actual) == float(documented), (
                f"{env_name}: default {actual!r} != .env.example {documented!r}"
            )
        else:
            assert str(actual) == documented, (
                f"{env_name}: default {actual!r} != .env.example {documented!r}"
            )


def test_missing_required_raises_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    with pytest.raises(ConfigError) as exc:
        s.require("hf_token")
    assert "HF_TOKEN" in str(exc.value)
    # A clear message, not a traceback dump.
    assert "set it in your .env" in str(exc.value)


def test_require_returns_value_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    s = Settings(_env_file=None)
    assert s.require("hf_token") == "hf_secret_value"


def test_secrets_redacted_in_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_topsecret_ABC")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg_topsecret_XYZ")
    monkeypatch.setenv("JARVISLABS_API_KEY", "jl_topsecret_123")
    s = Settings(_env_file=None)
    for rendered in (repr(s), str(s)):
        assert "hf_topsecret_ABC" not in rendered
        assert "pg_topsecret_XYZ" not in rendered
        assert "jl_topsecret_123" not in rendered
        assert "***REDACTED***" in rendered
    # Non-secret values remain visible for debuggability.
    assert "http://localhost:8000/v1" in repr(s)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu.example:8000/v1")
    s = Settings(_env_file=None)
    assert s.llm_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert s.llm_base_url == "http://gpu.example:8000/v1"


def test_no_real_secret_literals_in_env_example() -> None:
    """.env.example must not carry a filled-in token/key (blank secrets only)."""
    example = _parse_env_example()
    for key in ("HF_TOKEN", "JARVISLABS_API_KEY", "TMDB_API_KEY", "LANGFUSE_SECRET_KEY"):
        assert example.get(key, "") == "", f"{key} must be blank in .env.example"
    # No obvious HF token literal anywhere in the file.
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", _ENV_EXAMPLE.read_text(encoding="utf-8"))
