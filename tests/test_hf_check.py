"""Unit tests for the HF Hub auth check (P0 task 7). HfApi is mocked — no network/token."""

from __future__ import annotations

import pytest

from sutradhar.config import ConfigError, Settings
from sutradhar.serving import hf_check
from sutradhar.serving.hf_check import HFAuthError, hf_whoami


class _FakeApi:
    """Stand-in for HfApi; records the token it was given and returns a canned identity."""

    last_token: str | None = None

    def whoami(self, token: str | None = None) -> dict[str, str]:
        _FakeApi.last_token = token
        return {"name": "alice", "type": "user"}


class _RaisingApi:
    def whoami(self, token: str | None = None) -> dict[str, str]:
        raise RuntimeError("401 Unauthorized")


def _settings(**over: str) -> Settings:
    return Settings(_env_file=None, **over)


def test_whoami_valid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_check, "HfApi", _FakeApi)
    name = hf_whoami(_settings(HF_TOKEN="hf_valid_token"))
    assert name == "alice"
    assert _FakeApi.last_token == "hf_valid_token"


def test_whoami_missing_token_raises_named_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_check, "HfApi", _FakeApi)
    with pytest.raises(ConfigError) as exc:
        hf_whoami(_settings())  # no HF_TOKEN
    assert "HF_TOKEN" in str(exc.value)


def test_whoami_invalid_token_raises_clear_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_check, "HfApi", _RaisingApi)
    with pytest.raises(HFAuthError) as exc:
        hf_whoami(_settings(HF_TOKEN="hf_bad_token"))
    msg = str(exc.value)
    assert "auth failed" in msg.lower()
    assert "hf_bad_token" not in msg  # token never leaked into the message


def test_cli_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(hf_check, "hf_whoami", lambda: "alice")
    assert hf_check.main() == 0
    assert "alice" in capsys.readouterr().out


def test_cli_missing_token_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise() -> str:
        raise ConfigError("Missing required env var 'HF_TOKEN': set it in your .env")

    monkeypatch.setattr(hf_check, "hf_whoami", _raise)
    assert hf_check.main() == 1
    assert "HF_TOKEN" in capsys.readouterr().out
