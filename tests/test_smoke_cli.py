"""Unit tests for the smoke CLI exit codes + messaging (P0 task 6)."""

from __future__ import annotations

import pytest

from sutradhar.serving import smoke
from sutradhar.serving.llm_client import EndpointStatus


def _patch_status(monkeypatch: pytest.MonkeyPatch, status: EndpointStatus) -> None:
    monkeypatch.setattr(smoke, "run", lambda: status)


def test_cli_off_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_status(
        monkeypatch,
        EndpointStatus(
            status="off",
            model="m",
            sample_token=None,
            latency_ms=None,
            detail="endpoint OFF — bring up the on-demand GPU",
        ),
    )
    code = smoke.main()
    assert code == 0  # graceful, green in the paused-GPU default state
    out = capsys.readouterr().out
    assert "OFF" in out
    assert "endpoint OFF" in out


def test_cli_up_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_status(
        monkeypatch,
        EndpointStatus(
            status="up", model="m", sample_token="pong", latency_ms=12.3, detail="endpoint UP"
        ),
    )
    code = smoke.main()
    assert code == 0
    assert "pong" in capsys.readouterr().out


def test_cli_error_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_status(
        monkeypatch,
        EndpointStatus(
            status="error", model="m", sample_token=None, latency_ms=None, detail="boom"
        ),
    )
    assert smoke.main() == 1
