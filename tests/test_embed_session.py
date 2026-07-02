"""Offline tests for the P2 embed_session driver (DEC-P2-7 HF relay) — fully mocked:
no SDK, no network, no GPU, no HF Hub. Reuses the P0 fake harness; the key assertions
are guaranteed teardown, manifest verification before success, and the token/repo
plumbing of the startup script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sutradhar.config import Settings
from sutradhar.rag.artifacts import ArtifactRun, write_embedding_bank
from tests.test_jarvis_gpu import _base_deps, _FakeClient, jarvis


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        GPU_TYPE="A100",
        HF_TOKEN="hf_fine_grained_test",
        HF_ARTIFACT_REPO="tester/sutradhar-artifacts",
    )


class _FakeHub:
    """In-memory HF relay: upload/exists/download against a dict, sealed-run aware."""

    def __init__(self, tmp_path: Path, *, job_succeeds: bool = True) -> None:
        self.tmp_path = tmp_path
        self.job_succeeds = job_succeeds
        self.uploaded: dict[str, Path] = {}
        self.polls = 0

    def upload_file(self, local: Path, path_in_repo: str) -> None:
        self.uploaded[path_in_repo] = local

    def exists(self, path_in_repo: str) -> bool:
        if path_in_repo.endswith("/EXIT"):
            self.polls += 1
            return self.polls >= 2  # first poll misses, second sees the marker
        if path_in_repo.endswith("/out/MANIFEST.sha256"):
            return self.job_succeeds
        return False

    def fetch_log(self, run_id: str) -> str | None:
        return "Traceback: stub failure" if not self.job_succeeds else None

    def download_dir(self, prefix: str, dest: Path) -> None:
        # Materialize a real sealed run at <dest>/<prefix>/ exactly like snapshot_download.
        out = dest / prefix
        out.mkdir(parents=True, exist_ok=True)
        run = ArtifactRun(out)
        write_embedding_bank(
            run, "queries", ["a" * 64], np.zeros((1, 4), dtype=np.float32), [{1: 1.0}]
        )
        run.write_json("meta.json", {"stub": True})
        run.write_manifest()


def _ok_export(tmp_path: Path) -> Any:
    def run_export(out_path: Path) -> tuple[int, str, str]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"queries": []}), encoding="utf-8")
        return 0, "exported", ""

    return run_export


def _session_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "job_timeout_s": 100,
        "poll_interval_s": 1,
        "run_export": _ok_export(tmp_path),
        "artifacts_root": tmp_path / "artifacts",
        "inputs_path": tmp_path / "gpu_inputs.json",
    }


def test_happy_path_pulls_verified_run_and_destroys(tmp_path: Path) -> None:
    client = _FakeClient()
    hub = _FakeHub(tmp_path)
    ev = jarvis.embed_session(_settings(), _base_deps(client), hub, **_session_kwargs(tmp_path))
    assert ev.status == "up" and ev.booted and ev.destroyed
    assert client.destroyed == [4242]
    assert "RETRIEVAL_RUN=" in ev.detail
    run_id = ev.detail.split("RETRIEVAL_RUN=")[1].split()[0]
    ArtifactRun.open(tmp_path / "artifacts", run_id)  # really on disk, really sealed
    # Both inputs went up before the instance was created.
    assert any(k.endswith("/gpu_inputs.json") for k in hub.uploaded)
    assert any(k.endswith("/embed_and_score.py") for k in hub.uploaded)


def test_failed_job_still_destroys_instance(tmp_path: Path) -> None:
    client = _FakeClient()
    hub = _FakeHub(tmp_path, job_succeeds=False)  # EXIT appears but no sealed run
    ev = jarvis.embed_session(_settings(), _base_deps(client), hub, **_session_kwargs(tmp_path))
    assert ev.status == "error" and not ev.booted
    assert client.destroyed == [4242]  # teardown guaranteed


def test_timeout_still_destroys_instance(tmp_path: Path) -> None:
    client = _FakeClient()
    hub = _FakeHub(tmp_path)
    hub.exists = lambda path: False  # type: ignore[method-assign] — EXIT never appears
    ev = jarvis.embed_session(_settings(), _base_deps(client), hub, **_session_kwargs(tmp_path))
    assert ev.status == "off"
    assert client.destroyed == [4242]


def test_export_failure_never_creates_an_instance(tmp_path: Path) -> None:
    client = _FakeClient()
    kwargs = _session_kwargs(tmp_path)
    kwargs["run_export"] = lambda out: (1, "", "DB down")
    ev = jarvis.embed_session(_settings(), _base_deps(client), _FakeHub(tmp_path), **kwargs)
    assert ev.status == "error" and "export failed" in ev.detail
    assert client.destroyed == []  # no GPU was ever billed


def test_embed_startup_script_plumbs_token_repo_and_run() -> None:
    script = jarvis.build_embed_startup_script(
        "hf_scoped_token", "tester/sutradhar-artifacts", "run-42"
    )
    assert "export HF_TOKEN=hf_scoped_token" in script
    assert "'tester/sutradhar-artifacts', 'run-42'" in script
    assert "FlagEmbedding" in script and "pyarrow" in script
    assert "set -euo pipefail" in script and "-x" not in script.split("\n")[1]  # no token echo
    # The vLLM validate script stays token-free (P0 invariant untouched).
    assert "HF_TOKEN" not in jarvis.build_startup_script("google/gemma-4-E4B")
