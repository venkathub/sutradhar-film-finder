"""CI workflow meta-tests (P0 task 9): workflows are valid YAML, the PR path is Tier-1-safe,
and no secret literal is committed to tracked files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / ".github" / "workflows"
_TIER1 = _WORKFLOWS / "tier1.yml"

# Steps that would make a job touch a GPU or a neural model — forbidden in PR-triggered jobs.
_UNSAFE = re.compile(r"gpu-validate|gpu-nuke|jarvis|vllm serve|make gpu", re.IGNORECASE)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_tier1_is_valid_yaml_with_expected_jobs() -> None:
    wf = _load(_TIER1)
    assert wf["name"] == "tier-1"
    jobs = wf["jobs"]
    for job in ("lint-type-test", "integration", "secret-guard", "hf-auth"):
        assert job in jobs, f"tier-1 missing job {job!r}"


def test_tier1_pr_path_has_no_gpu_or_model_steps() -> None:
    """The PR-triggered jobs must never run a GPU/model step (cost + no secrets in Tier-1)."""
    wf = _load(_TIER1)
    # `on:` includes pull_request
    on = wf["on"] if "on" in wf else wf[True]  # PyYAML may parse bare `on:` as boolean True
    assert "pull_request" in on
    text = _TIER1.read_text(encoding="utf-8")
    for job_name, job in wf["jobs"].items():
        # hf-auth is explicitly excluded from PRs via its `if:`; others run on PRs.
        if job_name == "hf-auth":
            assert "pull_request" in job.get("if", ""), "hf-auth must be excluded from PRs"
            continue
        for step in job.get("steps", []):
            assert not _UNSAFE.search(str(step.get("run", ""))), (
                f"unsafe step in PR job {job_name}: {step}"
            )
    assert "artifact-validate stub" in text or "recorded artifacts" in text


def test_no_secret_literals_in_tracked_files() -> None:
    """git grep for HF tokens / private keys across tracked files (lockfile excluded)."""
    try:
        proc = subprocess.run(
            [
                "git",
                "grep",
                "-nE",
                r"hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",
                "--",
                ".",
                ":!*.lock",
            ],
            cwd=_REPO,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:  # git not available
        pytest.skip("git not available")
    # git grep exits 1 when there are NO matches — that's what we want.
    assert proc.returncode == 1, f"possible secret literal committed:\n{proc.stdout}"
