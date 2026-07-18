"""Repo-hygiene tripwire (P7 task 2; DEC-P7-1 finding 10, repo-hygiene half).

Asserts that no build/run debris is ever *tracked* by git. `.gitignore` alone
cannot guarantee this: a `git add -f`, a rule reshuffle, or a new artifact dir
outside an ignored parent silently starts tracking debris. This test makes the
clean state a CI-enforced invariant instead of a one-time cleanup.

Runs `git ls-files` (the tracked-file list — cheap, no working-tree scan) and
fails on any match against the denylist below. Tier-1 (no GPU, no model, no DB).
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Glob patterns over tracked paths. Keep in sync with .gitignore intent, but note
# this list is deliberately narrower: it names things that must NEVER be tracked,
# not everything that is merely ignored.
TRACKED_PATH_DENYLIST: tuple[str, ...] = (
    # Experiment-tracking / artifact stores (referenced by id, never committed)
    "mlruns/*",
    "mlartifacts/*",
    "data/mlflow-artifacts/*",
    # Data + artifact trees (versioned artifacts live on the HF Hub / in evals/*_runs)
    "data/raw/*",
    "data/interim/*",
    "data/processed/*",
    "data/artifacts/*",
    # Staging debris (DEC-P7-1: .staging was only transitively ignored before P7)
    "*.staging/*",
    "*/.staging/*",
    ".staging/*",
    # Model weights / checkpoints (HF Hub is the artifact registry)
    "*.gguf",
    "*.safetensors",
    "*.ckpt",
    "*.pt",
    "*.bin",
    "checkpoints/*",
    "adapters/*",
    # Secrets (only the example may be tracked)
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    # Local DB / vector-store volumes
    "pgdata/*",
    "qdrant_storage/*",
    "*.duckdb",
    "*.sqlite",
    # Caches
    "*/__pycache__/*",
    "*.pyc",
    "node_modules/*",
    "*/node_modules/*",
)

# Tracked paths that legitimately match a denylist pattern.
ALLOWLIST: tuple[str, ...] = (".env.example",)


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def test_no_debris_is_tracked() -> None:
    violations: list[tuple[str, str]] = []
    for path in _tracked_files():
        if path in ALLOWLIST:
            continue
        for pattern in TRACKED_PATH_DENYLIST:
            if fnmatch.fnmatch(path, pattern):
                violations.append((path, pattern))
                break
    assert not violations, (
        "Debris tracked by git (path, matched-denylist-pattern): "
        f"{violations} — remove with `git rm --cached` and fix .gitignore."
    )


def test_staging_rule_is_explicit_in_gitignore() -> None:
    """DEC-P7-1: `.staging/` must be ignored by name, not only via an ignored parent."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    rules = {line.strip() for line in gitignore if line.strip() and not line.startswith("#")}
    assert ".staging/" in rules, ".gitignore lost the explicit .staging/ rule"
