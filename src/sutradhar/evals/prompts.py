"""Frozen prompt artifacts: loading + hash pinning (P3 task 3, DEC-P3-4).

The base-model prompting strategy is FROZEN as three in-repo artifacts under
``evals/prompts/`` — system prompt, few-shot exemplars, and the intent taxonomy — combined
into one ``prompt_hash`` that every generation-run artifact and Table 2 stamp records
(P3_SPEC §2.2/§6.3). The P4 QLoRA before/after is only fair under an identical hash.

Pinning mechanics: ``prompts.lock.json`` records the per-file SHA-256 digests and the combined
``prompt_hash``. ``tests/test_prompt_artifacts.py`` recomputes both — any edit to a frozen file
fails CI until the lock is deliberately regenerated::

    uv run python -m sutradhar.evals.prompts --write-lock

Paths are repo-root-relative (same convention as ``sutradhar.evals.golden.GOLDEN_DIR``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path("evals/prompts")
LOCK_FILE = "prompts.lock.json"

# The frozen artifact set, in hash order. Adding/renaming a file is a prompt-version bump.
ARTIFACT_FILES = ("system_v1.md", "exemplars_v1.md", "intent_taxonomy_v1.json")

# P5 serving bundle (v1.1, DEC-P5-3/Q2): the frozen v1 files + the spotlighting appendix,
# recorded under a SEPARATE lock. prompts.lock.json (v1.0, hash 78215ccc…) stays untouched
# so the pinned Table 2 columns and their comparability gate keep holding; the serving
# stamp records the v1.1 hash from this lock instead.
SERVING_APPENDIX_FILE = "spotlighting_appendix_v1_1.md"
SERVING_ARTIFACT_FILES = (*ARTIFACT_FILES, SERVING_APPENDIX_FILE)
SERVING_LOCK_FILE = "prompts.serving.lock.json"

INTENT_PREAMBLE_PREFIX = "INTENT: "


@dataclass(frozen=True)
class PromptArtifacts:
    """The frozen prompt bundle + its pinned hashes (stamped into every generation run)."""

    system: str
    exemplars: str
    taxonomy: dict[str, Any]
    file_hashes: dict[str, str]
    prompt_hash: str
    # P5 serving bundle only (v1.1): the spotlighting appendix; None for the v1.0 bundle.
    appendix: str | None = None

    @property
    def intent_labels(self) -> frozenset[str]:
        return frozenset(self.taxonomy["intents"])

    @property
    def slot_keys(self) -> frozenset[str]:
        return frozenset(self.taxonomy["slot_keys"])

    def system_prompt(self) -> str:
        """The full frozen system message: system prompt + exemplars (+ v1.1 appendix on
        the serving bundle), one hashed unit.

        DEC-P3-4 option B: exemplars ride in the system message (native function-calling
        format stays available for the real tool traffic).
        """
        base = f"{self.system.rstrip()}\n\n{self.exemplars.rstrip()}\n"
        if self.appendix is None:
            return base
        return f"{base}\n{self.appendix.rstrip()}\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_hashes(
    directory: Path = PROMPTS_DIR,
    files: tuple[str, ...] = ARTIFACT_FILES,
) -> tuple[dict[str, str], str]:
    """Per-file SHA-256 digests + the combined prompt_hash (order- and name-sensitive)."""
    file_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in files:
        payload = (directory / name).read_bytes()
        file_hashes[name] = _sha256(payload)
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")
    return file_hashes, combined.hexdigest()


def load_lock(directory: Path = PROMPTS_DIR) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((directory / LOCK_FILE).read_text(encoding="utf-8"))
    return payload


def write_lock(directory: Path = PROMPTS_DIR) -> dict[str, Any]:
    """(Re)generate prompts.lock.json — a deliberate act; CI pins against the result."""
    file_hashes, prompt_hash = compute_hashes(directory)
    lock = {
        "$comment": (
            "Pinned hashes of the frozen prompt artifacts (DEC-P3-4). Regenerate ONLY on a "
            "deliberate prompt change: python -m sutradhar.evals.prompts --write-lock. "
            "Table 2 columns are comparable only under an identical prompt_hash."
        ),
        "files": file_hashes,
        "prompt_hash": prompt_hash,
    }
    (directory / LOCK_FILE).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def load_prompt_artifacts(directory: Path = PROMPTS_DIR) -> PromptArtifacts:
    """Load the frozen bundle, verifying the artifacts still match the committed lock."""
    file_hashes, prompt_hash = compute_hashes(directory)
    lock = load_lock(directory)
    if lock.get("files") != file_hashes or lock.get("prompt_hash") != prompt_hash:
        raise ValueError(
            f"Frozen prompt artifacts do not match {LOCK_FILE} — either revert the edit or "
            "deliberately re-pin: uv run python -m sutradhar.evals.prompts --write-lock"
        )
    taxonomy: dict[str, Any] = json.loads(
        (directory / "intent_taxonomy_v1.json").read_text(encoding="utf-8")
    )
    return PromptArtifacts(
        system=(directory / "system_v1.md").read_text(encoding="utf-8"),
        exemplars=(directory / "exemplars_v1.md").read_text(encoding="utf-8"),
        taxonomy=taxonomy,
        file_hashes=file_hashes,
        prompt_hash=prompt_hash,
    )


def write_serving_lock(directory: Path = PROMPTS_DIR) -> dict[str, Any]:
    """(Re)generate the v1.1 SERVING lock (P5, DEC-P5-3/Q2) — prompts.lock.json untouched."""
    file_hashes, prompt_hash = compute_hashes(directory, files=SERVING_ARTIFACT_FILES)
    lock = {
        "$comment": (
            "P5 serving bundle (v1.1, DEC-P5-3): the frozen v1 artifacts + the spotlighting "
            "appendix, under their own prompt_hash. The v1.0 prompts.lock.json stays pinned "
            "(Table 2 comparability); serving-run stamps record THIS hash. Regenerate: "
            "python -m sutradhar.evals.prompts --write-serving-lock."
        ),
        "files": file_hashes,
        "prompt_hash": prompt_hash,
    }
    (directory / SERVING_LOCK_FILE).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def load_serving_prompt_artifacts(directory: Path = PROMPTS_DIR) -> PromptArtifacts:
    """The v1.1 serving bundle (v1.0 files + spotlighting appendix), lock-verified.

    Verifies BOTH locks: the v1.0 sub-bundle must still match ``prompts.lock.json``
    (the appendix extends the frozen bundle, it never edits it) and the composed
    bundle must match ``prompts.serving.lock.json``.
    """
    base = load_prompt_artifacts(directory)  # v1.0 verification included
    file_hashes, prompt_hash = compute_hashes(directory, files=SERVING_ARTIFACT_FILES)
    lock = json.loads((directory / SERVING_LOCK_FILE).read_text(encoding="utf-8"))
    if lock.get("files") != file_hashes or lock.get("prompt_hash") != prompt_hash:
        raise ValueError(
            f"Serving prompt artifacts do not match {SERVING_LOCK_FILE} — either revert the "
            "edit or deliberately re-pin: uv run python -m sutradhar.evals.prompts "
            "--write-serving-lock"
        )
    return PromptArtifacts(
        system=base.system,
        exemplars=base.exemplars,
        taxonomy=base.taxonomy,
        file_hashes=file_hashes,
        prompt_hash=prompt_hash,
        appendix=(directory / SERVING_APPENDIX_FILE).read_text(encoding="utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect / re-pin the frozen prompt artifacts.")
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="(Re)generate prompts.lock.json from the current artifact files.",
    )
    parser.add_argument(
        "--write-serving-lock",
        action="store_true",
        help="(Re)generate the v1.1 serving lock (prompts.serving.lock.json) only.",
    )
    args = parser.parse_args()
    if args.write_lock:
        lock = write_lock()
        print(f"wrote {PROMPTS_DIR / LOCK_FILE}")
    elif args.write_serving_lock:
        lock = write_serving_lock()
        print(f"wrote {PROMPTS_DIR / SERVING_LOCK_FILE}")
    else:
        files, prompt_hash = compute_hashes()
        lock = {"files": files, "prompt_hash": prompt_hash}
    for name, digest in lock["files"].items():
        print(f"  {digest}  {name}")
    print(f"prompt_hash: {lock['prompt_hash']}")


if __name__ == "__main__":
    main()
