"""Raw-response snapshots (P1_SPEC §2.3): versioned, git-ignored, sha256-recorded.

Every connector persists what it fetched *before* touching the DB, so graph builds are
reproducible without re-hitting APIs (CI parses committed trimmed captures; rebuilds replay
snapshots with ``--offline``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_snapshot(base_dir: Path, name: str, payload: dict[str, Any]) -> str:
    """Persist a raw response as JSON; append its sha256 to the snapshot manifest."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{name}.json"
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1)
    path.write_text(blob, encoding="utf-8")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    manifest = base_dir / "MANIFEST.sha256"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(f"{digest}  {path.name}\n")
    return digest


def load_snapshot(base_dir: Path, name: str) -> dict[str, Any]:
    """Load a snapshot, verifying its recorded hash (tamper/corruption check)."""
    path = base_dir / f"{name}.json"
    blob = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    manifest = (base_dir / "MANIFEST.sha256").read_text(encoding="utf-8")
    if f"{digest}  {path.name}" not in manifest:
        raise ValueError(f"snapshot {path} does not match its recorded sha256")
    result: dict[str, Any] = json.loads(blob)
    return result


def latest_snapshot_dir(root: Path) -> Path:
    """Most recent timestamped snapshot directory under ``root`` (raises if none)."""
    candidates = sorted(d for d in root.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no snapshot under {root}; run a live fetch first")
    return candidates[-1]
