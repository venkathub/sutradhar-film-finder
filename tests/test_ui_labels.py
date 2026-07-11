"""P6 task 2 — the UI tool-label map is BYTE-DERIVED from the frozen v0 artifact.

``test_ui_tool_labels_generated`` is the drift gate named in P6_SPEC §4: the committed
``ui/app/src/generated/tool_labels.json`` must equal a fresh regeneration from
``docs/phases/tool_schema.v0.json`` — edit either side without the other and CI fails.
No hand-written tool or parameter name can reach the rendered trace view.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "ui" / "app" / "scripts" / "gen_tool_labels.py"
COMMITTED = REPO_ROOT / "ui" / "app" / "src" / "generated" / "tool_labels.json"
SCHEMA = REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json"

V0_TOOLS = ["get_versions", "get_work", "refine_filter", "resolve_title", "search_by_plot"]


def _regenerate() -> bytes:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--stdout"],
        check=True,
        capture_output=True,
    ).stdout


def test_ui_tool_labels_generated() -> None:
    """Byte-for-byte: committed map == fresh regeneration (the CI drift gate)."""
    assert COMMITTED.exists(), "run `make ui-gen` to create the committed label map"
    assert _regenerate() == COMMITTED.read_bytes(), (
        "ui/app/src/generated/tool_labels.json drifted from tool_schema.v0.json — "
        "run `make ui-gen` and commit the result"
    )


def test_label_map_carries_exactly_the_v0_tools() -> None:
    document = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert sorted(document["tools"]) == V0_TOOLS  # no extra, no missing
    assert document["schema_version"] == "v0"
    # Provenance stamp: the sha256 of the artifact the map was derived from.
    assert document["schema_sha256"] == hashlib.sha256(SCHEMA.read_bytes()).hexdigest()


def test_labels_and_params_are_derived_from_the_artifact() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    document = json.loads(COMMITTED.read_text(encoding="utf-8"))
    for name, tool in document["tools"].items():
        # Deterministic label rule — never free text.
        derived = name.replace("_", " ")
        assert tool["label"] == derived[:1].upper() + derived[1:]
        assert tool["description"] == schema["tools"][name]["description"]
        params = schema["tools"][name]["params"]
        assert sorted(tool["params"]) == sorted(params.get("properties", {}))
        required = set(params.get("required", []))
        for pname, param in tool["params"].items():
            assert param["required"] == (pname in required)
