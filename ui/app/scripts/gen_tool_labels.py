#!/usr/bin/env python3
"""Generate the UI tool-label map from the frozen v0 tool-schema artifact (P6 task 2).

The trace view's tool/param display labels are BYTE-DERIVED from
``docs/phases/tool_schema.v0.json`` — never hand-written (the DEC-P1-8
generated-tools-array posture, extended to the UI per P6_SPEC §2.8). The committed
output (``ui/app/src/generated/tool_labels.json``) is drift-gated twice:

- pytest Tier-1: ``tests/test_ui_labels.py::test_ui_tool_labels_generated``
  (regenerate → byte-compare);
- CI ui job: regenerate → ``git diff --exit-code``.

stdlib only (runs under any Python ≥3.10 — CI does not need the project venv).

Usage:
    python3 ui/app/scripts/gen_tool_labels.py            # (re)write the committed map
    python3 ui/app/scripts/gen_tool_labels.py --stdout   # emit to stdout (drift tests)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json"
OUTPUT_PATH = REPO_ROOT / "ui" / "app" / "src" / "generated" / "tool_labels.json"


def _label(name: str) -> str:
    """Deterministic display label: underscores → spaces, first letter capitalized."""
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def build_label_map(schema: dict[str, Any], schema_bytes: bytes) -> dict[str, Any]:
    """The label-map document — every field derived from the artifact, nothing invented."""
    tools: dict[str, Any] = {}
    for tool_name in sorted(schema["tools"]):
        tool = schema["tools"][tool_name]
        params: dict[str, Any] = {}
        properties = tool["params"].get("properties", {})
        required = set(tool["params"].get("required", []))
        for param_name in sorted(properties):
            params[param_name] = {
                "label": _label(param_name),
                "required": param_name in required,
            }
        tools[tool_name] = {
            "label": _label(tool_name),
            "description": tool["description"],
            "params": params,
        }
    return {
        "$comment": (
            "GENERATED from docs/phases/tool_schema.v0.json — do not edit by hand. "
            "Regenerate with `make ui-gen` (ui/app/scripts/gen_tool_labels.py)."
        ),
        "schema_version": schema["version"],
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "tools": tools,
    }


def render() -> bytes:
    schema_bytes = SCHEMA_PATH.read_bytes()
    schema = json.loads(schema_bytes)
    document = build_label_map(schema, schema_bytes)
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main(argv: list[str]) -> int:
    payload = render()
    if "--stdout" in argv:
        sys.stdout.buffer.write(payload)
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(payload)
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
