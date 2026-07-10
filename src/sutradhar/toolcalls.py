"""Tool-call schema plumbing: outbound generation + inbound validation (v0 artifact).

Promoted from ``sutradhar.evals.driver`` in P5 task 1 (P5_SPEC §2.1) so the eval driver and
the serving orchestrator share ONE home; ``sutradhar.evals.driver`` re-exports these names
for backward compatibility. Pure move — bodies are byte-identical to the P3 originals and
all DEC-P1-8 conformance tests pass unchanged.

Design rules (unchanged from P3):

- **Outbound tools are generated, never hand-written** — the ``tools`` array is derived from
  the frozen ``tool_schema.v0.json`` artifact, so drift is impossible (P3_SPEC §2.8).
- **Inbound calls are validated before execution** with the same jsonschema pattern as the
  DEC-P1-8 conformance test (params subschema + root ``$defs``), applied to model-emitted
  calls: hallucinated tool names, hallucinated parameters (``additionalProperties: false``)
  and wrong-typed arguments are all caught and *scored*, not raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

TOOL_SCHEMA_PATH = Path("docs/phases/tool_schema.v0.json")


def load_tool_schema(path: Path = TOOL_SCHEMA_PATH) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def params_subschema(schema: dict[str, Any], tool: str) -> dict[str, Any]:
    """The tool's params schema with the root $defs attached (same shape the DEC-P1-8
    conformance test validates golden expected_tool_calls against)."""
    sub = dict(schema["tools"][tool]["params"])
    sub["$defs"] = schema["$defs"]
    return sub


def openai_tools(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The outbound ``tools`` array, GENERATED from the frozen artifact (P3_SPEC §2.8)."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": params_subschema(schema, name),
            },
        }
        for name, tool in schema["tools"].items()
    ]


def validate_emitted_call(
    schema: dict[str, Any],
    tool: str,
    arguments: dict[str, Any] | None,
) -> list[str]:
    """Validation errors for one model-emitted call (empty list = valid)."""
    if tool not in schema["tools"]:
        return [f"hallucinated tool: {tool!r} is not in TOOL_SCHEMA v0"]
    if arguments is None:
        return ["malformed tool-call arguments: not a JSON object"]
    validator = Draft202012Validator(params_subschema(schema, tool))
    return [e.message for e in validator.iter_errors(arguments)]
