"""P5 task 1 — tool-call plumbing promoted to ``sutradhar.toolcalls`` (P5_SPEC §2.1).

The promotion is a PURE MOVE: ``sutradhar.evals.driver`` re-exports the same objects, so
all existing import sites (evals, finetune, tests) keep working unchanged, and the
generated ``tools`` array still derives from the frozen ``tool_schema.v0.json`` artifact.
"""

from __future__ import annotations

from sutradhar import toolcalls
from sutradhar.evals import driver

PROMOTED_NAMES = [
    "TOOL_SCHEMA_PATH",
    "load_tool_schema",
    "params_subschema",
    "openai_tools",
    "validate_emitted_call",
]

V0_TOOLS = ["resolve_title", "search_by_plot", "get_work", "get_versions", "refine_filter"]


def test_toolcalls_promotion() -> None:
    """Driver re-exports ARE the promoted objects — identity, not copies."""
    for name in PROMOTED_NAMES:
        assert getattr(driver, name) is getattr(toolcalls, name), (
            f"driver.{name} must re-export sutradhar.toolcalls.{name}"
        )


def test_openai_tools_from_new_home_match_v0_artifact() -> None:
    """The new import path still generates the five v0 tools, closed to extra params."""
    schema = toolcalls.load_tool_schema()
    tools = toolcalls.openai_tools(schema)
    assert [t["function"]["name"] for t in tools] == V0_TOOLS
    for tool in tools:
        params = tool["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert "$defs" in params  # root $defs attached (DEC-P1-8 subschema shape)


def test_validate_emitted_call_from_new_home() -> None:
    """Inbound validation behaves identically at the new import path."""
    schema = toolcalls.load_tool_schema()
    assert toolcalls.validate_emitted_call(schema, "resolve_title", {"title": "Papanasam"}) == []
    assert toolcalls.validate_emitted_call(schema, "made_up_tool", {}) == [
        "hallucinated tool: 'made_up_tool' is not in TOOL_SCHEMA v0"
    ]
    assert toolcalls.validate_emitted_call(schema, "resolve_title", None) == [
        "malformed tool-call arguments: not a JSON object"
    ]
    errors = toolcalls.validate_emitted_call(
        schema, "resolve_title", {"title": "Papanasam", "hallucinated_param": 1}
    )
    assert errors, "additionalProperties: false must reject hallucinated params"
