"""Tool-schema conformance tests (P1 task 15, spec §4 — required by the working agreement).

1. ``test_tool_schema_json_valid``: the frozen artifact is valid JSON Schema and stays in
   sync with the prose contract (doc drift fails CI).
2. ``test_golden_expected_tool_calls_validate``: no hallucinated tool or parameter names can
   be committed into the golden set (P3/P4 reuse this validator against model-emitted calls).
3. Signature-level repository conformance (result-shape round-trip is the integration half,
   ``tests/integration/test_repository_schema.py``).
"""

from __future__ import annotations

import inspect
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

from sutradhar.evals.golden import load_fixtures
from sutradhar.graph import repository

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json"
MD_PATH = REPO_ROOT / "docs" / "phases" / "TOOL_SCHEMA.md"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"

TOOLS = ("resolve_title", "search_by_plot", "get_work", "get_versions", "refine_filter")


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return data


def _tool_subschema(schema: dict[str, Any], tool: str, part: str) -> dict[str, Any]:
    sub = dict(schema["tools"][tool][part])
    sub["$defs"] = schema["$defs"]  # make local refs resolvable standalone
    return sub


# --- 1. Artifact validity + md<->json sync ---


def test_tool_schema_json_valid(schema: dict[str, Any]) -> None:
    assert schema["version"] == "v0" and schema["status"] == "FROZEN"
    assert set(schema["tools"]) == set(TOOLS)
    for tool in TOOLS:
        for part in ("params", "result"):
            sub = _tool_subschema(schema, tool, part)
            cls = validator_for(sub, default=Draft202012Validator)
            cls.check_schema(sub)  # raises if not valid JSON Schema


def test_md_and_json_stay_in_sync(schema: dict[str, Any]) -> None:
    """Doc drift fails CI: every tool + param in the artifact appears in the prose contract."""
    md = MD_PATH.read_text(encoding="utf-8")
    md_tools = set(re.findall(r"^### `(\w+)`", md, re.MULTILINE))
    assert md_tools == set(schema["tools"]), "tool inventory drifted between .md and .json"
    assert "FROZEN v0" in md
    for tool in TOOLS:
        section = md.split(f"### `{tool}`", 1)[1].split("###", 1)[0]
        for param in schema["tools"][tool]["params"]["properties"]:
            assert param in section, f"{tool}: param {param!r} missing from the .md block"


def test_enums_match_graph_reality(schema: dict[str, Any]) -> None:
    defs = schema["$defs"]
    assert set(defs["scope"]["enum"]) == {"indian", "all", "foreign"}
    assert set(defs["era"]["enum"]) == {"original", "newer", "older"}
    # Relationship enum = the 4 stored version-facing labels + derived is_original_of;
    # based_on is a work-level lineage fact exposed via get_work, not a version label.
    assert set(defs["relationship"]["enum"]) == {
        "is_original_of",
        "is_remake_of",
        "is_official_dub_of",
        "is_unofficial_remake_of",
        "is_sequel_of",
    }
    assert set(defs["sources"]["items"]["properties"]["source"]["enum"]) == {
        "wikidata",
        "tmdb",
        "imdb",
        "wikipedia",
        "human",
        "rule",
    }


# --- 2. Golden expected_tool_calls validate (no hallucinated tools/params committable) ---


def test_golden_expected_tool_calls_validate(schema: dict[str, Any]) -> None:
    fixtures = load_fixtures(GOLDEN_DIR)
    checked = 0
    for fixture in fixtures:
        for call in fixture.expected_tool_calls or []:
            assert call.tool in schema["tools"], f"{fixture.id}: hallucinated tool {call.tool!r}"
            sub = _tool_subschema(schema, call.tool, "params")
            validator = Draft202012Validator(sub)
            errors = [
                e.message
                for e in validator.iter_errors(call.arguments)
                # runtime placeholders ($work_id, [$version_set]) are strings — type-valid
            ]
            assert errors == [], f"{fixture.id}/{call.tool}: {errors}"
            checked += 1
    # P3 task 4 expansion: GS-07 x5 (2+2+2+2+2=10) + GS-08 x3 (5+5+5=15) + GS-02d-g (4) = 29.
    assert checked >= 29


def test_unknown_param_would_fail(schema: dict[str, Any]) -> None:
    """The validator has teeth: an invented parameter name is rejected."""
    sub = _tool_subschema(schema, "resolve_title", "params")
    validator = Draft202012Validator(sub)
    assert list(validator.iter_errors({"title": "Drishyam", "actor": "x"}))
    assert not list(validator.iter_errors({"title": "Drishyam"}))


# --- 3. Repository signature conformance (the "contract is satisfiable" proof, static half) ---


@pytest.mark.parametrize(
    ("fn_name", "tool", "infra_params"),
    [
        ("resolve_title", "resolve_title", set()),
        ("search_by_plot", "search_by_plot", {"retriever"}),  # injected infra (P2_SPEC §2.5)
        ("get_work", "get_work", set()),
        ("get_versions", "get_versions", set()),
        ("refine_filter", "refine_filter", set()),
    ],
)
def test_repository_matches_tool_schema(
    schema: dict[str, Any], fn_name: str, tool: str, infra_params: set[str]
) -> None:
    fn = getattr(repository, fn_name)
    fn_params = set(inspect.signature(fn).parameters) - {"session"} - infra_params
    schema_params = set(schema["tools"][tool]["params"]["properties"])
    assert fn_params == schema_params, f"{fn_name}: signature drifted from the frozen schema"
    # Required params carry no default in the function; optional ones do.
    required = set(schema["tools"][tool]["params"].get("required", []))
    sig = inspect.signature(fn)
    for name in fn_params:
        has_default = sig.parameters[name].default is not inspect.Parameter.empty
        assert has_default == (name not in required), (
            f"{fn_name}.{name}: required/optional drift vs schema"
        )
    # Infra params are keyword-only — they can never be mistaken for tool arguments.
    for name in infra_params:
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_search_by_plot_matches_tool_schema(schema: dict[str, Any]) -> None:
    """P2 flip of ``test_search_by_plot_not_implemented_yet``: the result MODEL round-trips
    against the frozen v0 result schema (live-call round-trip is the integration half)."""
    assert hasattr(repository, "search_by_plot")  # the P1 absence assertion, inverted
    sample = repository.SearchByPlotResult(
        results=[
            repository.PlotSearchHit(
                work_id=uuid.uuid4(),
                canonical_title="Drishyam",
                language="ml",
                year=2013,
                score=0.97,
            ),
            repository.PlotSearchHit(
                work_id=uuid.uuid4(),
                canonical_title="Unknown Year Work",
                language=None,
                year=None,
                score=0.11,
            ),
        ],
        abstain=False,
    )
    sub = _tool_subschema(schema, "search_by_plot", "result")
    payload = json.loads(sample.model_dump_json())
    errors = [e.message for e in Draft202012Validator(sub).iter_errors(payload)]
    assert errors == [], errors
    # And the abstain=true shape (v0 allows results alongside abstain).
    empty = repository.SearchByPlotResult(results=[], abstain=True)
    errors = [
        e.message
        for e in Draft202012Validator(sub).iter_errors(json.loads(empty.model_dump_json()))
    ]
    assert errors == []
