"""Frozen prompt artifacts: hash pinning + golden-set disjointness (P3 task 3, DEC-P3-4).

Enforces P3_SPEC §4: the frozen prompt/exemplar/taxonomy files are hash-pinned (any edit
fails until the lock is deliberately regenerated) and the exemplars stay disjoint from the
golden fixtures — no fixture query substring appears in the prompt bundle (no leakage into
the P4 before/after).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sutradhar.evals.golden import load_fixtures
from sutradhar.evals.prompts import (
    ARTIFACT_FILES,
    INTENT_PREAMBLE_PREFIX,
    PROMPTS_DIR,
    PromptArtifacts,
    compute_hashes,
    load_lock,
    load_prompt_artifacts,
    write_lock,
)

_REPO = Path(__file__).resolve().parents[1]
_PROMPTS = _REPO / PROMPTS_DIR
_GOLDEN = _REPO / "evals" / "golden"

_INTENT_LABELS = {
    "find_by_plot",
    "find_by_title",
    "list_versions",
    "refine",
    "disambiguate",
    "out_of_catalog",
}
_SLOT_KEYS = {"title", "plot_description", "actor", "language", "year", "era", "relationship"}


@pytest.fixture(scope="module")
def artifacts() -> PromptArtifacts:
    return load_prompt_artifacts(_PROMPTS)


def test_lock_matches_artifact_files() -> None:
    """The committed lock must equal a fresh recompute — the hash-pinning gate."""
    file_hashes, prompt_hash = compute_hashes(_PROMPTS)
    lock = load_lock(_PROMPTS)
    assert lock["files"] == file_hashes, (
        "frozen prompt artifact edited without re-pinning prompts.lock.json "
        "(deliberate change? run: uv run python -m sutradhar.evals.prompts --write-lock)"
    )
    assert lock["prompt_hash"] == prompt_hash
    assert set(lock["files"]) == set(ARTIFACT_FILES)


def test_any_edit_changes_prompt_hash_and_fails_load(tmp_path: Path) -> None:
    """A one-byte edit must flip prompt_hash and make load_prompt_artifacts refuse."""
    for name in ARTIFACT_FILES + ("prompts.lock.json",):
        (tmp_path / name).write_bytes((_PROMPTS / name).read_bytes())
    _, original_hash = compute_hashes(tmp_path)

    target = tmp_path / "system_v1.md"
    target.write_text(target.read_text(encoding="utf-8") + "x", encoding="utf-8")
    _, mutated_hash = compute_hashes(tmp_path)
    assert mutated_hash != original_hash
    with pytest.raises(ValueError, match="do not match"):
        load_prompt_artifacts(tmp_path)
    # Re-pinning (the deliberate act) makes it loadable again.
    write_lock(tmp_path)
    assert load_prompt_artifacts(tmp_path).prompt_hash == mutated_hash


def test_exemplars_disjoint_from_golden_fixtures(artifacts: PromptArtifacts) -> None:
    """No golden fixture query may appear in the prompt bundle (leakage guard, DEC-P3-4)."""
    bundle = artifacts.system_prompt().casefold()
    fixtures = load_fixtures(_GOLDEN)
    assert fixtures, "golden fixtures failed to load — disjointness check would be vacuous"
    for fixture in fixtures:
        queries = fixture.query if isinstance(fixture.query, list) else [fixture.query]
        for query in queries:
            assert query.strip().casefold() not in bundle, (
                f"golden fixture {fixture.id} query leaked into the prompt bundle: {query!r}"
            )


def test_taxonomy_frozen_labels_and_slots(artifacts: PromptArtifacts) -> None:
    """The six intent labels and seven slot keys are frozen (P3_SPEC §2.2)."""
    assert artifacts.intent_labels == _INTENT_LABELS
    assert artifacts.slot_keys == _SLOT_KEYS
    # Slot keys = refine_filter.by vocabulary + plot_description + title (nothing invented).
    schema = json.loads((_REPO / "docs/phases/tool_schema.v0.json").read_text(encoding="utf-8"))
    by_keys = set(schema["tools"]["refine_filter"]["params"]["properties"]["by"]["properties"])
    assert artifacts.slot_keys == by_keys | {"plot_description", "title"}


def test_system_prompt_carries_taxonomy_and_preamble_contract(
    artifacts: PromptArtifacts,
) -> None:
    """The system prompt must name every intent label and the preamble prefix verbatim."""
    for label in _INTENT_LABELS:
        assert f"`{label}`" in artifacts.system, f"intent label {label} missing from system prompt"
    assert INTENT_PREAMBLE_PREFIX.strip() + ":" in artifacts.system or (
        "INTENT:" in artifacts.system
    )
    combined = artifacts.system_prompt()
    assert combined.startswith(artifacts.system.rstrip()[:50])
    assert artifacts.exemplars.rstrip()[-50:] in combined


def test_exemplar_preambles_parse_against_taxonomy(artifacts: PromptArtifacts) -> None:
    """Every INTENT line in the exemplars must parse and use frozen labels/slot keys only."""
    preambles = [
        line.split(INTENT_PREAMBLE_PREFIX, 1)[1]
        for line in artifacts.exemplars.splitlines()
        if INTENT_PREAMBLE_PREFIX in line
    ]
    # DEC-P3-4: one code-mixed, one multi-turn refine (2 turns), one NO_MATCH => 4 preambles.
    assert len(preambles) == 4
    intents = []
    for raw in preambles:
        payload = json.loads(raw)
        assert set(payload) <= {"intent", "slots"}
        assert payload["intent"] in _INTENT_LABELS
        assert set(payload.get("slots", {})) <= _SLOT_KEYS
        intents.append(payload["intent"])
    # The three mandated behaviours are each demonstrated.
    assert "find_by_plot" in intents  # code-mixed plot query
    assert "refine" in intents  # multi-turn refinement
    assert "out_of_catalog" in intents  # NO_MATCH abstention


def test_exemplar_tool_calls_use_frozen_tool_names_only(artifacts: PromptArtifacts) -> None:
    """Exemplar tool traffic must not teach a tool outside TOOL_SCHEMA v0."""
    schema = json.loads((_REPO / "docs/phases/tool_schema.v0.json").read_text(encoding="utf-8"))
    frozen = set(schema["tools"])
    for line in artifacts.exemplars.splitlines():
        if line.startswith("[tool call] "):
            name = line.removeprefix("[tool call] ").split(" ", 1)[0]
            assert name in frozen, f"exemplar teaches unknown tool {name!r}"
