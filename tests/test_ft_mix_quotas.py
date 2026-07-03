"""Behaviour × language mix quotas for the scaffold generator (P4_SPEC §2.2 table)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.finetune.dataset import TrainingConversation
from sutradhar.finetune.scaffold import (
    BEHAVIOUR_SHARES,
    CODE_MIXED_LANGS,
    ScaffoldConfig,
    generate,
    mix_stats,
)
from sutradhar.finetune.snapshot import load_scaffold_snapshot

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIZE = 400


@pytest.fixture(scope="module")
def conversations() -> list[TrainingConversation]:
    snapshot = load_scaffold_snapshot(_REPO_ROOT / "finetune" / "scaffold_snapshot.json")
    return generate(snapshot, ScaffoldConfig(seed=11, size=_SIZE))


def test_behaviour_quotas_exact(conversations: list[TrainingConversation]) -> None:
    """Largest-remainder allocation: counts match the spec shares to within 1."""
    stats = mix_stats(conversations)
    for behaviour, share in BEHAVIOUR_SHARES.items():
        count = sum(stats.get(behaviour, {}).values())
        assert abs(count - share * _SIZE) <= 1, f"{behaviour}: {count} vs {share * _SIZE}"
    assert sum(sum(v.values()) for v in stats.values()) == _SIZE


def test_language_mix_thresholds(conversations: list[TrainingConversation]) -> None:
    """Spec: >=40% code-mixed romanized, >=10% native script, remainder en."""
    langs = [c.query_lang for c in conversations]
    code_mixed = sum(1 for lang in langs if lang in CODE_MIXED_LANGS)
    native = sum(1 for lang in langs if lang == "native")
    en = sum(1 for lang in langs if lang == "en")
    assert code_mixed / _SIZE >= 0.40
    assert native / _SIZE >= 0.10
    assert code_mixed + native + en == _SIZE
    # Every code-mixed language is actually represented.
    for lang in CODE_MIXED_LANGS:
        assert lang in set(langs), f"{lang} missing from the mix"


def test_no_match_share_present(conversations: list[TrainingConversation]) -> None:
    no_match = [c for c in conversations if c.behaviour == "out_of_catalog"]
    assert 0.10 <= len(no_match) / _SIZE <= 0.20  # spec: 15%
    # Mid-conversation NO_MATCH variants exist (abstention after a grounded turn).
    assert any(len(c.intent_labels) > 1 for c in no_match), "no mid-conversation NO_MATCH"


def test_multi_turn_refine_shapes(conversations: list[TrainingConversation]) -> None:
    refines = [c for c in conversations if c.behaviour == "refine"]
    assert refines
    for conv in refines:
        assert conv.intent_labels[0] in ("list_versions", "find_by_plot", "find_by_title")
        assert all(label == "refine" for label in conv.intent_labels[1:])
        assert 2 <= len(conv.intent_labels) <= 4  # base + 1..3 refine turns

    # Refine-to-empty (honest empty set) appears somewhere in the mix.
    def has_empty_refine(conv: TrainingConversation) -> bool:
        return any(
            t.tool_result == {"versions": []} for t in conv.turns if t.tool_result is not None
        )

    assert any(has_empty_refine(c) for c in refines), "no refine-to-empty variant generated"


def test_disambiguate_asks_one_question(conversations: list[TrainingConversation]) -> None:
    for conv in (c for c in conversations if c.behaviour == "disambiguate"):
        ask = next(
            t.content
            for t in conv.turns
            if t.role == "assistant" and t.content and t.tool_calls is None
        )
        assert ask.count("?") == 1, f"{conv.conv_id}: ask-back must ask exactly one question"
        assert conv.intent_labels[0] == "disambiguate"
