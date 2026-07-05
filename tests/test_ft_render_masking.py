"""Render + masking tests (P4 task 8; spec §4 ``test_ft_render_masking``).

Runs the REAL ``transformers.apply_chat_template`` assistant-masks code path (the one TRL
``assistant_only_loss`` uses) against the COMMITTED fixture tokenizer — no downloads, no
torch, no model weights. Assertions are on rendered token/mask ARRAYS, guarding the known
TRL liger silent-mask-drop failure by shape (all-ones mask must be flagged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.evals.driver import load_tool_schema, openai_tools
from sutradhar.finetune.dataset import TrainingConversation, read_jsonl
from sutradhar.finetune.render import (
    RenderedSample,
    render_stats,
    render_with_masks,
    to_trl_messages,
    to_trl_rows,
    verify_masking,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_TOKENIZER = _REPO_ROOT / "tests" / "fixtures" / "tokenizer"
_SAMPLE = _REPO_ROOT / "finetune" / "ft_v1_sample.jsonl"


@pytest.fixture(scope="module")
def tokenizer() -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_FIXTURE_TOKENIZER))


@pytest.fixture(scope="module")
def conversations() -> list[TrainingConversation]:
    return read_jsonl(_SAMPLE)


@pytest.fixture(scope="module")
def rendered(tokenizer: object, conversations: list[TrainingConversation]) -> list[RenderedSample]:
    return [
        render_with_masks(tokenizer, to_trl_messages(conv), conv_id=conv.conv_id)
        for conv in conversations
    ]


def test_trl_rows_shape(conversations: list[TrainingConversation]) -> None:
    schema = load_tool_schema(_REPO_ROOT / "docs" / "phases" / "tool_schema.v0.json")
    tools = openai_tools(schema)
    rows = to_trl_rows(conversations, tools)
    assert len(rows) == len(conversations)
    row = rows[0]
    assert set(row) == {"conv_id", "messages", "tools"}
    assert row["tools"] is tools  # generated from the frozen artifact, never hand-written
    names = {t["function"]["name"] for t in row["tools"]}
    assert names == {"resolve_title", "search_by_plot", "get_work", "get_versions", "refine_filter"}
    roles = [m["role"] for m in row["messages"]]
    assert roles[0] == "user"
    # Tool-calling assistant messages carry function dicts, not prose.
    for m in row["messages"]:
        if m["role"] == "assistant" and "tool_calls" in m:
            assert "content" not in m
            assert m["tool_calls"][0]["type"] == "function"


def test_masks_partition_every_sample_correctly(
    conversations: list[TrainingConversation], rendered: list[RenderedSample]
) -> None:
    """THE task-8 gate: assistant-only masking holds on token arrays for all 96 samples."""
    all_violations = []
    for conv, sample in zip(conversations, rendered, strict=True):
        all_violations.extend(verify_masking(conv, sample))
    assert all_violations == [], all_violations[:5]


def test_mask_is_never_degenerate(rendered: list[RenderedSample]) -> None:
    for sample in rendered:
        assert any(sample.assistant_masks), f"{sample.conv_id}: all-zero mask"
        assert not all(sample.assistant_masks), f"{sample.conv_id}: all-one mask (liger shape)"
        assert len(sample.assistant_masks) == len(sample.input_ids)


def test_all_ones_mask_is_flagged_as_liger_shape(
    conversations: list[TrainingConversation], rendered: list[RenderedSample]
) -> None:
    """Negative control: simulate the TRL liger silent-mask-drop and require detection."""
    sample = rendered[0].model_copy(update={"assistant_masks": [1] * len(rendered[0].input_ids)})
    violations = verify_masking(conversations[0], sample)
    assert any("liger" in v.detail for v in violations)


def test_all_zeros_mask_is_flagged(
    conversations: list[TrainingConversation], rendered: list[RenderedSample]
) -> None:
    sample = rendered[0].model_copy(update={"assistant_masks": [0] * len(rendered[0].input_ids)})
    violations = verify_masking(conversations[0], sample)
    assert any("all-zeros" in v.detail for v in violations)


def test_contracts_survive_rendering(rendered: list[RenderedSample]) -> None:
    with_preamble = [s for s in rendered if "INTENT: " in s.masked_text]
    assert len(with_preamble) == len(rendered)  # every sample has final answers
    assert any("**" in s.masked_text for s in rendered)  # bold titles present + trainable


def test_render_stats_shape(rendered: list[RenderedSample]) -> None:
    stats = render_stats(rendered, max_seq=4096)
    assert stats.samples == len(rendered)
    assert 0 < stats.token_p50 <= stats.token_p95 <= stats.token_max
    # The fixture tokenizer is tiny (2k vocab) so counts run long; the REAL base tokenizer
    # is checked in the ft-dryrun. Here we only assert the accounting works.
    assert stats.over_max_seq >= 0
