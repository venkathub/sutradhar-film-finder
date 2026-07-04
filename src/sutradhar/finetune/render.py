"""Chat-template rendering + assistant-only masking (P4 task 8; DEC-P4-5).

Transforms sealed :class:`TrainingConversation` rows into the TRL-native SFT shape
(``{"messages": […], "tools": […]}``, tool calls as OpenAI-style function dicts, tools
GENERATED from frozen ``tool_schema.v0.json`` — never hand-written), and verifies the
property the whole fine-tune stands on: **labels are masked everywhere except assistant
tokens**, asserted on rendered token/mask arrays, never assumed from config.

Why array-level assertion: TRL's ``assistant_only_loss=True`` derives masks from the chat
template's ``{% generation %}`` markers, and a known TRL bug silently DISCARDS those masks
under ``use_liger_kernel=True`` (loss over the whole sequence, no error — P4_SPEC §3 D5).
Liger is pinned OFF in the task-9 ``TrainConfig``; this module is the other half of the
guard — it proves the rendered masks partition the sequence correctly before any GPU
minute is spent.

Compute placement: tokenizer-only (``transformers`` runs torch-free on the laptop — no
model weights, no downloads in CI; unit tests use the committed fixture tokenizer under
``tests/fixtures/tokenizer/``; the ft-dryrun uses the real base-model tokenizer config).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from sutradhar.finetune.dataset import TrainingConversation


class MaskingViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conv_id: str
    detail: str


class RenderedSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conv_id: str
    input_ids: list[int]
    assistant_masks: list[int]
    masked_text: str  # decode of the masked (trainable) tokens
    unmasked_text: str  # decode of the context (loss-free) tokens


class RenderStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: int
    token_p50: int
    token_p95: int
    token_max: int
    over_max_seq: int  # samples exceeding the task-9 max_seq (truncation candidates)


# --- TrainingConversation -> TRL rows ---


def to_trl_messages(conv: TrainingConversation) -> list[dict[str, Any]]:
    """OpenAI-style messages: tool calls as function dicts, tool results as JSON content."""
    messages: list[dict[str, Any]] = []
    for turn in conv.turns:
        if turn.role == "assistant" and turn.tool_calls is not None:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": call.tool, "arguments": call.arguments},
                        }
                        for call in turn.tool_calls
                    ],
                }
            )
        elif turn.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(turn.tool_result, ensure_ascii=False, sort_keys=True),
                }
            )
        else:
            messages.append({"role": turn.role, "content": turn.content or ""})
    return messages


def to_trl_rows(
    conversations: list[TrainingConversation], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The TRL SFT dataset shape (trl>=0.19 native ``tools`` support, D5)."""
    return [
        {"conv_id": conv.conv_id, "messages": to_trl_messages(conv), "tools": tools}
        for conv in conversations
    ]


# --- Rendering + mask verification ---


def render_with_masks(
    tokenizer: Any,  # PreTrainedTokenizerBase — kept loose so laptop typing stays torch-free
    messages: list[dict[str, Any]],
    conv_id: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> RenderedSample:
    """apply_chat_template with assistant-token masks; decoded partitions for asserting."""
    kwargs: dict[str, Any] = {"return_dict": True, "return_assistant_tokens_mask": True}
    if tools is not None:
        kwargs["tools"] = tools
    encoded: Any = tokenizer.apply_chat_template(messages, **kwargs)
    ids: list[int] = [int(i) for i in encoded["input_ids"]]
    masks: list[int] = [int(m) for m in encoded["assistant_masks"]]
    masked = str(tokenizer.decode([i for i, m in zip(ids, masks, strict=True) if m]))
    unmasked = str(tokenizer.decode([i for i, m in zip(ids, masks, strict=True) if not m]))
    return RenderedSample(
        conv_id=conv_id,
        input_ids=ids,
        assistant_masks=masks,
        masked_text=masked,
        unmasked_text=unmasked,
    )


def verify_masking(conv: TrainingConversation, sample: RenderedSample) -> list[MaskingViolation]:
    """The array-level guard: trainable tokens = assistant turns, nothing else.

    - masks exist and are neither all-zero nor all-one (all-one == the liger failure
      shape: loss over the whole sequence);
    - every assistant answer/tool-name is inside the MASKED partition;
    - every user utterance and tool-result payload is OUTSIDE it;
    - the INTENT preamble + bold-title formatting survive rendering.
    """
    violations: list[MaskingViolation] = []

    def flag(detail: str) -> None:
        violations.append(MaskingViolation(conv_id=conv.conv_id, detail=detail))

    if not sample.assistant_masks or len(sample.assistant_masks) != len(sample.input_ids):
        flag("assistant_masks missing or misaligned")
        return violations
    if all(sample.assistant_masks):
        flag("mask is all-ones — loss over the whole sequence (liger-bug shape)")
    if not any(sample.assistant_masks):
        flag("mask is all-zeros — nothing trainable")

    for turn in conv.turns:
        if turn.role == "assistant" and turn.content and turn.tool_calls is None:
            if turn.content not in sample.masked_text:
                flag(f"assistant answer not fully masked-in: {turn.content[:60]!r}")
            if turn.content in sample.unmasked_text:
                flag("assistant answer leaked into the unmasked partition")
        if turn.role == "assistant" and turn.tool_calls:
            for call in turn.tool_calls:
                if call.tool not in sample.masked_text:
                    flag(f"tool call {call.tool!r} not in the masked partition")
        if turn.role == "user" and turn.content:
            if turn.content in sample.masked_text:
                flag(f"user utterance masked-in (would be trained on): {turn.content[:60]!r}")
            if turn.content not in sample.unmasked_text:
                flag(f"user utterance missing from the rendered context: {turn.content[:60]!r}")
        if turn.role == "tool" and turn.tool_result is not None:
            probe = json.dumps(turn.tool_result, ensure_ascii=False, sort_keys=True)[:40]
            if probe and probe in sample.masked_text:
                flag("tool result leaked into the trainable partition")

    final_answers = [
        t.content for t in conv.turns if t.role == "assistant" and t.content and not t.tool_calls
    ]
    if final_answers and "INTENT: " not in sample.masked_text:
        flag("INTENT preamble did not survive rendering")
    if any("**" in (a or "") for a in final_answers) and "**" not in sample.masked_text:
        flag("bold-title markup did not survive rendering")
    return violations


def render_stats(samples: list[RenderedSample], max_seq: int = 4096) -> RenderStats:
    lengths = sorted(len(s.input_ids) for s in samples)
    if not lengths:
        return RenderStats(samples=0, token_p50=0, token_p95=0, token_max=0, over_max_seq=0)

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    return RenderStats(
        samples=len(lengths),
        token_p50=pct(0.50),
        token_p95=pct(0.95),
        token_max=lengths[-1],
        over_max_seq=sum(1 for n in lengths if n > max_seq),
    )
