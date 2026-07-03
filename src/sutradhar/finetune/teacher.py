"""Teacher client + surface pass (P4 task 6; DEC-P4-1/P4-2).

The teacher rewrites ONLY linguistic surfaces — user utterances and assistant answer
prose — around task-5 sentinel locks; entities (titles, years) are physically outside its
reach and every rewrite is re-verified (``verify_locked``) before acceptance. A rejected
rewrite gets ONE retry, then the scaffold surface is kept and the rejection logged — the
dataset is always complete, and the **rejection rate is the DEC-P4-1 escalation signal**
(> 30% after one prompt revision → the recorded frontier-API escalation, ToS row first).

Client contract: OpenAI-compatible via ``TEACHER_BASE_URL/MODEL/API_KEY`` (DEC-P0-4 —
Sarvam-M on the ephemeral GPU and the frontier escalation are the same three env vars).
Blank ⇒ ``available`` is False and callers skip cleanly. Raw teacher outputs are cached as
a versioned artifact (``RewriteRecord`` JSONL) so the sealed dataset is reproducible from
the cache without replaying a non-deterministic model.

CI never calls a model: tests inject a fake ``rewrite`` callable (DEC-P2-6 posture).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sutradhar.config import Settings
from sutradhar.finetune.dataset import TeacherStamp, TrainingConversation
from sutradhar.finetune.validate import lock_entities, unlock_entities, verify_locked
from sutradhar.serving.llm_client import LLMClient

PROMPT_PATH = Path("finetune/prompts/teacher_rewrite_v1.md")

# The rewrite callable signature: (locked_text, register, kind) -> raw teacher output.
RewriteFn = Callable[[str, str, str], str]

# DEC-P4-1 escalation trigger: validator rejection rate above this after one prompt
# revision → frontier-API escalation (ToS row lands in LICENSING.md first).
ESCALATION_REJECTION_RATE = 0.30


class RewriteRecord(BaseModel):
    """One cached teacher call — the versioned raw-output artifact row."""

    model_config = ConfigDict(extra="forbid")

    conv_id: str
    turn_index: int
    kind: str  # "user" | "answer"
    target_register: str
    locked_input: str
    raw_output: str
    accepted: bool
    reasons: list[str] = []
    retries: int = 0


class TeacherRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts_total: int
    accepted: int
    rejected: int
    rejection_rate: float
    escalation_triggered: bool  # DEC-P4-1: > 30% → frontier escalation path
    stamp: TeacherStamp


def prompt_sha256(path: Path = PROMPT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_prompt(text: str, register: str, kind: str, path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    for key, value in (("register", register), ("kind", kind), ("text", text)):
        template = template.replace("{{" + key + "}}", value)
    return template


class TeacherClient:
    """Env-driven teacher over the shared OpenAI-compatible client (JudgeClient pattern)."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.Client | None = None,
        prompt_path: Path = PROMPT_PATH,
        temperature: float = 0.7,  # register diversity; reproducibility = the cache
        max_tokens: int = 1024,
    ) -> None:
        self._settings = settings
        self._prompt_path = prompt_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: LLMClient | None = None
        if self.available:
            teacher_settings = settings.model_copy(
                update={
                    "llm_base_url": settings.teacher_base_url,
                    "llm_model": settings.teacher_model or "",
                    "llm_api_key": settings.teacher_api_key or "EMPTY",
                    "llm_timeout_s": 120.0,
                }
            )
            self._client = LLMClient(teacher_settings, http_client=http_client)

    @property
    def available(self) -> bool:
        """False when TEACHER_BASE_URL/TEACHER_MODEL are unset — callers skip cleanly."""
        return bool(self._settings.teacher_base_url and self._settings.teacher_model)

    def stamp(self, revision: str = "main") -> TeacherStamp:
        return TeacherStamp(
            model=self._settings.teacher_model or "",
            revision=revision,
            prompt_sha256=prompt_sha256(self._prompt_path),
        )

    def rewrite(self, locked_text: str, register: str, kind: str) -> str:
        if self._client is None:
            raise RuntimeError("teacher off (TEACHER_BASE_URL unset) — surface pass skipped")
        prompt = render_prompt(locked_text, register, kind, self._prompt_path)
        result = self._client.chat(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        if result.status != "up":
            raise RuntimeError(f"teacher call failed: {result.detail}")
        return (result.content or "").strip()


# --- The surface pass ---


def _result_entities(conv: TrainingConversation) -> list[str]:
    """Lockable spans: every title + every year in the conversation's tool results."""
    entities: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("title", "matched_title", "canonical_title") and isinstance(value, str):
                    entities.add(value)
                if key == "year" and isinstance(value, int):
                    entities.add(str(value))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for turn in conv.turns:
        if turn.tool_result is not None:
            walk(turn.tool_result)
    return sorted(entities)


def _rewrite_one(
    rewrite: RewriteFn,
    text: str,
    entities: list[str],
    register: str,
    kind: str,
    *,
    max_retries: int = 1,
) -> tuple[str, bool, list[str], int, str, str]:
    """Lock → teach → verify (retry once) → unlock.

    Returns ``(final_text, accepted, reasons, retries, locked_input, raw_output)`` — on
    rejection the original text is kept (scaffold fallback; dataset stays complete) and
    ``raw_output`` is the last raw teacher output, cached for the audit artifact.
    """
    locked, mapping = lock_entities(text, entities)
    require_preamble = kind == "answer" and locked.startswith("INTENT: ")
    reasons: list[str] = []
    raw = ""
    for attempt in range(max_retries + 1):
        raw = rewrite(locked, register, kind)
        reasons = verify_locked(locked, raw, mapping, require_preamble=require_preamble)
        if not reasons:
            return unlock_entities(raw, mapping), True, [], attempt, locked, raw
    return text, False, reasons, max_retries, locked, raw


def surface_pass(
    conversations: list[TrainingConversation],
    rewrite: RewriteFn,
    stamp: TeacherStamp,
) -> tuple[list[TrainingConversation], list[RewriteRecord], TeacherRunSummary]:
    """Rewrite every user utterance + final prose answer; entities sentinel-locked.

    Returns (taught conversations, the raw-output cache records, the run summary).
    """
    taught: list[TrainingConversation] = []
    records: list[RewriteRecord] = []
    for conv in conversations:
        entities = _result_entities(conv)
        register = conv.query_lang
        new_conv = TrainingConversation.model_validate(conv.model_dump())
        for index, turn in enumerate(new_conv.turns):
            if turn.role == "user" and turn.content:
                kind = "user"
            elif turn.role == "assistant" and turn.content and turn.tool_calls is None:
                kind = "answer"
            else:
                continue
            final_text, accepted, reasons, retries, locked, raw = _rewrite_one(
                rewrite, turn.content or "", entities, register, kind
            )
            records.append(
                RewriteRecord(
                    conv_id=conv.conv_id,
                    turn_index=index,
                    kind=kind,
                    target_register=register,
                    locked_input=locked,
                    raw_output=raw,
                    accepted=accepted,
                    reasons=reasons,
                    retries=retries,
                )
            )
            if accepted:
                turn.content = final_text
        new_conv.teacher = stamp
        taught.append(new_conv)

    total = len(records)
    rejected = sum(1 for r in records if not r.accepted)
    rate = round(rejected / total, 4) if total else 0.0
    summary = TeacherRunSummary(
        texts_total=total,
        accepted=total - rejected,
        rejected=rejected,
        rejection_rate=rate,
        escalation_triggered=rate > ESCALATION_REJECTION_RATE,
        stamp=stamp,
    )
    return taught, records, summary
