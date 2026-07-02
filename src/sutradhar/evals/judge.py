"""LLM-as-judge client + governance plumbing (P3 task 7; DEC-P3-1, P3_SPEC §2.5).

The judge is a **self-hosted OSS model served by vLLM in a short ephemeral GPU session**
(`make gpu-judge`), reached through the same OpenAI-compatible surface as everything else:
``JUDGE_BASE_URL`` / ``JUDGE_MODEL`` / ``JUDGE_API_KEY``. The DEC-P3-1 A↔B swap
(self-hosted ↔ frontier escalation) is therefore **pure configuration** — this module has
no provider-specific code. Internally the client is an :class:`LLMClient` constructed over
the judge env values, inheriting the DEC-P0-4 off/error contract and the single-injected-
http-client testability.

Governance pins (recorded in every artifact's ``judge`` block + the Table 2 stamp):
``{model, revision, prompt file + sha256 hash, temperature 0, ragas version}``. Judge
rubrics live in-repo (``evals/prompts/judge_*.md``); their hashes are computed per file —
deliberately SEPARATE from the base-model ``prompt_hash`` (changing a rubric must not fake
a base-prompt change). gpt-oss-20b is a reasoning model, so calls pin ``reasoning_effort``
and use vLLM guided decoding (``guided_json``) for parseable, deterministic verdicts —
the same pattern the P1 extraction run proved (DEC-P1-4 amendment).

Malformed judge output NEVER crashes a batch: it becomes ``JudgeVerdict(error=…)``
(``judge_error`` in the artifact). An unset ``JUDGE_BASE_URL`` means "judge off" —
callers skip cleanly with a clear message (the P0 first-class "off" posture).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sutradhar.config import Settings
from sutradhar.serving.llm_client import LLMClient

PROMPTS_DIR = Path("evals/prompts")
COHERENCE_PROMPT = "judge_coherence_v1.md"
FAITHFULNESS_PROMPT = "judge_faithfulness_v1.md"

# Guided-decoding schema for the verdict (DEC-P3-1: parseable + deterministic at temp 0).
VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "criteria": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "rationale": {"type": "string"},
    },
    "required": ["score"],
    "additionalProperties": False,
}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def judge_prompt_hash(prompt_file: str, directory: Path = PROMPTS_DIR) -> str:
    return hashlib.sha256((directory / prompt_file).read_bytes()).hexdigest()


class JudgeConfig(BaseModel):
    """The frozen judge pin, stamped into every artifact (P3_SPEC §2.2/§2.5)."""

    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    revision: str | None = None  # HF revision SHA, frozen by task 13's κ gate
    prompt_file: str
    prompt_hash: str
    temperature: float = 0.0
    reasoning_effort: str = "low"
    ragas_version: str | None = None


class JudgeVerdict(BaseModel):
    """One judged item. ``error`` set (score None) = judge_error — recorded, not raised."""

    model_config = ConfigDict(extra="forbid")

    score: float | None = None
    criteria: dict[str, bool] = {}
    rationale: str | None = None
    error: str | None = None


def parse_verdict(text: str | None) -> JudgeVerdict:
    """Lenient rubric-output parse: bare JSON, fenced JSON, or JSON embedded in prose.
    Anything unusable → ``JudgeVerdict(error=…)``, never an exception (P3_SPEC §4)."""
    if not text or not text.strip():
        return JudgeVerdict(error="judge_error: empty output")
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return JudgeVerdict(error=f"judge_error: no JSON object in output: {text[:120]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return JudgeVerdict(error=f"judge_error: unparseable JSON ({exc})")
    if not isinstance(payload, dict):
        return JudgeVerdict(error="judge_error: output is not a JSON object")
    score = payload.get("score")
    if not isinstance(score, int | float) or not 0.0 <= float(score) <= 1.0:
        return JudgeVerdict(error=f"judge_error: score missing or out of [0,1]: {score!r}")
    criteria = payload.get("criteria", {})
    if not isinstance(criteria, dict):
        criteria = {}
    rationale = payload.get("rationale")
    return JudgeVerdict(
        score=float(score),
        criteria={k: bool(v) for k, v in criteria.items()},
        rationale=str(rationale) if rationale is not None else None,
    )


class JudgeClient:
    """Env-driven judge over the shared OpenAI-compatible client (see module doc)."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.Client | None = None,
        prompts_dir: Path = PROMPTS_DIR,
    ) -> None:
        self._settings = settings
        self._prompts_dir = prompts_dir
        self._client: LLMClient | None = None
        if self.available:
            judge_settings = settings.model_copy(
                update={
                    "llm_base_url": settings.judge_base_url,
                    "llm_model": settings.judge_model or "",
                    "llm_api_key": settings.judge_api_key or "EMPTY",
                    "llm_timeout_s": 120.0,
                }
            )
            self._client = LLMClient(judge_settings, http_client=http_client)

    @property
    def available(self) -> bool:
        """False when JUDGE_BASE_URL/JUDGE_MODEL are unset — callers skip cleanly."""
        return bool(self._settings.judge_base_url and self._settings.judge_model)

    def config(self, prompt_file: str, *, ragas_version: str | None = None) -> JudgeConfig:
        return JudgeConfig(
            model=self._settings.judge_model or "",
            base_url=self._settings.judge_base_url or "",
            prompt_file=prompt_file,
            prompt_hash=judge_prompt_hash(prompt_file, self._prompts_dir),
            ragas_version=ragas_version,
        )

    def _render(self, prompt_file: str, substitutions: dict[str, str]) -> str:
        template = (self._prompts_dir / prompt_file).read_text(encoding="utf-8")
        for key, value in substitutions.items():
            template = template.replace("{{" + key + "}}", value)
        return template

    def _judge(self, prompt: str) -> JudgeVerdict:
        if self._client is None:
            return JudgeVerdict(error="judge_error: judge off (JUDGE_BASE_URL unset)")
        result = self._client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            extra_body={
                "guided_json": VERDICT_JSON_SCHEMA,
                "reasoning_effort": "low",
            },
        )
        if result.status != "up":
            return JudgeVerdict(error=f"judge_error: endpoint {result.status}: {result.detail}")
        return parse_verdict(result.content)

    def judge_coherence(self, conversation: list[dict[str, str]]) -> JudgeVerdict:
        """GS-08 backtracking-coherence rubric over (user, assistant-answer) turns."""
        rendered = "\n".join(
            f"[turn {i + 1}] USER: {t['user']}\n[turn {i + 1}] ASSISTANT: {t['assistant']}"
            for i, t in enumerate(conversation)
        )
        return self._judge(self._render(COHERENCE_PROMPT, {"conversation": rendered}))

    def judge_faithfulness(self, answer: str, allowed_titles: list[str]) -> JudgeVerdict:
        """Supplementary faithfulness rubric (validation sample; the detector gates)."""
        return self._judge(
            self._render(
                FAITHFULNESS_PROMPT,
                {
                    "allowed_titles": "\n".join(f"- {t}" for t in sorted(allowed_titles))
                    or "(none)",
                    "answer": answer,
                },
            )
        )


# --- Human-agreement statistics (§2.5: percent agreement + Cohen's κ) ---


def percent_agreement(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's κ over categorical labels (binary here: coherent/faithful vs not).
    κ = (p_o − p_e) / (1 − p_e); κ = 1.0 when both raters are constant AND identical."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    n = len(a)
    p_o = percent_agreement(a, b)
    categories = set(a) | set(b)
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def binarize(score: float | None, threshold: float = 0.5) -> int:
    """Judge [0,1] score → binary label for κ against human binary labels.
    A judge_error (score None) counts as the negative class (a disagreement risk the
    judge owns, not a skipped item — errors must hurt agreement, not hide)."""
    return 1 if score is not None and score >= threshold else 0
