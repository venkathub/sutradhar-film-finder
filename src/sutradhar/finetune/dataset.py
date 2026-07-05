"""Training-dataset schema + card models (P4 task 2, P4_SPEC §2.2).

The synthetic SFT dataset is a JSONL file of :class:`TrainingConversation` rows plus a
:class:`DatasetCard` describing counts, provenance, decontamination, and the canonical
file hash. Everything here is *shape*: pydantic models with ``extra="forbid"`` and cheap
structural invariants (which roles may carry tool calls / results, label counts per user
turn). Deep semantics — v0 tool-call validation, invented-title detection, decontamination
math, quota checks — live in ``sutradhar.finetune.validate`` (task 5), by the spec's
component split.

Determinism contract (the property the card's ``sha256`` and Tier-1's
``test_ft_dataset_schema`` pin): serialization is canonical — sorted keys, compact
separators, ``ensure_ascii=False``, one ``\\n``-terminated line per conversation — so the
same conversations always produce a byte-identical JSONL and a stable hash, independent of
insertion order of model fields or the writing host.

The ``behaviour`` literal mirrors the FROZEN intent taxonomy
(``evals/prompts/intent_taxonomy_v1.json``, DEC-P3-4); a sync test guards the two against
silent drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

# Mirrors the frozen intent taxonomy v1 (evals/prompts/intent_taxonomy_v1.json).
# tests/test_ft_dataset_schema.py asserts this tuple == the taxonomy's intent keys.
BEHAVIOURS: tuple[str, ...] = (
    "find_by_plot",
    "find_by_title",
    "list_versions",
    "refine",
    "disambiguate",
    "out_of_catalog",
)

Behaviour = Literal[
    "find_by_plot",
    "find_by_title",
    "list_versions",
    "refine",
    "disambiguate",
    "out_of_catalog",
]

Role = Literal["system", "user", "assistant", "tool"]


class _StrictModel(BaseModel):
    """Base for all dataset models: unknown fields are schema violations, never ignored."""

    model_config = ConfigDict(extra="forbid")


class ToolCallRecord(_StrictModel):
    """One tool invocation in an assistant turn.

    ``tool`` must exist in ``tool_schema.v0.json`` and ``arguments`` must validate against
    its params schema — enforced by the task-5 validators (reusing the DEC-P1-8 validator),
    not here.
    """

    tool: str
    arguments: dict[str, Any]


class TrainingMessage(_StrictModel):
    """One turn. Assistant final answers carry the INTENT preamble line in ``content``."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCallRecord] | None = None  # assistant tool-calling turns only
    tool_result: dict[str, Any] | None = None  # tool role only; v0 result shape

    @model_validator(mode="after")
    def _structural_invariants(self) -> TrainingMessage:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError(f"tool_calls only valid on assistant turns, not {self.role!r}")
        if self.tool_calls is not None and not self.tool_calls:
            raise ValueError("tool_calls, when present, must be non-empty (use None instead)")
        if self.tool_result is not None and self.role != "tool":
            raise ValueError(f"tool_result only valid on tool turns, not {self.role!r}")
        if self.role == "tool" and self.tool_result is None:
            raise ValueError("tool turns must carry a tool_result")
        if self.content is None and self.tool_calls is None and self.tool_result is None:
            raise ValueError("empty message: needs content, tool_calls, or tool_result")
        return self


class TeacherStamp(_StrictModel):
    """Provenance of the teacher surface pass (None on a conversation = scaffold-only)."""

    model: str  # e.g. sarvamai/sarvam-m (env-driven; never hardcoded in code paths)
    revision: str  # pinned model revision / commit hash
    prompt_sha256: str  # hash of the rewrite-prompt template used


class TrainingConversation(_StrictModel):
    """One JSONL row: a grounded, labelled, multi-turn training conversation."""

    conv_id: str
    behaviour: Behaviour  # = frozen intent taxonomy (sync-tested against the artifact)
    query_lang: str  # ta-latin | hi-latin | kn-latin | te-latin | ml-latin | native | en
    turns: list[TrainingMessage]
    entity_ids: list[str]  # grounded work/version ids ([] for out_of_catalog)
    intent_labels: list[str]  # one per user turn (mirrors golden expected_intent)
    slot_labels: list[dict[str, Any]]  # one per user turn (frozen slot vocabulary only)
    scaffold_hash: str  # deterministic skeleton provenance
    teacher: TeacherStamp | None = None

    @model_validator(mode="after")
    def _structural_invariants(self) -> TrainingConversation:
        if not self.turns:
            raise ValueError("conversation must have at least one turn")
        user_turns = sum(1 for t in self.turns if t.role == "user")
        if user_turns == 0:
            raise ValueError("conversation must have at least one user turn")
        if len(self.intent_labels) != user_turns:
            raise ValueError(
                f"intent_labels must have one entry per user turn "
                f"({len(self.intent_labels)} labels for {user_turns} user turns)"
            )
        if len(self.slot_labels) != user_turns:
            raise ValueError(
                f"slot_labels must have one entry per user turn "
                f"({len(self.slot_labels)} labels for {user_turns} user turns)"
            )
        return self


class DecontReport(_StrictModel):
    """Decontamination evidence for the card (task-5 validators compute the numbers).

    Max rapidfuzz similarity of any training user utterance against each protected
    surface: golden fixture queries, frozen prompt exemplars, and ALL negative surfaces
    (GS-02 + evals/negatives/heldout.yaml) — P4_SPEC §2.2 / DEC-P4-3. A sealed dataset
    must have every max below ``threshold`` and ``violations == []``.
    """

    threshold: float
    max_similarity_golden: float
    max_similarity_exemplars: float
    max_similarity_negatives: float
    violations: list[str] = []  # conv_ids at/over threshold; must be empty to seal


class DatasetCard(_StrictModel):
    """The versioned dataset card committed in-repo alongside the sample (DEC-P4-7)."""

    dataset_id: str  # e.g. sutradhar-ft-v1 (FT_DATASET_ID)
    counts: dict[str, dict[str, int]]  # behaviour -> query_lang -> count
    graph_snapshot: str  # snapshot hash(es) the scaffolds were sampled from
    teacher: TeacherStamp | None
    seed: int
    decontamination: DecontReport
    split: dict[str, int]  # {"train": n, "val": m, "split_seed": s}
    licenses: list[str]  # provenance notes (Wikidata CC0, TMDB attribution, IMDb NC, teacher)
    sha256: str  # of the canonical JSONL


# ---------------------------------------------------------------------------
# Canonical (de)serialization + hashing
# ---------------------------------------------------------------------------


def canonical_json(model: BaseModel) -> str:
    """One canonical JSON encoding: sorted keys, compact, UTF-8 verbatim (no \\u escapes)."""
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def conversations_to_jsonl(conversations: Iterable[TrainingConversation]) -> str:
    """Render conversations as canonical JSONL text (one \\n-terminated line per row)."""
    return "".join(canonical_json(conv) + "\n" for conv in conversations)


def write_jsonl(path: Path, conversations: Sequence[TrainingConversation]) -> str:
    """Write the canonical JSONL and return its sha256 (the value the card pins)."""
    payload = conversations_to_jsonl(conversations)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[TrainingConversation]:
    """Load and re-validate every row (a hand-edited bad row fails here, not in training)."""
    conversations: list[TrainingConversation] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            conversations.append(TrainingConversation.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid TrainingConversation: {exc}") from exc
    return conversations


def dataset_sha256(path: Path) -> str:
    """sha256 of the JSONL file bytes — must equal the card's ``sha256`` for a sealed set."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_card(path: Path, card: DatasetCard) -> None:
    """Write the card deterministically (sorted keys, 2-space indent, trailing newline)."""
    payload = json.dumps(
        card.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def read_card(path: Path) -> DatasetCard:
    return DatasetCard.model_validate_json(path.read_text(encoding="utf-8"))
