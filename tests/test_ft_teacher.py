"""Teacher client + surface pass tests (P4 task 6) — fully mocked, zero model calls.

The fake ``rewrite`` callables simulate a faithful teacher, a sentinel-breaking teacher,
a preamble-dropping teacher, and a title-inventing teacher; the pass must accept only the
faithful one, keep the dataset complete on rejection, and account the DEC-P4-1 escalation
trigger. Taught output must STILL pass the task-5 validators (grounding survives the
teacher by construction).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.config import Settings
from sutradhar.finetune.dataset import TeacherStamp, TrainingConversation
from sutradhar.finetune.scaffold import ScaffoldConfig, generate
from sutradhar.finetune.snapshot import load_scaffold_snapshot
from sutradhar.finetune.teacher import (
    TeacherClient,
    _result_entities,
    prompt_sha256,
    render_prompt,
    surface_pass,
)
from sutradhar.finetune.validate import (
    SENTINEL_RE,
    validate_contracts,
    validate_grounding,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROMPT = _REPO_ROOT / "finetune" / "prompts" / "teacher_rewrite_v1.md"


def _stamp() -> TeacherStamp:
    return TeacherStamp(model="sarvamai/sarvam-m", revision="test", prompt_sha256="0" * 64)


@pytest.fixture(scope="module")
def conversations() -> list[TrainingConversation]:
    snapshot = load_scaffold_snapshot(_REPO_ROOT / "finetune" / "scaffold_snapshot.json")
    return generate(snapshot, ScaffoldConfig(seed=9, size=30))


# --- Fake teachers ---


def _faithful(locked: str, register: str, kind: str) -> str:
    """Keeps sentinels + preamble + structure; changes the connective prose."""
    if locked.startswith("INTENT: "):
        head, _, body = locked.partition("\n")
        lines = body.strip("\n").split("\n")
        rewritten = [f"arre, {line}!" if not line.startswith("- ") else line for line in lines]
        return head + "\n\n" + "\n".join(rewritten)
    return f"yaar {locked} pls"


def _sentinel_breaker(locked: str, register: str, kind: str) -> str:
    return SENTINEL_RE.sub("Some Other Film", locked)


def _preamble_dropper(locked: str, register: str, kind: str) -> str:
    if locked.startswith("INTENT: "):
        return locked.partition("\n")[2].strip()
    return locked


def _title_inventor(locked: str, register: str, kind: str) -> str:
    return locked + " btw also watch **Maestro Reloaded**!"


# --- Surface pass behaviour ---


def test_faithful_teacher_accepted_and_dataset_stays_valid(
    conversations: list[TrainingConversation],
) -> None:
    taught, records, summary = surface_pass(conversations, _faithful, _stamp())
    assert summary.rejected == 0 and not summary.escalation_triggered
    assert len(taught) == len(conversations)
    # Surfaces actually changed, entities stayed literal (unlocked back).
    changed = 0
    for before, after in zip(conversations, taught, strict=True):
        assert after.teacher is not None and after.teacher.model == "sarvamai/sarvam-m"
        for t_before, t_after in zip(before.turns, after.turns, strict=True):
            if t_before.role == "user" and t_before.content:
                assert "⟦" not in (t_after.content or "")
                if t_before.content != t_after.content:
                    changed += 1
    assert changed > 0
    # The task-5 gate re-earns grounding + contracts on the TAUGHT dataset.
    assert validate_grounding(taught) == []
    import json

    taxonomy = json.loads(
        (_REPO_ROOT / "evals" / "prompts" / "intent_taxonomy_v1.json").read_text()
    )
    assert validate_contracts(taught, set(taxonomy["intents"]), set(taxonomy["slot_keys"])) == []


def test_sentinel_breaker_rejected_scaffold_kept(
    conversations: list[TrainingConversation],
) -> None:
    taught, records, summary = surface_pass(conversations[:5], _sentinel_breaker, _stamp())
    # Rewrites touching entity-bearing texts are rejected; originals kept verbatim.
    rejected = [r for r in records if not r.accepted]
    assert rejected, "sentinel-breaking rewrites must be rejected"
    for r in rejected:
        assert any("locked span altered" in reason for reason in r.reasons)
        assert r.retries == 1  # one retry happened before giving up
    for before, after in zip(conversations[:5], taught, strict=True):
        entity_turns = [
            (b.content, a.content)
            for b, a in zip(before.turns, after.turns, strict=True)
            if b.role == "assistant" and b.content and b.tool_calls is None
        ]
        for b_content, a_content in entity_turns:
            if "**" in (b_content or ""):
                assert a_content == b_content  # scaffold surface kept
    assert validate_grounding(taught) == []


def test_preamble_dropper_rejected_on_answers(conversations: list[TrainingConversation]) -> None:
    _, records, summary = surface_pass(conversations[:5], _preamble_dropper, _stamp())
    answer_records = [r for r in records if r.kind == "answer"]
    assert answer_records
    assert all(not r.accepted for r in answer_records)
    assert all(any("preamble" in reason for reason in r.reasons) for r in answer_records)


def test_title_inventor_rejected(conversations: list[TrainingConversation]) -> None:
    taught, records, _ = surface_pass(conversations[:5], _title_inventor, _stamp())
    assert all(not r.accepted for r in records)
    assert validate_grounding(taught) == []  # nothing invented survived


def test_escalation_trigger_accounting(conversations: list[TrainingConversation]) -> None:
    _, _, summary = surface_pass(conversations[:10], _sentinel_breaker, _stamp())
    assert summary.rejection_rate > 0.30
    assert summary.escalation_triggered  # DEC-P4-1: > 30% -> frontier escalation path


# --- Entities + prompt plumbing ---


def test_result_entities_cover_titles_and_years(
    conversations: list[TrainingConversation],
) -> None:
    conv = next(c for c in conversations if c.behaviour == "find_by_title")
    entities = _result_entities(conv)
    assert any(not e.isdigit() for e in entities)  # titles
    assert any(e.isdigit() and len(e) == 4 for e in entities)  # years


def test_render_prompt_substitutes_everything() -> None:
    rendered = render_prompt("hello ⟦T1⟧", "ta-latin", "user", _PROMPT)
    assert "{{" not in rendered
    assert "hello ⟦T1⟧" in rendered and "ta-latin" in rendered


def test_prompt_hash_is_stable() -> None:
    assert prompt_sha256(_PROMPT) == prompt_sha256(_PROMPT)


# --- Client env contract (DEC-P0-4 posture; no network) ---


def test_client_off_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TEACHER_BASE_URL", "TEACHER_MODEL", "TEACHER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    client = TeacherClient(settings)
    assert not client.available
    with pytest.raises(RuntimeError, match="teacher off"):
        client.rewrite("text", "en", "user")


def test_client_available_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEACHER_BASE_URL", "http://teacher.example:8000/v1")
    monkeypatch.setenv("TEACHER_MODEL", "sarvamai/sarvam-m")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    client = TeacherClient(settings, prompt_path=_PROMPT)
    assert client.available
    stamp = client.stamp(revision="abc123")
    assert stamp.model == "sarvamai/sarvam-m"
    assert stamp.revision == "abc123"
    assert stamp.prompt_sha256 == prompt_sha256(_PROMPT)
