"""Judge module tests (P3 task 7; P3_SPEC §4 test_judge.py).

Covers: prompt-hashing stability; judge client env-wiring + redaction; endpoint-agnostic
wiring (the same client against a vLLM-style and a frontier-style endpoint — the DEC-P3-1
A↔B config swap); rubric parse of malformed judge output (never crashes, records
judge_error); κ/agreement math; worksheet build (foils) + report computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sutradhar.config import Settings
from sutradhar.evals.driver import FixtureTranscript
from sutradhar.evals.judge import (
    COHERENCE_PROMPT,
    FAITHFULNESS_PROMPT,
    JudgeClient,
    binarize,
    cohens_kappa,
    judge_prompt_hash,
    parse_verdict,
    percent_agreement,
)
from sutradhar.evals.judge_validation import (
    build_worksheet,
    coherence_foil,
    compute_report,
    faithfulness_foil,
    load_worksheet,
    save_worksheet,
)

_REPO = Path(__file__).resolve().parents[1]
_PROMPTS = _REPO / "evals" / "prompts"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "JUDGE_BASE_URL": "http://localhost:8001/v1",
        "JUDGE_MODEL": "openai/gpt-oss-20b",
        "JUDGE_API_KEY": "judge_secret_key_123",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _verdict_response(score: float = 1.0) -> httpx.Response:
    content = json.dumps({"score": score, "criteria": {"context_carried": True}, "rationale": "ok"})
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "judge",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        },
    )


def _client(handler: Any, **overrides: Any) -> JudgeClient:
    return JudgeClient(
        _settings(**overrides),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        prompts_dir=_PROMPTS,
    )


# --- Prompt hashing ---


def test_prompt_hash_stable_and_edit_sensitive(tmp_path: Path) -> None:
    h1 = judge_prompt_hash(COHERENCE_PROMPT, _PROMPTS)
    assert h1 == judge_prompt_hash(COHERENCE_PROMPT, _PROMPTS)  # stable
    assert h1 != judge_prompt_hash(FAITHFULNESS_PROMPT, _PROMPTS)  # per-file
    copy = tmp_path / COHERENCE_PROMPT
    copy.write_bytes((_PROMPTS / COHERENCE_PROMPT).read_bytes())
    assert judge_prompt_hash(COHERENCE_PROMPT, tmp_path) == h1
    copy.write_text(copy.read_text("utf-8") + "x", "utf-8")
    assert judge_prompt_hash(COHERENCE_PROMPT, tmp_path) != h1


def test_judge_config_pins_prompt_hash() -> None:
    judge = _client(lambda r: _verdict_response())
    config = judge.config(COHERENCE_PROMPT, ragas_version="0.4.3")
    assert config.model == "openai/gpt-oss-20b"
    assert config.prompt_hash == judge_prompt_hash(COHERENCE_PROMPT, _PROMPTS)
    assert config.temperature == 0.0
    assert config.ragas_version == "0.4.3"


# --- Env wiring + redaction + A<->B endpoint agnosticism ---


def test_judge_off_when_unset_skips_cleanly() -> None:
    judge = JudgeClient(_settings(JUDGE_BASE_URL="", JUDGE_MODEL=""), prompts_dir=_PROMPTS)
    assert judge.available is False
    verdict = judge.judge_coherence([{"user": "u", "assistant": "a"}])
    assert verdict.error is not None and "judge off" in verdict.error
    assert verdict.score is None  # skipped, never crashed


def test_judge_request_carries_env_wiring_and_key_never_leaks() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _verdict_response()

    verdict = _client(handler).judge_coherence([{"user": "u", "assistant": "a"}])
    assert verdict.score == 1.0
    assert seen["url"].startswith("http://localhost:8001/v1")
    assert seen["body"]["model"] == "openai/gpt-oss-20b"
    assert seen["auth"] == "Bearer judge_secret_key_123"
    assert seen["body"]["temperature"] == 0.0
    # DEC-P3-1 governance: guided decoding + pinned reasoning effort on the wire.
    assert seen["body"]["guided_json"]["required"] == ["score"]
    assert seen["body"]["reasoning_effort"] == "low"
    # The key never leaks into anything we might record.
    assert "judge_secret_key_123" not in json.dumps(verdict.model_dump())
    assert "judge_secret_key_123" not in json.dumps(
        _client(handler).config(COHERENCE_PROMPT).model_dump()
    )


def test_endpoint_agnostic_a_b_swap_is_pure_config() -> None:
    """Same client code against a vLLM-style and a frontier-style endpoint (DEC-P3-1)."""
    for base_url, model in [
        ("http://gpu-instance:8000/v1", "openai/gpt-oss-20b"),  # self-hosted vLLM
        ("https://api.frontier.example/v1", "frontier-judge-2026-01"),  # escalation
    ]:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request, seen: dict[str, Any] = seen) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["model"] = json.loads(request.content)["model"]
            return _verdict_response(0.67)

        judge = _client(handler, JUDGE_BASE_URL=base_url, JUDGE_MODEL=model)
        verdict = judge.judge_faithfulness("**Drishyam (2013)**", ["Drishyam"])
        assert verdict.score == 0.67
        assert seen["url"].startswith(base_url)
        assert seen["model"] == model


def test_judge_endpoint_down_records_error_not_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    verdict = _client(handler).judge_coherence([{"user": "u", "assistant": "a"}])
    assert verdict.score is None
    assert verdict.error is not None and "off" in verdict.error


# --- Malformed judge output (never crashes; judge_error recorded) ---


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "The conversation looks fine to me.",  # prose, no JSON
        '{"score": 1.0',  # truncated
        '{"criteria": {}}',  # score missing
        '{"score": 7}',  # out of range
        '{"score": "high"}',  # wrong type
    ],
)
def test_parse_verdict_malformed_records_judge_error(raw: str | None) -> None:
    verdict = parse_verdict(raw)
    assert verdict.score is None
    assert verdict.error is not None and verdict.error.startswith("judge_error")


def test_parse_verdict_accepts_fenced_and_embedded_json() -> None:
    fenced = '```json\n{"score": 0.67, "rationale": "ok"}\n```'
    assert parse_verdict(fenced).score == 0.67
    embedded = 'Here is my verdict: {"score": 0.33, "criteria": {"no_reanswer": false}} done.'
    verdict = parse_verdict(embedded)
    assert verdict.score == 0.33
    assert verdict.criteria == {"no_reanswer": False}


# --- Agreement statistics ---


def test_cohens_kappa_known_values() -> None:
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0  # perfect
    # 2x2 worked example: po=0.8, pe=0.5 -> kappa=0.6
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert cohens_kappa(a, b) == pytest.approx(0.6)
    assert percent_agreement(a, b) == pytest.approx(0.8)
    # Constant raters: identical -> 1.0; differing -> 0.0 (never a div-by-zero crash).
    assert cohens_kappa([1, 1], [1, 1]) == 1.0
    assert cohens_kappa([1, 1], [0, 0]) == 0.0
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_binarize_judge_error_counts_negative() -> None:
    assert binarize(0.67) == 1
    assert binarize(0.33) == 0
    assert binarize(None) == 0  # judge_error hurts agreement, never hides


# --- Worksheet build + report (the make judge-worksheet / judge-validate machinery) ---


def _transcript(fixture_id: str, answers: list[str | None], users: list[str]) -> FixtureTranscript:
    return FixtureTranscript(
        fixture_id=fixture_id,
        prompt_hash="h",
        chat_status="up",
        messages=[{"role": "user", "content": u} for u in users],
        calls=[],
        answers=answers,
    )


def test_build_worksheet_items_and_blind_foils(tmp_path: Path) -> None:
    transcripts = [
        _transcript("GS-08a", ["**A (2015)**", "**B (2013)**", "**C (2014)**"], ["q1", "q2", "q3"]),
        _transcript("GS-07a", ["**Papanasam (2015)** hai"], ["q"]),
    ]
    items, key = build_worksheet(transcripts)
    ids = [i.item_id for i in items]
    # multi-turn -> coherence pair + faithfulness pair; single-turn -> faithfulness pair.
    assert ids == [
        "coh-GS-08a",
        "coh-GS-08a-foil",
        "fai-GS-08a",
        "fai-GS-08a-foil",
        "fai-GS-07a",
        "fai-GS-07a-foil",
    ]
    assert key["coh-GS-08a-foil"]["is_foil"] is True
    # The worksheet itself is BLIND: no foil markers on items.
    assert all("foil" not in i.model_dump(exclude_none=True) for i in items)
    # Round-trip through the YAML file.
    save_worksheet(items, key, tmp_path)
    loaded = load_worksheet(tmp_path)
    assert [i.item_id for i in loaded] == ids
    assert all(i.human_label is None for i in loaded)


def test_foils_are_deterministic_and_wrong() -> None:
    turns = [
        {"user": "the Drishyam with Ajay Devgn", "assistant": "**Drishyam (2015, hi)**"},
        {"user": "no, the original one", "assistant": "**Drishyam (2013, ml)**"},
    ]
    foiled = coherence_foil(turns)
    assert foiled[1]["assistant"] == "**Drishyam (2015, hi)**"  # re-answers turn 1
    assert coherence_foil(turns) == foiled  # deterministic
    assert "Chokher Aloy" in faithfulness_foil("**Papanasam (2015)**.")


def test_compute_report_kappa_against_fake_judge(tmp_path: Path) -> None:
    """Judge that nails every real item and every foil -> kappa 1.0; report shape sane."""
    transcripts = [
        _transcript("GS-08a", ["**A (2015)**", "**B (2013)**", "**C (2014)**"], ["q1", "q2", "q3"]),
        _transcript("GS-07a", ["**Papanasam (2015)** hai"], ["q"]),
    ]
    items, key = build_worksheet(transcripts)
    for item in items:  # human labels: real = 1, foil = 0
        item.human_label = 0 if key[item.item_id]["is_foil"] else 1

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        bad = "Chokher Aloy" in prompt or prompt.count("**A (2015)**") > 1
        return _verdict_response(0.0 if bad else 1.0)

    report = compute_report(items, _client(handler))
    assert report.n_items == 6
    assert report.cohens_kappa == 1.0
    assert report.percent_agreement == 1.0
    assert "PASS" in report.gate
    assert set(report.per_kind) == {"coherence", "faithfulness"}
    assert report.judge["coherence"]["prompt_hash"] == judge_prompt_hash(COHERENCE_PROMPT, _PROMPTS)


def test_compute_report_requires_all_labels() -> None:
    items, _ = build_worksheet([_transcript("GS-07a", ["**X (2000)**"], ["q"])])
    with pytest.raises(ValueError, match="unlabelled"):
        compute_report(items, _client(lambda r: _verdict_response()))
