"""Scripted mock endpoint — the dry-run "model" (P3 task 11; P3_SPEC §1.13, §2.1).

A canned-transcript player behind ``httpx.MockTransport``: no model, no GPU, no network.
It plays a *well-behaved* assistant derived from the golden labels themselves — for each
generation fixture it emits the fixture's ``expected_tool_calls`` (placeholders bound to
REAL ids read out of the prior tool results in the request messages), then answers with
the frozen INTENT preamble + bold-title contract, grounded in the tool results.

Two faults are DELIBERATELY seeded so the committed dry-run artifact proves the gates
catch them end-to-end (never published to Table 2 — ``mode: "dry_run"``):

- **Seeded hallucinated tool call** (GS-07a, first round): ``lookup_movie`` — not in
  TOOL_SCHEMA v0. The driver must record the validation failure, feed the error back,
  and the conversation must recover (schema-validity < 1.0 in the artifact, visibly).
- **Seeded invented movie** (GS-07e answer): "Chokher Aloy (2016)" — never in any tool
  result. The detector must count exactly this one invention on the code_mixed slice
  while the GS-02 slice stays at 0 (the hard gate stays green).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from sutradhar.config import Settings
from sutradhar.evals.golden import GoldenFixture, load_fixtures
from sutradhar.serving import LLMClient

SEEDED_HALLUCINATED_TOOL_FIXTURE = "GS-07a"
SEEDED_INVENTION_FIXTURE = "GS-07e"
SEEDED_INVENTION_SENTENCE = (
    " Idara Bengali remake **Chokher Aloy (2016)** kuda chudandi."  # never in the catalog
)

_LANG_NAMES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "bn": "Bengali",
}


def _turns(fixture: GoldenFixture) -> list[str]:
    return fixture.query if isinstance(fixture.query, list) else [fixture.query]


def _intent_for_turn(fixture: GoldenFixture, turn: int) -> tuple[str, dict[str, Any]]:
    intents = (
        fixture.expected_intent
        if isinstance(fixture.expected_intent, list)
        else [fixture.expected_intent]
    )
    slots = (
        fixture.expected_slots
        if isinstance(fixture.expected_slots, list)
        else [fixture.expected_slots]
    )
    intent = intents[turn] if turn < len(intents) and intents[turn] else "find_by_title"
    turn_slots = slots[turn] if turn < len(slots) and slots[turn] is not None else {}
    return str(intent), dict(turn_slots)


def _allocate_calls(fixture: GoldenFixture) -> list[list[dict[str, Any]]]:
    """Expected calls per turn: the last (n_turns − 1) calls map one per later turn
    (the GS-08 refine pattern); everything earlier belongs to turn 1."""
    calls = [c.model_dump() for c in fixture.expected_tool_calls or []]
    n_turns = len(_turns(fixture))
    if n_turns == 1 or len(calls) < n_turns:
        return [calls] + [[] for _ in range(n_turns - 1)]
    head = calls[: len(calls) - (n_turns - 1)]
    tail = calls[len(calls) - (n_turns - 1) :]
    return [head] + [[c] for c in tail]


def _answer_versions(fixture: GoldenFixture) -> dict[int, list[Any]]:
    """Turn → expected versions to assert. Multi-turn: one version per non-abstain turn
    (the 'expected answer per turn, in order' convention); single-turn: all of them."""
    versions = fixture.expected.versions
    n_turns = len(_turns(fixture))
    if n_turns == 1:
        return {0: list(versions)}
    mapping: dict[int, list[Any]] = {}
    v = 0
    for turn in range(n_turns):
        intent, _ = _intent_for_turn(fixture, turn)
        if intent == "out_of_catalog":
            mapping[turn] = []
        elif v < len(versions):
            mapping[turn] = [versions[v]]
            v += 1
        else:
            mapping[turn] = []
    return mapping


class ScriptedModel:
    """The request handler: identifies fixture + turn from the messages, then plays."""

    def __init__(self, fixtures: list[GoldenFixture]) -> None:
        self._by_first_turn: dict[str, GoldenFixture] = {_turns(f)[0]: f for f in fixtures}
        self.requests: int = 0

    # -- message helpers --

    @staticmethod
    def _user_messages(messages: list[dict[str, Any]]) -> list[str]:
        return [str(m.get("content", "")) for m in messages if m.get("role") == "user"]

    @staticmethod
    def _tool_results_since_last_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "user":
                results = []
            elif m.get("role") == "tool":
                try:
                    results.append(json.loads(str(m.get("content", ""))))
                except json.JSONDecodeError:
                    results.append({})
        return results

    @staticmethod
    def _all_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m.get("role") == "tool":
                try:
                    out.append(json.loads(str(m.get("content", ""))))
                except json.JSONDecodeError:
                    out.append({})
        return out

    def _bind(self, args: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve $work_id / [$version_set] against ids the conversation returned."""
        results = self._all_tool_results(messages)
        bound = dict(args)
        for key, value in args.items():
            if value == "$work_id":
                bound[key] = self._first_work_id(results)
            elif value == ["$version_set"]:
                bound[key] = self._latest_version_set(results)
        return bound

    @staticmethod
    def _first_work_id(results: list[dict[str, Any]]) -> str:
        for r in results:
            for c in r.get("candidates", []) or r.get("results", []):
                if "work_id" in c:
                    return str(c["work_id"])
        return "unresolved"

    @staticmethod
    def _latest_version_set(results: list[dict[str, Any]]) -> list[str]:
        for r in reversed(results):
            if "original" in r and "versions" in r:  # get_versions shape
                ids = [r["original"]["version_id"]] if r.get("original") else []
                ids += [v["version_id"] for v in r["versions"]]
                return [str(i) for i in ids]
        return []

    # -- the play --

    def _answer(self, fixture: GoldenFixture, turn: int) -> str:
        intent, slots = _intent_for_turn(fixture, turn)
        preamble = "INTENT: " + json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)
        versions = _answer_versions(fixture).get(turn, [])
        if fixture.expected.no_match or intent == "out_of_catalog":
            body = (
                "I checked the catalog by title and story — that film has no match in it, "
                "so I can't list versions for you. NO_MATCH."
            )
        else:
            lines = []
            for v in versions:
                lang = _LANG_NAMES.get(v.language, v.language)
                tag = "original" if v.is_original else "remake"
                lines.append(f"- **{v.title} ({v.year}, {lang})** — {tag}")
            body = "Ye raha:\n" + "\n".join(lines) if lines else "Done."
        answer = f"{preamble}\n\n{body}"
        if fixture.id == SEEDED_INVENTION_FIXTURE:
            answer += SEEDED_INVENTION_SENTENCE
        return answer

    def _next_action(self, messages: list[dict[str, Any]]) -> dict[str, Any] | str:
        users = self._user_messages(messages)
        fixture = self._by_first_turn.get(users[0]) if users else None
        if fixture is None:
            return "I don't recognise this conversation (mock endpoint)."
        turn = len(users) - 1
        plan = _allocate_calls(fixture)
        turn_calls = plan[turn] if turn < len(plan) else []
        done = len(self._tool_results_since_last_user(messages))

        # Seeded hallucinated tool: GS-07a's very first action (recovers afterwards).
        if fixture.id == SEEDED_HALLUCINATED_TOOL_FIXTURE and turn == 0:
            if done == 0:
                return {"name": "lookup_movie", "arguments": {"name": users[0]}}
            done -= 1  # the error feedback consumed one tool slot

        if done < len(turn_calls):
            call = turn_calls[done]
            return {
                "name": call["tool"],
                "arguments": self._bind(call["arguments"], messages),
            }
        return self._answer(fixture, turn)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        body = json.loads(request.content)
        action = self._next_action(body["messages"])
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "tool_calls"
        if isinstance(action, str):
            message["content"] = action
            finish = "stop"
        else:
            message["tool_calls"] = [
                {
                    "id": f"call_{self.requests}",
                    "type": "function",
                    "function": {
                        "name": action["name"],
                        "arguments": json.dumps(action["arguments"], ensure_ascii=False),
                    },
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-mock-{self.requests}",
                "object": "chat.completion",
                "model": "mock",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            },
        )


def build_mock_client(settings: Settings) -> LLMClient:
    """The dry-run client: LLMClient over the scripted player (single MockTransport)."""
    fixtures = load_fixtures()
    model = ScriptedModel(fixtures)
    return LLMClient(
        settings,
        http_client=httpx.Client(transport=httpx.MockTransport(model)),
    )
