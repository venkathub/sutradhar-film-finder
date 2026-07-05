"""Injection eval runner (P5 task 11) — dry-run (scripted) + live (task-13 window).

Dry-run (Tier-1, no GPU): each fixture runs through the REAL orchestrator + guardrails
with a scripted LLM. The scripted model is a *naively compliant* attacker's dream — it
tries to echo canaries and assert the attacker's title — so the run proves the
DETERMINISTIC layers (datamarking, adversarial withholding, the output gate) drive ASR
and FP to 0 without any model cooperation. Live mode drives the same fixtures against the
window endpoint for the real defense-on/off ASR row (BENCHMARKS).

    make injection-eval                     # dry-run, defenses ON  -> committed summary
    python evals/run_injection_eval.py --defenses off   # baseline (window)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import typer

from sutradhar.config import Settings
from sutradhar.evals.injection import (
    EmittedCallView,
    InjectionFixture,
    load_injection_fixtures,
    score_injection,
    summarize,
    wrap_executor,
)
from sutradhar.evals.prompts import load_serving_prompt_artifacts
from sutradhar.serving import guardrails
from sutradhar.serving.orchestrator import Orchestrator, default_output_gate, default_spotlight
from sutradhar.serving.schemas import ChatResponse
from sutradhar.serving.sessions import InMemorySessionStore
from sutradhar.toolcalls import load_tool_schema, openai_tools

app = typer.Typer(add_completion=False)

INJECTION_RUNS_DIR = Path("evals/injection_runs")

# A tiny in-catalog fixture graph the scripted attacker path resolves against.
WORK_ID = str(uuid.uuid4())
V_ML, V_TA = str(uuid.uuid4()), str(uuid.uuid4())
GROUNDED_TITLES = ["Drishyam", "Papanasam"]

_BASE_RESULTS: dict[str, dict[str, Any]] = {
    "resolve_title": {
        "candidates": [
            {
                "work_id": WORK_ID,
                "version_id": V_TA,
                "matched_title": "Papanasam",
                "language": "ta",
                "year": 2015,
                "score": 1.0,
                "sources": [],
            }
        ],
        "ambiguous": False,
    },
    "get_work": {
        "work_id": WORK_ID,
        "canonical_title": "Drishyam",
        "work_type": "film",
        "original_language": "ml",
        "first_release_year": 2013,
        "source_work": None,
        "based_on": [],
        "sources": [{"source": "wikidata", "ref": "Q15401703"}],
        "confidence": "HIGH",
        "overview": "A man hides evidence to protect his family.",
    },
    "get_versions": {
        "original": {
            "version_id": V_ML,
            "title": "Drishyam",
            "language": "ml",
            "year": 2013,
            "cast_lead": ["Mohanlal"],
            "relationship": "is_original_of",
            "is_original": True,
            "sources": [{"source": "wikidata", "ref": "Q15401703"}],
            "confidence": "HIGH",
        },
        "versions": [
            {
                "version_id": V_TA,
                "title": "Papanasam",
                "language": "ta",
                "year": 2015,
                "cast_lead": ["Kamal Haasan"],
                "relationship": "is_remake_of",
                "is_original": False,
                "sources": [{"source": "wikidata", "ref": "Q18578149"}],
                "confidence": "HIGH",
            }
        ],
    },
}


def _base_executor(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool in _BASE_RESULTS:
        return {**_BASE_RESULTS[tool]}
    from sutradhar.evals.driver import ToolExecutionError

    raise ToolExecutionError(f"scripted executor has no result for {tool!r}")


ATTACKER_TITLE = "Zzyzx Road"  # out-of-catalog: the attacker's desired ungrounded assertion


class _CompliantAttackerModel:
    """A FAITHFUL worst-case victim: it obeys anything it *actually reads*. It echoes only
    canaries present in the messages it received (so spotlight-withholding is visible: a
    stripped canary can't be echoed) and always tries to assert the attacker's ungrounded
    title (so the output gate's effect is visible). What the defenses remove, this model
    cannot repeat — that is exactly what makes the dry-run's deterministic gate meaningful.

    Round 1 pulls the tool carrying a context payload so the model 'reads' it."""

    def __init__(self, fixture: InjectionFixture) -> None:
        self._fixture = fixture
        self._turn = 0

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        from sutradhar.serving.llm_client import ChatResult, ToolCall

        self._turn += 1
        target = (self._fixture.payload_target or "get_versions").split(":")[0]
        context_attack = self._fixture.attack_class in {"context_tool_result", "exfiltration"}
        if context_attack and self._turn == 1:
            args = (
                {"work_id": WORK_ID}
                if target in {"get_work", "get_versions"}
                else {"title": "Papanasam"}
            )
            return ChatResult(
                status="up",
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": target, "arguments": json.dumps(args)},
                        }
                    ],
                },
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name=target, arguments_raw=json.dumps(args), arguments=args),
                ),
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=12.0,
                detail="ok",
            )
        # Final round: echo the canary ONLY if it survived into TOOL content the model
        # received (the indirect-injection surface the datamarking/withholding defends).
        # A canary in the user's own query is not an indirect injection — the output gate
        # + read-only tools are what neutralize query-side attacks (title/redirect).
        tool_content = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "tool"
        )
        echoed = (
            f" {self._fixture.canary}"
            if self._fixture.canary and self._fixture.canary in tool_content
            else ""
        )
        answer = (
            'INTENT: {"intent": "list_versions", "slots": {}}\n'
            f"The original is **{ATTACKER_TITLE}** (2006).{echoed}"
        )
        return ChatResult(
            status="up",
            message={"role": "assistant", "content": answer},
            content=answer,
            tool_calls=(),
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=12.0,
            detail="ok",
        )


def _drive(
    fixture: InjectionFixture, model: Any, defenses: bool
) -> tuple[str, list[EmittedCallView], list[str]]:
    """Run one fixture through the real orchestrator + guardrails against ``model``.

    ``model`` is duck-typed on ``.chat`` — the scripted worst-case attacker (dry-run) or a
    real ``LLMClient`` (live window). The wrapper executor splices the context payload into
    the base graph result; the live graph is never touched (§2.9)."""
    artifacts = load_serving_prompt_artifacts()
    schema = load_tool_schema()
    store = InMemorySessionStore(3600)
    executor = wrap_executor(_base_executor, fixture)
    orchestrator = Orchestrator(
        model,
        store,
        executor,
        system_prompt=artifacts.system_prompt(),
        prompt_hash=artifacts.prompt_hash,
        schema=schema,
        spotlight=guardrails.spotlight if defenses else default_spotlight,
        output_gate=guardrails.output_gate if defenses else default_output_gate,
    )
    turns = fixture.query if isinstance(fixture.query, list) else [fixture.query]
    emitted: list[EmittedCallView] = []
    answer = ""
    conversation_id: str | None = None
    for turn in turns:
        outcome = orchestrator.run_turn(conversation_id, turn)
        if isinstance(outcome, ChatResponse):
            conversation_id = outcome.conversation_id
            answer = outcome.answer
    # Recover emitted calls from stored wire history (tool messages carry no call, so read
    # the assistant tool_calls) — enough for must_not_call scoring.
    state = store.load(conversation_id) if conversation_id else None
    if state is not None:
        for msg in state.messages:
            for call in msg.get("tool_calls", []) or []:
                fn = call.get("function", {})
                raw = fn.get("arguments", "{}")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                emitted.append(EmittedCallView(tool=fn.get("name", ""), arguments=parsed))
    return answer, emitted, GROUNDED_TITLES


def run_fixture_dry(
    fixture: InjectionFixture, defenses: bool
) -> tuple[str, list[EmittedCallView], list[str]]:
    """Dry-run: the worst-case compliant-attacker scripted model (no GPU)."""
    return _drive(fixture, _CompliantAttackerModel(fixture), defenses)


def run_fixture_live(
    fixture: InjectionFixture, defenses: bool
) -> tuple[str, list[EmittedCallView], list[str]]:
    """Live window: the real LLM at LLM_BASE_URL (task 13). Utility-under-attack is
    meaningful here because a real model actually attempts the legitimate task."""
    from sutradhar.serving.llm_client import LLMClient

    return _drive(fixture, LLMClient(Settings()), defenses)


def run_injection_suite(mode: str, defenses_on: bool) -> dict[str, Any]:
    """Score every fixture in ``mode`` (dry_run|live) with defenses on/off → summary dict."""
    live = mode == "live"
    fixtures = load_injection_fixtures()
    verdicts = []
    for fixture in fixtures:
        driver = run_fixture_live if live else run_fixture_dry
        answer, emitted, grounded = driver(fixture, defenses_on)
        verdicts.append(
            score_injection(
                fixture,
                answer=answer,
                emitted_calls=emitted,
                grounded_titles=grounded,
                defenses=defenses_on,
                withheld=_content_withheld(fixture, defenses_on),
                # Dry-run's worst-case model refuses the legit task by design, so utility is
                # a LIVE-only metric (a real model actually attempts it).
                score_utility=live,
            )
        )
    return {"mode": mode, **summarize(defenses_on, verdicts)}


@app.command()
def main(
    defenses: str = typer.Option("on", "--defenses", help="on|off (off = window baseline)"),
    mode: str = typer.Option("dry_run", "--mode", help="dry_run|live (live needs LLM_BASE_URL)"),
    out_dir: Path = typer.Option(INJECTION_RUNS_DIR, "--out-dir"),  # noqa: B008
    write: bool = typer.Option(True, "--write/--no-write"),
) -> None:
    defenses_on = defenses == "on"
    summary = run_injection_suite(mode, defenses_on)
    typer.echo(
        f"injection eval [{mode}] (defenses={'ON' if defenses_on else 'OFF'}): "
        f"ASR={summary['asr']} FP={summary['false_positive_rate']} "
        f"utility={summary['utility_under_attack']} "
        f"(attacks={summary['n_attacks']}, benign={summary['n_benign']})"
    )
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "dryrun" if mode == "dry_run" else "live"
        run_id = f"inj-{'on' if defenses_on else 'off'}-{suffix}"
        stamp = {
            "run_id": run_id,
            "prompt_hash": load_serving_prompt_artifacts().prompt_hash,
            "tools": [t["function"]["name"] for t in openai_tools(load_tool_schema())],
            **summary,
        }
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(stamp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        typer.echo(f"wrote {path}")


def _content_withheld(fixture: InjectionFixture, defenses: bool) -> bool:
    """Did the pattern layer withhold this fixture's payload? (FP source on benign.)"""
    if not defenses:
        return False
    text = fixture.payload or (
        fixture.query if isinstance(fixture.query, str) else " ".join(fixture.query)
    )
    return bool(guardrails.adversarial_flags(text))


if __name__ == "__main__":
    app()
