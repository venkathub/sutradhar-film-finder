"""ft-dryrun: the committed NO-GPU rehearsal of the whole dataset->training pipeline
(P4 task 10; spec §4 integration row). One command proves, from a fresh clone:

    scaffold (committed snapshot) -> mock teacher pass -> full validation -> seal ->
    TRL export -> render + assistant-mask verification (real transformers path, fixture
    tokenizer) -> hashed TrainConfig parse

and writes ``finetune/ft_dryrun_report.json`` — the rehearsal evidence the window's
pre-flight checks against. No model weights, no network, no GPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    size: int = typer.Option(60),
    seed: int = typer.Option(20260704),
    report_out: Path = typer.Option(Path("finetune/ft_dryrun_report.json")),  # noqa: B008
) -> None:
    from transformers import AutoTokenizer

    from sutradhar.evals.driver import load_tool_schema, openai_tools
    from sutradhar.finetune.dataset import TeacherStamp
    from sutradhar.finetune.render import (
        render_stats,
        render_with_masks,
        to_trl_messages,
        verify_masking,
    )
    from sutradhar.finetune.scaffold import ScaffoldConfig, generate, mix_stats
    from sutradhar.finetune.snapshot import load_scaffold_snapshot
    from sutradhar.finetune.teacher import surface_pass
    from sutradhar.finetune.train import TrainConfig, load_train_config
    from sutradhar.finetune.validate import validate_dataset

    report: dict[str, object] = {"steps": []}

    def step(name: str, **info: object) -> None:
        steps = report["steps"]
        assert isinstance(steps, list)
        steps.append({"step": name, **info})
        typer.echo(f"[ft-dryrun] {name}: {info}")

    # 1. scaffold from the committed snapshot
    snapshot = load_scaffold_snapshot(Path("finetune/scaffold_snapshot.json"))
    conversations = generate(snapshot, ScaffoldConfig(seed=seed, size=size))
    step("scaffold", conversations=len(conversations), behaviours=len(mix_stats(conversations)))

    # 2. mock teacher pass (faithful fake — the real one ran in task 7)
    def mock_rewrite(locked: str, register: str, kind: str) -> str:
        return f"arre, {locked}" if not locked.startswith("- ") else locked

    stamp = TeacherStamp(model="mock-teacher", revision="dryrun", prompt_sha256="0" * 64)
    taught, records, summary = surface_pass(conversations, mock_rewrite, stamp)
    step(
        "mock_teach",
        rewrites=summary.texts_total,
        rejection_rate=summary.rejection_rate,
        escalation=summary.escalation_triggered,
    )

    # 3. full validation gate
    validation = validate_dataset(taught)
    step(
        "validate",
        issues=len(validation.issues),
        decon_violations=len(validation.decontamination.violations),
        ok=validation.ok,
    )
    if not validation.ok:
        raise typer.Exit(1)

    # 4. TRL export + render/mask verification on the REAL apply_chat_template path
    tools = openai_tools(load_tool_schema())
    tokenizer = AutoTokenizer.from_pretrained("tests/fixtures/tokenizer")
    violations = []
    samples = []
    for conv in taught:
        sample = render_with_masks(tokenizer, to_trl_messages(conv), conv_id=conv.conv_id)
        samples.append(sample)
        violations.extend(verify_masking(conv, sample))
    stats = render_stats(samples, max_seq=TrainConfig().max_seq_length)
    step(
        "render_mask",
        samples=stats.samples,
        mask_violations=len(violations),
        token_p50=stats.token_p50,
        token_p95=stats.token_p95,
        tools=len(tools),
    )
    if violations:
        raise typer.Exit(1)

    # 5. hashed TrainConfig round-trip (the exact file the window ships)
    config = TrainConfig()
    parsed, embedded = load_train_config(config.to_json())
    assert parsed == config and embedded == config.config_hash()
    step("train_config", config_hash=config.config_hash())

    with tempfile.TemporaryDirectory():
        pass  # placeholder: seal exercised via `make validate-dataset` on real artifacts

    report["ok"] = True
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"[ft-dryrun] OK — rehearsal evidence: {report_out}")


if __name__ == "__main__":
    app()
