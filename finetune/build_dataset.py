"""P4 dataset-build CLI (P4_SPEC §2.9): scaffold → (teach → validate → seal, later tasks).

Subcommands grow task by task:
- ``snapshot``  (task 4) — export gate-view recordings to ``finetune/scaffold_snapshot.json``
- ``scaffold``  (task 4) — pure generation from the committed snapshot (no DB needed)

Compute placement: ``snapshot`` needs the local Postgres (gate views); ``scaffold`` is
string math over the committed file — CI-safe, laptop-safe, model-free.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import typer
import yaml
from sqlalchemy import select

from sutradhar.finetune.dataset import write_jsonl
from sutradhar.finetune.scaffold import ScaffoldConfig, generate, mix_stats
from sutradhar.finetune.snapshot import (
    SNAPSHOT_PATH,
    DecoyTheme,
    PlotExcerpt,
    ScaffoldSnapshot,
    WorkSnapshot,
    load_scaffold_snapshot,
    title_perturbations,
    write_scaffold_snapshot,
)
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.pipeline.seed import load_seed_slice

app = typer.Typer(add_completion=False)

TOOL_SCHEMA_PATH = Path("docs/phases/tool_schema.v0.json")
DEFAULT_SLICE = Path("data-pipeline/training_slice.yaml")
DEFAULT_DECOYS = Path("data-pipeline/training_decoys.yaml")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Lead/reception metadata sentences are not plot prose — a user describing a story never
# says "directed and co-written by Atlee" or quotes box-office records. The exporter scans
# for the first run of consecutive narrative sentences (the article's Plot section).
_META_SENTENCE_RE = re.compile(
    r"directed|produced|co-written|written by|starring|distributed|box office"
    r"|is a \d{4}|language film|film stars|screenplay|award|grossing|theatrical"
    r"|soundtrack|principal photography|filming|remake|remade|released|collaboration"
    r"|budget|sequel to|premiered|film adaptation|\bstars\b|\bcast\b|supporting role"
    r"|guest appearance|lead role|ensemble|reboot|cinematography|editing|comeback"
    r"|commercial cinema|status as a star|version had|conceived|based the film"
    r"|homage|success|making of|shot at|shot in|\bfilmed\b|notable for",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
# Plot sections are written in the narrative present; production/reception prose is past.
_PRESENT_TENSE_RE = re.compile(
    r"\b(is|are|has|works|follows|revolves|lives|moves|meets|finds|discovers|decides"
    r"|refuses|begins|tries|falls|helps|kills|saves|returns|realises|realizes|becomes"
    r"|leads|joins|plans|escapes|hides|leaves|takes|goes|comes|sees|learns)\b"
)


def _name_listy(sentence: str) -> bool:
    """Cast-list fragments: mostly-capitalized comma runs with no narrative content."""
    words = [w for w in re.split(r"[\s,]+", sentence) if w]
    caps = sum(1 for w in words if w[:1].isupper())
    return len(words) > 0 and caps / len(words) > 0.5


def _excerpts(texts: list[tuple[str | None, str]], limit: int = 2) -> list[PlotExcerpt]:
    """Story-shaped excerpts: the first run of >=2 consecutive narrative sentences."""
    ordered = sorted(texts, key=lambda t: (t[0] != "en", t[1]))
    out: list[PlotExcerpt] = []
    for lang, text in ordered:
        if lang != "en" or len(text) < 60:
            continue
        sentences = _SENTENCE_SPLIT.split(_PAREN_RE.sub("", text.strip()))
        clean = [
            len(s) > 40
            and not _META_SENTENCE_RE.search(s)
            and not _name_listy(s)
            and _PRESENT_TENSE_RE.search(s) is not None
            for s in sentences
        ]
        excerpt = ""
        for i in range(len(sentences) - 1):
            if clean[i] and clean[i + 1]:
                excerpt = " ".join(sentences[i : i + 2])[:320].strip()
                break
        if len(excerpt) >= 60:
            out.append(PlotExcerpt(language=lang or "en", excerpt=excerpt))
        if len(out) >= limit:
            break
    return out


@app.command()
def snapshot(
    slice_path: Path = typer.Option(DEFAULT_SLICE, "--slice"),  # noqa: B008 — typer idiom
    decoys_path: Path = typer.Option(DEFAULT_DECOYS, "--decoys"),  # noqa: B008 — typer idiom
    out: Path = typer.Option(SNAPSHOT_PATH),  # noqa: B008 — typer idiom
) -> None:
    """Export gate-view tool-result recordings for the scaffold generator."""
    from sutradhar.graph import repository
    from sutradhar.graph.schema import PlotText, Version

    slice_ = load_seed_slice(slice_path)
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    works_out: list[WorkSnapshot] = []
    with factory() as session:
        for wkey in sorted(slice_.works):
            seed_work = slice_.works[wkey]
            qids = [v.wikidata_qid for v in seed_work.versions.values() if v.wikidata_qid]
            work_id: uuid.UUID | None = None
            for qid in qids:
                row = session.execute(
                    select(Version.work_id).where(Version.wikidata_qid == qid)
                ).first()
                if row is not None:
                    work_id = row.work_id
                    break
            if work_id is None:
                typer.echo(f"  ! {wkey}: no gate-visible version — skipped", err=True)
                continue
            gw = repository.get_work(session, work_id)
            if gw is None:
                typer.echo(f"  ! {wkey}: work not gate-visible — skipped", err=True)
                continue
            variants = {"indian": repository.get_versions(session, work_id, scope="indian")}
            same_franchise = [
                k for k, w in slice_.works.items() if w.franchise == seed_work.franchise
            ]
            if len(same_franchise) > 1:
                variants["indian_sequels"] = repository.get_versions(
                    session, work_id, scope="indian", include_sequels=True
                )
            queries = {seed_work.primary_title}
            queries.update(v.title for v in seed_work.versions.values())
            for title in sorted(queries.copy()):
                queries.update(title_perturbations(title))
            resolved: dict[str, dict[str, Any]] = {}
            for q in sorted(queries):
                result = repository.resolve_title(session, q)
                if result.candidates:
                    resolved[q] = result.model_dump(mode="json")
            version_ids = [uuid.UUID(str(e.version_id)) for e in variants["indian"].versions]
            texts = session.execute(
                select(PlotText.language, PlotText.text).where(PlotText.version_id.in_(version_ids))
            ).all()
            works_out.append(
                WorkSnapshot(
                    work_key=wkey,
                    franchise=seed_work.franchise,
                    work_id=str(work_id),
                    canonical_title=gw.canonical_title,
                    original_language=gw.original_language,
                    get_work=gw.model_dump(mode="json"),
                    get_versions={k: v.model_dump(mode="json") for k, v in variants.items()},
                    resolve_title=resolved,
                    plot_excerpts=_excerpts([(r.language, r.text) for r in texts]),
                )
            )
    engine.dispose()

    decoys_raw = yaml.safe_load(decoys_path.read_text(encoding="utf-8"))
    decoys = [DecoyTheme(**d) for d in decoys_raw["decoy_themes"]]
    snap = ScaffoldSnapshot(
        slice_config=str(slice_path),
        tool_schema_sha256=hashlib.sha256(TOOL_SCHEMA_PATH.read_bytes()).hexdigest(),
        works=works_out,
        decoy_themes=decoys,
    )
    sha = write_scaffold_snapshot(out, snap)
    typer.echo(f"wrote {out} ({len(works_out)} works) sha256={sha}")


@app.command()
def scaffold(
    snapshot_path: Path = typer.Option(SNAPSHOT_PATH, "--snapshot"),  # noqa: B008 — typer idiom
    out: Path = typer.Option(  # noqa: B008 — typer idiom
        Path("data/artifacts/finetune/scaffolds.jsonl")
    ),
    seed: int = typer.Option(42),
    size: int = typer.Option(2000),
) -> None:
    """Generate the scaffold-only dataset (pure; no DB, no models)."""
    snap = load_scaffold_snapshot(snapshot_path)
    config = ScaffoldConfig(seed=seed, size=size)
    conversations = generate(snap, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = write_jsonl(out, conversations)
    stats = mix_stats(conversations)
    typer.echo(f"wrote {out} ({len(conversations)} conversations) sha256={sha}")
    typer.echo(json.dumps({b: sum(v.values()) for b, v in sorted(stats.items())}, indent=2))


@app.command()
def validate(
    dataset: Path = typer.Option(  # noqa: B008 — typer idiom
        Path("data/artifacts/finetune/scaffolds.jsonl"), "--dataset"
    ),
    report_out: Path = typer.Option(  # noqa: B008 — typer idiom
        Path("data/artifacts/finetune/validation_report.json")
    ),
) -> None:
    """Run every validation layer over a dataset JSONL (task 5; exit 1 on any issue)."""
    from sutradhar.finetune.dataset import read_jsonl
    from sutradhar.finetune.validate import validate_dataset

    conversations = read_jsonl(dataset)
    report = validate_dataset(conversations)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decon = report.decontamination
    typer.echo(
        f"conversations: {report.conversations}\n"
        f"issues:        {len(report.issues)}\n"
        f"decontamination max similarity: golden={decon.max_similarity_golden} "
        f"exemplars={decon.max_similarity_exemplars} negatives={decon.max_similarity_negatives} "
        f"(threshold {decon.threshold}; violations: {len(decon.violations)})\n"
        f"report: {report_out}"
    )
    for issue in report.issues[:20]:
        typer.echo(f"  ! {issue.conv_id} [{issue.kind}] {issue.detail}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("OK — dataset passes all validation layers")


if __name__ == "__main__":
    app()
