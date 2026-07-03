"""Emit the D3 entity-disjointness fixture list (P4 task 3 deliverable, DEC-P4-3).

Queries the *gate-visible* views (never raw tables — the layered-gate property training
data inherits, P4_SPEC §2.3) for every work/version belonging to the training slice, and
writes ``finetune/training_slice_entities.json``. Consumed by scaffold sampling (task 4)
and decontamination (task 5): training ``entity_ids`` must be a subset of these ids and
disjoint from all golden-fixture entities.

    uv run python finetune/export_training_entities.py
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import text

from sutradhar.graph.db import create_graph_engine
from sutradhar.pipeline.seed import load_seed_slice

app = typer.Typer(add_completion=False)

DEFAULT_SLICE = Path("data-pipeline/training_slice.yaml")
DEFAULT_OUT = Path("finetune/training_slice_entities.json")

_QUERY = text(
    "SELECT w.work_id::text AS wid, w.primary_title, v.version_id::text AS vid, "
    "v.title, v.language, v.release_year, v.wikidata_qid "
    "FROM ground_truth_versions v JOIN ground_truth_works w ON v.work_id = w.work_id "
    "WHERE w.work_id IN (SELECT v2.work_id FROM ground_truth_versions v2 "
    "WHERE v2.wikidata_qid = ANY(:qids)) "
    "ORDER BY w.primary_title, v.release_year, v.language"
)


@app.command()
def main(
    slice_path: Path = typer.Option(DEFAULT_SLICE, "--slice"),  # noqa: B008 — typer idiom
    out: Path = typer.Option(DEFAULT_OUT),  # noqa: B008 — typer idiom
    verified_at: str = typer.Option("2026-07-03", help="Ingestion/verification date stamp."),
) -> None:
    slice_ = load_seed_slice(slice_path)
    qids = sorted(qid for _, qid in slice_._iter_qids())  # noqa: SLF001 — same-package helper

    engine = create_graph_engine()
    with engine.connect() as conn:
        rows = conn.execute(_QUERY, {"qids": qids}).all()
    engine.dispose()

    works: dict[str, str] = {}
    versions: list[dict[str, object]] = []
    for r in rows:
        works[r.wid] = r.primary_title
        versions.append(
            {
                "version_id": r.vid,
                "work_id": r.wid,
                "title": r.title,
                "language": r.language,
                "year": r.release_year,
                "wikidata_qid": r.wikidata_qid,
            }
        )
    payload = {
        "$comment": (
            "P4 D3 entity-disjointness fixture list (DEC-P4-3): gate-visible training-slice "
            f"entities, emitted post-ingestion {verified_at}. Training entity_ids MUST be a "
            "subset of these ids and disjoint from all golden-fixture entities "
            "(tests/test_ft_decontamination, task 5)."
        ),
        "slice_config": str(slice_path),
        "verified_at": verified_at,
        "work_count": len(works),
        "version_count": len(versions),
        "works": [
            {"work_id": k, "primary_title": v}
            for k, v in sorted(works.items(), key=lambda kv: kv[1])
        ],
        "versions": versions,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}: {len(works)} works, {len(versions)} gate-visible versions")


if __name__ == "__main__":
    app()
