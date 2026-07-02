"""Human review gate CLI (P1 task 12, DEC-P1-6): confirm/reject candidates → promotion.

Two modes:
- **Interactive** (default): shows edge type, raw + bound endpoints, the supporting sentence,
  page@revision, model confidence; prompts y(confirm) / n(reject) / s(skip).
- **Batch** (``--decisions file.yaml``): applies a pre-reviewed decisions file — the audit
  artifact of a review session (each entry: candidate_id, verdict, optional endpoint
  bindings). ``--reviewer`` names the human on every audit field.

Additionally ``--verify-medium`` lists rule-derived MEDIUM edges (dub tracks) and verifies
them interactively — the same gate semantics applied to the builder's derivations.

    make review-candidates
    uv run python data-pipeline/review_candidates.py --decisions decisions.yaml --reviewer you
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.graph.schema import Version
from sutradhar.pipeline.review import (
    BindingError,
    ReviewReport,
    apply_decisions,
    list_medium_rule_edges,
    list_proposed,
    load_decisions,
    promote,
    reject,
    verify_medium_edges,
)

app = typer.Typer(add_completion=False)


def _echo_report(report: ReviewReport) -> None:
    typer.echo(
        f"confirmed: {report.confirmed} (edges created: {report.edges_created}, "
        f"corroborated: {report.edges_corroborated})\n"
        f"rejected:  {report.rejected}\n"
        f"skipped:   {report.skipped}\n"
        f"candidate precision (confirmed/decided): "
        f"{report.precision if report.precision is not None else 'n/a'}"
    )
    for err in report.errors:
        typer.echo(f"  ! {err}", err=True)


@app.command()
def main(
    reviewer: str = typer.Option(..., help="Reviewer name recorded on every audit field."),
    decisions: Path | None = typer.Option(  # noqa: B008 — typer idiom
        None, help="YAML decisions file (batch mode). Omit for interactive review."
    ),
    verify_medium: bool = typer.Option(
        False, help="Also review rule-derived MEDIUM edges for human verification."
    ),
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)

    with factory() as session:
        if decisions is not None:
            payload = yaml.safe_load(decisions.read_text(encoding="utf-8"))
            report = apply_decisions(session, load_decisions(payload), reviewer)
        else:
            report = ReviewReport()
            for candidate in list_proposed(session):
                src_v = (
                    session.get(Version, candidate.src_version_id)
                    if candidate.src_version_id
                    else None
                )
                dst_v = (
                    session.get(Version, candidate.dst_version_id)
                    if candidate.dst_version_id
                    else None
                )
                typer.echo(
                    f"\n[{candidate.edge_type}] "
                    f"{candidate.src_title_raw!r} -> {candidate.dst_title_raw!r}\n"
                    f"  bound: src={src_v.title + ' (' + src_v.language + ')' if src_v else '—'}"
                    f" dst={dst_v.title + ' (' + dst_v.language + ')' if dst_v else '—'}\n"
                    f'  evidence: "{candidate.supporting_sentence}"\n'
                    f"  page: {candidate.source_page}@{candidate.source_revision} "
                    f"model_conf={candidate.model_confidence}"
                )
                choice = typer.prompt("  [y]es / [n]o / [s]kip", default="s").strip().lower()
                if choice == "y":
                    try:
                        _edge, created = promote(session, candidate, reviewer)
                        report.confirmed += 1
                        report.edges_created += int(created)
                        report.edges_corroborated += int(not created)
                    except BindingError as exc:
                        typer.echo(f"  ! {exc} — use a decisions file with bindings", err=True)
                        report.skipped += 1
                elif choice == "n":
                    reject(session, candidate, reviewer)
                    report.rejected += 1
                else:
                    report.skipped += 1

        if verify_medium:
            pending = list_medium_rule_edges(session)
            to_verify = []
            for edge in pending:
                src = session.get(Version, edge.src_id)
                dst = session.get(Version, edge.dst_id)
                typer.echo(
                    f"\n[MEDIUM {edge.edge_type}] "
                    f"{src.title + ' (' + src.language + ')' if src else edge.src_id} -> "
                    f"{dst.title + ' (' + dst.language + ')' if dst else edge.dst_id} "
                    f"(rule-derived)"
                )
                if typer.prompt("  verify? [y/n]", default="n").strip().lower() == "y":
                    to_verify.append(edge.edge_id)
            report.medium_verified = verify_medium_edges(session, reviewer, to_verify)
            typer.echo(f"medium edges verified: {report.medium_verified}")

        session.commit()
    engine.dispose()
    _echo_report(report)


if __name__ == "__main__":
    app()
