"""Extract candidate edges from Wikipedia prose via the on-demand LLM (P1_SPEC §2.3 step 6).

The ONE P1 GPU-session job (DEC-P1-4: Gemma 4 E4B on the ephemeral A100, reached through the
env-driven ``LLM_BASE_URL``). Artifact-first: raw prompts+outputs land under
``data/raw/extraction/<run>/`` before any DB write; ``--offline`` replays the latest artifact
(what CI and rebuilds use — no model call).

    uv run python data-pipeline/extract_candidates.py            # live (GPU endpoint must be UP)
    uv run python data-pipeline/extract_candidates.py --offline  # replay latest artifact
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from sutradhar.config import get_settings
from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.pipeline.extract import collect_pages, load_candidates, run_extraction
from sutradhar.pipeline.snapshots import latest_snapshot_dir, load_snapshot, write_snapshot
from sutradhar.serving.llm_client import LLMClient

app = typer.Typer(add_completion=False)

ARTIFACT_ROOT = Path("data/raw/extraction")


@app.command()
def main(
    offline: bool = typer.Option(  # noqa: B008 — typer idiom
        False, help="Replay the latest recorded artifact; no model call."
    ),
    artifact_root: Path = typer.Option(  # noqa: B008 — typer idiom
        ARTIFACT_ROOT, help="Artifact base directory."
    ),
) -> None:
    settings = get_settings()
    engine = create_graph_engine()
    factory = create_session_factory(engine)

    with factory() as session:
        pages = collect_pages(session)
        if not pages:
            typer.echo("no plot_texts — run fetch-plots first", err=True)
            raise typer.Exit(1)

        if offline:
            run_dir = latest_snapshot_dir(artifact_root)
            artifact = load_snapshot(run_dir, "outputs")
            raw_outputs = dict(artifact["raw_outputs"])
            model_id = str(artifact["model_id"])
            pages = dict(artifact["pages"])  # replay against the exact recorded inputs
            typer.echo(f"replaying artifact {run_dir} ({len(raw_outputs)} outputs)")
            report = None
        else:
            client = LLMClient(settings)
            health = client.health()
            if health.status != "up":
                typer.echo(f"LLM endpoint not up ({health.detail})", err=True)
                raise typer.Exit(1)
            model_id = settings.llm_model
            typer.echo(f"extracting from {len(pages)} pages via {model_id} …")
            raw_outputs, report = run_extraction(client, pages)
            run_dir = artifact_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            write_snapshot(
                run_dir,
                "outputs",
                {"model_id": model_id, "pages": pages, "raw_outputs": raw_outputs},
            )
            typer.echo(f"artifact written to {run_dir}")

        final = load_candidates(session, raw_outputs, pages, model_id, report)
        session.commit()
    engine.dispose()

    typer.echo(
        f"pages processed:      {final.pages_processed}\n"
        f"malformed responses:  {final.responses_malformed} "
        f"(parse-failure rate {final.parse_failure_rate:.1%})\n"
        f"proposals:            {final.proposals_raw} "
        f"(unsupported dropped: {final.proposals_unsupported}, "
        f"duplicates: {final.proposals_duplicate})\n"
        f"candidates written:   {final.candidates_written} "
        f"(both ends resolved: {final.resolved_both_ends}, raw kept: {final.kept_raw_titles})\n"
        f"extraction run:       {final.run_hash}"
    )
    if final.errors:
        typer.echo("page errors:", err=True)
        for err in final.errors:
            typer.echo(f"  - {err}", err=True)


if __name__ == "__main__":
    app()
