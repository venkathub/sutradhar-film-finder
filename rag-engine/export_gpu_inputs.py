"""Export GPU-job inputs (P2 task 5): corpus texts per config + all eval queries.

uv run python rag-engine/export_gpu_inputs.py [--out data/interim/gpu_inputs.json]
"""

from __future__ import annotations

from pathlib import Path

import typer

from sutradhar.graph.db import create_graph_engine, create_session_factory
from sutradhar.rag.gpu_jobs import write_gpu_inputs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out: Path = typer.Option(Path("data/interim/gpu_inputs.json"), "--out"),  # noqa: B008
) -> None:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        path, inputs = write_gpu_inputs(session, out)
    engine.dispose()
    typer.echo(
        f"wrote {path} — {len(inputs['queries'])} queries, "
        + ", ".join(f"{k}: {len(v)} chunks" for k, v in sorted(inputs["configs"].items()))
    )


if __name__ == "__main__":
    app()
