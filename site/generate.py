"""Static always-available surface generator (P6 task 9, DEC-P6-3: GitHub Pages).

Renders ``site/dist/`` from committed evidence — the surface an interviewer can
always reach even when every server is off:

- ``index.html``     — the gating story in one screen + architecture + evidence links;
- ``benchmarks.html`` — ``docs/BENCHMARKS.md`` converted verbatim (SINGLE source of
  truth: this file contains **no metric literal**; a number on the page exists only
  because it exists in BENCHMARKS.md);
- ``assets/``        — the committed architecture diagram + MLflow evidence screenshots.

Static + precomputed only — no server code, never a neural model (CLAUDE.md).
Config via env (Settings): ``DEMO_VIDEO_URL`` (link rendered only when set — no dead
links), ``SITE_BASE_URL`` / ``REPO_URL`` for canonical/self links.

Run: ``make site-build`` → open ``site/dist/index.html``.
"""

from __future__ import annotations

import shutil
from html import escape
from pathlib import Path

import markdown

from sutradhar.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST = REPO_ROOT / "site" / "dist"
BENCHMARKS_MD = REPO_ROOT / "docs" / "BENCHMARKS.md"
DEFAULT_REPO_URL = "https://github.com/venkathub/sutradhar-film-finder"

# Committed evidence copied into the site (a missing file fails the build — the
# required-assets gate; a dead evidence link must be a CI failure, not a surprise).
EVIDENCE_ASSETS = {
    "architecture.svg": REPO_ROOT / "docs" / "assets" / "architecture.svg",
    "p3-mlflow-generation-dryrun.png": REPO_ROOT
    / "docs"
    / "assets"
    / "p3-mlflow-generation-dryrun.png",
    "mlflow_qlora_run.png": REPO_ROOT / "docs" / "evidence" / "p4" / "mlflow_qlora_run.png",
    "mlflow_registry_adapter.png": REPO_ROOT
    / "docs"
    / "evidence"
    / "p4"
    / "mlflow_registry_adapter.png",
}

CSS = """\
:root { --ink:#1d2733; --soft:#5b6b7c; --paper:#f7f8fa; --card:#fff; --accent:#b3541e;
        --line:#e3e7ec; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink);
       background:var(--paper); line-height:1.55; }
main { max-width:60rem; margin:0 auto; padding:1.5rem 1rem 4rem; }
h1 { color:var(--accent); margin-bottom:0.2rem; }
h2 { border-bottom:2px solid var(--line); padding-bottom:0.3rem; margin-top:2.2rem; }
table { border-collapse:collapse; width:100%; font-size:0.9rem; display:block; overflow-x:auto; }
th, td { border:1px solid var(--line); padding:0.4rem 0.6rem; text-align:left; }
th { background:var(--card); }
code { background:#eef2f6; padding:0.05rem 0.3rem; border-radius:4px; font-size:0.9em; }
pre code { display:block; padding:0.8rem; overflow-x:auto; }
blockquote { border-left:4px solid var(--accent); margin:1rem 0; padding:0.3rem 1rem;
             background:var(--card); color:var(--soft); }
img.diagram { width:100%; height:auto; border:1px solid var(--line); border-radius:8px;
              background:#fff; }
img.shot { max-width:100%; border:1px solid var(--line); border-radius:8px; }
.tagline { color:var(--soft); font-size:1.05rem; margin-top:0; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); gap:0.8rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:0.9rem 1rem; }
.card h3 { margin:0 0 0.4rem; font-size:1rem; }
.card p { margin:0; color:var(--soft); font-size:0.9rem; }
footer { border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem;
         color:var(--soft); font-size:0.8rem; }
nav a { margin-right:1rem; }
"""


def _page(title: str, body: str, *, site_base: str) -> str:
    canonical = f'\n  <link rel="canonical" href="{escape(site_base)}"/>' if site_base else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>{canonical}
  <link rel="stylesheet" href="style.css"/>
</head>
<body>
<main>
<nav><a href="index.html">Sutradhar</a><a href="benchmarks.html">Benchmark report</a></nav>
{body}
<footer>
  <p>Sutradhar is a non-commercial portfolio project. This site is static, generated from
  committed evidence — it never serves a model; the live demo runs on an on-demand GPU only.</p>
  <p>This product uses the TMDB API but is not endorsed or certified by TMDB. Plot text and
  remake evidence from Wikipedia (CC BY-SA 4.0, revision-pinned). Information courtesy of
  IMDb (https://www.imdb.com). Used with permission.</p>
</footer>
</main>
</body>
</html>
"""


def _landing_body(*, repo_url: str, demo_video_url: str | None) -> str:
    video_li = (
        f'<li><a href="{escape(demo_video_url)}">Recorded demo video</a> — the zero-GPU replay '
        "AND a live GPU turn, with the instance stopped on camera.</li>"
        if demo_video_url
        else ""
    )
    return f"""
<h1>Sutradhar</h1>
<p class="tagline">Find an Indian film from its story, plot, or cast — every language version,
the original flagged, every claim cited.</p>

<h2>The problem it proves</h2>
<p>The same story exists as <em>separate films</em> across languages (remakes) and as the
<em>same film with replaced audio</em> (dubs) — different relationships that must never be
conflated. Ask Sutradhar about <strong>Papanasam</strong> (Tamil) and it returns the Malayalam
original <strong>Drishyam</strong> (2013, Mohanlal) plus every remake —
<strong>Drishya</strong> (Kannada), <strong>Drushyam</strong> (Telugu),
<strong>Papanasam</strong> (Tamil, Kamal Haasan), <strong>Drishyam</strong> (Hindi, Ajay Devgn)
— each labelled with its typed relationship and grounded in a cited source. Queries arrive
code-mixed (Hinglish/Tanglish) and in native scripts; the graph answers across all of them.</p>

<h2>Architecture</h2>
<img class="diagram" src="assets/architecture.svg"
     alt="Sutradhar architecture: chat UI, FastAPI orchestration with guardrails, agent loop
over five schema-validated tools, Postgres+pgvector remake graph, hybrid retrieval, an
on-demand GPU (off by default) and the evals/observability rail."/>

<h2>Engineering posture (the part that is the point)</h2>
<div class="cards">
  <div class="card"><h3>Nothing inference-side runs 24/7</h3>
    <p>The GPU is rented per-minute, brought up to capture benchmarks and for live interview
    demos, then destroyed. The standing proof is this evidence, not a warm server.</p></div>
  <div class="card"><h3>Two tables, kept honest</h3>
    <p>Retrieval quality is model-independent and never presented as "before/after
    fine-tuning". The QLoRA fine-tune was measured, lost to the well-prompted base, and CUT
    under a pre-committed rule — a documented negative result.</p></div>
  <div class="card"><h3>Grounded or silent</h3>
    <p>Every claim carries provenance (Wikidata / TMDB / IMDb / revision-pinned Wikipedia /
    named rule / human gate). A deterministic output gate keeps the hallucinated-movie rate
    at zero on the served path; out-of-catalog queries abstain.</p></div>
  <div class="card"><h3>Eval-gated CI</h3>
    <p>Golden regressions (the Drishyam franchise, dub-vs-remake, false-merge traps,
    multi-turn backtracking) gate every merge — from the repository layer to the rendered
    DOM.</p></div>
</div>

<h2>Evidence</h2>
<ul>
  <li><a href="benchmarks.html">Benchmark report</a> — generated from the repository's
  <code>docs/BENCHMARKS.md</code>; numbers are never hand-copied to this site.</li>
  {video_li}
  <li><a href="{escape(repo_url)}">Source repository</a> — specs, decision log (every
  architectural choice dated and argued), runbook, and the full eval harness.</li>
  <li><a href="{escape(repo_url)}/blob/main/docs/RUNBOOK.md">Runbook</a> — the three
  rehearsed demo paths (zero-GPU, timed live GPU window, rebuild-from-scratch) with
  measured timings, costs, and the verified-teardown discipline.</li>
  <li>MLflow evidence (committed screenshots):
    <a href="assets/p3-mlflow-generation-dryrun.png">generation eval runs</a> ·
    <a href="assets/mlflow_qlora_run.png">the QLoRA training run</a> ·
    <a href="assets/mlflow_registry_adapter.png">model registry</a>.</li>
</ul>

<h2>Try it</h2>
<p>Zero-GPU, from a fresh clone: <code>make demo-up</code> → the chat UI comes up in
degradation mode and replays the recorded Papanasam story — citations, version cards and
the tool-call trace included — from pinned benchmark transcripts with real recorded GPU
latencies. The live experience is one <code>make gpu-serve</code> away.</p>
"""


def _benchmarks_body() -> str:
    md_text = BENCHMARKS_MD.read_text(encoding="utf-8")
    converted = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return (
        "<p><em>Generated verbatim from "
        "<code>docs/BENCHMARKS.md</code> — the single source of truth.</em></p>\n" + converted
    )


def build(dist: Path = DIST) -> Path:
    settings = Settings(_env_file=None)
    repo_url = DEFAULT_REPO_URL
    site_base = settings.site_base_url or ""
    demo_video_url = settings.demo_video_url

    if dist.exists():
        shutil.rmtree(dist)
    (dist / "assets").mkdir(parents=True)

    (dist / "style.css").write_text(CSS, encoding="utf-8")
    for name, source in EVIDENCE_ASSETS.items():
        if not source.exists():
            raise FileNotFoundError(f"required evidence asset missing: {source}")
        shutil.copyfile(source, dist / "assets" / name)

    (dist / "index.html").write_text(
        _page(
            "Sutradhar — cross-lingual film finder (portfolio evidence)",
            _landing_body(repo_url=repo_url, demo_video_url=demo_video_url),
            site_base=site_base,
        ),
        encoding="utf-8",
    )
    (dist / "benchmarks.html").write_text(
        _page(
            "Sutradhar — benchmark report",
            _benchmarks_body(),
            site_base=(site_base.rstrip("/") + "/benchmarks.html") if site_base else "",
        ),
        encoding="utf-8",
    )
    return dist


if __name__ == "__main__":
    out = build()
    pages = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({len(pages)} files): " + ", ".join(str(p) for p in pages))
