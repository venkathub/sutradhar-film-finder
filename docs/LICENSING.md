# Licensing & attribution

> Every data source and model Sutradhar uses, its license, how we use it, and the attribution we
> owe. **Status: updated for P4 (2026-07-04)** — updated whenever a source/model is added. This is
> a deliberate maturity signal (CLAUDE.md): license hygiene is part of the product.

## Data sources

| Source | License / terms | Our usage | Attribution / obligations |
|---|---|---|---|
| **IMDb non-commercial datasets** (`datasets.imdbws.com`, `title.akas.tsv.gz`) | **Personal / non-commercial use ONLY** (developer.imdb.com/non-commercial-datasets) | AKA/dub titles + `isOriginalTitle` corroboration, streamed and slice-filtered (`sutradhar.pipeline.imdb`); the raw dump is never stored or committed | "Information courtesy of IMDb (https://www.imdb.com). Used with permission." **This project is a non-commercial portfolio demo; IMDb-derived data must never ship in a commercial offering.** **P6 (met, executable):** the chat UI is the first surface that *displays* IMDb-derived AKA titles, so the courtesy line + the non-commercial note render in the UI footer — enforced by `tests/test_ui_attribution.py`. |
| **Wikidata** (SPARQL + `wbgetentities`) | **CC0 1.0** (public domain) | Relationship spine (P144/P4969/P155/P156), external-ID hub (P345/P4947), entity resolution | None required (CC0); we still record per-claim provenance in `sources[]` |
| **TMDB API** (v3) | Free developer API; **attribution required**; data community-editable | Multilingual titles, alternative titles, credits (`sutradhar.pipeline.tmdb`) | **"This product uses the TMDB API but is not endorsed or certified by TMDB."** TMDB logo required on any UI surface. **P6 (met, executable):** the chat UI footer renders the official TMDB logo + the exact notice, and the logo is **less prominent than Sutradhar's own mark** (an explicit TMDB FAQ condition) — enforced by `tests/test_ui_attribution.py` + a measured browser test (`Footer.test.tsx`). |
| **Wikipedia** (MediaWiki/REST API — never HTML-scraped) | **CC BY-SA 4.0** (attribution + share-alike) | Plot/synopsis prose for P2 embeddings (`plot_texts`, revision-pinned); candidate-edge extraction input (P1 task 11) | Per-article attribution: page URL + revision id stored on every `plot_texts` row; **derived text redistributions must remain CC BY-SA**. |

## Models (per DEC-0001)

| Model | License | Our usage |
|---|---|---|
| Gemma 4 E4B-it (`@ fee6332c…`, DEC-P4-9 pin correction) | Apache 2.0 | FT base + benchmark base column + P1 candidate-edge extraction (GPU sessions) |
| Qwen3-4B-Instruct-2507 | Apache 2.0 | Fallback base |
| BGE-M3 / bge-reranker-v2-m3 | MIT / Apache 2.0 | P2 embeddings / reranking |
| Sarvam-M 24B (`@ 01534a53…`) | Apache 2.0 | **P4 synthetic-data teacher (executed 2026-07-03)** — outputs unencumbered, folded into `sutradhar-ft-v1`; optional live showcase |
| ~~Sarvam-1 (2B)~~ | **non-commercial — AVOIDED** | not used (CLAUDE.md rule) |
| AI4Bharat IndicXlit | **CC BY-SA 4.0** | **Contingency only** (DEC-P1-5); if ever invoked, add share-alike attribution here |

## P4 derived artifacts (2026-07-04)

| Artifact | License / terms | Notes |
|---|---|---|
| **`sutradhar-ft-v1` dataset** (HF `venkat2393/sutradhar-ft-v1`) | Mixed — see the dataset card | **PRIVATE-first (DEC-P4-7)**: contains IMDb-derived AKA titles (non-commercial terms) and Wikipedia-plot-derived excerpts (CC BY-SA 4.0, revision-pinned); teacher surfaces are Apache-2.0 Sarvam-M outputs; graph facts CC0/TMDB-attributed. Stays private until a redaction pass removes the IMDb-derived rows. |
| **`finetune/scaffold_snapshot.json`** (committed) | CC BY-SA 4.0 applies to the plot excerpts inside | Same posture as the committed P2 retrieval artifacts: per-row provenance lives in `plot_texts` (page URL + revision id). |
| **QLoRA adapter** (HF `venkat2393/sutradhar-gemma4-e4b-qlora-v1`, PUBLIC) | Apache 2.0 | Trained on `sutradhar-ft-v1` from an Apache-2.0 base with an Apache-2.0 teacher; published as a documented negative result (DEC-P4-9). Weights contain no verbatim source text. |

## P6 distributed surfaces (2026-07-11)

| Surface | License / terms | Notes |
|---|---|---|
| **Static site** (GitHub Pages, `site/dist` from `site/generate.py`) | Repo license; embedded content per source rows above | Distributes **aggregated benchmark numbers, committed screenshots, and the architecture diagram only** — no bulk source data (no titles/plots/AKA dumps) is redistributed. Footer carries the TMDB notice, the CC BY-SA 4.0 label, and the IMDb courtesy line. Non-commercial portfolio use (inside GitHub Pages acceptable-use terms). |
| **Demo video** (GitHub Release asset, `DEMO_VIDEO_URL`) | Screen recording of the UI; displayed data per source rows above | The recorded frames show TMDB/Wikipedia/IMDb-derived facts as rendered by the attributed UI — the in-frame footer chrome (TMDB logo + notice, CC BY-SA, IMDb courtesy) satisfies the same obligations. Distributed via a Release asset (DEC-P6-3), never committed to git or the Pages site; non-commercial. |
| **UI screenshots** (`docs/evidence/p6/*.png`, committed) | Same as the demo video | Rehearsal-window captures of the attributed UI (footer chrome visible in the full-page shots). |

## Code

This repository: MIT (see `pyproject.toml`). `indic-transliteration` (Python): MIT.

| openai/gpt-oss-20b (LLM-as-judge, DEC-P3-1; rev 6cee5e81) | Apache 2.0 | Self-hosted on the ephemeral GPU for judge/RAGAS batch passes only; never serves users; weights pinned by revision SHA |
