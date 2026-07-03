# Licensing & attribution

> Every data source and model Sutradhar uses, its license, how we use it, and the attribution we
> owe. **Status: FINAL for P1 (2026-07-02)** — updated whenever a source/model is added. This is
> a deliberate maturity signal (CLAUDE.md): license hygiene is part of the product.

## Data sources

| Source | License / terms | Our usage | Attribution / obligations |
|---|---|---|---|
| **IMDb non-commercial datasets** (`datasets.imdbws.com`, `title.akas.tsv.gz`) | **Personal / non-commercial use ONLY** (developer.imdb.com/non-commercial-datasets) | AKA/dub titles + `isOriginalTitle` corroboration, streamed and slice-filtered (`sutradhar.pipeline.imdb`); the raw dump is never stored or committed | "Information courtesy of IMDb (https://www.imdb.com). Used with permission." **This project is a non-commercial portfolio demo; IMDb-derived data must never ship in a commercial offering.** |
| **Wikidata** (SPARQL + `wbgetentities`) | **CC0 1.0** (public domain) | Relationship spine (P144/P4969/P155/P156), external-ID hub (P345/P4947), entity resolution | None required (CC0); we still record per-claim provenance in `sources[]` |
| **TMDB API** (v3) | Free developer API; **attribution required**; data community-editable | Multilingual titles, alternative titles, credits (`sutradhar.pipeline.tmdb`) | **"This product uses the TMDB API but is not endorsed or certified by TMDB."** TMDB logo required on any UI surface (P6). |
| **Wikipedia** (MediaWiki/REST API — never HTML-scraped) | **CC BY-SA 4.0** (attribution + share-alike) | Plot/synopsis prose for P2 embeddings (`plot_texts`, revision-pinned); candidate-edge extraction input (P1 task 11) | Per-article attribution: page URL + revision id stored on every `plot_texts` row; **derived text redistributions must remain CC BY-SA**. |

## Models (per DEC-0001)

| Model | License | Our usage |
|---|---|---|
| Gemma 4 E4B | Apache 2.0 | FT base + P1 candidate-edge extraction (GPU session) |
| Qwen3-4B-Instruct-2507 | Apache 2.0 | Fallback base |
| BGE-M3 / bge-reranker-v2-m3 | MIT / Apache 2.0 | P2 embeddings / reranking |
| Sarvam-M 24B | Apache 2.0 | Optional P4 synthetic-data teacher + live showcase |
| ~~Sarvam-1 (2B)~~ | **non-commercial — AVOIDED** | not used (CLAUDE.md rule) |
| AI4Bharat IndicXlit | **CC BY-SA 4.0** | **Contingency only** (DEC-P1-5); if ever invoked, add share-alike attribution here |

## Code

This repository: MIT (see `pyproject.toml`). `indic-transliteration` (Python): MIT.

| openai/gpt-oss-20b (LLM-as-judge, DEC-P3-1; rev 6cee5e81) | Apache 2.0 | Self-hosted on the ephemeral GPU for judge/RAGAS batch passes only; never serves users; weights pinned by revision SHA |
