# data-pipeline

Ingestion and graph construction for Sutradhar's film catalog and remake/dub graph.

**Import package:** `sutradhar.pipeline`

## Planned architecture
- Ingest from TMDB (titles, translations, `alternative_titles`), Wikidata SPARQL (remake edges via
  P144 "based on" / P1877 "after a work by"), IMDb `title.akas` (AKA/dub titles), curated Kaggle
  Indian-movie sets to backfill South-Indian coverage.
- Build canonical **Work** nodes and per-language **Version** nodes with typed edges
  (`is_original_of` / `is_remake_of` / `is_official_dub_of` / `is_unofficial_remake_of` /
  `is_sequel_of`) in Postgres.
- Deterministic rule-based transliteration + normalization (no neural model on the laptop).
- Confidence gate: ground-truth only if HIGH confidence (≥2 independent sources agree, or an
  authoritative structured source) or human-verified; LLM-extracted edges land in `candidate_edges`
  and never enter the live graph unverified.

## Status
**Not built until P1.** P0 creates this directory as a stub only.
