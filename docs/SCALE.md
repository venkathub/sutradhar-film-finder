# SCALE.md — The 50k-Film Design Note (P7 task 18)

> **Status: DESIGN NOTE ONLY (2026-07-18).** ROADMAP §6.6 names catalog scale-up as future ops;
> the post-P6 review asked for the path to be *designed, not hand-waved*. This document is that
> design. **Nothing here is implemented in P7** (non-goal, P7_SPEC §1) — every section states
> the current seed-slice implementation honestly, the mechanism that replaces it at ~50k films
> (≈ 100–150k Versions, ≈ 1–2M chunks), and the trigger that says when.
>
> Sources (accessed 2026-07-18): pgvector 0.8.x iterative index scans + filtering docs;
> pgvector HNSW-vs-IVFFlat guidance; PostgreSQL pg_trgm docs; Wikidata SPARQL paging practice.

## 0. Scale model & triggers

| Quantity | Seed slice (today) | 50k target | Growth driver |
|---|---:|---:|---|
| Works | 16 | ~50k | discovery-mode ingestion (§4) |
| Versions | ~35 | ~100–150k | remakes/dubs per work |
| `version_title` rows (canonical+AKA+translit) | ~10² | ~1M | AKA/dub titles × scripts |
| Plot chunks (winner config `1024tok_15pct`) | 153 | ~1–2M | one plot per version, chunked |
| Golden fixtures | 85 (DEC-P7-4) | unchanged | eval set does NOT scale with catalog |

**General trigger discipline:** each change below activates at a *measured* threshold (recorded
in DECISIONS at the time), never preemptively — the seed slice must stay the simplest thing that
works.

## 1. Title resolution: pg_trgm/GIN replaces the O(N) in-Python fuzzy scan

**Today (honest):** `graph.repository.resolve_title` SELECTs the **entire** gate-visible title
index into Python and runs `rapidfuzz.best_matches` over every distinct `match_key` per query.
Correct and fast at 10² rows; at ~1M rows it is an O(N) per-query table scan with Python-side
scoring — unshippable.

**Design:**
- `CREATE EXTENSION pg_trgm;` + `CREATE INDEX ... ON version_title USING gin (match_key gin_trgm_ops);`
- Query becomes a two-stage funnel: **(1)** in-DB trigram candidate pull —
  `SELECT ... WHERE match_key % :query_key ORDER BY similarity(match_key, :query_key) DESC LIMIT 200`
  (index-served via GIN; `pg_trgm.similarity_threshold` tuned on GS-11) — then **(2)** the
  existing rapidfuzz scorer re-ranks only those ≤200 candidates, preserving today's scoring
  semantics (`score` stays the rapidfuzz-normalized 0–1 the frozen tool schema documents).
- Transliteration is unaffected: `match_key` is already the deterministic ITRANS romanization
  (DEC-P1-5), so cross-script matching keeps working — the trigram index sits on the same key.
- **Eval gate:** GS-07/GS-10/GS-11 title-resolution fixtures must be bit-identical through the
  funnel (the candidate pull may only *lose* candidates rapidfuzz would have scored < the
  current floor). A recall-vs-limit ablation (LIMIT ∈ {50, 200, 500}) picks the funnel width.
- **Trigger:** `version_title` > ~50k rows or resolve_title p95 > 50 ms measured.

## 2. Vector search: HNSW + iterative index scans for filtered queries

**Today (honest):** `chunk_embeddings.dense` is a `vector(1024)` column with **no ANN index** —
dense search is an exact sequential scan (fine and *exactly recall-1.0* at 153–458 chunks; the
bb22e78 migration records HNSW as the catalog-scale revisit, DEC-P2-1).

**Design:**
- **HNSW over IVFFlat**: build-once/query-many workload, no retraining on inserts (IVFFlat lists
  degrade as the catalog grows), recall/latency curve strictly better for our read-heavy shape.
  BGE-M3's 1024 dims sit comfortably under pgvector's 2,000-dim HNSW limit.
  `CREATE INDEX ON chunk_embeddings USING hnsw (dense vector_cosine_ops) WITH (m = 16, ef_construction = 64);`
- **The known trap — post-scan filtering:** our dense channel ALWAYS filters (gate views,
  `chunk_config`, `embed_model`, `index_version`; catalog-scale adds language/era filters). With
  classic HNSW, filters apply AFTER the index scan: `ef_search = 40` candidates that then hit a
  10%-selective filter return ~4 rows for a LIMIT 50 — silent recall collapse.
- **The fix — pgvector 0.8.x iterative index scans:**
  `SET hnsw.iterative_scan = strict_order;` + a bounded `hnsw.max_scan_tuples` — the scan keeps
  pulling candidates until the filtered LIMIT is satisfied or the bound is hit. `strict_order`
  (not `relaxed_order`) because fused RRF ranks are order-sensitive.
- **Highly-selective filters** (e.g. one work's chunks): planner should choose the btree on
  `(chunk_config, embed_model, index_version)` + exact scan instead — verified with
  `EXPLAIN ANALYZE` fixtures in CI.
- **Eval gate:** the P2 harness re-runs per index config; **Recall@10 ≥ 0.90 and version-set
  recall = 1.0 on GS-01/GS-06 remain the hard gate**, now measured *through* the HNSW path.
  Every index build is a fresh `index_version` — and per DEC-P7-3, **a rebuilt index without a
  matching re-calibration hard-fails serving** (`StaleCalibrationError`): θ is re-derived on the
  new score distribution before NO_MATCH abstention goes live. Sparse stays `sparsevec` +
  in-DB scoring; RRF fusion is index-agnostic.
- **Trigger:** > ~50k chunks, or dense-channel p95 > 100 ms measured.

## 3. Storage & compute placement (unchanged principles, bigger numbers)

- 1–2M × 1024-dim fp32 ≈ 4–8 GB vectors + HNSW overhead — a single mid-tier Postgres volume;
  no Qdrant migration needed (re-affirms DEC-P2-1 at 50k scale).
- Corpus embedding at 50k scale is a **batched on-demand GPU job** (≈ 2M chunks ≈ a few hours on
  the DEC-0003 A100 — a budgeted window with a persisted, versioned artifact, exactly the P2
  pattern; the laptop still never runs a model).

## 4. Ingestion: discovery mode, paginated SPARQL, delta re-ingest

**Today (honest):** ingestion is seed-driven — a hand-curated YAML (413 lines, 16 works) with
per-QID entity fetches; the SPARQL relationship spine is unpaginated (fine at seed scale).

**Design:**
- **Discovery mode:** replace the YAML *frontier* (not the gate): seed queries per language
  industry (`instance of film` × `original language` ∈ {ml, ta, te, hi, kn, bn, …}) stream
  candidate QIDs; each discovered work runs the SAME pipeline (spine → enrich → titles → gate).
  The verification gate is untouched — discovery widens *candidates*, never ground truth;
  low-confidence records simply never pass the gate views (the enforced HIGH/human-verified
  property scales as-is).
- **Paginated SPARQL:** all spine queries gain `ORDER BY ?item LIMIT 500 OFFSET n` slicing with
  per-page retry/backoff under the existing WMF UA policy; ingestion becomes resumable
  (page cursor persisted) so a 50k crawl survives interruption.
- **Delta re-ingest:** a `last_seen_revision` per source record + Wikidata's
  `schema:dateModified`; re-ingest touches only changed entities. The P7 task-6 provenance
  rules make this safe by construction: `sources[]` merges append-only, confidence is
  raise-only, human-verified records are immutable to the pipeline.
- **Human gate at scale:** candidate-edge review becomes sampling-based (per-language strata,
  precision CI instead of exhaustive review) — the same measured-precision posture as P1, with
  the sampling design logged as its own DEC entry when activated.

## 5. What deliberately does NOT change

- **The eval gate does not scale with the catalog** — golden fixtures grow by *scenario
  coverage* (DEC-P7-4), not by row count; Recall@10 ≥ 0.90 and version-set recall = 1.0 on
  GS-01/GS-06 gate every retrieval change at any scale.
- **Compute placement** — laptop/CI on persisted artifacts; every neural op in a budgeted
  on-demand GPU window (ROADMAP §2).
- **The verification gate** — HIGH (≥2 sources or authoritative) or human-verified, no
  unresolved conflicts; discovery mode feeds it, never bypasses it.
- **Tool schema v0** — `resolve_title`/`search_by_plot` signatures and score semantics are
  frozen; every change above is behind those contracts.
