# data-pipeline

Ingestion and graph construction for Sutradhar's film catalog and remake/dub graph.

**Import packages:** `sutradhar.pipeline` (ingestion/build — upcoming), `sutradhar.graph`
(store: schema, engine plumbing, repository).

## Graph store (P1 task 1 — built)

The Catalog + Remake-Graph schema lives in `src/sutradhar/graph/schema.py` (SQLAlchemy 2.0 typed
ORM, DEC-P1-2) with Alembic migrations under `alembic/` (repo root). Apply with:

```bash
make up          # Postgres (+pgvector) via docker compose
make db-migrate  # alembic upgrade head — env-driven POSTGRES_*, nothing hardcoded
```

### Tables

| Table | Holds |
|---|---|
| `work` | Canonical film-story lineage (or `literary_source`, for GS-05) |
| `version` | Per-language film Version of a Work (original / remake / dub track) |
| `version_title` | Cross-script title match index (`match_key`-indexed; canonical/aka/dub/transliteration) |
| `person`, `version_cast` | Cast/crew — evidence for the dub-vs-remake rule (lead-cast overlap) |
| `edges` | Single polymorphic typed-edge table (DEC-P1-1): remake/dub = version→version, sequel/`based_on` = work→work — shape CHECK-enforced, endpoint existence trigger-enforced |
| `candidate_edges` | LLM proposals awaiting the human gate — **not** an edge table, never read by views |
| `conflicts` | Multi-source disagreements, never silently resolved |
| `plot_texts` | Wikipedia/TMDB plot prose (revision-pinned, license recorded) — the P2 embedding corpus |

Every gated record/edge carries `confidence` (`HIGH`/`MEDIUM`) + inline `jsonb sources[]`
(DEC-P1-3) + `human_verified`. `is_original_of` is **derived** (from `version.is_original` +
incoming remake/dub edges), never stored — one source of truth.

### The verification gate as schema

Downstream consumers read **only** the `ground_truth_works` / `ground_truth_versions` /
`ground_truth_edges` views. Predicate (DEC-P1-7): `sources[]` non-empty **and** no open
`conflicts` row. Layered gates:

1. **Structural** — CANDIDATE tier is a separate table (`candidate_edges`); it cannot leak into
   any view by construction.
2. **View** — conflicted or provenance-less rows are invisible until resolved. MEDIUM rows pass,
   flagged (per the `DATA_SOURCES.md` tier table).
3. **Fixture** — the golden-set validator additionally requires HIGH or human-verified.

### Tests

```bash
uv run pytest tests/test_graph_schema.py       # hermetic: schema inventory, DSN builder
make up && uv run pytest -m integration        # constraints + gate views on real Postgres
```

Integration tests apply migrations themselves and roll back per-test; CI runs them in the Tier-1
compose job.

## Planned (remaining P1 tasks)
- Ingest from Wikidata SPARQL (relationship spine: P144/P1877/P4969, P155/P156/P179), TMDB
  (`translations`, `alternative_titles`, credits), IMDb `title.akas` (slice-filtered), Wikipedia
  REST (plots) — snapshot-first, API/dump only, never HTML scraping.
- Deterministic rule-based transliteration + normalization (`match_key`; no neural model on the
  laptop), rapidfuzz title resolution.
- Precedence-table conflict resolution; LLM candidate-edge extraction (GPU session) + typer
  review gate; coverage/lift reports; golden fixtures GS-01..GS-11.
