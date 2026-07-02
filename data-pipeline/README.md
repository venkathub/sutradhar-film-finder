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

Writes go through the pydantic domain models in `src/sutradhar/graph/models.py` (`SourceRef`,
`WorkRecord`, `VersionRecord`, `EdgeRecord`, …): empty `sources[]`, unknown source ids, malformed
QID/tconst patterns, and edge shape violations fail **before** any insert; the DB CHECKs and the
endpoint trigger are the backstop. `passes_gate_view()` / `golden_eligible()` mirror the two gate
layers in Python for pipeline and fixture code.

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
uv run pytest tests/test_graph_schema.py tests/test_graph_models.py  # hermetic units
make up && uv run pytest -m integration        # constraints + gate views on real Postgres
```

Integration tests apply migrations themselves and roll back per-test; CI runs them in the Tier-1
compose job.

## Seed slice (P1 task 3 — built)

`data-pipeline/seed_slice.yaml` is the committed vertical slice (P1_SPEC §2.7): 15 works /
31 versions covering every flagship chain — Drishyam (5 Indian + 2 foreign versions), Drishyam 2
(`is_sequel_of` work), Baahubali (bilingual double-original + hi/ml dub tracks), Devdas novella +
3 sibling adaptations (`based_on`, with the Tamil dub composing inside Devadasu), Vikram 1986/2022
(false-merge pair), the Manichitrathazhu transitive chain (Chandramukhi → Apthamitra proximate
edge), and 5 distractors. It is both the ingestion input **and** the curated-truth denominator for
the graph-coverage metric.

Every QID was confirmed against live Wikidata (2026-07-02; evidence per row in the YAML notes).
Notable curation findings, encoded not smoothed over:

- **Wikidata's P144 spine is incomplete:** only the hi-2015 and zh-2019 Drishyam remakes carry
  `based on` edges — the kn/te/ta/si remakes don't. That gap is the extraction layer's measured
  lift target.
- **Bilingual originals share one Wikidata item:** `version.wikidata_qid` is UNIQUE, so the QID
  sits on the primary original (Baahubali te) and the co-original (ta) carries none.
- **Devadasu 1953 dub-vs-bilingual is contested** (spec says Tamil dub; Wikidata P364 lists both
  te+ta) — flagged for the conflicts queue at ingestion, not silently picked.
- **Reported-but-unconfirmed versions** (Korean/Indonesian/US Drishyam, Dharmayuddhaya 2 — whose
  Wikidata item exists but is vandalized) live in a `backlog` list: name + reason only, no
  invented records, excluded from every denominator (§7 Q1).

Loader: `sutradhar.pipeline.seed.load_seed_slice()` — typed pydantic validation (dangling
relationship targets, works without originals, duplicate QIDs, literary sources with versions all
rejected at load).

## Wikidata spine ingest (P1 task 4 — built)

`sutradhar.pipeline.wikidata` + `data-pipeline/ingest_spine.py` (`make ingest-spine`):

- **Two-phase access** (verified best practice, P1_SPEC §2.9): one SPARQL discovery query returns
  QIDs only (P144/P4969 backlinks of the slice); entity detail comes from batched
  `wbgetentities` (≤50/call). Descriptive User-Agent (WMF policy, `HTTP_USER_AGENT`), gzip,
  429/503 `Retry-After` backoff. Endpoints env-driven (`WIKIDATA_API_URL`, `WIKIDATA_SPARQL_URL`).
- **Snapshot-first:** raw responses land in `data/raw/wikidata/<UTC-stamp>/` (git-ignored) with a
  sha256 `MANIFEST` before any DB write; `--offline` replays the latest snapshot, so rebuilds and
  CI never re-hit the API (CI parses a committed trimmed capture under `tests/fixtures/wikidata/`).
- **Idempotent upsert** keyed on QID (fallback `(work_id, language)` for QID-less dub tracks);
  QID-anchored rows are HIGH, QID-less dub tracks are MEDIUM (single human source) until
  corroborated. Seed-vs-Wikidata year disagreements open a `conflicts` row — never silently
  resolved — which hides the row from the gate views until resolution.
- **Metric honesty:** only edges **Wikidata asserts** are written (P144/P4969 → remake/based_on,
  P155/P156 → sequel). Seed `relationship:` entries are curated truth, not an edge source.

Live run 2026-07-02 (snapshot `20260702T055436Z`): 15 works, 31 versions, **12 edges**
(8 `is_remake_of`, 3 `based_on`, 1 `is_sequel_of`), 0 conflicts. Wikidata's spine is measurably
incomplete for the slice: the kn/te/si Drishyam remakes, both Drishyam-2 kn/te remakes, all dub
tracks, and both proximate Manichitrathazhu-chain edges have **no** structured assertion — that
gap is the extraction layer's lift target. 13 non-slice backlink QIDs were discovered and
reported for conditional-add review (§7 Q1), not auto-ingested.

## Planned (remaining P1 tasks)
- Ingest from Wikidata SPARQL (relationship spine: P144/P1877/P4969, P155/P156/P179), TMDB
  (`translations`, `alternative_titles`, credits), IMDb `title.akas` (slice-filtered), Wikipedia
  REST (plots) — snapshot-first, API/dump only, never HTML scraping.
- Deterministic rule-based transliteration + normalization (`match_key`; no neural model on the
  laptop), rapidfuzz title resolution.
- Precedence-table conflict resolution; LLM candidate-edge extraction (GPU session) + typer
  review gate; coverage/lift reports; golden fixtures GS-01..GS-11.
