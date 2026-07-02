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

## TMDB enrichment (P1 task 5 — built)

`sutradhar.pipeline.tmdb` + `data-pipeline/enrich_tmdb.py` (`make enrich-tmdb`):

- **One call per film** (`append_to_response=translations,alternative_titles,credits`, verified
  v3 contract §2.9); auth auto-detects a v4 Bearer token vs a v3 `api_key`; 429 `Retry-After`
  backoff; endpoint env-driven (`TMDB_API_URL`). Snapshot-first (`data/raw/tmdb/`), `--offline`
  replay, committed CI fixture (6 movies) under `tests/fixtures/tmdb/`.
- **Titles → `version_title`:** canonical (own language), alternative titles (kind=aka), and
  translations in a QID-less sibling's language mapped onto that sibling — kind=dub for dub
  tracks, canonical for a bilingual co-original. Interim `match_key` per row (task 8 upgrades to
  transliteration and re-keys idempotently).
- **Credits → `person` / `version_cast`:** billing order <5 = lead, else support; crew
  `job=Director` — the evidence base for the dub-vs-remake rule (task 9).
- **Precedence table as code** (`sutradhar.pipeline.precedence`, the `DATA_SOURCES.md` rows):
  `hub` / `primary_corroborate` / `majority` / `union` strategies. Rule-decidable disagreements
  are recorded as `conflicts(status=resolved, resolution={by: rule})` — never silent, row stays
  live; rule-undecidable splits (e.g. a 2-way year split) stay `open` and the gate views hide
  the row until human resolution.

Live run 2026-07-02 (snapshot `20260702T061520Z`): 27/27 versions enriched, **117 title rows
(30 canonical / 86 aka / 1 dub), 312 people, 382 cast rows, 0 conflicts** — seed, Wikidata, and
TMDB agree on every checked field. Honest gap: TMDB's translations give only the Baahubali *ml*
dub title; the ta/hi dub titles and Devadas (ta) must come from IMDb `title.akas` (task 6).

## IMDb `title.akas` loader (P1 task 6 — built)

`sutradhar.pipeline.imdb` + `data-pipeline/load_akas.py` (`make load-akas`):

- **Streamed + slice-filtered:** the multi-GB `title.akas.tsv.gz` is streamed from
  `datasets.imdbws.com` (env: `IMDB_DATASETS_URL`) and filtered on the fly to the slice's 27
  tconsts — the raw dump is never stored, never committed (36 s live). Only the filtered rows
  (546) land in the hash-recorded snapshot; a 239-row real capture is the committed CI fixture.
- **Column contract** per developer.imdb.com (§2.9): `\N` = null, `isOriginalTitle` flag.
  `isOriginalTitle` rows corroborate the **canonical** title; language-tagged rows map onto
  QID-less siblings (dub / co-original guard as in TMDB); the rest are `kind=aka`.
- **Union semantics** (`sutradhar.pipeline.titles`, shared with TMDB): a title observed by a
  second source **merges** that ref into `sources[]` — ≥2 independent sources = HIGH per value.
  The ref carries no per-row ordering, so multiple IMDb rows can never fake 2-source
  corroboration; a same-pass duplicate merges instead of duplicating (flush fix, caught live).
- **License:** IMDb datasets are **personal/non-commercial only** → `docs/LICENSING.md`
  (drafted this task).

Live run 2026-07-02 (snapshot `20260702T063039Z`): 546 filtered rows → **158 new titles
(2 dub titles mapped, incl. Baahubali's hi बाहुबली: एक शुरुआत — the TMDB gap filled), 37
corroborated (multi-source)**; version_title total 270. Re-run → all zeros. Honest gap:
Devadas (ta dub of Devadasu 1953) has no language-tagged akas row — its title must come from
Wikipedia/extraction (tasks 7/11).

## Wikipedia plot fetch (P1 task 7 — built)

`sutradhar.pipeline.wikipedia` + `data-pipeline/fetch_plots.py` (`make fetch-plots`):

- **Action API only, never HTML scraping** (DATA_SOURCES.md): one call per page —
  `prop=extracts|revisions|info` gives plaintext with `== wiki ==` section markers, the latest
  revision id, and the canonical URL. Endpoint env-templated (`WIKIPEDIA_API_URL` with `{lang}`).
- **Article resolution costs zero extra calls:** titles come from the Wikidata sitelinks already
  captured in the task-4 snapshot. Per version: the enwiki article + the version's own-language
  wiki article.
- **Revision-pinned + licensed per row:** every `plot_texts` row records `revision_id`,
  `source_url`, `license='CC BY-SA 4.0'`, `retrieved_at` (attribution obligations, LICENSING.md).
  An article edit re-pins (text + revision updated in place), never duplicates.
- **Stored text** = lead + a Plot/Synopsis-type section (multi-language heading list) when
  found, else the full extract — this is P2's embedding corpus and task 11's extraction input.

Live run 2026-07-02 (snapshot `20260702T064101Z`): **52 pages → 52 rows** (27 enwiki + 25
native-wiki; avg ~5.1 KB), all revision-pinned, re-run → 52 unchanged. QID-less dub tracks are
skipped and reported (their story lives in the parent film's article — extraction's input).

## Normalization / transliteration (P1 task 8 — built, DEC-P1-5 + measured amendment)

`sutradhar.pipeline.normalize` (+ `make rekey-titles`):

- **`match_key` pipeline:** NFC → script detection (Unicode-block majority) → deterministic
  **ITRANS** romanization (`indic-transliteration`; measured winner over IAST/ISO — avg 87.4 vs
  80.9 on real slice pairs, see DEC-P1-5 amendment) with Tamil digraph normalization and
  Devanagari/Bengali final-schwa deletion → casefold → strip diacritics → alnum-only →
  collapse character runs (`Paapanaasam → papanasam`, `दृश्यम → drishyam` **exact**).
- **Resolution** = exact key hit, then `best_matches` (rapidfuzz ratio, 0–1, threshold 0.80
  tuned on GS-11): perturbations ("Papanaasam", "Chandramuki", 1–2-char typos) resolve; decoys
  ("Inception", "Kaithi") stay below threshold; the Drishyam-family mutual near-matches all
  surface — the ambiguity signal `resolve_title` needs (GS-10).
- **No neural op** — pure Python, laptop/CI-safe; IndicXlit stays the unused contingency.
  Known limitation: non-Sanskrit Tamil letters (ன/ழ/ற), Sinhala, Han have no deterministic
  mapping — their Latin AKA/canonical rows in the same index carry the match instead.
- `make rekey-titles` (idempotent): seeds every version's canonical title into the index,
  recomputes all interim keys, populates `version_title.script`. Live 2026-07-02: +8 canonical
  rows, 88 keys recomputed, 278 scripts populated (index: 209 latn / 69 native-script rows).

## Planned (remaining P1 tasks)
- Ingest from Wikidata SPARQL (relationship spine: P144/P1877/P4969, P155/P156/P179), TMDB
  (`translations`, `alternative_titles`, credits), IMDb `title.akas` (slice-filtered), Wikipedia
  REST (plots) — snapshot-first, API/dump only, never HTML scraping.
- Deterministic rule-based transliteration + normalization (`match_key`; no neural model on the
  laptop), rapidfuzz title resolution.
- Precedence-table conflict resolution; LLM candidate-edge extraction (GPU session) + typer
  review gate; coverage/lift reports; golden fixtures GS-01..GS-11.
