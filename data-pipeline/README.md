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

## Graph builder (P1 task 9 — built)

`sutradhar.pipeline.build` + `data-pipeline/build_graph.py` (`make build-graph`;
`make ingest-seed` now chains the whole flow: spine → tmdb → akas → plots → rekey → build):

- **Dub-vs-remake rule as a pure function** (DATA_SOURCES.md): lead-cast overlap (relative to
  the smaller lead set) ≥ 0.5 → `is_official_dub_of`; disjoint → `is_remake_of`; missing
  evidence → abstain. Every version-level edge is **cross-checked**: agreement is counted as
  corroboration; disagreement opens an `edge_type` conflict — **never a silent re-type** —
  hiding the edge from the gate views until human resolution.
- **Dub-track edge derivation:** QID-less, non-original tracks (they exist only as tracks of
  their film — no external record of their own) get `is_official_dub_of` → primary original,
  `confidence=MEDIUM` with an honest `rule` source ref (DEC-P1-3 amendment). The human gate
  (task 12) promotes them.
- **What the builder never does:** write remake edges from seed curation — the Wikidata gap
  stays visible for extraction + review (lift attribution intact). Edge origins are separable
  by `sources[0].source`: 12 wikidata / 3 rule after the live run.
- Integrity checks: films without originals, >2 originals flagged as anomalies (live: none).

Live run 2026-07-02: **8/8 Wikidata remake edges rule-confirmed** (disjoint lead casts — every
remake typing corroborated by TMDB credits), **3 dub edges derived** (Baahubali hi/ml, Devadas
ta), 0 conflicts, 0 anomalies; graph = 15 works / 31 versions / 15 edges. Re-run → all zeros.
Named regressions green here: `test_gs04_dub_vs_remake`, `test_gs05_sibling_vs_remake`,
`test_gs10_false_merge`, + the rule-disagreement conflict gate.

## Repository — the tool contract, satisfied (P1 task 10)

`src/sutradhar/graph/repository.py` implements the graph-backed **TOOL_SCHEMA v0** signatures
as plain functions over the **ground-truth views only** (CANDIDATE rows and conflicted records
are structurally invisible — tested end-to-end):

| Tool (v0) | Behaviour |
|---|---|
| `resolve_title(title, language?)` | match-key index + rapidfuzz; exact = score 1.0; `ambiguous=true` when candidates span >1 Work ("Vikram" → 2 works, GS-10); native-script and perturbed queries resolve (GS-11) |
| `get_work(work_id)` | Work + `source_work` (the novella for Devadasu — GS-05) + `based_on[]` |
| `get_versions(work_id, scope, include_sequels)` | `scope` via `version.country`; `include_sequels` = transitive `is_sequel_of` walk (GS-06); labels: root original → derived `is_original_of`, sequel-work original → `is_sequel_of`, verified remake/dub edges → their type, no verified edge → `null` (honest gap until extraction+review) |
| `refine_filter(version_set, by)` | actor (lead-name match), language, year, `era` resolved against the set's original's year, relationship (GS-08 backtracking) |
| `search_by_plot` | **not implemented** — needs P2 retrieval + calibrated abstain; schema frozen at task 15 |

Named regressions green here: `test_gs02_no_hallucinated_movie` (decoys resolve to nothing),
`test_gs06_franchise_version_set_recall` (9/9 franchise versions, sequel-vs-remake labels never
conflated), `test_gs09_scoping` (indian/foreign/all partition exact), `test_gs09_transitive_lineage`
(full lineage, sole ml original — the proximate-edge assertion joins with the task-14 fixtures).

## LLM candidate-edge extraction (P1 task 11 — built + real GPU run done)

`sutradhar.pipeline.extract` + `data-pipeline/extract_candidates.py` (`make extract-candidates`;
GPU lifecycle: `infra/gpu/jarvis.py extract` — create → serve → extract → **destroy**, with
script-quota cleanup):

- **Honesty contract:** model output validates against a pydantic schema — malformed output is
  dropped + counted, never repaired; the `supporting_sentence` must appear **verbatim**
  (whitespace-normalized) in the source text or the proposal is dropped as unsupported; every
  candidate carries page + revision pin, model id, and an `extraction_run` hash (prompt + model
  + revisions). Title→version binding is conservative (unambiguous ≥0.9 `resolve_title` hit);
  raw strings always kept for the reviewer. **Nothing touches `edges`** — quarantine tested.
- **vLLM guided decoding required** (DEC-P1-4 amendment): free-form prompting of the 4B base
  gave 92.6% parse failures; `guided_json` + temperature 0 gave **7.4%**.

**Real GPU run 2026-07-02** (ephemeral A100, machine 437943, ~50 min ≈ $1, destroyed; artifact
`data/raw/extraction/20260702T085302Z`, run hash `3e37549f492bd2fc`): 27 pages → 72 proposals
→ 14 dropped by the verbatim guard → **58 candidates** (27 both-ends bound, 31 raw-kept),
including the Wikidata-missing edges the pipeline was built to recover (Drishya/Drushyam/
Dharmayuddhaya → Drishyam, **Chandramukhi → Apthamitra** proximate, Baahubali dub mentions) —
plus honest 4B noise (self-pairs, type confusion, inverted directions) left for the human gate
to measure as precision (task 12). CI replays a 5-page slice of the **real** artifact.

## Human review gate (P1 task 12 — built + real review pass done)

`sutradhar.pipeline.review` + `data-pipeline/review_candidates.py` (`make review-candidates`,
DEC-P1-6): interactive y/n/s CLI **and** batch mode over a committed decisions YAML (the audit
artifact of a session). Gate semantics, enforced and tested:

- **Promotion is the only candidate→edge path**: confirmed → `human_verified=true` HIGH edge
  (or corroboration — sources merged onto the existing edge, verified flag set, never a dup),
  `promoted_edge_id` linking the audit trail. Work-level types (`based_on`/`is_sequel_of`)
  promote at work level; endpoints may be reviewer-bound (resolution ≠ repair — the model's
  own bindings can be wrong, observed live). Unbindable confirms are refused, never partial.
- **Rejection** records reviewer + timestamp and never writes edges. **Skip** = out-of-slice
  truth (excluded from the precision denominator, logged for the breadth backlog).
- Rule-derived MEDIUM dub edges have an explicit verification queue (same gate semantics).

**Real review pass 2026-07-02** (reviewer: venkatesh; decisions file
`data-pipeline/review_decisions_20260702.yaml`, run `3e37549f492bd2fc`): 58 candidates →
**19 confirmed / 35 rejected / 4 skipped** → **candidate precision 0.352** (the honest 4B
number: inverted directions, self-pairs, type confusion all rejected). **6 edges created =
exactly the Wikidata gap** — Drishya, Drushyam, Dharmayuddhaya, Drishya 2, Drushyam 2, and the
Chandramukhi→Apthamitra **proximate** edge (GS-09B) — plus 13 corroborations and 3 dub-track
verifications. Graph: 15 → **21 gate-visible edges**; every curated Indian remake edge now
exists and GS-01 is unblocked.

## Reports (P1 task 13 — built + captured)

`sutradhar.pipeline.report` + `make graph-report` (exit code 1 if the flagship gate fails —
CI-usable): per-franchise **version coverage** vs the curated seed truth (backlog excluded from
every denominator by construction), supplementary curated-relationship **edge coverage**
(proximate targets counted, gaps named), **extraction lift** (precision, parse-failure rate,
verified-edges-beyond-Wikidata via provenance attribution), and the §6.1 reproducibility stamp
(code SHA, seed sha, snapshot manifest digests, model + run hash).

**Captured 2026-07-02** (full numbers in `docs/BENCHMARKS.md` "Graph coverage & extraction
lift"): flagship gate **PASS** (1.00 on all five flagships), edge coverage 19/20 (the Rajmohol
proximate edge has no stating source — recorded, not invented), precision 0.352, **6 verified
edges beyond Wikidata**, 10 corroborations.

## Planned (remaining P1 tasks)
- Ingest from Wikidata SPARQL (relationship spine: P144/P1877/P4969, P155/P156/P179), TMDB
  (`translations`, `alternative_titles`, credits), IMDb `title.akas` (slice-filtered), Wikipedia
  REST (plots) — snapshot-first, API/dump only, never HTML scraping.
- Deterministic rule-based transliteration + normalization (`match_key`; no neural model on the
  laptop), rapidfuzz title resolution.
- Precedence-table conflict resolution; LLM candidate-edge extraction (GPU session) + typer
  review gate; coverage/lift reports; golden fixtures GS-01..GS-11.
