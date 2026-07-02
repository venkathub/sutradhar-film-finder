# P1 Spec — Data pipeline + remake-graph (the hard problem's foundation)

> Phase spec for **P1** of the Sutradhar roadmap. Grounded in `docs/ROADMAP.md` (P1 entry/exit
> criteria, §2 vertical-slice + compute placement, §6 guardrails), `CLAUDE.md` (DoD, repo layout),
> `docs/DECISIONS.md` (DEC-0001..0003, DEC-P0-1..5 — settled, not reopened), `docs/DATA_SOURCES.md`
> (source inventory, precedence table, confidence tiers, verification gate),
> `docs/GOLDEN_SET_SCENARIOS.md` (GS-01..GS-11 + fixture schema), and `docs/phases/TOOL_SCHEMA.md`
> (v0 seed — **finalized and frozen as part of this phase's exit**).
>
> **Status:** **APPROVED — 2026-07-02.** Baseline for P1 execution. §7 open questions resolved
> (all recommendations accepted); decisions **DEC-P1-1..6 logged** in `docs/DECISIONS.md`
> (2026-07-02). Source-API contracts, tooling, and seed-slice ground truth **web-verified
> 2026-07-02** (§2.9); the seed slice was corrected against live sources (Drishyam 2 remake set,
> Manichitrathazhu transitive chain, Devadasu 1953 + its Tamil dub). Implementation proceeds per
> the §5 task order on `feature/p1-remake-graph`.
>
> _History: DRAFT written at grooming (2026-07-02); web-research pass folded in same day;
> approved with all proposed defaults accepted._
>
> **Current repo state (grounding):** P0 is complete: `src/sutradhar/{config,serving}` (typed
> `Settings`, OpenAI-compatible `LLMClient` with the `status:"off"` contract, hf-check),
> `infra/docker-compose.yml` (pgvector/pgvector Postgres + Redis, healthchecked), `Makefile`
> targets, two-tier CI, `infra/gpu/jarvis.py` (ephemeral create→serve→smoke→destroy lifecycle),
> `.env.example` incl. `TMDB_API_KEY` (unused until now). `/data-pipeline` and `/evals` are stub
> READMEs. There is **no Postgres schema, no ingestion code, no fixtures** yet. P1 builds the graph
> everything else stands on.

---

## 1. Scope

### In scope

1. **Postgres graph schema** (the Catalog + Remake-Graph store): canonical `work` nodes (film
   works **and** literary sources, for GS-05), per-language `version` nodes, a typed `edges` store
   (`is_remake_of` / `is_official_dub_of` / `is_unofficial_remake_of` / `is_sequel_of` /
   `based_on`; `is_original_of` derived — §2.2), plus `version_titles` (AKA/dub/transliteration
   index), `people` + `version_cast`, `candidate_edges`, `conflicts`, and `plot_texts`. Every
   record/edge carries `confidence` + `sources[]`.
2. **Ingestion connectors, per `DATA_SOURCES.md` (API/dump only — never HTML scraping):**
   - **Wikidata SPARQL** — the relationship spine: P144/P1877/P4969 (based-on/remake), P155/P156/
     P179 (sequel/series), external-ID hub (IMDb P345, TMDB), director P57, publication date P577.
   - **TMDB REST** — multilingual canonical titles (`/translations`), `alternative_titles`,
     `original_language`, credits (cast/crew), metadata gap-fill.
   - **IMDb `title.akas` TSV dump** — AKA/dub titles + region/language + "original" flag,
     **filtered to the seed slice's tconsts** (the raw multi-GB dump is never committed).
   - **Wikipedia REST API** — plot/synopsis prose per language version (revision-pinned, CC BY-SA
     attribution recorded), stored in `plot_texts` for P2 embeddings.
   Raw responses are persisted as versioned snapshot artifacts (`data/raw/`, git-ignored, hash
   recorded) so the graph build is reproducible without re-hitting APIs.
3. **Vertical seed slice (~30 records), not catalog breadth** (ROADMAP §2 build strategy). The
   slice covers every flagship chain the golden set needs: Drishyam family incl. Drishyam 2 +
   foreign adaptations (GS-01/03/06/08/09A), Baahubali (GS-04), Devdas + novella (GS-05), Vikram
   1986/2022 (GS-10), Manichitrathazhu → Bhool Bhulaiyaa (GS-09B), plus a handful of distractor
   films so GS-02/GS-03 negatives aren't trivially empty. Proposed composition in §2.7.
4. **Normalization + transliteration for cross-script title match:** deterministic rule-based
   `indic-transliteration` on the laptop producing a canonical `match_key` per title;
   `rapidfuzz` fuzzy match over the `version_titles` index (GS-11). Neural IndicXlit **only if**
   rule-based fails spot-checks, run in a rented-GPU session with outputs cached (ROADMAP §2
   compute placement).
5. **Multi-source conflict resolution:** the per-field precedence table from `DATA_SOURCES.md`
   implemented as code; disagreements are **never silently resolved** — both values + sources land
   in the `conflicts` queue for rule/human resolution.
6. **LLM candidate-edge extraction layer (rented-GPU session, ~1–2 h per DEC-0003):** an LLM reads
   Wikipedia lead/plot prose and *proposes* remake/dub edges Wikidata is missing → structured rows
   in `candidate_edges` (supporting sentence + source revision + model confidence). **Never
   straight into the graph.**
7. **Human-verification gate:** a review step (confirm/reject with the supporting sentence shown)
   that promotes confirmed candidates to `edges` with `human_verified = true`.
8. **Verification gate enforced in the schema layer:** ground-truth views expose a record/edge
   only if HIGH confidence (≥2 independent sources agree, or authoritative structured source) OR
   human-verified, AND no open `conflicts` entry, AND `sources[]`/`confidence` populated.
   CANDIDATE-tier rows are structurally excluded.
9. **Precision/recall-lift report:** verified edges added beyond Wikidata; candidate precision
   (confirmed/proposed) — written to `docs/BENCHMARKS.md` (new "graph" section, §6) and the module
   README.
10. **Graph-coverage metric (distinct from P2 retrieval recall):** per flagship franchise,
    `versions_present / versions_in_curated_truth` against a small human-curated truth file;
    exit requires 1.0 on the flagship chains.
11. **Seed golden set under `/evals/golden/`:** fixtures for **all** GS-01..GS-11 in the YAML
    schema from `GOLDEN_SET_SCENARIOS.md`, built only from HIGH/human-verified records, each
    fixture's IDs/relationships confirmed against sources during ingestion, plus a fixture
    **validator** that CI runs. Multi-turn fixtures (GS-07/GS-08) carry `expected_tool_calls`
    validated against the frozen tool schema (§4).
12. **Tool schema v0 frozen:** verify/refine `docs/phases/TOOL_SCHEMA.md` signatures against the
    real graph schema, emit a machine-readable `docs/phases/tool_schema.v0.json` (JSON Schema —
    the artifact P4 synthetic data and the tool-call-accuracy metric validate against), flip the
    doc status from DRAFT to **FROZEN v0**, and log it in `DECISIONS.md`. Read-only **repository
    query functions** backing the graph tools (`get_work`, `get_versions`, `refine_filter`,
    `resolve_title` sans dense retrieval) are implemented as plain Python functions to prove the
    contract is satisfiable (§2.5) — they are *not* exposed as an API (P5) and `search_by_plot`
    waits for P2.
13. **`docs/LICENSING.md`:** every source/model → license, our usage, attribution string (IMDb
    non-commercial, Wikidata CC0, TMDB attribution, Wikipedia CC BY-SA share-alike, models per
    DEC-0001).
14. **A 30-second demo path:** `make graph-demo` prints the Drishyam version set — original
    flagged, relationships labelled, per-claim sources — straight from the ground-truth views.

### Non-goals (explicit — prevents scope creep)

- **No catalog breadth.** The seed slice (~30 records) only; scaling to hundreds/thousands of
  films happens after the vertical slice is proven end-to-end (ROADMAP §2), as a later pass.
- **No embeddings, no vector index, no retrieval, no reranking, no chunking.** `plot_texts` are
  *stored* for P2; nothing is embedded. pgvector-vs-Qdrant and chunking are **P2 decisions**.
- **No retrieval metrics.** Graph coverage (R1, completeness) is measured here; Recall@k / MRR /
  version-set *retrieval* recall are P2. The two must not be conflated (they are different
  denominators).
- **No RAGAS / Langfuse / MLflow wiring.** P3. P1 metrics are computed by plain pytest + report
  scripts.
- **No fine-tuning, no synthetic FT data.** P4 (it depends on this phase's frozen tool schema).
- **No FastAPI routes, no tool *serving*, no orchestration, no model-emitted tool calls.** P5. P1
  ships repository query functions + a frozen contract, not an API.
- **No UI.** The `make graph-demo` path is a CLI print, not a product surface (P6).
- **No Java.** Deferred to P5 grooming per `CLAUDE.md` (§2.8).
- **No neural-model op on the laptop or in CI** (ROADMAP §2): transliteration on the laptop is
  rule-based only; the LLM extraction pass (and IndicXlit, if needed) runs in a short rented-GPU
  session and persists artifacts. CI validates against recorded artifacts.
- **No autonomous edge acceptance.** LLM output *never* enters `edges` without the human gate —
  this is the enforced property, not an aspiration.
- **No Wikipedia HTML scraping, no committing raw dumps** (size + license hygiene).

---

## 2. Design

### 2.1 Component breakdown

| Component | Code (import) | Entrypoint | Purpose |
|---|---|---|---|
| Schema + migrations | `sutradhar.graph.schema` | `make db-migrate` | Tables, enums, constraints, ground-truth views |
| Domain + provenance types | `sutradhar.graph.models` | — | Pydantic models: `SourceRef`, `Confidence`, node/edge records |
| Wikidata connector | `sutradhar.pipeline.wikidata` | `data-pipeline/ingest_spine.py` | SPARQL spine: QIDs, typed edges, external IDs |
| TMDB connector | `sutradhar.pipeline.tmdb` | `data-pipeline/enrich_tmdb.py` | Titles/translations/alt-titles/credits |
| IMDb akas loader | `sutradhar.pipeline.imdb` | `data-pipeline/load_akas.py` | Slice-filtered `title.akas` rows |
| Wikipedia plots | `sutradhar.pipeline.wikipedia` | `data-pipeline/fetch_plots.py` | Revision-pinned plot prose per version |
| Normalize/translit | `sutradhar.pipeline.normalize` | — | `match_key`, script detection, rapidfuzz match |
| Graph builder | `sutradhar.pipeline.build` | `data-pipeline/build_graph.py` | Entity resolution (QID hub), precedence, conflicts, dub-vs-remake rule |
| LLM extraction | `sutradhar.pipeline.extract` | `data-pipeline/extract_candidates.py` | GPU-session candidate-edge proposals via `LLMClient` |
| Review gate | `sutradhar.pipeline.review` | `make review-candidates` | Human confirm/reject → promotion |
| Repository (tool backing) | `sutradhar.graph.repository` | — | `get_work` / `get_versions` / `refine_filter` / `resolve_title` queries over ground-truth views |
| Golden set | `sutradhar.evals.golden` | `evals/build_golden.py` | Fixture builder + validator (`/evals/golden/*.yaml`) |
| Reports | `sutradhar.pipeline.report` | `make graph-report` | Coverage + extraction-lift report |
| Tool schema artifact | — | `docs/phases/tool_schema.v0.json` | Machine-checkable contract; synced-with-md test |

(Hyphenated dirs hold thin entrypoints importing `sutradhar.*`, per DEC-P0-2. New subpackages:
`sutradhar.graph`, `sutradhar.pipeline`, `sutradhar.evals`.)

### 2.2 Data model (the Work/Version graph)

**Modelling rules** (these encode the hard problem):

- A **Work** is one film-story lineage: the original film + all its remakes and dubs are
  **Versions of the same Work** (Drishyam family = 1 Work, ≥7 Versions incl. foreign).
- **Independent adaptations of a shared literary source are sibling Works**, each `based_on` a
  `literary_source` Work — never one remake chain (GS-05: Devdas 1955, 2002, Devadasu are three
  Works pointing at the novella node).
- **Sequels are separate Works** linked `is_sequel_of` at Work level; the sequel has its own
  Version set (GS-06: Drishyam 2 is a Work whose Malayalam original has a Hindi remake).
- **Dub vs remake is a Version-level edge type**: same story, different film+cast → `is_remake_of`;
  same film, replaced audio → `is_official_dub_of` (GS-04). Derived rule per `DATA_SOURCES.md`:
  lead-cast overlap across versions → dub; disjoint cast → remake; cross-checked against explicit
  Wikidata/Wikipedia statements, disagreements → `conflicts`.
- **Bilingual originals** (Baahubali, shot Telugu+Tamil): `is_original` may be true on **more than
  one** Version of a Work; dub edges point at the primary original. Encoded, tested in GS-04.
- **`is_original_of` is derived, not stored:** `is_original = true` on the Version + the inverse
  presentation of incoming `is_remake_of`/`is_official_dub_of` edges. One source of truth; no
  flag/edge divergence possible.
- **Same title + same actor ≠ same Work** (GS-10): entity resolution keys on the Wikidata QID hub
  (falling back to external-ID agreement), never on title/cast similarity.

**Tables (DDL sketch — final DDL in migrations):**

```sql
work (
  work_id         uuid PRIMARY KEY,
  work_type       text NOT NULL CHECK (work_type IN ('film','literary_source')),
  primary_title   text NOT NULL,
  original_language text,            -- null for literary sources
  first_release_year int,
  wikidata_qid    text UNIQUE,
  confidence      text NOT NULL CHECK (confidence IN ('HIGH','MEDIUM')),
  sources         jsonb NOT NULL,    -- SourceRef[]
  human_verified  boolean NOT NULL DEFAULT false,
  created_at / updated_at timestamptz
)

version (
  version_id      uuid PRIMARY KEY,
  work_id         uuid NOT NULL REFERENCES work,
  wikidata_qid    text UNIQUE, tmdb_id int, imdb_id text,
  title           text NOT NULL,     -- canonical title in this version's language
  language        text NOT NULL,     -- BCP-47-ish: ml, ta, te, hi, kn, si, zh, ...
  release_year    int,
  country         text,              -- drives GS-09 scope: indian | foreign
  is_original     boolean NOT NULL DEFAULT false,
  confidence / sources / human_verified / timestamps  -- as work
)

version_title (                       -- the cross-script match index (GS-07/10/11)
  title_id        uuid PRIMARY KEY,
  version_id      uuid NOT NULL REFERENCES version,
  title           text NOT NULL,
  kind            text CHECK (kind IN ('canonical','aka','dub','transliteration')),
  script          text,              -- deva | taml | mlym | latn | ...
  language        text,
  match_key       text NOT NULL,     -- normalized romanized key (§2.4), indexed
  sources         jsonb NOT NULL
)

person (person_id uuid PK, name text, wikidata_qid text UNIQUE, tmdb_id int, sources jsonb)
version_cast (
  version_id uuid REFERENCES version, person_id uuid REFERENCES person,
  role_kind text CHECK (role_kind IN ('lead','support','director')),
  billing_order int, sources jsonb,
  PRIMARY KEY (version_id, person_id, role_kind)
)

edges (                               -- single polymorphic edge table (DEC-P1-1)
  edge_id         uuid PRIMARY KEY,
  edge_type       text NOT NULL CHECK (edge_type IN
    ('is_remake_of','is_official_dub_of','is_unofficial_remake_of','is_sequel_of','based_on')),
  src_kind        text NOT NULL CHECK (src_kind IN ('version','work')),
  src_id          uuid NOT NULL,
  dst_kind        text NOT NULL CHECK (dst_kind IN ('version','work')),
  dst_id          uuid NOT NULL,
  confidence      text NOT NULL CHECK (confidence IN ('HIGH','MEDIUM')),
  sources         jsonb NOT NULL,
  human_verified  boolean NOT NULL DEFAULT false,
  UNIQUE (edge_type, src_id, dst_id),
  CHECK (src_id <> dst_id),
  -- type/kind shape rules enforced by CHECK:
  --   is_remake_of | is_official_dub_of | is_unofficial_remake_of : version -> version
  --   is_sequel_of | based_on                                     : work    -> work
)

candidate_edges (                     -- LLM proposals; NEVER read by ground-truth views
  candidate_id    uuid PRIMARY KEY,
  edge_type       text NOT NULL,     -- same enum as edges
  src_version_id  uuid REFERENCES version,   -- resolved when possible …
  dst_version_id  uuid REFERENCES version,
  src_title_raw   text, dst_title_raw text,  -- … raw strings when not
  supporting_sentence text NOT NULL,
  source_page     text NOT NULL,     -- wikipedia page + revision id (evidence pin)
  source_revision text NOT NULL,
  model_id        text NOT NULL, model_confidence real,
  extraction_run  text NOT NULL,     -- artifact/run hash (reproducibility stamp)
  status          text NOT NULL DEFAULT 'proposed'
                  CHECK (status IN ('proposed','confirmed','rejected')),
  reviewed_by text, reviewed_at timestamptz,
  promoted_edge_id uuid REFERENCES edges
)

conflicts (
  conflict_id     uuid PRIMARY KEY,
  entity_kind     text CHECK (entity_kind IN ('work','version','edge')),
  entity_id       uuid NOT NULL,
  field           text NOT NULL,     -- e.g. release_year, edge_type(dub-vs-remake)
  values          jsonb NOT NULL,    -- [{value, source}, ...] both sides preserved
  status          text NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved')),
  resolution      jsonb,             -- {rule|human, chosen_value, resolved_at}
)

plot_texts (                          -- P2 embedding corpus; content, not facts
  plot_id uuid PK, version_id uuid REFERENCES version,
  source text CHECK (source IN ('wikipedia','tmdb')),
  language text, text text NOT NULL,
  source_url text, revision_id text, license text NOT NULL, retrieved_at timestamptz
)
```

**The verification gate as schema (`DATA_SOURCES.md`, enforced):**

```sql
CREATE VIEW ground_truth_edges AS
  SELECT e.* FROM edges e
  WHERE (e.confidence = 'HIGH' OR e.human_verified)
    AND jsonb_array_length(e.sources) > 0
    AND NOT EXISTS (SELECT 1 FROM conflicts c
                    WHERE c.entity_kind = 'edge' AND c.entity_id = e.edge_id
                      AND c.status = 'open');
-- ground_truth_versions / ground_truth_works: same predicate shape.
```

Everything downstream — repository functions (§2.5), golden-fixture builder, `make graph-demo`,
and later P2 indexing — reads **only** the ground-truth views. `candidate_edges` is not an edge
table; it cannot leak by construction. MEDIUM rows pass the view (they are "live graph, flagged"
per the tier table) but the **fixture validator rejects them** (golden = HIGH/human-verified only).

**`SourceRef` (the `sources[]` element, pydantic-validated before any insert):**

```jsonc
{ "source": "wikidata" | "tmdb" | "imdb" | "wikipedia" | "human",
  "ref": "Q1618487" | "tmdb:266856" | "tt3417422" | "<page>@<revision>" | "reviewer",
  "field": "release_year",              // optional: which field this vouches for
  "retrieved_at": "2026-07-xx T…" }
```

### 2.3 Ingestion & build flow

```
seed_slice.yaml (committed: QIDs + curated truth per franchise)
  │
  ├─ 1. ingest_spine      Wikidata → raw snapshot → work/version skeletons,
  │                       typed edges (P144/P1877/P4969/P155/P156/P179), external IDs
  │                       (IMDb P345, TMDB P4947). Two-phase access (verified best practice,
  │                       §2.9): SPARQL returns QIDs only; per-entity detail via wbgetentities.
  │                       Descriptive User-Agent + gzip + 429/Retry-After backoff (Wikimedia
  │                       User-Agent policy + global API rate limits).
  ├─ 2. enrich_tmdb       per tmdb_id: ONE call with append_to_response=
  │                       translations,alternative_titles,credits (v3, Bearer auth)
  ├─ 3. load_akas         title.akas TSV filtered to slice tconsts → version_title rows
  │                       (columns per developer.imdb.com: titleId, ordering, title, region,
  │                       language, types, attributes, isOriginalTitle; '\N' = null)
  ├─ 4. fetch_plots       Wikipedia/MediaWiki API per language version → plot_texts
  │                       (revision id pinned from the page object; license recorded)
  ├─ 5. build_graph       entity resolution on the QID hub; per-field precedence table;
  │                       dub-vs-remake rule (lead-cast overlap + explicit statements);
  │                       disagreement → conflicts (never silent); match_key population
  ├─ 6. extract_candidates  [GPU session] LLM over Wikipedia lead/plot prose →
  │                       candidate_edges (+ persisted artifact: prompts, raw outputs, run hash)
  ├─ 7. review-candidates  human gate: confirm → promote to edges (human_verified=true) / reject
  ├─ 8. graph-report      coverage per franchise + candidate precision + lift beyond Wikidata
  └─ 9. build_golden      fixtures GS-01..GS-11 from ground-truth views → /evals/golden/*.yaml
                          + validator (CI, Tier-1)
```

Steps 1–5, 7–9 run on the laptop (API calls + Postgres — no neural op). Step 6 is the P1 GPU
session (DEC-0003: A100 40 GB, ~1–2 h): bring up the ephemeral instance (reusing the P0
`infra/gpu` lifecycle), serve the extraction model, run the batch, persist `candidate_edges` +
raw-output artifact, **destroy the instance**. Each step is idempotent (re-run = upsert keyed on
external IDs) and snapshot-first, so CI and rebuilds never re-hit APIs.

### 2.4 Normalization / transliteration (GS-07 slot ground, GS-11)

- `match_key(title) -> str`: Unicode NFC → script detection → native scripts transliterated to
  Latin via deterministic `indic-transliteration` (scheme per DEC-P1-5) → lowercase → strip
  diacritics/punctuation → collapse vowel-length doublets (`paapanaasam` → `papanasam`).
- Every title from every source (canonical, TMDB alt, IMDb aka, Wikipedia) lands in
  `version_title` with its `match_key`; **resolution = exact `match_key` hit, then rapidfuzz over
  the key index** (threshold tuned on GS-11 perturbations), returning scored candidates with
  `ambiguous=true` on collisions (GS-10) — exactly the `resolve_title` contract.
- Neural IndicXlit is a **contingency only** (rule-based expected to cover GS-11); if invoked it
  runs in the GPU session and its outputs are cached as `version_title(kind='transliteration')`.

### 2.5 Repository functions — proving the tool contract (TOOL_SCHEMA conformance)

P1 does not serve tools, but it must freeze a schema that the real graph can satisfy. The proof:
`sutradhar.graph.repository` implements the graph-backed tool signatures **exactly** as
`TOOL_SCHEMA.md` v0 defines them, as plain functions over the ground-truth views. P5 wraps these;
P4 generates synthetic calls against the same JSON Schema.

| Tool (v0) | P1 backing | Notes |
|---|---|---|
| `resolve_title(title, language?)` | `repository.resolve_title` via `version_title.match_key` + rapidfuzz | returns `candidates[] + ambiguous` (GS-10/11) |
| `get_work(work_id)` | `repository.get_work` | incl. `source_work`/`based_on` (GS-05) |
| `get_versions(work_id, scope, include_sequels)` | `repository.get_versions`; scope via `version.country`; `include_sequels` via recursive CTE over `is_sequel_of` (GS-06) | emits `relationship` labels + derived `is_original_of` presentation |
| `refine_filter(version_set, by)` | `repository.refine_filter` (SQL filter over version set; `era` resolved against `is_original`/year) | GS-08 backing |
| `search_by_plot` | **NOT implemented** (needs P2 retrieval + calibrated abstain) | schema frozen now; P2 implements |

**Schema changes proposed as part of the freeze** (to be noted in `DECISIONS.md`; no signature
breaks, v0 stays v0):
1. Add machine-readable **`docs/phases/tool_schema.v0.json`** (JSON Schema for each tool's
   params + result, with enums for `relationship`, `scope`, `era`) — the artifact tests, P4 data
   generation, and the tool-call-accuracy metric validate against. The `.md` remains the prose
   contract; a sync test keeps them consistent.
2. Tighten v0 field semantics discovered by implementation (e.g. `country`-based `scope`
   definition; `resolve_title.candidates[].score` = rapidfuzz-normalized 0–1). Wording-level
   only; any *signature* change found necessary during P1 bumps to v0.1 and is logged — none is
   currently anticipated.

### 2.6 LLM extraction layer (the R1 de-risker / portfolio feature)

- **Input:** Wikipedia lead + plot sections for slice films (and their linked "remake of"
  sentence contexts), chunked per section.
- **Prompt contract:** extract `{edge_type, src_title, dst_title, languages, supporting_sentence,
  confidence}` as JSON; the response is validated against a pydantic model — malformed output is
  dropped and counted, never "repaired" into the DB.
- **Resolution:** proposed titles are resolved to `version_id`s via `resolve_title`
  (`match_key`); unresolved proposals keep raw strings for the reviewer.
- **Client:** the P0 `LLMClient` (OpenAI-compatible, env-driven `LLM_BASE_URL`) — the extraction
  script does not know it's JarvisLabs. Model choice = DEC-P1-4.
- **Honesty metrics:** candidates proposed / confirmed / rejected; **verified edges added beyond
  Wikidata** (the recall lift); parse-failure rate. All in the `graph-report` and `BENCHMARKS.md`
  graph section.
- **Seeding note:** on the seed slice, Wikidata coverage of the flagship chains may already be
  complete — the extraction layer must still run and be measured (its value statement is the
  *mechanism + measured precision*; the recall lift becomes decisive at breadth scale-up).

### 2.7 Seed slice (proposed — confirm in §7 Q1; ground truth web-verified 2026-07-02, §2.9)

Committed as `data-pipeline/seed_slice.yaml` with per-franchise curated truth (the
graph-coverage denominator). Every row below was checked against live sources during grooming
and remains a **claim to re-verify during ingestion** (per the `GOLDEN_SET_SCENARIOS.md`
data-accuracy note); fixtures freeze only after source confirmation.

| Franchise / group | Records | Serves |
|---|---|---|
| Drishyam (2013 ml, Mohanlal, dir. Jeethu Joseph) + Drishya (kn 2014) + Drushyam (te 2014, Venkatesh) + Papanasam (ta 2015, Kamal Haasan) + Drishyam (hi 2015, Ajay Devgn) | 1 work, 5 versions | GS-01/03/07/08/11 |
| Drishyam foreign remakes: Dharmayuddhaya (si 2017), Sheep Without a Shepherd (zh 2019) [Korean/Indonesian/US versions reported — optional adds, verify] | +2 versions | GS-09A |
| **Drishyam 2 work (corrected):** Drishyam 2 (ml 2021, original) + Drishya 2 (kn 2021) + Drushyam 2 (te 2021, Venkatesh, dir. Jeethu Joseph) + Drishyam 2 (hi 2022) [+ Dharmayuddhaya 2 (si) reported — verify] | 2nd work, 4+ versions | GS-06 |
| Baahubali: The Beginning (2015) — **bilingual original shot simultaneously in te+ta** (Prabhas), dubbed hi/ml [+ more] | 1 work, 2 original-flagged versions + dub tracks | GS-04 |
| Devdas novella (literary_source) + Devdas (hi 1955, Bimal Roy) + Devdas (hi 2002) + **Devadasu (te 1953)** — sibling works `based_on` the novella; Devadasu additionally has a **Tamil dub _Devadas_** (a sibling adaptation with its own dub edge — proves the two edge types compose) | 1 source + 3 works | GS-05 |
| Vikram (1986 ta, Kamal Haasan) + Vikram (2022 ta, Kamal Haasan, dir. Lokesh Kanagaraj) | 2 unrelated works | GS-10 |
| **Manichitrathazhu chain (corrected — transitive):** Manichitrathazhu (ml 1993, original) → Apthamitra (kn 2004) → **Chandramukhi (ta 2005, a remake of Apthamitra, not of the ml original)**; plus Rajmohol (bn 2005) and Bhool Bhulaiyaa (hi 2007) | 1 work, 5+ versions; remake-of-a-remake edges | GS-09B + transitive-lineage test |
| Distractors: ~4–6 unrelated Indian films (mixed languages/actors, incl. another Mohanlal + another Kamal Haasan title) | noise floor | GS-02/03 realism |

**Modelling note (from the corrected data):** `is_remake_of` edges record the *proximate* source
(Chandramukhi → Apthamitra; Apthamitra → Manichitrathazhu). The Work groups the whole lineage, so
`get_versions` still returns the full set with the ultimate original flagged — and the proximate
edge is preserved evidence, not flattened away. GS-09B's "what's the original of Bhool Bhulaiyaa?"
resolves through the Work's `is_original` version, independent of edge depth.

### 2.8 Python vs Java

**All P1 code is Python** — SQLAlchemy/psycopg + Alembic (DEC-P1-2), httpx connectors, pydantic
validation, typer CLIs; this is data-pipeline + DB work squarely inside the CLAUDE.md Python
stack. The optional Java/Spring Boot gateway remains a **P5 grooming decision** ("decide in P5
grooming, don't assume"); a JVM adds nothing to ingestion and would fragment the single-package
layout (DEC-P0-2). **Recommendation: defer; P1 is Python-only.**

### 2.9 Web-verified source & tooling facts (accessed 2026-07-02)

External contracts and ground truth checked against live sources during grooming, per the P0
precedent (P0_SPEC §2.7), so the pipeline is built on real interfaces — consistent with the
mission's "rebuild from scratch" and data-licensing-maturity postures. These validate P1 inputs
only; they do **not** reopen settled decisions.

| Fact | Verified detail | Impact on P1 | Source |
|---|---|---|---|
| **Wikidata remake properties** | `P144` "based on" (with `P4969` "derivative work" as its inverse), `P1877` "after a work by"; sequel/series via P155/P156/P179; external-ID hub `P345` (IMDb) + **`P4947` (TMDB movie ID)** | Spine queries in `ingest_spine`; QID↔tmdb_id↔tconst linking keys | wikidata.org `Property:P144`, `Property_talk:P4969`, `Property:P4947` |
| **Wikidata/Wikimedia access etiquette** | Descriptive **User-Agent header required** (WMF User-Agent policy); gzip; on **429 honor `Retry-After`**; global Wikimedia API rate limits now apply across Action/REST APIs; community best practice for bulk work = **two-phase**: SPARQL for QIDs only, then `wbgetentities` for detail (avoids WDQS timeout/429 cascades) | Connector design in §2.3 step 1; snapshot-first so re-runs don't re-hit the endpoint | wikidata.org `Data access`; mediawiki.org `Wikimedia APIs/Rate limits`; WMF User-Agent policy |
| **TMDB API v3** | Docs at `developer.themoviedb.org`; auth = v3 `api_key` **or v4 Bearer token**; **`append_to_response`** bundles sub-requests → `GET /movie/{id}?append_to_response=translations,alternative_titles,credits` is **one call per film**; attribution required ("uses TMDB APIs but is not endorsed…") | `enrich_tmdb` = 1 request/film (rate-limit friendly); Bearer auth via `TMDB_API_KEY`; attribution string → `LICENSING.md` | developer.themoviedb.org (getting-started, append_to_response, auth) |
| **IMDb non-commercial datasets** | Documented at `developer.imdb.com/non-commercial-datasets/`; files at `datasets.imdbws.com`; gzipped TSV, UTF-8, header row, `\N` = null; `title.akas` columns: `titleId, ordering, title, region, language, types, attributes, isOriginalTitle`; **personal/non-commercial only** | `load_akas` parser contract + the `isOriginalTitle` corroboration signal; non-commercial limit → `LICENSING.md` | developer.imdb.com non-commercial datasets |
| **Wikipedia content access** | MediaWiki REST/Action APIs return the page object incl. **latest revision id and license**; content license **CC BY-SA 4.0** (attribution + share-alike); API/dump access is the sanctioned path (no HTML scraping); subject to the global rate limits above | `fetch_plots` pins `revision_id` + license per `plot_texts` row; attribution → `LICENSING.md` | mediawiki.org REST API reference; Wikimedia developer portal |
| **indic-transliteration (PyPI)** | `indic_transliteration.sanscript` actively maintained; supports Devanagari, Tamil, Malayalam, Telugu, Kannada, Bengali (+ more) and roman schemes incl. IAST/ISO-15919-style, HK, ITRANS, OPTITRANS; pure-Python deterministic (no neural op) | DEC-P1-5 option A is implementable as specced, laptop-safe | github.com/indic-transliteration/indic_transliteration_py; PyPI |
| **AI4Bharat IndicXlit (contingency)** | Transformer ~11M params, 21 Indic languages, roman↔native, trained on Aksharantar; published models are **CC BY-SA 4.0** | Stays the GPU-session contingency (DEC-P1-5); **if invoked, CC BY-SA share-alike/attribution must be added to `LICENSING.md`** | github.com/AI4Bharat/IndicXlit; ai4bharat model pages |
| **Drishyam ground truth** | Remakes: Drishya (kn 2014), Drushyam (te 2014), Papanasam (ta 2015), Drishyam (hi 2015); foreign: Dharmayuddhaya (si 2017), Sheep Without a Shepherd (zh 2019, altered ending); Korean/Indonesian/US versions reported | GS-01/GS-09A curated truth in `seed_slice.yaml` | en.wikipedia.org Drishyam, Drishyam (film series) |
| **Drishyam 2 ground truth (spec corrected)** | Drishyam 2 (ml 2021) remade as **Drishya 2 (kn 2021)**, **Drushyam 2 (te 2021, Jeethu Joseph, Venkatesh)**, **Drishyam 2 (hi 2022)**; Sinhala sequel-remake reported | GS-06 franchise truth grew by two versions vs the draft | en.wikipedia.org Drushyam 2, Drishyam (film series) |
| **Manichitrathazhu ground truth (spec corrected)** | Remade as Apthamitra (kn 2004); **Chandramukhi (ta 2005) is a remake of *Apthamitra***; also Rajmohol (bn 2005), Bhool Bhulaiyaa (hi 2007) | Remake-of-a-remake (transitive lineage) is now an explicit modelling rule + test (§2.7 note, §4) | en.wikipedia.org Manichitrathazhu |
| **Devdas ground truth (spec confirmed/enriched)** | Independent adaptations of the 1917 novella incl. Devadasu (**te 1953**, dubbed into Tamil as *Devadas*), Devdas (hi 1955), Devdas (hi 2002); many more exist (1928/1935/1936 …) | GS-05 sibling works + a dub edge *inside* a sibling adaptation; extra adaptations = natural breadth backlog | en.wikipedia.org Devadasu (1953 film) |
| **Baahubali ground truth (spec confirmed)** | Shot **simultaneously in Telugu and Tamil** (Prabhas et al.); released dubbed in Hindi, Malayalam (+ others) | Confirms the bilingual double-`is_original` rule + dub edges (GS-04) | en.wikipedia.org Baahubali: The Beginning |

> Anything above that drifts by implementation time is corrected in code/config and noted in the
> commit — never hardcoded against a stale assumption.

---

## 3. Decisions to make now

Only choices P1 requires that are **not** settled in `docs/DECISIONS.md`. (Settled and not
reopened: model stack DEC-0001; embedder A/B + vector store + chunking = P2 / DEC-0002; GPU/cost
DEC-0003; uv/layout/Make/LLM-client/GPU-lifecycle DEC-P0-1..5.) **All six accepted as recommended
and logged in `docs/DECISIONS.md` (2026-07-02)** — kept here with options + trade-offs as the
grooming record.

### DEC-P1-1 — Edge storage shape
- **A. Single polymorphic `edges` table (recommended).** One enum-typed table with
  `src/dst_kind` + CHECK constraints per type; traversal via recursive CTEs. *Pros:* one uniform
  provenance/confidence/gate/review pipeline for every edge type; one ground-truth view; adding an
  edge type is a value, not a table. *Cons:* polymorphic FKs are soft (mitigated by triggers/
  tests).
- **B. Two typed tables** (`version_edges`, `work_edges`) with hard FKs. *Pros:* referential
  integrity native. *Cons:* duplicated gate/review/provenance logic; the promotion path and views
  fork.
- **C. Graph extension (Apache AGE) / dedicated graph DB.** *Pros:* native traversal. *Cons:*
  reopens the settled "Postgres" call, new operational surface, overkill for a 2-hop graph.
- **Recommendation: A** — the verification gate and human-review pipeline are the product; keeping
  them single-path is worth soft FKs (hardened by a trigger + constraint tests).

### DEC-P1-2 — DB access + migrations
- **A. SQLAlchemy 2.0 (typed ORM) + Alembic (recommended).** *Pros:* typed models mypy-strict can
  see, standard migration story, testable, P2/P5 reuse the same session layer. *Cons:* ORM
  ceremony for what is mostly bulk upsert.
- **B. psycopg 3 + numbered raw-SQL migrations + tiny runner.** *Pros:* transparent SQL, zero
  abstraction. *Cons:* hand-rolled migration bookkeeping; typed-model duplication with pydantic.
- **C. SQLAlchemy Core (no ORM) + Alembic.** Middle path; still two model layers.
- **Recommendation: A** — the graph outlives P1 (P2 indexing, P5 tools read it); a typed, migrated
  schema is the maintainable default and Alembic autogenerate keeps diffs reviewable.

### DEC-P1-3 — Provenance (`sources[]`) representation
- **A. `jsonb` column per record/edge, pydantic-validated at the boundary (recommended).**
  *Pros:* provenance travels with the row (views, fixtures, tool results get it for free);
  matches the `TOOL_SCHEMA` result shape; no join fan-out. *Cons:* no DB-level FK to a sources
  registry; queries "all claims from source X" need jsonb ops.
- **B. Normalized `provenance` table** (`entity_kind, entity_id, source, ref, field, …`). *Pros:*
  relationally queryable, single shape. *Cons:* every read joins; every writer double-writes;
  more machinery than a ~30-record slice justifies.
- **C. Hybrid** (jsonb now, mirror table later if needed).
- **Recommendation: A** (with C's escape hatch noted) — the consumers (gate view, fixtures, tool
  results, UI citations) all want provenance *inline*.

### DEC-P1-4 — Extraction model for candidate edges (the P1 GPU job)
- **A. Gemma 4 E4B on the ephemeral A100 session (recommended).** *Pros:* already validated on
  vLLM in P0 (DEC-P0-5 evidence), ~$1–2 for the whole pass, zero new dependency, dogfoods the
  exact serving path P4 uses; extraction is gated by a human anyway, so model precision is a
  cost knob, not a correctness risk. *Cons:* a 4B extractor will propose noisier candidates
  (more human review time).
- **B. Frontier API (hosted).** *Pros:* highest extraction precision, no GPU rental. *Cons:* new
  external dependency + key, per-token cost, weakens the "our stack, rebuilt from scratch" story.
- **C. Sarvam-M 24B FP8 on a 40–48 GB card.** *Pros:* strong Indic reading. *Cons:* the DEC-0003
  cost profile of a 24B for a job a 4B + human gate covers; Sarvam-M's slot is P4 teacher.
- **Recommendation: A**, with B as the documented fallback **if** candidate precision on a
  spot-check falls below ~0.5 (making review time the real cost).

### DEC-P1-5 — Cross-script title normalization scheme (`match_key`)
- **A. Deterministic rule-based: native script → IAST/ISO-15919-style romanization
  (`indic-transliteration` `sanscript` — Devanagari/Tamil/Malayalam/Telugu/Kannada/Bengali
  verified supported, §2.9) → ASCII fold → lowercase → vowel-length collapse (recommended).**
  *Pros:* laptop-safe (pure Python, no neural op), reproducible, single indexed key, rapidfuzz
  handles residual drift (GS-11). *Cons:* imperfect for irregular popular spellings.
- **B. Neural IndicXlit romanization for every native-script title.** *Pros:* closest to how
  users actually romanize (Aksharantar-trained, 21 languages). *Cons:* a neural op → GPU session
  + cache for something rules mostly solve; harder to reproduce; models are **CC BY-SA 4.0**
  (adds a share-alike attribution obligation to `LICENSING.md`).
- **C. Multi-key storage** (store ISO-15919 + Harvard-Kyoto + popular-spelling variants per
  title). *Pros:* highest recall. *Cons:* index bloat; variants are better sourced from real AKA
  data (IMDb/TMDB already supply popular spellings).
- **Recommendation: A** — IMDb/TMDB AKAs already contribute the "popular spelling" variants into
  `version_title`, so the rule-based key + fuzzy match should clear GS-11; IndicXlit stays the
  measured contingency.

### DEC-P1-6 — Human-verification gate tooling
- **A. Typer CLI (`make review-candidates`) (recommended).** Shows supporting sentence + resolved
  entities, y/n/skip, writes `reviewed_by/at`, promotes on confirm. *Pros:* zero surface area,
  scriptable, testable; single-reviewer reality. *Cons:* less demo-shiny.
- **B. Minimal web page** (FastAPI + table). *Pros:* screenshots well for the portfolio. *Cons:*
  pre-empts P5's API surface; UI is P6's job; more code to test.
- **C. CSV export/import round-trip.** *Pros:* trivial. *Cons:* no audit trail integrity; easy to
  corrupt a promotion.
- **Recommendation: A** — the *gate semantics* (promotion sets `human_verified`, rejection is
  recorded, nothing bypasses) are the portfolio point; a screenshot of the CLI session serves the
  evidence need.

---

## 4. Test strategy

P1 touches neither retrieval nor generation, so **no Recall@k / RAGAS thresholds gate here**
(explicitly deferred to P2/P3). What gates P1: schema-enforced invariants, the named golden-set
graph regressions, gate enforcement, fixture validity, tool-schema conformance, and the
graph-coverage metric. All tests are **Tier-1** (no GPU, no model call): the extraction tests run
against a recorded fixture of model output, per ROADMAP §6.2.

### Unit tests (laptop/CI, no DB)
- `normalize/`: `match_key` idempotence; native-script → key equivalence (`பாபநாசம்` ↔
  "Papanasam"); GS-11 perturbations ("Papanaasam", "Drushyam", 1–2 char typos) resolve to the
  intended key via rapidfuzz above threshold; non-Indic scripts pass through.
- Dub-vs-remake rule as a pure function: same-lead-cast fixture → `is_official_dub_of`;
  disjoint-cast → `is_remake_of`; rule-vs-explicit-statement disagreement → conflict record,
  never a silent pick.
- Precedence table: per-field primary/fallback/conflict behaviour for each row of the
  `DATA_SOURCES.md` table; HIGH assignment requires ≥2 agreeing independent sources or the
  authoritative source.
- `SourceRef`/confidence pydantic validation: rejects empty `sources[]`, unknown source ids.
- Extraction output parsing: valid JSON → `CandidateEdge`; malformed → dropped + counted
  (recorded-fixture based).
- Connectors: response-parsing units against committed snapshot samples (no live API in CI).

### Integration tests (Postgres via compose; Tier-1 CI service container)
- **Schema constraints:** edge-type shape CHECKs (a `version→work` remake edge is rejected);
  self-edge rejected; duplicate `(edge_type, src, dst)` rejected; QID uniqueness.
- **Gate enforcement:** a CANDIDATE row can never appear via ground-truth views (by
  construction); a MEDIUM edge with an open conflict is excluded; promoting a candidate creates a
  `human_verified` edge and links `promoted_edge_id`; rejecting never writes to `edges`;
  golden-fixture validator rejects MEDIUM-backed fixtures.
- **Idempotent re-ingest:** running a connector twice produces no duplicates and no spurious
  conflicts.
- **Repository/tool backing:** `get_versions` scope filtering (indian/all/foreign), sequel
  traversal, `refine_filter` era/actor/language semantics, `resolve_title` ambiguity flag.

### Named golden-set regression tests (the ones this phase must ship)
Instantiated at the **graph layer** (their retrieval/generation counterparts re-gate in P2/P3):

| Test | Fixture | Asserts |
|---|---|---|
| `test_gs01_version_set_recall` | GS-01 | Drishyam Work's ground-truth Indian version set = exactly {ml 2013 original-flagged, kn 2014, te 2014, ta 2015, hi 2015}, each `is_remake_of`; **version-set recall = 1.0** |
| `test_gs06_franchise_version_set_recall` | GS-06 | `get_versions(include_sequels=true)` traverses `is_sequel_of` → Drishyam 2 + its remakes; sequel vs remake labels never conflated; recall = 1.0 |
| `test_gs02_no_hallucinated_movie` | GS-02 | decoy fixtures resolve to **no** graph record (`resolve_title` empty / no plot match candidate) and fixture `expected: NO_MATCH` validates; nothing in the graph is returnable for them |
| `test_gs04_dub_vs_remake` | GS-04 | every Baahubali language track is `is_official_dub_of` with shared lead cast; zero `is_remake_of` edges inside the Work; bilingual double-original encoded |
| `test_gs05_sibling_vs_remake` | GS-05 | Devdas adaptations are sibling Works each `based_on` the literary_source node; **zero** `is_remake_of` edges among them |
| `test_gs10_false_merge` | GS-10 | Vikram 1986 and Vikram 2022 are distinct Works with distinct version sets; `resolve_title("Vikram")` returns both with `ambiguous=true`; **false-merge rate = 0** |
| `test_gs09_scoping` | GS-09 | foreign Drishyam versions exist but are excluded at `scope="indian"`, returned at `scope="foreign"`; Manichitrathazhu flagged original of Bhool Bhulaiyaa |
| `test_gs09_transitive_lineage` | GS-09B (+§2.7 note) | remake-of-a-remake encoded: Chandramukhi's `is_remake_of` edge points at **Apthamitra** (proximate source preserved), yet `get_versions` on the Work returns the full lineage with **Manichitrathazhu** the sole `is_original` — "original of Bhool Bhulaiyaa" resolves correctly regardless of edge depth |

### Tool-schema conformance tests (required by the working agreement)
- `test_tool_schema_json_valid`: `tool_schema.v0.json` is valid JSON Schema; tool names + params
  exactly match the `.md` blocks (sync test — doc drift fails CI).
- `test_golden_expected_tool_calls_validate`: every `expected_tool_calls` sequence in GS-07/GS-08
  fixtures validates against `tool_schema.v0.json` — **no hallucinated tool or parameter names**
  can be committed into the golden set. (P3/P4 reuse this validator against *model-emitted*
  calls.)
- `test_repository_matches_tool_schema`: repository function signatures/result shapes round-trip
  through the JSON Schema (the "contract is satisfiable" proof, §2.5).

### Metrics that gate P1 (graph, not retrieval)
- **Graph coverage = 1.0** on the flagship franchises (Drishyam family incl. sequel + foreign,
  Baahubali, Devdas, Vikram pair, Manichitrathazhu chain) vs the curated truth in
  `seed_slice.yaml`.
- **Fixture validity = 100%**: every GS-01..GS-11 fixture passes the validator (HIGH/
  human-verified only, no open conflicts, sources populated, IDs confirmed).
- **Candidate precision reported** (confirmed/proposed) + edges-added-beyond-Wikidata: **report,
  not gate** (the honest number, whatever it is, per `DATA_SOURCES.md`).

---

## 5. Task breakdown (ordered, independently committable)

1. **Schema + migrations:** Alembic setup; `work`/`version`/`version_title`/`person`/
   `version_cast`/`edges`/`candidate_edges`/`conflicts`/`plot_texts` + enums, CHECKs, ground-truth
   views; constraint integration tests. (`make db-migrate`)
2. **Domain + provenance types:** pydantic `SourceRef`/`Confidence`/node/edge models; gate
   predicate helpers; unit tests.
3. **Seed slice committed:** `data-pipeline/seed_slice.yaml` (QIDs + curated per-franchise truth)
   + loader + schema validation test.
4. **Wikidata connector + spine ingest:** SPARQL queries, raw-snapshot persistence, skeleton
   works/versions/edges upsert; snapshot-based parse tests.
5. **TMDB connector + enrichment:** translations/alt-titles/credits; precedence-table application
   begins; conflict writes on disagreement; tests.
6. **IMDb akas loader:** slice-filtered TSV ingest → `version_title`; non-commercial note wired
   into LICENSING draft; tests.
7. **Wikipedia plot fetch:** REST, revision-pinned, license recorded → `plot_texts`; tests.
8. **Normalization/transliteration:** `match_key` + rapidfuzz resolution + `version_title`
   population; GS-11 unit suite.
9. **Graph builder:** QID-hub entity resolution, dub-vs-remake rule, conflicts queue,
   idempotency; GS-04/GS-05/GS-10 integration tests go green here.
10. **Repository functions:** `resolve_title`/`get_work`/`get_versions`/`refine_filter` over
    ground-truth views; GS-06/GS-09 traversal + scoping tests.
11. **LLM extraction (GPU session):** prompt + pydantic contract + batch script via `LLMClient`;
    recorded-output fixture for CI; run the real pass in one ephemeral A100 session (create →
    extract → persist artifact + `candidate_edges` → destroy); parse-failure metric.
12. **Human review gate:** typer CLI, promotion/rejection semantics, audit fields; gate
    enforcement tests; run the real review pass.
13. **Reports:** `make graph-report` → coverage per franchise, candidate precision, lift beyond
    Wikidata → `docs/BENCHMARKS.md` graph section + `data-pipeline/README.md`.
14. **Golden set:** fixture builder + validator + all GS-01..GS-11 fixtures under `/evals/golden/`
    (incl. `expected_tool_calls` on GS-07/GS-08); named regression tests wired into Tier-1 CI.
15. **Tool schema freeze:** `tool_schema.v0.json` + sync/conformance tests; `TOOL_SCHEMA.md`
    status → FROZEN v0; `DECISIONS.md` entry.
16. **Docs close-out:** `docs/LICENSING.md`; module READMEs (`data-pipeline`, `evals`);
    `DECISIONS.md` DEC-P1-1..6 finalized; `PORTFOLIO.md` bullet; `make graph-demo` (30-second
    path); `.env.example` additions if any (none anticipated beyond existing `TMDB_API_KEY`).

Each task = one reviewable conventional commit (or a small stack) on `feature/p1-remake-graph`.

---

## 6. Definition of Done (instantiated from CLAUDE.md)

- [ ] Code complete and matching this approved spec (all §5 tasks).
- [ ] Unit + integration tests passing (ruff, mypy-strict, pytest; Tier-1 CI green incl. the
      Postgres service container).
- [ ] **Named regressions green:** `test_gs01_version_set_recall`,
      `test_gs06_franchise_version_set_recall`, `test_gs02_no_hallucinated_movie`,
      `test_gs04_dub_vs_remake`, `test_gs05_sibling_vs_remake`, `test_gs10_false_merge`,
      `test_gs09_scoping`, plus the tool-schema conformance tests.
- [ ] **Verification gate enforced** and proven by tests; conflicts queue populated, never
      silently resolved; zero unresolved conflicts behind any golden fixture.
- [ ] **Graph coverage = 1.0** on flagship franchises; **candidate precision + Wikidata lift
      reported.** Benchmark tables: Table 1 (retrieval) and Table 2 (generation) are **not
      applicable in P1** — instead `docs/BENCHMARKS.md` gains a **"Graph coverage & extraction
      lift"** section (P1's evidence row, with the §6.1 reproducibility stamp: code SHA, snapshot
      hashes, extraction model revision, prompt hash).
- [ ] Seed golden set for **all GS-01..GS-11** frozen under `/evals/golden/`, validator-clean.
- [ ] **Tool schema v0 FROZEN** (`.md` + `.v0.json`), logged in `DECISIONS.md`.
- [ ] `docs/LICENSING.md` created; `DECISIONS.md` updated (DEC-P1-1..6 + freeze entry); module
      READMEs updated.
- [ ] Runs from scratch: fresh clone + `.env` → `make up && make db-migrate && make ingest-seed
      && make graph-report` (extraction step optional/recorded when no GPU).
- [ ] 30-second demo: `make graph-demo` prints the cited, relationship-labelled Drishyam version
      set with the original flagged.
- [ ] GPU hygiene: extraction session ephemeral (create→run→destroy), cost logged against the
      DEC-0003 envelope (~1–2 h).
- [ ] Resume-ready quantified bullet drafted in `docs/PORTFOLIO.md` (e.g. edges verified, candidate
      precision, coverage).

---

## 7. Open questions — RESOLVED (2026-07-02, recommendations accepted)

1. **Seed slice (§2.7): CONFIRMED at ~30 records**, catalog breadth stays out of P1
   (vertical-slice strategy). **Rajmohol (bn 2005) is included** — it completes the
   Manichitrathazhu lineage at negligible cost and strengthens the transitive-chain test.
   The reported Korean/Indonesian/US Drishyam versions and Dharmayuddhaya 2 are **conditional
   adds**: ingested only if a clean structured record (Wikidata QID or TMDB id) confirms them
   during ingestion; otherwise they go to the breadth backlog, never hand-invented — consistent
   with the verification gate.
2. **Golden-set size: CONFIRMED** — P1 ships 2–3 fixtures per category (~25–35 total) covering
   all GS-01..GS-11 against the seed slice; expansion toward ≥100 fixtures rides P2/P3 and the
   breadth scale-up. Rationale: fixtures must be HIGH/human-verified against ingested records,
   so fixture count is bounded by slice size in P1.
3. **Repository functions: CONFIRMED — implemented in P1** (§2.5). They prove the frozen tool
   contract is satisfiable against the real schema (the "facts come from tools" design rule),
   double as fixture-build machinery, and give P5 a tested query layer to wrap. `search_by_plot`
   remains P2.
4. **DEC-P1-1..6: ACCEPTED as recommended** (single polymorphic `edges` table; SQLAlchemy 2.0 +
   Alembic; inline jsonb `sources[]`; Gemma 4 E4B extraction with frontier-API fallback below
   ~0.5 candidate precision; rule-based `match_key` + rapidfuzz with IndicXlit contingency;
   typer-CLI review gate) — logged in `docs/DECISIONS.md` (2026-07-02).
