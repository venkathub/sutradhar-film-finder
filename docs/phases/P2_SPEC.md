# P2 Spec — RAG baseline + retrieval eval (the green-light gate for P4)

> Phase spec for **P2** of the Sutradhar roadmap. Grounded in `docs/ROADMAP.md` (P2 entry/exit
> criteria, §2 compute placement, §6 guardrails), `CLAUDE.md` (DoD, "RAG baseline + eval harness
> before fine-tuning"), `docs/DECISIONS.md` (DEC-0001..0003, DEC-P0-*, DEC-P1-* — settled, not
> reopened; DEC-0002 is *resolved by measurement here*), `docs/DATA_SOURCES.md` (CANDIDATE
> excluded from retrieval), `docs/GOLDEN_SET_SCENARIOS.md` (GS-01..GS-11 + the CI gate summary),
> and `docs/phases/TOOL_SCHEMA.md` (**FROZEN v0** — `search_by_plot` is schema-frozen and
> **implemented in this phase**, conforming, no version bump).
>
> **Status:** **COMPLETE — 2026-07-02. Exit gate met on the first pass; P4 green-lit.**
> Recall@10 = **1.000** (gate ≥ 0.90) in **all six** ablation cells; version-set recall
> **= 1.0 on GS-01 and GS-06** (the Papanasam/Drishyam row is a measured benchmark entry);
> NO_MATCH abstention calibrated (θ = 0.151747, **0 false accepts** on GS-02 + all 12 held-out
> test negatives, NO_MATCH recall 1.0 / precision 0.75). Winner: **1024tok_15pct / depth 20**
> (DEC-P2-3/4 measured values recorded); **DEC-0002 → Accepted** — BGE-M3 met the gate, the 9B
> challenger leg was skipped per Q1 (task 13 = no-op). One ephemeral A100 session
> (~10 GPU-minutes ≈ **$0.22**, 833 texts embedded, 44,217 rerank pairs) produced every neural
> artifact; sealed run `20260702T135315Z-f6583183` pinned as `RETRIEVAL_RUN`; Tier-1 CI
> recomputes every gating metric from the committed run artifact
> (`evals/retrieval_runs/…json`, DEC-P2-6). Evidence: `docs/BENCHMARKS.md` Table 1,
> `rag-engine/README.md` (ablation table + reproduce-from-scratch), `make rag-demo` (GPU off).
>
> **Execution deviations (all logged):** (1) GPU-job transport = **HF Hub relay** — a fork the
> spec left open, user-approved and logged as **DEC-P2-7**; consequence: `embed_and_score.py` is
> self-contained and the instance startup script carries a scoped HF token (mitigations in the
> DEC). (2) The planned `gpu` uv dependency group (§2.7) was **dropped** (DEC-P2-7 amendment):
> FlagEmbedding⇒`transformers<5`⇒`huggingface-hub<1.0` is unresolvable in one lockfile with the
> laptop's `hub>=1.0`; the startup script is the authoritative instance pin. (3) Chunk sizes use
> a deterministic char-class token **estimator**, not the XLM-R tokenizer (keeps the laptop
> neural-free per §2.7; sizes are ablation brackets — documented in `sutradhar.rag.chunking`).
> (4) `sutradhar.evals.report` / `make retrieval-report` (§2.1 row) was **not built**: Table 1 is
> written from `make retrieval-eval`'s output + the committed artifact, which already carries the
> stamp — a separate report script had no non-redundant consumer. (5) DEC-P2-5's
> zero-false-reject constraint was **measured infeasible** (code-mixed positives interleave with
> negatives on raw cross-encoder scores); resolution recorded in DEC-P2-5: no-hallucination
> outranks no-false-reject — 4 positives return `abstain=true` with correct results (all
> R@5 = 1.0), a named quantified P4 headroom target.
>
> _History: DRAFT + web-research pass + APPROVED (2026-07-02, all §3 recommendations accepted as
> DEC-P2-1..5; Q1/Q2 resolved, Q2 = DEC-P2-6). Executed same day: tasks 1–12, 14, 15 delivered in
> order (13 skipped by rule); 14 conventional commits on `feature/p2-rag-baseline`._
>
> **Current repo state (grounding).** P1 is complete: Postgres graph (`work`/`version`/`edges`/
> `version_title`/`plot_texts` + gate views `ground_truth_*`), 25 golden fixtures across
> GS-01..GS-11 (validator-clean, HIGH/human-verified only), repository functions for 4 of 5 v0
> tools (`search_by_plot` deliberately absent — asserted by
> `test_search_by_plot_not_implemented_yet`), flagship graph coverage 1.00, ITRANS `match_key` +
> rapidfuzz threshold 0.80 (DEC-P1-5), pgvector image already in compose
> (`pgvector/pgvector:0.8.4-pg17` — extension not yet enabled), `EMBED_MODEL`/`RERANK_MODEL` env
> pins in place, snapshot-artifact pattern (`data/raw/<src>/<ts>/ + MANIFEST.sha256`) established.
> The embedding corpus exists: **52 revision-pinned Wikipedia plot pages** (27 en + 25 native:
> ml/ta/te/hi/kn/si/zh; extract length min 1.1k / median 7.9k / max 53.6k chars — so whole-plot
> embedding does NOT fit BGE-M3's 8192-token window for the tail; chunking is required, not
> optional). The pinned pgvector 0.8.4 ships dense `vector`, `halfvec`, **and the `sparsevec`
> type** (native since 0.7.0) with in-DB inner-product/cosine operators — both hybrid legs can
> live in Postgres. **Nothing is embedded yet; there is no index, no retrieval, no retrieval
> metric.**

---

## 1. Scope

### In scope

1. **Chunk/document corpus build** from the **gate-visible graph only** (`ground_truth_versions`
   ⋈ `plot_texts`): plot prose chunked with per-Version metadata attached, plus one **metadata
   card** document per Version (title/AKAs/language/year/lead cast/relationship) so cast-anchored
   and title-adjacent queries have a dense target. CANDIDATE edges and conflict-hidden records are
   excluded **by construction** (they never appear in the views — `DATA_SOURCES.md` verification
   gate).
2. **Chunk schema in Postgres** (`chunks` + `chunk_embeddings` tables; pgvector extension enabled
   via migration) — schema in §2.3. Vector store choice is **DEC-P2-1** (§3).
3. **One batched GPU session** (DEC-0003: A100 40 GB, ~1–3 h) that produces **all** neural
   outputs as versioned artifacts:
   - corpus embeddings — BGE-M3 **dense + sparse lexical weights** — for **every chunking-ablation
     config** (the corpus is ~52 docs; extra configs are minutes, not hours);
   - query embeddings for every golden retrieval fixture **and** the held-out negative set;
   - the **full query × chunk cross-encoder score matrix** (`bge-reranker-v2-m3`) per config —
     ~10⁴–10⁵ pairs, minutes on the A100. Precomputing the full matrix makes fusion method, rerank
     depth, and top-k **free laptop-side parameters**: every ablation and every CI run is computed
     from stored scores with zero further GPU time (ROADMAP §2 compute placement).
4. **Hybrid retrieval pipeline** (laptop-side over stored artifacts): dense (pgvector cosine) +
   sparse (BGE-M3 lexical-weight overlap) → fusion (**DEC-P2-2**) → cross-encoder rerank
   (**DEC-P2-4**) → chunk→Work aggregation → calibrated abstention (**DEC-P2-5**) → Work results.
   The existing `resolve_title` (match_key + rapidfuzz) is the third, title channel; P2 wires it
   into the same evaluation, not a new implementation.
5. **`search_by_plot` implemented** in `sutradhar.graph.repository`, conforming exactly to frozen
   TOOL_SCHEMA v0 (params `description`, `top_k=10`; result `{results[], abstain}`). The P1 test
   asserting its absence is flipped to a conformance round-trip test. **No tool-schema change is
   needed — v0 stays v0** (wording-only status note in `TOOL_SCHEMA.md` that P2 landed the
   implementation).
6. **Retrieval eval harness** (`sutradhar.evals.retrieval`): Recall@1/5/10, MRR@10,
   **version-set recall** (retrieval → `get_versions` → expected-set coverage), computed per slice
   — plot-only (GS-03), flagship (GS-01), franchise (GS-06, `include_sequels=true`), code-mixed
   (GS-07, raw query → dense; no LLM intent parsing in P2), fuzzy/title (GS-11), negatives
   (GS-02). Ablation runner over chunking × fusion × rerank-depth from stored scores.
7. **NO_MATCH abstention calibration (GS-02):** author a **held-out negative set** (~24 queries:
   out-of-catalog plots + real-but-uncatalogued titles, GS-02-style, verified
   absent-from-slice-by-construction), split 50/50 calibration/test; tune the abstention threshold
   on the calibration half; report NO_MATCH precision/recall on the test half; **gate: zero false
   accepts on the GS-02 fixtures**. Threshold recorded in `DECISIONS.md`.
8. **Embedding A/B per DEC-0002's settled decision rule:** BGE-M3 is the default; the 9B
   `bge-multilingual-gemma2` leg runs **only if** BGE-M3 misses the exit gate (see open question
   Q1). Either way DEC-0002 flips to **Accepted** with measured numbers. **Why our gate, not a
   leaderboard, decides:** MIRACL — the multilingual benchmark behind both models' cross-lingual
   claims — covers Hindi/Bengali/Telugu but **not Tamil, Malayalam, or Kannada**; published
   numbers structurally under-cover the languages the mission lives on (Papanasam is Tamil, the
   original is Malayalam). The golden-set gate is the only measurement that answers our question.
9. **Exit gate enforcement: Recall@10 ≥ 0.90** across retrieval fixtures (GS-01/03/06/07/11) and
   **version-set recall = 1.0 on GS-01 and GS-06**. If unmet, iterate retrieval (chunking → fusion
   → 9B A/B, in that order); **P4 does not start**.
10. **Tier-1 CI regression gate:** a compact committed **retrieval-run artifact** (per-query ranked
    candidates + all channel scores, JSON, < 5 MB) lets CI recompute every retrieval metric with no
    GPU and no DB; a metric regression blocks merge.
11. **`docs/BENCHMARKS.md` Table 1 populated** (+ the Papanasam version-set row) with the ROADMAP
    §6.1 reproducibility stamp (embedder id+revision, index/chunker version, reranker, golden-set
    version, code SHA, artifact hashes).
12. **30-second demo:** `make rag-demo` — replays the Tanglish GS-07a and plot-only GS-03a queries
    from recorded artifacts through the full pipeline and prints the cited, relationship-labelled
    version set. Works with the GPU **off** (artifacts), which is itself the graceful-degradation
    story.
13. `/rag-engine` README (purpose, architecture, how to run, results); DECISIONS entries
    DEC-P2-1..6 (logged at grooming; measured values filled during execution);
    `docs/PORTFOLIO.md` bullet.

### Non-goals (explicit — prevents scope creep)

- **No generation.** No LLM answers, no prompts, no faithfulness/relevancy metrics — retrieval
  returns Works + version sets; sentence-level answering is P3/P5. (The demo prints structured
  results, not prose.)
- **No RAGAS / Langfuse / MLflow wiring.** P3. P2 metrics are computed by the pytest/eval harness
  and written to `BENCHMARKS.md` by a report script (same pattern as P1's `graph-report`).
- **No fine-tuning, no synthetic data.** P4 — and P4 is gated on this phase's exit criteria.
- **No FastAPI routes, no tool serving, no live query API.** P5. `search_by_plot` is a repository
  function; its live free-text path (which needs a GPU embedder) is exercised in P5's GPU-on
  integration tests. In P2 it runs against recorded query embeddings.
- **No embedding/reranking/neural-anything on the laptop or in CI** (ROADMAP §2). Laptop + CI
  consume persisted artifacts; only pgvector distance math (dense cosine + sparsevec inner
  product — SQL, not a model) runs locally.
- **No 24/7 embedder.** The GPU session is batch: embed → score → persist → **destroy** (the
  DEC-P0-5 ephemeral lifecycle, reused).
- **No catalog breadth.** Same ~30-record seed slice; scaling is post-vertical-slice.
- **No re-litigating settled decisions:** embedder default + A/B rule (DEC-0002), reranker
  (bge-reranker-v2-m3), GPU SKU (DEC-0003), match_key/ITRANS + fuzzy threshold 0.80 (DEC-P1-5),
  edge storage (DEC-P1-1), gate-view predicate (DEC-P1-7).
- **No tool-schema version bump.** `search_by_plot` implements frozen v0 exactly; any drift is a
  CI failure, not a renegotiation.
- **No conversational state, no backtracking logic** (P5), **no UI** (P6), **no Java** (deferred
  to P5 grooming per `CLAUDE.md`; see §2.8).
- **No Redis caching** — nothing is served yet; the cache has no consumer until P5.

---

## 2. Design

### 2.1 Component breakdown

| Component | Code (import) | Entrypoint | Runs on | Purpose |
|---|---|---|---|---|
| Chunk schema + migration | `sutradhar.graph.schema` (+ Alembic) | `make db-migrate` | laptop | `chunks`, `chunk_embeddings`, `CREATE EXTENSION vector` |
| Corpus builder + chunker | `sutradhar.rag.corpus`, `sutradhar.rag.chunking` | `rag-engine/build_corpus.py` | laptop | Gate-visible docs → deterministic chunks + metadata cards |
| Embedding/rerank GPU job | `sutradhar.rag.gpu_jobs` | `rag-engine/embed_and_score.py` | **GPU session** | BGE-M3 dense+sparse per config; query embeds; full rerank matrix → artifacts |
| Artifact store | `sutradhar.rag.artifacts` | — | laptop/CI | Versioned runs, MANIFEST.sha256, load/verify (P1 snapshot pattern) |
| Index loader | `sutradhar.rag.index` | `rag-engine/load_index.py` | laptop | Artifacts → pgvector rows (`chunk_embeddings`) |
| Sparse scoring + fusion | `sutradhar.rag.{sparse,fusion}` | — | laptop/CI | Query lexical-weight → sparsevec literal; in-DB `<#>` scoring; RRF k=60 fusion (pure SQL + math) |
| Retriever pipeline | `sutradhar.rag.retrieve` | — | laptop/CI | dense+sparse+title → fuse → rerank → aggregate → abstain |
| Tool implementation | `sutradhar.graph.repository.search_by_plot` | — | laptop/CI | v0-conformant wrapper over the retriever |
| Eval harness + ablations | `sutradhar.evals.retrieval` | `evals/run_retrieval_eval.py` | laptop/CI | Recall@k/MRR/version-set recall per slice; ablation grid; run artifact |
| Abstention calibration | `sutradhar.evals.calibration` | `evals/calibrate_no_match.py` | laptop | Threshold on calibration negatives; P/R on test negatives |
| Benchmark report | `sutradhar.evals.report` | `make retrieval-report` | laptop | Table 1 rows + reproducibility stamp |
| Demo | — | `make rag-demo` | laptop | GS-07a/GS-03a replay → cited version set |

### 2.2 Chunk/document model

Two document kinds per gate-visible Version:

1. **Plot chunks** — from `plot_texts` (Wikipedia primary, TMDB overview fill), recursive
   paragraph-boundary chunking (size/overlap = DEC-P2-3), each chunk prefixed with a **metadata
   header** so every embedded unit is self-identifying:
   `"{title} ({year}, {language_name}){ — remake of {original_title} ({orig_year}, {orig_lang})}. "`
   The header is what lets a Papanasam chunk carry its Drishyam lineage into dense space.
2. **Metadata card** — one synthetic document per Version: canonical title + AKA titles +
   language/year + lead cast + director + relationship label. This is the dense target for
   cast-anchored queries ("that Kamal Haasan movie …", GS-01/GS-10) whose plot text may not
   mention actors, and a cheap recall backstop for title-ish queries.

Native-script plots (25 of 52 pages) are chunked and embedded as-is — BGE-M3 is multilingual;
their headers stay English + native title (cross-lingual anchor).

### 2.3 Schema (Alembic migration; pgvector enabled)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  chunk_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  version_id    uuid NOT NULL REFERENCES version(version_id),
  work_id       uuid NOT NULL REFERENCES work(work_id),      -- denormalized for aggregation
  plot_id       uuid REFERENCES plot_texts(plot_id),         -- NULL for metadata cards
  kind          text NOT NULL CHECK (kind IN ('plot','metadata_card')),
  seq           int  NOT NULL,                               -- order within source doc
  text          text NOT NULL,                               -- header + body (what was embedded)
  language      text,
  chunker       text NOT NULL,                               -- e.g. 'recursive_para'
  chunk_config  text NOT NULL,                               -- e.g. '512tok_15pct' (ablation key)
  content_hash  text NOT NULL,                               -- sha256(text) — determinism + dedupe
  license       text NOT NULL,                               -- carries CC BY-SA from plot_texts
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (version_id, kind, chunker, chunk_config, seq)
);

CREATE TABLE chunk_embeddings (                              -- separate: supports the A/B + re-embeds
  chunk_id      uuid NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  embed_model   text NOT NULL,                               -- id+revision, e.g. 'BAAI/bge-m3@<sha>'
  index_version text NOT NULL,                               -- artifact run id
  dense         vector(1024) NOT NULL,                       -- 1024 = BGE-M3; dim per model row
  sparse        sparsevec NOT NULL,                          -- BGE-M3 lexical weights over the
                                                             -- 250,002-dim XLM-R vocab; a chunk has
                                                             -- ~10²–10³ non-zero tokens, far under
                                                             -- sparsevec's 16,000-nnz storage cap
  PRIMARY KEY (chunk_id, embed_model, index_version)
);
-- Sparse scoring runs IN-DB: `sparse <#> :query_sparse` (negative inner product, native since
-- pgvector 0.7.0) — no app-side sparse math. Indexes: exact scan is fine at seed-slice scale
-- (~10² chunks); at catalog scale add HNSW (dense: 1024 ≤ 2000-dim limit; sparse: HNSW's
-- 1,000-nnz limit would need weight-pruning) — noted, not built (YAGNI at 300 rows).
```

Provenance chain: `chunk → plot_texts (revision-pinned, licensed) → version → sources[]` — every
retrieved chunk can cite its Wikipedia revision, satisfying the grounding clause of the story.

### 2.4 Retrieval flow

```
query
  ├─ title channel: match_key → rapidfuzz over version_title      (exists; DEC-P1-5, θ=0.80)
  ├─ dense channel: query embedding (artifact | GPU endpoint) → pgvector cosine top-N chunks
  └─ sparse channel: BGE-M3 query lexical weights → in-DB sparsevec inner product (<#>) top-N
        ↓
  fusion (DEC-P2-2: RRF, k=60) → fused top-M chunks
        ↓
  cross-encoder rerank (bge-reranker-v2-m3, sigmoid-normalized [0,1] scores from the
  precomputed matrix | GPU endpoint)
        ↓
  chunk → Work aggregation (max chunk score per work), rank works
        ↓
  abstention: top sigmoid score < θ_no_match (DEC-P2-5, calibrated) → abstain=true (GS-02)
        ↓
  results[{work_id, canonical_title, language, year, score}]      (search_by_plot v0 shape)
        ↓ (eval + demo continue)
  get_versions(work_id, scope, include_sequels) → typed, original-flagged version set (P1 repo)
```

**Version-set recall** is measured over this whole flow: query → top-1 resolved Work →
`get_versions` → fraction of the fixture's expected versions present with correct labels. This is
what makes the Papanasam row an end-to-end retrieval claim, distinct from P1's graph-coverage
metric (different denominator — never blended, per `BENCHMARKS.md`).

### 2.5 Key interfaces and contracts

```python
class EmbeddingProvider(Protocol):        # sutradhar.rag.embeddings
    def embed(self, texts: list[str]) -> list[DenseSparse]: ...
# Implementations: ArtifactEmbeddings (laptop/CI — recorded vectors keyed by sha256(text);
# raises MissingArtifact on unseen text, never silently degrades) and, later, RemoteEmbeddings
# (OpenAI-compatible /v1/embeddings against an env-driven EMBED_BASE_URL — P5 live path; the
# DEC-P0-4 status:"off" contract applies).

class RerankProvider(Protocol):           # same artifact-first pattern
    def score(self, query: str, chunks: list[ChunkRef]) -> list[float]: ...

@dataclass(frozen=True)
class RetrievalConfig:                    # every knob is data → ablatable + stampable
    chunk_config: str; fusion: str; dense_top_n: int; sparse_top_n: int
    rerank_depth: int; top_k: int; no_match_threshold: float
    embed_model: str; index_version: str

def search_by_plot(session, description: str, top_k: int = 10, *,
                   retriever: Retriever) -> SearchByPlotResult   # mirrors tool_schema.v0.json
```

Contracts enforced by test: `SearchByPlotResult` round-trips against
`docs/phases/tool_schema.v0.json` (same conformance layer as DEC-P1-8); the retriever reads
**only** `ground_truth_*` views (CANDIDATE invisible); `ArtifactEmbeddings` verifies
MANIFEST.sha256 before serving a single vector.

### 2.6 GPU session plan (one batched job; DEC-0003 A100 40 GB)

Reuses the `infra/gpu/jarvis.py` ephemeral lifecycle: create → run `rag-engine/embed_and_score.py`
→ pull artifacts → destroy. Steps: (1) embed all chunk configs, corpus + metadata cards, BGE-M3
dense+sparse (FlagEmbedding); (2) embed all golden retrieval queries + all held-out negatives;
(3) score the **full** query×chunk cross-encoder matrix per chunk config; (4) write
`data/artifacts/retrieval/<run_id>/` (embeddings `.npy` + parquet scores + MANIFEST.sha256,
git-ignored) and the compact committed run summary under `evals/retrieval_runs/<run_id>.json`.
Estimated ≤ 1 h ≈ $1. The optional 9B A/B leg (DEC-0002) adds one more embed pass in the same or a
second session — only if triggered.

### 2.7 What changes where (repo)

- `src/sutradhar/rag/` — new subpackage (chunking, corpus, embeddings, sparse, fusion, retrieve,
  index, artifacts, gpu_jobs).
- `src/sutradhar/graph/repository.py` — `search_by_plot` added (only change to P1 code paths).
- `src/sutradhar/evals/` — `retrieval.py`, `calibration.py`, `report.py`.
- `rag-engine/` — entrypoint scripts + README (per repo conventions).
- `evals/golden/gs02_no_match.yaml` unchanged; **new** `evals/negatives/heldout.yaml`
  (calibration/test negatives — deliberately *not* golden fixtures: they exist to tune a
  threshold, and tuning on the golden set would contaminate the gate).
- `.env.example` — `EMBED_BASE_URL` (future live path, unset by default), `RETRIEVAL_RUN`
  (pinned artifact run id CI/demo read).
- Makefile — `build-corpus`, `load-index`, `retrieval-eval`, `calibrate-no-match`,
  `retrieval-report`, `rag-demo`, `gpu-embed` (ephemeral session driver).
- New deps: `FlagEmbedding` + torch land **only** in a GPU-session dependency group
  (`uv sync --group gpu` on the instance), keeping the laptop/CI environment neural-free;
  laptop adds only `numpy`/`pyarrow` + `pgvector` (the Python lib: `Vector`/`SparseVector`
  SQLAlchemy types — SQL glue, not a model).

### 2.8 Python vs (optional) Java

**All Python.** The phase is embeddings (FlagEmbedding/HF — Python-only), pgvector via
SQLAlchemy (already the P1 stack), and numpy-level fusion math. A Java leg here would fork the
artifact/eval toolchain for zero showcase value; `CLAUDE.md` explicitly defers the optional Java
moat (public gateway) to **P5 grooming**, where a Spring Boot gateway in front of FastAPI is a
coherent, bounded showcase. Restated here so P2 doesn't drift into it.

---

## 3. Decisions confirmed at grooming (logged as DEC-P2-1..5 in DECISIONS.md)

> All five recommendations below were **approved as proposed (2026-07-02)**. Measured parameters
> (winning chunk config, rerank depth, abstention θ) are filled into the corresponding DECISIONS
> entries during execution. Settled and therefore **not** listed: embedder default + A/B rule
> (DEC-0002), reranker model, GPU SKU (DEC-0003), fuzzy threshold 0.80 (DEC-P1-5), compute
> placement, artifact-first CI.

### DEC-P2-1 — Vector store: pgvector vs Qdrant (deferred to P2 by CLAUDE.md/ROADMAP)

| Option | For | Against |
|---|---|---|
| **A. Postgres + pgvector** (recommended) | Already running (`pgvector/pgvector:0.8.4-pg17` in compose); **native `sparsevec` type (since 0.7.0) scores the BGE-M3 sparse leg in-DB** — both hybrid legs in one store; embeddings live **next to** the graph → chunk→version→edges joins in one SQL hop; one store to migrate/backup; zero new service for a ~10²–10³-chunk corpus; pgvector's own docs demonstrate the RRF hybrid pattern | Fusion is client-side SQL (Qdrant fuses server-side); HNSW-on-sparse capped at 1,000 non-zero elements (irrelevant at slice scale — exact scan; would need weight-pruning at catalog scale) |
| B. Qdrant | Native named dense+sparse vectors with **server-side** `rrf`/`dbsf` fusion; nicer at 10⁶+ scale | Second stateful service in compose/CI; graph joins become app-side stitching; provenance/citation queries span two systems; overkill for the slice |
| C. Both (pgvector now, Qdrant behind an interface) | Optionality | Abstraction tax with no consumer; "decide by measurement" was the roadmap's ask — measure pgvector first |

**Recommendation: A.** The hard problem is graph-adjacent retrieval; keeping vectors in the graph
DB keeps every citation join trivial, and `sparsevec` removes Qdrant's former "native sparse"
edge. Revisit only at catalog-breadth scale (recorded as the revisit trigger).

### DEC-P2-2 — Sparse leg + fusion method

| Option | For | Against |
|---|---|---|
| **A. BGE-M3 native lexical weights (stored as `sparsevec`, scored in-DB via `<#>`) + RRF fusion, k=60** (recommended) | Sparse signal from the *same* model/pass as dense (free at embed time; the reason BGE-M3 is the DEC-0002 default); multilingual/transliteration-aware tokens over the 250k XLM-R vocab; scoring is native SQL; RRF is rank-based — no score normalization across heterogeneous channels; **k=60 is the cross-industry default** (Azure AI Search, OpenSearch 2.19, Qdrant `rrf`) | RRF ignores score magnitude; k=60 untuned (deliberately — one less overfittable knob on a small eval set) |
| B. Postgres FTS (tsvector/BM25-ish) as the sparse leg | Fully in-DB, no stored weights | English-stemmer-centric — weak on romanized Tamil/Hindi and native scripts, exactly our GS-07/GS-11 risk surface |
| C. Weighted score-sum fusion (α·dense + β·sparse) | Can outperform RRF when tuned | Two tuned weights + score normalization on a tiny eval set → overfitting risk; more DECISIONS surface for marginal gain |

**Recommendation: A.** If RRF underperforms in the ablation, C is the recorded fallback (measured,
same artifacts).

### DEC-P2-3 — Chunking of plot/metadata (ROADMAP pre-commits "recursive + per-Version metadata; size/overlap ablated")

| Option | For | Against |
|---|---|---|
| **A. Recursive paragraph-boundary chunks, ablate size ∈ {256, 512, 1024} tokens, 15% overlap; metadata header on every chunk; + 1 metadata card per version** (recommended, default 512) | Median plot ≈ 2k tokens, max ≈ 13k → chunking is mandatory anyway; **recursive 512-token + 10–20% overlap is the benchmark-validated default** (Vecta/NVIDIA 2026 guides), and multi-dataset research (arXiv 2505.21700) finds **512–1024 tokens wins for broader-context/narrative retrieval** — exactly our story-description queries — while smaller chunks favour short factoid answers, so the grid brackets both regimes; ablation is nearly free (artifacts §2.6); header carries lineage into every dense unit; card covers cast/title-anchored queries | Three configs to embed/score (cheap); header slightly biases dense space toward titles (measured by the ablation, plot-only slice GS-03 isolates it) |
| B. Whole-plot single chunk (truncate/summary-split at 8192) | Simplest; strongest doc-level recall | Max plot exceeds the window → silent truncation of exactly the long flagship pages; rerank quality drops on 8k-token passages |
| C. Semantic/embedding-guided chunking | Trendy | Needs an embedder to chunk (GPU dependency in the laptop path), non-deterministic across runs — breaks the reproducibility stamp |

**Recommendation: A.** Vectara's NAACL 2025 study (25 configs × 48 models) found chunking
configuration influences retrieval quality **as much as embedding-model choice** — which is also
why the §5 iteration order fixes chunking *before* reaching for the 9B embedder. The ablation
table (per config × slice) goes in the rag-engine README; winner recorded with numbers.

### DEC-P2-4 — Rerank depth + final top-k

| Option | For | Against |
|---|---|---|
| **A. Fuse top-50 chunks → rerank 50 → aggregate → top-10 works; ablate depth ∈ {20, 50}** (recommended) | 50 ≈ the whole fused candidate space at slice scale → rerank recall ceiling; precomputed matrix makes depth a free parameter; top_k=10 matches the v0 default and the Recall@10 gate | At catalog scale depth=50 costs latency (revisit trigger recorded) |
| B. Rerank top-20 only | Cheapest live-path latency later | Risks dropping a version-set member before the reranker sees it — directly threatens the GS-01/GS-06 = 1.0 gate |
| C. No reranker (fusion order only) | One less model | Wastes the already-settled reranker; cross-encoder is the standard lever to close the last recall/MRR gap |

**Recommendation: A.**

### DEC-P2-5 — NO_MATCH abstention signal (threshold calibrated on held-out negatives — ROADMAP)

| Option | For | Against |
|---|---|---|
| **A. Absolute top-1 cross-encoder score threshold, sigmoid-normalized to [0,1]** (recommended) | The sigmoid mapping is the **official BGE score semantics** (BAAI model card); cross-encoder relevance scores are query-conditioned and **markedly more stable across query types than cosine similarity** (documented failure mode of embedding-score abstention); single interpretable θ; identical signal in eval and (later) live path | One global θ; may need per-channel handling for pure-title queries (title channel already abstains via the 0.80 fuzzy floor, DEC-P1-5 — the two thresholds compose, GS-02b "Kaithi") |
| B. Top-1 vs top-2 margin | Captures "confidently ambiguous" | Punishes legitimate multi-version queries where siblings *should* score close (GS-01: five near-tied versions) — structurally wrong for this corpus |
| C. Fused-rank score threshold (pre-rerank) | No reranker dependency | Cosine/RRF magnitudes are poorly calibrated across query types; weakest option |

**Recommendation: A** — θ tuned on the calibration half of the negative set (the "canary +
margin" methodology: θ above the top negative score with an ε safety margin, then validated) to
maximize NO_MATCH F1 subject to **zero false rejects** on positive golden fixtures; reported P/R
on the test half; value + curve recorded in DECISIONS.

---

## 4. Test strategy

### Unit (Tier-1, every PR, no GPU, no DB)

- **Chunker:** determinism (same input → same `content_hash`), paragraph-boundary respect,
  size/overlap bounds, metadata-header correctness (incl. remake-lineage header), native-script
  passthrough, empty/short-plot edge cases.
- **Fusion math:** RRF correctness on hand-computed fixtures; tie-breaking determinism;
  chunk→Work max-aggregation.
- **Sparse literal builder:** query lexical weights → pgvector `sparsevec` literal
  (`{idx:w,…}/dims`, 1-based indices) — serialization edge cases; the in-DB scoring round-trip
  lives in integration.
- **Abstention:** θ boundary behaviour; `abstain=true` ⇒ `results` still well-formed (v0 allows
  both).
- **Artifact store:** MANIFEST hash verification (corrupt artifact → hard failure, never a silent
  fallback); `ArtifactEmbeddings` raises on unseen text.
- **Schema conformance:** `SearchByPlotResult` (and a sample emitted `search_by_plot` call)
  validates against `tool_schema.v0.json` — **no hallucinated tool or parameter names** (extends
  the DEC-P1-8 conformance layers; `test_search_by_plot_not_implemented_yet` is flipped to
  `test_search_by_plot_matches_tool_schema`).
- **Config:** `RetrievalConfig` serialization → reproducibility stamp fields.

### Integration (Tier-1 with `make up`; artifacts, no GPU)

- **Index build:** `chunks`/`chunk_embeddings` populated only from gate-visible versions; a
  CANDIDATE-backed or conflict-hidden version yields zero chunks (verification gate holds through
  retrieval); FK/uniqueness constraints; pgvector cosine query returns stored neighbors;
  **sparsevec `<#>` round-trip** — in-DB score equals the hand-computed inner product on toy
  vectors, zero token overlap → score 0.
- **Pipeline round-trip:** golden query (recorded embedding) → full retrieve → `search_by_plot`
  result shape → `get_versions` join.

### Named golden regression tests (Tier-1 CI gate; computed from the committed retrieval-run artifact)

| Test | Fixture(s) | Gate |
|---|---|---|
| `test_version_set_recall_gs01` | GS-01a/b | version-set recall **= 1.0** |
| `test_version_set_recall_gs06` | GS-06a/b (franchise, `include_sequels`) | version-set recall **= 1.0** |
| `test_no_hallucinated_movie_gs02` | GS-02a/b/c + test-half negatives | abstain on **all**; false-accept = 0 |
| `test_dub_vs_remake_gs04` | GS-04 (retrieve → `get_versions`) | every Baahubali version labelled `is_official_dub_of` (never `is_remake_of`) end-to-end through retrieval |
| `test_sibling_vs_remake_gs05` | GS-05 | Devdas adaptations returned as `based_on` siblings, never chained `is_remake_of` |
| `test_false_merge_gs10` | GS-10 ("Vikram Kamal Haasan") | two distinct Works, `ambiguous=true`; false-merge = 0 |
| `test_tool_calls_validate_v0` | GS-07 `expected_tool_calls` + emitted `search_by_plot` results | 100% validate against `tool_schema.v0.json` |
| `test_recall_gate` | all retrieval fixtures (GS-01/03/06/07/11) | **Recall@10 ≥ 0.90**; regression vs committed run blocks merge |

### Eval set + metric thresholds (the phase gate)

| Metric | Slice | Threshold |
|---|---|---|
| Recall@10 | all retrieval fixtures (GS-01, 03, 06, 07, 11) | **≥ 0.90** (hard exit gate → green light for P4) |
| Version-set recall | GS-01, GS-06 | **= 1.0** |
| No-hallucinated-movie | GS-02 + held-out test negatives | **= 0** false accepts |
| Title-match recall | GS-11 (all 4 perturbations) | = 1.0 (already passing in P1 at the repo layer; must hold through the pipeline) |
| NO_MATCH precision/recall | held-out test half | **reported** (threshold + curve in DECISIONS; no numeric gate beyond the 0-false-accept rule) |
| Recall@1/@5, MRR@10 | all slices | reported (Table 1), no gate |

GPU-touching steps (embedding, rerank matrix) run in the batched session; **Tier-2 CI is not
needed for P2's gate** — every gating metric is computable from committed artifacts in Tier-1
(ROADMAP §6.2).

---

## 5. Task breakdown (ordered, independently committable)

1. **Held-out negative set** — author `evals/negatives/heldout.yaml` (~24 queries,
   GS-02-schema-compatible, calibration/test split marked) + absent-from-slice validator test.
2. **Migration: pgvector + chunk schema** — enable extension; `chunks` + `chunk_embeddings`
   tables (§2.3); constraint tests.
3. **Chunker + corpus builder** — `sutradhar.rag.{chunking,corpus}` + `rag-engine/build_corpus.py`
   (`make build-corpus`); metadata headers + metadata cards; unit tests; populates `chunks` for all
   three ablation configs.
4. **Artifact store** — `sutradhar.rag.artifacts` (run ids, MANIFEST.sha256 write/verify, load
   API); `RETRIEVAL_RUN` env pin; unit tests.
5. **GPU job script** — `rag-engine/embed_and_score.py` (+ `gpu` dependency group, `make gpu-embed`
   ephemeral driver): corpus + query embeddings (dense+sparse, all configs), full rerank matrix →
   artifacts. Dry-runnable against a stub provider for tests.
6. **Run the GPU session** — one ephemeral A100 pass; commit `evals/retrieval_runs/<run_id>.json`
   + MANIFESTs; record cost/time.
7. **Index loader** — artifacts → `chunk_embeddings` (`make load-index`); integration tests
   (gate-visibility, pgvector round-trip).
8. **Sparse scoring + fusion + aggregation** — `sutradhar.rag.{sparse,fusion}` (sparsevec
   literal builder, in-DB `<#>` scoring, RRF k=60, chunk→Work max-aggregation); unit tests on
   hand-computed fixtures.
9. **Retriever pipeline + `search_by_plot`** — `sutradhar.rag.retrieve` +
   `repository.search_by_plot` (v0-conformant); flip the P1 absence test; conformance round-trip
   tests.
10. **Eval harness + ablation runner** — `sutradhar.evals.retrieval`, `evals/run_retrieval_eval.py`
    (`make retrieval-eval`); metrics per slice; ablation grid (chunk config × fusion × rerank
    depth) from stored scores; pick + record the winning config.
11. **Abstention calibration** — `evals/calibrate_no_match.py` (`make calibrate-no-match`); tune θ
    on calibration half; P/R on test half; wire θ into the retriever; GS-02 regression test green.
12. **Named golden regression tests + Tier-1 CI gate** — §4 table wired into `tier1.yml` against
    the committed run artifact; regression blocks merge.
13. **Gate check / A/B branch** — if Recall@10 < 0.90 or version-set recall < 1.0: iterate
    (chunking → fusion → **bge-multilingual-gemma2 leg per DEC-0002**, second GPU session) until
    green or the failure is documented. Skipped entirely if green (see Q1).
14. **Benchmarks + decisions + docs** — `BENCHMARKS.md` Table 1 + Papanasam row + reproducibility
    stamp; DEC-0002 → **Accepted** with numbers; measured values filled into DEC-P2-1..6;
    `TOOL_SCHEMA.md` status note (implementation landed, v0 unchanged); rag-engine README
    (+ ablation table); PORTFOLIO bullet.
15. **`make rag-demo`** — GS-07a Tanglish + GS-03a plot query replayed from artifacts → cited,
    relationship-labelled version set printed; README demo section.

---

## 6. Definition of Done (instantiates the CLAUDE.md generic DoD) — **ALL MET, 2026-07-02**

- [x] Code complete per this approved spec; `make check` green (lint + mypy-strict + **299 unit
      tests**).
- [x] Unit + integration tests written and passing (**108 integration**), including **all eight
      named regression tests** in §4 (version-set recall GS-01/GS-06, no-hallucinated-movie
      GS-02, dub-vs-remake GS-04, sibling-vs-remake GS-05, false-merge GS-10, tool-call v0
      validation, recall gate) + GS-11-through-pipeline bonus.
- [x] **Eval thresholds met:** Recall@10 = **1.000** ≥ 0.90 (retrieval fixtures); version-set
      recall = **1.0** on GS-01 **and** GS-06; **0 false accepts** on GS-02 + test negatives;
      NO_MATCH P/R reported (recall 1.0 / precision 0.75; θ = 0.151747 + curve in DECISIONS).
      (MLflow recording lands in P3 per the roadmap; P2 records to `BENCHMARKS.md` + the committed
      run artifact — same pattern as P1's graph report.)
- [x] **Benchmark tables:** `docs/BENCHMARKS.md` **Table 1** populated (6 config rows + the
      Papanasam/Drishyam version-set row = 1.0 + the GS-06 franchise row) with the §6.1
      reproducibility stamp. **Table 2 untouched** — retrieval metrics never appear there.
- [x] Tier-1 CI gates on the retrieval metrics from the committed artifact
      (`tests/test_golden_retrieval_regressions.py`); a regression blocks merge
      (drift check negative-control verified).
- [x] `DECISIONS.md`: DEC-P2-1..6 updated with measured values (winner **1024tok_15pct**, depth
      **20**, θ **0.151747** + curve); **DEC-0002 flipped to Accepted** (gate met by default,
      challenger not run); **DEC-P2-7** added (HF-relay transport, user-approved).
      `rag-engine/README.md` written (architecture, ablation table, how to run).
      `TOOL_SCHEMA.md` status note added (implementation landed, v0 unchanged).
- [x] Runs cleanly from scratch: fresh clone + `.env` → `make up db-migrate ingest-seed
      build-corpus load-index retrieval-eval` reproduces Table 1 from the pinned artifact run
      (no GPU needed); the GPU session itself reproducible via `make gpu-embed`
      (create→run→destroy; `make gpu-nuke` confirms no strays).
- [x] 30-second demo: `make rag-demo` (GPU off) prints the cited, relationship-labelled Drishyam
      version set (original flagged ★, revision-pinned Wikipedia citations) from the Tanglish
      GS-07a + plot-only GS-03a queries.
- [x] Resume-ready quantified bullets drafted in `docs/PORTFOLIO.md` (hybrid multilingual RAG,
      Recall@10 = 1.000, version-set recall 1.0, **~$0.22** GPU cost, zero-GPU CI gate).

---

## 7. Open questions — RESOLVED (2026-07-02)

**Q1 — Embedding A/B execution. RESOLVED: option (a).** If BGE-M3 passes the exit gate on the
first pass, the 9B `bge-multilingual-gemma2` leg is **skipped entirely** — strictly per
DEC-0002's settled rule; DEC-0002 flips to Accepted with "gate met by default, challenger not
needed". The 9B leg runs only as step 13's escalation path. (Recorded as a DEC-0002 execution
note.)

**Q2 — Committed run artifact in the repo. RESOLVED: yes.** The compact retrieval-run summary
(~1–5 MB JSON under `evals/retrieval_runs/`) is **committed to git** so Tier-1 CI recomputes
every gating metric with no GPU, no DB, no external fetch. Raw embeddings/matrices stay
git-ignored under `data/artifacts/` (MANIFEST-hashed, optionally pushed to HF Hub). (Logged as
DEC-P2-6.)

**Stated assumptions (accepted with the approval):** held-out negatives ≈ 24 authored queries
(50/50 split) is enough for a slice-scale θ; GS-07 retrieval in P2 embeds the **raw** code-mixed
query (no LLM intent parsing — that's the P4 headroom story); GS-09 scoping stays a graph/repo
concern (already covered by P1 tests) and contributes no retrieval fixture; metadata cards are
`kind='metadata_card'` chunks, not a schema fork.

---

## 8. Sources (web-research pass, accessed 2026-07-02)

- **pgvector `sparsevec` + hybrid:** pgvector README (`github.com/pgvector/pgvector`) — `sparsevec`
  since 0.7.0 (storage `8·nnz + 16` bytes, ≤ 16,000 non-zero elements; `<#>`/`<=>`/`<->` operators;
  HNSW-on-sparse ≤ 1,000 nnz; dense `vector` HNSW ≤ 2,000 dims), hybrid-search section with RRF +
  cross-encoder examples; pgvector 0.8.0 release notes (postgresql.org news); pgvector-python
  `SparseVector` (SQLAlchemy support).
- **BGE-M3:** `huggingface.co/BAAI/bge-m3` + `bge-model.com/bge/bge_m3.html` — dense + sparse
  lexical weights + ColBERT from one pass, 100+ languages, 8192-token inputs; `BGEM3FlagModel`
  (FlagEmbedding) returns lexical weights alongside dense vectors at no extra cost; hybrid
  (dense+sparse) is the model authors' recommended usage.
- **Reranker score semantics:** `huggingface.co/BAAI/bge-reranker-v2-m3` — raw similarity logits,
  "score can be mapped to [0,1] by sigmoid"; FlagEmbedding reranker examples.
- **Abstention practice:** cross-encoder scores calibrated/stable across query types vs cosine
  (tianpan.co "The Retrieval Emptiness Problem", 2026); canary/held-out negative threshold
  methodology (`github.com/alexsavio/rag-eval` abstention experiment); UAEval4RAG
  (arXiv 2412.12300) — unanswerable-query evaluation framing.
- **RRF k=60 as industry default:** Azure AI Search hybrid ranking docs; OpenSearch 2.19 RRF
  release; Qdrant hybrid queries (`rrf`/`dbsf` fusion).
- **Chunk size:** arXiv 2505.21700 "Rethinking Chunk Size for Long-Document Retrieval" —
  64–128 tok for short factoid answers, 512–1024 tok for broader-context retrieval; 2026
  benchmark guides (Vecta/NVIDIA-validated recursive 512 tok + 10–20% overlap default);
  Vectara NAACL 2025 (25 configs × 48 models: chunking ≈ embedder choice in impact).
- **MIRACL language coverage:** `bge-model.com` MIRACL evaluation page + MIRACL dataset docs —
  18 languages incl. hi/bn/te; **ta/ml/kn absent** → public multilingual leaderboards under-cover
  the mission's core languages (why the golden-set gate, not MTEB/MIRACL, resolves DEC-0002).
