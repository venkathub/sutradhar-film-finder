# rag-engine

Hybrid retrieval, reranking, grounding, and guardrails — the facts live here, never in weights.

**Import package:** `sutradhar.rag`

## Planned architecture
- Query normalization + transliteration (deterministic, rule-based).
- Hybrid retrieval: BGE-M3 dense + sparse over the catalog / remake graph / plot text.
- Cross-encoder reranking with `bge-reranker-v2-m3`.
- Grounding + source attribution (every claim cites a source); prompt-injection guardrails.
- Cross-lingual entity resolution across remakes and dubs (the Papanasam/Drishyam case).
- Retrieval eval gate: Recall@10 ≥ 0.90 on the golden set before any fine-tuning is invested.

## Status
**P2 complete — exit gate met** (Recall@10 = 1.000 ≥ 0.90; version-set recall GS-01/GS-06
= 1.0; abstention calibrated, 0 false accepts). Landed, in build order:

### Corpus builder (P2 task 3)

```
make build-corpus        # or: uv run python rag-engine/build_corpus.py [--config 512tok_15pct]
```

Builds the embeddable corpus from the **gate-visible graph only** (`ground_truth_versions`
⋈ `plot_texts` — CANDIDATE/conflict-hidden records excluded by construction) into the
`chunks` table, for every chunking-ablation config (DEC-P2-3: recursive paragraph-boundary,
{256, 512, 1024} est. tokens, 15% overlap). Two document kinds per Version:

- **Plot chunks** (`sutradhar.rag.chunking`) — every chunk is prefixed with a metadata
  header (`"Papanasam (Tamil, 2015) — remake of Drishyam (Malayalam, 2013). "`) so the
  remake/dub lineage rides each embedded unit into dense space; native-script plots are
  chunked as-is.
- **Metadata cards** (`sutradhar.rag.corpus`) — one synthetic doc per Version (title +
  AKAs incl. native scripts + lead cast + director + relationship): the dense target for
  cast-anchored and title-adjacent queries.

Chunk sizes use a deterministic character-class token **estimator** (Latin ≈ 4 chars/tok,
other scripts ≈ 2 — conservative for Indic), not the XLM-R tokenizer: the laptop stays
neural-free (ROADMAP §2) and sizes are ablation brackets, not exact budgets. Determinism
(same text → same `content_hash`) is the property CI and the artifact store depend on.

Live-graph result (2026-07-02): 31 gate-visible versions, 52 plot docs → 427 / 225 / 122
plot chunks (256/512/1024) + 31 metadata cards per config.

### GPU session evidence (P2 task 6 — executed 2026-07-02)

Pinned run **`RETRIEVAL_RUN=20260702T135315Z-f6583183`** (BGE-M3 + bge-reranker-v2-m3 on
an ephemeral JarvisLabs A100, DEC-P2-7 relay): 833 unique texts embedded (3 configs +
51 queries), **44,217 reranker pairs** (full matrix per config), sealed + verified,
instance destroyed. ~10 GPU-minutes ≈ **$0.22** across 4 attempts (3 environment fixes,
each diagnosed from the relayed `job.log` — see `evals/retrieval_runs/*.meta.json`).
Instance deps are pinned in the startup script (`transformers<5` for FlagEmbedding).

### Artifact store (P2 task 4)

`sutradhar.rag.artifacts` — versioned GPU-session outputs under
`data/artifacts/retrieval/<run_id>/` (git-ignored; P1 snapshot discipline reused):
every file is sha256-recorded in `MANIFEST.sha256`; `ArtifactRun.open` **hard-fails** on
a missing/mismatched/stray file — a corrupt run is never silently served.
`ArtifactEmbeddings` implements the `EmbeddingProvider` protocol by *lookup* keyed by
`sha256(text)` (row-aligned banks: `<bank>_hashes.json` + `<bank>_dense.npy` +
`<bank>_sparse.json`); an unseen text raises `MissingArtifactError` — the laptop/CI path
never degrades to a fake vector. The run CI/demo read is pinned by `RETRIEVAL_RUN` (env;
see `.env.example`); the live embedding path (`EMBED_BASE_URL`) stays unset until P5.

### GPU embed+score job (P2 task 5, DEC-P2-7)

```
make gpu-embed     # ephemeral: export → HF relay → A100 embeds+scores → pull → verify → destroy
# pieces, runnable standalone:
uv run python rag-engine/export_gpu_inputs.py                     # DB → gpu_inputs.json
python rag-engine/embed_and_score.py --inputs … --out … --run-id … [--stub]
```

One batched session produces every neural output as a sealed artifact run: BGE-M3
dense+sparse embeddings for **all** chunk configs + all golden/negative queries, and the
**full** query×chunk `bge-reranker-v2-m3` matrix per config (sigmoid-normalized, parquet)
— making fusion/depth/top-k free laptop-side parameters. `embed_and_score.py` is
**self-contained** (no `sutradhar` import — the HF relay ships this one file to the box);
its output format is locked to `sutradhar.rag.artifacts` by the `--stub` dry-run test.
`FlagEmbedding`/torch live only in the `gpu` dependency group (`uv sync --group gpu` on
the instance); the laptop/CI env stays neural-free. Transport per DEC-P2-7: private HF
dataset repo (`HF_ARTIFACT_REPO`, fine-grained token), instance destroyed in `finally`.

### Index loader (P2 task 7)

```
make load-index          # loads the RETRIEVAL_RUN artifact into chunk_embeddings
```

Joins the sealed run's corpus banks onto the `chunks` table **by `content_hash`** and
materializes pgvector rows (dense `vector(1024)` + `sparsevec`) for in-DB scoring.
Strict: MANIFEST-verified before any write; a DB chunk with no recorded vector is a
**hard error** (corpus drifted after the export — rebuild or re-embed); idempotent per
`(embed_model, index_version)`. Live result: 458/256/153 embeddings loaded; cosine
neighbor probe on the real vectors behaves (a Papanasam chunk's nearest neighbors are
its own family).

### Sparse channel + fusion (P2 task 8, DEC-P2-2)

`sutradhar.rag.sparse` — query lexical weights (0-based BGE-M3 token ids) → `sparsevec`
literal (1-based, via `pgvector.SparseVector`) → **in-DB** `<#>` top-N; zero-overlap
chunks are excluded (no RRF mass for no signal). `sutradhar.rag.fusion` — RRF **k=60**
(untuned industry default), deterministic key tie-breaking (the committed-artifact CI
gate depends on it), and chunk→Work **max** aggregation. Scores verified against
hand-computed inner products in integration.

### Retriever pipeline + `search_by_plot` (P2 task 9)

`sutradhar.rag.retrieve` wires §2.4: title channel (rapidfuzz over `match_key`, θ=0.80 →
the matched versions' **metadata-card chunks**, so it joins chunk-level RRF like any
other channel) + dense (pgvector cosine) + sparse (in-DB `<#>`) → RRF k=60 → cross-
encoder rerank from the **recorded full matrix** → chunk→Work max aggregation →
calibrated abstention. Every knob is data (`RetrievalConfig`, stampable). No neural op
on the laptop: `ArtifactEmbeddings` + `ArtifactReranker` serve recorded vectors/scores
and **raise** on anything unseen. `repository.search_by_plot` implements frozen
TOOL_SCHEMA **v0 exactly** (`description`, `top_k=10` → `{results[], abstain}`); the
retriever is injected keyword-only infrastructure, invisible to the tool surface.

Live smoke (real run, 512tok config): GS-11a "Papanaasam" → Drishyam @0.998; GS-01a →
Drishyam @0.863; Tanglish GS-07a → Drishyam top-1 → full 5-version labelled set.

### Eval harness + ablation grid (P2 task 10)

```
make retrieval-eval      # full grid; writes the committed evals/retrieval_runs/<run_id>.json
```

Recall@1/5/10, MRR@10, version-set recall per slice, over the 3×2 ablation grid
(chunk config × rerank depth) — computed entirely from stored artifacts, zero GPU.
The committed run artifact (DEC-P2-6, 0.54 MB) records per-query ranked works + channel
sizes + the raw abstention signal for all 13 retrieval fixtures **and** all 24 held-out
negatives, per cell; CI recomputes every gating metric from it with the same functions.

**Ablation results (run `20260702T135315Z-f6583183`, 13 retrieval fixtures — GS-01/03/06/07/11):**

| config | R@1 | R@5 | R@10 | MRR@10 | VSR GS-01 | VSR GS-06 |
|---|---|---|---|---|---|---|
| 256tok/d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 |
| 256tok/d50 | 0.769 | 1.000 | 1.000 | 0.846 | 1.0 | 1.0 |
| 512tok/d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 |
| 512tok/d50 | 0.923 | 1.000 | 1.000 | 0.949 | 1.0 | 1.0 |
| **1024tok/d20** ★ | **0.923** | **1.000** | **1.000** | **0.962** | **1.0** | **1.0** |
| 1024tok/d50 | 0.846 | 1.000 | 1.000 | 0.910 | 1.0 | 1.0 |

**Exit gate met on the first pass: Recall@10 = 1.000 (≥ 0.90) in every cell; version-set
recall GS-01 = GS-06 = 1.0.** Per the DEC-0002 execution note, the 9B challenger leg is
skipped. Known non-gating miss: GS-07b (Hinglish) ranks Drishyam #2 (R@5=1.0, VSR via
top-1 misses) — the raw-code-mixed-query limitation P2 accepts by design; LLM intent
parsing is the P4 headroom story.

### Abstention calibration (P2 task 11, DEC-P2-5)

```
make calibrate-no-match    # pure laptop math over the committed artifact
```

**θ = 0.151747** (1.35 × top calibration canary), calibrated on the 12-negative
calibration half only. **Zero false accepts on GS-02 + all 12 untouched test negatives
(NO_MATCH recall 1.0, precision 0.75).** Measured and documented: the zero-false-reject
constraint is infeasible on raw cross-encoder scores (code-mixed positives score below
fluent-English out-of-catalog negatives) — four positives (GS-03a/c, GS-07a/b) are
flagged `abstain=true` **with their correct results** (all still Recall@5 = 1.0): they
degrade to "low confidence", never to a hallucinated match. This interleave is a named
P4 headroom target. Full curve: `evals/retrieval_runs/<run>.calibration.json`.

### Named golden regressions = the Tier-1 CI gate (P2 task 12, DEC-P2-6)

`tests/test_golden_retrieval_regressions.py` — the P2_SPEC §4 table, recomputed from the
**committed** run artifact on every PR (no DB, no GPU, plain `uv run pytest`):
version-set recall GS-01/GS-06 = 1.0 · no-hallucinated-movie (GS-02 + test negatives,
0 false accepts) · dub-vs-remake GS-04 (incl. the bilingual double-original) ·
sibling-vs-remake GS-05 · false-merge GS-10 (two distinct Vikram works) · tool-call v0
validation (GS-07 calls + every recorded result) · **the Recall@10 ≥ 0.90 gate** with a
recomputation-vs-committed-metrics drift check (negative-control verified: a doctored
ranking fails). GS-11 fuzzy titles held through the full pipeline.

## Reproduce from scratch (fresh clone + .env; GPU optional)

```
make up db-migrate ingest-seed      # P1: graph + plots (offline snapshot replays)
make build-corpus                   # chunks for all ablation configs (deterministic)
make load-index                     # pinned RETRIEVAL_RUN artifacts -> pgvector
make retrieval-eval                 # Table 1 metrics from recorded scores (GPU OFF)
make calibrate-no-match             # DEC-P2-5 θ from the committed artifact (GPU OFF)
```

Only `make gpu-embed` (re-embedding after a corpus change) needs the GPU — one ephemeral
A100 session (~10 min ≈ $0.22), sealed artifacts pulled + verified, instance destroyed.
Everything above reproduces `docs/BENCHMARKS.md` Table 1 from the pinned run with no GPU
and no network beyond Postgres.

## Tests

```
uv run pytest tests/test_chunking.py tests/test_artifacts.py               # hermetic
uv run pytest -m integration tests/integration/test_build_corpus.py       # needs `make up`
```
