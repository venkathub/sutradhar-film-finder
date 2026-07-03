# Sutradhar Benchmarks

> **Two tables, kept honest.** Retrieval quality is **model-independent** — it depends only on the
> index/embedder/reranker, not on the generator. Fine-tuning does **not** touch retrieval, so
> retrieval numbers are **never** presented as "before/after fine-tuning". The two tables below are
> reported separately and filled by different phases.
>
> **P0 status:** this file is a **skeleton** — P0 fills **neither** table (no retrieval, no
> generation). Table 1 is populated in **P2**; Table 2 in **P3/P4**. Each result must be recorded to
> **MLflow** with a reproducibility stamp and linked here.

---

## Table 1 — Retrieval quality (model-independent)

Populated in **P2** (hybrid retrieval + reranker), gated by **Recall@10 ≥ 0.90** on the golden set
before any fine-tuning is invested. Metrics are computed over `docs/GOLDEN_SET_SCENARIOS.md`.

**Captured 2026-07-02** — 13 retrieval fixtures (GS-01/03/06/07/11: flagship, plot-only,
franchise, code-mixed, fuzzy-title) through the full §2.4 pipeline (title + BGE-M3 dense +
BGE-M3 sparse → RRF k=60 → bge-reranker-v2-m3 → Work aggregation → calibrated abstention).
**Exit gate met in every ablation cell on the first pass → P4 is green-lit; the DEC-0002 9B
challenger leg was not needed.** Reproduce with the GPU **off**: `make retrieval-eval`
(recomputes from the pinned artifact run), or from scratch per `rag-engine/README.md`.

| Config (chunk × rerank depth) | Recall@1 | Recall@5 | Recall@10 | MRR@10 | VSR GS-01 | VSR GS-06 | Notes |
|--------|---------:|---------:|----------:|-------:|----------:|----------:|-------|
| **1024tok_15pct / d20** ★ winner | **0.923** | **1.000** | **1.000** | **0.962** | **1.0** | **1.0** | DEC-P2-3/4 measured |
| 1024tok_15pct / d50 | 0.846 | 1.000 | 1.000 | 0.910 | 1.0 | 1.0 | |
| 512tok_15pct / d50 | 0.923 | 1.000 | 1.000 | 0.949 | 1.0 | 1.0 | |
| 512tok_15pct / d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 | |
| 256tok_15pct / d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 | |
| 256tok_15pct / d50 | 0.769 | 1.000 | 1.000 | 0.846 | 1.0 | 1.0 | |

Winner-cell per-slice detail: flagship / plot-only / franchise / fuzzy-title all 1.000 across
R@1/5/10; code-mixed R@1 0.5, R@5 1.0 (GS-07b Hinglish ranks Drishyam #2 on the raw query —
the accepted P2 limitation and the named P4 headroom target, per DEC-P2-5's measured
positive/negative interleave). **NO_MATCH abstention (DEC-P2-5): θ = 0.151747, 0 false accepts
on GS-02 + all 12 held-out test negatives (NO_MATCH recall 1.0, precision 0.75); 4 weak-scoring
positives flagged low-confidence with correct results (all R@5 = 1.0).**

**Version-set recall** = fraction of a Work's full set of language versions (original + all
remakes/dubs) returned for a query. The gating case:

| Query | Expected version set | Version-set recall | Notes |
|-------|----------------------|-------------------:|-------|
| "Papanaasam" (fuzzy, GS-11a) → and GS-01a plot query | Drishyam (2013 ML, **original**) + Drishya (2014 KN) + Drushyam (2014 TE) + Papanasam (2015 TA) + Drishyam (2015 HI) — all relationship-labelled | **1.0** | end-to-end: query → top-1 Work → `get_versions`; labels verified (remake ≠ dub) |
| "show me every Drishyam film" (franchise, GS-06a) | the 5 above + Drishyam 2 (2021 ML, sequel-original) + Drishya 2 (KN) + Drushyam 2 (TE) + Drishyam 2 (2022 HI) | **1.0** | `include_sequels` walk; sequel never conflated with remake |

**Reproducibility stamp (all rows):** embedder `BAAI/bge-m3` · reranker `BAAI/bge-reranker-v2-m3`
(sigmoid scores) · chunker `recursive_para` (deterministic, char-class token estimate) · artifact
run `20260702T135315Z-f6583183` (sealed, MANIFEST-verified; GPU job code SHA `f29ff6d`) · index
`chunk_embeddings@(BAAI/bge-m3, 20260702T135315Z-f6583183)` · golden set frozen 2026-07-02
(25 fixtures) + 24 held-out negatives · committed run artifact
`evals/retrieval_runs/20260702T135315Z-f6583183.json` (+ `.calibration.json`) — Tier-1 CI
recomputes every gating metric from it on each PR. **MLflow (self-hosted, DEC-P3-2):** run
`26dc04707c7d4efda4c07dff64a7b8ba` in experiment `sutradhar/retrieval` (Table 1 backfill,
logged 2026-07-03 by `make mlflow-backfill` — a log of the committed artifact, not a re-run;
discharges the P2 "(MLflow wiring lands in P3)" note).

---

## Table 2 — Generation / agent quality (base vs QLoRA)

Populated at the **top of the P4 GPU window** (base) and **P4** (QLoRA), captured by the P3
harness (`make benchmark-generation`) so both columns share identical serving conditions, with
evidence (MLflow run links, Langfuse traces, screenshots). If QLoRA does not measurably beat a
well-prompted base model here, it is cut and the reason documented (DEC-0001).

**Column definitions are FROZEN by P3** (P3_SPEC §2.4, byte-identical scorer code for both rows):

| Column | Definition (source of truth) |
|---|---|
| Tool-call accuracy | DEC-P3-5 **sequence-level headline** (expected sequence in order, benign schema-valid extras tolerated); call-level AST + schema-validity in the artifact |
| Code-mixed intent acc | per-turn exact match of the `INTENT:` preamble label, **GS-07 slice** |
| Slot-extraction acc | micro-F1 over expected (key,value) pairs, `match_key`-normalized titles |
| Backtracking coherence | frozen judge rubric over GS-08 conversations (mean [0,1]) |
| Faithfulness | **1 − hallucinated-movie rate** (deterministic detector; **gate: 0 inventions on GS-02**); RAGAS faithfulness reported as supplementary |
| Answer relevancy | RAGAS answer_relevancy via the frozen judge + BGE-M3 (DEC-P3-3) |
| Latency / throughput | p50/p95 wall-clock per assistant turn; completion tokens/sec — live runs only (null in dry-run, validator-enforced) |

| Model | Tool-call accuracy | Code-mixed intent acc | Slot-extraction acc | Backtracking coherence | Faithfulness (1 − hallucinated-movie rate) | Answer relevancy | GPU latency p50/p95 | Throughput (tok/s) |
|-------|-------------------:|----------------------:|--------------------:|-----------------------:|-------------------------------------------:|-----------------:|--------------------:|-------------------:|
| Base (Gemma-4-E4B, prompted) | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _captured at the top of the P4 GPU window by this harness_ |
| QLoRA (Gemma-4-E4B + adapter) | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _populated in P4_ |

**Frozen stamp fields (P3):** prompt bundle `prompt_hash 78215ccc…` (system + exemplars +
intent taxonomy, `evals/prompts/prompts.lock.json`) · TOOL_SCHEMA **v0** (sha256 recorded per
run) · retrieval replay pinned to run `20260702T135315Z-f6583183` (DEC-P3-8) · **judge frozen
(DEC-P3-1, κ = 0.738 measured):** `openai/gpt-oss-20b @ 6cee5e81ee83…`, coherence rubric
`judge_coherence_v1.md`, temp 0, low reasoning effort, guided JSON · ragas `0.4.3` · plus per
live run: base model revision, adapter commit, vLLM serving config, GPU type, decode params,
date, MLflow run URL, Langfuse trace links.

**Machinery evidence (P3 dry-run — mock model, NEVER published as Table 2 numbers):** committed
run `evals/generation_runs/20260703T012339Z-e7fff041.json` — 12/12 generation fixtures
end-to-end against the live graph; sequence accuracy 1.0; schema-validity 35/36 = **exactly the
seeded hallucinated tool caught**; faithfulness 17/18 = **exactly the seeded invented movie
caught**; **GS-02 inventions = 0 (the hard gate)**; latency/throughput null by the dry-run
honesty invariant. MLflow run `c2fb0eab52bd4691a8a70b35491d0dce` (`sutradhar/generation`);
Langfuse trace exported + committed (`…e7fff041.trace.json`, self-hosted instance per
DEC-P3-7). Tier-1 CI recomputes every deterministic metric from this artifact on each PR.

> **Retrieval metrics never appear in Table 2, and generation metrics never appear in Table 1.**

---

## Graph coverage & extraction lift (P1 — graph metrics, NOT retrieval)

> P1's evidence row (P1_SPEC §1.9/§1.10). **These are graph-completeness metrics against the
> curated seed truth — a different denominator from Table 1's retrieval recall; the two are never
> blended.** Reproduce: `make up && make db-migrate && make ingest-seed` (offline replays of the
> recorded snapshots; the extraction step replays its artifact), then `make graph-report`.

### Version coverage (gate-visible versions / curated truth) — captured 2026-07-02

| Franchise | Coverage | Notes |
|---|---:|---|
| drishyam (incl. sequel + foreign) | **11/11 = 1.00** | flagship |
| baahubali | **4/4 = 1.00** | flagship; bilingual double-original + 2 verified dub tracks |
| devdas | **4/4 = 1.00** | flagship; novella + 3 sibling adaptations |
| vikram | **2/2 = 1.00** | flagship; false-merge pair kept distinct |
| manichitrathazhu | **5/5 = 1.00** | flagship; transitive chain |
| distractors | 5/5 = 1.00 | noise floor |
| **Flagship gate (= 1.0)** | **PASS** | P1 exit criterion |

Supplementary — curated-relationship **edge** coverage: **19/20 = 0.95**. The one gap is
Rajmohol's *proximate* edge (→ Chandramukhi): no source states it (Wikipedia asserts the direct
Manichitrathazhu lineage, which IS a verified edge) — recorded, not invented.

### Extraction lift (report, not gate) — one ephemeral A100 session, 2026-07-02

| Metric | Value |
|---|---|
| Extractor | `google/gemma-4-E4B` on vLLM (`guided_json`, temp 0), run `3e37549f492bd2fc` |
| Pages processed | 27 (parse-failure rate **7.4%**; free-form prompting was 92.6% — DEC-P1-4 amendment) |
| Candidates | 58 proposed → **19 confirmed / 35 rejected / 4 skipped** (human gate, reviewer: venkatesh) |
| **Candidate precision** (confirmed/decided) | **0.352** — the honest 4B number; every reject on evidence |
| Proposals killed by the verbatim-evidence guard | 14 (hallucinated citations never reached review) |
| **Verified edges added beyond Wikidata** | **6** — Drishya, Drushyam, Dharmayuddhaya, Drishya 2, Drushyam 2 remake edges + the Chandramukhi→Apthamitra *proximate* edge (GS-09B) |
| Existing edges corroborated (wikidata+wikipedia+human) | 10 |
| GPU cost | ~50 min ephemeral A100 (created → served → extracted → destroyed; ≈ $1) |

**Reproducibility stamp:** code `4a8eff3` · seed slice `9a1b87eeff15` · snapshots
wikidata `20260702T055436Z:4ce71591d6d5` / tmdb `20260702T061520Z:3f6a68c5b1e0` /
imdb `20260702T063039Z:219500867f63` / wikipedia `20260702T064101Z:cd62bacb51c8` /
extraction `20260702T085302Z:debcdc270318` · decisions artifact
`data-pipeline/review_decisions_20260702.yaml`. (MLflow wiring lands in P3; P1 metrics are
computed by `make graph-report` + pytest per the spec's no-MLflow-in-P1 rule.)
