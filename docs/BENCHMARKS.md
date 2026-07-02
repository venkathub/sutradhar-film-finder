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

| Config | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Version-set recall | Notes |
|--------|---------:|---------:|----------:|-------:|-------------------:|-------|
| _BGE-M3 dense + sparse + reranker_ | _—_ | _—_ | _—_ | _—_ | _—_ | _populated in P2_ |

**Version-set recall** = fraction of a Work's full set of language versions (original + all
remakes/dubs) returned for a query. The gating case:

| Query | Expected version set | Version-set recall | Notes |
|-------|----------------------|-------------------:|-------|
| "Papanasam" (Tamil) | Drishyam (2013 ML, original) + Drishya (2014 KN) + Drushyam (2014 TE) + Papanasam (2015 TA) + Drishyam (2015 HI) | _—_ | _populated in P2_ |

_Reproducibility stamp (per row): embedder id + revision, index type, chunker, reranker, golden-set
version, date, MLflow run URL._

---

## Table 2 — Generation / agent quality (base vs QLoRA)

Populated in **P3** (base) and **P4** (QLoRA), captured during the on-demand GPU run with evidence
(MLflow run links, Langfuse traces, screenshots). If QLoRA does not measurably beat a well-prompted
base model here, it is cut and the reason documented (DEC-0001).

| Model | Tool-call accuracy | Code-mixed intent acc | Slot-extraction acc | Backtracking coherence | Faithfulness (1 − hallucinated-movie rate) | Answer relevancy | GPU latency p50/p95 | Throughput (tok/s) |
|-------|-------------------:|----------------------:|--------------------:|-----------------------:|-------------------------------------------:|-----------------:|--------------------:|-------------------:|
| Base (Gemma-4-E4B, prompted) | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _populated in P3_ |
| QLoRA (Gemma-4-E4B + adapter) | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _populated in P4_ |

_Reproducibility stamp: base model + revision, adapter commit, TOOL_SCHEMA version, serving config
(vLLM), GPU type, decode params, eval-set version, date, MLflow run URL, Langfuse trace links._

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
