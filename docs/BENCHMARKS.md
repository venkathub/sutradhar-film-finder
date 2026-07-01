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
