# evals

Evaluation harness, golden test set, and the two-table benchmark runner.

**Import package:** `sutradhar.evals`

## Planned architecture
- RAGAS-based eval harness wired to MLflow (metrics) and Langfuse (traces), CI-gated.
- Golden set derived from `docs/GOLDEN_SET_SCENARIOS.md` (incl. the Papanasam/Drishyam version-set
  case).
- **Two-table benchmark, kept honest** (`docs/BENCHMARKS.md`):
  1. Retrieval quality (model-independent): Recall@k, MRR, version-set recall.
  2. Generation/agent quality, base vs QLoRA: tool-call accuracy, code-mixed intent/slot accuracy,
     backtracking coherence, faithfulness, answer relevancy, plus GPU latency/throughput.
  Retrieval metrics are **never** presented as before/after fine-tuning.

## Status
**Not built until P3.** P0 creates this directory as a stub only (the two-table `BENCHMARKS.md`
skeleton is seeded in P0 task 11).
