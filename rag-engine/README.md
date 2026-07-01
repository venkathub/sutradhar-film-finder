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
**Not built until P2.** P0 creates this directory as a stub only.
