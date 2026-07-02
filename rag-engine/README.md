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
**P2 in progress.** Landed so far:

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

## Tests

```
uv run pytest tests/test_chunking.py                                  # hermetic
uv run pytest -m integration tests/integration/test_build_corpus.py  # needs `make up`
```
