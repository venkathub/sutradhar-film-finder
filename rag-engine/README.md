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

## Tests

```
uv run pytest tests/test_chunking.py tests/test_artifacts.py               # hermetic
uv run pytest -m integration tests/integration/test_build_corpus.py       # needs `make up`
```
