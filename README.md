# Sutradhar

> A production-grade, multilingual assistant that finds an Indian film from its story, plot, or cast — cross-lingual remake/dub aware.

Ask about **Papanasam** (Tamil) and Sutradhar knows the original is **Drishyam** (Malayalam), surfacing **all** language versions — the original plus every remake and official dub — with the original clearly flagged and every claim grounded in a cited source.

## The hard problem

Cross-lingual entity resolution across **remakes** and **dubs**. The same story exists as separate
films with separate casts across languages (remake) **and** as the same film with replaced audio
(dub) — different relationships that must not be conflated. The worked example that gates the build:

```
Drishyam (2013, Malayalam, Mohanlal) = ORIGINAL
  ├─ Drishya   (2014, Kannada)
  ├─ Drushyam  (2014, Telugu,   Venkatesh)
  ├─ Papanasam (2015, Tamil,    Kamal Haasan)
  └─ Drishyam  (2015, Hindi,    Ajay Devgn)
```

Any correct answer to a Papanasam query must return the Malayalam original plus all four Indian
remakes, labelled by relationship. Queries arrive code-mixed (Hinglish/Tanglish) and in native
scripts; titles match across scripts via transliteration.

## Architecture: hybrid, not pure fine-tuning

Fine-tuning teaches **behaviour and format**, not facts. The catalog and remake graph are
frequently-updated factual data, so they live in **retrieval**, never in weights.

- **RAG owns the facts** — catalog, remake/dub graph, grounding, citations.
- **QLoRA owns the behaviour** — code-mixed intent parsing, slot extraction, multi-turn
  backtracking, tool-calling, answering in the user's language/register.

If QLoRA does not measurably beat a well-prompted base model on the generation metrics, we cut it
and document why.

## Subsystems

1. **Chat UI** — find-a-movie chat; shows all language versions with the original flagged; renders citations and a trace view.
2. **API layer (FastAPI)** — orchestration, guardrails, caching, token/cost/latency tracking.
3. **RAG Engine** — query normalization + transliteration, hybrid retrieval (BGE-M3 dense + sparse), cross-encoder reranking (bge-reranker-v2-m3), grounding, prompt-injection guardrails.
4. **Catalog + Remake-Graph store** — Postgres modelling canonical Work nodes and per-language Version nodes with typed edges; embeddings in pgvector or Qdrant.
5. **Conversation/Intent model** — Gemma 4 E4B + QLoRA adapter (see `docs/DECISIONS.md` DEC-0001).
6. **Serving** — vLLM on a rented GPU, brought up **on-demand** for demo/benchmark, then stopped.

Cross-cutting: **Evals & Observability** (RAGAS + Langfuse + MLflow), CI-gated.

## Repo layout

| Path | Purpose |
|------|---------|
| `/data-pipeline` | Ingestion (TMDB/Wikidata/IMDb), remake-graph build, transliteration/normalize |
| `/finetune` | Synthetic data gen, QLoRA training, merge; optional GGUF quantize |
| `/rag-engine` | Embeddings, hybrid retrieval, reranker, grounding, guardrails |
| `/serving` | FastAPI app + vLLM serving adapter |
| `/evals` | RAGAS harness, golden test set, two-table benchmark runner |
| `/ui` | Chat + trace view |
| `/infra` | Docker, docker-compose, CI |
| `/docs` | ROADMAP, DECISIONS, RUNBOOK, BENCHMARKS, LICENSING, PORTFOLIO, DATA_SOURCES, GOLDEN_SET_SCENARIOS |

### Directory → import package

The `CLAUDE.md` subsystem directories are hyphenated and are **not** valid import names. All Python
code lives in one installable package, `sutradhar` (under `src/`, per `docs/DECISIONS.md` DEC-P0-2);
the hyphenated top-level dirs hold entrypoint scripts, Dockerfiles, and READMEs that import from
`sutradhar.*`. Directory name ≠ import name by design.

| Directory | Import package |
|-----------|----------------|
| `data-pipeline` | `sutradhar.pipeline` |
| `rag-engine` | `sutradhar.rag` |
| `serving` | `sutradhar.serving` |
| `finetune` | `sutradhar.finetune` |
| `evals` | `sutradhar.evals` |
| `infra` | — (containers / CI; no import package) |
| `ui` | — (frontend assets; no import package) |

## Cost discipline (a first-class feature)

Nothing inference-side runs 24/7. The GPU is rented (never owned), brought up only to capture the
benchmark and for live interview demos, then stopped. The standing portfolio evidence is the
**documented benchmark** from the live GPU run — not a live endpoint.

## Quickstart

Fresh clone → running stack in one command each (task runner is `make`; see `make help`):

```bash
cp .env.example .env       # fill HF_TOKEN etc. as needed; no secret goes in git
make setup                 # uv sync — install the locked environment
make up                    # start Postgres(+pgvector) + Redis, wait for healthy
make smoke                 # LLM connectivity: token if the GPU is up, graceful "endpoint OFF" if not
make hf-check              # verify Hugging Face auth (whoami)
make down                  # stop the stack
```

**30-second demo:** `make up && make smoke` shows the live stack and the graceful endpoint-OFF
message (the default — the on-demand GPU is normally paused).

| Target | Purpose |
|--------|---------|
| `setup` | `uv sync` the locked env |
| `fmt` / `lint` / `typecheck` | ruff format / ruff check / mypy (strict) |
| `test` / `test-int` | unit tests / integration tests (needs `make up`) |
| `check` | Tier-1 gate: lint + typecheck + unit tests |
| `up` / `down` / `down-v` | compose stack up / down / down + drop volume |
| `smoke` / `hf-check` | LLM connectivity smoke / HF auth check |
| `gpu-validate` / `gpu-nuke` | one-time ephemeral GPU validation / stray-instance safety |

## Status

**P0 — Foundation: complete.** Reproducible skeleton in place:

- `uv`-locked Python monorepo (`src/sutradhar`), ruff + mypy (strict) + pytest, two-tier CI.
- Env-driven `pydantic-settings` config with secret redaction; committed `.env.example`.
- Dockerized Postgres (+pgvector) + Redis with healthchecks (`make up`).
- Graceful OpenAI-compatible LLM smoke (`make smoke`) — green whether the GPU is up or off — and an
  HF Hub auth check (`make hf-check`).
- One-command on-demand GPU validation (`make gpu-validate`): create → vLLM serve → smoke → destroy,
  teardown guaranteed. **Live-validated Gemma-4-E4B on an A100 at ~98 tok/s for ~$0.34**
  (evidence in [`infra/README.md`](./infra/README.md); DEC-0001 follow-up discharged).
- Protected `main` (ruleset: required Tier-1 checks + PR + no force-push).

See [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) (two-table skeleton) and
[`docs/PORTFOLIO.md`](./docs/PORTFOLIO.md) for the quantified results (filled from P2 onward).
Next: **P1** — data ingestion + the Work/Version remake graph.

See [`CLAUDE.md`](./CLAUDE.md) for the full engineering operating agreement and [`docs/`](./docs)
for data sourcing, decisions, and golden-set scenarios.

## Licensing

Data sources carry mixed licenses (IMDb non-commercial, Wikidata CC0, TMDB developer terms,
Wikipedia CC BY-SA). This is a **non-commercial portfolio project**. See `docs/LICENSING.md`
(planned) and `docs/DATA_SOURCES.md`.
