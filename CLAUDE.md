# CLAUDE.md — Sutradhar Engineering Operating Agreement

## Mission
Build Sutradhar: a production-grade, multilingual assistant that finds an Indian film from its
story, plot, or cast, supports conversational backtracking (the user refines across turns), and
is cross-lingual remake/dub aware. If a user asks about "Papanasam" (Tamil), Sutradhar knows the
original is "Drishyam" (Malayalam) and surfaces ALL language versions — the original plus every
remake and official dub — with the original clearly flagged and every claim grounded in a cited
source. The goal is a deployed, demonstrable system that proves production AI-engineering depth:
hybrid RAG, cross-lingual entity resolution, QLoRA fine-tuning, evaluation, cost-aware infra, and
observability.

## Audience for this codebase
A senior backend engineer (10+ yrs Java full-stack) transitioning into AI engineering. Code,
READMEs, and decisions must read as the work of someone who can take AI from prototype to
reliable production. Favour clarity and correctness over cleverness.

## The hard problem (this is the spine of the project)
Cross-lingual entity resolution across remakes and dubs. The same story exists as separate films
with separate casts across languages (remake) AND as the same film with replaced audio (dub) —
these are different relationships and must not be conflated. Worked example that gates the build:
  Drishyam (2013, Malayalam, Mohanlal) = ORIGINAL
    -> Drishya (2014, Kannada) - Drushyam (2014, Telugu, Venkatesh)
    -> Papanasam (2015, Tamil, Kamal Haasan) - Drishyam (2015, Hindi, Ajay Devgn)
Any correct answer to a Papanasam query must return the Malayalam original plus all four Indian
remakes, labelled by relationship. Queries arrive code-mixed (Hinglish/Tanglish) and in native
scripts; titles must match across scripts via transliteration.

## Architecture decision: HYBRID, not pure fine-tuning
Fine-tuning teaches behaviour and format, NOT facts. The film catalog and remake graph are
frequently-updated factual data, so they live in retrieval, never in weights. The split:
- RAG owns the facts: catalog, remake/dub graph, grounding, citations.
- QLoRA owns the behaviour: code-mixed intent parsing, slot extraction, multi-turn backtracking,
  tool-calling, and answering in the user's language/register.
If QLoRA does not measurably beat a well-prompted base model on the generation metrics, we cut it
and document why. Knowing when fine-tuning did NOT help is a senior signal, not a failure.

## Subsystems
1. Chat UI — find-a-movie chat; shows all language versions with the original flagged; renders
   citations and a trace view.
2. API layer (FastAPI) — orchestration, request/response guardrails, caching, and token/cost/
   latency tracking. (Optional Java moat: the public gateway MAY be Spring Boot if I choose to
   showcase Java; decide in P5 grooming, don't assume.)
3. RAG Engine — query normalization + transliteration, hybrid retrieval (BGE-M3 dense + sparse),
   cross-encoder reranking (bge-reranker-v2-m3), grounding and source attribution, prompt-
   injection guardrails.
4. Catalog + Remake-Graph store — Postgres modelling canonical Work nodes and per-language
   Version nodes with typed edges (is_original_of / is_remake_of / is_official_dub_of /
   is_unofficial_remake_of / is_sequel_of); embeddings in pgvector OR Qdrant (decide in P2).
5. Conversation/Intent model — Gemma 4 E4B with a QLoRA adapter: code-mixed intent, slot
   extraction, backtracking, tool-calling. (Base model per docs/DECISIONS.md DEC-0001.)
6. Serving — vLLM on a rented GPU, brought up ON-DEMAND for a live demo (and for the benchmark
   capture), then stopped. There is NO 24/7 CPU-served model: we do not keep an LLM live on a CPU
   host. The standing portfolio evidence is the documented benchmark from the live GPU run, not a
   live endpoint. Endpoint chosen by env var, never hardcoded. (Optional: a GGUF/llama.cpp build is
   kept only as a portable local fallback, not as a deployed always-on service.)
Cross-cutting: Evals & Observability (RAGAS + Langfuse + MLflow), CI-gated.

## Tech stack
- Python primary: Hugging Face Transformers / PEFT / TRL (QLoRA), vLLM (GPU serving; llama.cpp/GGUF
  only as an optional portable fallback), FastAPI (API), RAGAS (eval), indic-transliteration +
  AI4Bharat IndicXlit (transliteration), rapidfuzz (fuzzy title match).
- Data sources: TMDB API (titles, translations, alternative_titles), Wikidata SPARQL (remake
  edges via P144 "based on" / P1877 "after a work by"), IMDb non-commercial dumps (title.akas
  for AKA/dub titles), curated Kaggle Indian-movie sets to backfill South-Indian coverage.
- Stores: Postgres (+ pgvector) or Qdrant; Redis optional cache.
- Models (see docs/DECISIONS.md DEC-0001 for rationale + sources; all IDs env-driven): serve the
  fine-tuned Gemma 4 E4B (base + QLoRA adapter via vLLM) on the on-demand GPU; base for benchmark =
  Gemma 4 E4B (Qwen3-4B-Instruct-2507 as fallback); embeddings = BGE-M3 (MIT); reranker =
  bge-reranker-v2-m3; optional bigger live showcase AND synthetic-data teacher = Sarvam-M 24B
  (Apache 2.0) on the GPU — NOT the fine-tune base (it is already Indic-specialized, leaving no
  beatable headroom). GGUF quantization is optional (only needed if a CPU fallback is ever wanted),
  not a deploy requirement.
- MLOps: MLflow (tracking + model registry, self-hosted), Langfuse (LLM tracing — self-hosted v3
  on an AIC Cloud VPS per DEC-P3-7; Langfuse Cloud free tier is the documented fallback),
  GitHub Actions (CI with eval gate), Docker + docker-compose, Hugging Face Hub (artifact registry
  / reproducibility bridge).

## Data & licensing (a maturity signal — keep it visible)
- IMDb datasets: personal/non-commercial ONLY. Fine for a portfolio demo; never commercialize.
- Wikidata: CC0 (public domain) — safe. The remake/dub/sequel graph spine.
- TMDB: free developer API (attribution required); verify current terms; data is community-editable.
- Wikipedia: CC BY-SA 4.0 (attribution + share-alike). Accessed via API/dump, NEVER HTML-scraped.
  Used for plot text (embeddings) and as a candidate-edge source under human verification.
- Sarvam-M (24B): Apache 2.0. Gemma 4 / Qwen3: Apache 2.0. Avoid Sarvam-1 (2B) — non-commercial.
- Sourcing strategy and per-field precedence live in docs/DATA_SOURCES.md. "High confidence" is an
  ENFORCED property: a record/edge is ground-truth only if HIGH confidence (>=2 independent sources
  agree, or an authoritative structured source) OR human-verified, with no unresolved conflict.
  LLM-extracted edges go to a candidate_edges table and never enter the live graph unverified.
- Maintain docs/LICENSING.md mapping every source/model to its license and our usage + attribution.

## Environment & infra constraints (budget is a first-class feature)
- Developer laptop is low-spec: it runs ONLY Claude Code + editor + Docker for app services. NO
  neural-model operation (LLM, embedding, reranker, or neural transliteration) trains, serves, or
  runs inference on the laptop — every model op runs on the rented on-demand GPU; the laptop and CI
  operate on persisted artifacts only. Deterministic rule-based transliteration is not a model.
- GPU is rented, never owned. JarvisLabs (per-minute billing; pause = storage-only) runs the neural
  jobs in short on-demand sessions — candidate-edge extraction (P1), embedding/index build +
  retrieval eval (P2), and the ONE-TIME QLoRA fine-tune + benchmark capture (P4) — then STOP the
  instance. After pushing artifacts to the HF Hub, the volume can be DELETED and the stack rebuilt
  from scratch — we never depend on a warm machine.
- NO 24/7 CPU DEPLOYMENT. We do not run the LLM on a CPU host for always-on uptime. There is no
  permanently-live inference endpoint. This is a deliberate cost decision, not a gap.
- The standing portfolio evidence is the BENCHMARK from the live GPU run, documented with strong
  evidence: the two results tables, MLflow run links, Langfuse traces, screenshots, GPU latency/
  throughput (tokens/sec), and a recorded demo video. These artifacts are always available (in the
  repo / README) even though the model is not. The evidence is the proof; the endpoint is on-demand.
- On-demand GPU is used for two things only: (a) the ONE-TIME training + benchmark-capture run, and
  (b) a QUICK live demo if an interviewer asks during a call. Resume JarvisLabs (sub-2-min), bring
  the stack up with one command, demo, then STOP. Default the live demo to the fine-tuned Gemma 4
  E4B (fast to load); the Sarvam-M 24B is an optional "if time permits" showcase.
- A small/cheap always-on host (e.g. a low-tier VPS or static host) MAY serve only the static
  surface — landing page, README, architecture diagram, the recorded demo video, and the benchmark
  report — plus, optionally, dashboards over precomputed/recorded results. It never serves any neural
  model (LLM, embedder, or reranker); any live query path requires the on-demand GPU.
- GPU is rented, never owned. After the training+benchmark run, push artifacts to the HF Hub, then
  STOP and delete the volume; the whole stack is rebuilt from scratch on demand, never kept warm.
- Cost discipline is part of the portfolio: "nothing inference-side runs 24/7; the GPU is up only
  to capture the benchmark and for live interview demos" is the lived proof of the cost-aware
  engineering the dashboards and benchmark advertise.
- All model config swappable by env (e.g. LLM_BASE_URL, LLM_MODEL, EMBED_MODEL), never hardcoded.

## Working agreement (how you, Claude Code, must operate)
- **One subsystem per session**, built in dependency order. Do not scaffold everything at once.
- **RAG baseline + eval harness before fine-tuning.** The retrieval system and its eval gate are
  built and passing BEFORE any QLoRA work. Never ship a retrieval or model change without an eval
  result. Gate: RAG Recall@10 >= 0.90 on the golden set before we invest in fine-tuning.
- **Two-table benchmark, kept honest.** Fine-tuning does not touch retrieval, so NEVER present
  retrieval metrics as "before/after FT." Keep them separate (see Definition of Done).
- **Plan before code.** For any non-trivial task, first output a short plan (files to touch,
  approach, test strategy, risks) and WAIT for my approval before writing code.
- **Test-backed.** Every feature ships with tests. A phase is not done until its tests and its
  eval thresholds pass in CI.
- **Small, reviewable diffs.** Prefer incremental commits with clear messages over giant drops.
- **Decisions are logged.** Any architectural choice (chunking, embedding model, vector store,
  graph schema, retrieval thresholds, base model, quantization) gets a dated entry in
  docs/DECISIONS.md with options considered + rationale. This is also my interview script.
- **No secrets in code.** Use env vars + .env.example. Never print real keys or tokens.
- **Ask, don't assume.** If a requirement is ambiguous, ask one focused question rather than
  guessing. State any assumption you do make, inline.
- **Honesty over agreeableness.** If my idea is weak or a shortcut creates tech debt, say so and
  propose the better path.

## Repo conventions
- /data-pipeline (ingestion from TMDB/Wikidata/IMDb, remake-graph build, transliteration/normalize)
- /finetune (synthetic data generation, QLoRA training, merge; optional quantize to GGUF)
- /rag-engine (embeddings, hybrid retrieval, reranker, grounding, guardrails)
- /serving (FastAPI app + vLLM serving adapter; optional llama.cpp/GGUF fallback)
- /evals (RAGAS harness, golden test set, the two-table benchmark runner)
- /ui (chat + trace view)
- /infra (Docker, docker-compose, CI)
- /docs (ROADMAP.md, DECISIONS.md, RUNBOOK.md, BENCHMARKS.md, LICENSING.md, PORTFOLIO.md,
  DATA_SOURCES.md, GOLDEN_SET_SCENARIOS.md, phases/P<#>_SPEC.md)
- Each module has its own README: purpose, architecture, setup, how to run tests, results/metrics.
- Conventional commit messages. Feature branches; CI must pass before merge to main.

## Definition of Done (every phase)
- [ ] Code complete and matches the approved phase spec.
- [ ] Unit + integration tests written and passing.
- [ ] Eval thresholds met (where the phase touches retrieval/generation) and recorded to MLflow.
- [ ] Where applicable, BOTH benchmark tables updated in docs/BENCHMARKS.md:
        (1) Retrieval quality (model-independent): Recall@k, MRR, version-set recall — incl. the
            Papanasam/Drishyam case.
        (2) Generation/agent quality, base vs QLoRA: tool-call accuracy, code-mixed intent/slot
            accuracy, backtracking coherence, faithfulness (1 - hallucinated-movie rate), answer
            relevancy — plus GPU latency/throughput (tokens/sec), captured during the live GPU run
            with evidence (MLflow links, Langfuse traces, screenshots).
- [ ] Module README + docs/DECISIONS.md updated.
- [ ] Runs cleanly from scratch via documented setup (fresh clone + .env).
- [ ] A 30-second demo path I can click/run.
- [ ] A resume-ready, quantified bullet drafted for docs/PORTFOLIO.md.