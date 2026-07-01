# Sutradhar Roadmap

> The build plan, in dependency order, for a production-grade multilingual film-finding assistant
> that is cross-lingual remake/dub aware. This roadmap is the spine that ties `CLAUDE.md` (operating
> agreement), `docs/DATA_SOURCES.md` (sourcing + confidence), and `docs/GOLDEN_SET_SCENARIOS.md`
> (eval scenarios GS-01..GS-11) into a single sequenced program of work.
>
> **Status:** APPROVED (2026-07-01). Baseline for execution. Changes go through `docs/DECISIONS.md`.

---

## 1. Product vision & the one story that forces the whole build

**Vision.** Sutradhar lets anyone find an Indian film from a half-remembered story, plot fragment,
or cast detail — typed in English, a native script, or code-mixed Tanglish/Hinglish — and answers
with *every* language version of that story, the original clearly flagged, remakes and official dubs
correctly distinguished, and every claim grounded in a cited source. It refines across conversational
turns when the user corrects or narrows the ask. It proves production AI-engineering depth: hybrid
retrieval over a typed remake graph, cross-lingual entity resolution, QLoRA behaviour-tuning, a
CI-gated eval harness, cost-aware on-demand GPU serving, and full observability.

**Model stack (decided in `docs/DECISIONS.md` DEC-0001, 2026-07-01; all IDs env-driven):**

| Role | Model | License | Why |
|---|---|---|---|
| **Fine-tune base** | **Gemma 4 E4B** | Apache 2.0 | Strong general 4B with *headroom* on Indic code-mix + our tool schema (→ **beatable base benchmark**); native function-call tokens; day-0 vLLM + single-GPU QLoRA; fast to load on-demand |
| **Fallback FT base** | **Qwen3-4B-Instruct-2507** | Apache 2.0 | Top sub-7B OOB tool-calling; used if Gemma 4 tooling is unstable (lead its FT story with GS-07/GS-08, not raw tool-calling) |
| **Optional 24B showcase + data-teacher** | **Sarvam-M 24B** | Apache 2.0 | Already Indic-specialized → **not** the FT base (no beatable headroom); instead the "big Indic" live-demo contrast and the P4 **synthetic-data teacher** |
| Embeddings / reranker | BGE-M3 / bge-reranker-v2-m3 | MIT / permissive | Unchanged |

> **"Beatable benchmark" principle:** the FT base is chosen to be *strong in general but unspecialized
> in our niche* (Tanglish/Hinglish intent+slot, backtracking, our tool schema), so QLoRA lift on
> GS-07/GS-08 is real and demonstrable — not an already-Indic model where FT can't win.

### The single end-to-end gating story

The gating story is told from **two views of the same conversation**: **(a)** what the user
experiences, and **(b)** the operational guarantees that had to be true for that experience to be
*trustworthy, reproducible, observable, and cheap*. View (a) forces the application subsystems;
view (b) forces the MLOps machinery. Neither view can be dropped without the story failing — that
is what makes it comprehensive.

#### (a) What the user experiences

> A user types in Tanglish: **"that Kamal Haasan movie where he hides a body to protect his family."**
> Sutradhar identifies **Papanasam (2015, Tamil)**, recognizes it as the Tamil **remake** of the
> Malayalam **original Drishyam (2013)**, returns **all** language versions — Kannada *Drishya* (2014),
> Telugu *Drushyam* (2014), Tamil *Papanasam* (2015), Hindi *Drishyam* (2015) — with the original
> flagged and each version labelled by relationship, **every claim citing its source**. The user
> **clicks a citation** and sees the exact Wikidata/TMDB record and confidence behind it. They probe
> with a decoy — **"is there a Marathi one set on Mars?"** — and Sutradhar answers **NO_MATCH**,
> inventing nothing. They correct: **"no, the newer one"**, and Sutradhar **refines across the turn**
> to the right version without losing context — then offers a **trace view** of how it reasoned.

#### (b) The operational guarantees behind that answer (the MLOps story)

> That answer only shipped because **retrieval had already passed Recall@10 ≥ 0.90 in CI** against
> the golden set, so the version set is *known-complete, not hoped-complete*. Every cited claim traces
> to a **confidence-gated, sourced graph record** — nothing LLM-guessed reached the answer. The
> correct handling of the Tanglish query and the "no, the newer one" backtrack exists because
> **QLoRA measurably beat the well-prompted base model** on those exact fixtures, a result **tracked
> in MLflow** and reproducible from an **adapter pinned on the HF Hub**. The whole turn — retrieval
> hits, tool calls, tokens, latency — is a **Langfuse trace**. The GPU that served it was brought up
> **on-demand, its tokens/sec recorded, then stopped**; and when the GPU is off (the default), the
> *same story replays from recorded benchmark evidence* instead of erroring.

Every clause of **both** views forces a subsystem into existence. You cannot satisfy the story
end-to-end without building all of Sutradhar — application **and** MLOps.

**Table A — user-facing clauses → application subsystem**

| Clause of the story | Subsystem it forces | Golden scenario |
|---|---|---|
| *"types in Tanglish …"* | **Conversation/Intent model** (code-mixed intent) + **RAG** query normalization / **transliteration** | GS-07, GS-11 |
| *"… movie where he hides a body to protect his family"* (plot, no title) | **RAG Engine** — dense semantic retrieval with no proper-noun anchor | GS-03 |
| *"that Kamal Haasan movie"* (cast/actor anchor) | **RAG Engine** sparse match + **Catalog** cast index; **de-dup** so Kamal-Haasan collisions don't merge | GS-10 |
| *"identifies Papanasam"* | **Catalog + Remake-Graph** canonical Work/Version resolution | GS-01 |
| *"the Tamil remake of the Malayalam original Drishyam"* | **Remake-Graph** typed edges (`is_remake_of` vs `is_original_of`), dub-vs-remake distinction | GS-01, GS-04, GS-05 |
| *"returns all language versions … original flagged"* | **Remake-Graph** version-set traversal + original flag; **UI** rendering | GS-01, GS-06 |
| *"every claim citing its source" / "clicks a citation"* | **RAG** grounding/attribution + **Data pipeline** `sources[]` + confidence surfaced to UI | all |
| *"is there a Marathi one set on Mars?" → NO_MATCH* | **Guardrail** — no-hallucinated-movie on out-of-catalog asks | GS-02 |
| *"no, the newer one" → refines across turns* | **Conversation/Intent model** multi-turn backtracking; **API** orchestration/state | GS-08 |
| *"offers a trace view"* | **UI** trace view fed by **Observability** | GS-08 |

**Table B — operational guarantees → MLOps subsystem + standing evidence artifact**

| Operational guarantee in the story | MLOps subsystem it forces | Phase | Standing evidence artifact |
|---|---|---|---|
| *"retrieval had already passed Recall@10 ≥ 0.90 in CI"* | **Eval harness + CI gate** (retrieval metrics on golden set) | P2/P3 | `BENCHMARKS.md` Table 1; green CI check |
| *"every cited claim traces to a confidence-gated, sourced graph record"* | **Data provenance + verification gate** (`sources[]`, confidence, `candidate_edges`) | P1 | Gate report; precision/recall-lift metric |
| *"QLoRA measurably beat the base model on those exact fixtures"* | **Experiment tracking + honest before/after benchmark** | P3 (base) + P4 (FT) | `BENCHMARKS.md` Table 2 (base vs QLoRA) |
| *"reproducible from an adapter pinned on the HF Hub"* | **Model registry / reproducibility bridge** (MLflow registry + HF Hub) | P4 | HF Hub repo: adapter + metrics; MLflow run link |
| *"the whole turn is a Langfuse trace"* | **Observability / tracing** | P3/P5 | Shared Langfuse trace links; UI trace view |
| *"GPU brought up on-demand, tokens/sec recorded, then stopped"* | **Cost-aware on-demand GPU serving** (vLLM, env-driven endpoint) | P4/P5 | tokens/sec + cost/latency dashboard screenshots; RUNBOOK |
| *"when the GPU is off, the same story replays from recorded evidence"* | **Graceful degradation** (evidence-backed static surface) | P5/P6 | Recorded demo video + benchmark report on the always-available host |

> **Why two views matter for this portfolio.** A find-a-movie demo proves you can wire an LLM. The
> *operational* view is what proves production AI-engineering: the answer is **gated** (CI eval),
> **grounded** (provenance), **measured** (MLflow before/after), **reproducible** (HF Hub),
> **observable** (Langfuse), and **cost-disciplined** (on-demand GPU, graceful degradation). The
> standing evidence artifacts in Table B are exactly what an interviewer can inspect when the live
> endpoint is off — which, by design, is most of the time.

---

## 2. Phase plan (P0 → P6)

Phases run in strict dependency order, one subsystem per session. A phase is not "done" until its
Definition of Done (exit criteria) is met **and** recorded per the DoD checklist in `CLAUDE.md`.
Cross-cutting engineering guardrails (artifact versioning, judge governance, injection defense,
two-tier CI) live in **§6** and apply to every phase.

> **Build strategy — vertical slice first.** Before scaling the catalog, take a **~20-film seed slice**
> that covers the flagship chains (Drishyam family GS-01/06, Baahubali GS-04, Devdas GS-05, Vikram
> collision GS-10, Bhool Bhulaiyaa GS-09) end-to-end through P1→P6. This exercises every gate and
> yields a working demo early; catalog breadth is scaled *after* the pipeline is proven.

> **Compute placement (strict).** The laptop runs only code, Postgres/Docker, data-pipeline
> orchestration (API/dump ingestion, graph build, deterministic rule-based transliteration), CI, and
> artifact storage. **Every neural-model operation runs on the rented on-demand GPU** — corpus and
> query embedding, cross-encoder reranking, neural transliteration (IndicXlit), LLM candidate-edge
> extraction, QLoRA fine-tune, and generation/serving. GPU work is done in short batched sessions and
> outputs are persisted as **versioned artifacts** (vector index, embedded queries, rerank scores,
> candidate edges, benchmarks) so laptop-side development and CI operate on artifacts and never invoke
> a model. Dense vector search itself runs inside Postgres/pgvector on stored embeddings (no model).

### P0 — Foundation: repo, infra, CI, GPU connectivity smoke test

- **Goal:** A clean, reproducible skeleton where every later phase plugs in; prove we can reach the
  on-demand GPU model endpoint without hardcoding anything.
- **Subsystems touched:** Cross-cutting infra (`/infra`), API skeleton (`/serving` shell), config.
- **Skills demonstrated:** Env-driven config, Docker/compose, CI design, secret hygiene, HF Hub
  auth, cost-aware "endpoint is off by default" wiring.
- **Entry criteria:** Roadmap approved. Repo has `CLAUDE.md`, `DATA_SOURCES.md`,
  `GOLDEN_SET_SCENARIOS.md`.
- **Exit criteria (DoD):**
  - Repo scaffold matches the `/…` layout in `CLAUDE.md`; each module has a stub README.
  - `.env.example` covers `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, DB/Redis, HF token, TMDB key,
    Langfuse keys — **no secret in code**.
  - `docker-compose` brings up Postgres(+pgvector)/Redis locally; `make`/task targets documented.
  - **Two-tier CI (GitHub Actions)** scaffolded: **Tier 1** (every PR, no GPU, no model calls) runs
    lint + unit tests and validates against recorded artifacts; **Tier 2** (manual/tagged dispatch
    during a GPU window, used from P2 on) runs the eval harness. Branch protection notes recorded.
  - HF Hub auth verified (whoami) from CI/local via token env var.
  - **Connectivity smoke test:** a scripted call to `LLM_BASE_URL` returns a token when the GPU is
    up, and **degrades to a clear "endpoint OFF" message** (not a crash) when it is down. Test is
    green in both states.
  - `docs/DECISIONS.md` seeded; first decision entries logged.

### P1 — Data pipeline + remake-graph (the hard problem's foundation)

- **Goal:** Build the canonical Work/Version graph in Postgres with typed edges, per-record
  confidence + `sources[]`, an enforced verification gate, an LLM candidate-edge extraction layer
  behind a human gate, and the seed golden set for GS-01..GS-11.
- **Subsystems touched:** Data pipeline (`/data-pipeline`), Catalog + Remake-Graph store, Evals
  (golden fixtures under `/evals/golden`).
- **Skills demonstrated:** Cross-lingual entity resolution, graph data modelling, multi-source
  conflict resolution, transliteration/normalization, LLM information-extraction with a
  human-in-the-loop gate, precision/recall measurement, data-licensing maturity.
- **Entry criteria:** P0 done. TMDB key + IMDb dumps + Wikidata SPARQL reachable; on-demand GPU
  endpoint available for the extraction and neural-transliteration passes.
- **Exit criteria (DoD):**
  - Ingestion per `DATA_SOURCES.md`: **Wikidata** (P144/P1877/P4969 + sequel P155/P156/P179 +
    ID linking) as the relationship spine; **TMDB** (multilingual titles/translations/cast);
    **IMDb `title.akas`** (dub/AKA titles); **Wikipedia prose** (plot text) — all via **API/dump,
    never HTML scraping**.
  - Postgres schema: canonical `Work` nodes + per-language `Version` nodes; typed edges
    (`is_original_of` / `is_remake_of` / `is_official_dub_of` / `is_unofficial_remake_of` /
    `is_sequel_of`); every record/edge carries `confidence` + `sources[]`.
  - Titles normalized + transliterated for cross-script match: deterministic rule-based
    `indic-transliteration` on the laptop; neural IndicXlit, where used, runs on a rented-GPU session
    with outputs cached. `conflicts` queue populated, never silently resolved.
  - **LLM extraction layer** (run on a rented-GPU session) proposes missing remake/dub edges from
    Wikipedia prose into `candidate_edges` (with supporting sentence + model confidence) — **never
    straight into the graph**. Human-verification gate promotes confirmed edges (`human_verified = true`).
  - **Verification gate enforced:** a record/edge is ground-truth only if HIGH confidence
    (≥2 independent sources agree, or authoritative structured source) **or** human-verified, with
    **no unresolved conflict** and populated `sources[]`/`confidence`.
  - **Precision/recall lift reported:** verified edges added beyond Wikidata, and candidate precision
    (confirmed/proposed) — recorded in docs.
  - **Seed golden set** built and verified against every scenario **GS-01..GS-11** (not just
    Papanasam/Drishyam); each fixture's IDs and relationship types confirmed against sources during
    ingestion; fixtures live under `/evals/golden/`.
  - Tests: schema constraints, edge-typing rules (esp. Baahubali dub GS-04, Devdas siblings GS-05,
    Vikram collision GS-10), transliteration match, gate enforcement.
  - **Graph-coverage metric (distinct from retrieval recall):** for each flagship franchise, report
    `versions_present / versions_in_curated_truth`, isolating *graph completeness* (R1) from the P2
    *retrieval* version-set recall.
  - **Tool/function schema frozen (v0)** in `docs/phases/TOOL_SCHEMA.md` — e.g. `resolve_title`,
    `get_versions`, `get_work`, `refine_filter`. This is a hard dependency for P4 synthetic-data and
    P5 orchestration and the base-vs-FT tool-call metric; it must be stable before P4.
  - `docs/LICENSING.md` created — every source/model mapped to license + our usage + attribution
    (IMDb non-commercial, Wikidata CC0, TMDB terms, Wikipedia CC BY-SA, model licenses per DEC-0001).

### P2 — RAG baseline + retrieval eval (the green-light gate for P4)

- **Goal:** A hybrid retrieval baseline good enough to justify fine-tuning — measured, not asserted.
  **No fine-tuning in this phase.**
- **Subsystems touched:** RAG Engine (embeddings, hybrid retrieval, reranker), vector store, Evals.
- **Skills demonstrated:** Hybrid (dense+sparse) retrieval, reranking, vector-store selection,
  retrieval evaluation (Recall@k / MRR / version-set recall), decision logging.
- **Entry criteria:** P1 done; golden fixtures frozen (HIGH/human-verified only); on-demand GPU
  available for embedding + reranking sessions.
- **Exit criteria (DoD):**
  - **Embedding A/B, decided by the gate (log in `DECISIONS.md` as DEC-0002):**
    **BGE-M3** (568M, hybrid dense+sparse) vs **bge-multilingual-gemma2** (9B, dense-only, stronger
    multilingual/MIRACL, needs a *separate* sparse signal), decided by Recall@10-vs-cost on the golden
    set. Default = BGE-M3 unless gemma2 clears a gate BGE-M3 cannot.
  - **All embedding + reranking runs on a rented-GPU session** (both models are neural; compute
    placement §2): the corpus is embedded once, golden queries are embedded and reranked per candidate
    config, and the outputs (vector index + query embeddings + rerank scores) are persisted as
    **versioned artifacts**. Recall@k / MRR / version-set recall are then computed on the laptop from
    the stored scores; configuration changes that alter embeddings or reranking trigger a fresh short
    GPU pass. Dense search runs in Postgres/pgvector (no model).
  - Hybrid indexing into **pgvector or Qdrant** (decision logged in `DECISIONS.md`) +
    **bge-reranker-v2-m3** cross-encoder reranking.
  - **Chunking strategy for plot/synopsis prose decided and logged** (`DECISIONS.md`): default
    **recursive** chunking with per-Version metadata (title/language/year) attached to each chunk;
    chunk size/overlap ablated on the plot-only slice (GS-03).
  - Retrieval metrics computed over the golden set: **Recall@k, MRR, version-set recall**, including
    the Papanasam/Drishyam case and plot-only (GS-03), franchise (GS-06), fuzzy (GS-11) slices.
  - **NO_MATCH abstention calibrated (GS-02):** the retrieval/rerank score threshold for abstention is
    tuned on a **held-out negative set**; NO_MATCH precision/recall reported; the chosen threshold
    recorded in `DECISIONS.md`.
  - **Exit gate: Recall@10 ≥ 0.90** across retrieval fixtures, **version-set recall = 1.0 on GS-01
    and GS-06** — this is the green light for P4. If not met, iterate retrieval; do **not** proceed.
  - Retrieval benchmark table written to `docs/BENCHMARKS.md` (Table 1 — model-independent).
  - Tests: retrieval regression tests wired so **Tier-1 CI** (no GPU) blocks a metric regression.

### P3 — Eval + observability harness (capture the PRE-fine-tune baseline)

- **Goal:** A CI-gated evaluation and observability harness, with the base-model generation
  benchmark **built and dry-run here**, then captured authoritatively at the *start of the single P4
  GPU window* so base and fine-tuned columns share identical serving conditions (incl. tokens/sec).
- **Subsystems touched:** Evals & Observability (RAGAS, Langfuse, MLflow), API (tracing hooks).
- **Skills demonstrated:** LLM eval (RAGAS), tracing (Langfuse), experiment tracking (MLflow),
  CI eval-gating, honest benchmark discipline (retrieval vs generation kept separate).
- **Entry criteria:** P2 gate passed (Recall@10 ≥ 0.90).
- **Exit criteria (DoD):**
  - RAGAS metrics wired (faithfulness / answer relevancy, etc.); Langfuse tracing on the request
    path; MLflow tracking + registry (self-hosted) recording runs.
  - **Tier-2 CI** runs the eval harness on dispatch and **gates merges** on gating metrics from
    `GOLDEN_SET_SCENARIOS.md` (incl. no-hallucinated-movie = 0 on GS-02). Retrieval + guardrail
    metrics run in **Tier-1** (no GPU) against recorded artifacts; generation metrics gate on the
    recorded run (see §6).
  - **Harness dry-run** against recorded fixtures / a mock endpoint (no model) here; the
    **authoritative PRE-fine-tune generation benchmark** of the well-prompted **base model**
    (Gemma 4 E4B; Qwen3-4B-Instruct-2507 fallback) is captured **at the top of the P4 GPU window** →
    `docs/BENCHMARKS.md` (Table 2, "base" column): tool-call accuracy, code-mixed intent/slot accuracy
    (GS-07), backtracking coherence
    (GS-08), faithfulness, answer relevancy. Same GPU, same vLLM config as the "QLoRA" column.
  - **LLM-as-judge governance (GS-08 coherence, RAGAS faithfulness):** judge model + version +
    prompt-hash pinned; **judge is a different model family** than the model under test (avoids
    self-preference bias, arXiv 2410.21819); judge validated against a small human-labelled sample
    with agreement reported. Frozen judge config recorded in `DECISIONS.md`.
  - Decision entry: base-model prompting strategy + judge config frozen for fair before/after.

### P4 — QLoRA fine-tune on the rented GPU (the one-time job)

- **Goal:** In a **single GPU rental**, capture the base generation column (per P3), fine-tune the
  behaviour model, capture the **AFTER benchmark with evidence**, publish artifacts, and **stop the
  GPU**. Prove FT beats the base — or document that it did not and cut it.
- **Subsystems touched:** Finetune (`/finetune`), Conversation/Intent model, Serving (transient),
  Evals/Observability.
- **Skills demonstrated:** Synthetic data generation grounded in real records, QLoRA/PEFT/TRL
  training, adapter merge, GPU cost discipline, reproducibility via HF Hub, rigorous before/after
  benchmarking.
- **Entry criteria:** P3 done (harness built + judge frozen); **tool schema v0 frozen** (P1);
  golden generation fixtures ready. (Base column is captured inside this phase's GPU window.)
- **Exit criteria (DoD):**
  - Synthetic multilingual/code-mixed **multi-turn + tool-calling** dataset generated **against the
    frozen v0 tool schema** (P1), **grounded in real graph records** (no invented films), using
    **Sarvam-M 24B (or a frontier API) as the Indic-code-mix data teacher**; dataset **documented +
    versioned** (dataset card + hash, see §6).
  - **Reproducible GPU environment:** a **pinned training container** (CUDA / torch / transformers /
    PEFT / TRL / vLLM versions locked) so "rebuild from scratch" is real and QLoRA numerics repeat.
  - **Order within the one window:** (1) capture **base** generation column → (2) **QLoRA fine-tune
    Gemma 4 E4B** (fallback Qwen3-4B-Instruct-2507) → (3) capture **QLoRA** column, identical vLLM
    config. Re-embed the corpus here **only if** the embedder changed (bge-multilingual-gemma2) or
    records changed since P2.
  - **AFTER benchmark WITH EVIDENCE** captured during the same live GPU run: MLflow run links, Langfuse
    traces, screenshots, **GPU latency/throughput (tokens/sec)** → `docs/BENCHMARKS.md` Table 2
    ("QLoRA" column) beside the base column captured in step (1).
  - Adapter merged; **adapter + metrics + recorded results pushed to HF Hub**; **GPU STOPPED**, volume
    deletable (stack rebuildable from scratch).
  - **Honest verdict logged:** GS-07/GS-08 are the fixtures where QLoRA must beat the base model; if
    it does not, record the finding and decide explicitly whether to keep the adapter.
  - (Optional) GGUF quantization **only** if a portable CPU fallback is wanted — not a deploy req.

### P5 — Serving, API & conversational backtracking

- **Goal:** The orchestration API and on-demand vLLM serving that answer the gating story end-to-end,
  with graceful degradation when the GPU is OFF (the default).
- **Subsystems touched:** API layer (FastAPI), Serving (vLLM, on-demand), Conversation/Intent model
  (backtracking), Observability dashboards.
- **Skills demonstrated:** FastAPI orchestration, tool-calling wiring, multi-turn state/backtracking,
  vLLM serving, token/cost/latency dashboards, graceful degradation as a feature.
- **Entry criteria:** P4 done; fine-tuned model + adapter on HF Hub; benchmark evidence recorded.
- **Exit criteria (DoD):**
  - FastAPI orchestration: query normalization → retrieval → grounding → LLM tool-calling →
    cited answer with the version set + original flag; guardrails (prompt-injection, no-hallucination).
  - **Indirect prompt-injection defense** for retrieved Wikipedia/plot text (attacker-influenceable
    content entering the prompt): **spotlighting / delimiting untrusted context**, a chunk-level
    adversarial-content check, and an eval slice of injection fixtures **in the query *and* in
    retrieved context** (BIPIA-style). Recorded as a guardrail metric, not just query-side GS-02.
  - **Multi-turn backtracking** works end-to-end (GS-08): "no, the newer one" refines within the
    version set without losing context.
  - vLLM serves the fine-tuned Gemma 4 E4B via `LLM_BASE_URL` (env, never hardcoded), **on-demand**.
  - Token/cost/latency dashboards live.
  - **Graceful degradation:** with the GPU endpoint OFF (default), the live query path is unavailable
    (query embedding, reranking, and generation all require the on-demand GPU per §2), so the app
    serves the **recorded benchmark/demo** and a **"request a live demo"** state — it never errors.
    **No always-on CPU model.**
  - Integration tests cover both GPU-on and GPU-off paths.

### P6 — UI, containerization, always-available surface & runbook

- **Goal:** The chat UI, the static portfolio surface that is always available without the LLM, and
  the runbook for a quick on-demand GPU bring-up for live interview demos.
- **Subsystems touched:** Chat UI (`/ui`), infra/containerization, docs.
- **Skills demonstrated:** Product-quality UI with citations + trace view, containerization,
  cheap static hosting, operational runbook writing, portfolio packaging.
- **Entry criteria:** P5 done.
- **Exit criteria (DoD):**
  - Chat UI: find-a-movie chat, all language versions with the original flagged, citations, trace
    view; renders the graceful-degradation state when the GPU is off.
  - Full stack containerized; one-command bring-up documented.
  - **Static always-available surface** on a cheap host: landing page + README + **architecture
    diagram** + **recorded demo video** + **benchmark report**. Never serves any neural model (LLM,
    embedder, or reranker) — static and precomputed content only.
  - **`docs/RUNBOOK.md`:** quick on-demand GPU bring-up (JarvisLabs resume < 2 min → one-command
    stack up → demo → **STOP**), plus teardown/rebuild-from-scratch steps.
  - Top-level README ties everything together; `docs/PORTFOLIO.md` has quantified bullets.
  - **No 24/7 inference deployment** exists — by design.

---

## 3. Risk register (top 5) & how phases de-risk them

| # | Risk | Why it bites | Primary de-risking phase(s) | Mitigation |
|---|---|---|---|---|
| R1 | **Sparse Wikidata remake coverage for South-Indian films** | The relationship spine has holes exactly where the project's flagship lives → low version-set recall | **P1** (built), P2 (measured) | Wikidata as high-precision base + **LLM candidate-edge extraction over Wikipedia prose behind a human gate**; measure & report the precision/recall lift; enforce the verification gate so recall gains stay honest |
| R2 | **Fine-tuning fails to beat the base model** | Wasted GPU spend; a weak "after" column | **P3** (honest base capture), **P4** (verdict) | Choose a general base with real Indic-niche headroom (Gemma 4 E4B), not an already-Indic model (DEC-0001); train on Sarvam-M-taught, record-grounded data; freeze base prompting + judge config in P3; keep retrieval vs generation benchmarks separate; if GS-07/08 do not improve, record the finding and cut the adapter |
| R3 | **Transliteration / cross-script match failures** | Tanglish/Hinglish and misspellings never resolve to the right Work → the gating story breaks at turn 1 | **P1** (normalize/translit), P2 (fuzzy+sparse eval) | indic-transliteration + IndicXlit normalization at ingest and query time; rapidfuzz fuzzy match; **GS-11** and **GS-07** fixtures gate it; title-match-under-noise metric |
| R4 | **On-demand GPU bring-up too slow for a live interview demo** | An interviewer asks for a live run and we stall | **P4** (rehearsed capture), **P6** (runbook) | JarvisLabs resume (sub-2-min) + **one-command** stack bring-up; default the live demo to the fast-loading fine-tuned **Gemma 4 E4B** (Sarvam-M 24B optional); rehearse and time it in the RUNBOOK |
| R5 | **Benchmark evidence not convincing without a live endpoint** | The standing portfolio has no live model; the evidence must carry it | **P3/P4** (evidence capture), **P6** (packaging) | Capture **two-table benchmark** + MLflow links + Langfuse traces + screenshots + tokens/sec + recorded demo video during the one live GPU run; host them on the always-available static surface so proof is permanent even though the endpoint is on-demand |

---

## 4. Skills proof map

Each in-demand AI/MLOps skill mapped to the exact phase/subsystem where it is demonstrated — so
nothing is left unproven.

| Skill | Where demonstrated | Evidence artifact |
|---|---|---|
| Reproducible infra / env-driven config / secret hygiene | **P0** — `/infra`, `.env.example`, CI | Compose up from clean clone; CI green; no secrets in code |
| HF Hub auth & artifact registry | **P0** (auth), **P4** (push adapter+metrics) | whoami in CI; HF repo with adapter + benchmark |
| Cross-lingual entity resolution | **P1** — Remake-Graph | GS-01/04/05/09/10 passing; typed edges |
| Graph data modelling (typed edges, Work/Version) | **P1** — Postgres schema | Schema + edge-typing tests |
| Multi-source conflict resolution & data provenance | **P1** — pipeline, `sources[]`/confidence, `conflicts` queue | Verification gate enforced; conflict queue |
| Transliteration / fuzzy cross-script matching | **P1** (build), **P2** (eval) | GS-07/GS-11; title-match-under-noise metric |
| LLM information extraction with human-in-the-loop | **P1** — `candidate_edges` + review gate | Precision/recall lift report |
| Data licensing / compliance maturity | **P1**, docs | `docs/LICENSING.md`; API/dump-only access |
| Hybrid retrieval (dense + sparse) | **P2** — RAG Engine (BGE-M3) | Recall@k / MRR |
| Reranking (cross-encoder) | **P2** — bge-reranker-v2-m3 | Rerank ablation in benchmark |
| Vector store selection & ops | **P2** — pgvector/Qdrant | `DECISIONS.md` entry + index |
| Retrieval evaluation | **P2** — `/evals` | Table 1 in `BENCHMARKS.md`; Recall@10 ≥ 0.90 gate |
| LLM eval (RAGAS) | **P3** — eval harness | Faithfulness/relevancy in CI |
| Observability / tracing (Langfuse) | **P3/P5** | Trace links; UI trace view |
| Experiment tracking & registry (MLflow) | **P3/P4** | MLflow run links |
| CI eval-gating | **P3** — GitHub Actions | Merge blocked on metric regression |
| Synthetic data generation (grounded) | **P4** — `/finetune` | Versioned dataset from real records |
| QLoRA / PEFT / TRL fine-tuning | **P4** | Adapter on HF Hub |
| Honest before/after benchmarking | **P3 (base) + P4 (FT)** | Two-column Table 2, kept separate from retrieval |
| GPU cost discipline (on-demand, stop-after) | **P4/P5/P6** | RUNBOOK; "nothing runs 24/7" |
| GPU serving (vLLM) + tokens/sec | **P4 (capture) / P5 (serve)** | Throughput in benchmark; env-driven endpoint |
| API orchestration + guardrails (FastAPI) | **P5** — API layer | Prompt-injection + no-hallucination tests |
| Multi-turn tool-calling & backtracking | **P4 (data) / P5 (serve)** | GS-08 passing end-to-end |
| Graceful degradation as a feature | **P5/P6** | GPU-off path shows evidence, never errors |
| Product UI with citations + trace view | **P6** — `/ui` | Demo video; cited answers |
| Containerization & one-command bring-up | **P6/P0** | `docker-compose`; RUNBOOK |
| Portfolio/runbook packaging | **P6** | Static surface + `PORTFOLIO.md` |
| Embedding model selection & A/B | **P2** — BGE-M3 vs bge-multilingual-gemma2 | DEC-0002; Recall@10 decider |
| Retrieval abstention calibration (NO_MATCH) | **P2/P3** — held-out negatives | NO_MATCH precision/recall; threshold in DECISIONS |
| Data/artifact versioning & lineage | **§6** — all phases | Reproducibility stamp on every benchmark row |
| LLM-as-judge governance | **P3 / §6.4** | Pinned cross-family judge; human-agreement report |
| Indirect prompt-injection defense | **P5 / §6.5** | BIPIA-style eval slice; spotlighting + chunk check |
| Reproducible training container | **P4 / §6.6** | Pinned CUDA/torch/vLLM image; rebuild-from-scratch |
| Cost-aware GPU instance selection | **§6.7 / DEC-0003** | Instance-per-job table; ~$10–25 total cost envelope |

---

## 5. Rough effort estimate (part-time)

Assumes part-time cadence (~8–12 focused hours/week), one subsystem per session, using the
**vertical-slice-first** strategy (§2). The widest ranges are **P1** (human verification of candidate
edges) and **P4** (first-time QLoRA + synthetic-data generation).

| Phase | Scope | Estimate (part-time) |
|---|---|---|
| **P0** | Scaffold, infra, two-tier CI, HF auth, GPU smoke test | 0.5–1 week |
| **P1** | Data pipeline + remake-graph + extraction gate + human verification + seed golden set + tool schema v0 + LICENSING | **3–5 weeks** (largest; the hard problem) |
| **P2** | Hybrid RAG + embedding A/B + abstention calibration + retrieval eval to Recall@10 ≥ 0.90 | 2–3 weeks |
| **P3** | RAGAS + Langfuse + MLflow, judge governance, two-tier CI gate, harness dry-run | 1–2 weeks |
| **P4** | Synthetic data + QLoRA FT + base+after benchmark in one GPU window + publish | 2–3 weeks (GPU time itself is hours) |
| **P5** | FastAPI + vLLM on-demand + backtracking + injection defense + dashboards + degradation | 1.5–2.5 weeks |
| **P6** | UI + containerization + static surface + RUNBOOK | 1–1.5 weeks |
| | **Total** | **~11–18 weeks part-time** |

> Sequencing note: the P2 gate (Recall@10 ≥ 0.90) is a hard checkpoint — if retrieval needs more
> iteration (or the embedding A/B forces a rebuild), P1/P2 absorb the extra time and P4 does not
> start early. The GPU is rented on-demand for each batched model job — P1 candidate-edge extraction,
> P2 embedding/index build + retrieval eval, and the combined P4 base+FT+after capture — and stopped
> after each; the P4 job runs only after the P2 gate is green.

---

## 6. Engineering guardrails & reproducibility (cross-cutting)

These apply to every phase and govern how artifacts, evaluation, and compute are managed.

### 6.1 Artifact & data versioning + reproducibility stamp
- **Everything that affects a metric is a versioned artifact:** graph snapshot, golden set, synthetic
  FT dataset, vector index, prompts, and the adapter. Use DVC or HF Datasets + **dataset cards**;
  the adapter lives on HF Hub (DEC-0001).
- **Every `BENCHMARKS.md` row carries a reproducibility stamp:** `{code SHA, data-snapshot hash,
  model revision/commit, index version, prompt-hash, judge config}`, so each result maps to an exact
  input set.

### 6.2 Two-tier CI
- **Tier 1 — every PR, no GPU, no model calls:** lint, unit/integration, **retrieval** metrics
  (Recall@k/MRR/version-set) and guardrail/no-match computed against cached retrieval artifacts. These
  block merges.
- **Tier 2 — manual/tagged dispatch (during a GPU window):** generation/agent metrics written to
  `BENCHMARKS.md`. Between GPU windows, CI gates on the recorded result, not a live call.

### 6.3 Prompt & schema management
- System prompt, tool schema (v0, P1), extraction prompts, and judge prompts are **hashed in-repo
  artifacts**; the active hash is recorded in each run.

### 6.4 LLM-as-judge governance
- Judge model + version + prompt-hash **pinned**; judge is a **different model family** than the model
  under test (self-preference bias, arXiv 2410.21819); **validated against a human-labelled sample**
  with agreement reported. Applies to GS-08 coherence and RAGAS faithfulness.

### 6.5 Safety / prompt-injection posture
- Treat retrieved Wikipedia/plot text as **untrusted content**. Defenses: spotlighting / delimiting,
  a chunk-level adversarial check, output filtering; evaluated with **BIPIA-style** fixtures injected
  in both query and context (P5). Query-side no-match (GS-02) is necessary but not sufficient.

### 6.6 Reproducible compute
- **All neural-model operations run on the rented on-demand GPU** (embedding, reranking, neural
  transliteration, LLM extraction, fine-tune, serving); the laptop and CI operate on persisted
  artifacts only (§2 compute placement).
- **Pinned training container** (CUDA/torch/transformers/PEFT/TRL/vLLM locked) for the GPU runs.
- Model/data **drift & refresh** (the catalog is frequently updated) is scoped as future ops: an
  incremental re-ingest + re-embed path, named here though out of scope for the portfolio slice.

### 6.7 GPU instance & cost envelope (see `docs/DECISIONS.md` DEC-0003)
- **Primary workhorse (P1 extraction, P2 embedding incl. the 9B A/B, P4 QLoRA FT + benchmark serving,
  P5/P6 demo) → JarvisLabs A100 40 GB (~$0.89/hr).** Ample for 4B QLoRA (~10–12 GB), fast 4B vLLM
  serving, fits the 9B embedder; per-minute billing + pause/resume.
- **Alternatives:** RTX 6000 Ada 48 GB (~$0.99/hr) for more headroom; 24 GB tier (RTX 4090 / A30) as a
  budget floor for 4B FT + serving only; A100/H100 80 GB (or FP8 on 40–48 GB) **only** for the optional
  Sarvam-M 24B showcase.
- **Cost envelope:** whole standing evidence ≈ **$10–25** total GPU spend (P4 the largest at ~4–8 h),
  plus **~$0.25–0.50 per live demo** from a paused instance. Storage-only while paused; volume deleted
  after the HF Hub push.

---

## 7. References

Model, retrieval, and evaluation choices in this roadmap are grounded in the following (accessed
2026-07-01); per-decision sourcing is in `docs/DECISIONS.md`.

- **Gemma 4 (Apache 2.0, native function-call tokens, vLLM):** Hugging Face model card `google/gemma-4-E4B`; HF blog "Welcome Gemma 4"; Google AI Gemma 4 model card; vLLM "Announcing Gemma 4 on vLLM" (2026-04).
- **Qwen3-4B-Instruct-2507 (Apache 2.0, tool-calling, 256K ctx):** Hugging Face `Qwen/Qwen3-4B-Instruct-2507`; Qwen docs (vLLM deployment).
- **Sarvam-M (24B, Apache 2.0, Mistral-Small base, Indic gains):** Sarvam AI "Sarvam-M" technical blog; `huggingface.co/sarvamai/sarvam-m`.
- **Sub-7B tool-calling & post-fine-tune equalization (BFCL v4):** Ertas AI, "On-Device Tool Calling 2026: Qwen3-4B vs Gemma 4 E4B vs Phi-4-Mini."
- **Embedding candidates:** BAAI `bge-m3` and `bge-multilingual-gemma2` (FlagEmbedding); MTEB/MIRACL multilingual results.
- **LLM-as-judge self-preference bias:** arXiv 2410.21819, "Self-Preference Bias in LLM-as-a-Judge."
- **Indirect prompt injection on RAG:** BIPIA (Benchmark for Indirect Prompt Injection Attacks); arXiv 2511.15759, "Securing AI Agents Against Prompt Injection Attacks."
- **GPU instance sizing & pricing:** JarvisLabs (`jarvislabs.ai`, `costbench.com`, `gpuvec.com`, `nodepedia.com`); QLoRA/vLLM VRAM (Unsloth requirements; koishiai 24 GB QLoRA guide; vLLM Mistral-Small-24B docs).

---

## Approval

**APPROVED — 2026-07-01.** This roadmap is the execution baseline. P0 is not started yet (awaiting a
separate go-ahead). When P0 begins, it opens with a short per-phase plan (files, approach, tests,
risks) for review before any code, per `CLAUDE.md`.
