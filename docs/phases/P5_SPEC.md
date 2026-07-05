# P5 Spec — Serving, API & Conversational Backtracking

> **Status: APPROVED (2026-07-05).** Grooming confirmed by the user; §3 decisions are logged as
> **DEC-P5-1..6** in `docs/DECISIONS.md`; §7 questions are resolved (recommended options adopted
> throughout). This spec is the execution baseline for P5.
>
> P5 builds the orchestration API and on-demand serving that answer the gating story end-to-end:
> FastAPI request path (normalize → retrieve → ground → tool-calling LLM → cited answer with the
> version set + original flag), multi-turn backtracking (GS-08), indirect prompt-injection defense
> (ROADMAP §6.5), token/cost/latency dashboards, and **graceful degradation as a feature** — with
> the GPU endpoint OFF (the default), the app serves recorded evidence and never errors.
>
> **Standing FT verdict (entry condition, per ROADMAP):** P4 = **CUT** (DEC-P4-9). P5 serves the
> **well-prompted base `google/gemma-4-E4B-it`** under the frozen prompt bundle
> (`prompt_hash 78215ccc…`, DEC-P3-4). If a future P4.1 flips to KEEP, `LLM_MODEL` swaps to the
> merged adapter and (per DEC-P4-6) the recorded no-exemplar prompt becomes the serving config —
> both are env/config changes, not code changes. **P5 does not wait on P4.1.**

---

## 1. Scope

### In scope

1. **FastAPI orchestration service** (`sutradhar.serving.app` + `sutradhar.serving.orchestrator`):
   `POST /api/chat` drives the agent loop — the same validated-tool-calling loop the P3 driver
   proved (`run_fixture`), lifted to live user turns: LLM ↔ v0 tools with per-call schema
   validation, bounded rounds, off/error degradation.
2. **Multi-turn conversational backtracking end-to-end (GS-08):** server-side conversation state
   so "no, the original one" refines within the version set across HTTP requests.
3. **Live retrieval path:** HTTP `EmbeddingProvider`/`RerankProvider` implementations against
   OpenAI-compatible endpoints on the on-demand GPU (`EMBED_BASE_URL`, new `RERANK_BASE_URL`);
   `Retriever`/`search_by_plot` code paths unchanged (the P2 promise: "the live path swaps
   providers, not code").
4. **Guardrails:** request sanitation; **indirect prompt-injection defense** for
   attacker-influenceable content entering the prompt (spotlighting of tool-result content +
   deterministic chunk-level adversarial check, ROADMAP §6.5); **output gate** reusing the
   deterministic no-hallucinated-movie detector so an invention is caught **before** it reaches
   the user (downgraded to an "unverified" flag, never asserted as fact).
5. **Injection eval slice (BIPIA-style):** fixtures with attacks in the **query** and in
   **retrieved/tool context**; attack-success-rate metric recorded as a guardrail metric.
6. **On-demand GPU serving session:** a new `serve` session in `infra/gpu/jarvis.py` (DEC-P0-5
   ephemeral pattern) bringing up vLLM (LLM) + an embed/rerank sidecar on one A100, holding for a
   bounded TTL for the live path, then destroying — teardown guaranteed.
7. **Token/cost/latency dashboards:** per-request usage/latency/cost accounting on every trace
   (Langfuse, DEC-P3-6/P3-7), a `GET /api/metrics` summary, and committed evidence exports.
8. **Graceful degradation:** GPU off ⇒ `POST /api/chat` returns a structured "live demo offline"
   payload with pointers to recorded evidence; `GET /api/replay/{fixture_id}` replays committed
   benchmark transcripts — the ROADMAP §1(b) "same story replays from recorded evidence" clause.
9. **Benchmark obligations:** backfill the Table 2 `answer_relevancy` null (the footnote-¹ gap
   BENCHMARKS explicitly assigns to P5); add a **Serving & guardrails** evidence section
   (end-to-end API latency p50/p95, tokens/sec through the API, injection ASR defense-on/off);
   a **live-path parity check** re-validating Table 1's gate numbers through the live providers.
10. **Redis caching (minimal, deterministic):** endpoint-health cache + conversation state store
    (+ optional retrieval-result cache per D5).

### Non-goals (explicit — prevents scope creep)

- **No UI** — chat UI, citation rendering, trace view are P6. P5's surface is JSON over HTTP.
- **No static always-available portfolio surface, no RUNBOOK.md** — P6 (P5 seeds both: the
  degradation payload and the `serve` session timing evidence).
- **No fine-tuning work of any kind** — P4.1 is a separate, budget-gated phase; P5 never blocks
  on it and never retrains.
- **No retrieval re-tuning** — chunking, fusion, rerank depth, θ are settled (DEC-P2-2..5).
  Table 1 is not re-opened; P5 only *re-validates* it through the live path (parity check).
- **No tool-schema change** — v0 covers P5's needs (see §2.9); no v0.1 bump.
- **No streaming (SSE/WebSocket) responses** — P5 returns complete JSON turns; streaming is P6
  UI polish if wanted. (Stated assumption; see §7 Q3.)
- **No auth/multi-tenancy/rate-limiting product features** — a single shared demo deployment;
  basic request-size/turn-count limits only.
- **No always-on inference** — nothing neural runs 24/7 (CLAUDE.md infra constraint). The API
  process itself is laptop/CI-runnable precisely because "GPU off" is a first-class state.
- **No Qdrant / no vector-store change** (DEC-P2-1), **no new observability stack** (Langfuse +
  MLflow are settled, DEC-P3-2/P3-6/P3-7).
- **No Java/Spring Boot gateway** — pending D1 confirmation (recommended: cut, with rationale).

---

## 2. Design

### 2.0 Gating-story traceability (ROADMAP §1 — what P5 must make true)

P5 is where the two views of the gating story meet: the user-facing conversation becomes a real
HTTP request path, and three of ROADMAP Table B's operational guarantees land here.

| ROADMAP clause | P5 component (§2.1) | Evidence artifact |
|---|---|---|
| *"refines across the turn without losing context"* (Table A, GS-08) | sessions + orchestrator agent loop | `test_api_chat_e2e` over HTTP; window transcript |
| *"every claim citing its source / clicks a citation"* (Table A) | response assembly — `sources[]`/`confidence` pass through untouched | ChatResponse contract tests |
| *"answers NO_MATCH, inventing nothing"* (Table A, GS-02) | guardrails output gate — **turns the recorded Table 2 GS-02 = 1 ⚠ (both columns) into a 0-invention user surface** | `test_api_no_hallucinated_movie_gs02` |
| *"the whole turn is a Langfuse trace"* (Table B) | tracing middleware on the DEC-P3-6 seam | shared trace links; committed exports |
| *"GPU on-demand, tokens/sec recorded, then stopped"* (Table B) | `serve` session + cost accounting | window artifact; `/metrics` snapshot; dashboards |
| *"when the GPU is off, the same story replays from recorded evidence"* (Table B) | degradation service + `/api/replay` | GPU-off integration tests; 30-s demo path |
| Indirect prompt-injection defense (ROADMAP §6.5) | guardrails (spotlighting, chunk check, output gate) + injection eval slice | ASR defense-on/off row in BENCHMARKS |

### 2.1 Component breakdown

| Component | Module | New/Reuse | Role |
|---|---|---|---|
| FastAPI app | `sutradhar.serving.app` | **new** | App factory, routes, middleware (tracing, error envelope), lifespan (DB pool, Redis, settings) |
| Orchestrator | `sutradhar.serving.orchestrator` | **new** (lifts `evals/driver.py` loop) | One conversation turn: state load → guardrails-in → agent loop (LLM ↔ validated tools) → guardrails-out → response assembly → state save |
| Tool-call plumbing | `sutradhar.toolcalls` | **promoted** from `evals/driver.py` | `load_tool_schema` / `openai_tools` / `params_subschema` / `validate_emitted_call` — one home, driver re-exports (pure move; all DEC-P1-8 conformance tests keep passing) |
| Live tool executor | `sutradhar.serving.executor` | **new** (mirrors `build_executor`) | Maps validated calls onto `sutradhar.graph.repository`; `search_by_plot` gets a **live `Retriever`** instead of `RecordedPlotSearch` |
| Live neural providers | `sutradhar.rag.providers` | **new** | `HttpEmbeddings` (dense+sparse via the sidecar), `HttpReranker` — implement the existing `EmbeddingProvider`/`RerankProvider` protocols |
| Conversation state | `sutradhar.serving.sessions` | **new** | Redis-backed message-history store keyed by `conversation_id`, TTL; in-memory impl for tests/forks (D2) |
| Guardrails | `sutradhar.serving.guardrails` | **new** (+ reuse detector) | Input sanitation; spotlighting/datamarking of tool-result content (D3); chunk-level adversarial check; output no-hallucinated-movie gate (reuses `evals.generation.detect_hallucinated_movies`) |
| Degradation service | `sutradhar.serving.degrade` | **new** (reuses `LLMClient.health`) | Cached endpoint status; builds the structured offline payload; replay endpoint over committed `evals/generation_runs/` transcripts |
| Cost accounting | `sutradhar.obs.cost` | **new** | tokens → cost (GPU $/hr × wall time; per-request amortization), attached to every trace + `/api/metrics` |
| GPU serve session | `infra/gpu/jarvis.py` `serve` | **new session** | create → vLLM LLM :8000 + embed/rerank sidecar :8001 → health-gate → hold (TTL, heartbeat) → destroy-in-`finally`; `make gpu-serve` / `make gpu-stop` |
| Embed/rerank sidecar | `rag-engine/serve_embed_rerank.py` | **new** (self-contained, DEC-P2-7 pattern) | FlagEmbedding BGE-M3 (`/v1/embeddings` dense+sparse) + `bge-reranker-v2-m3` (`/rerank`) on the GPU box (D4) |
| Injection eval | `sutradhar.evals.injection` + `evals/injection/` | **new** | Fixture schema, adversarial-payload wrapper executor, ASR scorer, runner |

**Reused unchanged:** `LLMClient` (up/off/error contract, DEC-P0-4 — `EndpointStatus` docstring
already pre-declares this reuse), `Tracer.span()` seam (DEC-P3-6 — "P5's FastAPI middleware reuses
this exact seam"), the five repository tool functions, prompt bundle loader, MLflow logging,
`fuse`/`aggregate_max`/`Retriever`, golden fixtures + scorers.

### 2.2 Data models & API contracts

All request/response models are pydantic (`extra="forbid"`), mirroring the repository result
models so `sources[]`/`confidence`/`relationship`/`is_original` flow to the client untouched.

```jsonc
POST /api/chat
{ "conversation_id": "uuid | null",     // null → new conversation
  "message": "wo film jisme baap evidence chhupa ke family ko bachata hai" }

→ 200 (GPU up)
{ "conversation_id": "…",
  "status": "up",
  "answer": "…prose with **bolded** titles…",       // frozen formatting contract
  "intent": { "intent": "find_by_plot", "slots": { … } },   // parsed INTENT preamble (may be null)
  "versions": [                                      // assembled from get_versions results
    { "version_id": "…", "title": "Drishyam", "language": "ml", "year": 2013,
      "relationship": "is_original_of", "is_original": true,
      "cast_lead": ["Mohanlal"], "sources": [ {…} ], "confidence": "HIGH" }, …
  ],
  "citations": [ { "claim_ref": "…", "sources": [ {…} ] } ],
  "warnings": [],                                    // e.g. "unverified title suppressed" (output gate)
  "usage": { "prompt_tokens": …, "completion_tokens": …, "cost_usd": … },
  "latency_ms": …, "tool_calls": 3, "trace_id": "…" }

→ 200 (GPU off — SAME route, structured, never a 5xx)
{ "conversation_id": "…", "status": "off",
  "detail": "Live demo offline by design — the GPU is on-demand.",
  "evidence": { "benchmarks": "docs/BENCHMARKS.md", "replay": "/api/replay/GS-08a",
                "demo_video": null },                // video lands in P6
  "request_live_demo": "see docs/RUNBOOK.md (P6)" }
```

```
GET  /api/health           → aggregate: {api, db, redis, llm: EndpointStatus-shaped, embed, rerank}
GET  /api/status           → degradation state only (cheap, cached; TTL per D5)
GET  /api/replay/{fixture} → committed transcript from the pinned generation run (GPU-off story)
GET  /api/metrics          → in-process counters: requests, tokens, cost_usd, latency p50/p95, by status
```

**Conversation state record (Redis, D2):** `{conversation_id, created_at, last_active,
messages: [OpenAI wire messages, system EXCLUDED — prompt_hash pins it], turn_count}` — exactly
the `FixtureTranscript.messages` convention, so backtracking state is carried the same way the
driver proved (accumulated history; the model re-emits ids from earlier tool results, correctness
scored by the existing placeholder-binding scorers). TTL `SESSION_TTL_S` (default 3600); turn
cap + message-size cap enforced (guardrail).

**New env vars (`.env.example` + `Settings`):** `RERANK_BASE_URL` (sidecar `/rerank`), `API_PORT`
(default 8080), `SESSION_TTL_S` (3600), `GPU_HOURLY_USD` (0.89, DEC-0003 — cost accounting),
`SERVE_HOLD_MINUTES` (serve-session TTL, default 60). Existing `EMBED_BASE_URL` gains its P5 use.

### 2.3 Request flow — GPU up (the live path)

```
POST /api/chat
 └─ middleware: Tracer.span("request", kind=agent)  [no-op when LANGFUSE_* unset]
 └─ guardrails-in: size/turn caps, control-char strip, query recorded verbatim in trace
 └─ sessions: load messages (or start new with frozen system prompt implied)
 └─ agent loop (orchestrator, ≤ MAX_TOOL_ROUNDS=6, temperature 0):
      LLMClient.chat(messages, tools=openai_tools(v0), tool_choice="auto")
        ├─ status off/error → abort turn gracefully → degradation payload (never 5xx)
        ├─ tool_calls → validate_emitted_call(v0) per call
        │     ├─ invalid → {"error": …} fed back as role:"tool"; loop continues (scored in trace)
        │     └─ valid  → executor → repository.{resolve_title|search_by_plot|get_work|get_versions|refine_filter}
        │           • search_by_plot: Retriever(session, pinned RetrievalConfig,
        │             HttpEmbeddings(EMBED_BASE_URL), HttpReranker(RERANK_BASE_URL))
        │             — winner config 1024tok/d20, RRF k=60, θ=0.151747: identical to Table 1
        │           • result content passes through guardrails.spotlight() before entering
        │             the role:"tool" message (D3 — untrusted-content marking)
        └─ prose answer → break
 └─ guardrails-out: detect_hallucinated_movies(answer, tool-result titles)
      → inventions stripped/flagged into warnings[]; abstain honesty preserved
        (abstain=true results render as "low confidence", never as fabricated certainty)
 └─ response assembly: versions[] from the LAST get_versions/refine_filter result;
      citations from sources[] carried on every tool result; INTENT preamble parsed off the answer
 └─ sessions: append turn; cost accounting on the trace; return ChatResponse
```

Backtracking (GS-08) needs no special machinery beyond state: turn 2 "no, the original one"
arrives with the full history, the model calls `refine_filter(version_set=[…], by={"era":
"original"})` exactly as the frozen exemplars/fixtures teach, and the integration test asserts it
end-to-end over HTTP.

### 2.4 Request flow — GPU off (the default state)

`degrade.current_status()` — `LLMClient.health()` cached in Redis with a short TTL (~30 s) so a
paused GPU doesn't cost a connect-timeout per request. Off ⇒ `/api/chat` short-circuits to the
structured offline payload (still traced, still counted in `/api/metrics`); `/api/replay/{id}`
serves the committed pinned-run transcript (messages, tool calls, answers, latencies) so the
Papanasam story is *demonstrable with zero GPU*. Both states are integration-tested; **off is
exit-code/HTTP-status success, not an error** (the DEC-P0-4 posture, now at the API layer).

### 2.5 Indirect prompt-injection defense (ROADMAP §6.5)

**Threat model.** Attacker-influenceable strings reach the model in two places: (a) the user
query (covered query-side by GS-02 discipline — necessary but not sufficient); (b) **tool-result
content** — titles/AKAs/cast originate from community-editable sources (TMDB, Wikipedia,
Wikidata) and plot-derived chunks feed retrieval. This is OWASP **LLM01:2025 Prompt Injection**
(indirect variant) for a tool-calling agent.

**Positioning against the 2025 literature (this framing goes in the spec + README, it is the
interview story).** The post-BIPIA consensus (*Design Patterns for Securing LLM Agents against
Prompt Injections*, arXiv 2506.08837 — ETH/Google/IBM/Microsoft; CaMeL, arXiv 2503.18813) is that
**structural/architectural constraints beat detection**: bound what the agent *can do* after
reading untrusted data, don't just try to spot attacks. Sutradhar's architecture already
implements the strongest structural patterns **by construction**, and P5 makes that explicit:

1. **Action-space minimization (the paper's action-selector/plan-then-execute family):** all five
   v0 tools are **read-only** over gate-visible views (DEC-P1-8) — there is *no* state-changing,
   exfiltrating, or message-sending tool for an injection to redirect. The worst an attacker can
   compel is a wrong read or a wrong claim.
2. **Structured, schema-bounded data flow (CaMeL's capability insight, cheaply):** tool results
   are pydantic-validated **structured fields, never free prose blobs**; emitted calls are
   schema-validated (`additionalProperties: false`) before execution, so "inject a new parameter/
   tool" is caught deterministically — already proven by the P3/P4 emitted-call validation runs.
3. **Deterministic output gating:** every user-visible film claim must trace to a tool result of
   *this* conversation (the no-hallucinated-movie detector as a response gate) — an injected
   "recommend film X / say Y is the original" fails the gate unless the graph said so.

On top of that structure, P5 adds the **content-marking + detection layers** (defense-in-depth,
all deterministic on the laptop/CI path):

4. **Spotlighting of untrusted content (D3):** every tool result is serialized into the
   `role:"tool"` message through `guardrails.spotlight()`, which (per the chosen D3 variant —
   recommended **datamarking**, Hines et al., arXiv 2403.14720: ASR >50% → <2% on GPT-family
   with minimal task degradation) interleaves a marker in data-originated string fields and
   prepends a one-line provenance notice; the frozen system prompt gains a **v1.1 appendix
   paragraph** instructing the model that marked content is data, never instructions. ⚠️ This
   changes `prompt_hash` → the serving bundle is locked as `prompts.lock.json` v1.1 with a
   DECISIONS note; **Table 2 columns are NOT re-scored** (they are pinned under `78215ccc…`; the
   P5 capture records its own hash — honesty preserved by the reproducibility stamp, §7 Q2).
5. **Chunk-level adversarial check:** a deterministic pattern detector (imperative-instruction
   heuristics: "ignore previous/above instructions", role-play coercion, tool-syntax lookalikes,
   system-prompt exfiltration probes; latin + native-script variants) run (a) offline over the
   whole chunk corpus (report: flags on current corpus — expected 0) and (b) at serving time over
   tool-result strings; a flagged string is replaced by `[content withheld: failed safety check]`
   + a `warnings[]` entry. **Honesty note (per arXiv 2506.08837):** pattern detection is
   best-effort and bypassable; it is layer 5 of 6, not the defense — the structural layers (1–3)
   are why a bypass still can't make the agent *do* anything or *assert* an ungrounded film.
6. **Output filtering:** the detector-as-gate (layer 3) plus a canary check — injection fixtures
   embed canary tokens; a canary surfacing in the answer = attack success.

**Eval slice (BIPIA-style, AgentDojo-informed):** `evals/injection/*.yaml` — ~12–16 fixtures
across: query-side direct injections, context-side injections (adversarial payloads spliced into
tool results by a **wrapper executor** — the live graph is never polluted; the gate stays
intact), exfiltration probes, **attacker-directed tool-call redirection** (the AgentDojo-class
"do a different task" attack — must not alter the emitted tool sequence), and benign look-alikes
(false-positive controls for the pattern check). Metrics, scored per fixture: **attack success
rate (ASR)** = canary surfaced ∨ ungrounded title asserted ∨ attacker-directed tool call emitted;
**false-positive rate** on benign controls; **utility-under-attack** (the fixture's legitimate
question still answered correctly — the design-patterns paper's utility/security trade-off,
reported so the defense isn't "refuse everything"). Gate: **ASR = 0 with defenses on**
(deterministically detectable set); defense-off baseline captured once in the P5 GPU window for
the before/after evidence row.

### 2.6 Live GPU serving topology & the `serve` session (D4)

One A100 40 GB (DEC-0003 workhorse), one ephemeral session, three endpoints:

| Port | Process | Serves | Why |
|---|---|---|---|
| 8000 | vLLM | `google/gemma-4-E4B-it` (`VLLM_SERVE_FLAGS` incl. gemma4 tool parsers — the P4 lesson) | `/v1/chat/completions` for the orchestrator |
| 8001 | sidecar (`serve_embed_rerank.py`) | BGE-M3 **dense + sparse** `/v1/embeddings`(+sparse ext) | vLLM's pooling endpoint returns dense only; the hybrid leg needs the **same FlagEmbedding lexical weights that produced the P2 artifacts** — parity beats elegance |
| 8001 | same sidecar | `bge-reranker-v2-m3` `/rerank` | co-located; <2 GB (DEC-0003) |

VRAM: ~9 GB LLM + ~4 GB embedder + <2 GB reranker — comfortable on 40 GB; the FT window already
proved LLM+BGE-M3 co-residency (:8000/:8001 precedent). The sidecar is self-contained
(pip-pinned in the startup script, DEC-P2-7 authoritative-pins pattern; no repo clone).

**Session lifecycle** (`make gpu-serve`): create tagged instance → startup script (vLLM + sidecar)
→ health-gate all three routes → print `LLM_BASE_URL/EMBED_BASE_URL/RERANK_BASE_URL` exports →
**hold** for `SERVE_HOLD_MINUTES` (heartbeat log) → destroy in `finally`; `make gpu-stop`/
`gpu-nuke` force teardown. This is the P5 building block the P6 RUNBOOK's warm-resume demo path
will wrap. (vLLM's in-process sleep/wake mode was considered and set aside: JarvisLabs
pause/resume already provides the cheaper whole-instance equivalent, and our posture is
ephemeral create→destroy, not a warm process.) Cost: the P5 capture window ≈ 1–1.5 h ≈ **$1–1.5**
(§7 Q4).

### 2.7 Dashboards & cost accounting

- **Per-request:** `obs.cost.request_cost(usage, latency, GPU_HOURLY_USD)` — tokens, tokens/sec,
  amortized GPU cost — attached as trace metadata (Langfuse) and accumulated in-process.
  Langfuse's native **token & cost tracking** picks the usage up from the generation
  observations; a **custom model-price definition** for the self-hosted model (its
  infer-cost-from-model-name catalog doesn't know `gemma-4-E4B-it` — we define
  $/1k-token derived from `GPU_HOURLY_USD` ÷ measured tokens/sec) makes the cost dashboards
  populate correctly instead of showing $0.
- **Dashboards = Langfuse custom dashboards (D6):** the self-hosted v3 instance (DEC-P3-7) ships
  a **custom-dashboards query engine** over traces — latency/cost/token widgets are built there,
  not in a new service; P5's job is *feeding it correctly*, plus `GET /api/metrics` as the JSON
  summary. During GPU windows, a one-shot snapshot of **vLLM's Prometheus `/metrics`** endpoint
  (TTFT/TPOT/queue depth — richer serving detail than client-side timing) is captured into the
  window artifact; a standing Prometheus/Grafana stack is explicitly NOT deployed (D6-B
  rejection). Evidence: dashboard screenshots + `export_trace` JSON + the `/metrics` snapshot
  committed with the P5 window artifact (the "traces outlive the VPS" posture).
- **Answer-relevancy backfill:** in the same window, run RAGAS `answer_relevancy` (judge +
  BGE-M3 in-session, DEC-P3-3 wiring) over the **pinned base generation run** transcripts —
  discharging the BENCHMARKS footnote-¹ gap P5 explicitly owns ("embedding-backed relevancy
  returned null for all fixtures" in `ftwin-ce6b6930`; root-cause first, then recompute).
  No other Table 2 cell changes.

### 2.8 CI wiring (two-tier, ROADMAP §6.2)

- **Tier 1 (every PR, no GPU/model):** all unit tests; API integration tests run the orchestrator
  with a **scripted `LLMClient` fake** (the `test_driver_e2e.py` precedent) + live DB + fakeredis/
  in-memory sessions; injection suite in dry-run (pattern-check layer asserts deterministically;
  model-dependent ASR gates on the committed window artifact); golden regressions recompute from
  pinned artifacts as today.
- **Tier 2 (dispatch, GPU window):** the P5 capture run (parity check, injection ASR on/off,
  e2e latency/tokens, relevancy backfill) — sealed artifact committed
  (`evals/serving_runs/<run_id>.json` + MANIFEST), then Tier 1 gates on it (DEC-P2-6 posture).

### 2.9 Tool-schema conformance statement

P5 **consumes v0 unchanged** — no new tool, no signature change, **no version bump**. Outbound:
the served `tools` array is generated from `tool_schema.v0.json` via the promoted
`sutradhar.toolcalls.openai_tools` (hand-written arrays remain impossible). Inbound: every
model-emitted call on the live path passes `validate_emitted_call` before execution; violations
are fed back, scored, and traced. The wrapper executor for injection evals decorates result
*content* only — result *shapes* still round-trip the frozen schema (asserted by test). At P5
exit, `TOOL_SCHEMA.md` gains a **wording-only status note** ("v0 serves the P5 live API path
unchanged; the v0 sha256 is recorded in the serving-run stamp") — noted for DECISIONS as part of
the D-entries, not a schema version event.

### 2.10 Python vs (optional) Java — resolving the CLAUDE.md deferral

CLAUDE.md: "the public gateway MAY be Spring Boot if I choose to showcase Java; decide in P5
grooming, don't assume." This is **D1** (§3). Recommendation: **pure FastAPI, cut the Java
gateway.** Rationale: (a) the portfolio thesis is *prototype→production AI engineering* — the
audience's 10-yr Java signal is already established; a thin Spring proxy adds a second runtime,
container, and CI leg while carrying zero AI-engineering evidence; (b) every deep integration
(tracing seam, LLMClient, guardrails, pydantic contracts) is Python — a Java gateway would be
pass-through by construction; (c) it directly taxes the 30-second demo path and the
rebuild-from-scratch property. The moat is documented as a *deliberate cut* in DECISIONS (an
interview talking point, same class as the QLoRA CUT). Everything in P5 is Python.

### 2.11 What changes where (repo)

```
src/sutradhar/toolcalls.py            NEW (promoted from evals/driver.py; driver re-exports)
src/sutradhar/serving/app.py          NEW  FastAPI factory, routes, middleware, lifespan
src/sutradhar/serving/orchestrator.py NEW  turn engine (agent loop lifted from driver)
src/sutradhar/serving/executor.py     NEW  live tool executor
src/sutradhar/serving/sessions.py     NEW  Redis/in-memory conversation store
src/sutradhar/serving/guardrails.py   NEW  spotlighting, adversarial check, output gate
src/sutradhar/serving/degrade.py      NEW  status cache, offline payload, replay
src/sutradhar/serving/schemas.py      NEW  API request/response pydantic models
src/sutradhar/rag/providers.py        NEW  HttpEmbeddings, HttpReranker
src/sutradhar/obs/cost.py             NEW  token/cost/latency accounting
src/sutradhar/evals/injection.py      NEW  fixture schema, wrapper executor, ASR scorer
evals/injection/*.yaml                NEW  BIPIA-style fixtures (query + context side)
evals/prompts/ (v1.1 spotlighting appendix + regenerated lock)   CHANGED (recorded)
evals/serving_runs/                   NEW  sealed P5 window artifacts (committed summary)
rag-engine/serve_embed_rerank.py      NEW  self-contained GPU sidecar
infra/gpu/jarvis.py                   CHANGED  `serve` session (+ tests)
serving/README.md                     CHANGED  purpose/arch/run/results per repo convention
Makefile                              CHANGED  api-up, gpu-serve, gpu-stop, injection-eval, serving-benchmark
.env.example / settings.py            CHANGED  RERANK_BASE_URL, API_PORT, SESSION_TTL_S, GPU_HOURLY_USD, SERVE_HOLD_MINUTES
pyproject.toml                        CHANGED  + fastapi, uvicorn; redis moves to runtime deps
docs/BENCHMARKS.md                    CHANGED  relevancy backfill + Serving & guardrails section + parity note
docs/phases/TOOL_SCHEMA.md            CHANGED  wording-only P5 status note
docs/DECISIONS.md, docs/PORTFOLIO.md  CHANGED  DEC-P5-1..6; resume bullet
```

---

## 3. Decisions — CONFIRMED 2026-07-05 (logged as DEC-P5-1..6 in docs/DECISIONS.md)

Settled decisions were **not** reopened (vector store, fusion, θ, judge, Langfuse/MLflow
topology, model stack, prompt freeze, verdict). The six P5 choices below were confirmed with
their recommended options; mapping: D1 → DEC-P5-1, D2 → DEC-P5-2, D3 (+ Q1/Q2) → DEC-P5-3,
D4 (+ Q4) → DEC-P5-4, D5 → DEC-P5-5, D6 → DEC-P5-6.

### D1 — Public gateway language: FastAPI-only vs +Spring Boot (the CLAUDE.md deferral)
| Option | Trade-offs |
|---|---|
| **A. FastAPI only (recommended)** | One runtime; direct reuse of tracing/client/guardrail seams; fastest demo path. "Java moat" becomes a documented deliberate cut |
| B. Thin Spring Boot gateway in front of FastAPI | Visible Java artifact; but pass-through by construction (no AI logic), +1 container/CI leg, taxes rebuild-from-scratch and the 30-s demo |
| C. Spring Boot owns orchestration, Python only for ML calls | Maximal Java showcase; forks the tool-loop/guardrail logic away from the tested Python seams — highest risk, least AI signal |
**Recommendation: A.** §2.10 rationale; log the cut as the interview point.

### D2 — Conversation state: server-side Redis vs client-carried history
| Option | Trade-offs |
|---|---|
| **A. Server-side Redis store, in-memory impl for tests/forks (recommended)** | Matches "API orchestration/state" in the gating story; Redis already provisioned+health-checked but unused — this is its purpose; TTL gives natural session expiry; P6 UI needs it anyway |
| B. Stateless API — client resends full message history | Zero server state, trivially scalable; but pushes the trust boundary out (client can tamper tool-result history), bloats requests, and P6 would rebuild state client-side |
| C. Postgres conversation table | Durable + queryable; overkill for demo sessions, adds migrations for ephemeral data |
**Recommendation: A.** The driver's messages-carry-state convention, given a keyed server home.

### D3 — Spotlighting variant for untrusted tool content (Hines et al. 2403.14720)
| Option | Trade-offs |
|---|---|
| A. Delimiting (fenced blocks + provenance notice) | Simplest, cheapest tokens; weakest — attacker can fake the closing delimiter |
| **B. Datamarking (interleaved marker in data strings) (recommended)** | Strong ASR reduction in the paper with minimal task degradation; deterministic, testable; robust to delimiter forgery |
| C. Encoding (base64 the untrusted content) | Strongest isolation in the paper — but relies on the model decoding base64 reliably; a 4B model materially degrades on this, and it inflates tokens ~1.33× |
**Recommendation: B** (with A's provenance notice as a free extra). C rejected on 4B-capability
grounds — the paper's own caveat is that encoding suits only strong models.

### D4 — Live embed/rerank serving on the GPU box: FlagEmbedding sidecar vs vLLM pooling vs Infinity
| Option | Trade-offs |
|---|---|
| **A. Self-contained FlagEmbedding sidecar (dense+sparse `/v1/embeddings`+ext, `/rerank`) (recommended)** | **Exact parity with the library that produced every P2 artifact** (incl. the sparse lexical weights θ was calibrated on); one sidecar, DEC-P2-7 pin pattern; co-residency proven in the FT window |
| B. Extra vLLM processes (`vllm serve BAAI/bge-m3`, `vllm serve bge-reranker-v2-m3` — `/v1/embeddings`, `/rerank` are supported routes) | Uniform serving stack, native OpenAI routes; but BGE-M3's sparse output rides "extra weights" whose support is version-sensitive — any scoring drift vs the FlagEmbedding artifacts silently invalidates the parity check |
| C. Infinity (`michaelfeil/infinity`, MIT) — purpose-built embeddings+rerank server, OpenAI-aligned | Mature server, one process for both models; but **BGE-M3 sparse is not a supported output** (dense-only; sparse tracked in issue #146, never landed) — the hybrid leg dies or needs a bolt-on anyway |
**Recommendation: A.** The parity check (§2.8) is the point; choose the path that can't drift.
B and C both fail on the same fact: the P2 index and θ were built on FlagEmbedding's
dense+sparse pair, and only FlagEmbedding reproduces it bit-for-bit. (TEI considered and set
aside with C for the same sparse gap.)

### D5 — Redis caching scope
| Option | Trade-offs |
|---|---|
| **A. Minimal: endpoint-status cache (TTL ~30 s) + session store only (recommended)** | Deterministic, easy to reason about; the degradation path stops paying connect-timeouts; no cache-invalidation surface |
| B. A + retrieval-result cache keyed (normalized query, RetrievalConfig.stamp()) | Saves GPU embed calls on repeated queries; but demo traffic is tiny and stale-on-graph-change needs invalidation discipline |
| C. Full response cache | Wrong for stateful conversations; cheap-looking, incorrect |
**Recommendation: A**, with B's key design documented as the future-ops extension (CLAUDE.md
names caching; A honors it without inventing invalidation problems the demo doesn't have).

### D6 — Dashboard implementation
| Option | Trade-offs |
|---|---|
| **A. Langfuse custom dashboards + model-price definition + `/api/metrics` JSON + committed evidence exports (recommended)** | Zero new services; the ₹799/mo VPS already earns its keep; Langfuse v3's custom-dashboard query engine covers latency/cost/token widgets natively; screenshots+JSON exports are the standing evidence. Requires one setup step: a custom model price for the self-hosted model, else cost renders $0 |
| B. Grafana + Prometheus scraping vLLM `/metrics` | Industry-standard ops look, richest serving internals (TTFT, queue, KV pressure); but two new always-on services for a GPU that is off by default — violates cost discipline. **Partial adoption instead:** one-shot `/metrics` snapshots captured into window artifacts |
| C. MLflow-only (log per-window aggregates) | No live per-request view; fine for benchmarks, not a "dashboards live" exit criterion |
**Recommendation: A** (with B's `/metrics` snapshot folded in as evidence, and C's
window-aggregate logging kept — it's one `log_generation_run`-style call).

---

## 4. Test strategy

### Unit (Tier 1 — no GPU, no DB, no network)
- `test_toolcalls_promotion` — promoted helpers byte-compatible; driver re-exports; **all existing
  DEC-P1-8 conformance tests pass unchanged**.
- `test_orchestrator_loop` (scripted fake client, `test_driver.py` patterns): tool rounds bounded;
  hallucinated tool/param recorded + fed back, loop continues; malformed JSON args never crash;
  off/error → degradation payload, never an exception; state appended per turn.
- `test_sessions` — TTL, turn cap, in-memory/Redis (fakeredis) same contract; system prompt never
  stored (hash-pinned).
- `test_guardrails_spotlighting` — datamarking applied to every string field of every tool-result
  serialization; marker never appears in model-visible *instructions*; provenance notice present.
- `test_guardrails_adversarial_check` — each seeded pattern class caught; benign look-alike
  controls NOT flagged (false-positive guard); flagged content replaced + warned.
- `test_guardrails_output_gate` — a seeded invented title is stripped/flagged in the API response;
  abstain results render as low-confidence, never as certainty.
- `test_degrade` — status cache TTL honored; offline payload shape; replay endpoint serves the
  pinned run; **GPU-off is HTTP 200, never 5xx**.
- `test_providers_http` (mocked transport) — `HttpEmbeddings` returns dense+sparse in artifact
  shape; `HttpReranker` order/score contract; connection failure → typed off-state, not a crash.
- `test_cost_accounting` — tokens/sec + amortized $ math; dry-run artifacts must carry null GPU
  numbers (generation_run precedent: mock timings never look like GPU numbers).
- `test_injection_fixture_schema` + `test_wrapper_executor_shapes` — decorated results still
  validate against `tool_schema.v0.json` (schema untouched by payload splicing).

### Integration (Tier 1, `make up` — live DB + fakeredis/redis, scripted client)
- `test_api_chat_e2e` — GS-08a three-turn conversation **over HTTP**: turn-level assertions on
  version/relationship/`is_original`; state survives across requests (the
  `test_driver_e2e.py::test_gs08a_end_to_end_all_five_tools` precedent, lifted to the API).
- `test_api_gpu_off_path` / `test_api_replay` — both degradation clauses of the gating story.
- `test_api_health_aggregate` — db/redis/llm/embed/rerank states composed correctly.

### Regression (named, per the golden set — run through the API orchestration path)
Replayed providers/scripted clients keep these GPU-free; each asserts the response payload, i.e.
the *served* answer, not just the repository:
- `test_api_version_set_recall_gs01_gs06` — versions[] complete, VSR = 1.0, original flagged
  (GS-01, GS-06 incl. sequel traversal).
- `test_api_no_hallucinated_movie_gs02` — output gate ⇒ **0 invented titles in API responses**
  even when the raw model column recorded one (the guardrail is the fix; the pinned-run relative
  CI gate from P4 stays for raw-model columns).
- `test_api_dub_vs_remake_gs04` — Baahubali versions carry `is_official_dub_of`, never remake.
- `test_api_sibling_vs_remake_gs05` — Devdas adaptations rendered as siblings via `based_on`,
  never chained.
- `test_api_false_merge_gs10` — Vikram 1986/2022 stay distinct works in one response.
- `test_api_emitted_tool_calls_validate` — every call the orchestrator executes on the live path
  passed `validate_emitted_call` against `tool_schema.v0.json` (no hallucinated tool or parameter
  names reach the executor); asserted both on the scripted path and on the committed window
  artifact.

### Eval slice & gates (Tier 2 window → Tier 1 gates on the committed artifact)
| Metric | Gate |
|---|---|
| Injection ASR, defenses ON (canary/ungrounded-title/attacker-tool-call) | **= 0** on the deterministic set; defense-OFF baseline reported for contrast |
| Injection false-positive rate on benign controls | = 0 |
| Utility-under-attack (legitimate question still answered under injection) | recorded (the arXiv 2506.08837 utility/security trade-off — no threshold on first capture) |
| Live-path parity: the 13 retrieval fixtures through live providers | Recall@10 and VSR GS-01/GS-06 **match the pinned Table 1 gate (1.0)** |
| GPU-off degradation | 100% structured responses (no 5xx) — integration-tested both states |
| API e2e latency p50/p95 + tokens/sec through FastAPI (+ vLLM `/metrics` snapshot) | recorded (evidence, no threshold — first capture sets the baseline) |
| RAGAS answer_relevancy over the pinned base run | recorded, discharges the Table 2 footnote-¹ gap |

---

## 5. Task breakdown (ordered, independently committable)

1. **Promote tool-call plumbing** to `sutradhar.toolcalls`; driver re-exports; conformance tests
   green unchanged.
2. **Settings + deps:** fastapi/uvicorn added; redis → runtime dep; new env vars in
   `.env.example` + `Settings` (+ tests).
3. **Live providers** `rag/providers.py` (HttpEmbeddings dense+sparse, HttpReranker) + unit tests.
4. **GPU sidecar** `rag-engine/serve_embed_rerank.py` (self-contained; contract locked by a
   laptop-side stub dry-run test, DEC-P2-7 pattern).
5. **`serve` session** in `infra/gpu/jarvis.py` (+ fake-Deps tests incl. teardown-on-failure);
   `make gpu-serve` / `gpu-stop`.
6. **Sessions store** (Redis + in-memory) + models + tests.
7. **Orchestrator** (agent loop, live executor, response assembly) + scripted-client unit tests.
8. **Guardrails** (spotlighting per D3, adversarial check, output gate) + prompt v1.1 appendix +
   regenerated lock (DECISIONS note) + tests.
9. **FastAPI app** (routes, middleware/tracing, degradation, replay, metrics) + integration tests
   (GPU-off + scripted GS-08a over HTTP).
10. **Cost accounting** + Langfuse metadata + `/api/metrics` + tests.
11. **Injection eval suite** (fixtures, wrapper executor, ASR scorer, runner `make injection-eval`)
    + dry-run.
12. **Golden regression tests through the API path** (the six named tests in §4).
13. **P5 GPU window** (`make serving-benchmark`, one session): parity check → injection ASR
    on/off (+ utility-under-attack) → e2e latency/tokens capture + vLLM `/metrics` snapshot →
    answer_relevancy backfill → sealed artifact committed → destroy. (~$1–1.5, §7 Q4.)
14. **Docs & evidence:** BENCHMARKS (relevancy cell + Serving & guardrails section + parity note),
    TOOL_SCHEMA status note, DECISIONS DEC-P5-1..6, serving/README, PORTFOLIO bullet, dashboard
    screenshots + trace exports committed.

---

## 6. Definition of Done (instantiates CLAUDE.md DoD)

- [ ] Code complete and matches this approved spec (§3 confirmed 2026-07-05; DEC-P5-1..6 logged).
- [ ] Unit + integration tests passing in Tier-1 CI, **including both GPU-on (scripted) and
      GPU-off paths** and the six named golden regressions through the API.
- [ ] Eval gates met and recorded to MLflow: injection ASR = 0 (defenses on), FP = 0 on benign
      controls, live-path parity = Table 1 gate values, GS-02 API-level inventions = 0.
- [ ] **Benchmark tables updated in `docs/BENCHMARKS.md`:**
      *Table 1* — untouched (config unchanged); a **live-path parity note** with the serving-run
      stamp re-validates it end-to-end.
      *Table 2* — `answer_relevancy` backfilled on the pinned base run (footnote ¹ discharged);
      no other cell changes (columns stay pinned under `prompt_hash 78215ccc…`).
      *New "Serving & guardrails" section* — API e2e latency p50/p95, tokens/sec through the API
      (+ vLLM `/metrics` snapshot), injection ASR defense-on/off + utility-under-attack,
      degradation posture — each row carrying the §6.1
      reproducibility stamp (code SHA, prompt v1.1 hash, tool-schema v0 sha256, serving-run id,
      model@revision, vLLM flags).
- [ ] `serving/README.md` written (purpose, architecture, run, results); `docs/DECISIONS.md` gains
      DEC-P5-1..6; `TOOL_SCHEMA.md` status note added.
- [ ] Runs cleanly from scratch: fresh clone + `.env` → `make up && make api-up` serves the
      GPU-off experience with zero GPU.
- [ ] **30-second demo path:** `make api-up` then one curl to `/api/chat` (structured offline
      answer with evidence links) and one to `/api/replay/GS-08a` (the recorded backtracking
      story) — no GPU, no wait. (With a GPU window: the same curl against the live path.)
- [ ] Resume-ready quantified bullet in `docs/PORTFOLIO.md` (e.g. injection ASR 0 vs baseline,
      e2e latency, $-per-demo, graceful-degradation design).

---

## 7. Open questions — RESOLVED (2026-07-05, user-confirmed with the recommended options)

- **Q1 — Injection fixture home → separate `evals/injection/` suite** with its own schema;
  `GOLDEN_SET_SCENARIOS.md` stays the frozen GS-01..11 catalog, gaining only a one-paragraph
  pointer. (Logged in DEC-P5-3.)
- **Q2 — Prompt v1.1 (spotlighting appendix) → APPROVED:** the serving prompt extends the frozen
  bundle as a new recorded lock (v1.1); **pinned Table 2 columns are never re-scored under it** —
  they stay pinned to `78215ccc…`; P5 artifacts record their own hash. (Logged in DEC-P5-3.)
- **Q3 — Streaming → non-streaming JSON confirmed for P5;** SSE deferred to P6 UI. (Recorded in
  DEC-P5-1 consequences.)
- **Q4 — Window budget → ≈ $1–1.5 APPROVED** for the single P5 capture session (parity +
  injection ASR on/off + latency/tokens + `/metrics` snapshot + relevancy backfill), within the
  DEC-0003 envelope. (Recorded in DEC-P5-4 consequences.)

---

## 8. Sources / research annex (web-research pass, accessed 2026-07-05)

**Prompt-injection defense (drives §2.5 layering + the D3 choice):**
- Beurer-Kellner et al., *Design Patterns for Securing LLM Agents against Prompt Injections*,
  arXiv 2506.08837 (ETH/Google/IBM/Microsoft/Invariant) — structural patterns (action-selector,
  plan-then-execute, context-minimization…) beat detection; detection heuristics are explicitly
  best-effort. Basis for §2.5's structure-first framing and the layer-5 honesty note.
- Debenedetti et al., *Defeating Prompt Injections by Design (CaMeL)*, arXiv 2503.18813 (Google
  DeepMind) — capability-tracked data flow around an untouched LLM; our cheap analog = pydantic-
  bounded tool results + schema-validated calls + read-only tool surface.
- Hines et al., *Defending Against Indirect Prompt Injection Attacks With Spotlighting*,
  arXiv 2403.14720 — delimiting/datamarking/encoding; datamarking/encoding cut ASR >50% → <2% on
  GPT-family; encoding depends on model strength (the D3-C rejection at 4B).
- **OWASP Top 10 for LLM Applications 2025** — LLM01:2025 Prompt Injection (indirect variant) is
  the taxonomy row this section answers; its named mitigations (input/output filtering, privilege
  control, human oversight for risky actions) map 1:1 onto §2.5 layers.
- BIPIA (Benchmark for Indirect Prompt Injection Attacks) — the query-side vs context-side
  fixture split ROADMAP §6.5 names, mirrored by `evals/injection/`.
- AgentDojo (agent-with-tools injection benchmark) — the attacker-directed tool-call-redirection
  fixture class and the utility-under-attack metric.
- arXiv 2511.15759, *Securing AI Agents Against Prompt Injection Attacks* (ROADMAP §7 reference).

**Serving & observability (drives D4/D6/§2.6/§2.7):**
- vLLM docs *Metrics* — Prometheus-compatible `/metrics` on the OpenAI-compatible server (TTFT/
  TPOT/queue/KV-cache metrics); captured as one-shot window snapshots, no standing Prometheus.
- vLLM docs *Embedding usages* + community threads — `vllm serve BAAI/bge-m3` /
  `bge-reranker-v2-m3` expose `/v1/embeddings` / `/score` / `/rerank`; BGE-M3 sparse/colbert are
  "extra weights" with version-sensitive support (the D4-B drift risk).
- `michaelfeil/infinity` (MIT) — embeddings+rerank server, OpenAI-aligned; **BGE-M3 sparse output
  not supported** (issue #146 open since 2024) — the D4-C rejection.
- BAAI BGE-M3 card / FlagEmbedding docs — lexical weights come only from the FlagEmbedding pass
  (the D4-A parity argument; same basis as DEC-P2-2).
- Langfuse docs *Token & cost tracking* — usage ingested per generation; **custom model price
  definitions required for self-hosted models** (else cost = $0); *Custom Dashboards* — v3 query
  engine over traces for latency/cost/token widgets (the D6-A basis, self-hosted per DEC-P3-7).

**In-repo precedents (the reuse contract):** `evals/driver.py` (agent loop, schema-generated
tools array, emitted-call validation), `LLMClient` up/off/error contract (DEC-P0-4),
`Tracer.span` seam (DEC-P3-6), FT-window LLM+embedder co-residency (DEC-P4-9 record), DEC-P2-7
self-contained-script + pin pattern, DEC-P2-6 committed-artifact CI gating, BENCHMARKS Table 2
footnote ¹ (the P5-owned relevancy gap) and its GS-02 = 1 ⚠ rows (the serving-layer output gate's
reason for existing).
