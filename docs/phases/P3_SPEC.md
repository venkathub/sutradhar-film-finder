# P3 Spec — Eval + observability harness (capture the PRE-fine-tune baseline)

> **Status: EXECUTED — PHASE COMPLETE (2026-07-03).** All 14 tasks delivered on
> `feature/p3-eval-observability` (17 commits); §6 Definition of Done checked item by item below.
> **Every §4 gate met:** GS-02 = 0 inventions on the committed run · seeded faults 3/3 caught ·
> **judge κ = 0.738 ≥ 0.6 (frozen: `openai/gpt-oss-20b @ 6cee5e81…`, one ephemeral A100 session,
> ~6 min, destroyed)** · dry-run complete (12/12 fixtures, Tier-1 recomputation matches) ·
> retrieval suite untouched and green · Langfuse live on the self-hosted instance (trace exported
> + committed) · MLflow runs recorded (dry-run `c2fb0eab…`, Table 1 backfill `26dc0470…`).
> Pinned artifact: `evals/generation_runs/20260703T012339Z-e7fff041.json`.
>
> **Rev 6 (2026-07-03, execution close-out) — deviations discovered live, all logged:**
> - **DEC-P3-8 (new):** `search_by_plot` in the harness replays the committed P2 retrieval run
>   per driven fixture (recorded providers key by query-hash; the model paraphrases — replay keeps
>   both Table 2 columns byte-identical on the tool surface).
> - **DEC-P3-4 amendments:** INTENT preamble placement pinned (final prose answer per turn);
>   bold-title formatting contract added for the deterministic detector (prompt re-pinned,
>   `prompt_hash 78215ccc…`).
> - **DEC-P3-7 amendments (live infra findings):** AIC checkout is **dashboard-only** (403 via API
>   key) → one-time manual purchase, find-or-create + phase 2 stay automated; the AIC edge
>   firewall opens **only SSH** → Caddy/443 impossible on this tier, public HTTPS rides an
>   **outbound cloudflared tunnel** (user-approved); ufw NAT self-lockout found + fixed (recovered
>   via API reinstall — the from-scratch property proven for real); swap is host-managed in LXC
>   (best-effort step); compose derives neither `DATABASE_URL` nor the S3 secret keys (written
>   explicitly + heal-in-place); web service is `langfuse-web`. Five findings, all folded into
>   `provision.py` + its fake-API/SSH tests.
> - **DEC-P3-3 amendment:** ragas 0.4.3 is import-broken against langchain-community ≥ 0.4 →
>   `[tool.uv]` constraint pin.
> - Judge validation used a 30-item worksheet (spec said ~24); labels by the project owner with
>   one assistant-flagged correction (`fai-GS-07e`) — methodology note in DEC-P3-1's amendment.
>
> *(Grooming history: Rev 1–5 approval notes preserved below.)*
>
> **Status at approval: APPROVED (2026-07-02).** Grooming complete — all recommendations
> user-confirmed; decisions logged as **DEC-P3-1..7** in `docs/DECISIONS.md`; open questions
> resolved (§7).
> **Rev 5 (2026-07-02):** Q1–Q6 resolved per recommendations; **DEC-P3-7 upgraded to an
> idempotent from-scratch bootstrap** (user requirement: the script must set up Langfuse from
> scratch if not installed — find-or-create instance, check-then-act configuration, safe
> re-runs); CLAUDE.md §Tech stack reconciled to DEC-P3-7.
> **Rev 2 (2026-07-02):** DEC-P3-1/DEC-P3-3 re-grounded after a web-research pass (§8) on the
> user's challenge — judge is now a **self-hosted OSS model on the ephemeral GPU**, not a frontier
> API; frontier demoted to documented escalation. Whole eval stack is OSS + self-hosted.
> **Rev 3 (2026-07-02):** in-depth model-stack research pass (§8). **DEC-0001/0002 re-verified,
> not reopened** (Gemma 4 E4B Apache 2.0 + vLLM recipe confirmed live; Qwen3-4B-Instruct-2507
> confirmed; Sarvam-M teacher rationale confirmed). Judge candidates re-ordered on measured
> evidence: **gpt-oss-20b primary** (OpenAI family — fully disjoint from Google/Alibaba/Mistral;
> Apache 2.0; ~14–16 GB → fits the A100 40 GB workhorse *with* BGE-M3); Llama-3.3-70B **dropped**
> (AWQ-INT4 ≈ 47 GB weights — does not close on the 48 GB Ada with KV cache).
> **Rev 4 (2026-07-02):** **DEC-P3-7 added** — Langfuse self-hosted (v3 docker compose) on an
> **AIC Cloud VPS** per user direction, replacing the CLAUDE.md "Langfuse Cloud free tier"
> default (kept as fallback); deployment plan, hardening, backups, and evidence-longevity
> mitigation specified; CLAUDE.md reconciliation noted.
>
> **Entry criteria (met):** P2 exit gate passed — Recall@10 = 1.000 (≥ 0.90) and version-set recall
> = 1.0 on GS-01/GS-06, committed run `20260702T135315Z-f6583183` (DEC-0002 accepted;
> `docs/BENCHMARKS.md` Table 1 populated). TOOL_SCHEMA **v0 FROZEN** (DEC-P1-8) with the
> machine-readable artifact `docs/phases/tool_schema.v0.json` and its three CI conformance layers.
>
> **ROADMAP P3 charter:** wire RAGAS + Langfuse + MLflow; make Tier-2 CI real; freeze LLM-as-judge
> governance and the base-model prompting strategy; **build and dry-run** the generation benchmark
> harness against recorded fixtures / a mock endpoint — the **authoritative PRE-fine-tune base
> capture happens at the top of the P4 GPU window** using this exact harness, so base and QLoRA
> columns share identical serving conditions.

---

## 1. Scope

### In scope

1. **Generation benchmark harness** (`sutradhar.evals.generation` + `evals/run_generation_eval.py`)
   that executes the conversational golden fixtures (GS-02 negatives, GS-07 code-mixed, GS-08
   multi-turn) against any OpenAI-compatible endpoint (`LLM_BASE_URL`) and scores every Table 2
   metric: tool-call accuracy, code-mixed intent accuracy, slot-extraction accuracy, backtracking
   coherence, faithfulness (1 − hallucinated-movie rate), answer relevancy, and (when live) GPU
   latency p50/p95 + tokens/sec. Output = a **committed, versioned generation-run artifact**
   (`evals/generation_runs/<run_id>.json`), mirroring the P2 retrieval-run pattern (DEC-P2-6).
2. **Conversation driver**: multi-turn loop that sends the frozen system prompt + TOOL_SCHEMA v0
   tools, validates every model-emitted tool call against `tool_schema.v0.json` **before**
   executing it against `sutradhar.graph.repository`, feeds tool results back, and records the
   full transcript (messages, calls, tool results, usage, latency) per fixture.
3. **`LLMClient.chat()`** extension: OpenAI tool-calling (`messages` + `tools` +
   `tool_choice`), preserving the DEC-P0-4 contract (`status="off"` is a first-class,
   never-crashing path; single injected `httpx.Client` so `MockTransport` mocks everything).
4. **LLM-as-judge governance (frozen this phase):** a **self-hosted OSS judge served by vLLM on
   the ephemeral GPU** (DEC-P3-1; frontier API only as the documented escalation), pinned as
   `{HF repo, revision SHA, prompt-hash}`; **different model family** than Gemma, than Qwen
   (fallback base) *and* than the Mistral/Sarvam teacher line (ROADMAP §6.4);
   backtracking-coherence rubric (GS-08); judge validated against a small human-labelled sample
   with agreement reported.
5. **RAGAS wiring:** faithfulness + answer relevancy computed through the pinned self-hosted
   judge endpoint + **BGE-M3 embeddings in the same GPU session** (RAGAS supports custom
   OpenAI-compatible LLMs/embeddings — DEC-P3-3); **no external eval API**. The headline
   faithfulness gate stays **deterministic** (no-hallucinated-movie detector, below).
6. **Deterministic no-hallucinated-movie detector:** every film title asserted in a final answer
   must resolve (match_key + rapidfuzz ≥ 0.80, DEC-P1-5) to a Work/Version present in that
   conversation's tool results; anything else counts as an invention. This is the GS-02 gate
   (= 0 inventions) and the "1 −" term in Table 2's faithfulness column.
7. **Langfuse tracing** on the existing chokepoints (`LLMClient`, repository tool functions, the
   driver, judge calls) via a thin no-op-safe wrapper — the same wrapper P5's FastAPI path reuses.
   Settings keys already exist (`LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`). **Backend = self-hosted
   Langfuse v3 on an AIC Cloud VPS (DEC-P3-7, user-directed; replaces the CLAUDE.md "Langfuse
   Cloud free tier" default — reconciliation noted for CLAUDE.md).**
8. **MLflow tracking + registry (self-hosted)**: compose service; every generation run and the
   existing P2 retrieval run logged with the §6.1 reproducibility stamp; registry stands ready for
   the P4 adapter. New env: `MLFLOW_TRACKING_URI`.
9. **Two-tier CI made real:** `tier2.yml` placeholder replaced by a dispatch-gated job that runs
   the generation harness against `LLM_BASE_URL` during a GPU window and uploads the sealed run;
   **Tier-1** gains `test_golden_generation_regressions.py`, recomputing generation gating metrics
   from the committed run artifact (pinned `GENERATION_RUN` env) — between GPU windows CI gates on
   the recorded run, never a live call (ROADMAP §6.2).
10. **Emitted-tool-call validation** as a metric *and* a test: the DEC-P1-8 validator
    (`test_golden_expected_tool_calls_validate` was built to be reused "against model-emitted
    calls") is applied to every call the model emits; hallucinated tool or parameter names are
    caught, scored as failures, and proven catchable by a seeded-fault test.
11. **Generation-slice golden fixture expansion** (P4 entry criterion "golden generation fixtures
    ready"): add `expected_intent` / `expected_slots` labels to the fixture schema and expand
    GS-02/GS-07/GS-08 to a stable count (**confirmed: GS-07 → 5, GS-08 → 3, GS-02-conversational
    → 4**). All new fixtures pass the
    existing golden-eligibility gate (`build_golden.py`, HIGH/human-verified only).
12. **Base-model prompting strategy frozen** (system prompt + exemplars as hashed in-repo
    artifacts, §6.3) so the P4 before/after is fair — decision DEC-P3-4 below.
13. **Harness dry-run** against a scripted mock endpoint (no model, no GPU): proves end-to-end
    scoring incl. detection of a seeded hallucinated tool call and a seeded invented movie; the
    dry-run artifact is committed with `mode: "dry_run"` and is **never** published to Table 2.

### Non-goals (explicit — prevents scope creep)

- **No QLoRA, no fine-tuning, no synthetic data generation** — P4.
- **No authoritative base-benchmark GPU capture in P3.** Per ROADMAP, the base column is captured
  at the *top of the P4 GPU window* with this harness (`make benchmark-generation`), so both
  Table 2 columns share one serving config. P3 ships the machinery + dry-run evidence only.
  (Consequence: P3's only GPU spend is **one short judge-validation session** — ≤ 1 h, < $1,
  inside the DEC-0003 envelope, ephemeral create→judge→destroy per DEC-P0-5 — no base-capture
  run, no training, nothing left running.)
- **No FastAPI request path / API orchestration** — P5. Tracing attaches to the existing
  chokepoints; full request-path tracing arrives with the API.
- **No retrieval changes.** Table 1 numbers are untouched (its reproducibility stamp gains its
  promised MLflow run link — a backfill log, not a re-run). No re-embedding, no new GPU artifacts.
- **No indirect prompt-injection defense / BIPIA slice** — P5 (§6.5). GS-02 query-side abstention
  is in scope; context-side injection is not.
- **No dashboards** (token/cost/latency dashboards are P5); Langfuse traces + MLflow runs suffice.
- **No TOOL_SCHEMA change.** The harness consumes frozen v0 as-is (verified below); v0 stays v0.
- **No golden-set changes outside the generation slice** (retrieval/graph fixtures stay frozen —
  the P2 committed artifact must keep validating).
- **No Java.** (See §2.9.)

---

## 2. Design

### 2.1 Component breakdown

| Component | Module (new unless noted) | Responsibility |
|---|---|---|
| Chat client | `sutradhar.serving.llm_client` (extend) | `chat(messages, tools, …) -> ChatResult` (OpenAI tool-calling); off/error contract unchanged |
| Conversation driver | `sutradhar.evals.driver` | Multi-turn fixture execution: prompt assembly, tool-call validation → repository execution → result feedback, bounded turns, transcript capture |
| Tool executor | `sutradhar.evals.driver` (uses `sutradhar.graph.repository`) | Maps validated calls to the five v0 repository functions; `search_by_plot` uses the artifact-backed retriever (recorded run) — no neural op on the laptop |
| Metric scorers | `sutradhar.evals.generation` | Pure functions: tool-call accuracy (DEC-P3-5), intent/slot accuracy, hallucinated-movie detector, aggregation; artifact models (`GenerationRunArtifact`) |
| Judge | `sutradhar.evals.judge` | Pinned cross-family judge client — OpenAI-compatible and env-driven (`JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY`), so the self-hosted vLLM judge (DEC-P3-1) vs the frontier escalation is pure config, not code; backtracking-coherence rubric; prompt hashing; human-agreement report |
| RAGAS adapter | `sutradhar.evals.ragas_metrics` | faithfulness + answer_relevancy through the pinned judge; ragas version pinned in the stamp |
| Tracing | `sutradhar.obs.tracing` | Thin Langfuse wrapper (span context managers); **no-op when keys unset**; wraps LLM, tool, judge calls |
| Experiment tracking | `sutradhar.obs.mlflow_log` | Log runs + params + metrics + artifacts with the §6.1 stamp; experiment naming conventions |
| Runner CLI | `evals/run_generation_eval.py` | Thin Typer CLI (same pattern as `run_retrieval_eval.py`); `make benchmark-generation` / `make generation-dryrun` |
| Mock endpoint | `tests/` + `evals/mock_llm.py` | Scripted OpenAI-compatible responses (httpx.MockTransport in tests; a canned-transcript player for the committed dry-run) |
| CI | `.github/workflows/tier2.yml` (replace placeholder), `tests/test_golden_generation_regressions.py` | Tier-2 dispatch harness run; Tier-1 recorded-artifact gate |

### 2.2 Data models / schema extensions

**Golden fixture schema (additive; `sutradhar.evals.golden`)** — the generation metrics need
labels the current schema lacks:

```python
class GoldenFixture(...):            # existing fields unchanged
    expected_intent: str | list[str] | None = None   # one label, or one per turn (multi-turn)
    expected_slots: dict[str, Any] | list[dict[str, Any]] | None = None  # ditto
```

- **Intent taxonomy (frozen as a hashed artifact alongside the system prompt):**
  `find_by_plot | find_by_title | list_versions | refine | disambiguate | out_of_catalog`.
  Six labels, chosen to map 1:1 onto the v0 tool surface (plot→`search_by_plot`,
  title→`resolve_title`, versions→`get_versions`, refine→`refine_filter`, disambiguate→GS-10
  ask-back, out_of_catalog→abstain/NO_MATCH).
- **Slot keys** reuse the `refine_filter.by` vocabulary plus `plot_description` and `title`:
  `{title, plot_description, actor, language, year, era, relationship}` — nothing invented; the
  slots are exactly what the frozen tool surface can consume.
- Existing GS-07/GS-08 fixtures gain these labels; the retrieval/graph fixtures don't need them
  (fields optional, `extra="forbid"` preserved).

**Generation-run artifact (`evals/generation_runs/<run_id>.json`)** — committed, mirrors
`EvalRunArtifact`:

```python
class GenerationRunArtifact(BaseModel):
    run_id: str                      # e.g. 20260715T…Z-<hash8>
    mode: Literal["dry_run", "live"] # dry_run = mock endpoint; live = GPU window
    model: str                       # LLM_MODEL + revision (or "mock")
    serving: dict                    # vLLM version/flags, GPU type, decode params (null in dry_run)
    prompt_hash: str                 # frozen system prompt + exemplars + intent taxonomy
    tool_schema_version: str         # "v0"
    judge: JudgeConfig               # model, version, prompt_hash, ragas version
    retrieval_run: str               # the pinned P2 artifact backing search_by_plot
    fixtures: list[FixtureResult]    # per fixture: transcript, per-turn scores, violations
    metrics: MetricsBlock            # the Table 2 aggregates + per-slice breakdown
    stamp: ReproStamp                # code SHA, data-snapshot hash, golden-set hash, dates
```

`FixtureResult.transcript` records every message, emitted tool call (raw + validation verdict +
bound arguments), tool result, token usage, and latency — this is what Langfuse traces mirror and
what the judge scores are computed from (auditable, replayable).

### 2.3 Request/data flow (one fixture)

```
fixture (golden YAML) ──► driver
  system prompt (frozen, hashed) + tools (from tool_schema.v0.json) + turn 1 user msg
        │
        ▼
  LLMClient.chat() ──► LLM_BASE_URL (vLLM in P4 window / mock in dry-run)
        │  assistant msg (may contain tool_calls)
        ▼
  validate each call against tool_schema.v0.json
        ├─ invalid (hallucinated tool/param) → scored failure; error message returned to
        │   model as the tool result; loop continues (bounded)          [metric: schema validity]
        └─ valid → execute sutradhar.graph.repository.<tool>(session, …)
                    (search_by_plot ← artifact retriever, RETRIEVAL_RUN pinned)
        │  tool results appended as tool-role messages
        ▼
  loop until assistant answers in prose or MAX_TOOL_ROUNDS (default 6) → next user turn
        │
        ▼
  scorers: tool-call accuracy (vs expected_tool_calls, placeholder-bound)
           intent/slot accuracy (vs expected_intent/expected_slots)
           hallucinated-movie detector (answer titles vs tool-result set, fuzzy ≥ 0.80)
           judge: backtracking coherence (GS-08 rubric), RAGAS faithfulness + relevancy
        │
        ▼
  FixtureResult → GenerationRunArtifact → MLflow run + Langfuse trace + committed JSON
```

**Placeholder binding:** golden `expected_tool_calls` carry `$work_id` / `[$version_set]`; the
scorer binds them to the ids actually returned by the model's earlier successful calls in the same
conversation, then compares argument-by-argument. A call chain that resolves a *different* work is
a scored mismatch, not a crash.

**Compute placement (ROADMAP §2, strict):** the driver, scorers, and detector are laptop/CI-safe
(DB + recorded artifacts + string math). The only neural ops are (a) the model under test — mock in
P3, vLLM-on-GPU in the P4 window; (b) the judge + RAGAS — the **self-hosted judge served by vLLM
in a short ephemeral GPU session**, run as a **batch pass over recorded transcripts** (judging
needs only text, so it never requires the model under test to be live, and 24/7 judge
availability is never needed — between windows CI gates on the recorded artifact, DEC-P2-6
posture); (c) answer-relevancy embeddings — **BGE-M3 in the same GPU session** (DEC-P3-3).
Nothing neural runs on the laptop, and **no external eval API is required**.

### 2.4 Metric definitions (Table 2 columns, pinned here so P4 reuses them byte-identically)

| Metric | Definition | Source of truth |
|---|---|---|
| Tool-call accuracy | Per DEC-P3-5 (below): call-level AST match + sequence-level all-correct-in-order; schema-validity reported alongside | `expected_tool_calls` + `tool_schema.v0.json` |
| Code-mixed intent accuracy | Exact-match of predicted intent label per turn, GS-07 slice (prediction parsed from a structured preamble the frozen prompt requires) | `expected_intent` |
| Slot-extraction accuracy | Micro-F1 over `expected_slots` key-value pairs; values normalized via `match_key` for titles, casefold otherwise | `expected_slots` |
| Backtracking coherence | Judge rubric per GS-08 conversation: per-turn correct version + context carried + no re-answering turn 1; mean of [0,1] scores | judge (frozen config) |
| Faithfulness (headline) | 1 − hallucinated-movie rate: inventions / titles asserted, per §1.6; **gate: 0 inventions on GS-02** | deterministic detector |
| Faithfulness (supplementary) | RAGAS faithfulness of the final answer against the conversation's tool results as context | RAGAS + judge |
| Answer relevancy | RAGAS answer_relevancy (question regeneration + embedding similarity) | RAGAS + judge + remote embeddings |
| GPU latency / throughput | p50/p95 wall-clock per assistant turn; completion tokens/sec from `usage` | live runs only (`null` in dry_run) |

### 2.5 Judge governance (frozen this phase; ROADMAP §6.4)

- Judge = **different model family** than Gemma, than Qwen (fallback base), *and* than the
  Mistral line (Sarvam-M's base — teacher-adjacent, so Prometheus-2 is excluded too) → DEC-P3-1
  recommends a **self-hosted OSS judge on the ephemeral GPU**, selected empirically by the
  human-agreement gate; frontier API only as the recorded escalation.
- Judge config = `{HF repo id, revision SHA, vLLM serving config, prompt file + hash,
  temperature 0, ragas version}` — recorded in `DECISIONS.md`, in every artifact's `judge` block,
  and in the Table 2 stamp. Pinned open weights make the judge **fully reproducible** — no
  provider model-deprecation risk under the stamp.
- Judging is a **batch pass over recorded transcripts** inside a GPU session; between windows CI
  gates on the recorded artifact — the judge never needs to be "always on".
- **Human-agreement validation:** ~24 judged items (GS-08 coherence + a faithfulness sample,
  incl. deliberately incoherent/hallucinated foils generated by seeding the mock) are
  independently labelled by the human reviewer; report percent agreement + Cohen's κ in
  `evals/README.md` — the same agreement-not-correlation methodology as the Judge's Verdict
  benchmark (arXiv 2510.09738; its 3-expert human-human baseline is κ ≈ 0.79–0.80, context for
  our target). Target: κ ≥ 0.6 (substantial) — below that, the rubric is revised and
  re-validated once, then the frontier escalation triggers (DEC-P3-1).
- Judge prompts live in-repo (`evals/prompts/judge_*.md`), hashed per §6.3.

### 2.6 Observability wiring

- **Langfuse** (`sutradhar.obs.tracing`): one trace per fixture conversation; spans for each
  `chat()` call, each tool execution, each judge call; metadata = fixture id, run_id, prompt_hash.
  Wrapper is import-safe and **no-ops when `LANGFUSE_*` unset** (Tier-1 CI, forks) — proven by
  test. **Backend: self-hosted Langfuse v3 on an AIC Cloud VPS (DEC-P3-7)** — single-node docker
  compose (web + worker + Postgres + ClickHouse + Redis + MinIO), pinned release tag, headless
  init with pinned project keys, Caddy TLS on 443 only, signup disabled, nightly off-box backups
  (compose ships none). Benchmark-cited traces are additionally **exported (JSON + screenshot)
  and committed** with the run artifact so standing evidence never depends on VPS uptime.
- **MLflow** (`sutradhar.obs.mlflow_log` + compose service): experiments `sutradhar/retrieval` and
  `sutradhar/generation`; each generation run logs params (model, prompt_hash, judge config,
  tool_schema_version, retrieval_run), metrics (Table 2 aggregates + slices), and artifacts (the
  run JSON). One backfill run logs the P2 retrieval Table 1 metrics so its stamp's "(MLflow wiring
  lands in P3)" note is discharged. Registry (DB-backed) stands ready for the P4 adapter.
  Topology = DEC-P3-2.

### 2.7 CI wiring (two-tier, ROADMAP §6.2)

- **Tier-1 (every PR, no GPU, no model calls):**
  - existing retrieval regression suite unchanged (`test_golden_retrieval_regressions.py`);
  - new `test_golden_generation_regressions.py`: loads the pinned `GENERATION_RUN` artifact,
    **recomputes** every deterministic metric (tool-call accuracy, schema validity, intent/slot,
    hallucinated-movie) from the recorded transcripts with the same scorer functions, asserts they
    match the artifact's metrics block, and enforces the hard gates (no-hallucinated-movie = 0 on
    GS-02; emitted-call schema validity fully accounted — every invalid call flagged). Judge
    scores are checked for presence/shape, not re-judged (no API calls on PRs).
  - Tier-1's P0 "validate against recorded artifacts (placeholder)" echo step is retired — the
    real validation is now entirely inside pytest.
- **Tier-2 (`tier2.yml`, workflow_dispatch only):** inputs `run_mode` (`dry_run|live`) + `reason`;
  runs `make benchmark-generation` against secrets-provided `LLM_BASE_URL`, then the judge/RAGAS
  batch pass against `JUDGE_BASE_URL` (the self-hosted judge served in the same GPU window);
  uploads
  the sealed run JSON as a workflow artifact. A human reviews and commits it (updating
  `GENERATION_RUN`) via PR — the same "committed artifact is the gate" posture as DEC-P2-6.
  Never runs on PRs (cost + secrets, per DEC-P0-5 guardrail).

### 2.8 Tool-schema conformance statement

This phase **calls** tools and **needs no new or changed tool** — the five frozen v0 signatures
cover every fixture (`resolve_title`, `search_by_plot`, `get_work`, `get_versions`,
`refine_filter`). The harness consumes `tool_schema.v0.json` in two directions:
(1) *outbound* — the `tools` array passed to `chat()` is generated from the JSON artifact (never
hand-written, so drift is impossible); (2) *inbound* — every emitted call is validated against it
before execution (reusing the DEC-P1-8 validator). `TOOL_SCHEMA.md` gets only a status-note line
("v0 consumed unchanged by the P3 harness; emitted-call validation live"), exactly like the P2
note — **no version bump, nothing to add to DECISIONS.md for the schema itself.**

### 2.9 Python vs (optional) Java

**All Python.** The harness is glue over the existing Python-only surfaces (pydantic fixtures,
SQLAlchemy repository, the `openai` SDK client) and the eval ecosystem (RAGAS, MLflow, Langfuse
SDKs) is Python-native. CLAUDE.md's optional Java moat is explicitly a **P5-grooming** decision
about the public gateway; nothing in an eval harness benefits from a JVM, and introducing one here
would fork the single-lockfile reproducibility story (DEC-P0-1) for zero portfolio signal.

### 2.10 What changes where (repo)

```
pyproject.toml                 + ragas, langfuse, mlflow (laptop-safe: SDKs only, no local models)
.env.example                   + MLFLOW_TRACKING_URI, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY,
                                 GENERATION_RUN   (LANGFUSE_* already present)
infra/docker-compose.yml       + mlflow service (DEC-P3-2)
infra/gpu/jarvis.py            + `judge` session (serve the pinned OSS judge + BGE-M3 via vLLM;
                                 DEC-P0-5 ephemeral create→run→destroy pattern)
infra/langfuse/                new: provision.py (AIC API: wallet→plans→checkout→verify),
                                 VPS runbook (README), Caddyfile, headless-init env template,
                                 backup script (pg_dump + ClickHouse BACKUP + MinIO sync, cron)
src/sutradhar/config/settings.py       + mlflow/judge/generation_run fields
src/sutradhar/serving/llm_client.py    + ChatResult, LLMClient.chat()
src/sutradhar/evals/{driver,generation,judge,ragas_metrics}.py   new
src/sutradhar/obs/{tracing,mlflow_log}.py                        new
src/sutradhar/evals/golden.py          + expected_intent/expected_slots
evals/run_generation_eval.py           new Typer CLI
evals/mock_llm.py                      scripted dry-run endpoint player
evals/prompts/{system_v1.md, exemplars_v1.md, judge_coherence_v1.md, …}   hashed artifacts
evals/golden/gs{02,07,08}*.yaml        labels + expansion (Q1)
evals/generation_runs/<run_id>.json    committed dry-run artifact
.github/workflows/{tier1.yml, tier2.yml}   retire stub step; real Tier-2
tests/test_{llm_chat,driver,generation_metrics,judge,tracing,mlflow_log,
            golden_generation_regressions}.py   new
Makefile                       + benchmark-generation, generation-dryrun, judge-validate,
                                 mlflow-up, langfuse-up (idempotent, DEC-P3-7)
docs/{BENCHMARKS.md, DECISIONS.md, phases/TOOL_SCHEMA.md, PORTFOLIO.md}, evals/README.md   updates
```

---

## 3. Decisions — CONFIRMED 2026-07-02 (logged as DEC-P3-1..7 in `docs/DECISIONS.md`)

> Settled and **not** reopened — but **re-verified by the Rev-3 research pass (§8)** so this phase
> stands on checked ground, not stale citations: **base = Gemma 4 E4B** (HF card live; Apache 2.0
> per the Google model card and the vLLM 2026-04 announcement; function calling served via the
> official vLLM Gemma-4 recipe — and already booted live on the A100 in P0); **fallback =
> Qwen3-4B-Instruct-2507** (Apache 2.0; the 2507 refresh's tool-use/multilingual gains confirmed);
> **teacher = Sarvam-M 24B** (Apache 2.0, Mistral-Small base; +20% Indic, **+86% romanized-Indic**
> — precisely the code-mix teacher rationale); **embedder/reranker = BGE-M3 /
> bge-reranker-v2-m3** (DEC-0002 measured at gate 1.000 — nothing to relitigate). Also settled:
> GPU SKU/cost envelope (DEC-0003), vector store & retrieval config (DEC-P2-1..4), NO_MATCH θ
> (DEC-P2-5), committed-artifact CI gating (DEC-P2-6), tool schema v0 (DEC-P1-8).

**Model stack after this grooming (all verified in the Rev-3 research pass; every ID env-driven):**

| Role | Model | License | Status | Where decided |
|---|---|---|---|---|
| Fine-tune base (under test, Table 2) | **Gemma 4 E4B** | Apache 2.0 | settled, re-verified (P0 live boot) | DEC-0001 |
| Fallback FT base | **Qwen3-4B-Instruct-2507** | Apache 2.0 | settled, re-verified | DEC-0001 |
| Synthetic-data teacher (P4) | **Sarvam-M 24B** (or frontier API) | Apache 2.0 | settled, re-verified | DEC-0001 |
| Embedder / reranker | **BGE-M3 / bge-reranker-v2-m3** | MIT / permissive | settled, **measured** (gate 1.000) | DEC-0002 |
| **Judge (new, this phase)** | **gpt-oss-20b** primary; Phi-4-14B alternate; frontier API escalation | Apache 2.0 / MIT | **proposed — frozen only after the κ gate** | DEC-P3-1 |
| RAGAS relevancy embedder | BGE-M3 (same GPU session) | MIT | proposed | DEC-P3-3 |

### DEC-P3-1 — Judge model (cross-family, pinned) — **revised: self-hosted OSS judge, gpt-oss-20b primary**

Family-independence map: model under test = **Gemma** (Google); fallback FT base = **Qwen**
(Alibaba); P4 data-teacher = **Sarvam-M** (Mistral-Small base → the whole Mistral line, including
Prometheus-2, is teacher-adjacent). The judge must sit outside all three. Selection method =
exactly the "Turing Test for judges" methodology (Judge's Verdict, arXiv 2510.09738: correlation
filter → Cohen's-κ human-agreement; **27/54 models reach Tier-1 and judge excellence is NOT
size-dependent** — which is what makes a self-hosted mid-size judge defensible at all).

| Option | Trade-offs |
|---|---|
| **(A) Self-hosted OSS judge served by vLLM on the ephemeral GPU** — **recommended**, candidates in measured order: **(1) gpt-oss-20b** (OpenAI family, Apache 2.0, MoE ≈ 14–16 GB VRAM → runs on the **A100 40 GB workhorse alongside BGE-M3 in one session**; explicit vLLM support; documented LLM-as-judge usage (TruLens cookbook); independent eval (arXiv 2508.12461) shows it matching/beating gpt-oss-120b on several benchmarks); **(2) Phi-4-14B** (MIT, Microsoft family, ~10 GB Q4) as the alternate if gpt-oss-20b's rubric behaviour disappoints | Mission-aligned: fully self-hosted, **zero external keys**; weights pinned by **HF revision SHA** → *more* reproducible than a frontier API (providers deprecate/rev models under the stamp); reuses the DEC-P0-4 OpenAI-compatible client and the DEC-P0-5 ephemeral driver unchanged; judging is a **batch pass over recorded transcripts**, so judge availability between GPU windows is not needed (CI gates on the recorded artifact, DEC-P2-6). Both candidates fit the **$0.89/hr A100 40 GB** — no SKU upgrade, judge pass ≈ $0.3–0.9. Con: quality ceiling below frontier judges — mitigated because our judging tasks are narrow rubric checks against recorded context, not the open-ended reasoning-correctness judging where even GPT-4o-class judges approach random (JudgeBench, arXiv 2410.12784); the **κ ≥ 0.6 human-agreement gate is the empirical selector**, run per candidate on the same labelled sample. |
| (B) Frontier API judge (pinned dated version) | Highest quality ceiling; no GPU session. Cons: external key + per-run spend; provider model-deprecation undercuts the reproducibility stamp; dilutes the self-hosted / cost-discipline portfolio story. **Kept as the documented escalation path:** adopted (and logged here) only if no OSS candidate reaches κ ≥ 0.6 after one rubric revision. |
| (C) Rejected candidates, with reasons | **Llama-3.3-70B-Instruct** — judge-quality literature likes it, but AWQ-INT4 is ≈ **47 GB weights alone** (nodepedia/VRAM guides): does not close on the 48 GB Ada once KV cache is added, and forcing an 80 GB card breaks the DEC-0003 envelope for a *judge*. **Prometheus-2** (purpose-built OSS evaluator, 72–85% human agreement) — Mistral-based → teacher-adjacent. **Sarvam-M** — is the teacher (teacher-as-judge contaminates the before/after). **Qwen-family** (incl. its strong large judges) — shares family with the fallback base. **GLM-4-32B** (MIT) — viable family-wise but ~66 GB bf16 / needs quantization to fit, with no judge-specific evidence advantage over gpt-oss-20b; kept as a bench-depth note only. |

**Recommendation: A with gpt-oss-20b first**, selected empirically: serve candidates in one short
ephemeral GPU session, score the ~24-item human-labelled sample (mirroring the Judge's Verdict
κ methodology), freeze the winner (`{repo, revision SHA, prompt hash}`). B is the recorded
escalation. Because the judge client is OpenAI-compatible and env-driven
(`JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY`), A↔B is a config change, not code — the same
swappability contract as DEC-P0-4. One extra governance note: gpt-oss-20b is a **reasoning**
model — the judge prompt pins reasoning effort and the rubric output schema (guided decoding, as
proven in the P1 extraction run, DEC-P1-4 amendment) so judge outputs stay parseable and
deterministic at temperature 0.

### DEC-P3-2 — MLflow topology (self-hosted)

| Option | Trade-offs |
|---|---|
| **(A) Compose service, backend = existing Postgres (separate `mlflow` database), artifacts under `./data/mlflow-artifacts/`** — **recommended** | Genuine "self-hosted" (CLAUDE.md); DB-backed → **model registry works** (needed by P4); zero new infra beyond one compose service; survives `make down`; URI env-driven. Con: one more container on the low-spec laptop (MLflow server is light). |
| (B) SQLite file store, `mlflow ui` on demand | Lightest. Cons: weaker concurrent-write story once CI/Tier-2 also logs; "server-less registry" is a second-class MLflow path; less convincing as the portfolio's "self-hosted MLflow". |
| (C) Managed (Databricks free / hosted) | Violates the stated self-hosted posture; external account dependency. |

**Recommendation: A.**

### DEC-P3-3 — RAGAS scope & execution locus

| Option | Trade-offs |
|---|---|
| **(A) RAGAS library (version-pinned); LLM = the DEC-P3-1 self-hosted judge endpoint; embeddings = BGE-M3 served in the same GPU session** — **recommended** | RAGAS explicitly supports custom / OpenAI-compatible LLMs and embeddings (`llm_factory`/`embedding_factory`, `BaseRagasLLM`/`BaseRagasEmbeddings`; fully-offline local-model runs are a documented pattern). Demonstrates the named RAGAS skill (skills-proof map); reuses the settled embedder (DEC-0002) for answer-relevancy; **zero external APIs — the entire eval stack is OSS + self-hosted**. Cons: RAGAS internal prompts evolve → pin the ragas version in the stamp; RAGAS scores are computed only inside GPU sessions (fine: they score recorded transcripts in the same batch judge pass, and CI gates on the recorded artifact between windows). |
| (B) Hand-rolled faithfulness/relevancy with the pinned judge (no ragas dep) | Full prompt-hash control, fewer deps. Con: loses the recognized-tool signal; re-implements well-trodden metrics; more code to test. |
| (C) RAGAS with a frontier-API LLM + embedding API | No GPU needed to score. Cons: external key + cost; inconsistent with the self-hosted judge decision; only justified if DEC-P3-1 escalates to its frontier fallback. |

**Recommendation: A**, with the deterministic hallucinated-movie detector (not RAGAS) remaining
the *gating* faithfulness signal — RAGAS numbers are reported, the detector gates.

### DEC-P3-4 — Base-model prompting strategy (frozen for the before/after)

| Option | Trade-offs |
|---|---|
| (A) Zero-shot: system prompt + tool schema only | Simplest, but an *under*-prompted base inflates QLoRA lift — fails the "well-prompted base" honesty bar (CLAUDE.md, R2). |
| **(B) System prompt + intent-taxonomy instructions + 3 few-shot exemplars (handcrafted, disjoint from the golden set: one code-mixed, one multi-turn refine, one NO_MATCH), native Gemma function-calling format** — **recommended** | A genuinely strong base = credible headroom claim; exemplars pinned + hashed; disjointness from golden fixtures is test-enforced (no leakage). Con: longer prompt (more tokens/turn — measured, and identical for both Table 2 columns so the comparison stays fair). |
| (C) ReAct-style scratchpad prompting | Sometimes stronger, but fights vLLM's native function-call parsing (DEC-0001 chose Gemma partly for those tokens) and makes tool-call extraction ambiguous → noisier tool-call metric. |

**Recommendation: B.** The frozen prompt + exemplars + taxonomy are one hashed artifact; the P4
QLoRA run must use the *same* system prompt (minus exemplars only if the adapter internalizes
them — that choice is P4's, recorded there).

### DEC-P3-5 — Tool-call accuracy scoring rule

| Option | Trade-offs |
|---|---|
| (A) Strict full-sequence exact match only | Unambiguous but brittle: one benign extra `get_work` zeroes the fixture; punishes valid alternate orderings; makes the metric noisy at small n. |
| **(B) BFCL-style two-level scoring — call-level AST match (tool name + placeholder-bound, normalized arguments) + sequence-level "expected sequence appears in order, extra schema-valid calls tolerated"; report both, Table 2 headline = sequence level** — **recommended** | Matches how BFCL v4 (the benchmark DEC-0001 cites) scores; partial credit makes base-vs-QLoRA deltas interpretable; tolerating benign extras avoids penalizing grounded caution. Con: two numbers to explain (mitigated: Table 2 carries the headline, the artifact carries both). |
| (C) Unordered set match | Ignores order — but order *is* the behaviour for GS-08 backtracking; wrong for our headline scenario. |

**Recommendation: B.** Schema-validity (hallucinated tool/param rate) is always reported as a
third, independent number — it is the "no hallucinated tool" test surface.

### DEC-P3-6 — Langfuse instrumentation boundary

| Option | Trade-offs |
|---|---|
| **(A) Thin explicit wrapper (`sutradhar.obs.tracing`: `trace()`/`span()` context managers) around LLMClient, tool executor, judge** — **recommended** | No-op-safe without keys (Tier-1/forks); unit-testable with a fake sink; one seam P5's FastAPI middleware reuses; no decorator import-time coupling to the SDK. Con: a little manual plumbing at each chokepoint (there are exactly four). |
| (B) Langfuse `@observe` decorators everywhere | Less code. Cons: SDK import required at module import time even when tracing is off; harder to fake in tests; decorator magic spreads a third-party contract across the codebase. |
| (C) OpenTelemetry + OTLP export to Langfuse | Most "enterprise", but a heavier dependency + config surface than a portfolio eval harness needs; Langfuse-native attributes get lossy. |

**Recommendation: A.**

### DEC-P3-7 — Langfuse deployment: self-hosted v3 on an AIC Cloud VPS (user-directed)

**Context.** CLAUDE.md's tech stack named *Langfuse Cloud free tier*; the user directs
self-hosting on **AIC Cloud** (aiccloud.in — INR-first VPS, monthly no-contract, Ubuntu images,
Docker-ready, entry ₹99/mo). Langfuse v3 OSS is a 6-container compose stack (web, worker,
Postgres, ClickHouse, Redis/Valkey, MinIO — all FOSS, no license key needed); official VM
guidance is **≥ 4 cores / 16 GiB / ~100 GB**, and the compose path explicitly ships **no HA and
no backups**. The mission permits a cheap always-on host serving dashboards over recorded results
(Langfuse is not a neural model), so this violates nothing.

| Option | Trade-offs |
|---|---|
| (A) Langfuse Cloud free tier (status quo) | $0, zero ops, trace links never die. Cons: vendor-hosted data; free-tier retention/limits; weaker "self-hosted observability" portfolio signal next to the self-hosted MLflow + OSS judge. **Kept as the documented fallback.** |
| **(B) Self-host on one AIC Cloud VPS, single-node docker compose** — **recommended (user-directed)** | Completes the all-self-hosted MLOps story; INR billing, no contract; the same VPS is the natural P6 always-available static-surface host (one box = the whole standing-evidence surface, optional read-only MLflow mirror — a P6 consolidation option, not reopening DEC-P3-2). Cons: monthly cost; DIY backups/upgrades; trace-link longevity tied to VPS uptime — mitigated by committing exported traces + screenshots with each benchmark artifact. |
| (C) On-demand VM, up only for GPU windows/demos | Cheapest. Con: trace links dead by default — guts the standing-evidence story the links exist for. Rejected. |

**Plan selection (verified live against the API, 2026-07-02 — public endpoint
`GET https://api.aiccloud.in/api/v1/public/essential-vps-plans`, no auth):** the implemented
product is the **Essential VPS** line (Proxmox **LXC**; the docs-page "Cloud VPS API" paths 404 —
not live). Catalogue extract:

| Plan slug | vCPU / RAM / NVMe | Dedicated IPv4 | ₹/mo (paise ÷ 100) | 6-mo (−5%) | 12-mo (−10%) |
|---|---|---|---|---|---|
| essential-4gb | 2 / 4 GB / 40 GB | **no** | ₹399 | — | — |
| **essential-8gb** ✅ | **4 / 8 GB / 80 GB, 200 Mbps** | **yes** | **₹799** | ₹759/mo (₹4,554 total) | ₹719/mo (₹8,629 total) |
| essential-16gb (escalation) | 6 / 16 GB / 160 GB | yes | ₹1,599 | ₹1,519/mo | ₹1,439/mo |

**→ `essential-8gb` is the pick** — it matches the DEC-P3-7 sizing (4 vCPU/8 GB/80 GB) *and* is
the **cheapest tier with a dedicated IPv4** (`dedicated_ipv4: false` below 8 GB — no public
443 for `LANGFUSE_HOST` without it, so smaller tiers are structurally out, not just tight).
`os_choices` includes `ubuntu-24.04`. Escalation on OOM/disk pressure =
`POST /api/v1/vps/:id/upgrade` (wallet-charged) to essential-16gb.

**API provisioning flow — idempotent from-scratch bootstrap (user requirement, confirmed
2026-07-02):** `make langfuse-up` → `infra/langfuse/provision.py` must **set up Langfuse from
scratch if not installed**, be safe to re-run, and converge from any intermediate state
(find-or-create, then check-then-act per step; no destructive op without an explicit
`--recreate`-style flag). Auth: `Authorization: Bearer <AICCLOUD_API_KEY>`.

*Phase 1 — instance (AIC API, find-or-create):*
1. `GET /api/v1/vps` — locate `sutradhar-obs-01`; **exists+running → skip to Phase 2**;
   exists+stopped → `POST /:id/start`; absent → continue.
2. `GET /api/v1/billing/wallet` — pre-check balance (paise);
   `POST /api/v1/billing/wallet/topup` (min ₹100) → Razorpay order → `…/topup/verify`.
3. `GET /api/v1/public/essential-vps-plans` — resolve the pinned `planSlug` (no hardcoded IDs).
4. `POST /api/v1/vps/checkout` `{planSlug: "essential-8gb", name: "sutradhar-obs-01",
   os: "ubuntu-24.04"}` → Razorpay `orderId` (+ ₹1 security fee). **Payment legs are browser-only
   Razorpay Checkout by design** → the script prints the order and waits;
   `POST /api/v1/vps/checkout/verify` with the Razorpay ids/signature **+ `sshKeys[]`** (key-only
   SSH from first boot) → `{id, ssh: {host, port, user, password}}` → password auth disabled
   during bootstrap.

*Phase 2 — configuration (SSH, every step check-then-act, re-run converges):* swap configured →
Docker installed → **Docker-in-LXC nesting validated day-0** → langfuse repo cloned at the pinned
release tag → secrets generated **once** and persisted (all `# CHANGEME`) → headless init with
pinned project keys → `docker compose up -d` → Caddy TLS (443 only) → `AUTH_DISABLE_SIGNUP=true`
→ backup cron installed → **health check** (`/api/public/health` over HTTPS) → prints
`LANGFUSE_HOST` + keys for the laptop `.env`. Already-satisfied steps are detected and skipped.

*Ops thereafter:* `GET /api/v1/vps/:id/stats` (CPU/mem/disk watch), `start/stop/restart`,
`reinstall`, `change-password`, `upgrade` — all API, evidence-scriptable. The provision script is
**mock-tested (fake API + fake SSH transcript)** like `extract_session` — CI never spends money.

**Two verified caveats:** (a) Essential VPS is **LXC**, so day-0 bootstrap validates
Docker-in-LXC (nesting) *before* the Langfuse compose-up — if blocked, escalate to AIC support /
a KVM-class product rather than fighting it; (b) the checkout body exposes **no term field** —
the 6/12-month discount may be dashboard-only; confirm at purchase, monthly otherwise.

**Deployment plan (B):** Ubuntu 24.04 on `essential-8gb`, **+ 4 GB swap** — a
deliberate right-size below the official 16 GB (that guidance targets production throughput; our
volume is a portfolio trickle; the known low-RAM failure mode is worker/ClickHouse OOM, hence the
8 GB floor + swap; upgrade path is vertical via the API). Pinned
release tag; every `# CHANGEME` secret rotated; **headless init** with pinned project keys
(reproducible from scratch; keys → `.env`); **Caddy TLS, port 443 only**
(`LANGFUSE_HOST=https://langfuse.<domain>`); MinIO :9090 stays internal (no multimodal uploads);
UFW + key-only SSH; `AUTH_DISABLE_SIGNUP=true` after init; `restart: unless-stopped`; **nightly
off-box backup** (pg_dump + ClickHouse BACKUP + MinIO sync → AIC storage or a private HF
dataset); disk watch. Upgrades = tag bump + `docker compose up --pull always`. Runbook + scripts
committed under `infra/langfuse/`.

**Consequences.** `LANGFUSE_HOST` points at the VPS (env-only change; the DEC-P3-6 wrapper and
keyless Tier-1 no-op behaviour are untouched); JarvisLabs GPU sessions trace to it over public
HTTPS; CLAUDE.md's "Langfuse Cloud free tier" line gets a reconciliation note referencing this
decision; cost added to the standing-evidence budget (**₹799/mo**, the only always-on spend
besides the P6 static surface it will eventually absorb; ~₹9,600/yr, or ₹8,629 on the 12-month
term if purchasable). `.env.example` gains `AICCLOUD_API_KEY` (dashboard-issued, SHA-256-hashed
server-side; never in code).

---

## 4. Test strategy

### Unit (Tier-1, every PR, no GPU, no DB, no network)

- `test_llm_chat.py` — `chat()` via `httpx.MockTransport`: tool-call parsing, multi-message
  round-trip, usage/latency capture; **off/error contract preserved** (connection refused →
  `status="off"` behaviour, never a crash — the P0 invariant extended to the chat path).
- `test_driver.py` — scripted mock model: bounded tool rounds; invalid emitted call → validation
  failure recorded + error fed back + loop continues; placeholder binding; multi-turn message
  assembly (turn 2 sees turn 1 context); transcript completeness.
- `test_generation_metrics.py` — pure scorers: DEC-P3-5 call/sequence scoring incl. benign-extra
  and wrong-order cases; intent exact-match per turn; slot micro-F1 with match_key normalization;
  **hallucinated-movie detector**: catches a seeded invented title, does not flag fuzzy variants
  of returned titles ("Papanaasam" vs "Papanasam"), respects the abstain path.
- `test_judge.py` — prompt hashing stability; judge client env-wiring + redaction;
  endpoint-agnostic wiring (same client against a mocked vLLM-style and frontier-style endpoint —
  proves the DEC-P3-1 A↔B config swap); rubric parse of malformed judge output (never crashes,
  records `judge_error`).
- `test_tracing.py` — wrapper no-ops with keys unset; spans emitted to a fake sink with keys set.
- `test_mlflow_log.py` — stamp completeness (every §6.1 field present) against a temp file-store.
- `test_prompt_artifacts.py` — frozen prompt/exemplar/taxonomy files hash-pinned; **exemplars
  disjoint from golden fixtures** (no fixture query substring appears in an exemplar).
- `test_langfuse_provision.py` — DEC-P3-7 bootstrap idempotency against a **fake AIC API + fake
  SSH transcript**: fresh-state run executes every step in order; re-run against a fully
  configured state is a **no-op** (every check short-circuits); partial-state run (e.g. Docker
  installed, Langfuse absent) resumes at the right step; instance-exists path never calls
  checkout; destructive paths require the explicit flag; no network/spend in CI.
- Schema round-trip: `GenerationRunArtifact` serialize/parse; `mode`/`serving` invariants
  (`dry_run` ⇒ latency/throughput null).

### Integration (Tier-1 with `make up`; DB + recorded artifacts, no GPU)

- Driver executes GS-08a end-to-end against the live Postgres graph + artifact retriever with the
  scripted mock model: all five v0 tools exercised, results conform to `tool_schema.v0.json`
  result shapes (reusing the P1 round-trip helpers).
- Golden validation: expanded GS-02/07/08 fixtures pass `build_golden.py` (golden-eligibility
  gate) and `test_golden_expected_tool_calls_validate` (`checked` count raised to cover the new
  fixtures).
- MLflow compose service: log-and-read-back a run (marked integration).

### Tool-call ↔ TOOL_SCHEMA validation (required by the phase charter)

- `test_emitted_tool_calls_validate.py` — the DEC-P1-8 validator applied to **model-emitted**
  calls: a seeded transcript containing (a) a hallucinated tool name (`lookup_movie`), (b) a
  hallucinated parameter (`get_versions(country=…)`), (c) a wrong-typed argument — all three must
  be caught, scored as schema-validity failures, and appear in the artifact's violations list.
  Conversely, every call in the committed dry-run artifact that executed successfully must
  validate. (This is the "no hallucinated tool or parameter names" gate.)

### Named golden regression tests (Tier-1; recomputed from committed artifacts)

Per `GOLDEN_SET_SCENARIOS.md` — retrieval/graph gates stay green and the generation surface adds
its own layer:

| Named test | Layer | Status in P3 |
|---|---|---|
| version-set recall = 1.0 on **GS-01/GS-06** | retrieval (P2 artifact) | already gating; must stay green untouched |
| no-hallucinated-movie = 0 on **GS-02** | **generation** (new) | detector gate on the committed generation run; hard Tier-1 assert |
| dub-vs-remake on **GS-04** | graph labels (P1/P2 tests) | stays green; generation answers additionally checked: any GS answer surfacing Baahubali versions must carry `is_official_dub_of`, never `is_remake_of` (label pass-through check on transcripts) |
| sibling-vs-remake on **GS-05** | graph labels | stays green; same pass-through check |
| false-merge = 0 on **GS-10** | graph (P1) | stays green; driver-level check: a GS-10 conversation must end in disambiguation or two clearly-separated works, never one merged set |

### Eval set + metric thresholds (what gates this phase)

P3 is a *harness* phase: the base model's absolute scores are **measured, not gated** (they are
the baseline). What gates P3:

| Gate | Threshold |
|---|---|
| Retrieval regression suite (P2) | unchanged, green |
| No-hallucinated-movie on GS-02 (recorded run) | **0 inventions** |
| Emitted-call schema-validity accounting | every invalid call flagged; seeded-fault tests catch 3/3 fault classes |
| Judge–human agreement (validation sample, ~24 items) | **κ ≥ 0.6** for the frozen OSS judge (candidates compared on the same sample in one GPU session; one rubric revision allowed, then escalate to the frontier fallback per DEC-P3-1) |
| Dry-run completeness | all generation-slice fixtures execute end-to-end on the mock; every Table 2 metric computed (latency/throughput null); artifact committed + Tier-1 recomputation matches |
| Tracing/tracking evidence | ≥ 1 Langfuse trace link **on the self-hosted instance** + the committed trace export (JSON/screenshot) + MLflow runs (generation dry-run + Table 1 backfill) recorded in docs |

---

## 5. Task breakdown (ordered, independently committable) — **all 14 delivered (2026-07-03)**

> Executed in order, one commit per task (task 10 split into 10a/10b; live-infra fixes and the
> worksheet landed as follow-up commits). Delivery notes inline below.

1. **Deps + config + compose:** add `ragas`/`langfuse`/`mlflow`; Settings fields + `.env.example`
   (`MLFLOW_TRACKING_URI`, `JUDGE_*`, `GENERATION_RUN`); MLflow compose service (DEC-P3-2);
   `make mlflow-up`; settings tests.
2. **`LLMClient.chat()`** + `ChatResult`; MockTransport tests; off/error contract tests.
3. **Frozen prompt artifacts:** system prompt + intent taxonomy + 3 exemplars (DEC-P3-4) under
   `evals/prompts/`, hash-pinned; disjointness test.
4. **Golden schema extension + fixture expansion:** `expected_intent`/`expected_slots` fields;
   label GS-07/GS-08; author the new GS-02/07/08 fixtures (Q1 confirmed: **GS-07 → 5,
   GS-08 → 3, GS-02-conversational → 4** — ≈12–15 generation fixtures) with verify_sources;
   `build_golden.py` green.
5. **Pure metric scorers** (`sutradhar.evals.generation`): DEC-P3-5 tool-call scoring +
   placeholder binding; intent/slot scorers; hallucinated-movie detector; unit tests.
6. **Conversation driver** (`sutradhar.evals.driver`): validate→execute→feedback loop over the
   repository; transcript capture; unit + integration tests.
7. **Judge module** (`sutradhar.evals.judge`): OpenAI-compatible client, coherence rubric,
   hashing (DEC-P3-1); `judge` session in `infra/gpu/jarvis.py` (serve the pinned OSS judge +
   BGE-M3 via vLLM, DEC-P0-5 ephemeral pattern); `make judge-validate` producing the
   human-agreement worksheet + report.
8. **RAGAS adapter** (`sutradhar.evals.ragas_metrics`, DEC-P3-3); version pinned; tests with a
   faked judge/embeddings backend.
9. **Artifact model + runner CLI:** `GenerationRunArtifact`; `evals/run_generation_eval.py`;
   `make benchmark-generation` / `make generation-dryrun`.
10. **Observability:** **idempotent `make langfuse-up`** (`infra/langfuse/provision.py` per
    DEC-P3-7: find-or-create `essential-8gb` via the AIC API → check-then-act SSH bootstrap that
    sets up Langfuse from scratch if not installed — swap, Docker(+LXC nesting check), pinned
    tag, secrets-once, headless init, Caddy TLS, backups, health check; re-run safe, no spend in
    CI — fake API + fake SSH transcript tests); `sutradhar.obs.tracing` (DEC-P3-6) wired
    into driver/LLMClient/judge against `LANGFUSE_HOST`; `sutradhar.obs.mlflow_log`; Table 1
    MLflow backfill run; trace-export helper (JSON + screenshot committed with benchmark
    artifacts); tests.
11. **Mock endpoint + committed dry-run:** `evals/mock_llm.py` scripted behaviours (incl. the
    seeded hallucinated tool call + invented movie); run the dry-run; commit
    `evals/generation_runs/<run_id>.json`; pin `GENERATION_RUN`.
12. **CI:** `test_golden_generation_regressions.py` + `test_emitted_tool_calls_validate.py` in
    Tier-1; retire the tier1 placeholder step; replace `tier2.yml` with the real dispatch job;
    update `test_ci_workflows.py` meta-tests.
13. **Judge human-agreement session:** label the ~24-item sample (needs you); run the judge
    candidate(s) over the sample in **one short ephemeral GPU session** (≤ 1 h, < $1,
    create→judge→destroy); compute κ per candidate; freeze the winning judge config
    (repo + revision SHA + prompt hash); record the report in `evals/README.md`; add the judge
    model's licence to `docs/LICENSING.md`.
14. **Docs + decisions:** log DEC-P3-1..6 in `DECISIONS.md` (post-confirmation); TOOL_SCHEMA
    status note; `BENCHMARKS.md` Table 2 stamp fields tied to the artifact (base row stays
    "_populated at the top of the P4 GPU window via `make benchmark-generation`_" with the dry-run
    evidence linked); module READMEs; `PORTFOLIO.md` bullet.

---

## 6. Definition of Done (instantiates the CLAUDE.md generic DoD) — **ALL MET 2026-07-03**

- [x] Code complete and matches this approved spec (scope §1, design §2) — all 13 in-scope
      items delivered; deviations logged as DEC-P3-8 + DEC-P3-3/4/7 amendments (see Rev 6).
- [x] Unit + integration tests written and passing (all of §4) in Tier-1 CI — **455 unit +
      108 integration** (1 pre-existing skip); every §4-named test file exists and passes.
- [x] Eval thresholds met and recorded: §4 gate table (GS-02 = **0 inventions** on the recorded
      run; **3/3** seeded fault classes caught; **judge κ = 0.738 ≥ 0.6**; dry-run complete
      12/12; retrieval suite untouched and green) — generation dry-run (`c2fb0eab…`) + Table 1
      backfill (`26dc0470…`) logged to **MLflow**.
- [x] Benchmark tables: **Table 1** — numbers unchanged; reproducibility stamp completed with its
      MLflow run link. **Table 2** — column definitions, stamp fields, judge config
      (`gpt-oss-20b @ 6cee5e81…`, κ = 0.738), and prompt hash (`78215ccc…`) pinned; base row
      explicitly pending capture at the top of the P4 GPU window **by this harness**, with the
      committed dry-run artifact + exported Langfuse trace linked as machinery evidence. (No live
      generation numbers published from the mock.)
- [x] Module READMEs (`evals/`, `serving/`, `infra/`, `infra/langfuse/`) + `docs/DECISIONS.md`
      updated (DEC-P3-1..8 + dated amendments; TOOL_SCHEMA v0 status note added, no bump;
      `docs/LICENSING.md` gains gpt-oss-20b).
- [x] Runs cleanly from scratch: fresh clone + `.env` → `make up db-migrate seed-graph-ci
      generation-dryrun` with **no GPU and no external API keys** (`seed-graph-ci` proven to
      reproduce the exact graph state in a scratch DB; judge/RAGAS/tracing/MLflow all skip
      cleanly when unset).
- [x] 30-second demo path: `make generation-dryrun` — 12/12 scored transcripts incl. multi-turn
      GS-08, validated tool calls, both seeded faults caught, gate line printed, and (keys
      present) a live Langfuse trace URL printed.
- [x] Resume-ready quantified bullets drafted in `docs/PORTFOLIO.md` (harness + 0-hallucination
      gate; judge governance κ = 0.738; all-self-hosted MLOps with five live infra findings).

---

## 7. Open questions — RESOLVED (2026-07-02, user-approved: "Q1–Q5 proceed with recommendation")

| Q | Resolution |
|---|---|
| **Q1 — fixture sizing** | **Approved as recommended:** GS-07 → 5, GS-08 → 3, GS-02-conversational → 4 (≈12–15 generation fixtures). Logged under DEC-P3-4 consequences. |
| **Q2 — judge candidates** | **Approved:** gpt-oss-20b primary → Phi-4-14B alternate → frontier API escalation only; frozen by the κ ≥ 0.6 gate (DEC-P3-1). |
| **Q3 — human labelling** | **Approved:** user labels the ~24-item judge-validation sample (blocks the judge freeze; task 13). |
| **Q4 — base-capture placement** | **Approved:** authoritative base column captured at the top of the P4 GPU window by this harness; P3 publishes no live Table 2 numbers; no early GPU read. |
| **Q5 — Tier-2 artifact handling** | **Approved:** Tier-2 uploads the sealed run as a workflow artifact; a human commits it via PR (DEC-P2-6 posture). |
| **Q6 — Langfuse VPS** | **Approved with an added requirement:** `essential-8gb` (₹799/mo; ~₹1,000 wallet top-up); the provision script must be an **idempotent from-scratch bootstrap** — it sets up Langfuse from scratch if not installed and safely no-ops when already configured (DEC-P3-7 updated). Execution-time inputs still needed from the user: `AICCLOUD_API_KEY`, the two browser Razorpay payment clicks, and the custom domain (interim: `<ip>.sslip.io` for Let's Encrypt). |

### Original questions (as asked at grooming, for the record)

- **Q1 — Generation fixture sizing.** GOLDEN_SET targets ≥5 per category long-term; today the
  generation slice is GS-07 ×2, GS-08 ×1, GS-02 ×N(retrieval-shaped). For a stable Table 2 I
  propose expanding in P3 to: **GS-07 → 5** (add Kanglish/Telugu-English/native-script variants),
  **GS-08 → 3** (one non-Drishyam franchise, one with a NO_MATCH mid-turn), **GS-02
  conversational → 4** — ≈12–15 generation fixtures, each ground-truth-verified as in P1. OK, or
  do you want the fuller ≥5-per-category build-out now (more verification work)?
- **Q2 — Judge candidates (revised after in-depth research).** DEC-P3-1 now proposes, in measured
  order: **(1) gpt-oss-20b** (OpenAI family — fully disjoint from Google/Alibaba/Mistral; Apache
  2.0; ~14–16 GB → co-serves with BGE-M3 on the $0.89/hr A100 40 GB; documented judge usage);
  **(2) Phi-4-14B** (MIT) as alternate. Llama-3.3-70B was **dropped on VRAM evidence** (AWQ-INT4
  ≈ 47 GB weights — needs an 80 GB card once KV is added, breaking the DEC-0003 envelope for a
  judge). Frontier API remains only the documented escalation if neither candidate reaches
  κ ≥ 0.6 after one rubric revision. Confirm this order — or name a different OSS judge
  (constraint: not Google/Gemma, not Qwen, not Mistral-line, fits ≤ 40 GB).
- **Q3 — Human labelling effort.** Judge validation needs ~24 items labelled by you (coherence +
  faithfulness, ~30–45 min). Confirm you'll do this in P3 (it blocks the judge freeze).
- **Q4 — Base-capture placement (confirming my reading).** The authoritative base column is
  captured at the *top of the P4 GPU window* with this harness; P3 publishes **no** live Table 2
  numbers. Confirm — or do you want an optional short P3 GPU session for an early base read
  (≈1 h A100, ~$1)?
- **Q5 — Assumption (stated, not blocking).** Tier-2 uploads the sealed run as a workflow
  artifact and a human commits it via PR (same trust posture as DEC-P2-6), rather than CI
  auto-committing. I'll proceed with this unless you object.
- **Q6 — Langfuse VPS go-ahead (DEC-P3-7, now concrete via the live API).** The plan catalogue
  was fetched from `GET /api/v1/public/essential-vps-plans` (public, verified 2026-07-02):
  recommendation = **`essential-8gb` — 4 vCPU / 8 GB / 80 GB NVMe / 200 Mbps, dedicated IPv4,
  ₹799/mo** (cheapest tier *with* a public IPv4 — tiers below 8 GB have `dedicated_ipv4: false`
  and can't serve `LANGFUSE_HOST` at all). Confirm: (a) **top up the wallet with ~₹1,000**
  (covers ₹799 + ₹1 security fee + headroom; top-up min is ₹100; or ~₹8,700 if the 12-mo −10%
  term is buyable via dashboard); (b) create your `AICCLOUD_API_KEY` (dashboard → Settings → API
  Keys) for `.env`; (c) a domain/subdomain for TLS (`langfuse.<domain>` — AIC's domains API can
  register one, or any registrar); (d) OK to plan the P6 consolidation (static surface +
  optional read-only MLflow mirror on this same VPS) — noted now, decided in P6. Note: the
  Razorpay payment legs (top-up + checkout) are browser-only by design — the script prepares the
  order, you click pay.

---

## 8. Sources (web-research pass, accessed 2026-07-02)

- **RAGAS custom/self-hosted models:** Ragas docs — "Customise models" (`docs.ragas.io/en/stable/
  howtos/customizations/customize_models/`: `llm_factory`/`embedding_factory`, provider-agnostic)
  and "Bring Your Own LLMs and Embeddings" (`BaseRagasLLM`/`BaseRagasEmbeddings`); fully-offline
  RAGAS evaluation with a locally-served OSS model is a documented community pattern
  (jheiduk.com RAGAS tutorial, local Qwen2.5, "no API key required").
- **Open judges vs frontier judges:** JudgeBench (arXiv 2410.12784, ICLR 2025) — on *hard*
  reasoning-correctness judging even GPT-4o-class judges are near random → judge tier is
  task-dependent, and our narrow rubric-vs-recorded-context tasks are the easy regime;
  "Judge's Verdict" (arXiv 2510.09738) — 54-model judge tiering via human-agreement ("Turing test
  for judges"; correlation alone insufficient) — the methodology our κ-gated candidate selection
  mirrors; JudgeLM (ICLR 2025 Spotlight) — fine-tuned open judges exceed 90% agreement,
  surpassing human-to-human agreement on their benchmark.
- **Prometheus-2** (`github.com/prometheus-eval/prometheus-eval`) — purpose-built OSS evaluator,
  72–85% human agreement on pairwise benchmarks; **Mistral-based → excluded here** by the
  teacher-family independence rule, cited as evidence that open-weight judges reach usable
  agreement levels.
- **Judge-selection framework:** "Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks"
  (ICLR 2025) — open framework for comparing judge reliability/alignment.

**Rev-3 in-depth model-stack pass (accessed 2026-07-02):**

- **Base stack re-verification (DEC-0001/0002 — checked, not reopened):** `google/gemma-4-E4B` HF
  card; Google AI Gemma 4 model card ("License: Apache 2.0"); vLLM blog 2026-04-02 "Announcing
  Gemma 4 on vLLM" (Apache 2.0, agentic/function-calling); vLLM Recipes "Gemma 4 Usage Guide"
  (custom tool-use protocol served via OpenAI-compatible API); `Qwen/Qwen3-4B-Instruct-2507` HF
  card + QwenLM GitHub (2507 refresh: instruction-following/tool-usage gains); `sarvamai/sarvam-m`
  HF card + sarvam.ai blog (Apache 2.0 Mistral-Small base; +20% Indic, +86% romanized-Indic —
  the teacher rationale); BFCL v4 leaderboard (`gorilla.cs.berkeley.edu/leaderboard.html`) as the
  tool-calling reference benchmark.
- **Judge candidate evidence:** `openai/gpt-oss-20b` HF card + OpenAI vLLM serving cookbook
  (Apache 2.0; explicit vLLM support; ~13–16 GB VRAM per LocalLLMs/apxml specs); TruLens
  "OpenAI OSS Models as Judge" cookbook (documented judge usage); arXiv 2508.12461 "Is GPT-OSS
  Good?" (gpt-oss-20b matches/beats gpt-oss-120b on several benchmarks under standardized
  settings); Judge's Verdict (arXiv 2510.09738) full text — two-step r ≥ 0.80 → κ z-score
  methodology, 27/54 Tier-1, "judge excellence is not solely dependent on model size", human
  baseline κ = 0.801.
- **Llama-3.3-70B rejection evidence:** nodepedia `llama-3.3-70b-instruct-awq` (min ~47 GB for
  4-bit), willitrunai/fitmyllm VRAM guides (~49 GB Q4_K_M, ~57 GB recommended) — exceeds the
  48 GB Ada with KV cache; would force an 80 GB SKU for a judge-only job, violating the DEC-0003
  cost envelope.

**Rev-4 Langfuse-hosting pass (accessed 2026-07-02, DEC-P3-7):**

- **Langfuse self-hosting:** langfuse.com "Docker Compose Deployment" (official ≥ 4 cores/16 GiB/
  ~100 GB VM guidance; only web:3000 + minio:9090 need exposure; compose lacks HA, scaling, and
  backup functionality; `# CHANGEME` secrets; upgrade via `docker compose up --pull always`);
  Langfuse v3 architecture discussion (github langfuse/discussions/1902 — ClickHouse/Redis/
  MinIO all FOSS, no license key); langfuse headless-initialization + authentication docs
  (signup disable, init env keys); community v3 self-host guides (jangwook.net 2026; qaskills.sh
  2026) + the v3 pitfalls write-up (worker OOM, ClickHouse config — the basis for the 8 GB +
  swap floor).
- **AIC Cloud:** aiccloud.in `/developers` (API docs: base `https://api.aiccloud.in`, Bearer/
  X-API-Key auth, OpenAPI 3.1 at `/openapi.json`, 69+ endpoints, per-key rate limits);
  **live OpenAPI spec** (Essential VPS = Proxmox LXC; `POST /api/v1/vps/checkout` →
  Razorpay → `…/checkout/verify` with `sshKeys[]`; wallet endpoints
  `/api/v1/billing/wallet[/topup]`, min top-up ₹100; docs-page "Cloud VPS API" paths not in the
  spec and 404 live); **live plan catalogue** `GET /api/v1/public/essential-vps-plans` (verified
  2026-07-02: 18 tiers ₹99–₹3,199/mo; `essential-8gb` = 4 vCPU/8 GB/80 GB/₹799/mo, first tier
  with `dedicated_ipv4: true`; term discounts 6 mo −5% / 12 mo −10%); aiccloud.in `/vps` +
  `/pricing` (INR-first, monthly no-contract, Ubuntu 20.04–24.04, full root, anti-DDoS).
