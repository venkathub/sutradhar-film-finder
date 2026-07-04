# P4 Spec — QLoRA fine-tune on the rented GPU (the one-time job)

> **Status: EXECUTED COMPLETE (2026-07-04). VERDICT: CUT** — the frozen D8 rule, applied
> untouched to window `ftwin-ce6b6930`: 1/3 primaries improved (slot F1 +0.236), intent +
> coherence regressed, GS-02 failed on both columns. Table 2 published
> (`docs/BENCHMARKS.md`), verdict + execution amendments in DEC-P4-9, adapter public as a
> negative result, dataset private per D7, conditional follow-up scoped as ROADMAP P4.1.
> Cost: ≈ $13–14 actual vs the ≈ $10 Q6 cap (chronicle in DEC-P4-9).
>
> **Status at approval: APPROVED (2026-07-03, Rev 3).** Grooming complete — all §3 recommendations
> user-confirmed (§7 Q1–Q6 resolved 2026-07-03); decisions logged as **DEC-P4-1..8** in
> `docs/DECISIONS.md`. The §5 task list executes in order on the execution go-ahead; the D8
> verdict rule (incl. the ≥ +0.05 margin clause frozen under the Q3 delegation) is committed
> **before** the GPU window and cannot move after the numbers exist.
>
> **Rev 3 (2026-07-03) — approval:** Q1 Sarvam-M teacher confirmed · Q2 training slice
> confirmed, no franchise changes · Q3+Q4 delegated to the spec recommendations (D8-A **with**
> the one-metric ≥ +0.05 margin clause; D6-A identical-prompt headline + supplementary
> no-exemplar capture) · Q5 repo names confirmed (`sutradhar-gemma4-e4b-qlora-v1`,
> `sutradhar-ft-v1` private-first) · Q6 ≈ $10 GPU cap confirmed.
>
> **Rev 2 (2026-07-03) — web-research + local-docs verification pass (§8), same discipline as
> P3's Rev-3:**
> - **Bug fixed (local docs):** the Rev-1 D3 candidate training-franchise list collided with the
>   **negative sets** — Kaithi (GS-02b), Salaar (GS-02d), Pushpa (GS-02g + held-out NEG title
>   "Pushpa: The Rise"), Andhadhun (held-out NEG title). Ingesting those families would put
>   their titles inside the rapidfuzz-0.80 radius the negatives are *enforced* to be outside of
>   (`tests/integration/test_negatives_absent.py`) and would silently flip GS-02/θ-calibration
>   ground truth. D3 now carries a **structural exclusion rule** (test-enforced) and a corrected
>   candidate list; decontamination scope widened to golden ∪ exemplars ∪ **all negatives**.
> - **D1 hardware nuance corrected (vLLM docs):** the A100 is Ampere — **no native FP8 tensor
>   cores**; Sarvam-M "FP8" on the A100 40 GB runs as **W8A16 weight-only (Marlin kernels)**
>   (~24 GB weights + KV headroom). True W8A8 FP8 needs Ada/Hopper → the RTX 6000 Ada 48 GB
>   (~$0.99/hr, DEC-0003 value alternative) is the recorded plan-B SKU for the teacher session.
> - **D1 legal risk confirmed:** frontier ToS (OpenAI explicitly; Anthropic equivalent) restrict
>   using outputs to develop competing models, and provider access is revocable — strengthens
>   the Apache-2.0 self-hosted teacher recommendation.
> - **D5 sharpened (TRL docs/issues):** TRL ≥ 0.19 has **native tool-calling SFT support**
>   (`tools` in dataset/SFTTrainer) and `assistant_only_loss=True` via chat-template assistant
>   masks — but a **known silent-failure bug** discards the masks when `use_liger_kernel=True`
>   → liger is pinned OFF and mask application is test-asserted (§4).
> - **D2/D3/D4 evidence attached:** APIGen/xLAM + ToolACE (verifiable multi-stage synthetic
>   tool-call data ≈ our scaffold+validators posture); LIMA/LIMIT (~1k curated conversations
>   suffice for behaviour/format learning → the 1.5–2.5k target is evidence-based, not vibes);
>   QLoRA paper + Lightning-AI/Unsloth hyperparameter guidance (all-linear targets > attn-only;
>   α = 2r; r=16 standard operating point).
> - **§2.4 serving note:** the QLoRA column serves the **merged** model because vLLM adapter
>   serving adds per-request LoRA overhead — merging keeps the tokens/sec column comparable to
>   base (zero-overhead serving), per vLLM LoRA docs.
>
> **Entry criteria (all met, verified against the repo):**
> - P3 EXECUTED COMPLETE (2026-07-03): harness built + dry-run green
>   (`evals/generation_runs/20260703T012339Z-e7fff041.json`, 12/12 fixtures, seeded faults 3/3
>   caught); judge FROZEN (`openai/gpt-oss-20b @ 6cee5e81…`, κ = 0.738 ≥ 0.6, DEC-P3-1); prompt
>   bundle frozen (`prompt_hash 78215ccc…`, DEC-P3-4); Table 2 column definitions frozen
>   (P3_SPEC §2.4 / DEC-P3-5).
> - TOOL_SCHEMA **v0 FROZEN** (DEC-P1-8) with machine-readable `tool_schema.v0.json` + three CI
>   conformance layers; all five tools implemented.
> - P2 exit gate passed: Recall@10 = 1.000, VSR GS-01/GS-06 = 1.0 (run
>   `20260702T135315Z-f6583183`, pinned; DEC-0002 accepted) — **the green light for P4**.
> - Golden generation fixtures ready: GS-07 ×5, GS-08 ×3, GS-02-conversational ×4 (12 fixtures,
>   `expected_intent`/`expected_slots`/`expected_tool_calls` labelled, golden-gate validated).
>
> **ROADMAP P4 charter:** in a **single GPU window** — (1) capture the authoritative **base**
> generation column with the P3 harness, (2) **QLoRA fine-tune Gemma 4 E4B** on a synthetic,
> record-grounded, code-mixed multi-turn + tool-calling dataset, (3) capture the **QLoRA** column
> under identical vLLM config — then push artifacts to HF Hub, **STOP the GPU**, and log the
> **honest verdict** (GS-07/GS-08 must beat the base, or the adapter is cut and the finding
> documented). Fine-tuning owns **behaviour, not facts** (CLAUDE.md): the catalog stays in
> retrieval; nothing in this phase touches Table 1.

---

## 1. Scope

### In scope

1. **Synthetic training dataset** (`sutradhar.finetune.dataset` + `finetune/` CLIs): multilingual
   / code-mixed (Tanglish, Hinglish, Kanglish, Tenglish, native scripts), **multi-turn**,
   **tool-calling** conversations targeting the **frozen TOOL_SCHEMA v0** and the frozen intent
   taxonomy / INTENT-preamble / bold-title contracts (DEC-P3-4 + amendments) — so the model is
   trained on *exactly* the surface the harness scores.
   - **Grounded in real graph records** (gate-visible Work/Version/edge rows only): tool calls,
     tool results, and every asserted film title in every training answer are constructed from
     the live graph — **no invented films by construction** (enforced by validators, §4).
   - Generated as **deterministic scaffolds** (programmatic sampling of graph records → tool-call
     sequences → ground-truth tool results → labelled answers) with a **teacher pass** for
     linguistic surface realization only (code-mixed user utterances, register-matched answer
     prose) — teacher per §3 D1, method per §3 D2.
   - **Documented + versioned**: dataset card (counts, mix, snapshot hashes, teacher config,
     seed, decontamination report, licensing notes) + sha256; hosted per §3 D7; a compact sample
     committed in-repo for CI.
2. **Training-slice graph expansion (entity disjointness, §3 D3):** ingest ~10–15 additional
   *non-golden* remake/dub/sibling/collision franchises through the **existing P1 pipeline and
   verification gate** (unchanged code, new slice config) so training entities are disjoint from
   the golden-fixture entities and the before/after measures behaviour transfer, not
   memorization. Golden fixtures and the P2 index are untouched.
3. **QLoRA training** (`finetune/train_qlora.py`, `sutradhar.finetune.train` config models):
   4-bit NF4 QLoRA of **Gemma 4 E4B** (fallback Qwen3-4B-Instruct-2507 — DEC-0001, only on a
   recorded blocker) with PEFT + TRL per §3 D4/D5; chat-template rendering with **assistant-only
   loss masking** and native Gemma function-call tokens; train/val split with val-loss checkpoint
   selection; all hyperparameters a hashed config artifact.
4. **Reproducible GPU environment (ROADMAP §6.6):** pinned pip set (torch / transformers / PEFT /
   TRL / bitsandbytes / vLLM exact versions) in the session startup script — the authoritative-
   pins-in-script pattern established by DEC-P2-7 — plus a captured `pip freeze` + CUDA/driver/
   image versions in the run stamp, so QLoRA numerics repeat from scratch.
5. **The single GPU window** (`finetune_session` in `infra/gpu/jarvis.py`, DEC-P0-5 ephemeral
   pattern; HF-relay transport per DEC-P2-7), strictly ordered:
   1. serve base via vLLM (P0-validated config) → **base column captured live** by
      `make benchmark-generation` (the P3 harness, byte-identical scorers);
   2. stop vLLM → **QLoRA train** (dataset via HF relay; metrics/loss curves relayed back);
   3. **merge adapter** → serve merged model with **byte-identical vLLM config** → **QLoRA
      column captured**;
   4. (per §3 D6) supplementary QLoRA capture without few-shot exemplars;
   5. swap-serve the **frozen judge** (gpt-oss-20b + BGE-M3, DEC-P3-1/P3-3) → batch judge +
      RAGAS pass over **both** columns' recorded transcripts in the same session;
   6. push adapter + tokenizer/config + metrics + sealed run artifacts to **HF Hub** →
      laptop pulls + MANIFEST-verifies → **instance DESTROYED**.
   Budget: ~4–8 h on the A100 40 GB workhorse ≈ **$4–8** (inside the DEC-0003 envelope).
6. **AFTER benchmark WITH EVIDENCE:** `docs/BENCHMARKS.md` Table 2 — base and QLoRA columns from
   the same window, same fixtures, same serving config, same judge; MLflow runs (experiment
   `sutradhar/generation`) + **model registered in the MLflow registry**; Langfuse traces
   (exported JSON + screenshot committed, DEC-P3-7 posture); GPU latency p50/p95 + tokens/sec
   both columns.
7. **Honest verdict logged** (DECISIONS entry): the §3 D8 keep/cut rule applied to GS-07
   intent/slot and GS-08 coherence with no-regression guards; if QLoRA does not win, the adapter
   is cut, the finding documented, and P5 proceeds on the well-prompted base — pre-committed
   here, before results exist.
8. **CI/repo integration:** Tier-1 gains dataset-integrity tests (schema-valid tool calls, no
   invented titles, decontamination) over the committed sample + card; the pinned
   `GENERATION_RUN` flips to the live window artifact so
   `test_golden_generation_regressions.py` gates on real base-vs-QLoRA numbers between windows;
   `finetune/README.md`, `docs/LICENSING.md` (teacher + dataset rows), `docs/PORTFOLIO.md`
   updated.

### Non-goals (explicit — prevents scope creep)

- **No retrieval changes, no re-embedding.** Embedder unchanged (BGE-M3, DEC-0002 accepted) and
  golden-slice records unchanged since P2 → the ROADMAP's "re-embed only if…" condition is not
  triggered. The training-slice ingestion (D3) adds rows to the graph but **not** to the P2
  chunk index or any retrieval fixture; Table 1 and run `20260702T135315Z-f6583183` stay
  byte-identical (Tier-1-enforced).
- **No golden-set changes.** Both Table 2 columns are measured on the 12 frozen generation
  fixtures; expanding the eval set mid-phase would break base/QLoRA comparability.
- **No TOOL_SCHEMA change.** Synthetic data targets frozen v0 (status note only, like P2/P3 —
  no version bump, no DECISIONS entry for the schema).
- **No prompt re-engineering.** The frozen bundle (`prompt_hash 78215ccc…`) is the base column's
  contract; the only sanctioned variation is the D6 exemplar question DEC-P3-4 explicitly
  deferred to P4.
- **No judge changes, no re-validation.** DEC-P3-1's frozen config scores both columns.
- **No DPO/RLHF/preference tuning, no continued pretraining** — SFT (QLoRA) only. If SFT loses,
  the verdict is recorded; we do not escalate method complexity inside this phase.
- **No GGUF quantization.** Optional portable fallback only (CLAUDE.md); nothing in the DoD
  needs it; revisit only on explicit request.
- **No serving/API/orchestration work** — P5 (which also owns the optional-Java gateway
  decision). The vLLM serving here is transient benchmark plumbing, not the product path.
- **No catalog-breadth programme.** D3 ingests a bounded training slice through existing
  pipeline code; scaling the catalog remains post-P6 ops (ROADMAP §6.6).
- **No always-on anything.** The window ends with the instance destroyed; standing evidence is
  the committed artifacts.

---

## 2. Design

### 2.1 Component breakdown

| Component | Module (new unless noted) | Responsibility |
|---|---|---|
| Dataset schema | `sutradhar.finetune.dataset` | Pydantic models (`TrainingConversation`, `TrainingMessage`, `ToolCallRecord`, `DatasetCard`); JSONL (de)serialization; sha256/card |
| Scaffold generator | `sutradhar.finetune.scaffold` | Deterministic (seeded) sampling of gate-visible graph records → behaviour-classed conversation skeletons: user-slot plans, v0 tool-call sequences, ground-truth tool results, labelled answers with INTENT preamble + bold-title contract |
| Teacher client + surface pass | `sutradhar.finetune.teacher` | OpenAI-compatible client (`TEACHER_BASE_URL/MODEL/API_KEY` — DEC-P0-4 contract, endpoint-swappable); rewrites **only** user utterances + assistant prose around locked entity placeholders; raw outputs cached as a versioned artifact |
| Dataset validators | `sutradhar.finetune.validate` | Tool-call schema validation (reuses the DEC-P1-8 validator), invented-title detector (reuses the P3 detector), golden/exemplar decontamination, mix quotas, placeholder-integrity, card/hash determinism |
| Render + masking | `sutradhar.finetune.render` | Gemma chat-template rendering with native function-call tokens; assistant-only label masking; token-length stats |
| Train config + script | `sutradhar.finetune.train` + `finetune/train_qlora.py` | Hashed `TrainConfig` (D4/D5); the self-contained on-box training script (loads base 4-bit NF4, PEFT LoRA, TRL SFT, val-loss checkpointing, merge, HF push) |
| GPU window driver | `infra/gpu/jarvis.py` (extend: `finetune_session`) | Ephemeral create → base capture → train → after capture(s) → judge pass → push → destroy; teardown in `finally` + `gpu-nuke` coverage; HF-relay transport (DEC-P2-7) |
| Verdict | `sutradhar.finetune.verdict` + `make ft-verdict` | Pure function over two `GenerationRunArtifact`s → keep/cut verdict per D8; printed table for the 30-second demo |
| Benchmark reuse (existing) | `sutradhar.evals.*`, `evals/run_generation_eval.py` | **Unchanged** — both columns captured by the P3 harness/scorers byte-identically |
| CI | `tests/test_ft_dataset_*.py`, `test_golden_generation_regressions.py` (re-pin) | Tier-1 dataset-integrity + live-run regression gates |

### 2.2 Data models

**Training conversation (JSONL rows; pydantic, `extra="forbid"`):**

```python
class ToolCallRecord(BaseModel):
    tool: str                      # must exist in tool_schema.v0.json
    arguments: dict[str, Any]      # must validate against the v0 params schema

class TrainingMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None            # assistant final answers carry the INTENT preamble line
    tool_calls: list[ToolCallRecord] | None = None   # assistant tool-calling turns
    tool_result: dict | None = None                  # tool role: v0 result shape, built from graph rows

class TrainingConversation(BaseModel):
    conv_id: str
    behaviour: Literal["find_by_plot", "find_by_title", "list_versions",
                       "refine", "disambiguate", "out_of_catalog"]   # = frozen intent taxonomy
    query_lang: str                # ta-latin | hi-latin | kn-latin | te-latin | ml-latin | native | en
    turns: list[TrainingMessage]
    entity_ids: list[str]          # grounded work/version ids ([] for out_of_catalog)
    intent_labels: list[str]       # per user turn (mirrors golden expected_intent)
    slot_labels: list[dict]        # per user turn (frozen slot vocabulary only)
    scaffold_hash: str             # deterministic skeleton provenance
    teacher: TeacherStamp | None   # model+revision/prompt-hash of the surface pass (None = scaffold-only)

class DatasetCard(BaseModel):
    dataset_id: str                # e.g. sutradhar-ft-v1
    counts: dict                   # per behaviour × language
    graph_snapshot: str            # snapshot hashes the scaffolds were sampled from
    teacher: TeacherStamp | None
    seed: int
    decontamination: DecontReport  # max fuzzy similarity vs golden queries, exemplars, AND all negative surfaces (GS-02 + held-out) + threshold
    split: dict                    # train/val counts, split seed
    licenses: list[str]            # provenance notes (Wikidata CC0, TMDB attribution, IMDb NC caveat, teacher terms)
    sha256: str                    # of the canonical JSONL
```

**Design invariants (validator-enforced, §4):**
- Every `ToolCallRecord` validates against `tool_schema.v0.json` (params, types, enums) — the
  training data can never teach a hallucinated tool or parameter.
- Every `tool_result` conforms to the v0 result shape and is derived from gate-visible graph rows
  (relationship labels, `is_original`, `sources[]`, `confidence` all real). `out_of_catalog`
  conversations carry an honest `{results: [], abstain: true}` result and a NO_MATCH answer.
- Every film title asserted in any assistant answer resolves (match_key + rapidfuzz ≥ 0.80,
  DEC-P1-5) to that conversation's tool results — the *same* detector that gates GS-02.
- Answers follow the frozen formatting contracts: INTENT preamble on the final prose answer of
  each user turn; asserted titles in `**bold**` and nothing else bold (DEC-P3-4 amendments).
- The teacher pass may not alter entities: titles/years/languages/ids are placeholder-locked
  (`⟦T1⟧`-style sentinels substituted after the teacher call); a diff guard rejects any sample
  where locked spans changed.

**Behaviour/language mix (quota-checked; final size per §3 D3):**

| Behaviour class | Share | Notes |
|---|---:|---|
| `find_by_plot`, code-mixed + native-script | 30% | the GS-07 headroom target; plot descriptions paraphrased from real plot text |
| `find_by_title` incl. fuzzy/transliterated titles | 15% | GS-11-style perturbations generated from `version_title` rows |
| `list_versions` (+ `include_sequels` franchise walks) | 10% | version sets with correct remake/dub/sequel labels |
| multi-turn `refine` / backtracking (2–4 turns, incl. corrections) | 25% | the GS-08 behaviour: `refine_filter` over a standing version set |
| `disambiguate` (title/actor collisions) | 5% | ask-back, never merge |
| `out_of_catalog` / NO_MATCH (incl. mid-conversation) | 15% | the guardrail behaviour; abstain + offer to refine, zero inventions |

Language spread across classes: ≥ 40% code-mixed romanized (ta/hi/kn/te/ml-latin), ≥ 10% native
script, remainder English — matched to the GS-07 slice definition but on **disjoint entities**
(D3) and decontaminated query surfaces.

### 2.3 Data flow — dataset build (laptop + one teacher pass)

```
gate-visible graph (live views; training slice per D3)
        │  seeded sampler
        ▼
scaffold generator ──► conversation skeletons
   (tool sequences + real tool results + labelled ground-truth answers, entities locked)
        │
        ▼
teacher surface pass (TEACHER_* endpoint; §3 D1/D2)         [only neural step; not on the laptop’s
   user utterances → code-mixed/native register              silicon — a remote endpoint per
   answer prose → user’s register, entities locked            compute placement §2 of ROADMAP]
        │  raw outputs cached (versioned artifact)
        ▼
validators: v0 schema · invented-title detector · placeholder integrity ·
            decontamination vs golden+exemplars+negatives · quotas · card + sha256
        │  rejects are logged with reasons (teacher rejection rate reported on the card)
        ▼
sutradhar-ft-v1.jsonl (+ train/val split) ──► hosted per D7; sample + card committed
```

### 2.4 Data flow — the GPU window (one rental, strict order)

```
pre-flight (laptop): dataset sealed · rehearsal dry-run green · Tier-1 green · pins locked
        │  make gpu-finetune
        ▼
JarvisLabs A100 40GB created (ephemeral; teardown in finally)
  [1] vllm serve gemma-4-E4B (P0-validated flags, --chat-template)
        └─ laptop: make benchmark-generation → BASE run artifact (live, 12 fixtures)   ~20 min
  [2] stop vLLM → train_qlora.py (pinned pips; dataset via HF relay;
        NF4 QLoRA per D4/D5; val-loss checkpoint; loss curves relayed)                 ~1–3 h
  [3] merge adapter → vllm serve merged model (byte-identical flags)
        └─ laptop: make benchmark-generation → QLoRA run artifact                      ~20 min
  [4] (D6) supplementary capture: QLoRA under the no-exemplar prompt variant           ~15 min
  [5] swap-serve judge (gpt-oss-20b @ 6cee5e81… + BGE-M3)
        └─ batch judge + RAGAS over BOTH columns' transcripts (same judge, same session) ~30–45 min
  [6] push to HF Hub: adapter (+tokenizer/config) · training metrics · sealed artifacts
        └─ laptop pulls + MANIFEST-verifies → instance DESTROYED                        (≈ $4–8 total)
        ▼
laptop: commit artifacts · MLflow runs + registry entry · Table 2 both columns ·
        make ft-verdict → DEC-P4 verdict entry · re-pin GENERATION_RUN
```

**Contingency (recorded, not improvised):** if the window dies mid-run, artifacts already
relayed to HF Hub survive; a fresh window **recaptures the base column too** (cheap, ~20 min) so
both columns always share one instance/serving config — the "identical serving conditions"
guarantee is per-window, never spliced across windows.

### 2.5 Benchmark & verdict definitions

- **Metrics:** frozen — P3_SPEC §2.4 / BENCHMARKS.md column definitions, byte-identical scorer
  code, both columns. Nothing is redefined in P4.
- **Fairness invariants:** same 12 fixtures; same frozen prompt bundle (`78215ccc…`) for the base
  column and the headline QLoRA column (per D6 recommendation); same retrieval replay
  (DEC-P3-8, run `20260702T135315Z-f6583183`); same judge + rubric; same vLLM flags/decode
  params; latency/throughput reported for both (**merged adapter, deliberately:** vLLM's
  per-request LoRA path adds serving overhead, so merging keeps the tokens/sec column an
  apples-to-apples architecture comparison — a >10% divergence flags a serving-config drift
  bug, not a result).
- **Verdict rule:** per §3 D8 (confirmed margins in §7 Q3), applied by `make ft-verdict` as a
  pure function over the two committed artifacts — the keep/cut decision is computed, not
  narrated. Small-n honesty: with 12 fixtures (~30 scored turns) every aggregate is reported as
  an exact fraction with per-slice counts; no significance theater.

### 2.6 Tool-schema conformance statement

This phase **emits training data against** and **evaluates calls against** frozen v0 — it needs
**no new or changed tool**. Outbound: scaffold tool sequences are generated *from*
`tool_schema.v0.json` (never hand-written). Inbound: every training-data call is validated by the
DEC-P1-8 validator in Tier-1, and every model-emitted call in both benchmark columns is validated
by the live P3 harness path. `TOOL_SCHEMA.md` gets a status-note line only ("v0 targeted
unchanged by P4 synthetic data + before/after benchmark; v0 sha256 in the dataset card and both
run stamps") — no version bump, nothing schema-related for DECISIONS.md.

### 2.7 Compute placement (ROADMAP §2, strict)

Laptop/CI: scaffold generation (DB + string math), validators, rendering/masking tests (tokenizer
only — no model weights loaded), verdict, artifact plumbing. GPU window: training, both benchmark
captures, judge/RAGAS pass. Teacher: a remote endpoint (frontier API or a Sarvam-M ephemeral
session per D1) — never laptop inference. CI never calls a model (DEC-P2-6 posture throughout).

### 2.8 Python vs (optional) Java

**All Python.** The entire phase lives in the HF ecosystem (PEFT/TRL/bitsandbytes/vLLM) and glue
over existing Python surfaces (pydantic, SQLAlchemy repository, the P3 harness). CLAUDE.md's
optional Java moat is explicitly a **P5-grooming** decision about the public gateway; a JVM adds
zero signal to a training phase and would fork the single-lockfile reproducibility story
(DEC-P0-1). Note: training deps (torch/TRL/bitsandbytes) are **GPU-side pins in the startup
script** (DEC-P2-7 precedent) — the laptop lockfile stays neural-free.

### 2.9 What changes where (repo)

```
.env.example                       + TEACHER_BASE_URL, TEACHER_MODEL, TEACHER_API_KEY,
                                     HF_ADAPTER_REPO, FT_DATASET_REPO, FT_DATASET_ID
src/sutradhar/finetune/            new package: dataset, scaffold, teacher, validate,
                                     render, train (config models), verdict
finetune/train_qlora.py            self-contained on-box training script (relay-shipped)
finetune/build_dataset.py          Typer CLI: scaffold → teach → validate → seal
infra/gpu/jarvis.py                + finetune_session (and, if D1=Sarvam-M, a teacher session)
data-pipeline/ (config only)       training-slice seed list (D3) through existing ingestion
evals/generation_runs/             + live base + QLoRA (+ no-exemplar) artifacts, traces
tests/test_ft_{dataset,scaffold,teacher,render,train_config,session,verdict}.py   new
tests/test_golden_generation_regressions.py    re-pinned to the live GENERATION_RUN
Makefile                           + build-ft-scaffold, teach-dataset, validate-dataset,
                                     ft-dryrun, gpu-finetune, ft-verdict
docs/{BENCHMARKS.md, DECISIONS.md, LICENSING.md, PORTFOLIO.md, phases/TOOL_SCHEMA.md}
finetune/README.md                 rewritten from stub (architecture, runbook, results)
```

---

## 3. Decisions — CONFIRMED 2026-07-03 (logged as DEC-P4-1..8 in `docs/DECISIONS.md`)

> Settled and consumed as-is: base/fallback/teacher-role models (DEC-0001), GPU SKU + cost
> envelope (DEC-0003), retrieval stack + replay (DEC-P2-1..6, DEC-P3-8), judge (DEC-P3-1),
> prompt freeze + formatting contracts (DEC-P3-4), scoring rules (DEC-P3-5), tracing/MLflow
> (DEC-P3-2/6/7), tool schema v0 (DEC-P1-8). The eight below are the residue P4 actually owns.

### D1 — Teacher execution: finalize the DEC-0001 disjunction ("Sarvam-M 24B *or* a frontier API")

| Option | Trade-offs |
|---|---|
| **(A) Sarvam-M 24B self-hosted, 8-bit weight-only on the ephemeral GPU** — **recommended** | The model DEC-0001 named for exactly this job (+86% romanized-Indic — the code-mix register is its documented strength); keeps the **fully self-hosted, zero-external-key** story P3 established for the judge; **Apache 2.0 → teacher outputs are unencumbered** (HF card confirms), so the dataset (and adapter trained on it) can be published cleanly; cost ≈ 1–2 h ≈ $1–2. **Hardware (corrected by the §8 research pass):** the A100 is Ampere — no native FP8 tensor cores; on the A100 40 GB the quantized path is **W8A16 weight-only FP8 via vLLM's Marlin kernels** (~24 GB weights + KV headroom — tight but serviceable at batch-small); if headroom or kernel support disappoints at bring-up, the recorded plan-B SKU is the **RTX 6000 Ada 48 GB (~$0.99/hr**, DEC-0003's named value alternative, native FP8**)**. Cons: output quality ceiling below frontier; small-batch throughput (fine for ~2k short rewrite prompts). |
| (B) Frontier API (Claude/GPT/Gemini class, pinned dated version) | Highest fluency ceiling; no GPU session. Cons: external key + spend; **provider ToS restrict using outputs to develop competing models** (OpenAI Terms of Use explicitly; Anthropic equivalent — §8 refs), and access is revocable mid-project (the 2025 Anthropic→OpenAI API revocation is the documented precedent) — a real licensing/dependency landmine for a portfolio whose LICENSING.md is a maturity signal; weakens the self-hosted story; model-deprecation under the reproducibility stamp. |
| (C) No teacher — scaffold-only templated surfaces | $0, fully deterministic. Cons: templated code-mix is exactly the shallow register QLoRA would then overfit to; DEC-0001 names teacher data quality as *the* single biggest lever on whether FT beats base — cutting it kneecaps R2. |

**Recommendation: A**, with **B as the recorded escalation** (mirroring DEC-P3-1's pattern):
trigger = teacher-output QC failure (validator rejection rate > 30% or a code-mix quality
spot-check the user fails) after one prompt revision. Either way the client is OpenAI-compatible
and env-driven — A↔B is config, not code.

### D2 — Dataset construction method

| Option | Trade-offs |
|---|---|
| **(A) Programmatic scaffold + teacher surface-realization only (entities placeholder-locked)** — **recommended** | Grounding, tool-call validity, label correctness, and formatting contracts hold **by construction** (the teacher physically cannot invent a film or a tool); labels (intent/slots/expected calls) are free and exact; deterministic skeletons → reproducible under the stamp; teacher spend minimized (short rewrite prompts). This is the same verify-then-keep posture the strongest public tool-call datasets use — APIGen/xLAM's three-stage verification (format → execution → semantic) and ToolACE's accuracy-first pipeline (§8) — applied at construction time instead of filter time. Con: conversational shapes are bounded by the scaffold library — mitigated by 6 behaviour classes × turn-count/correction/register variation. |
| (B) Teacher free-generates whole conversations; post-hoc validation filters | Maximum diversity. Cons: labels must be *extracted* (second error surface); high rejection rates waste teacher tokens; grounding becomes a filter, not an invariant — the exact failure class (invented films) our headline gate punishes. |
| (C) Paraphrase-augment the 12 golden fixtures | Cheapest. Cons: direct leakage of the eval surface — indefensible before/after; tiny entity/behaviour coverage. |

**Recommendation: A.**

### D3 — Dataset size & entity strategy

| Option | Trade-offs |
|---|---|
| **(A) ~1,500–2,500 conversations on a dedicated *training slice* of ~10–15 non-golden franchises (ingested via the existing P1 pipeline + gate), entities disjoint from all golden fixtures **and structurally excluded from every negative surface**; 95/5 train/val** — **recommended** | Entity disjointness turns the before/after into a **behaviour-transfer** claim (the interview-grade version: "trained on Rowdy Rathore/Kabir Singh-class families, evaluated on Drishyam-class families it never saw"); size is evidence-based (LIMA/LIMIT: ~1k curated, diverse conversations suffice for behaviour/format learning at far larger models — §8) and small enough for 1–3 h QLoRA; training data needs only **gate-visible** (HIGH/MEDIUM) records, not golden-grade, so review burden is light. Cons: ~0.5–1 day of ingestion + human-gate work; +1 short extraction session only if Wikidata coverage of the training slice is thin (reuse DEC-P1-4 machinery). |
| (B) Same size, sampled from the existing golden-slice graph | Zero ingestion work. Cons: training and eval share entities (Drishyam conversations in training, Drishyam fixtures in eval) — query-level decontamination can't cure entity leakage; the lift becomes partially memorization and a reviewer will see it. |
| (C) 8–10k conversations (xLAM-style scale) | The public tool-call datasets (xLAM-60k) are built for *general* function-calling across 3,673 APIs; our surface is **5 frozen tools + 6 behaviours** — format saturates far earlier (LIMA finding); 3–4× teacher + training cost/time; slower iteration if the verdict forces a second look. |

**Recommendation: A**, with a **structural exclusion rule (test-enforced, found by the Rev-2
docs sweep):** no training-slice franchise may contain any film that appears in — or falls
within the rapidfuzz-0.80 radius of — (a) any golden fixture (incl. the GS-02 negatives:
**Kaithi, Salaar, Pushpa, Inception**…), (b) any held-out negative title
(`evals/negatives/heldout.yaml`: **Master, Pushpa: The Rise, Jailer, Dangal, Kantara,
Andhadhun, तुम्बाड, Kumbalangi Nights, Super Deluxe, Ratsasan, Interstellar, Parasite**), or
(c) the frozen exemplar franchises (**Ghajini, Okkadu/Ghilli, Interstellar** — exemplars are
prompt surface for BOTH Table 2 columns; training on them would asymmetrically favour the
QLoRA row). The Rev-1 candidate list violated (a)/(b)/(c) four ways and is **replaced**:

- **Corrected candidate training-slice families (each verified at ingestion, per the golden
  data-accuracy rule; final list confirmed at task 3 against the exclusion test):**
  Vikramarkudu→Rowdy Rathore (te→hi remake); Anniyan + dub Aparichitudu (remake-vs-dub
  contrast); Kanchana/Muni series (sequel edges; distinct from the golden Manichitrathazhu
  lineage — **anything in that lineage: Chandramukhi, Apthamitra, Bhool Bhulaiyaa, is golden
  and excluded**); Pokiri→Wanted (te→hi); Bodyguard (ml→hi remake chain); Arjun Reddy→Kabir
  Singh (te→hi); U-Turn (kn→multi-language remakes); Mersal/Bigil-class dub tracks
  (pan-Indian dub-vs-remake signal); a same-title collision pair for `disambiguate`; plus 2–3
  out-of-catalog decoy *themes* for NO_MATCH scaffolds (themes, not real films — decoys must
  ALSO clear the negative-set radius check).
- The exclusion rule is enforced by `test_ft_training_slice_disjoint` (§4) **before** any
  ingestion lands, and re-asserted by the existing
  `tests/integration/test_negatives_absent.py` after ingestion (the negatives' absence
  invariant now guards the training slice too — no new test semantics, a new population).

### D4 — QLoRA configuration (rank / targets / quantization)

| Option | Trade-offs |
|---|---|
| **(A) r=16, α=32, dropout 0.05, targets = all linear projections (attn q/k/v/o + MLP gate/up/down), NF4 double-quant, bf16 compute, LR 2e-4 cosine, 2–3 epochs (val-loss selected), max_seq 4096, packing off** — **recommended** | The QLoRA-paper finding (all-linear targets matter more than rank — attn-only measurably underperforms) at the standard operating point the practitioner literature converges on (α = 2r heuristic; r=16 default; Lightning-AI "hundreds of experiments" + Unsloth hyperparameter guides, §8); ~10–12 GB peak (DEC-0003 sizing) — huge headroom on 40 GB; adapter ~50–100 MB (clean HF push). Con: none material at this scale. |
| (B) r=8, attention-only targets | Smallest/fastest. Cons: attn-only is the documented under-performer for tool-format + register learning; savings are irrelevant on a 40 GB card. |
| (C) r=64, α=128 | More capacity than a 6-behaviour SFT needs; slower, marginally higher overfit risk on ~2k conversations (Lightning-AI notes r>16 mainly pays on large diverse corpora); no evidence it buys anything at 4B. |

**Recommendation: A.** All values live in the hashed `TrainConfig`; if val loss signals under/
over-fit at execution, the change is a recorded amendment (like DEC-P2 measured-winner notes),
not silent tuning.

### D5 — Training stack

| Option | Trade-offs |
|---|---|
| **(A) Plain TRL `SFTTrainer` + PEFT + bitsandbytes (exact pins in the startup script)** — **recommended** | Literally the CLAUDE.md-named stack ("Transformers / PEFT / TRL (QLoRA)"); **TRL ≥ 0.19 natively supports tool-calling SFT** (a `tools` column rendered through the chat template) and **`assistant_only_loss=True`** via chat-template-generated assistant masks (§8) — exactly our data shape, no custom collator; most reproducible (no monkey-patching layer between the pins and the numerics). **Two pinned guardrails from the research pass:** (1) `use_liger_kernel` stays **OFF** — a known TRL bug silently discards assistant masks under liger, i.e. loss over the whole sequence with no error; (2) mask application is **test-asserted** on rendered samples (§4 `test_ft_render_masking`), never assumed. Con: ~1.5–2× slower than Unsloth — at 1–3 h total, that is dollars, not days. |
| (B) Unsloth | 2× faster, lower VRAM (DEC-0001 sources cite it). Cons: it is explicitly a **runtime patching layer** over TRL/transformers trainer classes (its own docs, §8) — a moving layer under the "QLoRA numerics repeat" §6.6 promise; another dependency family to pin/debug in a one-shot window. |
| (C) Axolotl | Config-driven convenience. Cons: heavy config surface for one job; less direct portfolio signal than showing the PEFT/TRL code. |

**Recommendation: A**, with B as the recorded fallback **only** if a measured window overrun
threatens the DEC-0003 envelope.

### D6 — QLoRA-column prompt: exemplars in or out (the question DEC-P3-4 deferred to P4)

| Option | Trade-offs |
|---|---|
| **(A) Headline Table 2 QLoRA row under the IDENTICAL full frozen prompt (system + taxonomy + 3 exemplars); a supplementary no-exemplar capture recorded in the artifact + a BENCHMARKS footnote quantifying prompt-token savings** — **recommended** | Strictest apples-to-apples (one variable: the adapter); immune to the "you just changed the prompt" critique; the no-exemplar capture (~15 min marginal) still documents the production win (adapter internalizes the exemplars → shorter, cheaper prompts) and is P5's serving config if the adapter is kept. Con: headline undersells the token-cost benefit — recovered by the footnote. |
| (B) Headline QLoRA row = no-exemplar "production config" | The classic prompting-vs-FT framing; shows internalization at headline level. Cons: two variables change between the rows (adapter *and* prompt); prompt_hash differs across columns in the same table — muddies the frozen-stamp story. |
| (C) Same-prompt only, skip the no-exemplar capture | Simplest. Con: loses the cheap, high-signal internalization evidence for P5. |

**Recommendation: A.**

### D7 — Dataset & adapter hosting/versioning

| Option | Trade-offs |
|---|---|
| **(A) HF Hub: adapter repo (`…/sutradhar-gemma4-e4b-qlora-v1`) + dataset repo (`…/sutradhar-ft-v1`), both with cards; dataset PRIVATE at first (IMDb-derived AKA titles + teacher provenance reviewed before any public flip); in-repo: dataset card, sha256, ~100-conversation committed sample for Tier-1** — **recommended** | ROADMAP §6.1 names HF Datasets + cards; HF Hub is already the artifact registry (DEC-P2-7 relay exists); committed sample keeps CI secret-free and fork-safe (DEC-P2-6 posture); private-first respects the IMDb non-commercial caveat until LICENSING.md review clears publication. Con: full-dataset access needs a token — acceptable, CI never needs it. |
| (B) Commit the full JSONL to git | Zero infra. Cons: MBs of generated text in git history; licensing review happens *after* publication by construction. |
| (C) DVC | Adds a whole tool for one artifact family HF Hub already covers. |

**Recommendation: A.**

### D8 — The keep/cut verdict rule (pre-committed before results exist)

| Option | Trade-offs |
|---|---|
| **(A) KEEP iff: strict improvement on ≥2 of the 3 primary metrics {GS-07 intent accuracy, GS-07 slot F1, GS-08 coherence} AND no primary metric regresses AND all guards hold: GS-02 inventions = 0, schema-validity ≥ base, tool-call sequence accuracy ≥ base** — **recommended** | Matches the ROADMAP's "GS-07/GS-08 are the fixtures where QLoRA must beat the base"; no-regression guards prevent a lift bought by breaking grounding or tool discipline; realistic at n = 12 fixtures (all-3-strict makes one noisy judge score veto a real win). Con: "2 of 3" needs one sentence of explanation in the verdict entry. |
| (B) Strict improvement on every Table 2 metric | Maximally clean headline. Con: answer relevancy / latency are not what FT targets; near-ties at small n would force cutting a genuinely better adapter. |
| (C) Any single-metric improvement | Too weak — invites keeping an adapter on noise; fails the honesty bar. |

**Decision (user-confirmed 2026-07-03; margins frozen under the Q3 delegation): A, with the
margin clause.** The frozen rule, implemented verbatim in `sutradhar.finetune.verdict` before
the GPU window:

> **KEEP the adapter iff** (i) strict improvement on **≥ 2 of the 3 primary metrics**
> {GS-07 intent accuracy, GS-07 slot F1, GS-08 coherence}; **(ii) at least one improving
> primary metric clears ≥ +0.05 absolute** (so judge/small-n noise alone can never trigger a
> keep at n = 12 fixtures); (iii) **no primary metric regresses**; (iv) **all guards hold**:
> GS-02 inventions = 0 on both columns, schema-validity ≥ base, tool-call sequence accuracy
> ≥ base. Anything else → **CUT**, finding recorded (DEC-0001 pre-commitment).

The rule is logged as DEC-P4-8 and cannot move after the numbers exist.

---

## 4. Test strategy

### Unit tests (Tier-1, no GPU, no model calls)

| Test | Asserts |
|---|---|
| `test_ft_dataset_schema` | Pydantic round-trip; `extra="forbid"`; JSONL determinism; card sha256 stability |
| `test_ft_scaffold_grounding` | Every asserted title in every scaffold answer resolves to that conversation's tool results (reuses the P3 detector); every `tool_result` shape validates against v0 result schemas; every entity id exists in the (seeded test) graph |
| `test_ft_tool_calls_validate` | **Every training-data tool call validates against `tool_schema.v0.json`** (reuses the DEC-P1-8 validator) — no hallucinated tool or parameter names are *trainable*, by CI construction |
| `test_ft_teacher_lock` | Placeholder integrity: a (faked) teacher output that alters a locked entity span, adds a title, or drops the INTENT preamble is rejected with a logged reason |
| `test_ft_decontamination` | No training user-utterance within the fuzzy threshold of any golden query, frozen exemplar, **or negative query (GS-02 + `evals/negatives/heldout.yaml`)**; **training `entity_ids` ∩ golden-fixture entities = ∅** (D3); threshold + report on the card |
| `test_ft_training_slice_disjoint` | **D3 structural exclusion rule:** every candidate training-slice title falls outside the rapidfuzz-0.80 radius of all golden-fixture titles, all GS-02 negative titles, all held-out negative titles, and all exemplar-franchise titles — runs against the slice config *before* ingestion; post-ingestion, `tests/integration/test_negatives_absent.py` re-asserts the negatives' absence over the grown title index |
| `test_ft_mix_quotas` | Behaviour × language quotas within tolerance; NO_MATCH share present |
| `test_ft_render_masking` | Chat-template render uses native Gemma function-call tokens; labels masked everywhere except assistant tokens — **asserted on rendered token/mask arrays, not config** (guards the known TRL liger-kernel silent-mask-drop failure; `use_liger_kernel` pinned OFF in `TrainConfig`); preamble + bold-title contract survive rendering |
| `test_ft_train_config` | `TrainConfig` parse/hash; D4 values pinned; no model weights loadable on the laptop path (import-safe) |
| `test_finetune_session` | Fake-transcript session driver (DEC-P0-5 pattern): step ordering (base → train → after → judge → push → destroy), **teardown-on-injected-failure still destroys**, HF-relay calls mocked, token never echoed |
| `test_ft_verdict_rule` | D8 as a pure function: crafted metric pairs → keep/cut/guard-violation outcomes, incl. the "2-of-3 with a regression elsewhere" edge |

### Integration tests (local DB)

- Scaffold generation against the seeded graph is deterministic under a pinned seed (same hash
  twice); samples from **live gate views only** (a CANDIDATE-tier edge can never appear in
  training data — same layered-gate property as P1).
- `make ft-dryrun`: end-to-end mini-build (mock teacher) → validators pass → card + sample
  emitted → render/masking runs on the real tokenizer config — the committed rehearsal evidence.

### Regression tests (existing suites — must stay green untouched, named per the golden set)

- **Retrieval/graph untouched:** `test_golden_retrieval_regressions.py` — **version-set recall
  = 1.0 on GS-01 and GS-06**; Table 1 recomputed byte-identical from the pinned run.
- **Graph label suites:** **dub-vs-remake on GS-04**, **sibling-vs-remake on GS-05**,
  **false-merge = 0 on GS-10** — the D3 training-slice ingestion runs the same edge-typing
  tests over the *new* rows too (the pipeline's tests are slice-agnostic).
- **No-hallucinated-movie on GS-02** — recomputed from the committed generation artifacts.
- **Tool-schema conformance:** `test_golden_expected_tool_calls_validate` +
  `test_repository_matches_tool_schema` + md↔json sync — unchanged and green.

### Eval gates for this phase (the live window; recorded artifact gates CI afterwards)

| Gate | Threshold |
|---|---|
| GS-02 no-hallucinated-movie | **= 0 inventions on BOTH columns** (hard gate — a hallucinating adapter fails the phase regardless of lift) |
| Emitted-call schema validity (no hallucinated tool/param names, model-emitted, validated against `tool_schema.v0.json`) | QLoRA ≥ base; target 1.0 |
| Tool-call sequence accuracy (DEC-P3-5 headline) | QLoRA ≥ base (guard) |
| GS-07 intent accuracy / slot F1, GS-08 coherence | **the D8 verdict rule** — the phase's R2 question |
| Retrieval Table 1 | byte-identical (untouched — asserted, not re-earned) |
| Latency/throughput | reported both columns; >10% tokens/sec divergence = serving-drift investigation before publishing |

Both columns are captured by the **byte-identical P3 harness** (scorers, judge, retrieval replay
pinned) — P4 adds no scorer code, so the before/after cannot drift by construction.

---

## 5. Task breakdown (ordered, independently committable)

1. **Log confirmed decisions** — DEC-P4-1..8 in `docs/DECISIONS.md` (after user confirmation);
   `.env.example` + settings gain `TEACHER_*`, `HF_ADAPTER_REPO`, `FT_DATASET_*`.
2. **Dataset schema + card models** (`sutradhar.finetune.dataset`) + unit tests.
3. **Training-slice ingestion (D3):** run `test_ft_training_slice_disjoint` against the slice
   config (the structural exclusion rule) **before** ingesting; then ingest ~10–15 non-golden
   franchises through the existing `make ingest-seed` chain + human gate; graph label tests +
   `test_negatives_absent` green over the grown index; entity-disjointness fixture list
   committed.
4. **Scaffold generator** (`sutradhar.finetune.scaffold`) + grounding/quota/determinism tests —
   produces a complete scaffold-only dataset (teacher-independent milestone).
5. **Validators** (`sutradhar.finetune.validate`): v0 call validation, invented-title detector
   reuse, decontamination, placeholder lock + tests.
6. **Teacher client + surface pass** (`sutradhar.finetune.teacher`) with mocked tests; if
   D1 = Sarvam-M: `teacher` session in `infra/gpu/jarvis.py` (ephemeral, fake-transcript tested).
7. **Teacher run (one session / API pass):** realize the dataset; QC spot-check; seal
   `sutradhar-ft-v1` (card, sha256, split); push per D7; commit sample + card; Tier-1 dataset
   tests wired.
8. **Render + masking** (`sutradhar.finetune.render`) + tests.
9. **`finetune/train_qlora.py` + `TrainConfig`** (hashed; D4/D5 pins); laptop-side parse/dry
   tests; startup-script pins for the training container (§6.6) recorded.
10. **`finetune_session` window driver** in `infra/gpu/jarvis.py` (+ `make gpu-finetune`) with
    fake-transcript tests incl. injected-failure teardown; `make ft-dryrun` rehearsal committed.
11. **Verdict module** (`sutradhar.finetune.verdict` + `make ft-verdict`) with the frozen D8
    rule + tests — **committed before the window**.
12. **THE GPU WINDOW (one-time):** execute §2.4 — base capture → train → QLoRA capture(s) →
    judge/RAGAS both columns → HF push → destroy. Evidence captured live (MLflow, Langfuse
    exports, screenshots, tokens/sec).
13. **Publish + verdict:** commit sealed artifacts; re-pin `GENERATION_RUN`; MLflow registry
    entry for the adapter; **Table 2 both columns** in `docs/BENCHMARKS.md` with the full stamp;
    DEC-P4 verdict entry (keep or cut, with numbers); exact model revisions recorded in
    `.env.example` + BENCHMARKS (DEC-0001 follow-up discharged).
14. **Docs close-out:** `finetune/README.md` (architecture, runbook, results), `LICENSING.md`
    (teacher + dataset rows, IMDb-derived caveat), `docs/PORTFOLIO.md` quantified bullet,
    TOOL_SCHEMA status note, P4_SPEC execution close-out.

---

## 6. Definition of Done (instantiates the CLAUDE.md generic DoD)

- [ ] Code complete and matching this approved spec (tasks 1–14).
- [ ] Unit + integration tests written and passing (§4); Tier-1 fully green with the re-pinned
      live `GENERATION_RUN`.
- [ ] Eval thresholds met and recorded to MLflow: GS-02 = 0 inventions on both columns; schema
      validity + sequence accuracy guards hold; the D8 verdict computed and logged — **either
      outcome (keep or cut) satisfies the DoD if it is measured, evidenced, and logged**.
- [ ] Benchmark tables: **Table 2 updated with BOTH columns** (base + QLoRA: tool-call accuracy,
      code-mixed intent/slot accuracy, backtracking coherence, faithfulness, answer relevancy,
      GPU latency p50/p95 + tokens/sec) captured in one live GPU window with evidence — MLflow
      run links, exported Langfuse traces, screenshots, serving config + model revisions in the
      stamp. **Table 1 untouched and asserted byte-identical** (the two-table honesty rule).
- [ ] Adapter merged; adapter + dataset card + metrics + sealed run artifacts on **HF Hub**;
      adapter registered in the MLflow registry; **GPU instance destroyed** (volume deletable —
      `make gpu-nuke` clean).
- [ ] `finetune/README.md` + `docs/DECISIONS.md` (DEC-P4-1..8 + verdict) + `docs/LICENSING.md`
      updated.
- [ ] Runs cleanly from scratch: fresh clone + `.env` → `make ft-dryrun` (no GPU) reproduces the
      rehearsal; the window itself reproducible via `make gpu-finetune` (documented, priced).
- [ ] 30-second demo path: `make ft-verdict` — prints the base-vs-QLoRA per-metric table with
      the keep/cut verdict from committed artifacts, GPU off.
- [ ] Resume-ready quantified bullet drafted for `docs/PORTFOLIO.md` (e.g. "QLoRA-tuned a 4B
      model on N synthetic record-grounded code-mixed conversations for $X of GPU time; +Y pts
      intent accuracy / +Z pts slot F1 over a well-prompted base under identical serving,
      zero hallucinated films across the benchmark").

---

## 7. Open questions — RESOLVED (user-confirmed 2026-07-03)

- **Q1 (D1) — CONFIRMED: Sarvam-M.** Teacher = **Sarvam-M 24B self-hosted** (8-bit weight-only
  on the A100 40 GB; RTX 6000 Ada 48 GB recorded plan-B SKU); frontier API stays the recorded
  escalation only (outputs-ToS row precedes any use). → DEC-P4-1.
- **Q2 (D3) — CONFIRMED, no additions.** Entity-disjoint training slice under the structural
  exclusion rule, ~1,500–2,500 conversations; the §3 D3 corrected candidate list stands as-is
  (final membership verified at task 3 against `test_ft_training_slice_disjoint`). → DEC-P4-3.
- **Q3 (D8) — DELEGATED to the spec recommendation; frozen:** KEEP requires strict improvement
  on ≥ 2 of 3 primary metrics **with at least one clearing ≥ +0.05 absolute**, no primary
  regression, all guards (the margin clause makes noise-only keeps impossible at n = 12). Rule
  text in §3 D8. → DEC-P4-8.
- **Q4 (D6) — DELEGATED to the spec recommendation; confirmed:** headline QLoRA row under the
  **identical full frozen prompt**; no-exemplar capture as a supplementary artifact row +
  BENCHMARKS footnote. → DEC-P4-6.
- **Q5 (D7) — CONFIRMED:** adapter repo **`sutradhar-gemma4-e4b-qlora-v1`** (public at publish
  time); dataset repo `sutradhar-ft-v1` **private-first** pending the LICENSING review. →
  DEC-P4-7.
- **Q6 (budget) — CONFIRMED:** total-phase GPU cap ≈ **$10** (teacher ~$2 + window ~$4–8),
  inside the DEC-0003 envelope.

---

## 8. Research annex (Rev 2, accessed 2026-07-03) — what was verified, what changed

> Same discipline as P3_SPEC's Rev-3 pass: every load-bearing claim in §3 was checked against
> primary sources (and the repo's own committed artifacts) before asking for approval. Settled
> decisions were **re-verified, not reopened**.

### 8.1 Local-docs findings (the ones that changed the spec)

- **Training-slice / negative-set collision (D3 — Rev-1 bug, fixed).** `evals/README.md` +
  `evals/negatives/heldout.yaml` + `evals/golden/gs02_no_match.yaml` show the negative surfaces
  contain real, deliberately-uncatalogued films: GS-02 → *Kaithi, Inception, Salaar (d),
  Pushpa (g)*; held-out negatives → *Master, Pushpa: The Rise, Jailer, Dangal, Kantara,
  Andhadhun, तुम्बाड (Tumbbad), Kumbalangi Nights, Super Deluxe, Ratsasan, Interstellar,
  Parasite*. Their absence from the title index at the rapidfuzz-0.80 radius is **enforced** by
  `tests/integration/test_negatives_absent.py`, and θ (DEC-P2-5) was calibrated against it.
  The Rev-1 candidate list (Kaithi→Bholaa, Pushpa, KGF, Andhadhun→Maestro) would have ingested
  four of these families, silently invalidating GS-02 ground truth and the abstention
  calibration. §3 D3 now carries the structural exclusion rule + corrected list;
  `test_ft_training_slice_disjoint` runs before ingestion.
- **Exemplar franchises are prompt surface for both columns.** `evals/prompts/exemplars_v1.md`
  uses Ghajini / Okkadu-Ghilli / Interstellar; training on them would asymmetrically favour the
  QLoRA row (it sees them in training AND in-prompt). Excluded from the training slice
  (Rev-1 wrongly allowed franchise-level overlap).
- **LICENSING.md** already carries the Sarvam-M row ("Optional P4 synthetic-data teacher") and
  the IMDb non-commercial caveat that motivates D7's private-first dataset posture — the spec
  adds rows (teacher-outputs provenance, FT dataset, adapter), it does not introduce new policy.

### 8.2 Web findings folded into decisions

| Claim in spec | Verified against | Outcome |
|---|---|---|
| Sarvam-M 24B: Apache 2.0, Mistral-Small base, ~24 GB at 8-bit / ~14 GB at Q4 | HF `sarvamai/sarvam-m` card; sarvam.ai blog; GPU-sizing calculators | Confirmed; teacher outputs unencumbered (D1) |
| "FP8 on the A100" (Rev-1 wording) | vLLM quantization docs: **Ampere = W8A16 weight-only FP8 via Marlin kernels only**; W8A8 FP8 needs Ada/Hopper | **Corrected** — D1 now says 8-bit weight-only on A100, RTX 6000 Ada 48 GB as plan-B SKU |
| Frontier teacher = ToS risk | OpenAI Terms of Use (outputs may not be used to develop competing models); Anthropic ToS equivalent; the 2025 Anthropic→OpenAI API-revocation precedent | Confirmed → strengthens D1-A; if the escalation triggers, the ToS row lands in LICENSING.md first |
| TRL supports our exact data shape | TRL SFTTrainer docs (`assistant_only_loss=True` via chat-template assistant masks; template auto-patching for known families); trl 0.19 release notes (native `tools` support in Dataset/SFTTrainer) | Confirmed (D5-A needs no custom collator) |
| TRL masking is safe by default | **huggingface/trl issue #3781: `assistant_only_loss` masks silently discarded when `use_liger_kernel=True`** | **Guardrail added** — liger pinned OFF; masks test-asserted on rendered arrays (§4) |
| Unsloth as primary trainer | Unsloth docs/DeepWiki: it is a **runtime patching layer** over TRL/transformers trainer classes | Confirmed as the §6.6 reproducibility concern → stays the recorded fallback only |
| ~2k conversations is enough | LIMA (arXiv 2305.11206): 1k curated examples align a 65B; LIMIT (Databricks): small high-quality sets suffice, eval-paradigm-dependent | Confirmed → D3 size target is evidence-based; more would buy register overfit, not capability |
| Scaffold+validate ≈ state of practice for tool-call data | APIGen (arXiv 2406.18518) / xLAM-60k: three-stage verification (format → execution → semantics); ToolACE (arXiv 2409.00920): accuracy/diversity-first synthesis | Confirmed → D2-A is the same posture applied at construction time |
| QLoRA config (D4) | QLoRA paper practice + Lightning-AI "hundreds of experiments" LoRA insights + Unsloth hyperparameter guide: all-linear targets > attn-only; α = 2r; r=16 standard; higher r pays only on large diverse corpora | Confirmed — D4-A unchanged |
| Merged vs adapter serving | vLLM LoRA docs: per-request adapter serving supported but adds runtime LoRA overhead; merged = zero-overhead standard serving | Confirmed → §2.4 serves the **merged** model for the QLoRA column (comparable tokens/sec) |

### 8.3 Sources

- Sarvam-M: `huggingface.co/sarvamai/sarvam-m`; `sarvam.ai/blogs/sarvam-m`.
- vLLM: quantization docs (FP8 hardware support; Marlin W8A16 on Ampere); LoRA adapter serving
  docs (`docs.vllm.ai/en/latest/features/lora/`).
- TRL: SFT trainer docs (`huggingface.co/docs/trl/sft_trainer` — `assistant_only_loss`, chat
  templates, tools); trl 0.19 tool-calling SFT (S. Diehl, "Fine-tuning With Tool Calling");
  issue #3781 (liger × assistant masks).
- Datasets/method: APIGen arXiv 2406.18518 + `Salesforce/xlam-function-calling-60k`; ToolACE
  arXiv 2409.00920; LIMA arXiv 2305.11206; Databricks LIMIT.
- LoRA hyperparameters: Lightning AI "Finetuning LLMs with LoRA and QLoRA: insights from
  hundreds of experiments"; Unsloth LoRA hyperparameters guide; HF PEFT LoRA developer guide.
- Frontier-output terms: OpenAI Terms of Use / Services Agreement; Anthropic ToS; 2025
  Anthropic→OpenAI API-revocation reporting (WIRED).
- In-repo: `evals/README.md`, `evals/negatives/heldout.yaml`, `evals/golden/gs02_no_match.yaml`,
  `evals/prompts/exemplars_v1.md`, `docs/LICENSING.md`, `infra/gpu/jarvis.py` (session
  patterns), `docs/BENCHMARKS.md` (frozen Table 2 stamp).
