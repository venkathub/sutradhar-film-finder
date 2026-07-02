# Sutradhar Architecture Decision Log

> Dated, append-only record of architectural choices — options considered, decision, and rationale.
> Per `CLAUDE.md`: any architectural choice (chunking, embedding model, vector store, graph schema,
> retrieval thresholds, base model, quantization) gets an entry here. This is also the interview script.

---

## DEC-0001 — LLM stack: fine-tune base, fallback, showcase, and data-teacher (2026-07-01)

**Status:** Accepted. `CLAUDE.md` and `README.md` reconciled to reference DEC-0001 (2026-07-01).

**Context.** The mission needs a small, permissively-licensed model to **QLoRA fine-tune for
behaviour** (code-mixed Tanglish/Hinglish intent + slot extraction, multi-turn backtracking,
tool-calling) and serve via vLLM on an **on-demand** GPU. Two hard constraints shape the choice:

1. **Beatable benchmark (de-risks R2).** Fine-tuning must *measurably beat* a well-prompted base on
   our generation metrics (GS-07 intent/slot, GS-08 backtracking, tool-calling). That requires a
   strong **general** base that has genuine **headroom in our niche** (Indic code-mix + our tool
   schema + backtracking). A base already specialized for Indic (e.g. Sarvam-M) would leave almost
   no headroom → FT can't win → wasted GPU. "Beatable" ≠ "weak"; it means *strong in general,
   unspecialized in our niche*.
2. **Accurate + cheap to serve on-demand.** Small (~4B), single-GPU QLoRA, day-0 vLLM support,
   fast to load for a sub-2-min live demo, Apache-2.0 (clean licensing for a portfolio).

**Options considered (mid-2026 landscape).**

| Model | Params | License | Indic / code-mix | Tool-calling (BFCL v4) | Role fit |
|---|---|---|---|---|---|
| **Gemma 4 E4B** | ~4B eff. | **Apache 2.0** | strong general multilingual; **headroom** on Tanglish/Hinglish | mid–high 80s; **native function-call tokens** (low output variance, nudges ahead post-FT) | **Primary FT base** |
| **Qwen3-4B-Instruct-2507** | 4B | **Apache 2.0** | strong general multilingual; headroom | **high 80s (top sub-7B OOB)**; 256K ctx (RoPE) | **Fallback FT base** |
| Phi-4-Mini-Instruct | 3.8B | MIT | weaker Indic breadth | low–mid 80s; best reasoning chains, fastest | Not chosen (weaker Indic) |
| Gemma 3 4B-it | 4B | Gemma custom license | good | good, prompt-JSON tool format | Superseded by Gemma 4 E4B |
| **Sarvam-M** | 24B | **Apache 2.0** | **already Indic-specialized** (+20% Indic vs Mistral-Small base; +86% on romanized-Indic GSM8K) | via base | **Optional 24B showcase + synthetic-data teacher — NOT the FT base** |

**Decision.**

- **Primary fine-tune base → Gemma 4 E4B (Apache 2.0).** Upgrade from Gemma 3 4B-it. Apache 2.0
  removes the Gemma-3 custom-license caveat; native function-call special tokens give cleaner,
  lower-variance tool-calls (helps post-FT tool-call accuracy); strong multilingual base with real
  headroom on code-mixed Indic intent/slot + backtracking (→ beatable base benchmark); proven
  single-GPU QLoRA (Unsloth/PEFT) and day-0 vLLM support; small and fast to load for the on-demand
  demo. Preserves continuity with the project's original Gemma choice.
- **Fallback fine-tune base → Qwen3-4B-Instruct-2507 (Apache 2.0).** Use if Gemma 4 tooling/serving
  is unstable in our stack. Caveat for the *beatable-benchmark* goal: its unusually strong OOB
  tool-calling priors leave **less headroom on the tool-call metric specifically** — so if we adopt
  it, lead the FT story with **Indic code-mix intent/slot (GS-07) + backtracking (GS-08)**, where
  headroom is real, and treat tool-calling as a secondary metric.
- **Optional live showcase (NOT the FT base) → Sarvam-M 24B (Apache 2.0).** The "big, already-Indic"
  contrast model for an "if time permits" live demo. Deliberately **not** the FT base: it is already
  tuned for our exact niche (≈ no beatable headroom), it is 24B (slower to load, costlier to FT and
  serve on-demand), and using it as the base would undercut the honest before/after story.
- **Synthetic-data teacher (P4) → Sarvam-M 24B (or a frontier API).** Repurpose Sarvam-M's Indic
  code-mix strength to **generate high-quality, grounded** multi-turn / tool-calling training data.
  This directly raises FT-data quality and is the single biggest lever on *whether QLoRA beats the
  base* (R2). Same knob the tool-calling literature confirms: post-FT accuracy is dataset-bound, not
  base-bound.
- **Unchanged:** embeddings = **BGE-M3** (MIT); reranker = **bge-reranker-v2-m3**.
- **All model IDs remain env-driven** (`LLM_MODEL`, `LLM_BASE_URL`, `EMBED_MODEL`) — never hardcoded.

**Why this satisfies "FT accurate with a beatable benchmark."** *Accurate*: a strong general 4B
base + Sarvam-M-taught, record-grounded synthetic data. *Beatable*: the base is unspecialized in
Indic code-mix + our tool schema + backtracking, so QLoRA has real, demonstrable lift on GS-07/GS-08.
If, after all this, QLoRA still does not beat the base, we record the finding and cut the adapter
(pre-committed in P4) — a senior signal, not a failure.

**Consequences.**
- P3 base benchmark and P4 FT both run against **Gemma 4 E4B** (fallback Qwen3-4B-Instruct-2507).
- The two-table benchmark stays honest: retrieval (model-independent) never mixed with generation.
- GGUF/llama.cpp remains optional (portable CPU fallback only), not a deploy requirement.

**Follow-ups.**
- ✅ `CLAUDE.md` (§Subsystems/§Tech stack/§Licensing/§Infra) and `README.md` (§Subsystems) reconciled
  to reference DEC-0001 (2026-07-01).
- Record exact `google/gemma-4-E4B` and `Qwen/Qwen3-4B-Instruct-2507` revisions/commit SHAs in
  `.env.example` + `BENCHMARKS.md` when the P4 run happens (reproducibility).
- ✅ Validate Gemma 4 E4B + vLLM on the rented GPU in a P0 smoke step before committing GPU budget;
  fall back to Qwen3-4B-Instruct-2507 if unstable. **Discharged 2026-07-01 (P0 task 8):**
  `google/gemma-4-E4B` (ungated) booted on a JarvisLabs **A100-40GB** under **vLLM 0.24.0**; the
  connectivity smoke returned `status="up"` (~98 tok/s single-stream); instance created→destroyed in
  one ephemeral cycle for **₹28.38 (~$0.34)**. No fallback needed. Finding: the base model ships **no
  chat template**, so vLLM must be served with `--chat-template` for `/v1/chat/completions` (folded
  into `infra/gpu/jarvis.py`; evidence in `infra/README.md`).

**Sources (accessed 2026-07-01).**
- Gemma 4 launch + Apache 2.0: HF blog `huggingface.co/blog/gemma4`; model card `ai.google.dev/gemma/docs/core/model_card_4`; `google/gemma-4-E4B`.
- Gemma 4 on vLLM: `vllm-project.github.io/2026/04/02/gemma4.html`. QLoRA single-GPU: Unsloth/PEFT walkthroughs (Medium, 2026).
- On-device tool-calling 2026 (BFCL v4, post-FT equalization, native FC tokens): `ertas.ai/blog/on-device-tool-calling-2026-qwen3-gemma4-phi4`.
- Qwen3-4B-Instruct-2507: `huggingface.co/Qwen/Qwen3-4B-Instruct-2507`; Qwen docs (vLLM, 32K→256K ctx).
- Sarvam-M (24B, Apache 2.0, Mistral-Small base, Indic gains): `sarvam.ai/blogs/sarvam-m`; `huggingface.co/sarvamai/sarvam-m`.

---

## DEC-0002 — Embedding model: A/B decided by the P2 retrieval gate (2026-07-01)

**Status:** **Accepted (2026-07-02, P2 execution).** BGE-M3 met the exit gate on the first pass —
Recall@10 = **1.000** (≥ 0.90) in **every** ablation cell and version-set recall = **1.0** on
GS-01 and GS-06 (run `20260702T135315Z-f6583183`; full grid in `docs/BENCHMARKS.md` Table 1).
Per the execution note below, the `bge-multilingual-gemma2` 9B challenger leg was therefore
**skipped entirely — gate met by default, challenger not run**; zero GPU time spent on it. The
9B leg remains the recorded escalation path if a future catalog-scale regression reopens the gate.

**Context.** The mission is Indic-heavy and cross-lingual; the P2 exit gate is Recall@10 ≥ 0.90 with
version-set recall = 1.0 on GS-01/GS-06. Embedding quality is the largest single lever on whether that
gate is reachable. Two credible open embedders sit at different points on the quality/cost curve.

**Options.**

| Option | Params | Retrieval | Cost / infra | Notes |
|---|---|---|---|---|
| **BGE-M3** (default) | 568M | **hybrid** dense + sparse + multi-vector, 100+ langs, 8192 ctx | cheaper; GPU embed pass is short | native sparse signal helps title/fuzzy match (GS-11) |
| **bge-multilingual-gemma2** | ~9B (gemma-2-9b) | **dense-only**, SOTA multilingual (MIRACL, MTEB-fr/pl) | ~3× storage, higher latency, longer GPU embed pass; needs a **separate** sparse signal | stronger cross-lingual recall per BAAI/MTEB |

**Decision rule.** Default to **BGE-M3**. Adopt **bge-multilingual-gemma2** only if it clears a Recall@10
target that BGE-M3 cannot, and the added storage/latency is acceptable; if adopted, pair it with an
explicit sparse retriever (e.g. BM25/SPLADE) to preserve the hybrid signal BGE-M3 gives for free.

**Compute.** Both embedders are neural → the corpus/query embedding pass runs on the **rented on-demand
GPU** (ROADMAP §2 compute placement), not the laptop. The resulting index is a versioned artifact.

**Consequences.** Reranker stays **bge-reranker-v2-m3** regardless. The chosen embedder + index version
are recorded in the P2 benchmark reproducibility stamp (ROADMAP §6.1). This entry is updated to
**Accepted** with the measured numbers when P2 completes.

**Sources (accessed 2026-07-01).** BAAI `bge-m3`, `bge-multilingual-gemma2` (FlagEmbedding, HF cards);
MTEB / MIRACL multilingual retrieval results.

---

## DEC-0003 — GPU instance selection & cost envelope (JarvisLabs) (2026-07-01)

**Status:** Accepted (sizing); exact SKU confirmed at bring-up per live availability. Names the rented
instance for each GPU job so the on-demand/cost-discipline posture is concrete, not aspirational.

**Context.** Per ROADMAP §2, **all neural-model operations run on the rented on-demand GPU**, in short
batched sessions on JarvisLabs (per-minute billing; pause = storage-only; delete volume after HF Hub
push). The workloads and their peak VRAM:

| Job (phase) | Model(s) | Approx peak VRAM | Notes |
|---|---|---|---|
| Candidate-edge extraction (P1) | instruct LLM (base 4B or hosted API) | ≤ 12 GB (4B) | batch inference over Wikipedia prose |
| Neural transliteration (P1) | IndicXlit | < 2 GB | small; usually rule-based on laptop suffices |
| Embedding + rerank (P2) | BGE-M3 568M / **bge-multilingual-gemma2 9B** / reranker 568M | ~4 GB / **~18–20 GB** / < 2 GB | 9B path (DEC-0002) sets the ceiling |
| **QLoRA fine-tune (P4)** | Gemma 4 E4B ~4B, 4-bit NF4 | **~10–12 GB** | fits 24 GB with headroom |
| Serve for benchmark + demo (P4/P5) | Gemma 4 E4B bf16 via vLLM | **~9 GB + KV cache** | 40 GB improves concurrency/tokens-sec |
| *Optional* 24B showcase / teacher | Sarvam-M 24B | bf16 ~48 GB / FP8 ~28 GB | needs 80 GB card, or FP8 on 40–48 GB |

**JarvisLabs options (mid-2026 pricing, indicative — confirm at bring-up).** A100 40 GB ≈ $0.89/hr;
A100 80 GB (SXM) ≈ $1.29/hr; RTX 6000 Ada 48 GB ≈ $0.99/hr; RTX Pro 6000 Blackwell 96 GB ≈ $1.89/hr;
H100 SXM 80 GB ≈ $2.69–2.80/hr; A30 / RTX 4090 24 GB budget tier; CPU VM ≈ $0.05/hr.

**Decision.**

- **Primary workhorse for all core jobs (P1 extraction, P2 embedding incl. the 9B A/B, P4 QLoRA FT +
  base/after benchmark serving, P5/P6 demo) → NVIDIA A100 40 GB (~$0.89/hr).** Best balance: ample for
  4B QLoRA, strong tokens/sec for the 4B serving benchmark, fits the 9B embedder, best cost-per-token,
  fast pause/resume for a sub-2-min live demo.
- **Value alternative → RTX 6000 Ada 48 GB (~$0.99/hr)** if A100 40 GB is unavailable — more headroom
  (fits Sarvam-M 24B in FP8/4-bit) at a similar rate.
- **Budget floor (FT + 4B serving only, skip the 9B embedder) → 24 GB tier (RTX 4090 / A30).** 24 GB
  is sufficient for 4B QLoRA and 4B vLLM serving.
- **Optional 24B showcase (Sarvam-M bf16) → A100 80 GB (~$1.29/hr) or H100 80 GB (~$2.69/hr);** or run
  Sarvam-M **FP8 (~28 GB)** on the 48 GB Ada / A100 40 GB to avoid the 80 GB card. For the **data-teacher**
  role, prefer a **frontier API** so no big card is rented.

**Cost envelope (the cost-discipline headline).** Producing the entire standing portfolio evidence:
P1 extraction ~1–2 h; P2 embedding + retrieval-eval ~1–3 h; **P4 (synthetic-data + QLoRA FT + base/after
capture) ~4–8 h** — all on A100 40 GB → **≈ $10–25 total GPU spend**, plus **~$0.25–0.50 per live demo**
(resume-from-paused). Storage-only while paused; volume deleted after HF Hub push.

**Consequences.** RUNBOOK (P6) documents the exact resume → one-command-up → demo → STOP flow for the
chosen SKU. `LLM_BASE_URL`/`LLM_MODEL` stay env-driven so the same stack targets any of these instances.

**Sources (accessed 2026-07-01).** JarvisLabs pricing (`jarvislabs.ai`, `costbench.com`, `gpuvec.com`,
`nodepedia.com`); QLoRA/vLLM VRAM sizing (Unsloth requirements; koishiai 24 GB QLoRA guide; vLLM +
Mistral-Small-24B serving docs); Sarvam-M FP8 serving (`sarvam.ai/blogs/sarvam-m`).

---

## DEC-P0-1 — Python dependency & packaging manager: uv (2026-07-01)

**Status:** Accepted (P0 grooming). Applies repo-wide from P0.

**Context.** P0 needs a reproducible, "rebuild-from-scratch" Python toolchain (CLAUDE.md infra;
ROADMAP §6.1/§6.6) that CI and Docker share without drift.

**Options.** (A) **uv** — Rust-based, single tool for resolve+venv+lock, `uv.lock`, fast CI/Docker.
(B) **Poetry** — mature, known, heavier/slower. (C) **pip-tools + venv** — minimal, manual, weakest DX.

**Decision.** **uv.** Commit `uv.lock`; CI runs `uv sync --frozen`; Docker uses `uv sync`. Speed +
deterministic lockfile directly serve the reproducibility and cost-discipline goals.

**Consequences.** `pyproject.toml` + `uv.lock` are the source of truth; contributors install uv.
If a hard blocker appears, Poetry is the documented fallback (same `pyproject.toml`).

**Sources (accessed 2026-07-01).** Astral uv docs (`docs.astral.sh/uv`); uv lockfile/CI + Docker
reproducibility guides (pydevtools; uv 2026 guides).

---

## DEC-P0-2 — Repo Python layout: single `src/sutradhar/` package (2026-07-01)

**Status:** Accepted (P0 grooming).

**Context.** The CLAUDE.md subsystem dirs are hyphenated (`data-pipeline`, `rag-engine`) — not
valid import names. P0 must define how they map to importable code.

**Options.** (A) **Single installable `src/sutradhar/` package** with subpackages
(`sutradhar.config`, `.serving`, `.rag`, `.pipeline`, …); hyphenated dirs hold entrypoints/
Dockerfiles/READMEs importing `sutradhar.*`. (B) **Multi-package workspace** — one installable pkg
per subsystem. (C) **Flat `sutradhar/` at repo root.**

**Decision.** **A.** One `pyproject.toml`, one venv, one test suite, one import root; a
dir→package mapping table lives in the top-level README. Preserves the CLAUDE.md directory story
while keeping imports clean.

**Consequences.** Subsystems are Python subpackages, not separate distributions; if a subsystem is
ever split into its own service, it is promoted out of the monopackage at that time (noted for a
future decision).

---

## DEC-P0-3 — Task runner: Makefile (2026-07-01)

**Status:** Accepted (P0 grooming).

**Context.** CLAUDE.md references "make/task targets"; P0 needs `setup/fmt/lint/typecheck/test/up/
down/smoke/hf-check/gpu-validate/gpu-nuke`.

**Options.** (A) **Makefile** — ubiquitous, zero install, CI-friendly. (B) **justfile** — cleaner
syntax, extra install. (C) **Taskfile (go-task)** — YAML, extra binary.

**Decision.** **Makefile.** Nothing extra to install on the low-spec laptop or in CI; matches
CLAUDE.md wording.

**Consequences.** Targets are the documented entrypoints in every module README and the RUNBOOK.

---

## DEC-P0-4 — LLM endpoint client contract: OpenAI-compatible (2026-07-01)

**Status:** Accepted (P0 grooming). Sets the contract reused by P3 tracing and P5 serving.

**Context.** The connectivity smoke test (P0 exit criteria) and all later generation calls target an
on-demand vLLM endpoint via env-driven `LLM_BASE_URL`/`LLM_MODEL`. The client contract must stay
swappable across vLLM, a frontier API (P4 data-teacher), and a local fallback.

**Options.** (A) **OpenAI-compatible via the `openai` SDK** against `LLM_BASE_URL` — vLLM serves this
natively (verified routes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`; `--api-key`
→ Bearer). (B) **Raw `httpx`** to `/v1/chat/completions` — no dep, hand-rolled, drift risk.
(C) **vLLM-native `/generate`** — couples to a non-standard route, breaks swappability.

**Decision.** **A.** Standardizing on the OpenAI-compatible contract now is what makes the endpoint
truly env-swappable later. `LLMClient.health()` probes `/health` → `/v1/models` → a 1-token
`/v1/chat/completions`; a connection-refused/timeout short-circuits to `status="off"` (exit 0, never
a crash) — the seed of the P5/P6 graceful-degradation thread.

**Consequences.** Adds the `openai` dep in P0. `status:"off"` is a first-class success path, not an
exception.

**Sources (accessed 2026-07-01).** vLLM "OpenAI-Compatible Server" docs
(`docs.vllm.ai/.../serving/openai_compatible_server`) — `/health`, `/v1/models`,
`/v1/chat/completions`, `--api-key`.

---

## DEC-P0-5 — JarvisLabs P0 GPU validation: full ephemeral create→smoke→destroy (2026-07-01)

**Status:** Accepted (P0 grooming). Implements the DEC-0001 follow-up (validate Gemma-4-E4B + vLLM
on the rented GPU before committing budget) and the P0 "smoke green in both states" exit criterion.

**Context.** P0 must prove the smoke test's *up* state against a real rented GPU without keeping any
standing machine. DEC-0003 names JarvisLabs (per-minute billing) as the on-demand provider.

**Options.** (A) **Manual** dashboard/`jl` bring-up, no committed automation — not reproducible from
scratch. (B) **Thin `gpu-up/gpu-down` resume/pause helper** — assumes a pre-existing (warm) paused
instance; leaves teardown to the operator. (C) **Full ephemeral automation** — one script
provisions a fresh instance, serves, validates, and destroys it.

**Decision.** **C — full ephemeral automation.** `make gpu-validate` drives the `jarvislabs` SDK/`jl`
CLI: **create** a fresh instance (`GPU_TYPE`, e.g. A100-40GB, via `JARVISLABS_API_KEY`) → `vllm
serve` Gemma-4-E4B (fallback `Qwen3-4B-Instruct-2507`) → health-wait on `/health` → `make smoke`
(expect `status="up"` + real token) → capture evidence → **destroy**. Chosen because it best
embodies the mission's "rebuild from scratch, never depend on a warm machine, volume deleted after
use" principle: the UP-state proof is reproducible from nothing.

**Guardrails.** (1) **Teardown guaranteed** via `try/finally` + a `make gpu-nuke` safety target that
destroys any stray tagged instance — a leaked billing GPU is treated as a defect (a teardown test
injects a smoke failure and asserts destroy still runs). (2) **Never on a PR** — developer- or
`workflow_dispatch`-invoked only (cost + secrets); Tier-1 CI stays fully mocked/stubbed. (3) The
*cold* create→destroy validation is distinct from the P6 **sub-2-min warm-resume** demo path (R4),
which remains in `docs/RUNBOOK.md`.

**Consequences.** Adds `JARVISLABS_API_KEY` (+ `GPU_TYPE`) to `.env.example`; drops any persistent
instance id (fresh each run). Cost bounded to one short create→destroy cycle (~$0.89/hr, minutes,
well under $1), within the DEC-0003 envelope. Evidence (log/screenshot + tokens/sec + create→up
time) is captured into `/infra/README.md` as the seed for the P6 RUNBOOK.

**Sources (accessed 2026-07-01).** JarvisLabs Python SDK + `jl` CLI docs (`docs.jarvislabs.ai/sdk`,
`/cli` — programmatic create/pause/resume/destroy, `jl run` auto-destroy; replaces deprecated
`jlclient`); JarvisLabs "Serving LLMs" tutorial (`vllm serve --port 8000` reached from the laptop
via tunnel).

---

## DEC-P1-1 — Edge storage: single polymorphic `edges` table (2026-07-02)

**Status:** Accepted (P1 grooming; spec `docs/phases/P1_SPEC.md` §2.2/§3).

**Context.** The remake graph needs five typed edges over two node kinds (`version→version` for
remake/dub, `work→work` for sequel/based_on), and every edge must flow through one provenance +
confidence + conflict + human-review pipeline (the verification gate is the product).

**Options.** (A) **Single polymorphic `edges` table** — enum `edge_type`, `src/dst_kind`
discriminators, CHECK constraints per type shape, traversal via recursive CTEs. (B) Two typed
tables (`version_edges`, `work_edges`) with hard FKs. (C) Graph extension (Apache AGE) / dedicated
graph DB.

**Decision.** **A.** One uniform gate/review/promotion path and one `ground_truth_edges` view for
every edge type; adding an edge type is a value, not a table. Soft polymorphic FKs are hardened by
a validation trigger + constraint integration tests. B forks the gate logic; C reopens the settled
Postgres call for a 2-hop graph.

**Consequences.** Type-shape rules (remake/dub = version→version; sequel/based_on = work→work) are
CHECK-enforced; `candidate_edges` promotion targets a single table; P2 indexing and P5 tools read
one view.

## DEC-P1-2 — DB access + migrations: SQLAlchemy 2.0 (typed ORM) + Alembic (2026-07-02)

**Status:** Accepted (P1 grooming).

**Options.** (A) **SQLAlchemy 2.0 typed ORM + Alembic.** (B) psycopg 3 + numbered raw-SQL
migrations + hand-rolled runner. (C) SQLAlchemy Core + Alembic.

**Decision.** **A.** The graph outlives P1 (P2 indexing, P5 tools read it); typed models that
mypy-strict can see, a standard migration story, and Alembic autogenerate keep schema diffs
reviewable. Bulk upserts use Core-level `insert…on_conflict` inside the ORM session where the ORM
would be ceremony.

**Consequences.** `sutradhar.graph.schema` (declarative models) + `alembic/` migrations are the
schema source of truth; `make db-migrate` is the documented entrypoint.

## DEC-P1-3 — Provenance: inline `jsonb sources[]`, pydantic-validated (2026-07-02)

**Status:** Accepted (P1 grooming).

**Options.** (A) **`jsonb` `sources[]` column per record/edge, validated by a pydantic
`SourceRef` model at the write boundary.** (B) Normalized `provenance` table. (C) Hybrid (jsonb
now, mirror table later).

**Decision.** **A**, with C's escape hatch noted. Every consumer of provenance — the gate views,
golden-fixture builder, tool results (`TOOL_SCHEMA.md` puts `sources[]` inline on every result),
and the P6 citation UI — wants it *with the row*, join-free. A normalized table is machinery a
~30-record slice (and even the breadth catalog) doesn't need; if source-centric queries ever
matter, a mirror table can be derived from the jsonb without a schema break.

**Consequences.** `SourceRef` rejects empty `sources[]`/unknown source ids before any insert;
"all claims from source X" queries use jsonb operators.

## DEC-P1-4 — Candidate-edge extraction model: Gemma 4 E4B on the ephemeral A100 (2026-07-02)

**Status:** Accepted (P1 grooming). Frontier API is the documented fallback.

**Context.** The P1 GPU job (DEC-0003: ~1–2 h) proposes remake/dub edges from Wikipedia prose into
`candidate_edges`. Every candidate passes a human gate, so extractor precision is a review-time
cost knob, not a correctness risk.

**Options.** (A) **Gemma 4 E4B served by vLLM on the ephemeral JarvisLabs A100** (the exact stack
validated in P0 / DEC-P0-5). (B) Frontier API. (C) Sarvam-M 24B FP8.

**Decision.** **A.** ~$1–2 for the whole pass, zero new dependency or key, and it dogfoods the
same serving path P4 uses. **Fallback trigger:** if a spot-check puts candidate precision below
~0.5 (human review time becomes the real cost), switch to a frontier API and record the switch
here. C is rejected: a 24B rental for a job a 4B + human gate covers; Sarvam-M's slot is the P4
data-teacher (DEC-0001).

**Consequences.** The extraction script talks only to `LLMClient`/`LLM_BASE_URL` (endpoint-
agnostic); prompts + raw outputs are persisted as a versioned artifact with a run hash
(reproducibility stamp, ROADMAP §6.1); parse-failure and precision metrics reported in the P1
graph report.

## DEC-P1-5 — Cross-script `match_key`: deterministic rule-based romanization + rapidfuzz (2026-07-02)

**Status:** Accepted (P1 grooming). IndicXlit remains a measured contingency.

**Options.** (A) **Rule-based:** native script → IAST/ISO-15919-style romanization via
`indic_transliteration.sanscript` (Devanagari/Tamil/Malayalam/Telugu/Kannada/Bengali verified
supported) → ASCII fold → lowercase → vowel-length collapse; resolution = exact `match_key` hit
then rapidfuzz over the key index. (B) Neural IndicXlit romanization for every native title.
(C) Multi-key storage (multiple romanization schemes per title).

**Decision.** **A.** Laptop-safe (pure Python — not a neural op, per ROADMAP §2 compute
placement), reproducible, one indexed key. The "popular spelling" variants B/C chase are already
supplied as real data by IMDb `title.akas` + TMDB `alternative_titles` into `version_title`.
**Contingency:** if rule-based + fuzzy fails GS-11 spot-checks, run IndicXlit in a rented-GPU
session with outputs cached as `version_title(kind='transliteration')` — noting IndicXlit models
are **CC BY-SA 4.0** (attribution/share-alike added to `LICENSING.md` if invoked).

**Consequences.** GS-11 (title-match under perturbation) gates this in P1 unit tests and again in
P2 retrieval; the rapidfuzz threshold is recorded when tuned.

## DEC-P1-6 — Human-verification gate tooling: typer CLI (2026-07-02)

**Status:** Accepted (P1 grooming).

**Options.** (A) **Typer CLI (`make review-candidates`)** — supporting sentence + resolved
entities shown; confirm/reject/skip; writes `reviewed_by/reviewed_at`; promotion sets
`human_verified=true` and links `promoted_edge_id`. (B) Minimal web review page. (C) CSV
export/import round-trip.

**Decision.** **A.** The portfolio point is the *gate semantics* (nothing bypasses the gate;
rejection is recorded; promotion is auditable), not the review surface. A CLI is zero extra
surface area, scriptable, and testable; a session screenshot serves the evidence need. B pre-empts
P5's API and P6's UI; C has no audit-trail integrity.

**Consequences.** Gate-enforcement integration tests drive the CLI's promotion/rejection paths;
the review session is part of the P1 exit evidence (candidate precision = confirmed/proposed).

## DEC-P1-7 — Ground-truth view predicate: MEDIUM passes the gate views (2026-07-02)

**Status:** Accepted (P1 task 1; clarifies a P1_SPEC internal inconsistency — user-confirmed).

**Context.** P1_SPEC §1.8 and its SQL sketch gate the `ground_truth_*` views on
`confidence = 'HIGH' OR human_verified` — but the prose directly beneath the sketch, the
`DATA_SOURCES.md` tier table ("MEDIUM → the live graph, flagged"), and the §4 test list ("a
MEDIUM edge **with an open conflict** is excluded") all say MEDIUM rows are live. The two
readings cannot both be implemented.

**Options.** (A) **MEDIUM passes the views** — predicate = `sources[]` non-empty AND no open
conflict; the golden-fixture validator (not the view) enforces HIGH/human-verified for fixtures.
(B) HIGH-or-verified only, per the SQL sketch literally.

**Decision.** **A.** Under B the MEDIUM tier is dead weight (write-only until promoted — nothing
downstream could ever read it) and the fixture validator's HIGH-only rule would be redundant.
A matches the tier table's intent: MEDIUM is live-but-flagged; consumers see the `confidence`
column and can filter. CANDIDATE remains excluded **by construction** (separate
`candidate_edges` table, never referenced by any view).

**Consequences.** View DDL (initial Alembic migration) implements predicate A;
`test_medium_edge_passes_gate_views` pins it. The golden-fixture validator (task 14) owns the
stricter HIGH/human-verified rule. Layered gates: structural exclusion (CANDIDATE) → conflict/
provenance gate (views) → fixture gate (HIGH only).

## DEC-P1-5 amendment — `match_key` romanization scheme: ITRANS, measured (2026-07-02)

**Status:** Accepted (P1 task 8; refines DEC-P1-5 option A within its "IAST/ISO-15919-style"
wording — the *goal* of the key is popular-spelling proximity, so the scheme is a measured
parameter, not a reopened decision).

**Measurement** (11 real slice title pairs, native script vs popular English spelling,
rapidfuzz ratio after fold): **ITRANS avg 87.4** with 2 exact hits vs **IAST 80.9 / ISO-15919
80.9** with 0 exact — IAST's bare consonants (`dṛśyam → drsyam`) lose the vowels popular
romanization keeps (`drishyam`). Two deterministic post-fixes raise ITRANS to **10/11 pairs
≥ 0.80 (avg 89.7)**:
1. **Tamil digraph normalization** — sanscript's Tamil scheme emits Sanskrit-positional
   aspirates (`ப→bha`, `ச→jha`); folded to the popular plain series (`p/ch/k/d`).
2. **Word-final schwa deletion** for Devanagari/Bengali (`दृश्यम → drishyam`, `एक → ek`),
   applied before casefold so long ā survives; Dravidian scripts keep final vowels.

**Pipeline:** NFC → script detect (Unicode-block majority) → ITRANS (+fixes) → casefold →
strip diacritics → alnum-only → collapse character runs (vowel length + gemination) →
collapse whitespace. Fuzzy resolution threshold **0.80**, tuned on the GS-11 perturbation
suite. Known limitation: non-Sanskrit Tamil letters (ன/ழ/ற) and Sinhala/Han have no
deterministic mapping — their Latin AKA/canonical rows in the same index carry the match;
IndicXlit remains the unused contingency.

**Consequences.** `sutradhar.pipeline.normalize` implements this; `make rekey-titles`
re-keys existing rows idempotently; `resolve_title.candidates[].score` = the rapidfuzz
0–1 value (TOOL_SCHEMA v0 semantics).

## DEC-P1-3 amendment — `SourceId` gains `rule` (2026-07-02, task 9)

**Context.** The dub-vs-remake rule *derives* edges (dub tracks) and evidence. Recording rule
output under `human` or an external source would be dishonest provenance; leaving `sources[]`
empty is gate-forbidden.

**Decision.** The pydantic `SourceId` enum (write-boundary contract) gains **`rule`** for
evidence produced by a documented deterministic rule (`ref` names the rule, e.g.
`dub-track-rule`, `lead-cast-overlap-rule`). Rule-only claims are **MEDIUM** by the tier table
("a derived rule with no corroboration") — live but flagged, promotable by the human gate.
DB-side nothing changes (`sources` is jsonb; no CHECK on content). Edge origins are now
separable by `sources[0].source`: wikidata / rule / (later) wikipedia-extraction — which is
what keeps the extraction-lift metric attributable.

## DEC-P1-4 amendment — extraction needs vLLM guided decoding (2026-07-02, task 11 GPU run)

**Measured on the live A100 session:** free-form JSON prompting of base Gemma 4 E4B produced a
**92.6% parse-failure rate** (single-quoted pseudo-dicts, bare `<end_of_turn>`, prompt echoes).
Re-running the same 27 pages with **vLLM `guided_json`** (schema-forced decoding from
`ExtractionResponse.model_json_schema()`, temperature 0) dropped it to **7.4%** (2/27 pages) and
yielded 72 proposals → 58 candidates after the verbatim-evidence guard (14 unsupported dropped).

**Adjustments (within DEC-P1-4 option A — no model change, no fallback triggered):**
1. `LLMClient.complete` accepts `temperature` + `extra_body` (guided decoding pass-through).
2. `parse_extraction_output` takes the FIRST well-formed JSON object and ignores trailing
   junk (guided decoding can emit continuation noise). Content is never repaired; pydantic
   still gates every field.
3. Observed 4B noise (self-pairs, edge-type confusion, inverted directions) is left in
   `candidate_edges` **by design** — the human gate measures it as precision, not a crash.

The frontier-API fallback stays untriggered pending the task-12 precision measurement.

## DEC-P1-8 — TOOL_SCHEMA v0 FROZEN (2026-07-02, P1 task 15)

**Status:** Accepted. `docs/phases/TOOL_SCHEMA.md` flipped DRAFT → **FROZEN v0**; the
machine-readable artifact is **`docs/phases/tool_schema.v0.json`** (JSON Schema 2020-12:
params + results + enums for all five tools).

**What froze.** The v0 seed signatures, unchanged — implementation required zero signature
breaks (the §2.5 "contract is satisfiable" bet held). Pinned wording-level semantics:
`resolve_title.score` = rapidfuzz 0–1 (exact = 1.0); `ambiguous` = multi-Work span; `scope` ↔
`version.country`; `include_sequels` = transitive work-level walk with the sequel work's
original labelled `is_sequel_of`; unverified relationship = `null`, never guessed;
`era` pivots on the set's original's year; `sources[].source` includes `rule`.

**Enforcement.** Three CI conformance layers: (1) `test_tool_schema_json_valid` +
md↔json sync test (doc drift fails CI); (2) `test_golden_expected_tool_calls_validate` —
no hallucinated tool/param names committable into the golden set (P3/P4 reuse this validator
against model-emitted calls); (3) `test_repository_matches_tool_schema` (signature drift) +
integration result-shape round-trips of real repository calls through the frozen schema.

**Consequences.** P4 synthetic data and the tool-call-accuracy metric target this exact
artifact; any change bumps to v0.1+ with a DECISIONS entry; `search_by_plot` stays
schema-only until P2 (its absence from the repository is itself asserted by test).

---

## DEC-P2-1 — Vector store: Postgres + pgvector (2026-07-02)

**Status:** Accepted (P2 grooming; spec `docs/phases/P2_SPEC.md` §3 — user-approved). Resolves
the pgvector-vs-Qdrant choice CLAUDE.md/ROADMAP explicitly deferred to P2.

**Options.** (A) **Postgres + pgvector** — already in compose (`pgvector/pgvector:0.8.4-pg17`);
embeddings next to the graph (chunk→version→edges joins in one SQL hop); **native `sparsevec`
type (since 0.7.0)** scores the BGE-M3 sparse leg in-DB. (B) Qdrant — named dense+sparse vectors,
server-side `rrf`/`dbsf` fusion; better at 10⁶+ scale. (C) Both behind an interface.

**Decision.** **A.** The hard problem is graph-adjacent retrieval; one store keeps every
citation/provenance join trivial, adds zero services, and `sparsevec` removes Qdrant's former
"native sparse" edge. **Revisit trigger:** catalog-breadth scale (HNSW-on-sparse's 1,000-nnz cap
and server-side fusion become relevant there; exact scan is correct at ~10² chunks).

**Consequences.** `chunks` + `chunk_embeddings` (dense `vector(1024)` + `sparse sparsevec`) land
in the graph DB via Alembic; `CREATE EXTENSION vector` in the migration; laptop adds only the
`pgvector` Python lib (SQLAlchemy types — SQL glue, not a model).

**Sources (accessed 2026-07-02).** pgvector README (`sparsevec` reference: `8·nnz+16` bytes,
≤16,000 nnz, `<#>`/`<=>` ops; HNSW limits; hybrid-search RRF example); Qdrant hybrid-queries docs.

## DEC-P2-2 — Hybrid sparse leg + fusion: BGE-M3 lexical weights in sparsevec + RRF k=60 (2026-07-02)

**Status:** Accepted (P2 grooming; user-approved).

**Options.** (A) **BGE-M3 native lexical weights stored as `sparsevec`, scored in-DB via `<#>`,
fused with RRF (k=60).** (B) Postgres FTS (tsvector) as the sparse leg. (C) Weighted score-sum
fusion (α·dense + β·sparse).

**Decision.** **A.** The sparse signal comes free from the same BGE-M3 pass as dense (the reason
it is the DEC-0002 default); its tokens are multilingual/transliteration-aware over the 250k
XLM-R vocab — B's English stemmers fail exactly on romanized Tamil/Hindi (GS-07/GS-11). RRF is
rank-based (no cross-channel score normalization) and k=60 is the cross-industry default (Azure
AI Search, OpenSearch 2.19, Qdrant `rrf`) — deliberately untuned: one less overfittable knob on a
small eval set. **Fallback:** if RRF underperforms in the ablation, C is measured from the same
artifacts and recorded here.

**Consequences.** `sutradhar.rag.{sparse,fusion}`: query lexical-weight → sparsevec literal,
in-DB scoring, RRF + chunk→Work max-aggregation as pure, unit-testable functions.

**Sources (accessed 2026-07-02).** BAAI `bge-m3` card + FlagEmbedding docs (lexical weights at no
extra cost; hybrid recommended); Azure AI Search RRF docs; OpenSearch 2.19 RRF; pgvector hybrid
example.

## DEC-P2-3 — Chunking of plot/metadata: recursive para-boundary + metadata header + metadata card; size ablated (2026-07-02)

**Status:** Accepted (P2 grooming; user-approved). **Measured winner to be filled at execution.**

**Options.** (A) **Recursive paragraph-boundary chunks, size ablated ∈ {256, 512, 1024} tokens
(default 512), 15% overlap; a metadata header on every chunk
(`"{title} ({year}, {lang}) — remake of {original} …"`); plus one `metadata_card` doc per Version
(titles/AKAs/cast/director/relationship).** (B) Whole-plot single chunk. (C) Semantic/
embedding-guided chunking.

**Decision.** **A.** B is physically broken: the corpus tail (max 53.6k chars ≈ 13k tokens)
exceeds BGE-M3's 8192 window — silent truncation of exactly the long flagship pages. C needs an
embedder to chunk (GPU in the laptop path) and is non-deterministic — breaks the reproducibility
stamp. The grid brackets both regimes the literature identifies (small chunks for factoid, 512–
1024 for broader narrative context — our story-description queries); recursive 512 + 10–20%
overlap is the benchmark-validated default. The header carries remake lineage into every dense
unit; the card gives cast/title-anchored queries (GS-01a, GS-10) a dense target plots may lack.
Chunking is ablated **before** any embedder swap: chunk config ≈ embedder choice in retrieval
impact (Vectara NAACL 2025), and iterating it costs artifact-replay time, not GPU rental.

**Consequences.** Deterministic chunker (`content_hash` pins it); ablation table in the
rag-engine README; winning config recorded here with numbers. _Measured winner (2026-07-02, run
`20260702T135315Z-f6583183`, 13 retrieval fixtures): **1024tok_15pct** — every config passed the
Recall@10 ≥ 0.90 gate at 1.000 and VSR-01/06 = 1.0; 1024tok won on Recall@1 0.923 / MRR@10 0.962
(at depth 20). Consistent with the arXiv 2505.21700 finding that 512–1024 tok favours
broader-context/narrative retrieval — our story-description queries. Full grid in the rag-engine
README ablation table._

**Sources (accessed 2026-07-02).** arXiv 2505.21700 (chunk size vs retrieval, multi-dataset);
2026 chunking benchmark guides (recursive 512 tok, 10–20% overlap default); Vectara NAACL 2025
(25 configs × 48 models).

## DEC-P2-4 — Rerank depth + final top-k: fuse-50 → rerank-50 → top-10; depth ablated (2026-07-02)

**Status:** Accepted (P2 grooming; user-approved). **Measured depth to be filled at execution.**

**Options.** (A) **Fuse top-50 chunks → rerank all 50 → aggregate → top-10 Works; ablate depth ∈
{20, 50}.** (B) Rerank top-20 only. (C) No reranker (fusion order only).

**Decision.** **A.** At slice scale, depth-50 ≈ the whole fused candidate space → the reranker's
recall ceiling; the precomputed full query×chunk score matrix (P2_SPEC §2.6) makes depth a free
laptop-side parameter, so measuring {20, 50} costs nothing. B risks dropping a version-set member
before the reranker sees it — a direct threat to the GS-01/GS-06 = 1.0 gate. C wastes the settled
reranker (`bge-reranker-v2-m3`), the standard lever for closing the last recall/MRR gap.
`top_k=10` matches the `search_by_plot` v0 default and the Recall@10 gate. **Revisit trigger:**
live-path latency at catalog scale. _Measured depth (2026-07-02, run
`20260702T135315Z-f6583183`): **20** — Recall@5/@10 identical at both depths (1.000); depth 20
beat 50 on Recall@1/MRR (0.923/0.962 vs 0.846/0.910 at 1024tok): at slice scale the extra 30
fused candidates only admit distractor chunks whose max-aggregated works can outrank the target.
Depth stays an env-free config knob; 50 remains the recorded catalog-scale starting point._

## DEC-P2-5 — NO_MATCH abstention: absolute top-1 reranker score (sigmoid), calibrated on held-out negatives (2026-07-02)

**Status:** Accepted (P2 grooming; user-approved). **θ to be filled at execution.**

**Options.** (A) **Absolute top-1 cross-encoder score threshold, sigmoid-normalized to [0,1].**
(B) Top-1 vs top-2 margin. (C) Fused-rank (pre-rerank) score threshold.

**Decision.** **A.** Sigmoid→[0,1] is the official BGE reranker score semantics; cross-encoder
relevance scores are query-conditioned and markedly more stable across query types than cosine
(C's documented failure mode). B is structurally wrong for this corpus: GS-01's five near-tied
sibling versions *should* score close — a margin rule punishes exactly the correct behaviour.
Calibration: canary + ε-margin methodology on the calibration half of the ~24-query held-out
negative set (`evals/negatives/heldout.yaml`, absent-from-slice-by-construction), maximizing
NO_MATCH F1 subject to **zero false rejects** on positive golden fixtures; P/R reported on the
test half; gate = 0 false accepts on GS-02. The title channel's 0.80 fuzzy floor (DEC-P1-5)
composes with θ for pure-title negatives (GS-02b "Kaithi").

_Measured outcome (2026-07-02, run `20260702T135315Z-f6583183`, winner cell 1024tok/d20):_
**θ = 0.151747** = 1.35 × the top calibration canary (NEG-17 "toddy-shop accountant wins the
Fields Medal" = 0.11241 — thematically Indian negatives score highest, as expected). **Hard gate
met: 0 false accepts on GS-02 and on all 12 untouched test negatives (NO_MATCH recall = 1.0;
precision 0.75 over the validation population).** The zero-false-reject side constraint was
**measured infeasible** — witness: GS-07a (Tanglish positive) = 0.00084 ≤ NEG-17 = 0.11241; the
cross-encoder scores code-mixed/vague-plot positives below fluent-English out-of-catalog plots.
Consequence, chosen deliberately: the no-hallucination gate outranks the no-false-reject
preference, so four weak-scoring positives (GS-03a/c, GS-07a/b) return `abstain=true` **with
their correct results** (v0 allows both) — they degrade to "low confidence", never to a
fabricated match, and all four still rank the right Work top-5 (Recall@5 = 1.0). This measured
positive/negative interleave on raw cross-encoder scores is a primary P4 headroom target
(code-mixed intent parsing before retrieval). Identical false-reject set across all six ablation
cells (structural, not config noise). θ wired as
`sutradhar.rag.retrieve.CALIBRATED_NO_MATCH_THRESHOLD` (drift-checked against the artifact by
`test_recorded_calibration_outcome_holds`); full curve committed at
`evals/retrieval_runs/<run>.calibration.json`.

**Sources (accessed 2026-07-02).** BAAI `bge-reranker-v2-m3` card (sigmoid mapping); retrieval-
abstention practice (cross-encoder score stability; canary-threshold methodology); UAEval4RAG
(arXiv 2412.12300).

## DEC-P2-6 — Tier-1 CI gates on a committed retrieval-run artifact (2026-07-02)

**Status:** Accepted (P2 grooming; user-confirmed — spec §7 Q2).

**Context.** Tier-1 CI (no GPU, no model calls — ROADMAP §6.2) must recompute Recall@k / MRR /
version-set recall / abstention metrics on every PR and block regressions.

**Options.** (A) **Commit the compact retrieval-run summary (~1–5 MB JSON,
`evals/retrieval_runs/<run_id>.json`: per-query ranked candidates + all channel scores) to git;**
raw embeddings/matrices stay git-ignored under `data/artifacts/` (MANIFEST-hashed, optionally on
HF Hub). (B) HF Hub + download step in CI.

**Decision.** **A.** Zero external dependency and secret-free for forks; CI recomputes every
gating metric from the pinned run (`RETRIEVAL_RUN` env). B adds a network/auth dependency to the
merge gate for no benefit at this artifact size. The committed run id appears in the Table 1
reproducibility stamp, so every benchmark row maps to an exact committed input.

## DEC-0002 execution note (2026-07-02, P2 grooming — user-confirmed)

Within DEC-0002's settled decision rule (default BGE-M3; adopt the 9B challenger only if it
clears a gate BGE-M3 cannot): **if BGE-M3 meets the P2 exit gate on the first pass, the
`bge-multilingual-gemma2` leg is skipped entirely** — no GPU time is spent on the challenger, and
DEC-0002 flips to Accepted recording "gate met by default; challenger not run". The 9B leg
remains the escalation path inside the P2 iteration loop (chunking → fusion → embedder, in that
order). Context worth recording: MIRACL — the public benchmark behind both embedders'
cross-lingual claims — covers hi/bn/te but **not ta/ml/kn**, so the golden-set gate (not
leaderboards) is the only measurement that answers Sutradhar's question.

## DEC-P2-7 — GPU-job transport: HF Hub relay (2026-07-02, execution — user-confirmed)

**Context.** P2 task 5's `make gpu-embed` must ship `gpu_inputs.json` + the job code to the
ephemeral JarvisLabs instance and pull the sealed artifact run back (P2_SPEC §2.6 names the
lifecycle but not the channel). The P0/P1 sessions never moved files — they drove a remote vLLM
API — so there was no precedent to reuse.

**Options.** (A) **HF Hub relay:** laptop uploads inputs + the *self-contained*
`rag-engine/embed_and_score.py` to a private HF **dataset** repo (`HF_ARTIFACT_REPO`); the
instance startup script downloads both, runs the job, uploads the sealed run + an `EXIT` marker;
the laptop polls, downloads into `data/artifacts/retrieval/<run_id>/`, MANIFEST-verifies, then
destroys the instance. (B) SSH/scp direct — no relay repo, but untested SDK SSH surface + key
management. (C) Manual runbook — least code, but no automated ephemeral driver (weak DEC-P0-5
story).

**Decision.** **A** (user-approved at task 5). It matches CLAUDE.md's "HF Hub = artifact
registry / reproducibility bridge" and the delete-the-volume lifecycle, and is fully
mock-testable like `extract_session`. Two deliberate consequences: (1) `embed_and_score.py` is
**self-contained** (stdlib+numpy+pyarrow+FlagEmbedding, no `sutradhar` import — no repo clone on
the box); its output format is locked to `sutradhar.rag.artifacts` by the laptop-side stub
dry-run test. (2) Unlike the token-free vLLM validate script (P0 invariant, still enforced by
test), the embed startup script **must embed an HF token** to upload results; mitigation: use a
fine-grained token scoped to `HF_ARTIFACT_REPO` only, no `set -x` (never echoed), and the script
slot is removed in the same `finally` as instance teardown.

**DEC-P2-7 execution amendment (2026-07-02, task 6).** The planned `gpu` uv dependency group
(P2_SPEC §2.7) was dropped: FlagEmbedding requires `transformers<5`, which caps
`huggingface-hub<1.0` — unresolvable in one lockfile with the laptop's `huggingface_hub>=1.0`
(modern-token API, P0). Under the HF relay the box never sees the repo, so the **authoritative
instance pins live in `build_embed_startup_script`** (`pip install pyarrow huggingface_hub
'transformers<5' FlagEmbedding`); noted in `pyproject.toml`. Session evidence: 4 attempts
(instances 438174/438176/438178/438179 — driver log-relay gap, py3.10 `datetime.UTC`,
transformers-v5 reranker API break, then success), ~10 GPU-minutes total ≈ $0.22, sealed run
`20260702T135315Z-f6583183` (833 unique texts, 44,217 rerank pairs) pulled, verified, pinned as
`RETRIEVAL_RUN`; committed record in `evals/retrieval_runs/`.
