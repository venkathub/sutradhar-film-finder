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

---

## DEC-P3-1 — Generation judge: self-hosted OSS via vLLM, gpt-oss-20b primary (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved; spec `docs/phases/P3_SPEC.md` §3). Frontier API
is the **documented escalation only**.

**Context.** Table 2 needs an LLM judge (GS-08 backtracking coherence; RAGAS) under ROADMAP §6.4
governance: pinned, cross-family, human-validated. Independence map: **Gemma** (under test),
**Qwen** (fallback base), **Mistral line** (Sarvam-M teacher base — excludes Prometheus-2).

**Options.** (A) **Self-hosted OSS judge on the ephemeral GPU** — candidates in measured order:
**gpt-oss-20b** (OpenAI family; Apache 2.0; MoE ~14–16 GB → co-serves with BGE-M3 on the
A100 40 GB workhorse; official vLLM cookbook; documented judge usage (TruLens); arXiv 2508.12461
shows it matching/beating gpt-oss-120b on several benchmarks), then **Phi-4-14B** (MIT).
(B) Frontier API (pinned dated version) — quality ceiling, but external key + model-deprecation
risk under the reproducibility stamp + dilutes the self-hosted story. (C) Rejected: Llama-3.3-70B
(AWQ-INT4 ≈ 47 GB weights — breaks the 48 GB Ada with KV; an 80 GB card for a judge violates
DEC-0003); Prometheus-2 (Mistral/teacher-adjacent); Sarvam-M (is the teacher); Qwen family.

**Decision.** **A, selected empirically:** serve candidates in one short ephemeral session; freeze
the first to reach **κ ≥ 0.6** against the ~24-item human-labelled sample (Judge's Verdict
methodology, arXiv 2510.09738 — agreement over correlation; 27/54 judges Tier-1; explicitly
size-independent). One rubric revision allowed, then escalate to B and record it here. Judge runs
as a **batch pass over recorded transcripts** (no always-on requirement; CI gates on the
committed artifact per DEC-P2-6). Pinned as `{HF repo, revision SHA, vLLM config, prompt hash,
temperature 0}`; rubric output schema-forced via guided decoding (per the DEC-P1-4 amendment
finding). gpt-oss-20b reasoning effort pinned in the judge prompt.

**Consequences.** `JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY` env (OpenAI-compatible → OSS↔
frontier is config, not code — DEC-P0-4 contract); `judge` session added to `infra/gpu/jarvis.py`
(DEC-P0-5 ephemeral pattern); judge licence rows added to `LICENSING.md`.

## DEC-P3-2 — MLflow topology: compose service on the existing Postgres (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved).

**Options.** (A) **Compose service; backend = existing Postgres (separate `mlflow` database);
artifacts under `./data/mlflow-artifacts/`.** (B) SQLite file store + `mlflow ui`. (C) Managed.

**Decision.** **A.** Genuine "self-hosted" (CLAUDE.md), DB-backed **model registry** ready for the
P4 adapter, one light container, survives `make down`. `MLFLOW_TRACKING_URI` env-driven.

**Consequences.** Experiments `sutradhar/retrieval` + `sutradhar/generation`; every run logs the
§6.1 reproducibility stamp; a one-time backfill run logs the P2 Table 1 metrics, discharging its
"(MLflow wiring lands in P3)" stamp note.

## DEC-P3-3 — RAGAS: pinned lib; LLM = self-hosted judge; embeddings = BGE-M3 in-session (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved).

**Decision.** RAGAS (version-pinned) computes faithfulness + answer_relevancy through the
DEC-P3-1 judge endpoint with **BGE-M3 embeddings served in the same GPU session** (custom
OpenAI-compatible LLMs/embeddings are first-class in ragas via its model factories /
`BaseRagasLLM`/`BaseRagasEmbeddings`). **Zero external eval APIs** — the whole eval stack is
OSS + self-hosted. Rejected: hand-rolled metrics (loses the recognized-tool signal); frontier-API
RAGAS (only if DEC-P3-1 escalates). The **deterministic no-hallucinated-movie detector remains
the gating faithfulness signal**; RAGAS numbers are reported, the detector gates.

**Consequences.** ragas version recorded in every artifact stamp; RAGAS/judge scores are computed
only inside GPU sessions over recorded transcripts; Tier-1 CI never calls a model.

**Amendment (2026-07-03, P3 task 8 — implementation notes).** (1) Pinned **ragas 0.4.3** (the
modern `metrics.collections` API: `Faithfulness(llm=…)`, `AnswerRelevancy(llm=…, embeddings=…)`
over `llm_factory` + `OpenAIEmbeddings` on OpenAI-compatible clients — exactly the wiring option A
promised). (2) **Dependency pin:** ragas 0.4.3 imports `langchain_community.chat_models.vertexai`
at import time; that module was removed in langchain-community 0.4.x and ragas carries no upper
bound → `[tool.uv] constraint-dependencies = ["langchain-community>=0.3,<0.4"]` holds the
transitive stack; revisit on the next ragas bump. (3) Batch-safety contract: per-sample metric
failures are recorded as `ragas_error` fields, never raised; judge/embeddings unset →
`build_scorer` returns `(None, reason)` and callers skip cleanly.

## DEC-P3-4 — Base-model prompting freeze: system prompt + intent taxonomy + 3 disjoint exemplars (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved). This is the "well-prompted base" honesty bar.

**Decision.** System prompt + 6-label intent taxonomy (`find_by_plot | find_by_title |
list_versions | refine | disambiguate | out_of_catalog`) + **3 handcrafted few-shot exemplars**
(one code-mixed, one multi-turn refine, one NO_MATCH), native Gemma function-calling format.
Rejected: zero-shot (under-prompted base inflates QLoRA lift — dishonest headroom); ReAct
scratchpad (fights native FC tokens; noisier tool-call parsing). Artifacts live under
`evals/prompts/`, hash-pinned (§6.3); **exemplar↔golden-set disjointness is test-enforced**;
P4 evaluates the QLoRA column under the same system prompt.

**Consequences.** Prompt hash appears in every generation artifact + the Table 2 stamp. Confirmed
generation-fixture expansion (grooming Q1): **GS-07 → 5, GS-08 → 3, GS-02-conversational → 4**
(~12–15 generation fixtures), all ground-truth-verified per the P1 gate before freezing.

**Amendment (2026-07-03, P3 task 3 — preamble placement pinned at implementation).** §2.4 required
a "structured preamble the frozen prompt requires" for per-turn intent/slot prediction but left its
placement open. Pinned: the preamble (`INTENT: {"intent": …, "slots": …}`, one JSON line) sits on
the **final prose answer of each user turn** (the assistant message without tool calls), NOT on the
first response — `out_of_catalog` is only knowable *after* tool results, so a first-response
preamble would force the model to predict abstention before searching. Tool-calling messages carry
no preamble. Contract frozen in `evals/prompts/intent_taxonomy_v1.json` (`preamble` block) and
`system_v1.md`; artifacts hash-pinned in `evals/prompts/prompts.lock.json`
(regenerate only via `python -m sutradhar.evals.prompts --write-lock`).
Exemplars use deliberately out-of-golden-set franchises (Ghajini, Okkadu/Ghilli, Interstellar);
disjointness + taxonomy conformance are test-enforced (`tests/test_prompt_artifacts.py`).

**Amendment (2026-07-03, P3 task 5 — bold-title formatting contract, deliberate re-pin).** The
deterministic no-hallucinated-movie detector needs a machine-readable title surface. Added to
`system_v1.md`: every asserted film title is wrapped in `**bold**` and nothing else is bold
(exemplars revised to match — "original" un-bolded). Detector extraction = bold spans (contract)
+ an unbolded `Title (year)` fallback pattern with a language/meta-word guard; a prose invention
carrying neither marker is out of deterministic reach — RAGAS faithfulness is the documented
supplementary net (P3_SPEC §2.4). Lock regenerated: **`prompt_hash 78215ccc…`** (the P4
before/after runs under this hash).

## DEC-P3-5 — Tool-call accuracy scoring: BFCL-style two-level (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved).

**Decision.** (1) **Call-level AST match** — tool name + placeholder-bound, normalized arguments;
(2) **sequence-level headline** — expected sequence appears in order, benign schema-valid extras
tolerated; (3) **schema-validity rate** (hallucinated tool/param rate against
`tool_schema.v0.json`) always reported as an independent number. Rejected: strict-only (one
benign `get_work` zeroes a fixture; noisy at small n); unordered set (order *is* the behaviour
in GS-08). Table 2 carries the sequence-level headline; the artifact carries all three. The same
scorer bytes score base (P4-window top) and QLoRA columns.

## DEC-P3-6 — Langfuse instrumentation boundary: thin explicit wrapper (2026-07-02)

**Status:** Accepted (P3 grooming; user-approved).

**Decision.** `sutradhar.obs.tracing` — explicit `trace()`/`span()` context managers wrapped
around exactly four chokepoints (LLMClient.chat, tool executor, judge client, driver). **No-ops
when `LANGFUSE_*` is unset** (Tier-1 CI, forks) — proven by a fake-sink unit test; no SDK
import-time coupling. Rejected: `@observe` decorators (third-party contract spread through the
codebase; harder to fake); OpenTelemetry+OTLP (heavier than a portfolio harness needs).
P5's FastAPI middleware reuses this exact seam.

## DEC-P3-7 — Langfuse deployment: self-hosted v3 on AIC Cloud `essential-8gb`, API-provisioned, idempotent from-scratch bootstrap (2026-07-02)

**Status:** Accepted (P3 grooming; **user-directed**). Replaces the CLAUDE.md "Langfuse Cloud
free tier" default — CLAUDE.md §Tech stack reconciled (2026-07-02); Cloud free tier remains the
documented fallback.

**Context (verified live against the AIC Cloud API, 2026-07-02).** Public catalogue
`GET https://api.aiccloud.in/api/v1/public/essential-vps-plans` (no auth): the implemented
product is **Essential VPS (Proxmox LXC)** — the docs-page "Cloud VPS API" paths 404.
**`essential-8gb` = 4 vCPU / 8 GB / 80 GB NVMe / 200 Mbps at ₹799/mo** and is the **cheapest tier
with `dedicated_ipv4: true`** (below 8 GB there is no public IPv4 → cannot serve `LANGFUSE_HOST`
— smaller tiers are structurally out, not merely tight). Langfuse v3 = 6-container FOSS compose
stack (web, worker, Postgres, ClickHouse, Redis, MinIO); official guidance ≥ 4 cores/16 GiB is
right-sized to 8 GB + 4 GB swap for our trickle volume (documented low-RAM failure = worker/
ClickHouse OOM); compose ships **no backups** — we add them.

**Decision.** One `essential-8gb` VPS (Ubuntu 24.04), provisioned and configured by an
**idempotent from-scratch bootstrap** (user requirement): `make langfuse-up` →
`infra/langfuse/provision.py`, safe to re-run, converges from any state, sets up Langfuse from
scratch **if not installed** —
- **API phase (find-or-create):** locate instance by name (`GET /api/v1/vps`); if absent →
  wallet pre-check (`GET /api/v1/billing/wallet`, top-up min ₹100) → `POST /api/v1/vps/checkout`
  `{planSlug: essential-8gb, name: sutradhar-obs-01, os: ubuntu-24.04}` →
  `POST /api/v1/vps/checkout/verify` with `sshKeys[]` (Razorpay payment legs are browser-only by
  design: the script prepares the order and waits); if present but stopped → `start`.
- **SSH phase (check-then-act per step):** swap configured → Docker installed → **Docker-in-LXC
  nesting validated day-0** (if blocked: AIC support / KVM-class product, recorded here) →
  langfuse repo at pinned release tag → secrets generated once (all `# CHANGEME`) and persisted →
  **headless init** with pinned project keys → compose up → Caddy TLS (**443 only**; MinIO/CH/PG/
  Redis never exposed; `AUTH_DISABLE_SIGNUP=true`) → backup cron (nightly pg_dump + ClickHouse
  BACKUP + MinIO sync, off-box) → health check → prints `LANGFUSE_HOST` + keys for `.env`.
- **No destructive operation without an explicit flag**; mock-tested (fake API + fake SSH
  transcript) like `extract_session` — CI never spends money.

**Consequences.** `AICCLOUD_API_KEY` added to `.env.example`; `LANGFUSE_HOST` → the VPS (custom
domain supplied at execution; `<ip>.sslip.io` interim for Let's Encrypt); benchmark-cited traces
are **exported (JSON + screenshot) and committed** with each run artifact so evidence outlives
the VPS; standing-evidence budget +**₹799/mo** (12-mo −10% term if dashboard-purchasable — the
checkout API exposes no term field); escalation = API `upgrade` to `essential-16gb` (₹1,599/mo)
on OOM/disk pressure; the same VPS is the intended P6 consolidation host (static surface +
optional read-only MLflow mirror — decided in P6). Rejected: Langfuse Cloud free tier as default
($0 but vendor-hosted, weaker self-hosted story — kept as fallback); on-demand VM (dead trace
links gut the standing evidence).

## DEC-P3-8 — `search_by_plot` in the generation harness: per-fixture replay of the committed retrieval run (2026-07-03)

**Status:** Accepted (P3 task 6, implementation decision under the P3_SPEC §2.1 "artifact-backed
retriever" requirement).

**Context.** The P2 artifact providers are keyed by `sha256(query_text)` and raise
`MissingArtifactError` for unseen text — but the model under test *paraphrases* plot
descriptions, so its emitted `search_by_plot(description=…)` will never hash-match a recorded
query. Running the fusion pipeline on model text would require live embeddings (a neural op the
laptop/CI path forbids, ROADMAP §2).

**Decision.** The driver's tool executor replays the **recorded per-fixture result** from the
committed P2 retrieval-run artifact (`RecordedPlotSearch`: pinned run, winner config, keyed by
the driven fixture id). Fixtures absent from the recorded run (the P3 conversational negatives —
out-of-catalog by construction, golden-validated) replay as honest abstention
(`results: [], abstain: true`).

**Consequences.** (1) No neural op on the laptop; Tier-1 CI and the dry-run are fully
deterministic. (2) Base (top of the P4 window) and QLoRA columns see **byte-identical tool
behaviour** — retrieval quality cannot leak into the generation before/after (the two-table
honesty rule). (3) The model's description *quality* is still measured — by the DEC-P3-5
call-level argument match against `expected_tool_calls`, not by retrieval. (4) Limitation,
stated: replay answers "what would the pinned retriever have returned for this fixture's query",
not "for the model's paraphrase" — acceptable because the golden queries ARE the benchmark's
ground truth; the live P5 query path uses the real retriever with live embeddings.

**Rejected.** (a) Live embeddings in the harness (breaks laptop/CI compute placement and makes
Table 2 depend on GPU retrieval state); (b) abstain-on-unseen for ALL queries (would falsely
abstain GS-07a/b whose recorded queries exist); (c) nearest-recorded-query fuzzy fallback
(non-deterministic tool surface under paraphrase drift — noisier than pinning by fixture).

**Amendment (2026-07-03, P3 task 10 — live API test findings, DEC-P3-7).** First run against the
real AIC API (wallet funded ₹1000, key valid) surfaced two corrections: (1) **the checkout leg is
dashboard-only** — `POST /api/v1/vps/checkout` returns `403 "This endpoint is not available via
API key"` (the grooming-time plan assumed API checkout with browser-only Razorpay legs; in
reality the *entire purchase* happens in the dashboard). `provision.py` now detects the 403 and
stops with exact instructions (plan slug, instance name `sutradhar-obs-01`, OS, SSH key), and the
re-run finds the instance and proceeds — find-or-create and the whole phase-2 bootstrap remain
fully automated. (2) The live plans payload prices as `price_monthly_paise` (not `price_paise`);
a zero-price guard now hard-stops before any spend on catalogue shape drift. Both paths are
mock-tested with the corrected real shapes. Caveat list gains: (c) instance creation is a
one-time dashboard step by design.

**Amendment 2 (2026-07-03, P3 task 10 — phase-2 live bootstrap findings, DEC-P3-7).** Executed the
full from-scratch bootstrap on the real `essential-8gb` box (LXC, Proxmox, NATed). Five findings,
all folded back into `provision.py` + its fake-transcript tests:
(1) **AIC's managed edge firewall opens ONLY the SSH NAT (external 20036 → container 22),
read-only, "cannot be modified"** — inbound 443/80 are permanently blocked, so the planned
Caddy + Let's Encrypt public HTTPS is impossible on this tier. **Public HTTPS now rides an
outbound `cloudflared` tunnel** (systemd-managed quick tunnel; user-confirmed choice; a named
tunnel on a real domain is the drop-in upgrade, planned alongside the P6 static surface). The
health gate checks end-to-end THROUGH the tunnel edge.
(2) The NAT also made `ufw allow <external-port>` a **self-lockout** (internal sshd is 22) —
recovered via API `reinstall` (clean-slate proof of the from-scratch property); ufw now allows
both 22 and the external port.
(3) `swapon` is not permitted inside the LXC container (host-managed swap) → swap step is
best-effort (`optional=True`), warn-and-continue.
(4) The pinned compose does NOT derive `DATABASE_URL` or the three `LANGFUSE_S3_*_SECRET_ACCESS_KEY`
values from `POSTGRES_PASSWORD`/`MINIO_ROOT_PASSWORD` (defaults are literals) → the secrets step
writes them explicitly, and a **heal-in-place** path fixes an existing `.env` without ever
rotating secrets against an initialized volume, then re-ups compose.
(5) Web service name is `langfuse-web`, not `web`.
**Outcome:** instance healthy (`v3.203.3` answering over the tunnel), signup disabled, key-only
sshd, ufw active, backups cron'd; laptop traced the full generation dry-run to it
(run `20260703T012339Z-e7fff041`, GS-08c trace exported + committed, MLflow run `c2fb0eab…`).
Standing cost unchanged (₹799/mo); tunnel adds ₹0.

**Amendment (2026-07-03, P3 task 13 — judge FROZEN, DEC-P3-1).** Human-agreement validation
executed: 30-item blind worksheet (6 coherence + 24 faithfulness, 15 deterministic foils) from
the committed dry-run transcripts, labelled by the user (one label corrected on review —
`fai-GS-07e`, whose non-foil answer carries the seeded invention), judged by **gpt-oss-20b served
by vLLM in one ephemeral A100 session** (create→judge→destroy: machine 438566, up in 297 s,
destroyed, ≈6 min ≪ the ≤1 h envelope). **Result: percent agreement 0.867, Cohen's κ = 0.738 —
PASS (≥ 0.6)**; coherence slice perfect (κ = 1.0, n=6), faithfulness κ = 0.673 (n=24); 0
judge_errors (guided decoding + pinned low reasoning effort worked as governed). All 4
disagreements share one mode: the judge forgave trailing invented-film recommendations on
otherwise-grounded foils — reinforcing that the deterministic detector GATES and the judge is
supplementary (DEC-P3-3). **Frozen judge config:** `openai/gpt-oss-20b @ revision
6cee5e81ee83917806bbde320786a8fb61efebee`, coherence rubric `judge_coherence_v1.md`
(hash b08612f1…), faithfulness rubric `judge_faithfulness_v1.md`, temperature 0,
reasoning_effort low, guided JSON (plus a tested plain-retry fallback). Phi-4-14B alternate and
the frontier escalation were NOT needed. Methodology note: labels were produced by the project
owner with an assistant-flagged single-item review correction; report + verdicts committed at
`evals/judge_validation/report.json`.

---

## DEC-P4-1 — Synthetic-data teacher: Sarvam-M 24B self-hosted, 8-bit weight-only on the ephemeral GPU (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed — spec `docs/phases/P4_SPEC.md` §3 D1 / §7 Q1).
Finalizes the DEC-0001 disjunction ("Sarvam-M 24B *or* a frontier API") in Sarvam-M's favour.

**Options.** (A) **Sarvam-M 24B self-hosted** in one ephemeral session — on the A100 40 GB
(Ampere: no native FP8, so the quantized path is **W8A16 weight-only FP8 via vLLM Marlin
kernels**, ~24 GB weights + KV headroom); recorded plan-B SKU = RTX 6000 Ada 48 GB (~$0.99/hr,
native FP8, DEC-0003's named value alternative). (B) Frontier API (pinned dated version).
(C) No teacher — templated scaffold surfaces only.

**Decision.** **A.** The model DEC-0001 named for exactly this job (+86% romanized-Indic — the
code-mix register is its documented strength); fully self-hosted, zero external keys (the
DEC-P3-1 posture); **Apache 2.0 → teacher outputs unencumbered**, so dataset + adapter publish
cleanly. B is the **recorded escalation only** — trigger: validator rejection rate > 30% or a
failed code-mix quality spot-check after one prompt revision; before any frontier use, the
outputs-ToS row lands in `LICENSING.md` (frontier ToS restrict using outputs to develop
competing models, and access is revocable — P4_SPEC §8 refs). C rejected: templated code-mix is
the shallow register QLoRA would overfit to; teacher data quality is the single biggest lever on
R2 (DEC-0001). Client is OpenAI-compatible + env-driven (`TEACHER_BASE_URL/MODEL/API_KEY`) —
A↔B is config, not code (DEC-P0-4 contract). Cost ≈ 1–2 h ≈ $1–2.

## DEC-P4-2 — Dataset construction: programmatic scaffolds + teacher surface-realization only (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed — P4_SPEC §3 D2).

**Options.** (A) **Deterministic (seeded) scaffolds from gate-visible graph records** — tool-call
sequences, real tool results, labelled answers — with the teacher rewriting **only** user
utterances + assistant prose around **placeholder-locked entities**. (B) Teacher free-generates
whole conversations, post-hoc filtered. (C) Paraphrase-augment the 12 golden fixtures.

**Decision.** **A.** Grounding, v0 tool-call validity, label exactness, and the frozen formatting
contracts (INTENT preamble, bold-title) hold **by construction** — the teacher physically cannot
invent a film or a tool; a diff guard rejects any sample whose locked spans changed. Same
verify-then-keep posture as APIGen/xLAM's three-stage verification and ToolACE (P4_SPEC §8),
applied at construction time instead of filter time. B makes grounding a filter (the exact
failure class the GS-02 gate punishes) and labels an extraction problem; C leaks the eval
surface. Teacher raw outputs cached as a versioned artifact; rejection rate reported on the
dataset card.

## DEC-P4-3 — Training data: ~1,500–2,500 conversations on an entity-disjoint training slice, structurally excluded from every negative surface (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed, no franchise changes — P4_SPEC §3 D3 / §7 Q2).

**Context (the Rev-2 finding that shaped this).** The negative surfaces contain real,
deliberately-uncatalogued films (GS-02: Kaithi, Salaar, Pushpa, Inception; held-out negatives:
Master, Pushpa: The Rise, Jailer, Dangal, Kantara, Andhadhun, Tumbbad, Kumbalangi Nights, Super
Deluxe, Ratsasan, Interstellar, Parasite) whose absence from the title index at the
rapidfuzz-0.80 radius is **test-enforced** (`test_negatives_absent.py`) and baked into the θ
calibration (DEC-P2-5). A draft candidate list violated this four ways and was corrected.

**Decision.** ~10–15 **non-golden** franchises ingested via the existing P1 pipeline + human
gate (gate-visible HIGH/MEDIUM suffices — golden-grade not required), yielding ~1,500–2,500
conversations (95/5 train/val; LIMA/LIMIT evidence: ~1k curated conversations suffice for
behaviour/format learning — more buys register overfit, not capability). **Structural exclusion
rule (test-enforced by `test_ft_training_slice_disjoint`, run BEFORE ingestion):** no training
franchise may contain any film within the rapidfuzz-0.80 radius of (a) any golden fixture incl.
GS-02 negatives, (b) any held-out negative title, (c) the frozen exemplar franchises (Ghajini,
Okkadu/Ghilli, Interstellar — prompt surface for BOTH Table 2 columns). Candidate families
(verified at ingestion): Vikramarkudu→Rowdy Rathore, Anniyan+Aparichitudu, Kanchana/Muni series
(Manichitrathazhu lineage excluded — golden), Pokiri→Wanted, Bodyguard (ml→hi), Arjun
Reddy→Kabir Singh, U-Turn, Mersal/Bigil-class dub tracks, one collision pair, 2–3 NO_MATCH decoy
themes (also radius-checked). **Consequence:** entity disjointness turns the before/after into a
behaviour-transfer claim, not memorization; decontamination scope = golden ∪ exemplars ∪ all
negatives, reported on the dataset card.

**Execution note (2026-07-03, P4 task 3 — ingested).** Final membership: **11 franchises / 15
works / 34 gate-visible versions** (`data-pipeline/training_slice.yaml`; every QID verified live
against Wikidata P31/P577/P364 before commit): Vikramarkudu→Siruthai+Rowdy Rathore,
Anniyan+Aparichitudu(dub), Muni→Kanchana→Kanchana 2 (sequel ladder)+Laxmii(remake),
Pokiri→Pokkiri+Wanted, Bodyguard ml→ta/hi/te, Arjun Reddy→Kabir Singh+Adithya Varma, U Turn
kn→te/ta(bilingual)+hi, Mersal+Adirindhi(dub), Bigil+Whistle(dub), Premam ml→te,
Ala Vaikunthapurramloo→Shehzada, and the Don 1978/2006-vs-Don 2022 (ta) collision pair; 3
NO_MATCH decoy themes in `training_decoys.yaml`. Exclusion test green BEFORE ingestion; closest
protected approach 0.667 (Vikramarkudu~Vikram), threshold 0.80. Ingestion via the existing chain
with a behaviour-preserving `--slice` option on the slice-driven CLIs + dedicated
`data/raw/*-training` snapshot roots (`make ingest-training`) so golden `--offline` replays stay
untouched. **Pipeline fix exposed by the new data shape:** the QID-less version-upsert fallback
keyed on `(work_id, language)` and merged the first same-language remake (Don hi-1978→hi-2006)
into a self-edge; re-keyed to `(work_id, language, release_year)` (regression-tested in
`test_ingest_spine.py`; dub-track idempotency preserved). One conflict opened and human-resolved
(Adithya Varma year: seed+TMDB=2019 vs stale Wikidata P577=2018 → 2019; new
`data-pipeline/resolve_conflicts.py` applies the audited `conflict_resolutions.yaml` — P1 shipped
zero conflicts so no applier existed). Discovered P144 backlinks (Bikram Singha Q4907444,
kn Bodyguard Q4936999) → backlog only, user-reviewed. Post-ingestion: graph = 30 works / 65
versions / 34 edges; full integration suite green incl. `test_negatives_absent` over the grown
index; Table 1 pinned run untouched. Entity fixture list committed:
`finetune/training_slice_entities.json` (emitted by `finetune/export_training_entities.py` from
gate views only).

## DEC-P4-4 — QLoRA configuration: r=16, α=32, all-linear targets, NF4 (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed — P4_SPEC §3 D4). Values live in the hashed
`TrainConfig`; any execution-time change is a recorded amendment, never silent tuning.

**Decision.** r=16, α=32 (α = 2r heuristic), dropout 0.05, **targets = all linear projections**
(attn q/k/v/o + MLP gate/up/down — the QLoRA-paper finding: all-linear matters more than rank;
attn-only is the documented under-performer), NF4 double-quant, bf16 compute, LR 2e-4 cosine,
2–3 epochs (val-loss checkpoint selection), max_seq 4096, packing off. ~10–12 GB peak (DEC-0003
sizing) on the A100 40 GB; adapter ~50–100 MB. Rejected: r=8/attn-only (underperforms, savings
irrelevant at 40 GB); r=64 (no evidence it pays at 4B on ~2k conversations; higher overfit risk).
Sources: QLoRA paper practice, Lightning-AI LoRA-insights series, Unsloth hyperparameter guide
(P4_SPEC §8).

## DEC-P4-5 — Training stack: plain TRL SFTTrainer + PEFT + bitsandbytes; liger OFF (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed — P4_SPEC §3 D5).

**Decision.** Plain **TRL `SFTTrainer` + PEFT + bitsandbytes**, exact versions pinned in the GPU
startup script (authoritative-pins-in-script pattern, DEC-P2-7 precedent; laptop lockfile stays
neural-free). TRL ≥ 0.19 natively supports our exact data shape (`tools` column through the chat
template; `assistant_only_loss=True` via chat-template assistant masks). **Two research-pass
guardrails:** (1) `use_liger_kernel` pinned **OFF** — known TRL bug (issue #3781) silently
discards assistant masks under liger, i.e. loss over the whole sequence with no error;
(2) masking is **test-asserted on rendered token/mask arrays** (`test_ft_render_masking`), never
assumed from config. Rejected: Unsloth (a runtime patching layer over TRL/transformers trainer
classes — a moving layer under the §6.6 "QLoRA numerics repeat" promise; kept as the recorded
fallback ONLY on a measured window overrun threatening the DEC-0003 envelope); Axolotl (config
surface without portfolio signal).

## DEC-P4-6 — QLoRA benchmark prompt: identical full frozen prompt for the headline row; supplementary no-exemplar capture (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed per recommendation — P4_SPEC §3 D6 / §7 Q4).
Resolves the exemplar question DEC-P3-4 explicitly deferred to P4.

**Decision.** The headline Table 2 QLoRA row is captured under the **byte-identical frozen
prompt bundle** (`prompt_hash 78215ccc…` — system + taxonomy + 3 exemplars): one variable
changes between the rows (the adapter), immune to the "you just changed the prompt" critique.
A **supplementary no-exemplar capture** (~15 min marginal in the same window) is recorded in the
artifact + a BENCHMARKS footnote quantifying prompt-token savings — the evidence that the
adapter internalizes the exemplars, and the intended P5 serving config if the adapter is kept.
Rejected: no-exemplar headline (two variables, prompt_hash differs across rows of one table);
skipping the supplementary capture (loses cheap, high-signal internalization evidence).

## DEC-P4-7 — Dataset & adapter hosting: HF Hub with cards; dataset private-first (2026-07-03)

**Status:** Accepted (P4 grooming; user-confirmed — P4_SPEC §3 D7 / §7 Q5).

**Decision.** Adapter → **`sutradhar-gemma4-e4b-qlora-v1`** (public at publish time; registered
in the MLflow registry, DEC-P3-2). Dataset → **`sutradhar-ft-v1`**, **PRIVATE-first**: the
IMDb-derived AKA titles in the graph and the teacher provenance are reviewed against
`LICENSING.md` before any public flip. In-repo for Tier-1 CI: the dataset card, sha256, and a
~100-conversation committed sample (secret-free, fork-safe — DEC-P2-6 posture). Rejected: full
JSONL in git (MBs of generated text; publication-before-review by construction); DVC (a whole
tool for one artifact family HF Hub already covers — ROADMAP §6.1 names HF Datasets + cards).

## DEC-P4-8 — Keep/cut verdict rule, frozen BEFORE the GPU window (2026-07-03)

**Status:** Accepted (P4 grooming; margins frozen under the user's Q3 delegation — P4_SPEC §3
D8 / §7 Q3). Implemented verbatim in `sutradhar.finetune.verdict` and committed before any
training; the rule cannot move after the numbers exist.

**Decision.** **KEEP the adapter iff** (i) strict improvement on **≥ 2 of the 3 primary
metrics** {GS-07 code-mixed intent accuracy, GS-07 slot F1, GS-08 backtracking coherence};
(ii) **at least one improving primary metric clears ≥ +0.05 absolute** — so judge/small-n noise
alone can never trigger a keep at n = 12 fixtures; (iii) **no primary metric regresses**;
(iv) **all guards hold**: GS-02 inventions = 0 on both columns (hard gate regardless of lift),
emitted-call schema-validity ≥ base, tool-call sequence accuracy ≥ base. Anything else → **CUT**:
the adapter is dropped, the finding + numbers recorded here, and P5 proceeds on the well-prompted
base (the DEC-0001 pre-commitment — knowing when fine-tuning did NOT help is the senior signal).
Rejected: strict improvement on every Table 2 metric (relevancy/latency aren't FT targets;
near-ties at small n would cut a genuinely better adapter); any-single-metric (keeps adapters on
noise). `make ft-verdict` computes the verdict as a pure function over the two committed
generation-run artifacts — the 30-second demo path.

## DEC-P4-9 — P4 execution record + THE VERDICT: adapter CUT under the frozen rule (2026-07-04)

**Status:** Accepted (P4 tasks 12–13 execution; verdict computed by `make ft-verdict` over the
committed window artifacts — the DEC-P4-8 rule as frozen on 2026-07-03, applied untouched).

**Verdict: CUT.** Window `ftwin-ce6b6930` (one A100, both columns, byte-identical serving):
clause (i) 1/3 primaries improved — GS-07 slot F1 0.364→0.600 (+0.236, margin met); clause (iii)
2 regressions — GS-07 intent 0.400→0.200, GS-08 coherence 0.667→0.333; clause (iv) 3 guard
failures — schema validity 1.00→0.94, GS-02 = 1 on BOTH columns (base invented "Pushpa", QLoRA
fuzzy-attached "Salaar"). Meanwhile tool-call sequence accuracy rose 0.083→0.417: the adapter
learned FORM (slots, tool discipline) and lost JUDGMENT (intent, coherence). Per the DEC-0001
pre-commitment, **P5 proceeds on the well-prompted base**; a conditional, budget-gated second
iteration is scoped as ROADMAP P4.1 (transcript-diagnosed data defects: unconditional
ask-back-on-ambiguity teaching, missing title-abstention class, no loop-termination examples —
with a pre-registered guard amendment because the base's own GS-02 failure makes a KEEP
unreachable under the original wording).

**Execution amendments (recorded, dated):**
- **Model pin corrected:** `google/gemma-4-E4B` → **`google/gemma-4-E4B-it`**
  (`@ fee6332c1abaafb…`). The bare base ships no chat template; the -it template natively renders
  TOOL_SCHEMA tools + the gemma4 call format. Train-time derivative template
  (`finetune/gemma4_train_template.jinja`) adds `{% generation %}` markers only — proven
  byte-identical over all 96 sealed samples, 0 mask violations. (DEC-0001 follow-up discharged.)
- **`VLLM_SERVE_FLAGS` env** (default `--enable-auto-tool-choice --tool-call-parser gemma4
  --reasoning-parser gemma4`): tool-bearing requests 400 without the family parser — found live;
  guarded thereafter by an ON-BOX tools self-test before any capture marker, plus a laptop
  rc-guard (0 completed fixtures = abort, not a benchmark).
- **Resumable window** (the load-bearing lesson): the adapter is checkpointed to the HF relay
  immediately after training; `FT_RESUME_FROM` lets a fresh window skip training entirely. The
  first window died at merged-serving AFTER a successful train (val loss 0.0502) — the adapter
  was hand-rescued over SSH, and the publishing window resumed from it for ~$1.5 instead of a
  full retrain. Also fixed en route: training pins resolved as a set (`uv pip compile`), isolated
  train venv, LoRA targets resolved against the real module tree (Gemma4ClippableLinear → inner
  `.linear`, multimodal towers excluded), multimodal merge (AutoModelForMultimodalLM +
  KV-sharing tensor graft + processor packaging), `--served-model-name` for the merged column.
- **Tier-1 pin + gate (user-confirmed):** committed `evals/generation_runs/PINNED_RUN` → the live
  base column (env override intact); the GS-02 CI gate became RELATIVE (recomputed inventions ==
  the pinned artifact's recorded value — no NEW hallucinations between windows) because the =0
  target is unmet by both live columns; =0 remains a hard clause in `make ft-verdict` and the
  absolute CI assertion returns when a pinned column achieves it.

**Cost accounting (honest):** teacher session ≈ $7 (est. $2 — think-mode latency + disk/proxy
failures) + window attempts ≈ $6.2 (8 serving/packaging failures incl. a duplicate-loop leak
≈ $0.4, one clean resume window ≈ $1.5) ≈ **$13–14 total vs the ≈ $10 Q6 cap**. Every failure
mode is now a committed fix + test; the resume design caps any future window failure at the
cost of the phase it died in.

---

## DEC-P5-1 — Public gateway: FastAPI only; the optional Java/Spring moat is deliberately CUT (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec `docs/phases/P5_SPEC.md` §3 D1).
Resolves the CLAUDE.md deferral ("the public gateway MAY be Spring Boot…; decide in P5 grooming,
don't assume").

**Options.** (A) **FastAPI only.** (B) Thin Spring Boot gateway proxying FastAPI. (C) Spring Boot
owns orchestration, Python only for ML calls.

**Decision.** **A.** The portfolio thesis is *prototype→production AI engineering* — the
audience's 10-yr Java signal is already established; a pass-through Java proxy carries zero AI
signal while adding a second runtime, container, and CI leg that taxes the 30-second demo path
and the rebuild-from-scratch property. Every deep integration seam (tracing, LLMClient,
guardrails, pydantic contracts) is Python; under B the gateway is pass-through *by construction*,
and C forks the tested tool-loop/guardrail logic away from those seams. The **cut itself is the
interview point** — same class of senior signal as the QLoRA CUT (DEC-P4-9): knowing what not to
build.

**Consequences.** Everything in P5 is Python; `fastapi` + `uvicorn` land as runtime deps.
**Non-streaming JSON responses confirmed for P5** (spec §7 Q3) — SSE/streaming is P6 UI polish if
wanted. The CLAUDE.md "optional Java moat" clause is discharged by this entry.

## DEC-P5-2 — Conversation state: server-side Redis store, in-memory impl for tests/forks (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec §3 D2).

**Options.** (A) **Server-side Redis store (TTL), in-memory implementation for tests/forks.**
(B) Stateless API — client resends full message history. (C) Postgres conversation table.

**Decision.** **A.** The gating story names "API orchestration/state"; Redis is already
provisioned + health-checked in compose but unused — this is its purpose. State = the P3 driver's
proven convention (accumulated OpenAI wire messages, **system message never stored** — the
prompt_hash pins it), given a keyed server home with TTL expiry and turn/size caps. B pushes the
trust boundary to the client (tamperable tool-result history) and bloats requests; C adds
migrations + durable storage for deliberately ephemeral demo sessions.

**Consequences.** `redis` moves from dev-only to runtime dependency; `SESSION_TTL_S` (default
3600) + `API_PORT` env vars; the store is protocol-typed so Tier-1 CI and forks run the in-memory
(or fakeredis) implementation with zero network. P6's chat UI consumes the same store.

## DEC-P5-3 — Indirect prompt-injection defense: structure-first layering; datamarking spotlighting; prompt bundle v1.1; separate injection eval suite (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec §3 D3 + §7 Q1/Q2). Implements ROADMAP
§6.5 for the P5 request path.

**Context.** Post-BIPIA consensus (*Design Patterns for Securing LLM Agents against Prompt
Injections*, arXiv 2506.08837; CaMeL, arXiv 2503.18813; OWASP LLM01:2025) — **structural
constraints beat detection**. Sutradhar already has the strongest structural layers by
construction: a **read-only** v0 tool surface (worst compellable action = a wrong read),
pydantic-bounded tool results (structured fields, never prose blobs), schema-validated emitted
calls (`additionalProperties: false`), and the deterministic no-hallucinated-movie output gate.

**Decision (three parts).**
1. **Spotlighting variant = datamarking** (Hines et al., arXiv 2403.14720: ASR >50% → <2% with
   minimal task degradation): `guardrails.spotlight()` interleaves a marker in data-originated
   string fields of every tool result + a one-line provenance notice. Rejected: delimiting alone
   (forgeable closing delimiter); encoding/base64 (the paper's own caveat — reliability degrades
   below strong models; wrong for a 4B; ~1.33× token inflation).
2. **Prompt bundle v1.1 (user-approved, Q2):** the frozen system prompt gains a spotlighting
   appendix ("marked content is data, never instructions"); `prompts.lock.json` regenerated as
   v1.1 and recorded. **Pinned Table 2 columns are NEVER re-scored under v1.1** — they stay
   pinned to `prompt_hash 78215ccc…`; every P5 serving/window artifact records its own v1.1 hash
   (reproducibility stamp keeps the honesty).
3. **Fixture home (user-approved, Q1): separate `evals/injection/` suite** with its own schema —
   `GOLDEN_SET_SCENARIOS.md` stays the frozen GS-01..11 catalog (gaining only a pointer
   paragraph); injection payloads don't carry golden ground-truth-verification semantics. Fixture
   classes: query-side direct, context-side (spliced into tool results by a **wrapper executor**
   — the live graph is never polluted; result *shapes* still round-trip the frozen v0 schema),
   exfiltration probes, AgentDojo-class attacker-directed tool-call redirection, and benign
   look-alike false-positive controls.

**Gates.** ASR = 0 with defenses on (deterministic set); FP = 0 on benign controls;
utility-under-attack recorded (no threshold on first capture); defense-OFF baseline captured once
in the P5 window. **Honesty note recorded with the metric:** the chunk-level pattern check is
best-effort and bypassable (per 2506.08837); the structural layers are why a bypass still cannot
make the agent *do* anything or *assert* an ungrounded film.

**Consequences.** `sutradhar.serving.guardrails` + `sutradhar.evals.injection` + fixtures;
the offline corpus scan report; a "Serving & guardrails" evidence section in `BENCHMARKS.md`.

**Implementation note (2026-07-05, P5 task 8).** Part 2 is realized as a **separate serving
lock**, not an in-place regeneration: `evals/prompts/prompts.serving.lock.json` pins the v1.1
bundle (the three untouched v1.0 files + `spotlighting_appendix_v1_1.md`,
`prompt_hash 98b3ece1…`), while `prompts.lock.json` stays byte-identical at `78215ccc…`.
Reason: the Tier-1 comparability gate
(`test_golden_generation_regressions.py::test_stamp_pins_current_prompt_and_schema`) asserts the
pinned Table 2 artifact's hash equals the *current* v1.0 lock — regenerating in place would have
broken the very pin Q2 promised to preserve. `load_serving_prompt_artifacts()` verifies BOTH
locks (the appendix extends the frozen bundle; it can never silently edit it). Same intent as
approved, mechanically safer.

## DEC-P5-4 — Live embed/rerank serving + `serve` session: self-contained FlagEmbedding sidecar co-located with vLLM on one ephemeral A100 (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec §3 D4 + §7 Q4).

**Options.** (A) **Self-contained FlagEmbedding sidecar** (BGE-M3 dense+sparse
`/v1/embeddings`+ext, `bge-reranker-v2-m3` `/rerank`) on :8001 beside vLLM :8000.
(B) Extra vLLM pooling processes (`vllm serve BAAI/bge-m3` etc. — supported routes).
(C) Infinity (`michaelfeil/infinity`, MIT).

**Decision.** **A.** The live path must reproduce **bit-for-bit the dense+sparse pair the P2
index and the θ = 0.151747 calibration were built on** — only FlagEmbedding does. B: BGE-M3's
sparse output rides "extra weights" with version-sensitive vLLM support — silent scoring drift
would invalidate the parity check unnoticed. C: verified — **BGE-M3 sparse output is not
supported in Infinity** (issue #146, never landed); the hybrid leg dies. (TEI set aside for the
same sparse gap.) Sidecar follows the DEC-P2-7 self-contained-script + authoritative-pins
pattern; LLM+embedder co-residency on the A100 40 GB was proven in the FT window.

**Consequences.** `RERANK_BASE_URL` env (joins `EMBED_BASE_URL`); `rag/providers.py`
(`HttpEmbeddings`, `HttpReranker`) implement the existing protocols — `Retriever`/
`search_by_plot` code unchanged (the P2 "swap providers, not code" promise). New **`serve`**
session in `infra/gpu/jarvis.py`: create → vLLM + sidecar → health-gate three routes → hold
`SERVE_HOLD_MINUTES` (heartbeat) → **destroy in `finally`**; `make gpu-serve`/`gpu-stop`. vLLM
in-process sleep/wake considered and set aside (JarvisLabs pause/resume + ephemeral posture
already covers it). **Q4 budget approved: ≈ $1–1.5** for the single P5 capture window (parity
check, injection ASR on/off, e2e latency/tokens + `/metrics` snapshot, relevancy backfill) —
within the DEC-0003 envelope. This session is the building block P6's RUNBOOK warm-resume demo
wraps.

**Execution notes (2026-07-05, task 13 — three live-window fixes, ~$3–4 total spend, teardown
`nuke`-clean every time; all recorded because they are the interview-grade "what broke live"
story):**
1. **Endpoint disambiguation.** vLLM *and* the FlagEmbedding sidecar both answer `/health`, and
   JarvisLabs `notebooksn` proxy URLs **carry no port** — so the port-swap heuristic was a no-op
   and smokes/embeddings hit the wrong service. Replaced by `_resolve_serve_endpoints`: the LLM
   is the candidate whose **chat smoke** passes; the sidecar is the other healthy candidate,
   awaited with instance-refresh (the `:8001` proxy registers *after* creation). This was the
   true root cause of the **P4 footnote-¹ null relevancy** (RAGAS had been embedding against the
   judge, not BGE-M3).
2. **`vllm serve --task embed` is rejected by current vLLM** (`unrecognized arguments`, found
   on-box). The judge session now serves BGE-M3 via the **same FlagEmbedding sidecar**
   (`build_judge_sidecar_startup_script`), not the old P3/P4 vLLM embed task.
3. **Sidecar OpenAI-compat.** RAGAS's `OpenAIEmbeddings` sends `input` as a bare string +
   `encoding_format=base64`; the sidecar's `extra="forbid"` 422'd every call. `/v1/embeddings`
   now accepts string|list and tolerates standard extra fields (rerank keeps `extra="forbid"`).

Two-session topology (user-confirmed 2026-07-05): the serve window (Gemma + sidecar) runs
parity/injection/latency; a **separate brief judge session** (gpt-oss-20b + BGE-M3 sidecar) runs
the answer-relevancy backfill — the judge does **not** co-reside with the serve stack on one
A100-40GB. `phase=serve|relevancy` + `merge_run` re-run a single leg and merge into the sealed
artifact (cost discipline while iterating). Sealed run **`servewin-25c029d3`**: parity
Recall@10 = 1.0 / VSR-01/06 = 1.0; injection ASR **0.0 on / 0.273 off**, FP 0.0; latency p50/p95
4535/5395 ms, 76 tok/s; answer_relevancy 0.571 (12/12). MLflow `sutradhar/serving`
run `d453c73e`.

## DEC-P5-5 — Redis caching scope: minimal — endpoint-status cache + session store only (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec §3 D5).

**Options.** (A) **Minimal:** health-status cache (TTL ~30 s) + conversation sessions.
(B) A + retrieval-result cache keyed `(normalized query, RetrievalConfig.stamp())`. (C) Full
response cache.

**Decision.** **A.** Deterministic and reasoning-friendly; the degradation path stops paying a
connect-timeout per request when the GPU is paused (the concrete win). B's saving is real GPU
embed calls, but demo traffic is tiny and stale-on-graph-change would need invalidation
discipline the portfolio slice doesn't earn; **B's key design is documented as the named
future-ops extension** (honours CLAUDE.md's "caching" without inventing invalidation problems).
C is wrong for stateful conversations.

## DEC-P5-6 — Dashboards: Langfuse custom dashboards + custom model price + `/api/metrics`; one-shot vLLM `/metrics` snapshots; NO standing Prometheus/Grafana (2026-07-05)

**Status:** Accepted (P5 grooming; user-approved — spec §3 D6). Also owns the Table 2
footnote-¹ discharge.

**Options.** (A) **Langfuse custom dashboards** (self-hosted v3, DEC-P3-7) + `/api/metrics` JSON
+ committed evidence exports. (B) Grafana + Prometheus scraping vLLM `/metrics`. (C) MLflow-only
window aggregates.

**Decision.** **A**, with B partially adopted and C kept. Langfuse v3's custom-dashboard query
engine renders latency/cost/token widgets from trace metadata — zero new services; the ₹799/mo
VPS earns its keep. **Required setup step (research finding):** a **custom model price
definition** for the self-hosted model, derived from `GPU_HOURLY_USD` ÷ measured tokens/sec —
Langfuse cannot infer pricing for `gemma-4-E4B-it`, so cost dashboards would render $0 without
it. B as a standing stack violates cost discipline for a GPU that is off by default; instead,
**one-shot snapshots of vLLM's Prometheus `/metrics`** (TTFT/TPOT/queue/KV pressure) are captured
into each window artifact. C's per-window aggregate logging is kept (one
`log_generation_run`-style call).

**Consequences.** `obs/cost.py` (`GPU_HOURLY_USD` env, default 0.89 per DEC-0003); trace metadata
feeds the dashboards; dashboard screenshots + `export_trace` JSON + `/metrics` snapshot committed
per window (evidence outlives the VPS). **The RAGAS answer_relevancy backfill over the pinned
base run runs in the P5 window** (root-cause the `ftwin-ce6b6930` null first), discharging the
BENCHMARKS footnote ¹; no other Table 2 cell changes.

---

## DEC-P6-1 — UI stack: Vite + React + TypeScript SPA, built to static assets (2026-07-10)

**Status:** Accepted (P6 grooming; user-confirmed — spec `docs/phases/P6_SPEC.md` §3 D1 + §7 Q3).

**Options.** (A) **Vite + React + TS SPA** built to static assets, served by the app container and
consumable by the static host. (B) Server-rendered Jinja2 + htmx (single Python runtime, zero
node). (C) Streamlit/Gradio.

**Decision.** **A.** "Product UI with citations + trace view" is a named skill row (ROADMAP §4);
A is the only option producing pure static assets both serving surfaces can use, with
componentized version cards/trace view and first-class testing. B reads as "backend engineer
avoided frontend"; C has the weakest product signal and a heavyweight runtime for what should be
static files. **Toolchain pins (research-verified 2026-07-10, spec §8):** Node 24 LTS "Krypton"
(Active LTS; 22 is maintenance), Vite 8.x (stable on Rolldown since 2026-03-12; 8.1 current),
React 19, Vitest 4 (Browser Mode, Playwright provider) — exact versions locked in
`ui/app/package.json` + committed lockfile (the DEC-P0-1 lockfile discipline applied to node).
Node is **build-time only** — never deployed, never serving; ROADMAP §2 compute placement
untouched.

**Consequences.** `ui/app/` is frontend assets (no `sutradhar.*` package, per the P0 stub note);
`make ui-build`/`ui-dev` targets; a node job joins Tier-1 CI; the UI tool-label map is
**generated from `tool_schema.v0.json`** with a drift test (DEC-P1-8 posture extended to the UI).

## DEC-P6-2 — Response delivery: non-streaming JSON + progress states; SSE/token streaming CUT (2026-07-10)

**Status:** Accepted (P6 grooming; user-confirmed — spec §3 D2). Finalizes the DEC-P5-1
"SSE/streaming is P6 UI polish if wanted" deferral as a **deliberate cut**.

**Options.** (A) **Non-streaming JSON (the pinned P5 contract) + staged progress affordance in
the UI.** (B) SSE phase events (tool-call started/finished), final JSON unchanged. (C) Full token
streaming.

**Decision.** **A.** C is structurally wrong for this system: the no-hallucinated-movie **output
gate must see the complete answer before display** — token streaming would show-then-retract an
invention, breaking the guardrail story; and the agent loop interleaves tool rounds, so tokens
arrive late regardless. B adds an endpoint + orchestrator callback surface + a second tested
client path for turns measured at p50/p95 4.5/5.4 s (`servewin-25c029d3`) — a spinner's problem,
not an architecture's. B is recorded as the documented future upgrade if live traffic ever
warrants it.

**Consequences.** One rendering path for live and replayed turns; the P5 ChatResponse contract
changes only additively (DEC-P6-4).

## DEC-P6-3 — Standing surfaces: GitHub Pages static site; demo video as a GitHub Release asset; VPS stays Langfuse-only; MLflow mirror and standing degradation app CUT (2026-07-10)

**Status:** Accepted (P6 grooming; user-confirmed — spec §3 D3 + §7 Q1/Q2/Q4/Q5). Resolves the
DEC-P3-7 "same VPS is the intended P6 consolidation host — decided in P6" clause.

**Options (site).** (A) **GitHub Pages from `site/` via the official Actions deploy flow,
github.io URL.** (B) AIC VPS (Caddy + cloudflared named tunnel on a real domain). (C) Cloudflare
Pages/Netlify.

**Decision.** **A** (limits verified 2026-07-10: 1 GB site / 100 GB-mo soft bandwidth / 10
builds-hr; free on the already-public repo; a non-commercial portfolio fits the Pages terms).
B would hang "always-available" off an outbound tunnel + a ₹799/mo box with **no inbound 443**
(P3 finding) — a single point of failure for the one artifact that must never be down — and a
named tunnel adds a domain-delegated-to-Cloudflare dependency. No custom domain
(user-confirmed); github.io is the canonical URL. Four bundled sub-decisions, all
user-confirmed:
1. **Demo video = GitHub Release asset** (< 2 GiB/file, no bandwidth/total-size limit on release
   assets — verified), linked via `DEMO_VIDEO_URL` from the site and the offline payload; never
   committed to git history or the 1 GB-capped Pages site. Rejected: YouTube unlisted (evidence
   on a revocable off-platform account).
2. **VPS stays Langfuse-only** — its benchmark-cited traces are already exported + committed
   (DEC-P3-7 posture), so the site never depends on it.
3. **Read-only MLflow mirror on the VPS: CUT** — committed screenshots + run links already serve
   the evidence; a public MLflow adds ops/attack surface for zero new proof.
4. **No standing degradation-mode app** — the static surface is the only permanent deployment;
   the app remains a one-command local/live-window artifact. A permanently-up chat endpoint
   would muddy the "nothing inference-side runs 24/7" story CLAUDE.md makes a feature.

**Consequences.** `site/` generator renders the benchmark page **from `BENCHMARKS.md`** (no
hand-copied numbers) + diagram + video + evidence links; link-check + required-assets tests gate
deploy; Pages deploy job on merge to main.

## DEC-P6-4 — Trace-view data source: additive `trace[]` on ChatResponse, assembled in-process (2026-07-10)

**Status:** Accepted (P6 grooming; user-confirmed — spec §3 D4).

**Options.** (A) **`ChatResponse.trace: list[TraceStep]`** assembled by the orchestrator from the
per-call records it already validates. (B) UI queries the Langfuse API. (C) Persist tool records
in Redis behind a new `GET /api/trace/{conversation_id}`.

**Decision.** **A.** The data already exists at the validation seam; deterministic and testable
with the scripted fake client; replay transcripts adapt to the same shape (one rendering path).
B is disqualifying — observability credentials in a browser + VPS coupling for a demo UI. C
grows session state for data the client wanted at turn time. `TraceStep` carries tool name +
arguments + `valid`/`validation_error` + a **bounded** `result_summary` + latency; the change is
additive (existing consumers unaffected); Langfuse remains the ops view via `trace_id`.

**Consequences.** Every rendered tool call re-validates against `tool_schema.v0.json` in CI
(`test_ui_trace_tool_calls_validate`) — no hallucinated tool/param name can reach a rendered
trace. v0 is consumed unchanged: **no version bump** (wording-only status note at P6 exit).

## DEC-P6-5 — App containerization: single multi-stage image; FastAPI serves the built UI (2026-07-10)

**Status:** Accepted (P6 grooming; user-confirmed — spec §3 D5).

**Options.** (A) **One `app` image**: `node:24` build stage → uv runtime stage per the official
Astral multi-stage pattern (`uv sync --frozen --no-dev` with cache mounts; final image = venv +
app only, no uv/compilers); FastAPI mounts the built UI (same-origin, no CORS). (B) Separate
nginx container for the UI. (C) No app container.

**Decision.** **A.** One image + one new compose service = the fastest `make demo-up` and the
cleanest rebuild-from-scratch. B adds a container/CI leg and reverse-proxy config a demo stack
without TLS/scale doesn't earn (the DEC-P5-1 rationale, again) — recorded as the future-ops
split if the app ever fronts real traffic. C fails the "full stack containerized, one-command
bring-up" exit criterion outright.

**Consequences.** `make demo-up` = compose up (postgres, redis, app) → migrate → seed from
recorded fixtures → open UI: fresh clone to working zero-GPU demo in one command, CI-proven from
a fresh checkout; model endpoints stay env-driven and empty by default (off = first-class).

## DEC-P6 execution addendum (2026-07-11)

**Status:** P6 EXECUTED — D1–D5 implemented exactly as decided at grooming (no reopening):
Vite 8.1.4 / React 19.2.7 / Vitest 4.1.10 / Node 24 pinned + lockfile (DEC-P6-1);
non-streaming JSON + deterministic D2 progress states, SSE stayed cut (DEC-P6-2); GitHub Pages
deploy workflow live, video-as-Release-asset wiring in place, VPS untouched/Langfuse-only,
MLflow mirror stayed cut (DEC-P6-3); additive `ChatResponse.trace[]` assembled in-process
(DEC-P6-4); single multi-stage `app` image, `make demo-up` CI-proven from a fresh checkout
(DEC-P6-5). Rehearsal windows measured (545/530 s cold bring-up, 25 s warm `demo-up`,
$0.21 + $0.17, teardown `nuke`-verified both times — `docs/RUNBOOK.md` + BENCHMARKS
degradation evidence). **Demo video recorded and published (2026-07-11):** one-take Playwright
screen capture via the committed recorder (`ui/app/e2e/record_demo.mjs`) — zero-GPU replay →
live turns → the GPU stopped on camera — Release asset `p6-demo-v1`, `DEMO_VIDEO_URL` set
(repo variable + env); a narrated re-record stays optional per the RUNBOOK script.
