# P0 Spec — Foundation: repo, infra, CI, GPU connectivity smoke test

> Phase spec for **P0** of the Sutradhar roadmap. Grounded in `docs/ROADMAP.md` (P0 entry/exit
> criteria + §6 guardrails), `CLAUDE.md` (repo layout, DoD, working agreement), `docs/DECISIONS.md`
> (DEC-0001..DEC-0003 — already settled, not reopened here), `docs/DATA_SOURCES.md`,
> `docs/GOLDEN_SET_SCENARIOS.md`, and `docs/phases/TOOL_SCHEMA.md` (frozen tool contract — P0 does
> not implement or call any tool; it only wires the endpoint they will later run on).
>
> **Status:** **APPROVED — 2026-07-01.** Baseline for P0 execution. Decisions **DEC-P0-1..5** logged
> in `docs/DECISIONS.md`; the §7 confirmations are resolved (all proposed defaults accepted).
> Implementation of the §5 tasks is **not yet started** (awaiting a separate go-ahead).
>
> **Current repo state (grounding):** `main` contains `CLAUDE.md`, `README.md`, `.gitignore`
> (already ignores `.env`, model artifacts, data, DB volumes), and `docs/` (ROADMAP, DECISIONS,
> DATA_SOURCES, GOLDEN_SET_SCENARIOS, phases/TOOL_SCHEMA). No `/infra`, `/serving`, or any code
> exists yet. P0 builds the skeleton the other six phases plug into.

---

## 1. Scope

### In scope
1. **Repo scaffold** matching the `/…` layout in `CLAUDE.md`: `/data-pipeline`, `/finetune`,
   `/rag-engine`, `/serving`, `/evals`, `/ui`, `/infra`, plus `/docs` (exists). Each module gets a
   stub `README.md` (purpose, planned architecture, "not built until P<n>").
2. **Python project bootstrap:** dependency/packaging manager, `pyproject.toml`, lint/format/type
   config (ruff + mypy), `pytest` config, `src/` package layout.
3. **Env-driven config subsystem** (the `/serving` shell): a single typed `Settings` object
   (pydantic-settings) that reads every runtime knob from the environment, plus a committed
   `.env.example`. No secret in code.
4. **Local infra via `docker-compose`:** Postgres **with pgvector** + Redis, with healthchecks and
   documented `make`/task targets to bring the stack up/down.
5. **LLM connectivity smoke test:** a scripted call to `LLM_BASE_URL` that returns a token when the
   on-demand GPU endpoint is up, and **degrades to a clear "endpoint OFF" status (not a crash)**
   when it is down. Green in both states. The endpoint is a **JarvisLabs on-demand vLLM** instance
   (DEC-0003), reached via an **env-driven** URL — an SSH tunnel `http://localhost:8000/v1` or the
   instance URL — never hardcoded.
6. **On-demand GPU (JarvisLabs) full-lifecycle validation:** an **automated** `make gpu-validate`
   that **creates a fresh** JarvisLabs instance (`jarvislabs` SDK / `jl` CLI), `vllm serve`s the
   model, health-waits, runs the smoke in the *up* state, captures evidence, and **destroys** the
   instance (teardown guaranteed via `try/finally` + a `make gpu-nuke` safety target). This (a)
   proves the smoke's *up* state against a **real, from-scratch** GPU, and (b) discharges the
   **DEC-0001 follow-up** — validate **Gemma-4-E4B + vLLM** on the rented GPU before committing
   budget, falling back to `Qwen3-4B-Instruct-2507` if unstable. Ephemeral by design (create→destroy,
   no warm machine); developer/`workflow_dispatch`-invoked, never on a PR. A **seed mini-runbook**
   lands in `/infra/README.md` (full `docs/RUNBOOK.md` is P6).
7. **HF Hub auth check:** `whoami` via the token env var, runnable locally and in CI.
8. **Two-tier CI (GitHub Actions):** Tier-1 (every PR, no GPU, no model calls) = lint + type + unit
   tests + a placeholder "validate against recorded artifacts" step; Tier-2 (`workflow_dispatch`,
   used from P2 on) = eval-harness placeholder that runs only inside a GPU window. Branch-protection
   notes recorded.
9. **Docs seeding:** the P0 decision entries `DEC-P0-1..5` are logged in `docs/DECISIONS.md`
   (done at grooming); create the empty `docs/BENCHMARKS.md` skeleton with **both** table headers
   (Table 1 retrieval, Table 2 generation) so later phases fill them; create a **stub**
   `docs/PORTFOLIO.md` carrying the P0 bullet (P6 finalizes); note the branch-protection policy in
   `/infra/README.md`.
10. **A 30-second demo path:** `make up && make smoke` shows the stack live and the endpoint-OFF
   graceful message (the default, GPU paused).

### Non-goals (explicit — prevents scope creep)
- **No data ingestion.** No TMDB/Wikidata/IMDb/Wikipedia calls, no graph, no Postgres *schema*
  (only the container + pgvector extension availability). That is **P1**.
- **No tool implementation.** `resolve_title`, `search_by_plot`, `get_work`, `get_versions`,
  `refine_filter` are **not** implemented or called. P0 only proves the LLM endpoint is reachable.
- **No embeddings, retrieval, reranking, or vector index.** That is **P2**.
- **No RAGAS/Langfuse/MLflow wiring, no eval harness logic.** Only the empty Tier-2 CI shell and the
  `BENCHMARKS.md` skeleton. That is **P3**.
- **No fine-tuning, no synthetic data, no vLLM serving config.** That is **P4/P5**. P0 validates
  Gemma-4-E4B-on-vLLM reachability *only* as a client smoke test against whatever `LLM_BASE_URL`
  points to (which is normally OFF).
- **No FastAPI application routes / orchestration.** The `/serving` shell in P0 is config + LLM
  client + smoke CLI only; the request-orchestration API is **P5**.
- **No UI.** `/ui` gets a stub README only; the chat/trace UI is **P6**.
- **No Java.** The optional Spring Boot gateway decision is explicitly deferred to **P5 grooming**
  per `CLAUDE.md`; P0 is 100% Python (see §2).
- **No `docs/LICENSING.md`, `RUNBOOK.md`, `PORTFOLIO.md`** — those are owned by P1/P6.
- **JarvisLabs automation is one ephemeral create→validate→destroy run, not standing ops.** P0
  automates exactly that single lifecycle (`make gpu-validate`); it does **not** build the sub-2-min
  **warm-resume demo** path, scheduled/CI-on-PR GPU triggers, cost/latency dashboards, teardown
  pipelines, or the full `docs/RUNBOOK.md` — those are **P4/P5/P6**. No standing GPU, ever.
- **No neural-model op on the laptop or in CI** (ROADMAP §2/§6.6). The smoke test is a *client*; it
  invokes no local model. The one GPU touch runs on the **rented JarvisLabs instance**, then stops.

---

## 2. Design

### 2.1 Component breakdown

| Component | Path | Purpose (P0) |
|---|---|---|
| Repo scaffold + module stubs | `/{data-pipeline,finetune,rag-engine,serving,evals,ui,infra}` | Directory skeleton; each has a stub README |
| Python project | `/pyproject.toml`, `/src/sutradhar/` | Packaging, deps, ruff/mypy/pytest config, importable package |
| Config subsystem | `/src/sutradhar/config/settings.py` | Typed, env-driven `Settings`; the single source of runtime config |
| Env template | `/.env.example` | Every knob documented; no real secret |
| LLM client + smoke | `/src/sutradhar/serving/llm_client.py`, `/serving/smoke.py` (CLI) | Reach `LLM_BASE_URL`; return token or `off` status |
| HF auth check | `/src/sutradhar/serving/hf_check.py` | `whoami` via token env var |
| Local infra | `/infra/docker-compose.yml`, `/infra/README.md` | Postgres+pgvector, Redis, healthchecks, branch-protection notes |
| Task runner | `/Makefile` (see §3 DEC-P0-3) | `setup/lint/typecheck/test/up/down/smoke/hf-check/fmt` targets |
| CI | `/.github/workflows/tier1.yml`, `/.github/workflows/tier2.yml` | Two-tier CI shells |
| Benchmarks skeleton | `/docs/BENCHMARKS.md` | Empty Table 1 + Table 2 headers for later phases |

### 2.2 Data models / schemas (P0)

P0 has **no domain schema** (the Work/Version graph is P1; the chunk/document schema is P2). The
only P0 contracts are:

**(a) `Settings` contract** — env-driven, all IDs swappable, no hardcoding (CLAUDE.md):

| Env var | Meaning | Required | Example / default |
|---|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible base URL of the on-demand vLLM endpoint (JarvisLabs instance URL, or SSH-tunnelled `localhost:8000/v1`) | yes | `http://localhost:8000/v1` |
| `LLM_MODEL` | Served model id (DEC-0001: `google/gemma-4-E4B`, fallback `Qwen/Qwen3-4B-Instruct-2507`) | yes | `google/gemma-4-E4B` |
| `LLM_API_KEY` | Bearer for the endpoint (vLLM `--api-key` / proxy) | no | `EMPTY` |
| `LLM_TIMEOUT_S` | Client timeout for the smoke call | no | `10` |
| `EMBED_MODEL` | Embedding model id (DEC-0002 default) | yes | `BAAI/bge-m3` |
| `RERANK_MODEL` | Reranker id | no | `BAAI/bge-reranker-v2-m3` |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | Postgres+pgvector connection | yes | `localhost/5432/sutradhar/sutradhar/…` |
| `REDIS_URL` | Redis connection | no | `redis://localhost:6379/0` |
| `HF_TOKEN` | Hugging Face Hub token (auth/whoami; artifact push in P4) | yes (for hf-check) | — |
| `JARVISLABS_API_KEY` | JarvisLabs API token for the automated **create→smoke→destroy** lifecycle (`make gpu-validate`, §2.5) | yes (for gpu-validate) | — |
| `GPU_TYPE` | JarvisLabs instance type provisioned per run (DEC-0003 default) | no in P0 | `A100` |
| `TMDB_API_KEY` | TMDB developer key (unused until P1) | no in P0 | — |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Langfuse tracing (unused until P3) | no in P0 | — |

Rules: required-but-missing → a **clear startup error naming the var** (not a stack trace);
secrets never logged (repr/`__str__` redacts `*_KEY`, `*_TOKEN`, `*_PASSWORD`). Postgres runs from
the **official multi-arch `pgvector/pgvector` image**, pinned by tag (e.g. `pgvector/pgvector:0.8.x-pg17`
— exact tag confirmed at bring-up against Docker Hub; §2.7).

**(b) Endpoint health/smoke contract** — the structure returned by the smoke test (reused by P5
graceful-degradation and P3 tracing hooks):

```jsonc
EndpointStatus {
  status: "up" | "off" | "error",   // "off" = unreachable/connection refused/timeout (GPU paused)
  model: string | null,             // echoed LLM_MODEL or the model the endpoint reports
  sample_token: string | null,      // first token(s) of a 1-token completion when up
  latency_ms: number | null,
  detail: string                    // human-readable, e.g. "endpoint OFF — bring up the GPU (see RUNBOOK)"
}
```

`status: "off"` is a **first-class success path**, not an exception — this is the P0 seed of the
"graceful degradation as a feature" thread (ROADMAP P5/P6, Table B row 7).

### 2.3 Key interfaces & contracts

- **`Settings`** (pydantic-settings): loaded once, injected; the *only* place env is read.
- **`LLMClient`** (thin, OpenAI-compatible — see §3 DEC-P0-4): constructed from `Settings`.
  - `health() -> EndpointStatus` — never raises for a down endpoint; catches connection/timeout
    errors and returns `status="off"`. Raises only on genuinely unexpected programmer errors.
  - **Probe sequence (web-verified vLLM routes, §2.7):** (1) `GET {LLM_BASE_URL%/v1}/health` →
    liveness (vLLM returns **200, empty body** when up); (2) `GET /v1/models` → confirm the served
    id matches `LLM_MODEL`; (3) one `POST /v1/chat/completions` with `max_tokens=1` → capture
    `sample_token` + `latency_ms`. Any connection-refused/timeout at step 1 short-circuits to
    `status="off"` (the paused-GPU default).
  - `complete(prompt, max_tokens=1) -> str` — minimal single-token round-trip used by `health()`.
  - Auth via `Authorization: Bearer {LLM_API_KEY}` (vLLM `--api-key`); `EMPTY` when unset.
  - Designed so P3 (tracing) and P5 (orchestration/serving) extend it rather than replace it.
- **`hf_whoami() -> {name} | raises`** — wraps `HfApi().whoami()` (`huggingface_hub` **v1.0**);
  reads `HF_TOKEN` from env (modern HF tokens require API v2). Clear message when the token is
  absent/invalid. `make hf-check` may also shell out to the current CLI `hf auth whoami`.
### 2.4 Request / data flow (P0)

```
make up            → docker-compose: Postgres(+pgvector) + Redis, healthchecked
make smoke         → Settings.load() → LLMClient(settings).health()
                        ├─ endpoint UP  → 200 + token  → EndpointStatus{status:"up",  sample_token}
                        └─ endpoint OFF → conn refused  → EndpointStatus{status:"off", detail:"endpoint OFF …"}  (exit 0)
make hf-check      → hf_whoami() → prints HF username  (fails clearly if token missing)
CI Tier-1 (PR)     → ruff + mypy + pytest (mocked endpoint, both up/off states) + artifact-validate stub
CI Tier-2 (dispatch)→ eval-harness placeholder (no-op until P2/P3), runs only in a GPU window
```

No neural model runs anywhere in this flow (the default); the LLM path is a *client call* to an
external endpoint that is normally paused. The one-time exception is the JarvisLabs UP-state
validation below.

### 2.5 On-demand GPU (JarvisLabs) bring-up for the UP-state smoke

To make the smoke test "green in **both** states" honest, P0 exercises the *up* state once against a
real **JarvisLabs** GPU (DEC-0003) and simultaneously discharges the DEC-0001 follow-up (validate
Gemma-4-E4B + vLLM on the rented GPU). Per **DEC-P0-5** this is done as **full ephemeral
automation** — a fresh instance is **created**, validated, and **destroyed** in one scripted run, so
the proof is reproducible from nothing (no warm machine).

**`make gpu-validate` — create → serve → smoke → destroy (web-verified surface, §2.7):**
```
 1. Create instance   → jarvislabs SDK/jl: launch fresh GPU (GPU_TYPE, e.g. A100-40GB) via JARVISLABS_API_KEY
 2. Serve model       → vllm serve google/gemma-4-E4B --port 8000   (fallback Qwen3-4B-Instruct-2507)
 3. Health-wait       → poll GET /health until 200 (or timeout → fail)
 4. Expose to laptop  → SSH tunnel :8000  →  LLM_BASE_URL=http://localhost:8000/v1
 5. Validate          → make smoke → EndpointStatus{status:"up", sample_token, latency_ms}
                        → record: boots? model? tokens/sec glimpse? create→up wall-clock
 6. Destroy (finally) → jarvislabs SDK/jl: DESTROY the instance — runs even if step 5 failed
```

- **Guaranteed teardown:** destroy is in a `try/finally`; a separate `make gpu-nuke` safety target
  destroys any stray tagged instance so a failed run can never leak a billing GPU.
- **Env-driven, never hardcoded:** the app only ever sees `LLM_BASE_URL`/`LLM_MODEL`; nothing in the
  serving code path knows it is JarvisLabs. The lifecycle script is isolated under `infra/gpu/`.
- **Evidence captured** (log + screenshot + which model booted + rough tokens/sec + create→up time)
  into `/infra/README.md`'s seed mini-runbook — raw material P6 turns into `docs/RUNBOOK.md`.
- **Invocation:** developer-run or a `workflow_dispatch` GitHub job — **never on a PR** (cost +
  secrets). Tier-1 CI stays fully mocked/stubbed.
- **Cost:** one short create→destroy cycle (minutes on A100-40GB, ~$0.89/hr → well under $1), within
  the DEC-0003 envelope; nothing runs 24/7, volume not retained.

> Note the deliberate split from P6: this is the *cold, from-scratch* validation (create→destroy).
> The **sub-2-min warm-resume** path for live interview demos (R4) is a different flow owned by
> `docs/RUNBOOK.md` in P6 and is out of P0 scope.

### 2.6 Tool-schema conformance

P0 neither exposes, implements, nor calls any tool in `docs/phases/TOOL_SCHEMA.md`, so **no schema
change is required and none is proposed**. The tool contract stays frozen at v0 (finalized in P1).
P0's only forward-looking touch is a lightweight test asserting `TOOL_SCHEMA.md`'s tool blocks
parse (see §4), so later phases inherit a machine-checkable contract.

### 2.7 Web-verified tooling facts (accessed 2026-07-01)

External facts checked against current (mid-2026) reality so the scaffold is built on live
contracts, consistent with the mission's env-driven, "rebuild from scratch" posture:

| Choice | Verified fact (2026) | Impact on P0 | Source |
|---|---|---|---|
| **uv** (DEC-P0-1) | De-facto fast Rust-based manager; `uv.lock` + `uv sync` are the reproducible-CI/Docker standard, replacing pip/poetry/pip-tools | Commit `uv.lock`; CI uses `uv sync --frozen`; Docker uses `uv sync` | Astral uv docs; uv lockfile/CI guides |
| **pgvector image** | Official multi-arch `pgvector/pgvector:<ext>-pg{16,17}` images published on Docker Hub, built on the upstream `postgres` image | Pin the tag in `docker-compose.yml`; integration test runs `CREATE EXTENSION vector` | Docker Hub `pgvector/pgvector`; pgvector repo Dockerfile |
| **vLLM endpoints** (DEC-P0-4) | OpenAI-compatible server exposes `GET /health` (200, empty), `GET /v1/models`, `POST /v1/chat/completions`; `--api-key` sets a Bearer token | `LLMClient.health()` probe sequence in §2.3 | vLLM OpenAI-compatible server docs |
| **huggingface_hub v1.0** | CLI moved to `hf …`; auth check is `hf auth whoami`; programmatic `HfApi().whoami()`; **modern tokens require API v2** (legacy 401s otherwise) | Use `HfApi().whoami()` + `HF_TOKEN`; note API-v2 token requirement in `.env.example` | HF Hub auth docs; huggingface_hub v1.0 HfApi ref |
| **JarvisLabs bring-up** (DEC-P0-5) | Modern `jarvislabs` Python SDK + `jl` CLI support programmatic **create/pause/resume/destroy** (API-token auth; `jl run` can auto-destroy after a job), replacing deprecated `jlclient`; docs show `vllm serve --port 8000` reached from the laptop via tunnel | Drives `make gpu-validate` create→serve→smoke→**destroy** (§2.5) | JarvisLabs SDK/CLI docs; JarvisLabs "Serving LLMs" tutorial |
| **Model/embedder ids** | `google/gemma-4-E4B` (fallback `Qwen/Qwen3-4B-Instruct-2507`), `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` remain the DEC-0001/DEC-0002 pins | Used only as **env defaults** in `.env.example`; not loaded in P0 | DEC-0001 / DEC-0002 (already sourced) |

> These validate P0 tooling only; they do **not** reopen DEC-0001/0002/0003. If any tag/route has
> drifted at implementation time, the value is corrected in `.env.example`/compose and noted in the
> commit — never hardcoded.

### 2.8 Python vs Java

**All P0 code is Python** (CLAUDE.md tech stack: FastAPI, pydantic-settings, huggingface_hub,
pytest, ruff, mypy). The optional Java/Spring Boot public gateway is explicitly a **P5 grooming
decision** per CLAUDE.md ("decide in P5 grooming, don't assume") — introducing it now would add a
JVM toolchain to CI for no P0 benefit and pre-empt a later decision. **Recommendation: defer; P0 is
Python-only.**

---

## 3. Decisions to make now

Only choices this phase needs that are **not** already settled in `docs/DECISIONS.md`
(DEC-0001 model stack, DEC-0002 embedder A/B, DEC-0003 GPU/cost are settled and untouched). Each
will be logged as a dated `DECISIONS.md` entry after your sign-off.

### DEC-P0-1 — Python dependency & packaging manager
- **A. `uv` (recommended).** Fast, single-tool resolver+venv+lock; `uv.lock` is reproducible;
  first-class in modern CI; low ceremony. Trade-off: newer tool, less "traditional."
- **B. Poetry.** Mature, widely known, good lockfile. Trade-off: slower, heavier, more config.
- **C. pip-tools + venv.** Minimal, transparent. Trade-off: manual, multi-step, weakest DX.
- **Recommendation: A (uv)** — reproducibility + speed matter for the "rebuild from scratch" and CI
  goals; aligns with the cost/speed discipline theme.

### DEC-P0-2 — Repo Python layout (hyphenated subsystem dirs vs importable packages)
The `CLAUDE.md` dirs are hyphenated (`data-pipeline`, `rag-engine`) — not valid import names.
- **A. Single installable package `src/sutradhar/` with subpackages (recommended)**
  (`sutradhar.config`, `sutradhar.serving`, `sutradhar.rag`, `sutradhar.pipeline`, …); the
  hyphenated top-level dirs hold entrypoint scripts, Dockerfiles, and READMEs that import from
  `sutradhar.*`. One `pyproject.toml`, one venv, one test suite. Trade-off: top-level dir names and
  import names differ (documented mapping).
- **B. Multi-package workspace** (each subsystem its own installable package). Trade-off: closer to
  microservice boundaries but heavy for a solo portfolio; N pyprojects, cross-package version pain.
- **C. Flat `sutradhar/` package at repo root.** Trade-off: simplest, but clutters root and blurs
  the CLAUDE.md module layout.
- **Recommendation: A** — keeps the CLAUDE.md directory story intact while giving one clean,
  testable import root; a short dir→package mapping table goes in the top-level README.

### DEC-P0-3 — Task runner
- **A. `Makefile` (recommended).** Ubiquitous, zero install, CLAUDE.md already says "make/task
  targets," CI-friendly. Trade-off: Make syntax quirks.
- **B. `justfile`.** Cleaner syntax. Trade-off: extra install on laptop + CI.
- **C. Taskfile (go-task).** YAML, nice deps. Trade-off: extra binary, least common.
- **Recommendation: A (Makefile)** — matches CLAUDE.md wording and needs nothing extra installed.

### DEC-P0-4 — LLM endpoint client contract for the smoke test
- **A. OpenAI-compatible client via the `openai` SDK against `LLM_BASE_URL` (recommended).** vLLM
  serves an OpenAI-compatible API natively (DEC-0001) with verified routes `GET /health` (liveness),
  `GET /v1/models` (confirm served id), and `POST /v1/chat/completions` (§2.7); using the standard
  client means P5 serving reuses the exact contract. Trade-off: adds the `openai` dep in P0.
- **B. Raw `httpx` call to `/v1/chat/completions`.** No extra dep, full control. Trade-off: we
  hand-roll what the SDK gives free; drift risk vs the real serving path.
- **C. vLLM-native `/generate`.** Trade-off: couples us to vLLM's non-standard route; breaks the
  env-swappable "any OpenAI-compatible endpoint" posture.
- **Recommendation: A** — standardizing on the OpenAI-compatible contract now is what makes
  `LLM_BASE_URL`/`LLM_MODEL` truly swappable later (vLLM, a frontier API for the P4 data-teacher, or
  a local fallback).

### DEC-P0-5 — JarvisLabs bring-up for the P0 UP-state smoke — **DECIDED: full ephemeral automation**
The smoke test must be green in the *up* state and DEC-0001 wants a P0 GPU validation. How much of
the GPU lifecycle does P0 automate?
- **A. Manual dashboard/`jl` bring-up, no committed automation.** Zero new code/secrets. Trade-off:
  not one-command; not reproducible from scratch.
- **B. Thin `make gpu-up`/`gpu-down` resume/pause helper.** One command, but assumes a pre-existing
  paused instance (a warm machine) and leaves teardown to the operator.
- **C. Full ephemeral automation (CHOSEN): `create → serve → health-wait → smoke → capture →
  destroy`.** One committed script (`make gpu-validate`) driving the `jarvislabs` SDK/`jl` CLI:
  provisions a **fresh** instance, `vllm serve`s Gemma-4-E4B, waits for `/health`, runs `make smoke`,
  records evidence, then **destroys** the instance — with teardown in a `finally` so a failed smoke
  never leaks a running GPU. Needs `JARVISLABS_API_KEY` + a `GPU_TYPE`/config; no persistent
  instance id.
- **Decision: C.** Chosen because it best embodies the mission's "**rebuild from scratch, never
  depend on a warm machine, volume deleted after use**" principle (ROADMAP §2, CLAUDE.md infra): the
  UP-state proof is fully reproducible from nothing, and cost is bounded to one short run
  (create→destroy, minutes on A100-40GB, well under $1). **Guardrails:** (1) teardown is guaranteed
  (`try/finally` + a `make gpu-nuke` safety target that destroys any stray tagged instance); (2) the
  script is **developer/`workflow_dispatch`-invoked only — never on a PR** (cost + secrets); (3)
  first-run model-weight download makes this slower than a warm resume, which is fine for a one-time
  validation (the *sub-2-min warm-resume demo* path stays P6/RUNBOOK, a different flow).

> These five are the whole P0 decision surface. Everything else (ruff/mypy/pytest as the lint/type/
> test stack, pydantic-settings for config, `pgvector/pgvector` image, Postgres+Redis choice)
> follows directly from `CLAUDE.md`/ROADMAP and is applied without a separate decision.

---

## 4. Test strategy

P0 **does not touch retrieval or generation**, so **no eval set or metric threshold gates P0** —
this is stated explicitly so no one mistakes the missing eval gate for an omission. The P0 gate is
behavioural: the stack comes up from a clean clone and the smoke test is green in **both** endpoint
states. (Golden-set/RAGAS gates begin in P2/P3.)

### Unit tests (Tier-1 CI, no GPU, no model)
- **Config:** `Settings` loads from env; a missing **required** var raises a clear, var-named error;
  `repr`/log redacts `*_KEY`/`*_TOKEN`/`*_PASSWORD`; defaults match `.env.example`.
- **LLM smoke — OFF path:** mock the client transport to raise connection-refused/timeout →
  `health()` returns `status="off"` with the guidance `detail`, **process exit 0** (never raises,
  never crashes). This is the graceful-degradation seed.
- **LLM smoke — UP path:** mock a 200 with a one-token completion → `status="up"`, `sample_token`
  populated, `latency_ms` set.
- **LLM smoke — ERROR path:** mock a 500/malformed body → `status="error"` (distinct from `off`).
- **HF check:** `hf_whoami` with a fake token (mocked) returns a name; missing `HF_TOKEN` → clear
  error message, not a traceback.
- **TOOL_SCHEMA parse test (forward-looking):** assert the five tool blocks in
  `docs/phases/TOOL_SCHEMA.md` are present and their fenced signatures parse — so future phases that
  emit tool calls can validate against a machine-checkable contract (no hallucinated tool/param
  names). *P0 emits no tool calls, so the emitted-call validation itself is deferred to P4/P5, per
  the task instruction — this test seeds it.*

### Integration tests (local; a CI job that boots the compose services)
- `docker-compose up` → Postgres reachable; **`CREATE EXTENSION IF NOT EXISTS vector;` succeeds**
  (proves the pgvector image, not the P1 schema); Redis `PING` → `PONG`.
- `make smoke` against a **local stub OpenAI server** returns `status="up"`; against nothing
  listening returns `status="off"` exit 0.

### CI meta-tests
- Workflows are valid YAML and reference only Tier-1-safe steps in the PR job (no model calls, no
  GPU); Tier-2 is `workflow_dispatch`-only.
- No secret literals in tracked files (a simple grep/`gitleaks`-style guard); `.env` is
  git-ignored (already true) while `.env.example` is tracked.

### One-time GPU validation (NOT in CI — invoked deliberately, evidence captured)
- **JarvisLabs full-lifecycle smoke (§2.5):** `make gpu-validate` **creates** a fresh rented GPU,
  serves **Gemma-4-E4B on vLLM** (or the Qwen fallback), health-waits, runs `make smoke` → expects
  `status="up"` with a real `sample_token`, then **destroys** the instance. Records: does the model
  boot on the A100-40GB? tokens/sec glimpse? create→up wall-clock. This is the only place a neural
  model is touched in P0, and the instance exists only for the duration of the run.
- **Teardown test:** simulate a smoke failure (inject an error after create) and assert the
  `finally` still **destroys** the instance — a leaked billing GPU is a defect. `make gpu-nuke`
  covers stray instances.
- This intentionally does **not** run on PRs (no GPU/secrets in Tier-1 CI, cost discipline); it is
  developer- or `workflow_dispatch`-invoked. CI relies on the captured evidence + the mocked/stub
  unit and integration tests above.

### Named golden regression tests — status for P0
The GS-01/GS-06/GS-02/GS-04/GS-05/GS-10 regression tests named in the grooming brief are
**retrieval/graph/generation** tests owned by **P1 (graph)**, **P2 (retrieval)**, and **P3
(generation)**. They **cannot** run in P0 (no graph, no index, no model). P0's obligation is to
**stand up the Tier-1/Tier-2 CI harness those tests will plug into** and the `/evals` stub dir. This
is called out so the coverage gap is deliberate and traceable, not forgotten.

---

## 5. Task breakdown (ordered, independently committable)

1. **Python bootstrap** — `pyproject.toml` (pkg manager per DEC-P0-1), `src/sutradhar/` package,
   ruff + mypy + pytest config, lockfile. *(commit: `chore: python project bootstrap`)*
2. **Module scaffold** — create `/data-pipeline /finetune /rag-engine /serving /evals /ui /infra`
   with stub READMEs + dir→package mapping table in root README. *(`chore: scaffold module dirs`)*
3. **Config subsystem** — `Settings` (pydantic-settings) + redaction + `.env.example` + unit tests.
   *(`feat(config): env-driven settings`)*
4. **Local infra** — `infra/docker-compose.yml` (Postgres+pgvector, Redis, healthchecks) +
   `infra/README.md` + integration test (extension + ping). *(`feat(infra): local compose stack`)*
5. **Task runner** — `Makefile` targets (`setup/fmt/lint/typecheck/test/up/down/smoke/hf-check`).
   *(`chore: make targets`)*
6. **LLM connectivity smoke** — `LLMClient` (OpenAI-compatible per DEC-P0-4) + `serving/smoke.py`
   CLI + unit tests for up/off/error. *(`feat(serving): llm connectivity smoke test`)*
7. **HF Hub auth check** — `hf_check.py` + `make hf-check` + test. *(`feat(serving): hf whoami check`)*
8. **JarvisLabs full-lifecycle GPU validation** — per DEC-P0-5: `infra/gpu/jarvis.py` +
   `make gpu-validate` (create→`vllm serve`→health-wait→`make smoke`→capture→**destroy** in
   `try/finally`) + `make gpu-nuke` safety target; run it once, capture evidence (boots? model?
   tokens/sec? create→up time) into `infra/README.md`; discharge the DEC-0001 vLLM-on-GPU follow-up.
   *(`feat(infra): jarvislabs ephemeral create→smoke→destroy gpu validation`)*
9. **Tier-1 CI** — `.github/workflows/tier1.yml` (lint/type/test/artifact-validate stub) + secret
   guard. *(`ci: tier-1 pr checks`)*
10. **Tier-2 CI** — `.github/workflows/tier2.yml` (`workflow_dispatch` eval-harness placeholder) +
   branch-protection notes in `infra/README.md`. *(`ci: tier-2 gpu-window shell`)*
11. **Docs seed** — `docs/BENCHMARKS.md` (empty Table 1 + Table 2 headers), stub `docs/PORTFOLIO.md`
    (P0 bullet), update `README.md` status. *(DEC-P0-1..5 already logged in `docs/DECISIONS.md` at
    grooming.)* *(`docs: seed benchmarks + portfolio stub`)*

---

## 6. Definition of Done (P0)

Instantiates the generic DoD from `CLAUDE.md` for this phase:

- [ ] Code complete and matches this approved spec; repo scaffold matches the `CLAUDE.md` layout;
      each module has a stub README.
- [ ] `.env.example` covers `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, DB/Redis, `HF_TOKEN`,
      `TMDB_API_KEY`, Langfuse keys — **no secret in code** (redaction test + secret guard green).
- [ ] `docker-compose` brings up Postgres(+pgvector) + Redis; `make` targets documented and run.
- [ ] **Two-tier CI** scaffolded and green: Tier-1 (PR, no GPU) runs lint + type + unit/integration
      tests + artifact-validate stub; Tier-2 (`workflow_dispatch`) shell present. Branch-protection
      notes recorded in `/infra/README.md`.
- [ ] **HF Hub `whoami`** verified via token env var, locally and in CI.
- [ ] **Connectivity smoke test** returns a token when the GPU endpoint is up and a clear
      "endpoint OFF" status (exit 0, no crash) when down — **green in both states**.
- [ ] **JarvisLabs full-lifecycle validation (§2.5):** `make gpu-validate` **creates** a fresh
      instance → `vllm serve` Gemma-4-E4B → health-wait → `make smoke` returns `status="up"` with a
      real token → **destroys** the instance (teardown guaranteed; no leaked GPU). Evidence
      (log/screenshot + tokens/sec + create→up time) captured in `/infra/README.md`. **DEC-0001
      follow-up discharged** (Gemma-4-E4B + vLLM boots on the rented GPU, else fallback recorded).
- [ ] Unit + integration tests written and passing in CI.
- [ ] **Eval thresholds: N/A for P0** (no retrieval/generation) — recorded as a deliberate,
      documented exception, not an omission.
- [ ] **Benchmark tables: none updated with results.** P0 **creates the `docs/BENCHMARKS.md`
      skeleton** (Table 1 retrieval + Table 2 generation headers) that P2 (Table 1) and P3/P4
      (Table 2) later fill. *(This is the phase's only legitimate "updates neither table" case.)*
- [ ] Module READMEs updated; `docs/DECISIONS.md` carries **DEC-P0-1..5** (logged at grooming,
      2026-07-01).
- [ ] Runs cleanly from scratch: fresh clone + `.env` → `make setup && make up && make smoke`.
- [ ] **30-second demo path:** `make up && make smoke` (shows live stack + graceful endpoint-OFF
      message); `make hf-check` (shows HF auth).
- [ ] Resume-ready quantified bullet drafted for `docs/PORTFOLIO.md` (e.g. "reproducible,
      env-driven, two-tier-CI skeleton with cost-aware on-demand-GPU wiring and graceful
      degradation, standing up from a clean clone in one command").

---

## 7. Resolved confirmations (all accepted 2026-07-01)

1. **DEC-P0-1..5** (§3) — **CONFIRMED:** uv / single-`src/sutradhar`-package / Makefile /
   OpenAI-compatible client / full ephemeral JarvisLabs `create→smoke→destroy` automation. Logged as
   `DEC-P0-1..5` in `docs/DECISIONS.md`.
2. **`docs/PORTFOLIO.md` in P0 — CONFIRMED (stub now, P6 finalizes).** P0 creates a **stub**
   `PORTFOLIO.md` carrying the P0 quantified bullet (satisfies the generic DoD); P6 owns the final
   version. Handled in **task 11**.
3. **`docs/BENCHMARKS.md` skeleton in P0 — CONFIRMED.** P0 creates the **empty two-table skeleton**
   (Table 1 retrieval + Table 2 generation headers) that P2/P3/P4 later fill. Handled in **task 11**.
4. **Local stub LLM server for tests — CONFIRMED.** Unit tests **mock the client transport**
   (up/off/error paths); a **tiny local stub OpenAI server** is used only for the integration
   `make smoke` path — **no external network / no real endpoint in CI**. The real-endpoint check is
   the out-of-CI `make gpu-validate` run (§2.5).

---

## 8. Sources (web-verified, accessed 2026-07-01)

Facts underpinning §2.7 and the P0 decisions. DEC-0001/0002/0003 sources live in `docs/DECISIONS.md`
and are not repeated.

- **uv (packaging, DEC-P0-1):** Astral uv docs (`docs.astral.sh/uv`); uv lockfile reproducibility +
  CI/Docker guides (pydevtools, uv 2026 guides).
- **pgvector Docker image:** Docker Hub `pgvector/pgvector` tags; pgvector GitHub repo `Dockerfile`;
  pgvector Docker deployment notes (multi-arch, `postgres`-based, pg16/pg17).
- **vLLM OpenAI-compatible server (DEC-P0-4):** vLLM "OpenAI-Compatible Server" docs
  (`docs.vllm.ai/.../serving/openai_compatible_server`) — `/health`, `/v1/models`,
  `/v1/chat/completions`, `--api-key`.
- **Hugging Face auth:** HF Hub Authentication docs; `huggingface_hub` v1.0 `HfApi` reference
  (`whoami`); `hf auth whoami` CLI; modern-token/API-v2 requirement (huggingface_hub issue #3479).
- **JarvisLabs on-demand GPU (DEC-P0-5 / DEC-0003):** JarvisLabs Python SDK + `jl` CLI docs
  (`docs.jarvislabs.ai/sdk`, `/cli` — `create/pause/resume/destroy`, `jl run` auto-destroy; replaces
  deprecated `jlclient`); JarvisLabs "Serving LLMs" tutorial (`vllm serve --port 8000` reached from
  the laptop). Pricing/sizing already sourced in DEC-0003.
