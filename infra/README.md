# infra

Local dev stack, CI, and on-demand GPU lifecycle for Sutradhar.

Containers and workflows — no `sutradhar.*` import package.

## Planned architecture (P0)
- `docker-compose.yml`: Postgres (+pgvector) + Redis with healthchecks (P0 task 4).
- On-demand GPU (JarvisLabs) ephemeral `create → serve → smoke → destroy` validation under
  `infra/gpu/` (P0 task 8).
- Two-tier GitHub Actions CI: Tier-1 (every PR, no GPU/model) and Tier-2 (`workflow_dispatch`,
  GPU-window) — workflows live in `.github/workflows/` (P0 tasks 9–10).

## Local stack

`infra/docker-compose.yml` brings up two healthchecked services, driven by the repo-root `.env`
(the same `POSTGRES_*` / `REDIS_URL` values `src/sutradhar/config` reads; inline defaults let it run
without one):

| Service | Image (pinned) | Port | Healthcheck |
|---------|----------------|------|-------------|
| `postgres` | `pgvector/pgvector:0.8.4-pg17` | `${POSTGRES_PORT:-5432}` | `pg_isready` |
| `redis` | `redis:7-alpine` | `${REDIS_PORT:-6379}` | `redis-cli ping` |
| `app` (profile `demo`) | multi-stage build (`infra/app/Dockerfile`) | `${API_PORT:-8080}` | `GET /api/health` |
| `mlflow` (profile `mlflow`) | `ghcr.io/mlflow/mlflow:v3.14.0` + psycopg2 (`infra/mlflow/Dockerfile`) | `${MLFLOW_PORT:-5000}` | `GET /health` |

Postgres data persists in the named volume `pgdata` (git-ignored). The pgvector tag was verified on
Docker Hub on 2026-07-01 (P0_SPEC §2.7); pinned exactly for reproducibility. The Work/Version graph
schema is applied on top with `make db-migrate` (P1; Alembic — see `data-pipeline/README.md`).

### The app image + `make demo-up` (P6 task 8, DEC-P6-5)

`infra/app/Dockerfile` builds ONE image = API + built UI: a `node:24` build stage (build-time
only, never deployed) compiles `ui/app` to static assets; the official Astral uv multi-stage
pattern (`uv sync --frozen --no-dev`, cache mounts, non-editable install) produces the venv; the
final `python:3.12-slim` runtime carries only the venv + the runtime artifact tree (alembic,
prompt bundles, pinned replay/retrieval runs, the offline seeding fixtures, `ui/app/dist`) —
no uv, no node, no compilers, no baked-in endpoints or secrets (`tests/test_demo_stack.py`).

```bash
make demo-up    # fresh clone, zero GPU, zero secrets:
                #   .env from template -> compose (demo profile) -> migrate -> seed-graph-ci
                #   -> http://localhost:8080/ (offline notice + replay browser = the 30 s demo)
make demo-down  # stop (pgdata volume kept)
```

**Live flip:** `make gpu-serve` prints `LLM_BASE_URL` / `EMBED_BASE_URL` / `RERANK_BASE_URL`;
export them (+ `RETRIEVAL_RUN`) and rerun `make demo-up` — compose passes them through to the
`app` service; the endpoints are env-driven and empty by default, so the flip is exports, never
a rebuild. CI proves the fresh-checkout path in the tier-1 `demo-smoke` job. The snap-Docker
`$HOME` staging workaround below applies to `demo-up` unchanged (stage the repo, run there).

### Snap-confined Docker workaround (repo outside `$HOME`)

Snap-packaged Docker can only read files under `$HOME` (non-hidden paths) — if the repo lives
elsewhere (e.g. `/data/...`), `make up` fails with `open /var/lib/snapd/void/...: no such file`.
Workaround: stage the compose inputs into a visible `$HOME` dir, preserving the repo layout so the
compose file's `../.env` reference resolves:

```bash
mkdir -p ~/sutradhar-stack/infra/mlflow ~/sutradhar-stack/data/mlflow-artifacts
cp infra/docker-compose.yml ~/sutradhar-stack/infra/
cp infra/mlflow/Dockerfile ~/sutradhar-stack/infra/mlflow/   # mlflow build context
cp .env ~/sutradhar-stack/.env      # or: touch ~/sutradhar-stack/.env
cd ~/sutradhar-stack && docker compose -f infra/docker-compose.yml up -d --wait
```

Re-copy after editing any of these files. (`make db-migrate` and the tests are unaffected — they
reach Postgres over localhost. Under this workaround the MLflow artifact store lives at
`~/sutradhar-stack/data/mlflow-artifacts/` instead of the repo's `data/mlflow-artifacts/`.)

Bring it up / down (the `make up` / `make down` wrappers land in P0 task 5):

```bash
docker compose -f infra/docker-compose.yml up -d --wait   # waits for healthy
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml down            # add -v to drop the pgdata volume
```

Integration tests (opt-in marker `integration`, auto-skip when services are down):

```bash
docker compose -f infra/docker-compose.yml up -d --wait
uv run pytest -m integration            # CREATE EXTENSION vector; + Redis PING->PONG
```

## Self-hosted MLflow (P3, DEC-P3-2)

`make mlflow-up` starts the MLflow tracking server + model registry as a **profile-gated** compose
service (`--profile mlflow`), so plain `make up` stays two-container on the low-spec laptop:

- **Backend store:** a separate `mlflow` database in the same compose Postgres — DB-backed so the
  **model registry works** (needed for the P4 adapter). `make mlflow-up` creates the database
  idempotently (check-then-act) before starting the service, so it works on existing `pgdata`
  volumes, not just fresh ones.
- **Artifact store:** `data/mlflow-artifacts/` (git-ignored), bind-mounted into the container and
  served by the tracking server (`--serve-artifacts`).
- **Client config:** `MLFLOW_TRACKING_URI` (default `http://localhost:5000`) — env-driven, read
  once by `src/sutradhar/config/settings.py`, never hardcoded.
- Experiments: `sutradhar/retrieval` and `sutradhar/generation` (P3_SPEC §2.6; the logging helper
  `sutradhar.obs.mlflow_log` lands in P3 task 10).

```bash
make mlflow-up      # postgres up -> create `mlflow` DB if missing -> mlflow up --wait
make mlflow-down    # stop the service; runs persist in Postgres + data/mlflow-artifacts/
```


## On-demand GPU — seed mini-runbook

`make gpu-validate` runs `infra/gpu/jarvis.py`: it creates a fresh JarvisLabs A100, serves
`LLM_MODEL` on vLLM, health-waits, runs the connectivity smoke against it, captures evidence, and
**destroys** the instance (teardown guaranteed in `try/finally`; `make gpu-nuke` destroys any stray
`sutradhar-p0-validate` instance). Developer / `workflow_dispatch` invoked only — never on a PR.

### Serve session (P5 task 5, P5_SPEC §2.6 / DEC-P5-4)

`make gpu-serve` is the live-demo window: one ephemeral A100 running **vLLM** (`LLM_MODEL` +
`VLLM_SERVE_FLAGS`, `:8000`) and the **embed/rerank sidecar** (`rag-engine/serve_embed_rerank.py`
heredoc-embedded into the startup script — no repo clone, no HF relay; BGE-M3 dense+sparse
`/v1/embeddings` + `bge-reranker-v2-m3` `/v1/rerank`, `:8001`, in its own venv because
FlagEmbedding pins `transformers<5` while vLLM tracks latest). The session health-gates vLLM,
the sidecar, and a chat smoke, then prints the three exports the API needs —
`LLM_BASE_URL` / `EMBED_BASE_URL` / `RERANK_BASE_URL` — and **holds for `SERVE_HOLD_MINUTES`**
(heartbeat log, Ctrl-C to end early), destroying the instance in `finally` either way.
`make gpu-stop` force-tears-down from another terminal. Fake-Deps tests cover
teardown-on-failure, the sidecar-health gate, and the Ctrl-C path.

### Evidence — first validation run (2026-07-01, discharges the DEC-0001 vLLM-on-GPU follow-up)

| Field | Result |
|-------|--------|
| Provider / GPU | JarvisLabs, **A100-PCIE-40GB** (region IN2), ₹84.24/hr |
| Instance | machine_id `437621`, container (`pytorch` template), port 8000 exposed |
| vLLM | `0.24.0`, `pip install -U vllm` on a fresh container |
| Model booted | **`google/gemma-4-E4B`** ✅ (ungated; loaded in ~46 s, `max_model_len=131072`) |
| create → first `/health` 200 | ~5.5 min cold (incl. vLLM install + weight load + `torch.compile` ~52 s); ~100 s on warm compile cache |
| Smoke (`make smoke`) | **`status="up"`**, `sample_token='{"'`, `latency_ms≈196` |
| Throughput glimpse | ~**98 tok/s** single-stream (128 tokens / 1.31 s, greedy) |
| Teardown | `destroy(437621)` → clean; **0 instances remaining** |
| Cost | ₹1884.11 → ₹1855.73 = **₹28.38 (~$0.34)** for the whole create→destroy cycle |

**Findings folded into `infra/gpu/jarvis.py`:**
1. **Reach path:** a container's `public_ip:8000` is firewalled; the port is reachable only via the
   proxied `https://<id>N.notebooksn.jarvislabs.net` endpoint (here the port-8000 mapping was the
   2nd `endpoints[]` URL). `candidate_base_urls()` probes `public_ip` **and** every `endpoints[]`
   entry and uses the first that returns `/health` 200, so this is handled automatically.
2. **Chat template:** base `gemma-4-E4B` ships no chat template → `/v1/chat/completions` 400s
   without one. The startup script now writes a Gemma template and passes `--chat-template`
   (`GEMMA_CHAT_TEMPLATE` in `jarvis.py`); text `/v1/completions` works without it.

> This was the *cold, from-scratch* validation (create→destroy). The sub-2-min **warm-resume** demo
> path (R4) is a separate flow owned by `docs/RUNBOOK.md` in P6. No standing GPU, ever.

## Branch-protection policy

**`main` is protected.** Merges require a green **Tier-1** run and a review; no direct pushes.

- **Required status checks (Tier-1, `.github/workflows/tier1.yml`):** `lint-type-test`,
  `integration`, and `secret-guard` must pass before a PR can merge to `main`.
- **Require a pull request before merging** (at least 1 approving review); **no direct pushes** to
  `main`; require branches to be up to date before merge.
- **Tier-2 (`.github/workflows/tier2.yml`) never gates PRs** — it is `workflow_dispatch`-only and
  runs GPU/eval work deliberately inside an on-demand GPU window (P2/P3 onward). Keeping GPU/secret
  work off the PR path is a cost-and-safety decision (P0_SPEC §2.5, §4).
- **Secrets:** set the repo/org secret **`HF_TOKEN`** so the optional Tier-1 `hf-auth` job can run
  `make hf-check`; it is skipped (not failed) when the secret is absent, and never runs on PRs.

> These are the intended GitHub branch-protection settings for this repo; apply them in
> Settings → Branches. The full operational runbook is `docs/RUNBOOK.md` (P6).

### Applied (2026-07-01)

Classic branch protection and rulesets are **not available on a private free-tier repo**
(GitHub returns 403: "Upgrade to GitHub Pro or make this repository public"). The repo was made
**public** and protection applied as a **repository ruleset** (`main protection (P0)`, id `18391957`,
enforcement `active`) on `refs/heads/main`:

| Rule | Setting |
|------|---------|
| `required_status_checks` (strict) | `lint · type · unit tests`, `compose stack (Postgres+pgvector, Redis)`, `secret guard` |
| `pull_request` | required; `0` approving reviews (solo repo — raise when collaborators are added) |
| `non_fast_forward` | force-pushes to `main` blocked |
| `deletion` | `main` cannot be deleted |

Managed via `gh api …/rulesets`. Required-check contexts are the Tier-1 job **names** in
`tier1.yml`; if a job name changes, update the ruleset context to match.

## Status
**P0 builds the local compose stack, CI shells, and the one-time GPU validation.** The full
`docs/RUNBOOK.md`, warm-resume demo path, and cost dashboards are **P4/P5/P6**.

## P7 — Container hardening (task 5)

Both images run as non-root (`app` user in the app image; uid-1000 `mlflow` in the MLflow
image, matching the dev-host owner of the bind-mounted artifacts dir) and the app image
carries an image-level HEALTHCHECK on `/api/health`. The tier-1 demo-smoke CI job asserts
the BUILT image config via `docker inspect`; `tests/test_dockerfile_hardening.py` guards
the Dockerfiles textually.
