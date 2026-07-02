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

Postgres data persists in the named volume `pgdata` (git-ignored). The pgvector tag was verified on
Docker Hub on 2026-07-01 (P0_SPEC §2.7); pinned exactly for reproducibility. The Work/Version graph
schema is applied on top with `make db-migrate` (P1; Alembic — see `data-pipeline/README.md`).

### Snap-confined Docker workaround (repo outside `$HOME`)

Snap-packaged Docker can only read files under `$HOME` (non-hidden paths) — if the repo lives
elsewhere (e.g. `/data/...`), `make up` fails with `open /var/lib/snapd/void/...: no such file`.
Workaround: stage the compose inputs into a visible `$HOME` dir, preserving the repo layout so the
compose file's `../.env` reference resolves:

```bash
mkdir -p ~/sutradhar-stack/infra
cp infra/docker-compose.yml ~/sutradhar-stack/infra/
cp .env ~/sutradhar-stack/.env      # or: touch ~/sutradhar-stack/.env
cd ~/sutradhar-stack && docker compose -f infra/docker-compose.yml up -d --wait
```

Re-copy after editing either file. (`make db-migrate` and the tests are unaffected — they reach
Postgres over localhost.)

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


## On-demand GPU — seed mini-runbook

`make gpu-validate` runs `infra/gpu/jarvis.py`: it creates a fresh JarvisLabs A100, serves
`LLM_MODEL` on vLLM, health-waits, runs the connectivity smoke against it, captures evidence, and
**destroys** the instance (teardown guaranteed in `try/finally`; `make gpu-nuke` destroys any stray
`sutradhar-p0-validate` instance). Developer / `workflow_dispatch` invoked only — never on a PR.

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
