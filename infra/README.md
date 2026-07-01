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
Docker Hub on 2026-07-01 (P0_SPEC §2.7); pinned exactly for reproducibility. **No schema is created
here** — the Work/Version schema is P1. P0 only proves the pgvector image works
(`CREATE EXTENSION vector`).

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
<!-- Populated in P0 task 8 with evidence from the make gpu-validate run:
     boots? model? tokens/sec glimpse? create→up wall-clock. Full RUNBOOK is P6. -->

## Branch-protection policy
<!-- Populated in P0 task 10: Tier-1 required on PRs to main; Tier-2 is workflow_dispatch-only. -->

## Status
**P0 builds the local compose stack, CI shells, and the one-time GPU validation.** The full
`docs/RUNBOOK.md`, warm-resume demo path, and cost dashboards are **P4/P5/P6**.
