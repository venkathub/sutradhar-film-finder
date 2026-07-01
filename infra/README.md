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
<!-- Populated in P0 task 4 (docker-compose bring-up: make up / make down). -->

## On-demand GPU — seed mini-runbook
<!-- Populated in P0 task 8 with evidence from the make gpu-validate run:
     boots? model? tokens/sec glimpse? create→up wall-clock. Full RUNBOOK is P6. -->

## Branch-protection policy
<!-- Populated in P0 task 10: Tier-1 required on PRs to main; Tier-2 is workflow_dispatch-only. -->

## Status
**P0 builds the local compose stack, CI shells, and the one-time GPU validation.** The full
`docs/RUNBOOK.md`, warm-resume demo path, and cost dashboards are **P4/P5/P6**.
