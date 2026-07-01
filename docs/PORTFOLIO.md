# Portfolio — Sutradhar

> Resume-ready, quantified bullets. **Stub** — P0 seeds the first bullet; **P6 finalizes** the full
> set with the benchmark evidence (two tables), demo video, and MLflow/Langfuse links.

## P0 — Foundation: reproducible skeleton, cost-aware GPU wiring, two-tier CI

- Bootstrapped a reproducible, env-driven multilingual-RAG platform skeleton — `uv`-locked Python
  monorepo, typed `pydantic-settings` config with secret redaction, Dockerized Postgres + pgvector
  and Redis (healthchecked), and a graceful OpenAI-compatible LLM smoke test that stays green
  whether the on-demand GPU is up **or** off (endpoint-OFF as a first-class path, not a crash).
- Engineered cost-aware **on-demand GPU** automation that creates → serves (vLLM) → smoke-tests →
  **destroys** a rented A100 in one command with teardown guaranteed via `try/finally`; validated
  **Gemma-4-E4B** live at **~98 tok/s** single-stream for **~$0.34** total, with the full
  create→destroy cycle leaving **zero** running instances.
- Gated the repo behind **two-tier CI** (hermetic PR checks — lint, type, unit + integration,
  secret-guard; plus a `workflow_dispatch` GPU-window shell) and a **protected `main`** ruleset;
  the whole stack stands up from a clean clone in one command (`make setup && make up && make smoke`).

_Metrics to be added in later phases: retrieval Recall@10 / version-set recall (P2), base-vs-QLoRA
generation quality + GPU throughput (P3/P4). See `docs/BENCHMARKS.md`._
