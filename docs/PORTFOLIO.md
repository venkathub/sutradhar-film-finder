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

## P1 — Catalog + remake-graph: verified cross-lingual film graph with a human-gated LLM pipeline

- Built a **provenance-first cross-lingual film graph** (Postgres/SQLAlchemy: Works, per-language
  Versions, 5 typed edge kinds) from **4 sources** (Wikidata SPARQL, TMDB, IMDb dumps, Wikipedia
  API — snapshot-first, hash-stamped, fully offline-replayable), reaching **100% version coverage
  on all 5 flagship remake franchises (31/31)** with every record carrying `confidence` +
  `sources[]` and multi-source disagreements queued as conflicts — **never silently resolved**
  (verification gate enforced *in the schema* via ground-truth views + triggers, 279 tests).
- Shipped an **LLM extraction → human review pipeline** that recovered the remake edges Wikidata
  is missing: an ephemeral A100 session (~$1, created→destroyed) ran Gemma-4-E4B over 27
  Wikipedia articles with vLLM guided decoding (parse failures **92.6% → 7.4%**, measured) and a
  verbatim-evidence guard; the audited human gate (58 candidates → 19 confirmed, precision
  0.352 honestly reported) added **6 verified edges beyond Wikidata** — including the
  remake-of-a-remake edge (Chandramukhi→Apthamitra) no structured source asserts.
- Engineered **deterministic cross-script title resolution** (ITRANS transliteration chosen by
  measurement over IAST/ISO — 89.7 vs 80.9 avg similarity on real native↔popular pairs — plus
  rapidfuzz) so `பாபநாசம்`, "Papanaasam", and `दृश्यम` all resolve correctly; froze a
  machine-checkable **tool contract** (JSON Schema + CI conformance gates) and a **25-fixture
  golden set** across 11 scenario categories, every fixture validator-proven against
  HIGH/human-verified graph records only.
