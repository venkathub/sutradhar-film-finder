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

## P2 — Hybrid multilingual RAG baseline: measured retrieval gate, GPU-free CI evals

- Built a **hybrid multilingual retrieval engine** (BGE-M3 dense + BGE-M3 sparse lexical weights,
  both scored **inside Postgres/pgvector** — `sparsevec` inner product, no second vector store —
  fused with RRF k=60, reranked by bge-reranker-v2-m3, aggregated chunk→Work) that hit the phase
  gate **on the first pass: Recall@10 = 1.000 (gate ≥ 0.90) in all 6 ablation cells** and
  **version-set recall = 1.0** on the flagship cross-lingual case — a Tamil "Papanaasam" query
  returns the Malayalam original + all 4 Indian remakes, relationship-labelled, end-to-end
  (query → Work → typed version set); the planned 9B embedder A/B was **skipped by the
  pre-registered decision rule**, saving the GPU spend.
- Engineered a **compute-placement discipline** that keeps every neural op off the laptop and CI:
  one ephemeral A100 session (**~10 GPU-minutes ≈ $0.22**, HF-Hub relay, created→destroyed,
  MANIFEST-sealed artifacts) embedded 833 texts and precomputed the **full 44k-pair query×chunk
  reranker matrix**, making chunking/fusion/depth ablations and **every CI eval run zero-GPU** —
  Tier-1 CI recomputes all gating metrics (Recall@k, MRR, version-set recall, no-hallucination)
  from a 0.5 MB committed artifact on every PR and blocks merge on regression.
- **Calibrated abstention with the failure documented, not hidden**: NO_MATCH threshold tuned on
  a 24-query held-out negative set (calibration/test split) → **zero false accepts on all
  out-of-catalog queries (NO_MATCH recall 1.0)**; measured that raw cross-encoder scores rank
  code-mixed positives below fluent-English negatives (zero-false-reject infeasible — witnessed
  per-fixture), chose no-hallucination over no-false-reject, and recorded the interleave as the
  quantified fine-tuning headroom target for P4.

## P3 — Eval + observability harness: judge governance, 0-hallucination gate, all-self-hosted MLOps

- Built a **CI-gated generation benchmark harness** (multi-turn conversation driver over an
  OpenAI-compatible endpoint) where the tools array is **generated from a frozen JSON tool
  schema** and every model-emitted call is **schema-validated before execution** — 3/3 seeded
  fault classes (hallucinated tool, hallucinated parameter, wrong-typed argument) caught and
  scored, proven by a committed 12-fixture dry-run whose only validity/faithfulness deductions
  are **exactly the two deliberately seeded faults** (schema-validity 35/36, faithfulness 17/18,
  **0 hallucinated movies on the out-of-catalog gate**); Tier-1 CI recomputes every metric from
  the committed transcripts with the same scorer bytes on each PR and fails on any drift — or on
  a stale artifact after any prompt/schema re-pin.
- Froze **LLM-as-judge governance the measured way**: a self-hosted cross-family judge
  (gpt-oss-20b via vLLM, pinned to an HF revision SHA + hashed rubric + temp 0 + guided JSON)
  validated against a 30-item blind human-labelled sample with deterministic foils in **one
  ephemeral A100 session (~6 min, <$1, auto-destroyed)** — **Cohen's κ = 0.738** (coherence
  slice κ = 1.0), with the disagreement analysis motivating the design: the deterministic
  no-hallucinated-movie detector **gates**, the judge stays supplementary; RAGAS runs through
  the same judge + self-served BGE-M3 with **zero external eval APIs**.
- Stood up the **all-self-hosted observability stack** and kept it honest about cost and
  reality: MLflow (compose, DB-backed registry) + Langfuse v3 self-hosted on a ₹799/mo VPS via
  an **idempotent from-scratch bootstrap** (find-or-create API provisioning + check-then-act
  SSH steps, mock-tested with fake API/SSH transcripts — CI never spends) that survived real
  infra: a provider firewall that only opens SSH (solved with an outbound cloudflared tunnel),
  a NAT self-lockout (recovered via API reinstall — proving the from-scratch property), and
  three compose env-derivation gaps — **five live findings, each folded back into code, tests,
  and the decision log**; benchmark-cited traces are exported and committed so evidence
  outlives the infrastructure.

## P4 — QLoRA fine-tune: a rigorously measured negative result (the senior signal)

- **Built a fully synthetic, record-grounded, code-mixed SFT dataset (2,000 multi-turn
  tool-calling conversations, 6,426 teacher rewrites)** where hallucination is structurally
  impossible: tool sequences are generated *from* the frozen JSON tool schema, every tool result
  is a byte-recorded output of the live graph tools, entities are sentinel-locked through the
  Sarvam-M 24B surface-realization pass (28% of rewrites machine-rejected back to scaffold
  surfaces), and every training utterance is decontaminated (rapidfuzz < 0.80) against the eval
  set, the prompt exemplars, AND all negative fixtures — the decontamination gate caught 3
  scaffold templates reproducing protected eval phrasings before a single GPU minute was spent.
- **Ran the one-time train+benchmark window as a resumable, phase-checkpointed protocol** over an
  HF-Hub relay (no SSH dependency): base column, QLoRA train (r=16 all-linear NF4, assistant-only
  loss asserted on token arrays — guarding a known TRL silent-mask-drop bug), adapter
  checkpointed to the relay the moment training ended, merged-model captures, and a judge+RAGAS
  pass — all in one A100 rental with byte-identical serving (tokens/sec divergence 5.7%). When
  the first window died *after* training, the resume design replayed everything else for ~$1.5
  instead of a retrain.
- **Published the honest verdict: CUT, under a keep/cut rule frozen before any number existed**
  (≥2-of-3 primary improvements, ≥ +0.05 margin, zero regressions, zero-hallucination guards).
  The adapter improved tool-call sequence accuracy 0.083 → 0.417 and slot F1 +0.24 but regressed
  intent accuracy and multi-turn coherence — and the transcript-level diagnosis traced every
  regression to three specific training-data defects (documented, with a budget-gated fix phase
  scoped). Negative result, positive evidence: benchmark tables, MLflow registry entry, exported
  Langfuse traces, and a public adapter card all say so out loud.

## P5 — Serving, API & guardrails: the gating story as a deployed request path

- **Shipped the end-to-end request path** (FastAPI orchestration: normalize → hybrid retrieve →
  schema-validated tool-calling agent loop → deterministically gated, cited answer) and proved it
  live: the winner retrieval config re-validated **through live GPU providers at Recall@10 = 1.0
  and version-set recall = 1.0** — bit-identical to the recorded-artifact gate, demonstrating "the
  live path swaps providers, not code" — plus server-side multi-turn backtracking over HTTP and
  six named golden regressions asserted on the *served* JSON payload, not just the repository.
- **Engineered layered indirect-prompt-injection defense and measured it live: attack-success
  rate 0.273 → 0.000** (14-fixture BIPIA/AgentDojo-style suite: query- and context-side attacks,
  exfiltration, tool-redirection, plus benign false-positive controls at 0.0 FP). Structural
  layers (read-only tool surface, schema-validated calls, a deterministic no-hallucinated-movie
  output gate) do the real work; datamarking-spotlight + the gate close the rest — framed against
  the 2025 design-patterns literature. A live capture even surfaced a real gate gap (an
  ungrounded title with the year inside the bold span slipped a naive check), which I fixed,
  regression-tested, and re-confirmed live at ASR 0.
- **Made graceful degradation a feature, not a gap:** with the on-demand GPU off (the default),
  `POST /api/chat` returns a structured HTTP 200 offline payload and `/api/replay/{fixture}`
  serves committed benchmark transcripts — the full Papanasam story is demonstrable with **zero
  GPU and zero database** (integration-tested both states), with per-request token/cost/latency
  accounting feeding `/api/metrics` and Langfuse.
- **Captured the deployed-path benchmark on a strict budget and debugged it honestly:** one
  `make serving-benchmark` window (two ephemeral A100 sessions, teardown-guaranteed, ~$3–4 across
  live iterations) recording **API p50/p95 latency 4535/5395 ms at 76 tok/s**, and it **root-caused
  and discharged the P4 answer-relevancy evidence gap** (null for all fixtures) down to a
  proxy-URL endpoint-disambiguation bug — recomputing RAGAS answer_relevancy = **0.571 (12/12
  scored)** over the pinned base run. Sealed artifact + MANIFEST, MLflow `sutradhar/serving`, and
  the honest "what broke live" trail all committed.
