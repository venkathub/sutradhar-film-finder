# P7 Spec — Credibility Hardening: Doc-Truth Reconciliation, Security & Data-Integrity Fixes, Evidence Strengthening

> **Status: EXECUTED (approved 2026-07-18; tasks 1–20 complete 2026-07-19 — see §8).**
> Grounded in `docs/ROADMAP.md` §2 (P7 entry,
> CLOSED — APPROVED FOR GROOMING 2026-07-18), the post-P6 external review findings (to be logged
> verbatim as DEC-P7-1 at execution start, per P7 entry criteria), and a code survey of the
> touched paths (file/line references below verified 2026-07-18).
>
> **Prime directive of this phase:** *claims are reconciled to evidence, never the reverse.* No
> frozen benchmark artifact is re-scored. No metric changes except from a future, clearly-dated
> capture window (which is budget-gated and NOT part of default P7). Default GPU spend: **$0**.

---

## 1. Scope

### In scope (the four DoD groups from ROADMAP P7)

**A. Doc-truth reconciliation** — every standing claim matches recorded evidence, with dates:
1. FT-verdict truth: `CLAUDE.md` (Subsystems §5, "QLoRA owns the behaviour" framing), `README.md`,
   and `ROADMAP.md` §1(b) still narrate "QLoRA measurably beat the base" — the recorded verdict is
   **CUT** (DEC-P4-9); the served model is the **well-prompted base**. Rewrite to "the FT question
   was settled by a pre-registered verdict (CUT, DEC-P4-9)" — the negative result IS the MLOps proof.
2. Demo-timing truth: replace every "sub-2-min resume" claim (`CLAUDE.md` ×2, `ROADMAP.md` R4,
   DEC-0003 cross-refs) with the **measured 545 s (~9.1 min) ephemeral-create** posture from
   `docs/RUNBOOK.md`; document the live-demo choreography (start bring-up at meeting open, walk
   the recorded replays while it boots).
3. Cost truth: `docs/PORTFOLIO.md:181` presents "≈ $12–17" (an estimate) as the project total.
   Recompute one dated project-total **from the audited per-phase actuals in DECISIONS.md**
   (e.g. DEC-P4-9's ≈ $7 teacher, ≈ $13–14 P4 total; P0/P1/P2/P5 window actuals). No total is
   pre-asserted in this spec — the task is the recomputation.
4. Two-layer hallucination framing: wherever "0 hallucinated movies" appears (`PORTFOLIO.md:75`
   et al.), state model-layer **GS-02 = 1 ⚠ (both Table 2 columns)** beside served-layer
   **0-via-output-gate** (`sutradhar.serving.guardrails.output_gate`), referencing the DEC-P4-9
   relative-CI-gate amendment. The honest version is the stronger story.
5. New README section — **"Why the demo works despite base tool-call accuracy 0.083"**:
   deterministic orchestration, schema-validated tool loop (`validate_emitted_call` before every
   execution), and the output gate carry the product. Confronted before an interviewer asks.
6. Golden-set target truth: `GOLDEN_SET_SCENARIOS.md:19` targets "≥ 100 fixtures total"; the
   committed set has **34** golden fixtures (+12 held-out negatives, +14 injection fixtures).
   Either meet it or formally revise it down with a dated DEC entry (→ decision D3, §3).
7. Git-verifiable pre-registration: annotate DEC-P4-8/DEC-P4-9 (and the P4.1 amendment if ever
   run) with commit SHAs proving the rule predates the numbers. Main is squash-merged (DEC-P4-8
   and the numbers both land in `e817139`), so the proof comes from **PR #5 branch commits** —
   **verified reachable via the GitHub API (2026-07-18)**: rule frozen in `0fd94477`
   (2026-07-04T03:06:31Z, "frozen DEC-P4-8 verdict rule — committed BEFORE the window") vs
   window executed in `710025c5` (2026-07-04T10:13:20Z) and Table 2 published in `95ea3d39`
   (2026-07-04T10:28:54Z). Task 12 records exactly these SHAs + timestamps in the DEC entries.

**B. Correctness & data-integrity fixes** (each ships with a regression test):
1. **Spine upsert provenance bug** — `src/sutradhar/pipeline/wikidata.py` `_upsert_work`
   (L276–309) / `_upsert_version` (L312–370): update branches assign
   `existing.sources = sources_to_jsonb(sources)` wholesale (replacing provenance later merged by
   `enrich_tmdb`/`load_akas`, which correctly union via `precedence.py`), and `_upsert_version`
   writes `existing.confidence = confidence.value` unconditionally with no `human_verified`
   check. Fix: **merge** `sources[]` (union + dedupe, reusing the `precedence.py` union logic)
   and never downgrade confidence or overwrite curated sources on a `human_verified = true`
   record.
2. **NO_MATCH threshold placement** — `CALIBRATED_NO_MATCH_THRESHOLD = 0.151747` is hardcoded at
   `src/sutradhar/rag/retrieve.py:47`, silently decoupled from the calibration artifact
   (`evals/retrieval_runs/20260702T135315Z-f6583183.calibration.json`) it was derived from. Move
   the binding into the retrieval-run artifact load path with a **staleness check**: if the live
   index / embed model does not match the calibration run's stamp, **hard fail** (no silent
   reuse). Mechanism → decision D2, §3.
3. **DB-owned uniqueness** — one Alembic migration (#3) adding: unique index on `person.tmdb_id`
   (currently bare, `c636e4be00a5` L101); a unique constraint backing the `candidate_edges`
   dedup key `(edge_type, src_title_raw, dst_title_raw, source_page)` (currently app-side
   SELECT-then-skip only, `extract.py` L202–212); a unique constraint on the
   `(work_id, language, release_year)` version fallback key used by `_upsert_version`
   (L330–337). Pre-migration duplicate audit + documented resolution for any existing violation.

**C. Security & serving hardening (the paid request path):**
1. **Auth + rate limiting on `POST /api/chat`** (`src/sutradhar/serving/app.py` L220) — the
   endpoint that burns GPU seconds is currently open. Token-header auth; limits keyed by **auth
   token first, client IP fallback** (per-IP alone is weak behind NAT/proxies). Redis is already
   in compose + deps; slowapi is NOT yet a dependency. Implementation → decision D1, §3.
   Documented in the RUNBOOK demo flow.
2. **500 envelope leak** — `trace_and_envelope` (app.py L211–216) returns
   `f"{type(exc).__name__}: {exc}"[:300]` to the client (same pattern at L361/L372). Fix:
   generic message + request id to the client; `str(exc)` to logs only.
3. **Container hardening** — `infra/app/Dockerfile` runs as root with no image-level
   `HEALTHCHECK` (compose has one; the image does not). Add non-root `USER` + `HEALTHCHECK`;
   same `USER` treatment for `infra/mlflow/Dockerfile`.
4. **Repo hygiene** — survey verdict: already largely clean (`mlruns/` untracked + ignored;
   `git status` clean; `data/artifacts/` ignored). Remaining: delete local
   `data/artifacts/retrieval/.staging/` debris, add an explicit `.staging/` ignore rule, and add
   a CI hygiene tripwire so regressions can't land.

**D. Evidence strengthening (no frozen artifact re-scored):**
1. **Blind test-retest second pass** on the existing 30-item judge-validation worksheet
   (`evals/judge_validation/worksheet.yaml`; current single-annotator judge–human κ = 0.738).
   **Resolved at grooming (DEC-P7-6):** the second pass is by the *original* annotator, so it is
   reported as **intra-rater (test-retest) κ** — explicitly labelled as an upper-bound *proxy*,
   never presented as a human–human inter-rater ceiling. Blind = labels + foil key stripped,
   items reshuffled; ≥14-day gap satisfied (original labels 2026-07-02/03). Plus a
   **real-items-only κ** (foils excluded). Protocol committed as
   `evals/judge_validation/PROTOCOL.md` **before** labelling (per arXiv 2606.00093). A true
   second human, if later available, adds the inter-rater ceiling additively.
2. **Generation golden-set expansion**: GS-07 n=5 → **n ≥ 10**, GS-08 n=3 → **n ≥ 10**. New
   fixtures only ADD; each carries `expected_intent` / `expected_slots` /
   `expected_tool_calls` that must validate against `tool_schema.v0.json` in Tier-1. Scored in
   the **approved capture window** (DEC-P7-7, ~$3–5) as new dated BENCHMARKS rows.
3. **Injection suite widening, AgentDojo-style**: add obfuscation variants to
   `evals/injection/fixtures.yaml` (14 → ~25): encoding (base64/leet), homoglyph,
   split-across-fields; scored as the paired utility/security metrics (benign utility,
   utility-under-attack, ASR). Published claim re-framed: a static suite bounds *these* attacks
   only; ASR 0.000 is never presented as adaptive-attacker robustness (2025 adaptive-attack
   results defeated all twelve published defenses). New fixtures runnable with the existing
   `evals/run_injection_eval.py` off/on harness; authoritative numbers only from a dated window.
4. **`docs/SCALE.md`** — the 50k-film design note making ROADMAP §6.6's "named as future ops"
   concrete: pg_trgm/GIN for title resolution (replacing the O(N) in-Python rapidfuzz scan),
   pgvector **HNSW with 0.8.x iterative index scans for filtered search** (our queries always
   filter through gate views/language — the known HNSW-recall trap; BGE-M3's 1024 dims fit HNSW
   limits), discovery-mode ingestion beyond the seed YAML, paginated SPARQL, delta re-ingest.
   Design note only — no implementation.

### Non-goals (scope fence — from ROADMAP P7, restated as binding)

- **No re-scoring of any frozen benchmark artifact.** Table 1 / Table 2 numbers do not move.
- **No streaming, no response/semantic cache, no multi-worker serving, no load testing.**
- **No catalog scale-up implementation** — `docs/SCALE.md` is a design note only.
- **No P4.1 execution** (separately gated, own budget, own approval).
- **GPU spend: one approved window only.** The fixture-scoring capture window (~$3–5) is
  **approved (DEC-P7-7, 2026-07-18)** and runs as the final task; all doc/bug/security work
  proceeds independently and never waits on it. No other GPU spend in P7. Observability for the
  window = **Langfuse Cloud free tier** (the DEC-P3-7 documented fallback — the self-hosted VPS
  was destroyed; standing cost now ₹0, recorded in the doc-truth pass).
- **No new tools, no TOOL_SCHEMA change.** v0 remains frozen; P7 emits no tool-surface change
  (verified: nothing in scope touches tool signatures). If any fix were to force one, that would
  be a v0→v1 bump + DEC entry — none is anticipated.
- **No reopening settled decisions** (vector store, chunking, θ *value*, judge config, FT verdict,
  serving topology, UI stack). The θ fix relocates the *binding*, not the calibrated value.

---

## 2. Design

### 2.1 Component breakdown (what changes where)

| Component | Files (verified) | Change |
|---|---|---|
| Docs (truth pass) | `README.md`, `CLAUDE.md`, `docs/ROADMAP.md` §1(b)/R4, `docs/PORTFOLIO.md`, `docs/BENCHMARKS.md` (annotations only), `docs/GOLDEN_SET_SCENARIOS.md`, `docs/RUNBOOK.md`, `docs/DECISIONS.md` | Reconciliation edits, dated; DEC-P7-x entries; SHA annotations on DEC-P4-8/9 |
| Data pipeline | `src/sutradhar/pipeline/wikidata.py` (L276–370) | Source-merge + human_verified guard in `_upsert_work`/`_upsert_version` |
| RAG engine | `src/sutradhar/rag/retrieve.py` (L39–63), `src/sutradhar/rag/index.py::load_index` (L54+) | θ binding moved to artifact + staleness check |
| DB schema | new `alembic/versions/<rev3>_uniqueness_hardening.py` | 3 unique constraints/indexes + duplicate pre-audit |
| API layer | `src/sutradhar/serving/app.py` (L199–260, L361, L372) | Auth dependency, rate limiter, scrubbed error envelope with request id |
| Infra | `infra/app/Dockerfile`, `infra/mlflow/Dockerfile`, `.gitignore`, CI workflow | non-root USER, HEALTHCHECK, hygiene tripwire |
| Evals | `evals/judge_validation/` (new `worksheet.blind.yaml`, `report_interrater.json`, protocol doc), `evals/golden/gs07_code_mixed.yaml`, `evals/golden/gs08_backtracking.yaml`, `evals/injection/fixtures.yaml` | Second-rater flow; fixture expansion; injection variants |
| New doc | `docs/SCALE.md` | 50k design note |

### 2.2 Key designs & contracts

**Auth + rate-limit request flow (paid path):**

```
POST /api/chat
  → [1] auth: Authorization: Bearer <token> checked against API_AUTH_TOKENS (env, comma-sep;
        never in code; absent env var in non-dev ⇒ endpoint returns 503 "auth not configured",
        dev/test profile may explicitly disable)
  → [2] rate limit: key = sha256(token) if authed else client IP (proxy-aware via
        X-Forwarded-For only when TRUST_PROXY=1); backend = existing Redis; limits env-tuned
        (e.g. CHAT_RATE_LIMIT="10/minute") — fakeredis in tests
  → [3] existing session caps (SessionLimitError) unchanged
  → [4] orchestrator (unchanged) → output_gate (unchanged)
401 (bad/missing token) and 429 (limited) use the same envelope shape as other errors.
Health/static/degradation routes stay unauthenticated (they serve the always-available surface).
```

**Error envelope contract (all 5xx):**

```jsonc
{ "error": "internal_error", "request_id": "<uuid>" }   // client-visible
// log line (server only): request_id + type(exc) + str(exc) + stack
```
`request_id` is generated per-request in `trace_and_envelope` and echoed in a response header
(`X-Request-Id`) — this also strengthens the Langfuse trace-correlation story.

**θ binding + staleness contract:**

```
retrieval-run artifact (committed): evals/retrieval_runs/<run>.calibration.json  → theta
                                    evals/retrieval_runs/<run>.meta.json         → embed_model,
                                                                                   manifest_sha256, code_sha
config:  CALIBRATION_RUN_ID (single pinned constant, e.g. in rag/config or env)
load path (index.py::load_index / RetrievalConfig): read theta FROM the pinned run's
  calibration artifact; assert run.embed_model == live index embed_model AND the index the
  run was calibrated on matches (manifest/index-version comparison). Mismatch ⇒ raise
  StaleCalibrationError (hard fail — "recalibrate before serving"), never silent reuse.
retrieve.py keeps NO numeric literal; tests/test_calibration.py updated to assert the
  loaded value round-trips the artifact (value itself unchanged: 0.151747).
```

**Spine upsert provenance contract:**

```
on update of an existing Work/Version:
  sources_new = union_dedupe(existing.sources, incoming.sources)   # reuse precedence.py union
  if existing.human_verified:
      confidence unchanged (never downgraded); curated fields not overwritten by pipeline values
  else:
      confidence = max(existing, incoming) by tier (HIGH > MEDIUM) — re-ingest can raise, never lower
conflicting scalar values (e.g. year) still go to the conflicts queue (existing behaviour, unchanged)
```

**Second-pass protocol (committed BEFORE labelling as `evals/judge_validation/PROTOCOL.md`):**
binary scale identical to the original pass (`human_label` 0/1 per `kind`); ties impossible
(binary); invalid/unscorable output labelled 0 with a note; no abstention option; annotator sees
`worksheet.blind.yaml` (labels + foil key stripped, order reshuffled) only. Outputs (per
DEC-P7-6): **intra-rater test-retest κ** (labelled as such — proxy, not a human–human ceiling),
judge–human κ for the second pass, real-items-only κ (foils excluded) — computed by the existing
`cohens_kappa()` (`src/sutradhar/evals/judge.py:228`) via a new
`judge_validate.py testretest` subcommand. Frozen `report.json` is not modified; the new report
is additive (`report_testretest.json`).

### 2.3 Python vs Java

**Python only.** All touched code is the existing single `src/sutradhar/` package (DEC-P0-2);
the optional Java/Spring moat was deliberately CUT in DEC-P5-1 and P7 explicitly does not reopen
settled decisions. Nothing in this phase (doc edits, an Alembic migration, FastAPI middleware,
eval tooling) would benefit from a second runtime; adding one would itself be the kind of
unforced complexity this phase exists to remove.

### 2.4 Tool-schema conformance

P7 exposes/implements/changes **no tool**. TOOL_SCHEMA.md v0 stays frozen; at most it gains a
wording-only P7 status note (consistent with the P2–P6 notes) recording that the new GS-07/GS-08
fixtures' `expected_tool_calls` are validated against `tool_schema.v0.json` by the existing
DEC-P1-8 validator. No version bump, nothing new for DECISIONS.md on the tool surface.

---

## 3. Decisions to make now (not settled in DECISIONS.md)

> **CONFIRMED 2026-07-18** — the user accepted all four recommendations; logged as
> **DEC-P7-2 (D1), DEC-P7-3 (D2), DEC-P7-4 (D3), DEC-P7-5 (D4)**, plus **DEC-P7-6**
> (test-retest reframe, Q1) and **DEC-P7-7** (capture window approved + Langfuse Cloud, Q2).
> Retained below as the options record.
>
> Settled and NOT reopened: θ value (DEC-P2-5), judge config (DEC-P3-1), FT verdict (DEC-P4-9),
> FastAPI-only gateway (DEC-P5-1), Redis scope (DEC-P5-5), containerization shape (DEC-P6-5).
> The ROADMAP P7 entry names slowapi as the "default candidate" and prescribes token-first
> keying — but the implementation choice itself is not yet a DEC entry. These four are.

**D1 — Rate-limiter + auth implementation (→ DEC-P7-2)**

| Option | Trade-offs |
|---|---|
| **(a) slowapi + existing Redis (recommended)** | Purpose-built Starlette/FastAPI limiter; Redis backend matches DEC-P5-5 infra; moving-window strategies for free; small, well-known dep — reviewer-legible. Cons: adds a dep; its decorator API needs a thin wrapper to key by token-then-IP |
| (b) Hand-rolled middleware (Redis INCR/EXPIRE fixed window) | Zero new deps; full control of the token-first key. Cons: we own correctness (race conditions, window semantics) on a security control — exactly where NIH is a smell |
| (c) fastapi-limiter | Async-native, Redis-backed. Cons: less maintained than slowapi; brings aioredis assumptions; no advantage over (a) for our sync-style app |

**Recommendation: (a)** — matches the ROADMAP's named candidate; custom key function
(token-hash → IP fallback) is ~10 lines. Auth itself: static bearer token(s) from
`API_AUTH_TOKENS` env (comma-separated, allowing per-interviewer tokens + revocation by removal)
checked in a FastAPI dependency — no JWT/OAuth machinery for a portfolio demo endpoint (that
would be scope creep, and DEC-P5-x already keeps sessions server-side).

**D2 — θ binding mechanism (→ DEC-P7-3)**

| Option | Trade-offs |
|---|---|
| **(a) Pinned `CALIBRATION_RUN_ID` + load-time read of the committed calibration artifact + staleness assertion (recommended)** | Threshold lives with the run that produced it (the ROADMAP's literal ask); staleness = artifact-vs-index stamp comparison; Tier-1 testable offline; re-calibration = commit new artifact + bump one pinned id (a reviewable diff) |
| (b) Write θ into the index artifact's `meta.json` at build time | Strongest coupling (index and θ physically inseparable). Cons: requires regenerating/annotating the existing frozen index artifact — brushes against "no frozen artifact is modified"; conflates two artifacts with different lifecycles (index build vs calibration) |
| (c) `NO_MATCH_THRESHOLD` env var | Trivially swappable. Cons: exactly the failure mode P7 is fixing — a number detached from its provenance, silently reusable against the wrong index |

**Recommendation: (a).** The frozen calibration artifact already contains θ; nothing is
re-scored, only re-bound.

**D3 — Golden-set ≥100 target: meet or revise (→ DEC-P7-4)**

Current: 34 golden + 12 held-out negatives + 14 injection = 60 committed fixtures.

| Option | Trade-offs |
|---|---|
| **(a) Formally revise the target + targeted expansion (recommended):** GS-07→10, GS-08→10, injection→~25 ⇒ ~48 golden + 12 negatives + ~25 injection ≈ **85 total**, with a dated DEC entry revising "≥100 golden" to the measured, slice-justified number | Honest (no silently abandoned target — the DEC entry IS the fix); effort lands where the review said n is weakest (generation slices); no GPU needed to author |
| (b) Meet 100 literally (author ~66 new golden fixtures across all GS) | Preserves the round number. Cons: bulk fixtures in already-saturated retrieval slices (Recall@10 = 1.000 across the grid) add authoring + verification cost with near-zero evidence value; risks quantity-over-quality fixtures — the opposite of credibility |
| (c) Revise down with no expansion | Cheapest. Cons: leaves the review's actual finding (tiny generation n) unaddressed; the κ/variance criticism stands |

**Recommendation: (a).** The review's point is metric *stability* on generation slices, not the
round number.

**D4 — Injection-suite scoring frame (→ DEC-P7-5, small)**

| Option | Trade-offs |
|---|---|
| **(a) Adopt AgentDojo's paired metrics (benign utility / utility-under-attack / ASR) as the reported triple for the widened suite (recommended)** | Field-standard framing; makes the "defense that breaks utility is no defense" trade-off explicit; maps cleanly onto the existing off/on runner |
| (b) Keep ASR-only reporting | No harness change. Cons: exactly the one-number overclaim the review flagged; can't show the defense preserves benign behaviour |

**Recommendation: (a);** authoritative numbers still only from a dated capture window — the
Tier-1 suite validates fixture shape and replays recorded runs.

---

## 4. Test strategy

P7 touches the retrieval load path and the eval fixtures, so the standing regression suite must
stay green throughout, and every fix lands with its own regression test. **No new authoritative
metric is produced in this phase** (any scoring of new fixtures awaits an approved capture
window).

**Named golden regression tests (existing, must remain green — Tier-1, on committed artifacts):**
- Version-set recall = 1.0 on **GS-01** and **GS-06** (Drishyam family / Drishyam-2 lineage).
- No-hallucinated-movie on **GS-02** (retrieval abstention + output-gate layers).
- Dub-vs-remake on **GS-04** (Baahubali — `is_official_dub_of` never conflated with remake).
- Sibling-vs-remake on **GS-05** (Devdas — siblings of a source work, not a chain).
- False-merge on **GS-10** (Vikram collision — Kamal-Haasan overlaps never merge Works).

**New unit/integration tests per fix:**

| Fix | Test |
|---|---|
| Spine upsert provenance | `test_ingest_spine.py` additions: re-ingest after `enrich_tmdb` ⇒ `sources[]` is the union (nothing lost); re-ingest over a `human_verified=true` version ⇒ confidence and curated sources untouched; non-verified re-ingest can raise MEDIUM→HIGH but never lower |
| θ binding | `test_calibration.py` rework: θ loads from the pinned run artifact and equals 0.151747; tampered/mismatched `embed_model` or manifest ⇒ `StaleCalibrationError`; no numeric θ literal remains in `rag/` source (grep-based tripwire) |
| Migration #3 | Alembic upgrade/downgrade round-trip on a seeded DB; inserting duplicate `person.tmdb_id` / candidate-edge dedup-tuple / `(work_id, language, release_year)` ⇒ IntegrityError; app-side dedup path still reports `proposals_duplicate` (no behaviour change on the happy path) |
| Auth + rate limit | 401 without/with-bad token; 200 with valid token; 429 after limit exceeded (fakeredis); token-keyed and IP-fallback keying each exercised; health/static routes unaffected; envelope shape on 401/429 matches contract |
| Error envelope | Handler raising an exception containing a fake DSN ⇒ response body contains neither the DSN nor the exception message, does contain `request_id`; log record contains both; same for the L361/L372 status paths |
| Docker hardening | CI/step or test asserting the built image's config: `User` non-empty/non-root and `Healthcheck` present (`docker inspect` in the existing image-build CI job) |
| Repo hygiene | Tripwire test: `git ls-files` intersects a denylist (`mlruns/`, `data/artifacts/`, `*.staging*`, weight extensions) ⇒ fail |
| Doc truth | Claims-lint tripwire: grep-based test failing on re-introduction of retired claims in standing docs (e.g. `sub-2-min` outside RUNBOOK-historical context, the `$12–17` estimate-as-actual string, "QLoRA measurably beat" in present-tense claim positions). Kept deliberately narrow to avoid false positives |
| New GS-07/GS-08 fixtures | Every `expected_tool_calls` entry validates against `tool_schema.v0.json` via the existing DEC-P1-8 validator (extends `tests/test_emitted_tool_calls_validate.py` / golden-set validation) — **no hallucinated tool or parameter names can be committed** |
| Injection fixtures | Shape/canary-uniqueness validation for the widened suite; off/on runner executes the new fixtures in dry-run (mock endpoint) without error |
| Test-retest flow | `worksheet.blind.yaml` provably contains no `human_label`/key fields and is order-reshuffled; `testretest` subcommand reproduces κ on a synthetic fully-agreeing pair (κ=1.0) and a known-disagreement pair; frozen `report.json` byte-identical after the run |

**Eval set / metric thresholds gating this phase:** the Tier-1 gates are *unchanged* — committed
retrieval-run metrics (Recall@10 ≥ 0.90, version-set recall = 1.0 on GS-01/GS-06, NO_MATCH
outcomes) and the recorded generation/injection runs must re-verify byte-stably. P7's own gate is
**green Tier-1 CI + zero frozen-artifact diffs**, not a new metric threshold. If the optional
capture window is approved, its numbers land as a **new dated row/column** in BENCHMARKS.md with
a full reproducibility stamp — never overwriting the frozen rows.

---

## 5. Task breakdown (ordered, independently committable)

Each task = one conventional commit (or small PR-internal sequence); Tier-1 CI green after each.

> **GPU usage: task 20 ONLY.** Tasks 1–19 run entirely on the laptop/CI (code, docs, fixtures,
> Postgres/Docker, blind labelling — no model op anywhere). Task 20 is the single approved
> ephemeral A100-40GB session (~$3–5, DEC-P7-7), runnable any time after tasks 15–16 land.

1. **DEC-P7-1..7 logged** ✅ **DONE 2026-07-18**: review findings pre-registered (DEC-P7-1) +
   the four §3 decisions as user-confirmed (DEC-P7-2..5) + test-retest reframe (DEC-P7-6) +
   approved capture window / Langfuse Cloud posture (DEC-P7-7).
2. **Repo hygiene**: delete `.staging` debris, explicit ignore rule, `git ls-files` tripwire test.
3. **Error envelope fix** + request-id header + tests (app.py L211–216, L361, L372).
4. **Auth + rate limiting** on `/api/chat` (per DEC-P7-2) + tests + `.env.example` additions
   (`API_AUTH_TOKENS`, `CHAT_RATE_LIMIT`, `TRUST_PROXY`) + RUNBOOK demo-flow note.
5. **Container hardening**: non-root USER + image HEALTHCHECK (app + mlflow Dockerfiles) + CI
   image-config assertion.
6. **Spine upsert provenance fix** + regression tests (wikidata.py L276–370).
7. **Migration #3** (three uniqueness constraints) + duplicate pre-audit + round-trip tests.
8. **θ re-binding** (per DEC-P7-3): pinned run id, artifact load, `StaleCalibrationError`,
   tests; remove the literal from `retrieve.py`.
9. **Doc-truth pass 1 — FT verdict**: CLAUDE.md §5 / README / ROADMAP §1(b) rewritten to the
   CUT-verdict framing; new README section "Why the demo works despite base tool-call accuracy
   0.083".
10. **Doc-truth pass 2 — timing + cost**: sub-2-min → measured-545 s posture everywhere (+ demo
    choreography note); PORTFOLIO project-total recomputed from DECISIONS actuals, dated,
    estimate language purged; **standing-cost line updated for the destroyed Langfuse VPS**
    (₹799/mo → ₹0; DEC-P3-7 gets a dated status annotation; Cloud free tier is the active
    fallback, self-hosted bootstrap remains documented).
11. **Doc-truth pass 3 — hallucination framing**: two-layer statement beside every
    "0 hallucinated" claim; BENCHMARKS.md dated annotations (no number changes).
12. **Pre-registration SHAs**: annotate DEC-P4-8/9 with the verified PR #5 commit SHAs
    (`0fd94477` → `710025c5` → `95ea3d39`, resolved Q3).
13. **Claims-lint tripwire test** (locks tasks 9–11 in place).
14. **Golden-set target resolution** (per DEC-P7-4): GOLDEN_SET_SCENARIOS.md dated revision.
15. **GS-07/GS-08 expansion** to n ≥ 10 each, schema-validated, scored in task 20.
16. **Injection-suite widening** (per DEC-P7-5): obfuscation variants + BU/UA/ASR harness
    support + honest static-suite framing in docs.
17. **Test-retest package** (per DEC-P7-6): PROTOCOL.md (committed first), blind-worksheet
    generator (labels/key stripped, reshuffled), `testretest` subcommand + tests → the user's
    blind labelling pass (~30–45 min, no GPU) → additive `report_testretest.json` + doc update
    (intra-rater κ honestly labelled, real-items-only κ).
18. **`docs/SCALE.md`** 50k design note.
19. **Close-out (pre-window)**: module READMEs, PORTFOLIO bullet for P7, DoD checklist sweep.
20. **Approved capture window — the only GPU task** (DEC-P7-7, ~$3–5, separately runnable):
    Langfuse Cloud keys in env → one ephemeral A100-40GB session (serve base model + judge per
    the frozen P3 config) scoring expanded GS-07/GS-08 + injection BU/UA/ASR, identical
    harness/judge config, full reproducibility stamp → **new dated rows** in BENCHMARKS.md
    (frozen rows byte-untouched) → teardown nuke-verified; evidence links added to
    PORTFOLIO/BENCHMARKS.

---

## 6. Definition of Done (instantiated from CLAUDE.md)

- [ ] Code complete and matches this approved spec (all four scope groups; non-goals respected).
- [ ] Unit + integration tests written and passing — every fix has its named regression test
      (§4 table); the five named golden regression tests remain green.
- [ ] Eval thresholds: **no new thresholds introduced; no frozen artifact re-scored.** Tier-1
      gates (retrieval metrics on committed artifacts, recorded generation/injection runs,
      schema-conformance suites) green throughout. MLflow untouched (no new runs by default).
- [ ] Benchmark tables: **no frozen metric cell changes.** `docs/BENCHMARKS.md` gains dated
      *annotations* (two-layer hallucination framing on Table 2; honest static-suite framing on
      the injection numbers) **plus the new clearly-dated rows from the approved capture window
      (task 20)** — expanded GS-07/GS-08 scores and the injection BU/UA/ASR triple, each with a
      full reproducibility stamp; existing rows byte-untouched.
- [ ] `docs/DECISIONS.md` updated: DEC-P7-1 (review findings, pre-registered) + DEC-P7-2..5
      (§3, as confirmed) + the golden-target revision + pre-registration SHA annotations.
      Module READMEs (serving, pipeline, rag, evals, infra) updated where behaviour changed.
- [ ] Runs cleanly from scratch: fresh clone + `.env` (with the new auth/limit vars) → compose up
      → migrations apply (incl. #3) → Tier-1 suite green; hardened image builds and passes its
      config assertions.
- [ ] 30-second demo path: `make demo-degraded` (or equivalent documented one-liner) showing the
      GPU-off surface now returns 401/429 correctly on `/api/chat` without a token, the scrubbed
      error envelope, and the reconciled README/benchmark pages — clickable proof of the
      hardening.
- [ ] `docs/PORTFOLIO.md`: quantified P7 bullet drafted (e.g. audited N standing claims to
      recorded evidence; secured the cost-bearing endpoint with token-first rate limiting;
      closed 3 data-integrity gaps with DB-owned constraints; human–human κ ceiling reported;
      50k scale path designed — final numbers filled at close-out, never pre-asserted).

---

## 7. Questions — ALL RESOLVED (2026-07-18)

- **Q1 — Second annotator. RESOLVED:** the original annotator (the user) performs the second
  pass ⇒ reframed as **blind intra-rater test-retest κ**, honestly labelled as a proxy — never
  presented as a human–human ceiling (DEC-P7-6). ≥14-day gap satisfied. A future second human
  adds the inter-rater ceiling additively.
- **Q2 — Capture window. RESOLVED — APPROVED** (~$3–5, DEC-P7-7) as task 20. Clarification
  recorded: **LangChain is not used** by this project (transitive RAGAS dep only); the destroyed
  VPS hosted **Langfuse**, whose tracing wrapper is no-op-safe — the window uses the **Langfuse
  Cloud free tier** (the documented DEC-P3-7 fallback), and the doc-truth pass records the VPS
  destruction as standing cost ₹799/mo → ₹0.
- **Q3 — Pre-registration proof. RESOLVED:** PR #5's 23 branch commits are retained and
  reachable via the GitHub API
  (`repos/venkathub/sutradhar-film-finder/pulls/5/commits`); the rule-before-numbers chain is
  `0fd94477` (03:06Z) → `710025c5` (10:13Z) → `95ea3d39` (10:28Z), all 2026-07-04. Task 12 is
  fully executable.

**APPROVED FOR EXECUTION (2026-07-18).** D1–D4 confirmed per recommendations; DEC-P7-1..7
logged in `docs/DECISIONS.md` before any code, per CLAUDE.md. Execution starts at task 2.

---

## 8. Execution status (updated 2026-07-18)

**Tasks 1–19 EXECUTED** on `feature/p7-credibility-hardening`, one committed diff per task,
Tier-1 (822 unit) + integration (124, live Postgres) suites green throughout; no frozen
benchmark artifact re-scored, no metric cell changed. Notable execution findings:
- The `uq_candidate_edges_dedup` constraint **exposed a real latent bug** on landing: 3
  within-batch duplicate proposals in the recorded extraction artifact, invisible to the
  SELECT-based dedup under `autoflush=False` — fixed with a batch-local seen-set (task 7).
- The claims-lint tripwire (task 13) caught its own author twice during tasks 13/19 —
  evidence it bites.
- The blind worksheet (`worksheet.blind.yaml`) is generated and committed unlabelled.

**Tasks 17 + 20 EXECUTED (2026-07-19) — P7 COMPLETE (section updated 2026-07-19):**
- **Task 17:** the rater's blind pass landed 30/30 → **intra-rater κ = 0.933 (29/30
  agreement; the single flip was on a foil), real-items-only κ = 1.000 (n = 15)**,
  second-pass-vs-judge κ = 0.670 computed offline — `report_testretest.json`, frozen
  `report.json` byte-untouched. Bounds (does not close) the single-annotator limitation.
- **Task 20:** two ephemeral A100 sessions (450702 serve → captures; 450712 judge+sidecar →
  rejudge + RAGAS via the committed `infra/gpu/p7_judge_window.py`), both nuke-verified;
  **₹51.09 ≈ $0.61 dashboard actuals**. Generation run `20260719T063002Z-1bf3cd3e` (24/24;
  exported Langfuse trace committed; MLflow run `846967f0…` backfilled to a durable local
  sqlite store — topology deviation disclosed in the DEC-P7-7 addendum; RAGAS faithfulness
  covered 8/24, disclosed) + live injection runs on the widened suite (**ASR 0.000 /
  BU 1.000 / UA 0.800 defenses-ON**). New dated BENCHMARKS sections only; frozen rows
  byte-untouched; Tier-1 pin deliberately unmoved. Operational lessons recorded: killed
  Session-A hold (manual stop + idle billing), leaked script slots blocking the next create.
- **PR #9 review pass:** all five blocking findings resolved in-branch (rate-limit bypass
  via unauthenticated token rotation — valid-token-only bucketing + timing-safe compares +
  regression test; MLflow three-way doc contradiction; ROADMAP second-annotator stale claim;
  undisclosed RAGAS coverage gap; missing committed trace export), plus the non-blocking
  corrections (flip count, counts 46/83, cost framing, spec header, mlflow HEALTHCHECK,
  Retry-After derivation, INJ-16 payload byte, GIN/GiST wording, decontamination notes).
