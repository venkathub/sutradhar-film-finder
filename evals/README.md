# evals

Evaluation harnesses for Sutradhar. **Import package:** `sutradhar.evals`

## Golden set (P1 — frozen)

`evals/golden/*.yaml` holds **34 fixtures across all 11 scenario categories**
(GS-01..GS-11 per `docs/GOLDEN_SET_SCENARIOS.md`), authored only from verified graph facts
(each `verify_source` names the QIDs / pinned pages). `make golden-validate` runs the
validator against the live graph:

- **schema** (pydantic; `expected_tool_calls` on GS-07/GS-08 validate against TOOL_SCHEMA v0);
- **golden eligibility** (DEC-P1-7 layer 3): every expected version/relationship must be
  gate-visible AND HIGH-or-human-verified — MEDIUM-backed or conflict-hidden fixtures are
  rejected (tested by mutation);
- **NO_MATCH proofs**: GS-02 negatives verify nothing in the graph resolves for them;
- category completeness (all 11 GS categories present).

CI (Tier-1 integration job) rebuilds the reviewed graph from recorded fixtures — including a
CI-mirrored review pass with the same semantics as the committed
`data-pipeline/review_decisions_20260702.yaml` — and re-validates every fixture plus the named
regressions (`test_gs01_version_set_recall`, `test_gs04_dub_vs_remake`, …).

Frozen 2026-07-02 against graph state: 15 works / 31 versions / 21 gate-visible edges
(reproducibility stamp in `docs/BENCHMARKS.md`).

### Generation slice (P3 task 4 — the Table 2 fixtures)

12 fixtures carry the additional generation labels (`expected_intent`, `expected_slots` —
one entry per turn for multi-turn; labels drawn from the frozen taxonomy, test-enforced by
`tests/test_golden_generation_labels.py`) plus `expected_tool_calls` validated against
TOOL_SCHEMA v0:

| Slice | Fixtures | Notes |
|---|---|---|
| GS-07 (code-mixed) | a Tanglish, b Hinglish (plot); **c Kanglish, d Tenglish (+actor slot), e native-script Kannada** (title path) | intent/slot accuracy |
| GS-08 (backtracking) | a Drishyam; **b Manichitrathazhu franchise; c Drishyam 2 with a mid-turn NO_MATCH** (no Tamil version exists — abstain without losing the set) | backtracking coherence |
| GS-02-conversational | **d Hinglish title (Salaar), e Hinglish plot, f Tanglish plot+language, g native-Telugu title (Pushpa)** | 0-inventions gate, end-to-end |

GS-02a/b/c stay retrieval-shaped and untouched (the P2 committed artifact keeps validating);
all new queries were verified against the live title index before freezing (negatives resolve
to nothing — including their extracted title forms; positives resolve at score 1.0). All 34
fixtures pass `make golden-validate` (expanded 2026-07-03, same graph state).

## Held-out negatives (P2 — abstention calibration)

`evals/negatives/heldout.yaml` holds **24 NO_MATCH queries** (12 out-of-catalog plot
descriptions incl. code-mixed/native-script registers + 12 real-but-uncatalogued titles),
split 50/50 `calibration`/`test` (6 plot + 6 title in each half). **Import:**
`sutradhar.evals.negatives`.

These are deliberately **not** golden fixtures: they exist to tune the NO_MATCH abstention
threshold θ (DEC-P2-5) — θ is tuned on the calibration half only; NO_MATCH precision/recall
is *reported* on the test half. Tuning on GS-02 would contaminate the golden gate, so the
two sets never mix.

"Absent from slice by construction" is **enforced, not asserted**: the integration test
(`tests/integration/test_negatives_absent.py`) rebuilds the full title index (canonical +
TMDB + IMDb AKAs) and requires `resolve_title` to return zero candidates for every negative
at the rapidfuzz 0.80 radius (DEC-P1-5). A collision re-authors the negative, never moves
the threshold.

## Retrieval eval harness + committed run artifacts (P2)

`sutradhar.evals.retrieval` (metrics + ablation runner) and `sutradhar.evals.calibration`
(NO_MATCH θ, DEC-P2-5) — entrypoints `make retrieval-eval` / `make calibrate-no-match`.
`evals/retrieval_runs/` holds the **committed** per-run evidence (DEC-P2-6): the compact
eval artifact (per-query ranked works + abstention signals for every ablation cell),
the calibration report/curve, and the GPU-session record. Tier-1 CI
(`tests/test_golden_retrieval_regressions.py`) recomputes every gating metric from these
files on each PR — no DB, no GPU; a regression blocks merge. Results: `docs/BENCHMARKS.md`
Table 1 (P2 exit gate met: Recall@10 = 1.000, version-set recall = 1.0).

## Frozen prompt artifacts (P3 — DEC-P3-4)

`evals/prompts/` holds the **hash-pinned base-model prompting strategy** the generation benchmark
runs under (loader: `sutradhar.evals.prompts`):

- `system_v1.md` — system prompt: grounding rules (assert only tool-returned films; cite;
  label all five relationship types; flag the original; abstain out-of-catalog), tool-usage
  and multi-turn behaviour, the **INTENT preamble contract**: every final prose answer per
  user turn begins with one machine-read line `INTENT: {"intent": …, "slots": …}` — the source
  the intent/slot-accuracy scorers parse (placement rationale in DEC-P3-4's 2026-07-03
  amendment) — and the **bold-title formatting contract**: every asserted film title in
  `**bold**`, nothing else bold — the deterministic surface the no-hallucinated-movie
  detector extracts (task-5 amendment).
- `exemplars_v1.md` — 3 handcrafted few-shot exemplars (one code-mixed Hinglish plot query, one
  multi-turn refine, one NO_MATCH abstention), appended to the system message as one hashed unit.
  Franchises (Ghajini, Okkadu/Ghilli, Interstellar) are deliberately **outside the golden set**;
  disjointness is test-enforced (no fixture query substring may appear in the bundle).
- `intent_taxonomy_v1.json` — the frozen 6-label taxonomy (`find_by_plot | find_by_title |
  list_versions | refine | disambiguate | out_of_catalog`) + 7 slot keys (= `refine_filter.by`
  vocabulary + `plot_description` + `title`) + the preamble format.
- `prompts.lock.json` — pinned SHA-256 per file + the combined **`prompt_hash`** stamped into
  every generation-run artifact and the Table 2 stamp. Any edit fails
  `tests/test_prompt_artifacts.py` until deliberately re-pinned:
  `uv run python -m sutradhar.evals.prompts --write-lock`. The P4 QLoRA before/after is only
  comparable under an identical `prompt_hash`.

## Generation metric scorers (P3 task 5 — `sutradhar.evals.generation`)

Pure, laptop/CI-safe scoring functions (string math only; the task-6 driver produces
transcripts, these score them, Tier-1 re-runs the same bytes over the committed artifact):

- **Tool-call accuracy (DEC-P3-5):** call-level AST match (name + placeholder-bound,
  normalized args; free-text `description` compared by token overlap, titles by `match_key`)
  + sequence-level headline (expected sequence in order; benign schema-valid extras
  tolerated) + schema-validity as the independent third number. `$work_id`/`[$version_set]`
  bind only to ids earlier tool results actually returned — an unseen id is a scored
  mismatch, never a crash.
- **Intent accuracy:** per-turn exact match of the parsed `INTENT:` preamble label
  (missing/malformed preamble = wrong, never an exception).
- **Slot accuracy:** micro-F1 over (key, value) pairs; titles normalized via `match_key`
  (native-script ↔ romanized matches), everything else casefolded.
- **No-hallucinated-movie detector (the GS-02 gate):** asserted titles (bold spans per the
  frozen contract + an unbolded `Title (year)` fallback with a language/meta-word guard)
  must fuzzy-resolve (≥ 0.80, DEC-P1-5) to a tool-result title of the same conversation;
  "Papanaasam"-style variants pass, inventions are counted; abstentions assert nothing.

## Conversation driver (P3 task 6 — `sutradhar.evals.driver`)

The multi-turn validate→execute→feedback loop that runs one golden fixture conversation
against any OpenAI-compatible endpoint (`LLMClient.chat()`; mock in dry-run, vLLM in the
P4 GPU window):

- **Outbound tools are generated from `tool_schema.v0.json`** — never hand-written, so
  drift from the frozen schema is impossible (P3_SPEC §2.8).
- **Every emitted call is validated BEFORE execution** (the DEC-P1-8 jsonschema validator
  applied to model output): hallucinated tools, hallucinated parameters and wrong-typed
  arguments are recorded as scored verdicts, the error is fed back as the tool result, and
  the loop continues — bounded by `MAX_TOOL_ROUNDS = 6` per turn, never a crash.
- **Valid calls execute against `sutradhar.graph.repository`** (the five frozen v0 tools).
  `search_by_plot` **replays the committed P2 retrieval run** (pinned `RETRIEVAL_RUN`,
  per-fixture records): no neural op on the laptop, and both Table 2 columns see
  byte-identical tool behaviour; fixtures absent from the recorded run (the P3
  conversational negatives, out-of-catalog by construction) replay as honest abstention.
  Description *quality* is still scored by tool-call accuracy.
- **Full transcript capture** (`FixtureTranscript`): every message, call (raw + verdict +
  bound arguments + result/error), per-round usage and latency, per-turn final answers —
  what the task-9 artifact commits and Langfuse traces mirror. Endpoint off/error →
  `chat_status` recorded, fixture aborts gracefully (DEC-P0-4).

Integration proof (`tests/integration/test_driver_e2e.py`): GS-08a end-to-end against the
live Postgres graph with a scripted mock model that binds REAL ids from prior tool results —
all five v0 tools exercised, every result v0-shape-valid, DEC-P3-5 sequence score 1.0 with
benign extras tolerated, zero hallucinated movies.

## LLM-as-judge (P3 task 7 — `sutradhar.evals.judge`, DEC-P3-1)

Self-hosted OSS judge (gpt-oss-20b primary, Phi-4-14B alternate) served by vLLM in a
**short ephemeral GPU session** (`make gpu-judge`: serve `JUDGE_MODEL` + BGE-M3 → run the
κ report → destroy; teardown guaranteed). The client is OpenAI-compatible and env-driven
(`JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY`), so the DEC-P3-1 self-hosted ↔ frontier
escalation is pure config — proven by test against both endpoint styles.

- **Governance pins:** rubric prompts live in-repo (`evals/prompts/judge_coherence_v1.md`,
  `judge_faithfulness_v1.md`) and hash into `JudgeConfig` (separate from the base-model
  `prompt_hash`); temperature 0; guided decoding + pinned low reasoning effort on the wire
  (gpt-oss-20b is a reasoning model). Malformed judge output → `judge_error` recorded,
  never a crash; a judge_error binarizes to the negative class so it *hurts* agreement
  rather than hiding.
- **Human-agreement validation** (`sutradhar.evals.judge_validation`, blocks the judge
  freeze — task 13): `make judge-worksheet` builds the ~24-item blind worksheet from
  committed transcripts (+ deterministic foils: a re-answer-turn-1 coherence foil and an
  invented-movie faithfulness foil; provenance in a separate key file), the human fills
  every `human_label`, `make judge-validate` scores it with the pinned judge and reports
  percent agreement + **Cohen's κ (gate: ≥ 0.6)** → `evals/judge_validation/report.json`.
  **Judge–human agreement result (2026-07-03, task 13): percent agreement 0.867,
  Cohen's κ = 0.738 — PASS** (coherence κ = 1.0 n=6; faithfulness κ = 0.673 n=24; 0
  judge_errors; one ephemeral A100 session, ≈6 min, destroyed). Frozen judge:
  `openai/gpt-oss-20b @ 6cee5e81…`, rubric hashes pinned, temperature 0, low reasoning
  effort, guided JSON. All 4 disagreements were the judge forgiving trailing invented-film
  mentions on grounded foils — the deterministic detector gates for exactly this reason.
  Full verdicts: `evals/judge_validation/report.json` (labels by the project owner; one
  item corrected on review — see DEC-P3-1 amendment).

## RAGAS adapter (P3 task 8 — `sutradhar.evals.ragas_metrics`, DEC-P3-3)

Supplementary Table 2 signals — RAGAS **faithfulness** + **answer relevancy** — computed
with **zero external eval APIs**: LLM = the pinned self-hosted judge endpoint
(`JUDGE_BASE_URL`/`JUDGE_MODEL` via `ragas.llms.llm_factory`), embeddings = **BGE-M3 in
the same judge GPU session** (`EMBED_BASE_URL`, DEC-0002's embedder reused). The
**gating** faithfulness signal remains the deterministic detector; RAGAS numbers are
reported, the detector gates.

- `build_scorer(settings)` → `(None, reason)` when judge/embeddings are unset — callers
  skip cleanly (the P0 "off" posture). Construction is network-free; scoring runs as a
  batch pass over recorded transcripts inside the GPU session.
- Per-sample metric failures land in `RagasScores.*_error`, never raised into the batch;
  the ragas version is stamped on every scores object (RAGAS internal prompts evolve —
  the pin keeps runs comparable).
- Tests drive the **real ragas pipelines** through fakes subclassing the real base
  contracts (`InstructorBaseRagasLLM` / `BaseRagasEmbedding`) — statement-ratio
  faithfulness, cosine relevancy, noncommittal zeroing, NaN and failure paths, all offline.
- Dependency note: `ragas 0.4.3` is import-broken against `langchain-community ≥ 0.4`
  (removed `vertexai` module); a `[tool.uv] constraint-dependencies` pin holds the stack
  at `<0.4` until a ragas release absorbs it (see DEC-P3-3 amendment).

## Generation-run artifact + runner (P3 task 9 — `sutradhar.evals.generation_run`)

The committed Table 2 evidence unit, mirroring the P2 retrieval-run pattern (DEC-P2-6):
`evals/generation_runs/<run_id>.json` embeds every fixture's **full transcript**, its
deterministic scores (tool-call two-level + validity, intent, slots, hallucination per
turn), the Table 2 aggregates (`MetricsBlock` incl. per-slice breakdown and the
`gs02_inventions` hard gate), and a reproducibility stamp (code SHA, golden-set hash,
`prompt_hash`, tool-schema version+sha256, pinned `retrieval_run`, ragas version).

- **Honesty invariants, validator-enforced:** `mode="dry_run"` ⇒ `serving=null` and null
  latency/throughput — mock timings can never masquerade as GPU numbers; Table 2 publishes
  only `mode="live"` runs captured by this same harness at the top of the P4 GPU window.
- **Enrichment passes** (`apply_judge_scores`, `apply_ragas_scores`) fill the supplementary
  judge-coherence and RAGAS fields in place during the ephemeral GPU session; deterministic
  metrics never depend on them (Tier-1 checks presence/shape only).
- Runner: `evals/run_generation_eval.py` — `make generation-dryrun` (mock endpoint,
  task 11) / `make benchmark-generation` (live `LLM_BASE_URL` + `--with-judge
  --with-ragas`). Exit code 3 = the GS-02 zero-inventions gate failed.

### Committed dry-run (P3 task 11 — machinery evidence, never Table 2)

`evals/generation_runs/20260703T012339Z-e7fff041.json` — the scripted mock
(`evals/mock_llm.py`, a canned-transcript player derived from the golden labels with
placeholder binding against real tool results) driven through the full harness against the
live graph + the pinned P2 retrieval replay:

| Signal | Value | Meaning |
|---|---|---|
| fixtures completed | 12/12 | all generation-slice conversations end-to-end |
| tool-call sequence / call level | 1.0 / 1.0 | expected sequences matched, placeholder-bound |
| schema validity | 35/36 (0.972) | **exactly the seeded `lookup_movie` hallucinated tool on GS-07a** — caught by validation, error fed back, conversation recovered |
| faithfulness | 17/18 (0.944) | **exactly the seeded invented movie ("Chokher Aloy") on GS-07e** — caught by the detector |
| GS-02 inventions | **0** | the hard gate, green |
| intent / slots | 1.0 / 1.0 | preamble parsing + micro-F1 across all 12 fixtures |
| latency / tokens-per-sec | null | dry_run invariant (mock timings never look like GPU numbers) |

MLflow evidence: generation dry-run → run `c2fb0eab52bd4691a8a70b35491d0dce`
(`sutradhar/generation`); Table 1 backfill → run `26dc04707c7d4efda4c07dff64a7b8ba`
(`sutradhar/retrieval`, discharging the P2 stamp note). Langfuse evidence: the dry-run
traced live to the self-hosted instance (DEC-P3-7 on AIC Cloud, public HTTPS via
cloudflared tunnel); the GS-08c trace (14 observations: agent → generation/tool spans)
is exported and committed as `20260703T012339Z-e7fff041.trace.json` **plus a screenshot**
(`…e7fff041.trace.png`; MLflow run view: `docs/assets/p3-mlflow-generation-dryrun.png`)
so the evidence outlives the tunnel URL.

## Observability (P3 task 10 — `sutradhar.obs`, DEC-P3-6/P3-2/P3-7)

- **Langfuse tracing** (`sutradhar.obs.tracing`): one thin explicit span seam (`Tracer.span`)
  around exactly four chokepoints — fixture loop (`agent`), each `chat()` round
  (`generation`), each tool execution (`tool`), judge calls (`evaluator`). **No-ops without
  `LANGFUSE_*` keys** (test-proven: the SDK is never even imported); P5's FastAPI middleware
  reuses the same seam. Backend = self-hosted Langfuse v3 on the AIC Cloud VPS —
  `make langfuse-up` is the DEC-P3-7 idempotent from-scratch bootstrap (see
  `infra/langfuse/README.md`). Benchmark-cited traces are exported
  (`sutradhar.obs.tracing.export_trace`) and committed so evidence outlives the VPS.
- **MLflow** (`sutradhar.obs.mlflow_log`): experiments `sutradhar/generation` (every run:
  §6.1 stamp as params, Table 2 aggregates as metrics, run JSON as artifact) and
  `sutradhar/retrieval` (`make mlflow-backfill` logs the committed P2 Table 1 run,
  discharging its "(MLflow wiring lands in P3)" stamp note). The runner logs automatically
  and degrades with a clear message when the server is down — observability never fails
  an eval.

## Two-tier CI (P3 task 12 — ROADMAP §6.2, made real)

- **Tier-1 (every PR — no GPU, no model calls, no network):**
  `tests/test_golden_generation_regressions.py` loads the pinned `GENERATION_RUN` artifact
  (default: latest committed) and **recomputes every deterministic metric from the recorded
  transcripts with the same scorer bytes**, asserting equality with the committed metrics
  block (drift gate) plus the hard gates: GS-02 = 0 inventions, every invalid call flagged +
  accounted, the two seeded faults visibly caught, dry-run honesty invariants, stamp pins
  current prompt/schema/golden hashes (a stale artifact after a re-pin fails CI). Judge/RAGAS
  fields are shape-checked, never re-judged. `tests/test_emitted_tool_calls_validate.py`
  proves the 3 seeded fault classes (hallucinated tool, hallucinated parameter, wrong type)
  are caught 3/3 and that every executed call in the committed run validates against frozen
  v0. The P0 "artifact-validate stub" step is retired — validation lives entirely in pytest.
- **Tier-2 (`workflow_dispatch` only, never PRs):** inputs `run_mode` (`dry_run`|`live`) +
  `reason`; brings up Postgres, seeds the graph from recorded fixtures
  (`make seed-graph-ci`, proven to reproduce the exact 15-work/31-version state with all 34
  fixtures golden-valid), gates on `build_golden.py`, runs `make benchmark-generation`
  (live: secrets-provided `LLM_BASE_URL` + judge/RAGAS endpoints) or `make generation-dryrun`,
  and uploads the sealed run JSON as a workflow artifact — **a human reviews and commits it
  via PR** (grooming Q5; CI never auto-commits benchmark numbers).

## Planned (P3+)
- Retrieval eval harness + metrics — **landed in P2** (see above).
- Frozen prompt artifacts — **landed in P3 task 3** (see above).
- RAGAS harness; MLflow/Langfuse wiring; the two-table benchmark runner (base vs QLoRA)
  reusing `expected_tool_calls` (P3+).
